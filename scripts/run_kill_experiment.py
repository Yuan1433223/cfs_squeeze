"""Decisive 'kill-or-confirm' experiment (REQUIRES GPU + transformers + model).

Question (Sec. 4.4): at MATCHED visual-token count on dense-text DocVQA, does
frequency-DIFFERENTIATED compression (csf) beat content-HOMOGENEOUS compression
(uniform) -- and stay close to the dense upper bound?

Three arms share ONE model (toggled, no reload):
    dense   : module.enabled=False           -> no compression (upper bound)
    uniform : selection_mode="random"         -> exempt SAME rho fraction, content-agnostic
    csf     : selection_mode="freq"           -> exempt top-rho by high-freq saliency
uniform and csf use identical rho/stride, so their token counts match exactly;
the only difference is WHICH tokens are exempt -> isolates frequency-awareness.

Run on the ModelScope DSW (24G):
    python scripts/run_kill_experiment.py --subset 200 --rho 0.25 --stride 2
Gate M1 (smoke_test_gpu.py) must pass first.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `import csf_squeeze` work when run as `python scripts/run_kill_experiment.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---- ANLS metric (standard for DocVQA) -------------------------------------
def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds, threshold: float = 0.5) -> float:
    pred = _normalize(pred)
    best = 0.0
    for g in _as_answer_list(golds):
        g = _normalize(g)
        denom = max(len(pred), len(g), 1)
        score = 1.0 - _levenshtein(pred, g) / denom
        best = max(best, score if score >= threshold else 0.0)
    return best


def _normalize(s) -> str:
    """Lowercase, strip surrounding whitespace/punctuation/quotes (DocVQA-style)."""
    import re
    s = str(s).strip().lower()
    s = re.sub(r"^[\s\"'.;:,]+|[\s\"'.;:,]+$", "", s)
    return s


def _as_answer_list(golds):
    """DocVQA answers come as list[str], a single str, or a dict with 'answer(s)'.
    Normalize all of these to a flat list of strings."""
    if golds is None:
        return []
    if isinstance(golds, dict):
        golds = golds.get("answers") or golds.get("answer") or list(golds.values())
    if isinstance(golds, (str, bytes)):
        return [golds]
    try:
        return [g for g in golds]
    except TypeError:
        return [golds]


def _clean_generation(text: str) -> str:
    """Strip common verbose prefixes so a short gold answer can match."""
    import re
    t = text.strip()
    t = re.sub(r"(?i)^(the\s+answer\s+is|answer\s*:|it\s+is|this\s+is)\s*", "", t)
    return t.splitlines()[0].strip() if t else t


def load_docvqa(subset: int):
    """Load a DocVQA validation subset (image, question, answers). Tries ModelScope
    then HF datasets; adjust the dataset id/fields to what's available on the box."""
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load("lmms-lab/DocVQA", subset_name="DocVQA", split="validation")
    except Exception:
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    items = []
    for i, r in enumerate(ds):
        if i >= subset:
            break
        items.append({"image": r["image"], "question": r["question"], "answers": r.get("answers", [])})
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--subset", type=int, default=200)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--arms", default="dense,uniform,csf")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--debug", type=int, default=0, help="print this many (q, gold, pred, anls) samples")
    args = p.parse_args()

    import torch
    from modelscope import snapshot_download
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelCls

    from csf_squeeze.integrate_qwen3vl import build_module_from_hf, patch_qwen3vl

    model_dir = snapshot_download(args.model)
    proc = AutoProcessor.from_pretrained(model_dir)
    # Avoid device_map (needs a recent accelerate); load then move to cuda.
    model = ModelCls.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()

    module = build_module_from_hf(
        model.config, keep_ratio=args.rho, downsample_stride=args.stride
    ).to("cuda", dtype=torch.bfloat16)
    patch_qwen3vl(model, module)   # NOTE: untrained router => training-free signal;
    #                                load LoRA + router weights here for the trained result.

    data = load_docvqa(args.subset)
    print(f"[data] {len(data)} DocVQA samples | rho={args.rho} stride={args.stride}")

    def configure(arm: str):
        if arm == "dense":
            module.enabled = False
        elif arm == "uniform":
            module.enabled, module.selection_mode = True, "random"
        elif arm == "csf":
            # training-free run: select by high-frequency operator energy (content-based).
            # For the trained result, load LoRA + router weights and use "freq".
            module.enabled, module.selection_mode = True, "energy"
        else:
            raise ValueError(arm)

    # Official DocVQA-style instruction: elicit a SHORT answer so ANLS can match.
    SHORT_ANSWER = " Answer the question using a single word or phrase."

    @torch.no_grad()
    def run_arm(arm: str):
        configure(arm)
        tok_counts, scores = [], []
        for si, r in enumerate(data):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": r["image"]},
                {"type": "text", "text": r["question"] + SHORT_ANSWER}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[r["image"]], return_tensors="pt").to("cuda")
            thw = inputs["image_grid_thw"]
            n_vis = int((thw.prod(-1) // model.model.visual.spatial_merge_size ** 2).sum())
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            # effective visual tokens: dense = full; compressed arms = measured by the module
            tok_counts.append(n_vis if arm == "dense" else module.last_n_out)
            gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            pred = _clean_generation(gen)
            sc = anls(pred, r["answers"])
            scores.append(sc)
            if args.debug and si < args.debug:
                print(f"  [{arm}] Q={r['question'][:50]!r} "
                      f"gold={_as_answer_list(r['answers'])[:3]} pred={pred[:50]!r} anls={sc:.3f}")
        return sum(scores) / len(scores), sum(tok_counts) / len(tok_counts)

    print(f"\n{'arm':<10}{'ANLS':>10}{'avg_vis_tok':>14}")
    results = {}
    for arm in args.arms.split(","):
        acc, toks = run_arm(arm)
        results[arm] = (acc, toks)
        print(f"{arm:<10}{acc:>10.4f}{toks:>14.1f}")

    if "csf" in results and "uniform" in results:
        gap = results["csf"][0] - results["uniform"][0]
        print(f"\n[confirm criterion] ANLS(csf) - ANLS(uniform) = {gap:+.4f} "
              f"(>0 at matched tokens confirms frequency-differentiation; "
              f"also check csf close to dense).")


if __name__ == "__main__":
    main()
