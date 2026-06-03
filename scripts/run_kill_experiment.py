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
    pred = pred.strip().lower()
    best = 0.0
    for g in golds:
        g = str(g).strip().lower()
        denom = max(len(pred), len(g), 1)
        score = 1.0 - _levenshtein(pred, g) / denom
        best = max(best, score if score >= threshold else 0.0)
    return best


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
    p.add_argument("--max_new_tokens", type=int, default=64)
    args = p.parse_args()

    import torch
    from modelscope import snapshot_download
    from transformers import AutoProcessor, AutoModelForImageTextToText

    from csf_squeeze.integrate_qwen3vl import build_module_from_hf, patch_qwen3vl

    model_dir = snapshot_download(args.model)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

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
            module.enabled, module.selection_mode = True, "freq"
        else:
            raise ValueError(arm)

    @torch.no_grad()
    def run_arm(arm: str):
        configure(arm)
        tok_counts, scores = [], []
        for r in data:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": r["image"]},
                {"type": "text", "text": r["question"]}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[r["image"]], return_tensors="pt").to("cuda")
            thw = inputs["image_grid_thw"]
            n_vis = int((thw.prod(-1) // model.model.visual.spatial_merge_size ** 2).sum())
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            # effective visual tokens: dense = full; compressed arms = measured by the module
            tok_counts.append(n_vis if arm == "dense" else module.last_n_out)
            gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            scores.append(anls(gen, r["answers"]))
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
