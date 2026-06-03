"""Shared DocVQA evaluation utilities for CSF-Squeeze experiments.

Pure helpers (ANLS, answer parsing, output cleaning) are CPU-testable and have
no heavy imports at module load; model/data loading defer their imports so this
file stays importable on the CPU dev box.
"""
from __future__ import annotations

import os
import re
import sys

# Make `import csf_squeeze` work when scripts are run from anywhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# DocVQA convention: prompt the model for a short answer so ANLS can match.
SHORT_ANSWER = " Answer the question using a single word or phrase."


# --------------------------------------------------------------------------- #
# Metric (ANLS) and text normalization                                        #
# --------------------------------------------------------------------------- #
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


def _normalize(s) -> str:
    s = str(s).strip().lower()
    return re.sub(r"^[\s\"'.;:,]+|[\s\"'.;:,]+$", "", s)


def _as_answer_list(golds):
    """Normalize DocVQA answers (list[str] / str / dict) to a flat list of str."""
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
    t = text.strip()
    t = re.sub(r"(?i)^(the\s+answer\s+is|answer\s*:|it\s+is|this\s+is)\s*", "", t)
    return t.splitlines()[0].strip() if t else t


def anls(pred: str, golds, threshold: float = 0.5) -> float:
    pred = _normalize(pred)
    best = 0.0
    for g in _as_answer_list(golds):
        g = _normalize(g)
        denom = max(len(pred), len(g), 1)
        score = 1.0 - _levenshtein(pred, g) / denom
        best = max(best, score if score >= threshold else 0.0)
    return best


# --------------------------------------------------------------------------- #
# Data + model loading (GPU/transformers required)                            #
# --------------------------------------------------------------------------- #
def load_benchmark(name: str, subset: int):
    """Load a dense-text VQA validation subset. Supported: docvqa, infovqa.

    Tries ModelScope then HF datasets; both expose (image, question, answers).
    Dataset ids/fields below match the lmms-lab mirrors; adjust if the box differs.
    """
    name = name.lower()
    spec = {
        "docvqa": ("lmms-lab/DocVQA", "DocVQA"),
        "infovqa": ("lmms-lab/DocVQA", "InfographicVQA"),
    }
    if name not in spec:
        raise ValueError(f"unknown benchmark {name!r}; supported: {list(spec)}")
    repo, subset_name = spec[name]
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(repo, subset_name=subset_name, split="validation")
    except Exception:
        from datasets import load_dataset
        ds = load_dataset(repo, subset_name, split="validation")
    items = []
    for i, r in enumerate(ds):
        if i >= subset:
            break
        items.append({"image": r["image"], "question": r["question"], "answers": r.get("answers", [])})
    return items


def load_docvqa(subset: int):
    """Backward-compatible alias for the DocVQA subset."""
    return load_benchmark("docvqa", subset)


def load_model_and_module(model_name: str, keep_ratio: float, stride: int):
    """Load Qwen3-VL + a patched CSF-Squeeze module. Returns (model, proc, module)."""
    import torch
    from modelscope import snapshot_download
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelCls
    from csf_squeeze.integrate_qwen3vl import build_module_from_hf, patch_qwen3vl

    model_dir = snapshot_download(model_name)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = ModelCls.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    module = build_module_from_hf(
        model.config, keep_ratio=keep_ratio, downsample_stride=stride
    ).to("cuda", dtype=torch.bfloat16)
    patch_qwen3vl(model, module)
    return model, proc, module


def configure_arm(module, arm: str, keep_ratio=None, stride=None):
    """Set the module's compression policy for one experiment arm.

    Arms:
        dense        no compression (upper bound)
        uniform      content-homogeneous exemption (same rho fraction, random)
        csf          frequency-differentiated, DeepStack-consistent (ours)
        csf-naive    frequency-differentiated but DeepStack-UNAWARE (ablation,
                     Sec. 4.6-2): isolates the value of consistent propagation.
    """
    if keep_ratio is not None:
        module.config.keep_ratio = keep_ratio
    if stride is not None:
        module.config.downsample_stride = stride
    module.deepstack_mode = "consistent"     # default unless the arm overrides
    if arm == "dense":
        module.enabled = False
    elif arm == "uniform":
        module.enabled, module.selection_mode = True, "random"
    elif arm == "csf":
        # training-free probe; use "freq" with trained LoRA + router weights instead.
        module.enabled, module.selection_mode = True, "energy"
    elif arm == "csf-naive":
        module.enabled, module.selection_mode = True, "energy"
        module.deepstack_mode = "naive"
    else:
        raise ValueError(f"unknown arm: {arm}")


def eval_arm(model, proc, module, data, arm, max_new_tokens=32, debug=0):
    """Evaluate one arm over ``data``. Returns (mean_anls, mean_visual_tokens)."""
    import torch

    merge = model.model.visual.spatial_merge_size
    scores, tok_counts = [], []
    with torch.no_grad():
        for si, r in enumerate(data):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": r["image"]},
                {"type": "text", "text": r["question"] + SHORT_ANSWER}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[r["image"]], return_tensors="pt").to("cuda")
            thw = inputs["image_grid_thw"]
            n_vis = int((thw.prod(-1) // merge ** 2).sum())
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            tok_counts.append(n_vis if arm == "dense" else module.last_n_out)
            gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            pred = _clean_generation(gen)
            sc = anls(pred, r["answers"])
            scores.append(sc)
            if debug and si < debug:
                print(f"  [{arm}] Q={r['question'][:50]!r} "
                      f"gold={_as_answer_list(r['answers'])[:3]} pred={pred[:50]!r} anls={sc:.3f}")
    return sum(scores) / len(scores), sum(tok_counts) / len(tok_counts)
