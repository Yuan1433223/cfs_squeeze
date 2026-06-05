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


def _progress(iterable, total=None, desc=None):
    """tqdm if available (cloud env has it via transformers); plain iter otherwise."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    except Exception:
        return iterable

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
    """Load a VQA validation subset. Supported:
        docvqa, infovqa : free-form short answer + ANLS
        mmbench         : multi-choice (A/B/C/D)        + accuracy
        pope            : yes/no                         + accuracy

    Each item carries ``task`` so eval routines pick the right scoring.
    """
    name = name.lower()
    if name in ("docvqa", "infovqa"):
        return _load_docvqa_family(name, subset)
    if name == "mmbench":
        return _load_mmbench(subset)
    if name == "pope":
        return _load_pope(subset)
    raise ValueError(f"unknown benchmark {name!r}")


def _ms_download_file(repo: str, file_path: str, revision: str = "master") -> str:
    """Download one file from a ModelScope dataset via the public API.

    Bypasses ``datasets``/``MsDataset.load`` entirely so no HuggingFace Hub
    metadata validation occurs. Cached on local disk by URL.
    """
    import requests
    cache_root = os.path.expanduser("~/.cache/cfs_squeeze/datasets")
    os.makedirs(cache_root, exist_ok=True)
    safe = file_path.replace("/", "_").replace("\\", "_")
    local = os.path.join(cache_root, f"{repo.replace('/', '_')}__{safe}")
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    url = (f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo"
           f"?Source=SDK&Revision={revision}"
           f"&FilePath={file_path.replace('/', '%2F')}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = local + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local)
    return local


def _ms_list_files(repo: str, revision: str = "master"):
    """List file paths in a ModelScope dataset (best-effort)."""
    import requests
    url = (f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo/files"
           f"?Revision={revision}&Recursive=true")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json().get("Data", {})
    files = data.get("Files") or data.get("files") or []
    return [f.get("Path") or f.get("path") for f in files if isinstance(f, dict)]


def _decode_image(field):
    """Parquet 'image' columns may be {bytes, path} dicts, raw bytes, or PIL.Image."""
    import io
    from PIL import Image
    if isinstance(field, dict):
        b = field.get("bytes") or field.get("data")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if field.get("path"):
            return Image.open(field["path"]).convert("RGB")
    if isinstance(field, (bytes, bytearray)):
        return Image.open(io.BytesIO(field)).convert("RGB")
    return field


def _ms_load(candidates, split):
    """DEPRECATED: kept for the docvqa family which already worked via this path.

    For new datasets we use :func:`_ms_download_file` directly to avoid the
    HF Hub metadata validation that triggers ``Couldn't reach ... on the Hub``
    on the cloud DSW even after a successful ModelScope download.
    """
    from datasets import load_dataset
    last_err = None
    try:
        from modelscope import dataset_snapshot_download
    except Exception:
        from modelscope.msdatasets import MsDataset
        for repo, sub in candidates:
            try:
                kw = dict(split=split, trust_remote_code=True)
                if sub is not None:
                    kw["subset_name"] = sub
                return MsDataset.load(repo, **kw).to_hf_dataset()
            except Exception as e:
                last_err = e
        raise RuntimeError(f"MsDataset.load all failed; last={last_err!r}")

    for repo, sub in candidates:
        try:
            local_dir = dataset_snapshot_download(dataset_id=repo)
            ds = _scan_local_dataset(local_dir, split, sub)
            if ds is not None:
                return ds
            if sub is not None:
                return load_dataset(local_dir, sub, split=split, trust_remote_code=True)
            return load_dataset(local_dir, split=split, trust_remote_code=True)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"None of the ModelScope candidates loaded: {candidates}. "
        f"Last error: {type(last_err).__name__}: {str(last_err)[:200]}"
    )


def _scan_local_dataset(local_dir, split, sub):
    """Best-effort: find a split-named parquet/json file under local_dir.

    Patterns observed in ModelScope mirrors:
      lmms-lab/POPE/Full/<split>-00000-of-00001.parquet
      lmms-lab/DocVQA/<sub>/{validation,train}-*.parquet
      *<split>*.{parquet,json,jsonl}
    Tries narrower-then-wider globs and returns the first one ``datasets`` can read.
    """
    import glob
    from datasets import load_dataset
    base_candidates = []
    if sub:
        base_candidates.append(os.path.join(local_dir, sub))
    base_candidates.extend([os.path.join(local_dir, "Full"), local_dir])
    for base in base_candidates:
        if not os.path.isdir(base):
            continue
        for ext, fmt in (("parquet", "parquet"), ("jsonl", "json"), ("json", "json")):
            patterns = [
                os.path.join(base, f"{split}*.{ext}"),
                os.path.join(base, f"*{split}*.{ext}"),
                os.path.join(base, "**", f"*{split}*.{ext}"),
            ]
            for pat in patterns:
                files = sorted(glob.glob(pat, recursive=True))
                if files:
                    try:
                        return load_dataset(fmt, data_files=files, split="train")
                    except Exception:
                        continue
    return None


def _load_docvqa_family(name: str, subset: int):
    spec = {"docvqa": "DocVQA", "infovqa": "InfographicVQA"}
    ds = _ms_load([("lmms-lab/DocVQA", spec[name])], split="validation")
    items = []
    for i, r in enumerate(ds):
        if i >= subset:
            break
        items.append(dict(task="anls", image=r["image"], question=r["question"],
                          answers=r.get("answers", [])))
    return items


def _load_mmbench(subset: int):
    """MMBench (dev): multi-choice. Direct ModelScope file download (no datasets/HF).

    The ``/repo/files`` listing endpoint is unreliable across mirrors (we've seen
    405 Method Not Allowed), so we bypass listing and try a small set of known
    file paths against several mirrors. The first that resolves (HTTP 200 + a
    readable parquet/tsv) wins.
    """
    import pandas as pd
    candidates = [
        # (repo, file_path) -- ordered by likelihood:
        ("AI-ModelScope/MMBench", "MMBench_DEV_EN/dev-00000-of-00001.parquet"),
        ("AI-ModelScope/MMBench", "dev/dev-00000-of-00001.parquet"),
        ("AI-ModelScope/MMBench", "MMBench_DEV_EN/test-00000-of-00001.parquet"),
        ("AI-ModelScope/MMBench", "mmbench_dev_en_20231003.tsv"),
        ("AI-ModelScope/MMBench_DEV_EN", "dev-00000-of-00001.parquet"),
        ("AI-ModelScope/MMBench_DEV_EN", "test-00000-of-00001.parquet"),
        ("modelscope/MMBench", "MMBench_DEV_EN/dev-00000-of-00001.parquet"),
        ("modelscope/MMBench", "mmbench_dev_en_20231003.tsv"),
    ]
    last_err = None
    for repo, fp in candidates:
        try:
            local = _ms_download_file(repo, fp)
        except Exception as e:
            last_err = e
            continue
        try:
            df = pd.read_parquet(local) if local.endswith(".parquet") else \
                 pd.read_csv(local, sep="\t")
        except Exception as e:
            last_err = e
            continue
        items = []
        img_col = "image" if "image" in df.columns else (
            "image_path" if "image_path" in df.columns else None)
        if img_col is None:
            last_err = RuntimeError(f"no image column in {fp}; cols={list(df.columns)[:8]}")
            continue
        for _, r in df.iterrows():
            if len(items) >= subset:
                break
            opts = {k: r[k] for k in ("A", "B", "C", "D")
                    if k in df.columns and pd.notna(r[k])}
            items.append(dict(task="mc",
                              image=_decode_image(r[img_col]),
                              question=str(r["question"]),
                              options=opts,
                              answer=str(r.get("answer", "")).strip().upper()))
        if items:
            return items
    raise RuntimeError(f"MMBench: no candidate file worked. Last error: {last_err!r}")


def _load_pope(subset: int):
    """POPE: yes/no. Read parquet files directly from a ModelScope mirror."""
    import pandas as pd
    repo = "lmms-lab/POPE"
    splits = ("adversarial", "popular", "random")
    per_split = max(1, subset // len(splits))
    items, last_err = [], None
    for split in splits:
        try:
            local = _ms_download_file(repo, f"Full/{split}-00000-of-00001.parquet")
            df = pd.read_parquet(local)
            taken = 0
            for _, r in df.iterrows():
                if taken >= per_split or len(items) >= subset:
                    break
                items.append(dict(task="yesno",
                                  image=_decode_image(r["image"]),
                                  question=str(r["question"]),
                                  answer=str(r.get("answer", "")).strip().lower()))
                taken += 1
        except Exception as e:
            last_err = e
            continue
    if not items:
        raise RuntimeError(f"POPE: no split could be loaded. Last error: {last_err!r}")
    return items[:subset]


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
    """Evaluate one arm over ``data``. Returns (mean_score, mean_visual_tokens).

    Dispatches by item['task']:
        anls   -> ANLS (DocVQA / InfographicVQA, short free-form)
        mc     -> exact-letter accuracy (MMBench, A/B/C/D)
        yesno  -> exact-yes/no accuracy (POPE)
    """
    import torch

    merge = model.model.visual.spatial_merge_size
    scores, tok_counts = [], []
    running_score = 0.0
    pbar = _progress(range(len(data)), total=len(data), desc=f"eval/{arm}")
    with torch.no_grad():
        for si in pbar:
            r = data[si]
            task = r.get("task", "anls")
            text_q, gen_max, score_fn = _format_question(r, task, max_new_tokens)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": r["image"]},
                {"type": "text", "text": text_q}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[r["image"]], return_tensors="pt").to("cuda")
            thw = inputs["image_grid_thw"]
            n_vis = int((thw.prod(-1) // merge ** 2).sum())
            out = model.generate(**inputs, max_new_tokens=gen_max, do_sample=False)
            tok_counts.append(n_vis if arm == "dense" else module.last_n_out)
            gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            sc = score_fn(gen, r)
            scores.append(sc)
            running_score = sum(scores) / len(scores)
            if hasattr(pbar, "set_postfix_str"):
                pbar.set_postfix_str(f"score={running_score:.3f}")
            if debug and si < debug:
                print(f"  [{arm}/{task}] Q={r['question'][:40]!r} pred={gen.strip()[:50]!r} score={sc:.3f}")
    return sum(scores) / len(scores), sum(tok_counts) / len(tok_counts)


def _format_question(r, task, max_new_tokens):
    """Return (text, gen_max, score_fn) tailored to the task."""
    if task == "anls":
        return (r["question"] + SHORT_ANSWER, max_new_tokens,
                lambda gen, r: anls(_clean_generation(gen), r["answers"]))
    if task == "mc":
        opts = r.get("options", {})
        opt_lines = "\n".join(f"{k}. {v}" for k, v in opts.items())
        prompt = (f"{r['question']}\n{opt_lines}\n"
                  "Answer with the option letter (A, B, C or D) only.")
        gold = r.get("answer", "").strip().upper()
        return prompt, 4, lambda gen, r: float(_extract_letter(gen) == gold)
    if task == "yesno":
        prompt = r["question"] + " Answer yes or no."
        gold = r.get("answer", "").strip().lower()
        return prompt, 4, lambda gen, r: float(_extract_yesno(gen) == gold)
    raise ValueError(f"unknown task {task!r}")


def _extract_letter(text: str) -> str:
    import re
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else ""


def _extract_yesno(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes") or " yes" in t.split(".")[0]:
        return "yes"
    if t.startswith("no") or " no" in t.split(".")[0]:
        return "no"
    return ""


# --------------------------------------------------------------------------- #
# Efficiency timing (TTFT, throughput) -- decoupled from accuracy evaluation. #
# --------------------------------------------------------------------------- #
def time_arm(model, proc, module, data, arm, gen_tokens=64, warmup=2):
    """Per-image TTFT (s) and decode throughput (tokens/s).

    TTFT measured as the wall-clock to produce the FIRST output token from a fully
    prepared input batch (single-image, batch=1) -- the latency users feel before
    streaming starts. Throughput measured over a fixed-length greedy continuation
    of ``gen_tokens`` after the first token. CUDA events used for accurate GPU timing.
    """
    import time
    import torch

    merge = model.model.visual.spatial_merge_size
    ttft, decode_tps, vis_toks = [], [], []
    pbar = _progress(range(len(data)), total=len(data), desc=f"time/{arm}")
    with torch.no_grad():
        for si in pbar:
            r = data[si]
            task = r.get("task", "anls")
            text_q, _, _ = _format_question(r, task, gen_tokens)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": r["image"]},
                {"type": "text", "text": text_q}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[r["image"]], return_tensors="pt").to("cuda")
            thw = inputs["image_grid_thw"]
            n_vis = int((thw.prod(-1) // merge ** 2).sum())

            # TTFT: time to first generated token.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            t_first = time.perf_counter() - t0

            # Decode throughput: gen_tokens additional tokens.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False)
            torch.cuda.synchronize()
            t_gen = time.perf_counter() - t0
            n_new = out.shape[1] - inputs["input_ids"].shape[1]

            if si < warmup:                        # discard warmup iterations
                if hasattr(pbar, "set_postfix_str"):
                    pbar.set_postfix_str("warmup")
                continue
            ttft.append(t_first)
            decode_tps.append(n_new / max(t_gen, 1e-6))
            vis_toks.append(n_vis if arm == "dense" else module.last_n_out)
            if hasattr(pbar, "set_postfix_str"):
                pbar.set_postfix_str(f"TTFT={1000*t_first:.0f}ms tps={n_new/max(t_gen,1e-6):.1f}")

    n = max(len(ttft), 1)
    return dict(
        arm=arm,
        ttft_ms=1000.0 * sum(ttft) / n,
        throughput_tps=sum(decode_tps) / n,
        avg_visual_tokens=sum(vis_toks) / n,
        n_samples=n,
    )
