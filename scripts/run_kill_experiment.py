"""Decisive 'kill-or-confirm' experiment skeleton (REQUIRES GPU + transformers).

Goal (Sec. 4.4): at MATCHED compression ratio on a dense-text benchmark
(DocVQA / InfoVQA), does frequency-DIFFERENTIATED compression (CSF-Squeeze) beat
content-HOMOGENEOUS compression (uniform downsample of all tokens)? If yes on a
small subset, the paper's core thesis is confirmed and full runs are justified.

This file does NOT run on the CPU-only dev box. It is GPU-ready scaffolding:
    pip/uv add transformers accelerate datasets pillow
    python scripts/run_kill_experiment.py --model Qwen/Qwen3-VL-4B-Instruct \
        --benchmark docvqa --subset 200 --rho 0.25 --stride 2

Both arms use the SAME DeepStack-consistent propagation; they differ ONLY in the
token-selection policy, isolating the effect of frequency-awareness.
"""
from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CSF-Squeeze kill experiment (GPU)")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--benchmark", default="docvqa", choices=["docvqa", "infovqa", "textvqa"])
    p.add_argument("--subset", type=int, default=200, help="number of eval samples")
    p.add_argument("--rho", type=float, default=0.25, help="high-frequency keep ratio")
    p.add_argument("--stride", type=int, default=2, help="low-frequency pooling stride")
    p.add_argument("--arm", default="all", choices=["dense", "uniform", "csf", "all"])
    p.add_argument("--device", default="cuda")
    return p


def main():  # pragma: no cover - needs GPU + transformers
    args = build_argparser().parse_args()

    # Deferred heavy imports so the repo stays importable on the CPU dev box.
    import torch  # noqa: F401
    from transformers import AutoConfig  # noqa: F401
    from csf_squeeze.integrate_qwen3vl import build_module_from_hf  # noqa: F401

    raise NotImplementedError(
        "GPU experiment scaffold. Steps to implement on the rented single GPU:\n"
        "  1. Load Qwen3-VL-4B (+ LoRA r=64) and its processor.\n"
        "  2. cfg = AutoConfig.from_pretrained(args.model); "
        "module = build_module_from_hf(cfg, keep_ratio=args.rho, downsample_stride=args.stride).\n"
        "  3. Define three arms sharing DeepStack-consistent propagation:\n"
        "       - dense  : no compression (accuracy upper bound).\n"
        "       - uniform: replace the selection policy with uniform downsample of\n"
        "                  ALL tokens to the SAME average token count as CSF.\n"
        "       - csf    : frequency-differentiated (high-freq exempt + low-freq pool).\n"
        "  4. Evaluate ANLS/accuracy on the benchmark subset; log avg visual tokens,\n"
        "     TTFT, throughput.\n"
        "  5. Confirm criterion: at matched token count, csf - uniform > 0 on dense\n"
        "     text, and csf ~ dense within tolerance. If not, pivot before scaling.\n"
    )


if __name__ == "__main__":
    main()
