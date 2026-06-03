"""Decisive 'kill-or-confirm' experiment (REQUIRES GPU + transformers + model).

Question (Sec. 4.4): at MATCHED visual-token count on dense-text DocVQA, does
frequency-DIFFERENTIATED compression (csf) beat content-HOMOGENEOUS compression
(uniform) -- and stay close to the dense upper bound?

Three arms share ONE model (toggled, no reload):
    dense   : no compression (upper bound)
    uniform : exempt the SAME rho fraction, content-agnostically
    csf     : exempt the top-rho fraction by high-frequency saliency
uniform and csf use identical rho/stride, so token counts match; the only
difference is WHICH tokens are exempt -> isolates frequency-awareness.

Usage (ModelScope DSW, 24G):
    python scripts/run_kill_experiment.py --subset 200 --rho 0.25 --stride 2
    python scripts/run_kill_experiment.py --subset 20 --arms dense --debug 5
"""
from __future__ import annotations

import argparse

from _docvqa_eval import load_benchmark, load_model_and_module, configure_arm, eval_arm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--benchmark", default="docvqa", choices=["docvqa", "infovqa"])
    p.add_argument("--subset", type=int, default=200)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--arms", default="dense,uniform,csf",
                   help="comma list of: dense,uniform,csf,csf-naive")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--debug", type=int, default=0, help="print this many (q, gold, pred, anls) samples")
    args = p.parse_args()

    model, proc, module = load_model_and_module(args.model, args.rho, args.stride)
    data = load_benchmark(args.benchmark, args.subset)
    print(f"[data] {len(data)} {args.benchmark} samples | rho={args.rho} stride={args.stride}")

    print(f"\n{'arm':<12}{'ANLS':>10}{'avg_vis_tok':>14}")
    results = {}
    for arm in args.arms.split(","):
        configure_arm(module, arm, keep_ratio=args.rho, stride=args.stride)
        acc, toks = eval_arm(model, proc, module, data, arm,
                             max_new_tokens=args.max_new_tokens, debug=args.debug)
        results[arm] = (acc, toks)
        print(f"{arm:<12}{acc:>10.4f}{toks:>14.1f}")

    if "csf" in results and "uniform" in results:
        gap = results["csf"][0] - results["uniform"][0]
        print(f"\n[confirm criterion] ANLS(csf) - ANLS(uniform) = {gap:+.4f} "
              f"(>0 at matched tokens confirms frequency-differentiation; "
              f"also check csf close to dense).")
    if "csf" in results and "csf-naive" in results:
        d = results["csf"][0] - results["csf-naive"][0]
        print(f"[ablation] ANLS(csf) - ANLS(csf-naive) = {d:+.4f} "
              f"(>0 quantifies the value of DeepStack-consistent propagation).")


if __name__ == "__main__":
    main()
