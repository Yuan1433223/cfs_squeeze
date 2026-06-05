"""Efficiency benchmark: TTFT and decode throughput (REQUIRES GPU).

Measures the engineering payoff of CSF-Squeeze on a single GPU at batch size 1,
which is the latency users feel before streaming starts. Three arms compared at
matched protocol; only the visual-token count differs.

Outputs (under --outdir):
    efficiency.csv, efficiency.json   per-arm metrics

Run on the ModelScope DSW (24G):
    python scripts/run_efficiency.py --subset 30 --rho 0.35 --gen-tokens 64
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from _docvqa_eval import load_benchmark, load_model_and_module, configure_arm, time_arm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--benchmark", default="docvqa", choices=["docvqa", "infovqa"])
    p.add_argument("--subset", type=int, default=30,
                   help="number of images to time (small is fine; we report the mean)")
    p.add_argument("--rho", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--arms", default="dense,csf-25,csf-35",
                   help="comma list; csf-XX uses rho=0.XX (XX in {15,25,35,50})")
    p.add_argument("--gen-tokens", type=int, default=64,
                   help="decode-throughput window after the first token")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    model, proc, module = load_model_and_module(args.model, args.rho, args.stride)
    data = load_benchmark(args.benchmark, args.subset)
    print(f"[data] {len(data)} {args.benchmark} samples | warmup={args.warmup} gen_tokens={args.gen_tokens}")

    rows = []
    print(f"\n{'arm':<10}{'TTFT(ms)':>12}{'tps':>10}{'avg_vis_tok':>14}")
    for arm_spec in args.arms.split(","):
        # Allow shortcuts like csf-35 -> arm=csf at rho=0.35.
        if arm_spec.startswith("csf-"):
            rho = int(arm_spec.split("-")[1]) / 100.0
            configure_arm(module, "csf", keep_ratio=rho, stride=args.stride)
            label = arm_spec
        elif arm_spec.startswith("uniform-"):
            rho = int(arm_spec.split("-")[1]) / 100.0
            configure_arm(module, "uniform", keep_ratio=rho, stride=args.stride)
            label = arm_spec
        else:
            configure_arm(module, arm_spec, keep_ratio=args.rho, stride=args.stride)
            label = arm_spec

        m = time_arm(model, proc, module, data, label,
                     gen_tokens=args.gen_tokens, warmup=args.warmup)
        rows.append(m)
        print(f"{label:<10}{m['ttft_ms']:>12.1f}{m['throughput_tps']:>10.2f}{m['avg_visual_tokens']:>14.1f}")

    # Speedups vs dense.
    dense = next((r for r in rows if r["arm"] == "dense"), None)
    if dense is not None:
        print()
        for r in rows:
            if r["arm"] == "dense":
                continue
            ttft_x = dense["ttft_ms"] / max(r["ttft_ms"], 1e-6)
            tps_x = r["throughput_tps"] / max(dense["throughput_tps"], 1e-6)
            tok_x = dense["avg_visual_tokens"] / max(r["avg_visual_tokens"], 1e-6)
            print(f"  {r['arm']:<10}: TTFT speedup {ttft_x:.2f}x | "
                  f"throughput {tps_x:.2f}x | tokens cut {tok_x:.2f}x")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "efficiency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.outdir, "efficiency.json"), "w") as f:
        json.dump({"meta": vars(args), "rows": rows}, f, indent=2)
    print(f"\n[saved] {args.outdir}/efficiency.csv")


if __name__ == "__main__":
    main()
