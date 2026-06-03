"""One-shot experiment runner for the CSF-Squeeze paper (REQUIRES GPU).

Runs the paper's core experiments end-to-end, saving each result block to
``--outdir`` (CSV + JSON) and printing a readable summary with the key gaps.
Designed for a single 24G GPU (ModelScope DSW); ~1-2h at the default subset
sizes, much less with --quick.

Blocks (toggle with --only):
    pareto-doc     Pareto sweep on DocVQA (the headline accuracy-vs-cost figure)
    pareto-info    Pareto sweep on InfographicVQA (denser text, csf's home turf)
    ablation-ds    DeepStack-consistent vs DeepStack-unaware (csf vs csf-naive)
    confirm        single-point 3-arm confirmation (dense/uniform/csf) at --rho

Usage:
    python scripts/run_all.py                       # full run, all blocks
    python scripts/run_all.py --quick               # small subsets to smoke the pipeline
    python scripts/run_all.py --only pareto-doc,ablation-ds
    python scripts/run_all.py --from-dir results    # re-plot/re-summarize saved CSVs
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from _docvqa_eval import load_benchmark, load_model_and_module, configure_arm, eval_arm


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _save(rows, meta, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    fields = ["arm", "rho", "stride", "anls", "vis_tok"]
    with open(os.path.join(outdir, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  [saved] {outdir}/{tag}.csv")


def _load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                arm=r["arm"],
                rho=None if r["rho"] in ("", "None") else float(r["rho"]),
                stride=None if r["stride"] in ("", "None") else int(r["stride"]),
                anls=float(r["anls"]), vis_tok=float(r["vis_tok"])))
    return rows


def _plot_if_possible(rows, outdir, tag, title):
    try:
        from pareto_sweep import plot_pareto
        # plot_pareto writes pareto.{pdf,png}; render into a tagged subdir to avoid clobber.
        sub = os.path.join(outdir, tag)
        os.makedirs(sub, exist_ok=True)
        plot_pareto(rows, sub, title=title)
    except Exception as e:
        print(f"  [plot skipped] {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# experiment blocks                                                            #
# --------------------------------------------------------------------------- #
def block_pareto(model, proc, module, benchmark, subset, rhos, strides, mnt):
    data = load_benchmark(benchmark, subset)
    rows = []
    configure_arm(module, "dense")
    acc, tok = eval_arm(model, proc, module, data, "dense", mnt)
    rows.append(dict(arm="dense", rho=None, stride=None, anls=acc, vis_tok=tok))
    print(f"  dense                : ANLS={acc:.4f} tok={tok:.1f}")
    for rho in rhos:
        for s in strides:
            for arm in ("uniform", "csf"):
                configure_arm(module, arm, keep_ratio=rho, stride=s)
                acc, tok = eval_arm(model, proc, module, data, arm, mnt)
                rows.append(dict(arm=arm, rho=rho, stride=s, anls=acc, vis_tok=tok))
                print(f"  {arm:<7} rho={rho:<4} s={s} : ANLS={acc:.4f} tok={tok:.1f}")
    return rows, dict(benchmark=benchmark, subset=len(data), rhos=rhos, strides=strides)


def block_arms(model, proc, module, benchmark, subset, rho, stride, arms, mnt):
    data = load_benchmark(benchmark, subset)
    rows = []
    for arm in arms:
        configure_arm(module, arm, keep_ratio=rho, stride=stride)
        acc, tok = eval_arm(model, proc, module, data, arm, mnt)
        rows.append(dict(arm=arm, rho=rho, stride=stride, anls=acc, vis_tok=tok))
        print(f"  {arm:<10} rho={rho} s={stride} : ANLS={acc:.4f} tok={tok:.1f}")
    return rows, dict(benchmark=benchmark, subset=len(data), rho=rho, stride=stride, arms=arms)


def _gap(rows, a, b):
    da = next((r for r in rows if r["arm"] == a), None)
    db = next((r for r in rows if r["arm"] == b), None)
    if da and db:
        return da["anls"] - db["anls"]
    return None


# --------------------------------------------------------------------------- #
def summarize(blocks):
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for tag, rows in blocks.items():
        print(f"\n[{tag}]")
        for r in rows:
            cfg = "" if r["rho"] is None else f"rho={r['rho']} s={r['stride']}"
            print(f"  {r['arm']:<10} {cfg:<14} ANLS={r['anls']:.4f}  tok={r['vis_tok']:.1f}")
        # headline gaps
        for a, b, label in [("csf", "uniform", "freq-differentiation"),
                            ("csf", "csf-naive", "DeepStack-consistent propagation")]:
            g = _gap(rows, a, b)
            if g is not None:
                verdict = "CONFIRMED" if g > 0 else "NOT confirmed"
                print(f"    -> ANLS({a}) - ANLS({b}) = {g:+.4f}  [{label}: {verdict}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--only", default="", help="comma subset of: pareto-doc,pareto-info,ablation-ds,confirm")
    p.add_argument("--rhos", default="0.5,0.35,0.25,0.15")
    p.add_argument("--strides", default="2")
    p.add_argument("--rho", type=float, default=0.35, help="operating point for confirm/ablation blocks")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--pareto-subset", type=int, default=100)
    p.add_argument("--point-subset", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--outdir", default="results")
    p.add_argument("--quick", action="store_true", help="tiny subsets to smoke the full pipeline")
    p.add_argument("--from-dir", default=None, help="skip GPU runs; summarize/plot saved CSVs in this dir")
    args = p.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    strides = [int(x) for x in args.strides.split(",")]
    blocks_wanted = [b for b in (args.only.split(",") if args.only else
                                 ["pareto-doc", "pareto-info", "ablation-ds", "confirm"]) if b]

    # Re-summarize / re-plot from disk without touching the GPU.
    if args.from_dir:
        blocks = {}
        for tag in ("pareto-doc", "pareto-info", "ablation-ds", "confirm"):
            path = os.path.join(args.from_dir, f"{tag}.csv")
            if os.path.exists(path):
                blocks[tag] = _load_csv(path)
                if tag.startswith("pareto"):
                    _plot_if_possible(blocks[tag], args.from_dir, tag, tag)
        summarize(blocks)
        return

    p_sub = 8 if args.quick else args.pareto_subset
    c_sub = 8 if args.quick else args.point_subset
    mnt = args.max_new_tokens

    model, proc, module = load_model_and_module(args.model, rhos[0], strides[0])
    blocks = {}

    if "pareto-doc" in blocks_wanted:
        print("\n=== Block: Pareto on DocVQA ===")
        rows, meta = block_pareto(model, proc, module, "docvqa", p_sub, rhos, strides, mnt)
        _save(rows, meta, args.outdir, "pareto-doc")
        _plot_if_possible(rows, args.outdir, "pareto-doc",
                          f"Accuracy vs. compression on DOCVQA (Qwen3-VL-4B, n={meta['subset']})")
        blocks["pareto-doc"] = rows

    if "pareto-info" in blocks_wanted:
        print("\n=== Block: Pareto on InfographicVQA ===")
        rows, meta = block_pareto(model, proc, module, "infovqa", p_sub, rhos, strides, mnt)
        _save(rows, meta, args.outdir, "pareto-info")
        _plot_if_possible(rows, args.outdir, "pareto-info",
                          f"Accuracy vs. compression on INFOVQA (Qwen3-VL-4B, n={meta['subset']})")
        blocks["pareto-info"] = rows

    if "ablation-ds" in blocks_wanted:
        print("\n=== Block: DeepStack propagation ablation (DocVQA) ===")
        rows, meta = block_arms(model, proc, module, "docvqa", c_sub, args.rho, args.stride,
                                ["dense", "csf", "csf-naive"], mnt)
        _save(rows, meta, args.outdir, "ablation-ds")
        blocks["ablation-ds"] = rows

    if "confirm" in blocks_wanted:
        print("\n=== Block: 3-arm confirmation (DocVQA) ===")
        rows, meta = block_arms(model, proc, module, "docvqa", c_sub, args.rho, args.stride,
                                ["dense", "uniform", "csf"], mnt)
        _save(rows, meta, args.outdir, "confirm")
        blocks["confirm"] = rows

    summarize(blocks)


if __name__ == "__main__":
    main()
