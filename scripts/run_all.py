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

from _docvqa_eval import load_benchmark, load_model_and_module, configure_arm, eval_arm, time_arm


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _save(rows, meta, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    # widen field set so efficiency rows (ttft_ms, throughput_tps) round-trip too.
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(os.path.join(outdir, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"  [saved] {outdir}/{tag}.csv")


def _load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in ("", "None", None):
                    row[k] = None
                elif k in ("rho", "anls", "vis_tok", "ttft_ms", "throughput_tps", "score"):
                    row[k] = float(v)
                elif k == "stride":
                    row[k] = int(v)
                else:
                    row[k] = v
            rows.append(row)
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
        print(f"  {arm:<10} rho={rho} s={stride} : score={acc:.4f} tok={tok:.1f}")
    return rows, dict(benchmark=benchmark, subset=len(data), rho=rho, stride=stride, arms=arms)


def block_general(model, proc, module, benchmark, subset, rho, stride, mnt):
    """csf vs dense on a non-document benchmark (MMBench / POPE) -- 'do no harm' check."""
    data = load_benchmark(benchmark, subset)
    rows = []
    for arm in ("dense", "csf"):
        configure_arm(module, arm, keep_ratio=rho, stride=stride)
        acc, tok = eval_arm(model, proc, module, data, arm, mnt)
        rows.append(dict(arm=arm, rho=rho if arm != "dense" else None,
                         stride=stride if arm != "dense" else None,
                         score=acc, vis_tok=tok))
        print(f"  {arm:<6} {benchmark:<8} : acc={acc:.4f} tok={tok:.1f}")
    return rows, dict(benchmark=benchmark, subset=len(data), rho=rho, stride=stride)


def block_efficiency(model, proc, module, subset, rho, stride, gen_tokens):
    """TTFT and decode throughput on DocVQA (latency users feel)."""
    data = load_benchmark("docvqa", subset)
    rows = []
    print(f"  {'arm':<10}{'TTFT(ms)':>10}{'tps':>10}{'avg_tok':>12}")
    for label in ("dense", "csf-50", "csf-35", "csf-25"):
        if label == "dense":
            configure_arm(module, "dense")
        else:
            r = int(label.split("-")[1]) / 100.0
            configure_arm(module, "csf", keep_ratio=r, stride=stride)
        m = time_arm(model, proc, module, data, label, gen_tokens=gen_tokens, warmup=2)
        # store with the same column conventions as other blocks for easy CSV reuse.
        rows.append(dict(arm=label, rho=None if label == "dense" else r,
                         stride=None if label == "dense" else stride,
                         ttft_ms=m["ttft_ms"], throughput_tps=m["throughput_tps"],
                         vis_tok=m["avg_visual_tokens"]))
        print(f"  {label:<10}{m['ttft_ms']:>10.1f}{m['throughput_tps']:>10.2f}{m['avg_visual_tokens']:>12.1f}")
    dense = rows[0]
    for r in rows[1:]:
        ttft_x = dense["ttft_ms"] / max(r["ttft_ms"], 1e-6)
        print(f"    {r['arm']:<10}: TTFT speedup {ttft_x:.2f}x | tokens {dense['vis_tok']/max(r['vis_tok'],1e-6):.2f}x cut")
    return rows, dict(subset=len(data), gen_tokens=gen_tokens, rho=rho, stride=stride)


def _score(r):
    """Extract the accuracy/score number regardless of column name (anls/score)."""
    if "anls" in r and r.get("anls") is not None:
        return r["anls"]
    if "score" in r and r.get("score") is not None:
        return r["score"]
    return None


def _gap(rows, a, b):
    da = next((r for r in rows if r["arm"] == a), None)
    db = next((r for r in rows if r["arm"] == b), None)
    if da and db:
        sa, sb = _score(da), _score(db)
        if sa is not None and sb is not None:
            return sa - sb
    return None


# --------------------------------------------------------------------------- #
def summarize(blocks):
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for tag, rows in blocks.items():
        print(f"\n[{tag}]")
        for r in rows:
            cfg = ""
            if r.get("rho") is not None:
                cfg = f"rho={r['rho']} s={r['stride']}"
            if "ttft_ms" in r:
                print(f"  {r['arm']:<10} {cfg:<14} TTFT={r['ttft_ms']:.1f}ms  "
                      f"tps={r['throughput_tps']:.2f}  tok={r['vis_tok']:.1f}")
            else:
                s = _score(r)
                key = "anls" if "anls" in r and r.get("anls") is not None else "acc"
                print(f"  {r['arm']:<10} {cfg:<14} {key}={s if s is not None else 0:.4f}  "
                      f"tok={r.get('vis_tok', 0):.1f}")
        # headline gaps (only meaningful for accuracy blocks).
        for a, b, label in [("csf", "uniform", "freq-differentiation"),
                            ("csf", "csf-naive", "DeepStack-consistent propagation")]:
            g = _gap(rows, a, b)
            if g is not None:
                verdict = "CONFIRMED" if g > 0 else "NOT confirmed"
                print(f"    -> ANLS({a}) - ANLS({b}) = {g:+.4f}  [{label}: {verdict}]")
        # general blocks: dense vs csf delta -> "do no harm" check.
        if tag in ("mmbench", "pope"):
            d = _gap(rows, "csf", "dense")
            if d is not None:
                msg = "no harm" if d >= -0.01 else f"degraded by {-d:.3f}"
                print(f"    -> csf - dense = {d:+.4f}  [general capability: {msg}]")
        # efficiency: speedup vs dense.
        if tag == "efficiency":
            dense = next((r for r in rows if r["arm"] == "dense"), None)
            if dense is not None:
                for r in rows:
                    if r["arm"] == "dense":
                        continue
                    sp = dense["ttft_ms"] / max(r["ttft_ms"], 1e-6)
                    print(f"    -> {r['arm']:<10} TTFT speedup vs dense = {sp:.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--only", default="",
                   help="comma subset of: pareto-doc,pareto-info,ablation-ds,confirm,mmbench,pope,efficiency")
    p.add_argument("--rhos", default="0.5,0.35,0.25,0.15")
    p.add_argument("--strides", default="2")
    p.add_argument("--rho", type=float, default=0.35, help="operating point for confirm/ablation/general blocks")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--pareto-subset", type=int, default=100)
    p.add_argument("--point-subset", type=int, default=200)
    p.add_argument("--general-subset", type=int, default=200, help="size for MMBench/POPE")
    p.add_argument("--efficiency-subset", type=int, default=30, help="size for TTFT/throughput timing")
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--outdir", default="results")
    p.add_argument("--quick", action="store_true", help="tiny subsets to smoke the full pipeline")
    p.add_argument("--from-dir", default=None, help="skip GPU runs; summarize/plot saved CSVs in this dir")
    p.add_argument("--ckpt", default=None,
                   help="path to a trained checkpoint dir (LoRA adapter + csf_router.pt) "
                        "from scripts/train_csf.py; switches the csf arm to the learned router")
    args = p.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    strides = [int(x) for x in args.strides.split(",")]
    blocks_wanted = [b for b in (args.only.split(",") if args.only else
                                 ["pareto-doc", "pareto-info", "ablation-ds", "confirm",
                                  "mmbench", "pope", "efficiency"]) if b]

    # Re-summarize / re-plot from disk without touching the GPU.
    if args.from_dir:
        blocks = {}
        for tag in ("pareto-doc", "pareto-info", "ablation-ds", "confirm",
                    "mmbench", "pope", "efficiency"):
            path = os.path.join(args.from_dir, f"{tag}.csv")
            if os.path.exists(path):
                blocks[tag] = _load_csv(path)
                if tag.startswith("pareto"):
                    _plot_if_possible(blocks[tag], args.from_dir, tag, tag)
        summarize(blocks)
        return

    p_sub = 8 if args.quick else args.pareto_subset
    c_sub = 8 if args.quick else args.point_subset
    g_sub = 8 if args.quick else args.general_subset
    e_sub = 4 if args.quick else args.efficiency_subset
    mnt = args.max_new_tokens

    model, proc, module = load_model_and_module(args.model, rhos[0], strides[0], ckpt=args.ckpt)
    blocks = {}

    # Top-level block progress: lets the user see overall ETA across blocks.
    try:
        from tqdm import tqdm as _tqdm
        outer = _tqdm(total=len(blocks_wanted), desc="blocks", position=0,
                      bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    except Exception:
        outer = None
    def _tick(name):
        if outer is not None:
            outer.update(1)
            outer.set_description_str(f"blocks (last={name})")

    if "pareto-doc" in blocks_wanted:
        print("\n=== Block: Pareto on DocVQA ===")
        rows, meta = block_pareto(model, proc, module, "docvqa", p_sub, rhos, strides, mnt)
        _save(rows, meta, args.outdir, "pareto-doc")
        _plot_if_possible(rows, args.outdir, "pareto-doc",
                          f"Accuracy vs. compression on DOCVQA (Qwen3-VL-4B, n={meta['subset']})")
        blocks["pareto-doc"] = rows
        _tick("pareto-doc")

    if "pareto-info" in blocks_wanted:
        print("\n=== Block: Pareto on InfographicVQA ===")
        rows, meta = block_pareto(model, proc, module, "infovqa", p_sub, rhos, strides, mnt)
        _save(rows, meta, args.outdir, "pareto-info")
        _plot_if_possible(rows, args.outdir, "pareto-info",
                          f"Accuracy vs. compression on INFOVQA (Qwen3-VL-4B, n={meta['subset']})")
        blocks["pareto-info"] = rows
        _tick("pareto-info")

    if "ablation-ds" in blocks_wanted:
        print("\n=== Block: DeepStack propagation ablation (DocVQA) ===")
        rows, meta = block_arms(model, proc, module, "docvqa", c_sub, args.rho, args.stride,
                                ["dense", "csf", "csf-naive"], mnt)
        _save(rows, meta, args.outdir, "ablation-ds")
        blocks["ablation-ds"] = rows
        _tick("ablation-ds")

    if "confirm" in blocks_wanted:
        print("\n=== Block: 3-arm confirmation (DocVQA) ===")
        rows, meta = block_arms(model, proc, module, "docvqa", c_sub, args.rho, args.stride,
                                ["dense", "uniform", "csf"], mnt)
        _save(rows, meta, args.outdir, "confirm")
        blocks["confirm"] = rows
        _tick("confirm")

    if "mmbench" in blocks_wanted:
        print("\n=== Block: MMBench (do-no-harm on general capability) ===")
        rows, meta = block_general(model, proc, module, "mmbench", g_sub, args.rho, args.stride, mnt)
        _save(rows, meta, args.outdir, "mmbench")
        blocks["mmbench"] = rows
        _tick("mmbench")

    if "pope" in blocks_wanted:
        print("\n=== Block: POPE (object hallucination) ===")
        rows, meta = block_general(model, proc, module, "pope", g_sub, args.rho, args.stride, mnt)
        _save(rows, meta, args.outdir, "pope")
        blocks["pope"] = rows
        _tick("pope")

    if "efficiency" in blocks_wanted:
        print("\n=== Block: Efficiency (TTFT, throughput on DocVQA) ===")
        rows, meta = block_efficiency(model, proc, module, e_sub, args.rho, args.stride, args.gen_tokens)
        _save(rows, meta, args.outdir, "efficiency")
        blocks["efficiency"] = rows
        _tick("efficiency")

    if outer is not None:
        outer.close()
    summarize(blocks)


if __name__ == "__main__":
    main()
