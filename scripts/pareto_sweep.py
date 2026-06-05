"""Pareto sweep for CSF-Squeeze (REQUIRES GPU + transformers + model).

Sweeps a grid of compression settings for the uniform and csf arms, measures the
accuracy-vs-cost trade-off on DocVQA, and renders a publication-quality Pareto
figure (accuracy vs. average visual tokens) with the dense upper bound.

The thesis to visualize: as compression grows (fewer tokens, leftward), the
content-homogeneous baseline (uniform) degrades sharply on dense text, while
frequency-differentiated compression (csf) stays close to the dense bound.

Usage (ModelScope DSW, 24G):
    python scripts/pareto_sweep.py --subset 100 --rhos 0.5,0.35,0.25,0.15 --strides 2
    python scripts/pareto_sweep.py --from-csv results/pareto.csv   # re-plot only

Outputs (under --outdir, default ./results):
    pareto.csv, pareto.json   raw numbers (reproducible, re-plottable)
    pareto.pdf, pareto.png    the figure (vector + 300-dpi raster)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from _docvqa_eval import load_benchmark, load_model_and_module, configure_arm, eval_arm

# Method display styling (publication palette; colorblind-safe).
STYLE = {
    "dense":   dict(label="Dense (upper bound)", color="#333333"),
    "uniform": dict(label="Uniform (content-homogeneous)", color="#E69F00",
                    marker="s", linestyle="--"),
    "csf":     dict(label="CSF-Squeeze (frequency-differentiated)", color="#0072B2",
                    marker="o", linestyle="-"),
}


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
def run_sweep(args):
    rhos = [float(x) for x in args.rhos.split(",")]
    strides = [int(x) for x in args.strides.split(",")]
    model, proc, module = load_model_and_module(args.model, rhos[0], strides[0], ckpt=args.ckpt)
    data = load_benchmark(args.benchmark, args.subset)
    print(f"[data] {len(data)} {args.benchmark} samples | rhos={rhos} strides={strides}")

    rows = []

    # Dense once (independent of rho/stride) -> the accuracy upper bound.
    configure_arm(module, "dense")
    d_acc, d_tok = eval_arm(model, proc, module, data, "dense", args.max_new_tokens)
    rows.append(dict(arm="dense", rho=None, stride=None, anls=d_acc, vis_tok=d_tok))
    print(f"  dense                : ANLS={d_acc:.4f}  tokens={d_tok:.1f}")

    # uniform and csf over the grid.
    for rho in rhos:
        for s in strides:
            for arm in ("uniform", "csf"):
                configure_arm(module, arm, keep_ratio=rho, stride=s)
                acc, tok = eval_arm(model, proc, module, data, arm, args.max_new_tokens)
                rows.append(dict(arm=arm, rho=rho, stride=s, anls=acc, vis_tok=tok))
                print(f"  {arm:<7} rho={rho:<4} s={s} : ANLS={acc:.4f}  tokens={tok:.1f}")
    return rows, dict(subset=len(data), rhos=rhos, strides=strides,
                      model=args.model, benchmark=args.benchmark)


# --------------------------------------------------------------------------- #
# I/O                                                                          #
# --------------------------------------------------------------------------- #
def save_results(rows, meta, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "pareto.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    with open(os.path.join(outdir, "pareto.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "rho", "stride", "anls", "vis_tok"])
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {outdir}/pareto.csv, pareto.json")


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                arm=r["arm"],
                rho=None if r["rho"] in ("", "None") else float(r["rho"]),
                stride=None if r["stride"] in ("", "None") else int(r["stride"]),
                anls=float(r["anls"]), vis_tok=float(r["vis_tok"]),
            ))
    return rows


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def plot_pareto(rows, outdir, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })

    dense = next((r for r in rows if r["arm"] == "dense"), None)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    # dense upper-bound reference line.
    if dense is not None:
        ax.axhline(dense["anls"], color=STYLE["dense"]["color"], lw=1.2, ls=":",
                   zorder=1, label=STYLE["dense"]["label"])

    # uniform and csf curves, sorted by token count (ascending = more compression).
    for arm in ("uniform", "csf"):
        pts = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["vis_tok"])
        if not pts:
            continue
        xs = [p["vis_tok"] for p in pts]
        ys = [p["anls"] for p in pts]
        st = STYLE[arm]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                lw=1.8, ms=6, mfc=st["color"], mec="white", mew=0.7, zorder=3, label=st["label"])
        # annotate each point with its rho (the primary compression lever).
        for p in pts:
            ax.annotate(rf"$\rho{{=}}{p['rho']:g}$", (p["vis_tok"], p["anls"]),
                        textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=8, color=st["color"])

    ax.set_xlabel("Average visual tokens per image  (fewer = more compression)")
    ax.set_ylabel("DocVQA ANLS")
    ax.set_title(title or "Accuracy vs. compression on dense-text DocVQA", fontsize=11, pad=10)
    ax.invert_xaxis()                       # more compression toward the right
    ax.legend(loc="lower left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    pdf = os.path.join(outdir, "pareto.pdf")
    png = os.path.join(outdir, "pareto.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[saved] {pdf}, {png}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--benchmark", default="docvqa", choices=["docvqa", "infovqa"])
    p.add_argument("--subset", type=int, default=100)
    p.add_argument("--rhos", default="0.5,0.35,0.25,0.15")
    p.add_argument("--strides", default="2")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--outdir", default="results")
    p.add_argument("--from-csv", default=None, help="skip the sweep; re-plot from a saved CSV")
    p.add_argument("--ckpt", default=None,
                   help="trained checkpoint dir (LoRA + csf_router.pt) -> learned router for csf arm")
    args = p.parse_args()

    if args.from_csv:
        rows = load_csv(args.from_csv)
        plot_pareto(rows, args.outdir)
        return

    rows, meta = run_sweep(args)
    save_results(rows, meta, args.outdir)
    title = (f"Accuracy vs. compression on {args.benchmark.upper()} "
             f"(Qwen3-VL-4B, n={meta['subset']})")
    try:
        plot_pareto(rows, args.outdir, title=title)
    except Exception as e:
        print(f"[plot skipped] {type(e).__name__}: {e} (data saved; re-plot with --from-csv)")


if __name__ == "__main__":
    main()
