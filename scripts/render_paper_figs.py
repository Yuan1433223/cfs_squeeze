"""Render publication figures from the saved CSVs in result/.

Reads four CSVs and writes matched .pdf + .png pairs into result/result_pdf/:
    pareto-doc (1).csv   -> pareto_docvqa.{pdf,png}    (Pareto, DocVQA)
    pareto-info (1).csv  -> pareto_infovqa.{pdf,png}   (Pareto, InfoVQA)
    confirm (1).csv      -> confirm_docvqa.{pdf,png}   (3-bar: dense/uniform/csf)
    ablation-ds (1).csv  -> ablation_deepstack.{pdf,png} (3-bar: dense/csf/csf-naive)

Re-uses the publication palette / font setup from pareto_sweep.py so all four
figures look like one paper. Run on CPU; only matplotlib is required.

    python scripts/render_paper_figs.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Publication palette (colorblind-safe, same as pareto_sweep.py).
COLOR = {
    "dense":     "#333333",
    "uniform":   "#E69F00",
    "csf":       "#0072B2",
    "csf-naive": "#D55E00",
}
LABEL = {
    "dense":     "Dense (upper bound)",
    "uniform":   "Uniform (content-homogeneous)",
    "csf":       "CSF-Squeeze (frequency-differentiated)",
    "csf-naive": "CSF w/o DeepStack-consistent prop.",
}
MARKER = {"uniform": "s", "csf": "o"}
LS = {"uniform": "--", "csf": "-"}


def _set_rc():
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


def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                arm=r["arm"].strip(),
                rho=None if r["rho"] in ("", "None") else float(r["rho"]),
                stride=None if r["stride"] in ("", "None") else int(r["stride"]),
                anls=float(r["anls"]),
                vis_tok=float(r["vis_tok"]),
            ))
    return rows


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, stem + ".pdf")
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")


def _read_efficiency(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                arm=r["arm"].strip(),
                rho=None if r["rho"] in ("", "None") else float(r["rho"]),
                ttft_ms=float(r["ttft_ms"]),
                throughput_tps=float(r["throughput_tps"]),
                vis_tok=float(r["vis_tok"]),
            ))
    return rows


def _read_score_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                arm=r["arm"].strip(),
                rho=None if r["rho"] in ("", "None") else float(r["rho"]),
                score=float(r["score"]),
                vis_tok=float(r["vis_tok"]),
            ))
    return rows


# --------------------------------------------------------------------------- #
def plot_pareto(rows, outdir, stem, benchmark_label, ylabel="ANLS"):
    _set_rc()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    dense = next((r for r in rows if r["arm"] == "dense"), None)
    if dense is not None:
        ax.axhline(dense["anls"], color=COLOR["dense"], lw=1.2, ls=":",
                   zorder=1, label=LABEL["dense"])
    for arm in ("uniform", "csf"):
        pts = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["vis_tok"])
        if not pts:
            continue
        xs = [p["vis_tok"] for p in pts]
        ys = [p["anls"] for p in pts]
        ax.plot(xs, ys, color=COLOR[arm], marker=MARKER[arm], linestyle=LS[arm],
                lw=1.8, ms=6, mfc=COLOR[arm], mec="white", mew=0.7, zorder=3,
                label=LABEL[arm])
        for p in pts:
            ax.annotate(rf"$\rho{{=}}{p['rho']:g}$", (p["vis_tok"], p["anls"]),
                        textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=8, color=COLOR[arm])
    ax.set_xlabel("Average visual tokens per image  (fewer = more compression)")
    ax.set_ylabel(f"{benchmark_label} {ylabel}")
    ax.set_title(f"Accuracy vs. compression on {benchmark_label} (Qwen3-VL-4B)",
                 fontsize=11, pad=10)
    ax.invert_xaxis()
    ax.legend(loc="lower left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save(fig, outdir, stem)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def plot_bars(rows, order, outdir, stem, title, ylabel="DocVQA ANLS",
              annotate_tokens=True):
    """Grouped bar chart: ANLS bar (left) + visual tokens annotation (right)."""
    _set_rc()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    by_arm = {r["arm"]: r for r in rows}
    arms = [a for a in order if a in by_arm]
    xs = list(range(len(arms)))
    ys = [by_arm[a]["anls"] for a in arms]
    colors = [COLOR.get(a, "#777777") for a in arms]
    bars = ax.bar(xs, ys, color=colors, width=0.55, edgecolor="white", linewidth=0.7)
    for b, a in zip(bars, arms):
        h = b.get_height()
        txt = f"{h:.3f}"
        if annotate_tokens:
            txt += f"\n({by_arm[a]['vis_tok']:.0f} tok)"
        ax.text(b.get_x() + b.get_width()/2, h + 0.015, txt,
                ha="center", va="bottom", fontsize=9, color="#222222")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(a, a) for a in arms], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(ys) * 1.18)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(axis="x", visible=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save(fig, outdir, stem)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def plot_efficiency(rows, outdir, stem):
    """Two-panel efficiency figure: TTFT and throughput vs visual tokens."""
    _set_rc()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    dense = next((r for r in rows if r["arm"] == "dense"), None)
    csf_pts = sorted([r for r in rows if r["arm"].startswith("csf")],
                     key=lambda r: r["vis_tok"])

    ax = axes[0]
    if dense is not None:
        ax.axhline(dense["ttft_ms"], color=COLOR["dense"], lw=1.2, ls=":",
                   zorder=1, label=f"Dense ({dense['ttft_ms']:.0f} ms)")
    xs = [p["vis_tok"] for p in csf_pts]
    ys = [p["ttft_ms"] for p in csf_pts]
    ax.plot(xs, ys, color=COLOR["csf"], marker="o", linestyle="-",
            lw=1.8, ms=6, mfc=COLOR["csf"], mec="white", mew=0.7, zorder=3,
            label="CSF-Squeeze")
    for p in csf_pts:
        ax.annotate(rf"$\rho{{=}}{p['rho']:g}$", (p["vis_tok"], p["ttft_ms"]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=COLOR["csf"])
    ax.set_xlabel("Average visual tokens per image")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time-to-first-token vs. compression", fontsize=11, pad=10)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    if dense is not None:
        ax.axhline(dense["throughput_tps"], color=COLOR["dense"], lw=1.2, ls=":",
                   zorder=1, label=f"Dense ({dense['throughput_tps']:.2f} tok/s)")
    ys = [p["throughput_tps"] for p in csf_pts]
    ax.plot(xs, ys, color=COLOR["csf"], marker="o", linestyle="-",
            lw=1.8, ms=6, mfc=COLOR["csf"], mec="white", mew=0.7, zorder=3,
            label="CSF-Squeeze")
    for p in csf_pts:
        ax.annotate(rf"$\rho{{=}}{p['rho']:g}$",
                    (p["vis_tok"], p["throughput_tps"]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=COLOR["csf"])
    ax.set_xlabel("Average visual tokens per image")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.set_title("Decoding throughput vs. compression", fontsize=11, pad=10)
    ax.invert_xaxis()
    ax.legend(loc="lower left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Efficiency on Qwen3-VL-4B (single 24G GPU, bf16)",
                 fontsize=11.5, y=1.02)
    _save(fig, outdir, stem)
    plt.close(fig)


def plot_cross_benchmark(mmbench_rows, pope_rows, outdir, stem):
    """Side-by-side bar comparison: dense vs csf on MMBench and POPE."""
    _set_rc()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, rows, title, ylabel in [
        (axes[0], mmbench_rows, "MMBench (multiple-choice accuracy)", "Score"),
        (axes[1], pope_rows, "POPE (hallucination, accuracy)", "Score"),
    ]:
        by_arm = {r["arm"]: r for r in rows}
        arms = [a for a in ("dense", "csf") if a in by_arm]
        xs = list(range(len(arms)))
        ys = [by_arm[a]["score"] for a in arms]
        colors = [COLOR.get(a, "#777777") for a in arms]
        bars = ax.bar(xs, ys, color=colors, width=0.5, edgecolor="white",
                      linewidth=0.7)
        for b, a in zip(bars, arms):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012,
                    f"{h:.3f}\n({by_arm[a]['vis_tok']:.0f} tok)",
                    ha="center", va="bottom", fontsize=9, color="#222222")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL.get(a, a) for a in arms], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(ys) * 1.20)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(r"Cross-benchmark generalization at $\rho{=}0.35$ "
                 "(Qwen3-VL-4B)", fontsize=11.5, y=1.02)
    _save(fig, outdir, stem)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src = os.path.join(root, "result")
    out = os.path.join(src, "result_pdf")

    plot_pareto(_read_csv(os.path.join(src, "pareto-doc (1).csv")),
                out, "pareto_docvqa", "DocVQA")
    plot_pareto(_read_csv(os.path.join(src, "pareto-info (1).csv")),
                out, "pareto_infovqa", "InfographicVQA")

    plot_bars(_read_csv(os.path.join(src, "confirm (1).csv")),
              order=["dense", "uniform", "csf"],
              outdir=out, stem="confirm_docvqa",
              title=r"DocVQA confirmation at $\rho{=}0.35$, stride $s{=}2$")

    plot_bars(_read_csv(os.path.join(src, "ablation-ds (1).csv")),
              order=["dense", "csf", "csf-naive"],
              outdir=out, stem="ablation_deepstack",
              title=r"DeepStack-consistent propagation ablation ($\rho{=}0.35$)")

    plot_efficiency(_read_efficiency(os.path.join(src, "efficiency.csv")),
                    out, "efficiency")

    plot_cross_benchmark(
        mmbench_rows=_read_score_csv(os.path.join(src, "mmbench.csv")),
        pope_rows=_read_score_csv(os.path.join(src, "pope.csv")),
        outdir=out, stem="cross_benchmark",
    )

    print("[done] all figures rendered to", out)


if __name__ == "__main__":
    main()
