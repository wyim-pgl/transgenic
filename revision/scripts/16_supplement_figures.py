#!/usr/bin/env python3
"""Supplementary revision figures: S2 (TSS/TES) and S3 (GSF vocabulary coverage)."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "Figures"))
import figstyle as _figstyle
_figstyle.apply(8)
import numpy as np

BASE = Path("/data/gpfs/assoc/pgl/data/Transgenic")
REV = BASE / "transgenic/revision"
OUT = REV / "figures"

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333", "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.dpi": 300,
})

SPECIES = ["A_thaliana", "B_distachyon", "B_rapa", "G_max", "L_sativa",
           "O_sativa", "P_patens", "P_trichocarpa", "S_bicolor", "S_italica",
           "S_lycopersicum", "V_vinifera", "Z_mays"]
SP_LABEL = {s: s.replace("_", ". ") for s in SPECIES}


def figure_s2():
    rows = {}
    with open(REV / "results" / "tss_tes_summary.csv") as fh:
        for r in csv.DictReader(fh):
            rows[r["species_variant"]] = r
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    series = [("transgenic400M", "de novo 400M", "#D55E00"),
              ("transgenic400Mprompt", "prompted 400M", "#009E73")]
    cats = [("TSS_exact_pct", "TSS_within50_pct", "TSS_within100_pct"),
            ("TES_exact_pct", "TES_within50_pct", "TES_within100_pct")]
    titles = ["A  TSS accuracy", "B  TES accuracy"]
    x = np.arange(len(SPECIES))
    w = 0.38
    for ax, (ex, w50, w100), title in zip(axes, cats, titles):
        for i, (suffix, lab, col) in enumerate(series):
            exact = [float(rows[f"{sp}_{suffix}"][ex]) for sp in SPECIES]
            near = [float(rows[f"{sp}_{suffix}"][w50]) for sp in SPECIES]
            ax.bar(x + (i - 0.5) * w, near, w * 0.9, color=col, alpha=0.45,
                   edgecolor="#333333", linewidth=0.25,
                   label=f"{lab} (within ±50 nt)" if ax is axes[0] else None)
            ax.bar(x + (i - 0.5) * w, exact, w * 0.9, color=col,
                   edgecolor="#333333", linewidth=0.25,
                   label=f"{lab} (exact)" if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([SP_LABEL[s] for s in SPECIES], rotation=38,
                           ha="right", style="italic", fontsize=5.5)
        ax.set_ylim(0, 109)
        ax.set_ylabel("Transcripts (%)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8, pad=20)
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left",
                   bbox_to_anchor=(0, 1.005), borderaxespad=0, columnspacing=1.4,
                   handlelength=1.4)
    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figureS2_tss_tes.{ext}")
    plt.close(fig)
    print("figureS2_tss_tes saved")


def figure_s3():
    rows = {}
    with open(REV / "results" / "feature_stats_summary.csv") as fh:
        for r in csv.DictReader(fh):
            rows[r["species"]] = r
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)
    panels = [("cds_max", "cds_p99", 150, "A  CDS segments"),
              ("utr5_max", "utr5_p99", 50, "B  5′-UTR segments"),
              ("utr3_max", "utr3_p99", 50, "C  3′-UTR segments")]
    x = np.arange(len(SPECIES))
    w = 0.38
    for ax, (mx, p99, limit, title) in zip(axes, panels):
        sel = [sp for sp in SPECIES if sp in rows]
        xx = np.arange(len(sel))
        vmax = [float(rows[sp][mx]) for sp in sel]
        v99 = [float(rows[sp][p99]) for sp in sel]
        ax.bar(xx - w / 2, vmax, w * 0.9, color="#0072B2",
               edgecolor="#333333", linewidth=0.25, label="Max")
        ax.bar(xx + w / 2, v99, w * 0.9, color="#56B4E9",
               edgecolor="#333333", linewidth=0.25, label="99th percentile")
        ax.axhline(limit, color="#D55E00", linewidth=1.0, linestyle="--",
                   label=f"GSF limit ({limit})")
        ax.set_xticks(xx)
        ax.set_xticklabels([SP_LABEL[s] for s in sel], rotation=38,
                           ha="right", style="italic", fontsize=5.5)
        ax.set_ylabel("Segments")
        ax.set_yscale("log")
        ax.set_ylim(1, 500)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8, pad=20)
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left",
                   bbox_to_anchor=(0, 1.005), borderaxespad=0, columnspacing=1.4,
                   handlelength=1.4)
    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figureS3_vocabulary_coverage.{ext}")
    plt.close(fig)
    print("figureS3_vocabulary_coverage saved")


if __name__ == "__main__":
    figure_s2()
    figure_s3()
