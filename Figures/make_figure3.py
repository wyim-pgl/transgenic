#!/usr/bin/env python3
"""Figure 3 regeneration. Performance analysis (regenerated test-set evaluation).

  A: F1 at base/exon/intron/intron-chain/transcript/locus level per species
     (de novo 400M, held-out test split + Z. mays).
  B: base-level recall/precision/F1 of de novo (dashed edge) vs prompted
     predictions per species.
  C: gene length vs base-level F1 (per-gene).
  D: number of reference CDS features vs base-level F1 (per-gene).

Inputs:
  fig3A_metrics.csv          (from fig3_evaluate.sh, de novo)
  fig3A_metrics_prompted.csv (same, prompted; optional until 3B run completes)
  fig3_per_gene_metrics.csv  (from fig3_per_gene_metrics.py)

Outputs: transgenic/Figures/figure3_performance.{pdf,png} (300 dpi)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import figstyle as _figstyle
_figstyle.apply(7)
import numpy as np

BASE = Path("/data/gpfs/assoc/pgl/data/Transgenic")
RES = BASE / "transgenic/revision/results/fig3_regen"
OUT = BASE / "transgenic/Figures"

plt.rcParams.update({
    "font.size": 7, "font.family": "sans-serif", "figure.dpi": 300,
    "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
})

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#E69F00"
SKY, VERM, PURP, GREY = "#56B4E9", "#D55E00", "#CC79A7", "#666666"

SPECIES = ["A_thaliana", "B_distachyon", "G_max", "O_sativa", "P_patens",
           "P_trichocarpa", "S_bicolor", "S_italica", "V_vinifera", "Z_mays"]
LEVELS = ["base", "exon", "intron", "ichain", "tx", "locus"]
LEVEL_NAMES = ["Base", "Exon", "Intron", "Intron chain", "Transcript", "Locus"]
LCOL = [BLUE, GREEN, ORANGE, SKY, VERM, PURP]

def f1(sn, pr):
    sn, pr = sn / 100, pr / 100
    return 2 * sn * pr / (sn + pr) if sn + pr else 0.0

def load_metrics(path):
    d = {}
    if not path.exists():
        return d
    with open(path) as fh:
        for row in csv.DictReader(fh):
            sp = row["species"]
            vals = [float(row[f"{lvl}_{m}"]) for lvl in ["base", "exon", "intron", "ichain", "tx", "locus"] for m in ("sn", "pr")]
            d[sp] = {lvl: f1(vals[i * 2], vals[i * 2 + 1]) * 100 for i, lvl in enumerate(LEVELS)}
            d[sp]["base_sn"] = float(row["base_sn"])
            d[sp]["base_pr"] = float(row["base_pr"])
    return d

denovo = load_metrics(RES / "fig3A_metrics.csv")
prompted = load_metrics(RES / "fig3A_metrics_prompted.csv")

fig = plt.figure(figsize=(7.2, 8.8))

# ---- Panel A: per-species per-level F1 (de novo) ---------------------------
axA = fig.add_axes([0.07, 0.56, 0.90, 0.30])
axA.text(-0.10, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
n, k = len(SPECIES), len(LEVELS)
w = 0.8 / k
x = np.arange(n)
for i, lvl in enumerate(LEVELS):
    vals = [denovo.get(sp, {}).get(lvl, 0.0) for sp in SPECIES]
    axA.bar(x + (i - k / 2 + 0.5) * w, vals, w * 0.9, color=LCOL[i],
            label=LEVEL_NAMES[i], edgecolor="none")
axA.set_xticks(x)
axA.set_xticklabels([s.replace("_", ". ") for s in SPECIES], rotation=45, ha="right")
axA.set_ylabel("F1 (%)")
axA.set_ylim(0, 100)
axA.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
axA.set_title("de novo (400M), held-out test set", fontsize=8)

# ---- Panel B: de novo vs prompted base metrics ------------------------------
axB = fig.add_axes([0.07, 0.30, 0.42, 0.17])
axB.text(-0.16, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
metrics = [("base_sn", "Recall"), ("base_pr", "Precision")]
x = np.arange(n)
w2 = 0.35
have_prompt = bool(prompted)
for j, (key, label) in enumerate(metrics):
    off = (j - 0.5) * w2 * 1.1
    dv = [denovo.get(sp, {}).get(key, 0.0) for sp in SPECIES]
    axB.bar(x + off - w2 / 2, dv, w2 / 2 * 0.9, color=LCOL[j], edgecolor="#333333",
            linewidth=0.5, linestyle="--", label=f"de novo {label}" if j == 0 else f"de novo {label}")
    if have_prompt:
        pv = [prompted.get(sp, {}).get(key, 0.0) for sp in SPECIES]
        axB.bar(x + off + w2 / 2, pv, w2 / 2 * 0.9, color=LCOL[j], alpha=0.45,
                edgecolor="none", label=f"prompted {label}")
axB.set_xticks(x)
axB.set_xticklabels([s.replace("_", ". ") for s in SPECIES], rotation=45, ha="right")
axB.set_ylabel("Base level (%)")
axB.set_ylim(0, 105)
axB.legend(ncol=2, frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 1.28))
axB.set_title("de novo (dashed) vs prompted", fontsize=8)

# ---- Panels C/D: per-gene scatter -------------------------------------------
pg = {}
pg_file = RES / "fig3_per_gene_metrics.csv"
if pg_file.exists():
    with open(pg_file) as fh:
        for row in csv.DictReader(fh):
            pg.setdefault(row["species"], []).append(
                (int(row["gene_len"]), int(row["n_cds"]), float(row["f1"]) * 100))

def scatter(ax, key_idx, xlabel, title, log=True):
    xs, ys = [], []
    for sp, rows in pg.items():
        xs += [r[key_idx] for r in rows]
        ys += [r[2] for r in rows]
    xs, ys = np.array(xs), np.array(ys)
    ax.scatter(xs, ys, s=1, alpha=0.15, color=BLUE, edgecolor="none", rasterized=True)
    if len(xs):
        q = np.quantile(xs, np.linspace(0, 1, 30))
        med = [np.median(ys[(xs >= q[i]) & (xs < q[i + 1])]) for i in range(len(q) - 1)]
        xc = [(q[i] + q[i + 1]) / 2 for i in range(len(q) - 1)]
        ax.plot(xc, med, color=VERM, lw=1.2)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Base F1 (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(title, fontsize=8)

axC = fig.add_axes([0.55, 0.30, 0.42, 0.17])
axC.text(-0.14, 1.04, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")
if pg:
    scatter(axC, 0, "Gene length (bp)", "gene length vs base F1")
else:
    axC.text(0.5, 0.5, "pending", transform=axC.transAxes, ha="center", color=GREY)

axD = fig.add_axes([0.07, 0.05, 0.90, 0.17])
axD.text(-0.07, 1.04, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")
if pg:
    xs, ys = [], []
    for sp, rows in pg.items():
        xs += [r[1] for r in rows]
        ys += [r[2] for r in rows]
    xs, ys = np.array(xs), np.array(ys)
    axD.scatter(xs, ys, s=1, alpha=0.15, color=GREEN, edgecolor="none", rasterized=True)
    if len(xs):
        bins = np.arange(0, min(xs.max(), 60) + 2)
        med = [np.median(ys[(xs >= b) & (xs < b + 1)]) for b in bins[:-1]]
        axD.plot(bins[:-1] + 0.5, med, color=VERM, lw=1.2)
    axD.set_xlabel("Number of reference CDS features")
    axD.set_ylabel("Base F1 (%)")
    axD.set_ylim(-3, 103)
    axD.set_xlim(0, min(xs.max() if len(xs) else 60, 60))
    axD.set_title("CDS feature count vs base F1", fontsize=8)
else:
    axD.text(0.5, 0.5, "pending", transform=axD.transAxes, ha="center", color=GREY)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure3_performance.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("saved figure3_performance")
