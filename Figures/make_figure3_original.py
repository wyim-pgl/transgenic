#!/usr/bin/env python3
"""Figure 3 rebuilt from the ORIGINAL preserved test-set artifacts.

These are the artefacts that produced the published Figure 3, recovered from
jlomas@pgl-gpu:.../TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_768L12_E22-* and
staged under transgenic/revision/results/fig3_original/. Using them reproduces the
published panels exactly, unlike the 2026-07-28 reconstruction, which used a
re-seeded split and a per-gene aggregation that did not match the paper.

Anchor check (published: A. thaliana 92.2, Z. mays 71.1 base F1):
  TAIR10 noPost base F1 = 92.2, Zm0 noPost base F1 = 71.2 — exact.

Panels:
  A  per-species base/exon/intron/intron-chain/transcript/locus F1 (de novo noPost)
  B  de novo (noPost) vs prompted (completion) base-level Sn/Pr per species
  C  TAIR10 base F1 vs gene-length bin
  D  TAIR10 base F1 vs CDS-count bin

Definition, now settled: base-level F1 = 2·Sn·Pr/(Sn+Pr) from GFFCompare on the
noPost prediction against the test-set labels, per organism (panel A) or per
gene-length / CDS-count bin (panels C/D). Bins come from binLengthGff.py /
binCDSGFF.py + gffcompare, exactly as in the original pipeline.

Output: transgenic/Figures/figure3_performance_original.{pdf,png} (300 dpi)
"""

from __future__ import annotations

import glob
import re
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
ORIG = BASE / "transgenic/revision/results/fig3_original"
OUT = BASE / "transgenic/Figures"

plt.rcParams.update({
    "font.size": 7, "font.family": "sans-serif", "figure.dpi": 300,
    "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "svg.fonttype": "none",
})

# Okabe-Ito
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
SKY, VERM, PURP, GREY = "#56B4E9", "#D55E00", "#CC79A7", "#666666"

# GFFCompare organism id -> display name (publication order: eudicot, monocot, moss, maize)
ORG = [
    ("TAIR10", "A. thaliana"), ("Glyma", "G. max"), ("Potri", "P. trichocarpa"),
    ("Vitvi", "V. vinifera"), ("Bradi", "B. distachyon"), ("MSUv7", "O. sativa"),
    ("Sobic", "S. bicolor"), ("Seita", "S. italica"), ("Pp3", "P. patens"),
    ("Zm0", "Z. mays"),
]
LEVELS = ["Base", "Exon", "Intron", "Intron chain", "Transcript", "Locus"]
LCOL = [BLUE, GREEN, ORANGE, SKY, VERM, PURP]


def parse_levels(path: Path) -> dict[str, float]:
    """Return {level: F1} from a GFFCompare .stats file."""
    txt = path.read_text()
    out = {}
    for lv in LEVELS:
        m = re.search(rf"{re.escape(lv)} level:\s+([\d.]+)\s+\|\s+([\d.]+)", txt)
        if m:
            sn, pr = float(m.group(1)), float(m.group(2))
            out[lv] = 0.0 if sn + pr == 0 else 2 * sn * pr / (sn + pr)
    return out


def parse_base(path: Path) -> tuple[float, float, int]:
    txt = path.read_text()
    m = re.search(r"Base level:\s+([\d.]+)\s+\|\s+([\d.]+)", txt)
    q = re.search(r"Query mRNAs :\s+(\d+)", txt)
    if not m:
        return (0.0, 0.0, 0)
    return float(m.group(1)), float(m.group(2)), int(q.group(1)) if q else 0


def f1(sn: float, pr: float) -> float:
    return 0.0 if sn + pr == 0 else 2 * sn * pr / (sn + pr)


# ------------------------------------------------------------------ figure ----
fig = plt.figure(figsize=(7.2, 8.6))

# ---- Panel A -----------------------------------------------------------------
axA = fig.add_axes([0.07, 0.71, 0.90, 0.22])
axA.text(-0.065, 1.06, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
names = [n for _, n in ORG]
x = np.arange(len(ORG))
k = len(LEVELS)
w = 0.8 / k
for i, lv in enumerate(LEVELS):
    vals = []
    for org, _ in ORG:
        p = ORIG / "denovo" / f"{org}_noPost.stats"
        vals.append(parse_levels(p).get(lv, 0.0) if p.exists() else 0.0)
    axA.bar(x + (i - k / 2 + 0.5) * w, vals, w * 0.9, color=LCOL[i], label=lv,
            edgecolor="none")
axA.set_xticks(x)
axA.set_xticklabels(names, rotation=40, ha="right", style="italic")
axA.set_ylabel("F1 (%)")
axA.set_ylim(0, 100)
axA.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
           handlelength=1.1, columnspacing=1.2)

# ---- Panel B: de novo vs prompted base Sn/Pr ---------------------------------
axB = fig.add_axes([0.07, 0.40, 0.90, 0.18])
axB.text(-0.065, 1.06, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
w2 = 0.2
for j, (label, col) in enumerate([("Recall", BLUE), ("Precision", ORANGE)]):
    dn, pr = [], []
    for org, _ in ORG:
        d = parse_base(ORIG / "denovo" / f"{org}_noPost.stats")
        p = parse_base(ORIG / "prompted_stats" / f"{org}_prompt.stats")
        dn.append(d[0] if label == "Recall" else d[1])
        pr.append(p[0] if label == "Recall" else p[1])
    off = (j - 0.5) * (2 * w2 + 0.04)
    axB.bar(x + off, dn, w2, color=col, edgecolor="#333333", linewidth=0.4,
            linestyle="--", label=f"de novo {label}")
    axB.bar(x + off + w2, pr, w2, color=col, alpha=0.5, edgecolor="none",
            label=f"prompted {label}")
axB.set_xticks(x)
axB.set_xticklabels(names, rotation=40, ha="right", style="italic")
axB.set_ylabel("Base level (%)")
axB.set_ylim(0, 105)
axB.legend(ncol=4, frameon=False, fontsize=5.5, loc="lower center",
           bbox_to_anchor=(0.5, 1.01), handlelength=1.1, columnspacing=1.0)

# ---- Panels C/D: TAIR10 length and CDS bins ----------------------------------
def load_bins(folder: str) -> tuple[list[int], list[float], list[int]]:
    rows = []
    for f in glob.glob(str(ORIG / "denovo" / folder / "TAIR10-*.stats")):
        b = int(re.search(r"TAIR10-(\d+)\.stats", f).group(1))
        sn, pr, n = parse_base(Path(f))
        rows.append((b, f1(sn, pr), n))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def spearman(xv, yv) -> float:
    import math
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2 + 1
            for kk in range(i, j + 1):
                r[o[kk]] = a
            i = j + 1
        return r
    rx, ry = rank(xv), rank(yv); mx = sum(rx) / len(rx); my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


axC = fig.add_axes([0.07, 0.07, 0.40, 0.22])
axC.text(-0.16, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")
lx, ly, ln = load_bins("binByLength")
# size marker by n so the noisy small-n tail reads as noisy
sizes = [max(6, min(90, n / 6)) for n in ln]
axC.scatter(lx, ly, s=sizes, color=BLUE, alpha=0.7, edgecolor="#333333", linewidth=0.3)
# n-weighted trend over the well-populated range only
big = [(b, y) for b, y, n in zip(lx, ly, ln) if n >= 30]
if len(big) > 2:
    bx, by = zip(*big)
    z = np.polyfit(bx, by, 1)
    xs = np.array([min(bx), max(bx)])
    axC.plot(xs, np.polyval(z, xs), color=VERM, lw=1.3)
rho_all = spearman(lx, ly)
rho_big = spearman([b for b, _, n in zip(lx, ly, ln) if n >= 30],
                   [y for _, y, n in zip(lx, ly, ln) if n >= 30])
axC.set_xlabel("Gene length bin (bp)")
axC.set_ylabel("Base F1 (%)")
axC.set_ylim(-3, 103)
axC.set_title(f"A. thaliana  (ρ={rho_all:+.2f}, n≥30 ρ={rho_big:+.2f})", fontsize=7)

axD = fig.add_axes([0.57, 0.07, 0.40, 0.22])
axD.text(-0.16, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")
cx, cy, cn = load_bins("binByCDS")
sizes = [max(6, min(90, n / 6)) for n in cn]
axD.scatter(cx, cy, s=sizes, color=GREEN, alpha=0.7, edgecolor="#333333", linewidth=0.3)
big = [(b, y) for b, y, n in zip(cx, cy, cn) if n >= 30]
if len(big) > 2:
    bx, by = zip(*big)
    z = np.polyfit(bx, by, 1)
    xs = np.array([min(bx), max(bx)])
    axD.plot(xs, np.polyval(z, xs), color=VERM, lw=1.3)
rho_all = spearman(cx, cy)
rho_big = spearman([b for b, _, n in zip(cx, cy, cn) if n >= 30],
                   [y for _, y, n in zip(cx, cy, cn) if n >= 30])
axD.set_xlabel("Number of reference CDS features")
axD.set_ylabel("Base F1 (%)")
axD.set_ylim(-3, 103)
axD.set_xlim(0, 60)
axD.set_title(f"A. thaliana  (ρ={rho_all:+.2f}, n≥30 ρ={rho_big:+.2f})", fontsize=7)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure3_performance_original.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")

# ------------------------------------------------------------- console report -
print("Figure 3 rebuilt from original artifacts -> figure3_performance_original.{pdf,png}")
print("\nAnchor check (published A. thaliana 92.2, Z. mays 71.1 base F1):")
for org, name in [("TAIR10", "A. thaliana"), ("Zm0", "Z. mays")]:
    sn, pr, _ = parse_base(ORIG / "denovo" / f"{org}_noPost.stats")
    print(f"  {name:14s} noPost base F1 = {f1(sn, pr):.1f}")
print(f"\nPanel C (length):  all-bin ρ = {spearman(lx, ly):+.3f}")
print(f"Panel D (CDS):     all-bin ρ = {spearman(cx, cy):+.3f}")
