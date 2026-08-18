#!/usr/bin/env python3
"""Figure 3: panels C/D from the ORIGINAL preserved artifacts, panel A from the
audited re-inference, panel B from the alternative-only AtRTD3 evaluation.

Panel A (revised under roadmap R11): the original per-species GFFCompare stats
are contaminated for some species by "runaway" predicted transcripts — a single
blown-out UTR/CDS becomes one multi-Mb pseudo-exon that moves base-mass-weighted
base-level precision 13-24 points while count-weighted levels move <1 point
(original G. max Pr 66.0 and P. trichocarpa 72.4 are such artifacts). The original
de novo prediction GFF3s were not archived, so the filter cannot be applied to
them; panel A therefore reports the repaired-RC re-inference with runaway
transcripts excluded (predicted mRNA span > same-species reference maximum):
transgenic/revision/results/fig3a_divergence/gffcompare_noRunaway/nr_*.stats.
Caveats carried as legend footnotes: V. vinifera base Sn is 4.4 points below the
original computation (unexplained); the Z. mays evaluation set differs from the
original (36,352 vs 7,458 reference loci), so its bars are not comparable with
the originally reported 71.1.

Panel B (revised, matches the manuscript legend): base-level recall/precision/F1
of de novo vs reference-prompted predictions scored in A. thaliana against the
alternative-transcript-only AtRTD3 reference (Table S4a):
transgenic/revision/results/altonly_fixed/*_vs_AtRTD3altfix.stats.

Anchor check: canonical panel A base F1 — A. thaliana 92.0, Z. mays 74.0
(original artifacts reproduce 92.2 / 71.2 and remain staged under
transgenic/revision/results/fig3_original/).

Panels:
  A  per-species base/exon/intron/intron-chain/transcript/locus F1
     (de novo, repaired re-inference, runaway-filtered)
  B  de novo vs prompted base-level Sn/Pr/F1, alt-only AtRTD3 (A. thaliana)
  C  TAIR10 base F1 vs gene-length bin (original artifacts)
  D  TAIR10 base F1 vs CDS-count bin (original artifacts)

Definition: base-level F1 = 2·Sn·Pr/(Sn+Pr) from GFFCompare, per organism
(panel A) or per gene-length / CDS-count bin (panels C/D). Bins come from
binLengthGff.py / binCDSGFF.py + gffcompare, exactly as in the original pipeline.

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
NR = BASE / "transgenic/revision/results/fig3a_divergence/gffcompare_noRunaway"
ALTONLY = BASE / "transgenic/revision/results/altonly_fixed"
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
# panel A canonical source: nr_<species>.stats keyed by species name, same order.
# Daggers mark the two legend footnotes (V. vinifera Sn divergence; Z. mays
# evaluation-set change) on the tick labels.
NR_SP = {
    "TAIR10": "A_thaliana", "Glyma": "G_max", "Potri": "P_trichocarpa",
    "Vitvi": "V_vinifera", "Bradi": "B_distachyon", "MSUv7": "O_sativa",
    "Sobic": "S_bicolor", "Seita": "S_italica", "Pp3": "P_patens",
    "Zm0": "Z_mays",
}
NR_MARK = {"Vitvi": "†", "Zm0": "‡"}  # † / ‡
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

# ---- Panel A: canonical re-inference, runaway-filtered (R11) -----------------
axA = fig.add_axes([0.07, 0.71, 0.90, 0.22])
axA.text(-0.065, 1.06, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
names = [n + NR_MARK.get(org, "") for org, n in ORG]
x = np.arange(len(ORG))
k = len(LEVELS)
w = 0.8 / k
for i, lv in enumerate(LEVELS):
    vals = []
    for org, _ in ORG:
        p = NR / f"nr_{NR_SP[org]}.stats"
        vals.append(parse_levels(p).get(lv, 0.0) if p.exists() else 0.0)
    axA.bar(x + (i - k / 2 + 0.5) * w, vals, w * 0.9, color=LCOL[i], label=lv,
            edgecolor="none")
axA.set_xticks(x)
axA.set_xticklabels(names, rotation=40, ha="right", style="italic")
axA.set_ylabel("F1 (%)")
axA.set_ylim(0, 100)
axA.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
           handlelength=1.1, columnspacing=1.2)

# ---- Panel B: de novo vs prompted, alt-only AtRTD3 (matches manuscript legend)
axB = fig.add_axes([0.30, 0.40, 0.44, 0.18])
axB.text(-0.14, 1.06, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
_dn = parse_base(ALTONLY / "A_thaliana_transgenic400M_vs_AtRTD3altfix.stats")
_pr = parse_base(ALTONLY / "A_thaliana_transgenic400Mprompt_beam1_vs_AtRTD3altfix.stats")
metricsB = ["Recall", "Precision", "F1"]
dn_vals = [_dn[0], _dn[1], f1(_dn[0], _dn[1])]
pr_vals = [_pr[0], _pr[1], f1(_pr[0], _pr[1])]
xb = np.arange(len(metricsB))
w2 = 0.32
axB.bar(xb - w2 / 2, dn_vals, w2 * 0.92, color="white", edgecolor=BLUE,
        linewidth=0.9, linestyle="--", label="de novo")
axB.bar(xb + w2 / 2, pr_vals, w2 * 0.92, color=BLUE, edgecolor="none",
        label="prompted")
for xi, v in zip(xb - w2 / 2, dn_vals):
    axB.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
for xi, v in zip(xb + w2 / 2, pr_vals):
    axB.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
axB.set_xticks(xb)
axB.set_xticklabels(metricsB)
axB.set_ylabel("Base level (%)")
axB.set_ylim(0, 95)
axB.set_title(r"$\it{A.\ thaliana}$ vs alternative-only AtRTD3 (Table S4a)",
              fontsize=6.5)
axB.legend(ncol=2, frameon=False, fontsize=6, loc="lower center",
           bbox_to_anchor=(0.5, 1.10), handlelength=1.3, columnspacing=1.4)

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
print("Figure 3 rebuilt -> figure3_performance_original.{pdf,png}")
print("\nPanel A canonical anchors (expected A. thaliana 92.0, Z. mays 74.0 base F1):")
for org, name in [("TAIR10", "A. thaliana"), ("Zm0", "Z. mays")]:
    sn, pr, _ = parse_base(NR / f"nr_{NR_SP[org]}.stats")
    print(f"  {name:14s} noRunaway base F1 = {f1(sn, pr):.1f}")
print("Original-artifact anchors (expected 92.2 / 71.2, staged, not plotted):")
for org, name in [("TAIR10", "A. thaliana"), ("Zm0", "Z. mays")]:
    sn, pr, _ = parse_base(ORIG / "denovo" / f"{org}_noPost.stats")
    print(f"  {name:14s} noPost base F1 = {f1(sn, pr):.1f}")
print("\nPanel B (alt-only AtRTD3): de novo "
      f"{dn_vals[0]:.1f}/{dn_vals[1]:.1f}/{dn_vals[2]:.1f}, "
      f"prompted {pr_vals[0]:.1f}/{pr_vals[1]:.1f}/{pr_vals[2]:.1f} (Sn/Pr/F1)")
print(f"\nPanel C (length):  all-bin ρ = {spearman(lx, ly):+.3f}")
print(f"Panel D (CDS):     all-bin ρ = {spearman(cx, cy):+.3f}")
