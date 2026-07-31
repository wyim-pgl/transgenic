#!/usr/bin/env python3
"""Figure 2. Training, evaluation, and testing datasets.

Recreated from the manuscript legend:
  - Nine plant species (4 eudicots, 4 monocots, 1 moss) used for
    training/evaluation/testing; maize fully withheld from training.
  - Per-gene sequences padded with adjacent genomic sequence to the next
    multiple of 6,144 nt, paired with GSF labels.
  - 75% / 10% / 15% train/eval/test split.

Outputs: transgenic/Figures/figure2_datasets.{pdf,png} (300 dpi)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import figstyle as _figstyle
_figstyle.apply(8)
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

OUT = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/Figures")
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "figure.dpi": 300})

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#E69F00"
SKY, VERM, PURP, GREY = "#56B4E9", "#D55E00", "#CC79A7", "#666666"

fig = plt.figure(figsize=(7.2, 5.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# species boxes (height proportional to species count, stacked with gaps)
groups = [
    ("Eudicots (4)", ["A. thaliana", "G. max", "P. trichocarpa", "V. vinifera"], BLUE),
    ("Monocots (4)", ["O. sativa", "S. bicolor", "B. distachyon", "S. italica"], GREEN),
    ("Moss (1)", ["P. patens"], PURP),
]
y_top = 96
for gname, species, color in groups:
    h = 8 + 4.2 * len(species)
    b = FancyBboxPatch((3, y_top - h), 26, h, boxstyle="round,pad=0.6",
                       fc="#FFFFFF", ec=color, lw=1.4)
    ax.add_patch(b)
    ax.text(16, y_top - 4, gname, ha="center", fontsize=8, fontweight="bold", color=color)
    for si, sp in enumerate(species):
        ax.text(16, y_top - 8.5 - si * 4.2, sp, ha="center", fontsize=6.6,
                style="italic", color="#333333")
    y_top -= h + 3

# withheld maize box (directly below the stacked groups)
h = 12
b = FancyBboxPatch((3, y_top - h), 26, h, boxstyle="round,pad=0.6",
                   fc="#F0F0F0", ec=GREY, lw=1.4, linestyle="--", hatch="///")
ax.add_patch(b)
ax.text(16, y_top - 5, "Z. mays", ha="center", fontsize=7.5, style="italic", color=GREY)
ax.text(16, y_top - 9, "withheld from training (test only)", ha="center", fontsize=5.6, color=GREY)
maize_right = (29, y_top - h / 2)

# arrow to padding step
ax.annotate("", xy=(36, 50), xytext=(30, 50),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333333"))

# per-gene extraction: DNA padded to 6144 multiple + GSF label
ax.text(49, 92, "per-gene (sequence, GSF label) pairs", fontsize=8.5,
        fontweight="bold", ha="center")
# gene bar with flanks
ax.add_patch(Rectangle((38, 72), 14, 6, fc=BLUE, ec="#333333", lw=0.8))
ax.add_patch(Rectangle((52, 72), 4, 6, fc=SKY, ec="#333333", lw=0.8))
ax.add_patch(Rectangle((56, 72), 4, 6, fc=SKY, ec="#333333", lw=0.8))
ax.text(45, 75, "gene", ha="center", va="center", fontsize=6.5, color="white")
ax.text(54, 75, "5′", ha="center", va="center", fontsize=6, color="white")
ax.text(58, 75, "3′", ha="center", va="center", fontsize=6, color="white")
ax.annotate("", xy=(60, 69), xytext=(38, 69),
            arrowprops=dict(arrowstyle="<->", lw=0.9, color=GREY))
ax.text(49, 64, "padded to next multiple of 6,144 nt (max 49,152 nt)",
        ha="center", fontsize=6.4, color=GREY)
ax.text(49, 55, "0|CDS1|30|+|A;80|CDS2|120|+|B>CDS1|CDS2",
        ha="center", family="monospace", fontsize=6.6, color=BLUE)
ax.text(49, 50, "GSF label (decoder target)", ha="center", fontsize=6.4, color=GREY)

# arrow to split
ax.annotate("", xy=(66, 50), xytext=(62, 50),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333333"))

# 75/10/15 split bar
ax.text(82, 92, "dataset split", fontsize=8.5, fontweight="bold", ha="center")
segs = [("training 75%", 75, BLUE), ("eval 10%", 10, GREEN), ("test 15%", 15, VERM)]
x0, total_w = 68, 28
y = 66
for name, frac, c in segs:
    w = total_w * frac / 100
    ax.add_patch(Rectangle((x0, y), w, 8, fc=c, ec="#333333", lw=0.6))
    if frac >= 15:
        ax.text(x0 + w / 2, y + 4, name, ha="center", va="center", fontsize=6.2,
                color="white")
    else:
        ax.text(x0 + w / 2, y + 11, name, ha="center", va="bottom", fontsize=6.0, color=c)
    x0 += w
ax.text(82, 56, "9 species: random 75/10/15 split of gene models",
        ha="center", fontsize=6.4, color=GREY)
ax.text(82, 50, "Z. mays: entire genome assigned to test set",
        ha="center", fontsize=6.4, color=GREY)
ax.annotate("", xy=(68, 58), xytext=maize_right,
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=GREY, linestyle="--"))

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure2_datasets.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("saved")
