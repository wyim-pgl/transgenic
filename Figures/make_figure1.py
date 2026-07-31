#!/usr/bin/env python3
"""Figure 1. Gene sentence format tokenization and model architecture.

Recreated from the manuscript legend and the TransGenic source code:
  (A) GFF -> GSF conversion for a hypothetical two-transcript gene.
  (B) Encoder-decoder architecture: HyenaDNA encoder -> Conv1d downsampling
      block -> Longformer decoder, with optional GSF prompt.
  (C) Two-stage strided-convolution downsampling block (6x compression)
      with U-Net-style skip connection and relative positional biases
      (HyenaDownsampleWithRelPosBias in modeling_HyenaTransgenic.py).

Outputs: transgenic/Figures/figure1_architecture.{pdf,png} (300 dpi)
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
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "figure.dpi": 300})

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#E69F00"
SKY, VERM, PURP, GREY = "#56B4E9", "#D55E00", "#CC79A7", "#666666"

fig = plt.figure(figsize=(7.2, 8.8))

# ---------------------------------------------------------------- Panel A
axA = fig.add_axes([0.03, 0.705, 0.94, 0.265])
axA.axis("off")
axA.set_xlim(0, 100)
axA.set_ylim(0, 100)
axA.text(1, 98, "A", fontsize=13, fontweight="bold", va="top")

gff_rows = [
    ("Chr1", "gene",  "100", "400", "+"),
    ("Chr1", "mRNA",  "100", "400", "+"),
    ("Chr1", "CDS",   "100", "130", "0"),
    ("Chr1", "CDS",   "180", "220", "0"),
    ("Chr1", "CDS",   "280", "350", "0"),
    ("Chr1", "mRNA",  "180", "400", "+"),
    ("Chr1", "CDS",   "180", "220", "0"),
    ("Chr1", "CDS",   "280", "350", "0"),
]
axA.text(2, 89, "GFF annotation", fontsize=9, fontweight="bold")
for i, (c, f, s, e, st) in enumerate(gff_rows):
    axA.text(2, 81 - i * 8.6, f"{c}   {f:<5}  {s:>3} - {e:<3}  {st}",
             family="monospace", fontsize=7)

axA.annotate("", xy=(36, 52), xytext=(27, 52),
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREY))
axA.text(31.5, 57, "reduce", ha="center", fontsize=7, color=GREY)

axA.text(40, 89, "Gene Sentence Format (GSF)", fontsize=9, fontweight="bold")
box = FancyBboxPatch((40, 34), 58, 44, boxstyle="round,pad=1.4",
                     fc="#F7F7F7", ec=GREY, lw=0.8)
axA.add_patch(box)
axA.text(42, 71, "0|CDS1|30|+|A;80|CDS2|120|+|B;180|CDS3|250|+|A",
         family="monospace", fontsize=7.4, color=BLUE)
axA.text(42, 64.5, "feature list: start|feature|end|strand|phase, ';'-separated",
         fontsize=6.3, color=BLUE, style="italic")
axA.text(42, 55, ">", family="monospace", fontsize=9, color=VERM, fontweight="bold")
axA.text(42, 45, "CDS1|CDS2|CDS3;CDS2|CDS3",
         family="monospace", fontsize=7.4, color=GREEN)
axA.text(42, 38.5, "transcript list: isoforms separated by ';'",
         fontsize=6.3, color=GREEN, style="italic")

# ---------------------------------------------------------------- Panel B
axB = fig.add_axes([0.03, 0.40, 0.94, 0.26])
axB.axis("off")
axB.set_xlim(0, 100)
axB.set_ylim(0, 100)
axB.text(1, 98, "B", fontsize=13, fontweight="bold", va="top")

def module(ax, x, y, w, h, label, sub, fc):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0",
                       fc=fc, ec="#333333", lw=0.9)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h * 0.66, label, ha="center", fontsize=8.2,
            fontweight="bold", color="white")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", fontsize=6.0, color="white")

axB.text(2, 66, "DNA", fontsize=8.5, family="monospace")
axB.text(2, 58, "ATGCCGTA...", fontsize=6.5, family="monospace", color=GREY)
axB.annotate("", xy=(15, 57), xytext=(12, 57),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))

module(axB, 15, 44, 22, 28, "HyenaDNA encoder", "implicit long convolutions\n12 layers, 768 dim", BLUE)
axB.annotate("", xy=(40, 57), xytext=(37, 57),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))
axB.text(38.5, 62, "1 nt", ha="center", fontsize=6.5, color=GREY)

module(axB, 40, 44, 22, 28, "Downsampling block", "strided Conv1d ×2\n1 nt → 6-mer  (C)", GREEN)
axB.annotate("", xy=(65, 57), xytext=(62, 57),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))
axB.text(63.5, 62, "6-mer", ha="center", fontsize=6.5, color=GREY)

module(axB, 65, 44, 22, 28, "Longformer decoder", "12 layers, 6 heads\nwindow 1024", VERM)
axB.annotate("", xy=(90, 57), xytext=(87, 57),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))

axB.text(91, 66, "GSF annotation", fontsize=8, family="monospace")
axB.text(91, 58, "0|CDS1|30|+|A;...", fontsize=6.2, family="monospace", color=GREY)

prompt = FancyBboxPatch((48, 8), 34, 18, boxstyle="round,pad=1.0",
                        fc="#FFFFFF", ec=PURP, lw=1.2, linestyle="--")
axB.add_patch(prompt)
axB.text(65, 19.5, "optional GSF prompt", ha="center", fontsize=7.5, color=PURP)
axB.text(65, 13, "(primary transcript → isoform completion)", ha="center",
         fontsize=6.0, color=PURP)
axB.annotate("", xy=(74, 43), xytext=(70, 27),
             arrowprops=dict(arrowstyle="-|>", lw=1.2, color=PURP, linestyle="--"))

# ---------------------------------------------------------------- Panel C
axC = fig.add_axes([0.03, 0.03, 0.94, 0.335])
axC.axis("off")
axC.set_xlim(0, 100)
axC.set_ylim(0, 100)
axC.text(1, 98, "C", fontsize=13, fontweight="bold", va="top")

def tensor_bar(ax, x, y, w, h, label, fc, sub):
    r = Rectangle((x, y), w, h, fc=fc, ec="#333333", lw=0.8)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=6.6, color="white", rotation=90)
    ax.text(x + w / 2, y - 8, sub, ha="center", va="top", fontsize=6.2, color=GREY)

# flow: input -> stage1 -> stage2
tensor_bar(axC, 6, 45, 8, 26, "n×768×L", BLUE, "encoder output\n(single-nt)")
axC.annotate("", xy=(17.5, 58), xytext=(14.5, 58),
             arrowprops=dict(arrowstyle="-|>", lw=1.8, color=BLUE))
axC.text(16, 76, "Conv1d\nk=6, s=3", ha="center", va="bottom", fontsize=6, color=BLUE)

tensor_bar(axC, 18, 45, 8, 26, "n×1152×L/3", GREEN, "stage 1")
axC.annotate("", xy=(29.5, 58), xytext=(26.5, 58),
             arrowprops=dict(arrowstyle="-|>", lw=1.8, color=BLUE))
axC.text(28, 76, "Conv1d\nk=2, s=2", ha="center", va="bottom", fontsize=6, color=BLUE)

tensor_bar(axC, 30, 45, 8, 26, "n×1536×L/6", VERM, "decoder input\n(6-mer)")

# skip connection (stage1 -> avgpool -> add to stage2), drawn below the bars
axC.annotate("", xy=(38, 41), xytext=(22, 41),
             arrowprops=dict(arrowstyle="-|>", lw=1.2, color=ORANGE,
                             connectionstyle="arc3,rad=0.25"))
axC.text(30, 24, "AvgPool k=2, s=2  (skip connection)", ha="center",
         fontsize=6.2, color=ORANGE)
axC.text(41.5, 55, "+", fontsize=13, fontweight="bold", color="#333333")

# positional bias injections
for i, c in enumerate([SKY, GREEN, PURP]):
    r = Rectangle((50 + i * 5, 68 - i * 7), 3.4, 5, fc=c, ec="#333333", lw=0.7)
    axC.add_patch(r)
axC.text(64, 78, "relative positional biases", fontsize=6.4, color=GREY)
axC.text(64, 72, "(learnable, injected at each stage)", fontsize=6.0, color=GREY)
axC.annotate("", xy=(57.5, 62), xytext=(63, 71),
             arrowprops=dict(arrowstyle="->", lw=0.8, color=GREY))

axC.text(50, 45, "two-stage strided convolution:", fontsize=6.6,
         fontweight="bold", color="#333333", va="top")
axC.text(50, 39, "stage 1: 768 → 1152 channels, 3× compression\n"
                 "stage 2: 1152 → 1536 channels, 2× compression\n"
                 "total: 6× sequence-length reduction\n"
                 "(HyenaDownsampleWithRelPosBias)",
         fontsize=6.2, color="#333333", va="top")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure1_architecture.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")
print("saved")
