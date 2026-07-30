#!/usr/bin/env python3
"""
Revision figures for the TransGenic manuscript (Plant Communications).

Outputs (PDF + PNG, 300 dpi):
  figure5_gffcompare_f1   - 13 species x 7 tools, base- & transcript-level F1
  figureS_busco           - BUSCO completeness, 13 species
  figure6_as_evaluation   - 4-panel alternative splicing evaluation

Palette: Okabe-Ito (colorblind-safe). TransGenic prompt variants use a
lighter tint of the base model hue with a black edge for print legibility.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/data/gpfs/assoc/pgl/data/Transgenic")
CMP = BASE / "transgenic_comparison"
REV = BASE / "transgenic/revision"
OUT = REV / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
})

SPECIES = ["A_thaliana", "B_distachyon", "B_rapa", "G_max", "L_sativa",
           "O_sativa", "P_patens", "P_trichocarpa", "S_bicolor", "S_italica",
           "S_lycopersicum", "V_vinifera", "Z_mays"]
SP_LABEL = {s: s.replace("_", ". ") for s in SPECIES}
SP_LABEL = {s: s.replace("_", ". ") for s in SPECIES}  # italic thin space

TOOLS = [
    ("annevo", "ANNEVO", "#0072B2"),
    ("helixer", "Helixer", "#E69F00"),
    ("tiberius", "Tiberius", "#009E73"),
    ("transgenic160M", "TransGenic 160M", "#CC79A7"),
    ("transgenic160Mprompt", "TransGenic 160M + prompt", "#EAD0E4"),
    ("transgenic400M", "TransGenic 400M", "#D55E00"),
    ("transgenic400Mprompt", "TransGenic 400M + prompt", "#F3BE93"),
]


def f1(sn, pr):
    return 2 * sn * pr / (sn + pr) if (sn + pr) else 0.0


# ---------------------------------------------------------------- Figure 5
def figure5():
    rows = {}
    with open(CMP / "gffcompare_summary.csv") as fh:
        for r in csv.DictReader(fh):
            rows[(r["Species"], r["Tool"])] = r

    levels = [("Base", "Base_Sensitivity", "Base_Precision"),
              ("Transcript", "Transcript_Sensitivity", "Transcript_Precision")]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    width = 0.115
    x = np.arange(len(SPECIES))
    for ax, (lvl, snk, prk) in zip(axes, levels):
        for i, (tool, label, color) in enumerate(TOOLS):
            vals = []
            for sp in SPECIES:
                r = rows.get((sp, tool))
                vals.append(f1(float(r[snk]), float(r[prk])) if r else np.nan)
            off = (i - (len(TOOLS) - 1) / 2) * width
            ax.bar(x + off, vals, width * 0.92, label=label, color=color,
                   edgecolor="#333333", linewidth=0.25)
        ax.set_ylabel(f"{lvl}-level F1 (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SP_LABEL[s] for s in SPECIES], rotation=38,
                            ha="right", style="italic")
    axes[0].legend(ncol=4, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 1.32))
    axes[0].set_title("A", loc="left", fontweight="bold", fontsize=9)
    axes[1].set_title("B", loc="left", fontweight="bold", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure5_gffcompare_f1.{ext}")
    plt.close(fig)
    print("figure5_gffcompare_f1 saved")


# ---------------------------------------------------------------- BUSCO
def figure_busco():
    rows = {}
    # The raw summary splits each run directory on its last underscore, which leaks the
    # tool prefix into the Species column for tools whose names contain an underscore
    # (tiberius_softmasked, transgenic*_prompt_denovo). None of those appear in TOOLS, so
    # the figure was never affected, but read the normalized file when it is available so
    # that a future TOOLS addition cannot silently drop rows.
    # Regenerate it with: python 18_normalize_busco_summary.py
    busco_csv = CMP / "busco_summary_final.normalized.csv"
    if not busco_csv.exists():
        busco_csv = CMP / "busco_summary_final.csv"
    with open(busco_csv) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["Species"], {})[r["Tool"]] = r

    tools = [t for t in TOOLS]
    fig, axes = plt.subplots(4, 4, figsize=(7.0, 6.4), sharey=True)
    axes = axes.ravel()
    for ai, sp in enumerate(SPECIES):
        ax = axes[ai]
        data = rows.get(sp, {})
        for i, (tool, label, color) in enumerate(tools):
            r = data.get(tool)
            if not r:
                continue
            comp = float(r["Complete (%)"])
            total = int(r["Total BUSCOs"])
            frag = 100.0 * int(r["Fragmented (F)"]) / total
            ax.bar(i, comp, 0.82, color=color, edgecolor="#333333",
                   linewidth=0.25)
            ax.bar(i, frag, 0.82, bottom=comp, color="none",
                   edgecolor="#333333", linewidth=0.25, hatch="////")
        ax.set_title(SP_LABEL[sp], fontsize=7, style="italic", pad=2)
        ax.set_ylim(0, 109)
        ax.set_xticks([])
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if ai % 4 == 0:
            ax.set_ylabel("BUSCO (%)")
    for k in range(len(SPECIES), len(axes)):
        axes[k].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#333333",
                             linewidth=0.3, label=l) for _, l, c in tools]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none",
                                 edgecolor="#333333", hatch="////",
                                 linewidth=0.3, label="Fragmented"))
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figureS_busco.{ext}")
    plt.close(fig)
    print("figureS_busco saved")


# ---------------------------------------------------------------- Figure 6
def load_summary(d):
    return json.load(open(REV / "results" / d / "summary_report.json"))


def figure6():
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.30)

    # --- A: transcript-level P/R/F1 -------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    combos = [("de novo\n(vs TAIR10)", "denovo400M_vs_TAIR10"),
              ("prompted\n(vs TAIR10)", "prompted400Mbeam1_vs_TAIR10"),
              ("de novo\n(vs AtRTD3)", "denovo400M_vs_AtRTD3"),
              ("prompted\n(vs AtRTD3)", "prompted400Mbeam1_vs_AtRTD3")]
    metrics = [("isoform_recall", "Recall", "#0072B2"),
               ("isoform_precision", "Precision", "#E69F00"),
               ("isoform_f1", "F1", "#009E73")]
    w = 0.26
    x = np.arange(len(combos))
    for i, (key, lab, col) in enumerate(metrics):
        vals = [100 * load_summary(d)["transcript_level_metrics"][key]
                for _, d in combos]
        ax.bar(x + (i - 1) * w, vals, w * 0.9, label=lab, color=col,
               edgecolor="#333333", linewidth=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in combos], fontsize=6.2)
    ax.set_ylabel("Transcript level (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A  Isoform-level accuracy", loc="left",
                 fontweight="bold", fontsize=8)

    # --- B: isoform count outcome per gene (prompted, vs TAIR10) ---------
    ax = fig.add_subplot(gs[0, 1])
    cats = [("prompted400Mbeam1_vs_TAIR10", "vs TAIR10"),
            ("prompted400Mbeam1_vs_AtRTD3", "vs AtRTD3")]
    series = [("exact_count_matches", "Exact count", "#009E73"),
              ("underpredictions", "Under-predicted", "#E69F00"),
              ("overpredictions", "Over-predicted", "#D55E00"),
              ("missed_genes", "No prediction", "#999999")]
    w = 0.38
    x = np.arange(len(cats))
    bottoms = np.zeros(len(cats))
    totals = []
    for d, _ in cats:
        a = load_summary(d)["isoform_count_analysis"]
        totals.append(sum(a[k] for k, _, _ in series))
    for k, lab, col in series:
        vals = []
        for (d, _), tot in zip(cats, totals):
            a = load_summary(d)["isoform_count_analysis"]
            vals.append(100.0 * a[k] / tot)
        ax.bar(x, vals, w * 0.9, bottom=bottoms, label=lab, color=col,
               edgecolor="#333333", linewidth=0.25)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cats], fontsize=7)
    ax.set_ylabel("Genes (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=6, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("B  Isoform-count outcome (prompted)",
                 loc="left", fontweight="bold", fontsize=8)

    # --- C: splice event recovery ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    events = ["SE", "A5SS", "A3SS", "IR"]
    refs = [("prompted400Mbeam1_vs_TAIR10", "vs TAIR10", "#0072B2"),
            ("prompted400Mbeam1_vs_AtRTD3", "vs AtRTD3", "#56B4E9")]
    w = 0.34
    x = np.arange(len(events))
    for i, (d, lab, col) in enumerate(refs):
        rep = json.load(open(REV / "results" / d / "splice_events_report.json"))
        rec = [100 * rep["per_event_type"][e]["recall"] for e in events]
        prec = [100 * rep["per_event_type"][e]["precision"] for e in events]
        ax.bar(x + (i - 0.5) * w, rec, w * 0.9, label=f"Recall ({lab})",
               color=col, edgecolor="#333333", linewidth=0.25)
        ax.bar(x + (i - 0.5) * w, prec, w * 0.9, fill=False,
               edgecolor=col, linewidth=0.9, linestyle="--",
               label=f"Precision ({lab})")
    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylabel("Event rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=5.8, ncol=2)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("C  AS event recovery (prompted)",
                 loc="left", fontweight="bold", fontsize=8)

    # --- D: structural self-consistency -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    sc = {}
    with open(REV / "results" / "selfconsistency_summary.csv") as fh:
        for r in csv.DictReader(fh):
            sc[r["species_variant"]] = float(r["pct_fully_consistent"])
    try:
        beam1 = json.load(open(REV / "results" / "selfconsistency" /
                               "A_thaliana_transgenic400Mprompt_beam1.json"))
        sc["A_thaliana_transgenic400Mprompt"] = beam1["pct_fully_consistent"]
    except FileNotFoundError:
        pass
    series = [("transgenic400M", "de novo 400M", "#D55E00"),
              ("transgenic400Mprompt", "prompted 400M", "#009E73"),
              ("REF", "reference", "#999999")]
    x = np.arange(len(SPECIES))
    w = 0.26
    for i, (suffix, lab, col) in enumerate(series):
        vals = [sc.get(f"{sp}_{suffix}", np.nan) for sp in SPECIES]
        ax.bar(x + (i - 1) * w, vals, w * 0.9, label=lab, color=col,
               edgecolor="#333333", linewidth=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([SP_LABEL[s] for s in SPECIES], rotation=38,
                       ha="right", style="italic", fontsize=5.8)
    ax.set_ylabel("Fully consistent\ntranscripts (%)")
    ax.set_ylim(0, 109)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("D  ORF self-consistency",
                 loc="left", fontweight="bold", fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure6_as_evaluation.{ext}")
    plt.close(fig)
    print("figure6_as_evaluation saved")


if __name__ == "__main__":
    figure5()
    figure_busco()
    figure6()
