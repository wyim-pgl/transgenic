#!/usr/bin/env python3
"""Figure 4. Prompted alternative transcript predictions at three example loci.

Tracks per locus: TAIR10 reference / TransGenic (prompt completion, top beam) /
AtRTD3 IsoSeq transcriptome. Exons are boxes (CDS thick, UTR thin), introns are
connecting lines. The prompt transcript is marked (*); predictions matching an
AtRTD3 model are marked with a green triangle.

Loci (re-selected from parsed_tmap.csv per the legend descriptions; the
original three loci are unknown — see issues/issue5):
  A: AT1G02630 — both TAIR10 isoforms (.1, .2) reproduced; shared with AtRTD3
  B: AT1G19350 — 5 TAIR10 isoforms; .1 and .4 recovered (non-primary, AtRTD3-supported)
  C: AT1G01080 — .1 reproduced; .10 predicted, supported by AtRTD3, absent from TAIR10

Outputs: transgenic/Figures/figure4_example_loci.{pdf,png} (300 dpi)
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REV = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision")
CMP = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic_comparison")
OUT = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/Figures")

LOCI = [
    ("A", "AT1G02630", "A_thaliana_g000347"),
    ("B", "AT1G19350", "A_thaliana_g003895"),
    ("C", "AT1G01080", "A_thaliana_g000017"),
]

# ---------------------------------------------------------------- parsing
def parse_gtf_tx(path, wanted_genes=None, gene_attr="curly"):
    """Return {gene_id: {tx_id: {'strand':s,'exons':[(s,e)],'cds':[(s,e)],'utr':[(s,e)]}}}"""
    genes = defaultdict(lambda: defaultdict(lambda: {"strand": ".", "exons": [], "cds": [], "utr": []}))
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        feat, s, e, strand = f[2], int(f[3]), int(f[4]), f[6]
        attrs = f[8]
        if gene_attr == "curly":
            mt = re.search(r'transcript_id "([^"]+)"', attrs)
            mg = re.search(r'gene_id "([^"]+)"', attrs)
        else:
            mt = re.search(r'ID=([^;]+)', attrs) if feat == "mRNA" else None
            mg = re.search(r'Parent=([^;]+)', attrs)
        if feat == "mRNA" and gene_attr != "curly":
            continue
        if not mt or not mg:
            continue
        tx, g = mt.group(1), mg.group(1)
        g_norm = g.replace(".TAIR10", "")
        if wanted_genes and g_norm not in wanted_genes and tx.replace(".TAIR10", "") not in wanted_genes:
            continue
        t = genes[g_norm][tx]
        t["strand"] = strand if strand in "+-" else t["strand"]
        if feat == "exon":
            t["exons"].append((s, e))
        elif feat == "CDS":
            t["cds"].append((s, e))
            t["exons"].append((s, e))
        elif feat in ("five_prime_utr", "three_prime_utr", "five_prime_UTR", "three_prime_UTR"):
            t["utr"].append((s, e))
            t["exons"].append((s, e))
    return genes

def parse_transgenic(path, wanted_gm):
    """TransGenic standardized GFF3: gene/mRNA/exon/CDS with ID/Parent/GM."""
    genes = defaultdict(lambda: defaultdict(lambda: {"strand": ".", "exons": [], "cds": [], "utr": []}))
    gid2gm = {}
    mrna2gene = {}
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        rows.append((f[2], int(f[3]), int(f[4]), f[6], f[8]))
    for feat, s, e, strand, attrs in rows:
        if feat == "gene":
            mid = re.search(r'ID=([^;]+)', attrs)
            mgm = re.search(r'GM=([^;]+)', attrs)
            if mid and mgm:
                gid2gm[mid.group(1)] = mgm.group(1)
        elif feat == "mRNA":
            mid = re.search(r'ID=([^;]+)', attrs)
            mp = re.search(r'Parent=([^;]+)', attrs)
            if mid and mp:
                mrna2gene[mid.group(1)] = mp.group(1)
    for feat, s, e, strand, attrs in rows:
        if feat in ("gene", "mRNA"):
            continue
        mp = re.search(r'Parent=([^;]+)', attrs)
        if not mp:
            continue
        mrna = mp.group(1)
        gm = gid2gm.get(mrna2gene.get(mrna, ""), None)
        gm_norm = gm.replace(".TAIR10", "") if gm else None
        if gm_norm not in wanted_gm:
            continue
        t = genes[gm_norm][mrna]
        t["strand"] = strand if strand in "+-" else t["strand"]
        if feat == "exon":
            t["exons"].append((s, e))
        elif feat == "CDS":
            t["cds"].append((s, e))
        elif feat in ("five_prime_UTR", "three_prime_UTR"):
            t["utr"].append((s, e))
            t["exons"].append((s, e))
    return genes

tair = parse_gtf_tx(REV / "data/TAIR10/TAIR10.gtf", {g for _, g, _ in LOCI})
rtd = parse_gtf_tx(REV / "data/AtRTD3/AtRTD3.gtf", {g for _, g, _ in LOCI})
tg = parse_transgenic(CMP / "standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3",
                      {g for _, g, _ in LOCI})

def cds_type(t, s, e):
    if (s, e) in t["cds"]:
        return "cds"
    if (s, e) in t["utr"]:
        return "utr"
    return "exon"

def draw_track(ax, transcripts, y0, dy, color, label, mark=None, strand="+"):
    """Draw one track of gene models. mark: set of tx ids to flag with green triangle."""
    ax.text(-0.01, y0, label, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7, rotation=90)
    for i, (tx, t) in enumerate(sorted(transcripts.items())):
        y = y0 + i * dy
        exons = sorted(set(t["exons"]))
        if not exons:
            continue
        lo, hi = exons[0][0], exons[-1][1]
        ax.plot([lo, hi], [y, y], color=color, lw=0.7, zorder=1)
        for (s, e) in exons:
            kind = cds_type(t, s, e)
            h = 0.55 if kind == "cds" else 0.28
            ax.add_patch(Rectangle((s, y - h / 2), e - s, h, fc=color, ec="none", zorder=2))
        if mark and tx in mark:
            ax.plot([hi], [y + 0.55], marker="v", color="#009E73", markersize=5, zorder=3)
        ax.text(hi, y + 0.75, tx.split(".TAIR10")[0], fontsize=5.2, ha="right", color="#333333")
    return y0 + len(transcripts) * dy

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "figure.dpi": 300})
fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2))

COL = {"tair": "#0072B2", "tg": "#D55E00", "rtd": "#666666"}

for ax, (panel, tair_gene, qry_gene) in zip(axes, LOCI):
    t_tair = tair.get(tair_gene, {})
    t_rtd_all = rtd.get(tair_gene, {})
    t_tg = tg.get(tair_gene, {})

    # which TransGenic transcripts match an AtRTD3 model: compare intron chains
    def merged(t):
        ex = sorted(set(t["exons"]))
        if not ex:
            return []
        out = [list(ex[0])]
        for s, e in ex[1:]:
            if s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]

    def ichain(t):
        m = merged(t)
        return tuple((a[1], b[0]) for a, b in zip(m, m[1:]))

    rtd_chains = {ichain(t) for t in t_rtd_all.values() if t["exons"]}
    tg_match = {tx for tx, t in t_tg.items() if ichain(t) in rtd_chains}

    y = 1.0
    ax.set_xlim(0, 1)
    # genomic limits
    all_ex = [e for t in list(t_tair.values()) + list(t_rtd_all.values()) + list(t_tg.values()) for e in t["exons"]]
    glo = min(s for s, _ in all_ex) - 200
    ghi = max(e for _, e in all_ex) + 200

    def track(ax, transcripts, label, color, y, mark=None):
        n = len(transcripts)
        yy = y
        for i, (tx, t) in enumerate(sorted(transcripts.items())):
            yy = y + i * 1.0
            exons = merged(t)
            if not exons:
                continue
            lo, hi = exons[0][0], exons[-1][1]
            ax.plot([lo, hi], [yy, yy], color=color, lw=0.7, zorder=1)
            for (s, e) in exons:
                kind = cds_type(t, s, e)
                h = 0.5 if kind == "cds" else 0.25
                ax.add_patch(Rectangle((s, yy - h / 2), e - s, h, fc=color, ec="none", zorder=2))
            tag = tx.replace(".TAIR10", "")
            if color == COL["tg"] and tx.endswith(".t1"):
                tag += " (*)"  # prompt transcript
            ax.text(hi + (ghi - glo) * 0.005, yy, tag, fontsize=5.4, va="center", color="#333333")
            if mark and tx in mark:
                ax.plot([lo - (ghi - glo) * 0.01], [yy], marker="v", color="#009E73",
                        markersize=5, zorder=3, linestyle="none")
        ax.text(glo, yy + 1.2, label, ha="left", va="bottom",
                fontsize=7.5, color=color, fontweight="bold")
        return yy + 3.2

    y = track(ax, t_rtd_all, "AtRTD3", COL["rtd"], y)
    y = track(ax, t_tg, "TransGenic (*)", COL["tg"], y, mark=tg_match)
    y = track(ax, t_tair, "TAIR10", COL["tair"], y)

    ax.set_ylim(-1, y)
    ax.set_xlim(glo, ghi + (ghi - glo) * 0.12)
    ax.set_yticks([])
    ax.set_title(f"{panel}. {tair_gene} locus", fontsize=9, loc="left", fontweight="bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel(f"Chr1 (bp)", fontsize=7)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure4_example_loci.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("saved")
