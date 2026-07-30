#!/usr/bin/env python3
"""Draw Panel-C example loci under one common rule set.

A Panel-C locus is one where TransGenic, in prompt-completion mode, predicts an
isoform whose CDS intron-chain matches an AtRTD3 long-read transcript exactly but
matches no TAIR10 transcript. Loci were found by scanning the original prompted
test-set predictions (`fig4_forensics/raw_TAIR10_hyenaTest_prediction_noPost.gff3`)
with `find_panelC.py`; per-locus GFF3/GTF extracts live next to that script under
`fig4_forensics/panelC_examples/`.

Every figure in the set follows the same rules so the panels are comparable:

  Track order      TAIR10 (grey) -> TransGenic novel (orange) -> TransGenic
                   reproduced (blue) -> AtRTD3 exact chain match (dark green) ->
                   AtRTD3 feature support (green) -> AtRTD3 others (pale green)
  Track selection  TAIR10: one row per distinct CDS intron-chain.
                   TransGenic: every predicted transcript.
                   AtRTD3: the exact chain match, then up to MAX_SUPPORT transcripts
                   carrying the highlighted feature, then up to MAX_CONTEXT others,
                   each group sorted by numeric suffix (deterministic) and never
                   drawing the same transcript twice.
  Glyphs           CDS = tall box, UTR / non-coding exon = short box, intron = line.
  Highlight        the first category that applies:
                   1. exon          - an exon of the novel isoform overlaps no TAIR10 exon
                   2. junction      - the novel isoform uses a junction no TAIR10 isoform uses
                   3. retained      - the novel CDS spans a TAIR10 intron that no TAIR10
                                      isoform leaves unspliced
                   4. unspliced_utr - same, but the span falls in a predicted UTR, which
                                      usually just means splicing stopped past the stop codon
                   5. combination   - every junction already occurs in some TAIR10 isoform
                                      and only the chain is new; the full symmetric
                                      difference is shaded. Weakest visual evidence.
  Annotation       every panel states TAIR10 support (0 by construction) and AtRTD3
                   support split into "identical" / "splice-site-shared" /
                   "overlap-only" - never a bare overlap count, which conflates a
                   cassette exon with a longer exon that merely spans the interval.

`verify_panelC_figures.py` (next to the data) re-derives these claims from the
source files and fails loudly if a drawn track contradicts its colour.

Outputs `figure4_panelC_<locus>.{pdf,png}` per locus plus the combined
`figure4_panelC_all.{pdf,png}` sheet, all at 300 dpi.

Run: python3 make_figure4_panelC.py [locus ...]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Resolved from this file so the script runs from a fresh clone, not just from the
# path it was written in.
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "revision" / "results" / "fig4_forensics" / "panelC_examples"
OUT = REPO / "Figures"

# Ordered so the combined sheet reads strongest-evidence first.
LOCI = ["AT2G37450", "AT4G22540", "AT3G56730", "AT1G78940", "AT3G29185", "AT3G13740"]
# The three loci whose novel feature is directly visible (a cassette exon or a
# junction no TAIR10 isoform uses). The other three differ only in chain
# combination or in UTR splicing, so they go to the supplement, not the figure.
STRONG = ["AT2G37450", "AT4G22540", "AT3G56730"]
# The rebuilt Figure 4. (A) is the locus recovered forensically from the published
# model's plot-prep file; (B) is the only locus in the test set where every distinct
# TAIR10 chain came back; (C)-(E) are the Panel-C loci with a directly visible novel
# feature. See fig4_forensics/FIG4_LOCUS_FORENSICS.md and panelC_examples/README.md.
FIGURE4 = ["AT1G43770", "AT1G44575", "AT2G37450", "AT4G22540", "AT3G56730"]

# Okabe-Ito derived, colour-vision-safe.
GREY = "#8C8C8C"
BLUE = "#0072B2"
ORANGE = "#D55E00"
DGREEN = "#00664A"
GREEN = "#009E73"
PALE = "#9AD5C0"
SHADE = "#F0C000"

MAX_SUPPORT = 2      # AtRTD3 splice-site sharers drawn per panel
MAX_CONTEXT = 2      # AtRTD3 non-supporting isoforms drawn per panel
H_CDS, H_UTR = 0.34, 0.17
ROW_H = 0.30         # inches per transcript row - fixed, so all panels share a scale
FIG_W = 7.6

UTR_FEATS = {"five_prime_UTR", "three_prime_UTR", "five_prime_utr", "three_prime_utr"}


# --------------------------------------------------------------------------- parsing

def parse(path: Path, gtf: bool) -> dict:
    """Return {transcript: {'cds', 'utr', 'exon', 'strand', 'chr'}} for one locus."""
    tx: dict = defaultdict(lambda: {"cds": [], "utr": [], "exon": [], "strand": ".", "chr": ""})
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text().splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        # Only structural rows define a transcript. Skipping gene/mRNA rows matters:
        # an mRNA row carries Parent=<gene>, so keying on Parent alone would invent a
        # phantom empty transcript per gene and leave a blank track in the figure.
        if f[2] not in ({"CDS", "exon"} | UTR_FEATS):
            continue
        m = (re.search(r'transcript_id "([^"]+)"', f[8]) if gtf
             else re.search(r"Parent=([^;]+)", f[8]))
        if not m:
            continue
        rec = tx[m.group(1)]
        rec["strand"], rec["chr"] = f[6], f[0]
        seg = (int(f[3]), int(f[4]))
        if f[2] == "CDS":
            rec["cds"].append(seg)
        elif f[2] == "exon":
            rec["exon"].append(seg)
        else:
            rec["utr"].append(seg)
    return dict(tx)


def chain(segs: list) -> tuple:
    """CDS intron chain: the gaps between sorted exons."""
    e = sorted(segs)
    return tuple((e[i][1] + 1, e[i + 1][0] - 1) for i in range(len(e) - 1))


def overlaps(a: tuple, b: tuple) -> bool:
    return not (a[1] < b[0] or a[0] > b[1])


def thick_thin(rec: dict) -> tuple[list, list]:
    """Split a record into (CDS boxes, thin boxes). AtRTD3 GTF gives exon+CDS, the
    GFF3 tracks give CDS+UTR; both reduce to the same two-tier glyph."""
    cds = sorted(rec["cds"])
    thin = sorted(rec["utr"] or rec["exon"])
    return cds, thin


def span(rec: dict) -> tuple:
    xs = [x for seg in (rec["cds"] + rec["utr"] + rec["exon"]) for x in seg]
    return (min(xs), max(xs)) if xs else (0, 0)


def suffix(tx: str) -> int:
    m = re.search(r"\.(\d+)$", tx)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- analysis

def analyse(locus: str) -> dict:
    """Resolve the novel isoform, its AtRTD3 match, and what to highlight."""
    pred = parse(DATA / f"{locus}_pred.gff3", gtf=False)
    tair = parse(DATA / f"{locus}_tair.gff3", gtf=False)
    art = parse(DATA / f"{locus}_atrtd.gtf", gtf=True)

    tair_chains = {chain(r["cds"]) for r in tair.values()}
    tair_junctions = {j for c in tair_chains for j in c}
    tair_exons = [s for r in tair.values() for s in (r["cds"] + r["utr"])]
    art_chains = {t: chain(r["cds"]) for t, r in art.items() if r["cds"]}

    novel_tx = novel_match = None
    novel_chain: tuple = ()
    for pt in sorted(pred, key=suffix):
        pc = chain(pred[pt]["cds"])
        if len(pc) < 2 or pc in tair_chains:
            continue
        hits = sorted((t for t, c in art_chains.items() if c == pc), key=suffix)
        if hits:
            novel_tx, novel_match, novel_chain = pt, hits[0], pc
            break
    if novel_tx is None:
        # No AtRTD3-supported chain that TAIR10 lacks. If instead the model reproduced
        # two or more distinct TAIR10 chains and AtRTD3 documents them too, this is the
        # "reproduced shared alternative transcripts" panel (Fig. 4A), not a failure.
        art_chain_set = set(art_chains.values())
        pred_chains = {chain(r["cds"]) for r in pred.values() if len(chain(r["cds"])) >= 1}
        reproduced = [c for c in pred_chains if c in tair_chains]
        shared = [c for c in reproduced if c in art_chain_set]
        if len(shared) >= 2:
            return _analyse_reproduced(locus, pred, tair, art, tair_chains,
                                       art_chains, reproduced, shared)
        raise ValueError(f"{locus}: no AtRTD3-matched novel isoform and no reproduced pair")

    novel_exons = [s for s in (pred[novel_tx]["cds"] + pred[novel_tx]["utr"])
                   if not any(overlaps(s, b) for b in tair_exons)]
    novel_junctions = [j for j in novel_chain if j not in tair_junctions]
    tair_only = sorted(tair_junctions - set(novel_chain))
    # A TAIR10 intron the novel isoform covers with exon sequence is retained. Whether
    # CDS or UTR covers it matters: retention inside the CDS is a coding splice event,
    # retention that falls in a UTR usually just means the prediction stopped splicing
    # past its stop codon. Keep them apart instead of calling both "retained intron".
    def covered_by(segs):
        return [j for j in tair_only if any(s <= j[0] and e >= j[1] for s, e in segs)]

    def tair_leaves_unspliced(j):
        """Some TAIR10 isoform already spans this intron with exon sequence."""
        return any(s <= j[0] and e >= j[1] for s, e in tair_exons)

    # Retention only counts as absent-from-TAIR10 if no TAIR10 isoform already
    # leaves the same intron unspliced. Without this test a locus whose isoforms
    # differ by which intron they retain would be labelled "retained intron,
    # 0 TAIR10 transcripts" while TAIR10 plainly contains that retention.
    retained_cds = [j for j in covered_by(sorted(pred[novel_tx]["cds"]))
                    if not tair_leaves_unspliced(j)]
    retained_utr = [j for j in covered_by(sorted(pred[novel_tx]["utr"]))
                    if not tair_leaves_unspliced(j) and j not in retained_cds]

    if novel_exons:
        mode = "exon"
        feats = [(min(s for s, _ in novel_exons), max(e for _, e in novel_exons))]
    elif novel_junctions:
        mode = "junction"
        feats = novel_junctions
    elif retained_cds:
        mode = "retained"
        feats = retained_cds
    elif retained_utr:
        mode = "unspliced_utr"
        feats = retained_utr
    else:
        # No single novel exon, junction or retention: every difference from TAIR10 is
        # a junction that some TAIR10 isoform already uses (or already omits). Only the
        # combination is new - a real isoform, but a weak visual example. Shade the
        # full symmetric difference so a reader sees exactly what differs.
        mode = "combination"
        feats = sorted({j for j in novel_chain
                        if not all(j in c for c in tair_chains)} | set(tair_only))

    # AtRTD3 support for the highlighted feature, in disjoint categories
    key = feats[0]
    identical, site_shared, overlap_only, other = [], [], [], []
    for t, r in art.items():
        segs = sorted(r["exon"] or r["cds"])
        if mode == "exon":
            if key in segs:
                identical.append(t)
            elif any(s == key[0] or e == key[1] for s, e in segs):
                site_shared.append(t)
            elif any(overlaps(key, s) for s in segs):
                # spans the interval inside a longer exon - NOT the same cassette exon.
                # Kept separate so it is never counted as support for the exon.
                overlap_only.append(t)
            else:
                other.append(t)
        elif mode in ("retained", "unspliced_utr"):
            # support = long-read transcripts that also keep this intron as exon
            if any(s <= key[0] and e >= key[1] for s, e in segs):
                identical.append(t)
            elif key in chain(segs):
                site_shared.append(t)   # transcripts that splice it out instead
            else:
                other.append(t)
        else:
            jc = chain(segs)
            if key in jc:
                identical.append(t)
            elif any(j[0] == key[0] or j[1] == key[1] for j in jc):
                site_shared.append(t)
            else:
                other.append(t)
    for grp in (identical, site_shared, overlap_only, other):
        grp.sort(key=suffix)

    return dict(locus=locus, pred=pred, tair=tair, art=art,
                novel_tx=novel_tx, novel_match=novel_match, mode=mode, feats=feats,
                identical=identical, site_shared=site_shared,
                overlap_only=overlap_only, other=other,
                chrom=next(iter(tair.values()))["chr"],
                strand=next(iter(tair.values()))["strand"])



def _analyse_reproduced(locus, pred, tair, art, tair_chains, art_chains,
                        reproduced, shared):
    """Panel-A/B case: the model reproduced >=2 distinct TAIR10 chains and >=2 of them
    are documented by AtRTD3. The feature to highlight is what distinguishes the
    reproduced chains from each other - the alternatively spliced introns."""
    longest = max(reproduced, key=len)
    others = [c for c in reproduced if c != longest]
    feats = sorted({j for c in others for j in set(longest) ^ set(c)}
                   or set(longest))
    # every AtRTD3 transcript carrying one of the reproduced chains counts as support
    identical = sorted((t for t, c in art_chains.items() if c in reproduced), key=suffix)
    other = sorted((t for t in art if t not in identical), key=suffix)
    return dict(locus=locus, pred=pred, tair=tair, art=art,
                novel_tx=None, novel_match=identical[0] if identical else None,
                mode="reproduced", feats=feats,
                identical=identical, site_shared=[], overlap_only=[], other=other,
                n_reproduced=len(reproduced), n_shared=len(shared),
                chrom=next(iter(tair.values()))["chr"],
                strand=next(iter(tair.values()))["strand"])

def build_rows(a: dict) -> tuple[list, list]:
    """Rows as (label, record, colour, highlight) plus (n, group-name) bands."""
    rows, bands = [], []

    seen: set = set()
    tair_rows = []
    for t in sorted(a["tair"], key=lambda k: span(a["tair"][k])):
        c = chain(a["tair"][t]["cds"])
        if c in seen:
            continue
        seen.add(c)
        tair_rows.append((f".{len(tair_rows) + 1}", a["tair"][t], GREY, False))
    rows += tair_rows
    bands.append((len(tair_rows), "TAIR10"))

    if a["novel_tx"] is None:
        # one row per distinct predicted chain, so six identical predictions of the
        # same isoform do not become six rows
        pred_rows, seen_p = [], set()
        for t in sorted(a["pred"], key=suffix):
            c = chain(a["pred"][t]["cds"])
            if c in seen_p:
                continue
            seen_p.add(c)
            pred_rows.append((f"pred {len(pred_rows) + 1}", a["pred"][t], BLUE, True))
    else:
        pred_rows = [("novel", a["pred"][a["novel_tx"]], ORANGE, True)]
        for t in sorted(a["pred"], key=suffix):
            if t != a["novel_tx"]:
                pred_rows.append(("reproduced", a["pred"][t], BLUE, False))
    rows += pred_rows
    bands.append((len(pred_rows), "TransGenic"))

    # Dark green is reserved for the transcript whose whole CDS chain equals the
    # novel prediction. Transcripts that merely carry the highlighted feature are
    # mid green - otherwise a retained-intron panel would show three "exact match"
    # rows whose chains in fact differ. `placed` stops a transcript being drawn
    # twice when it falls into two categories (e.g. the chain match is also the
    # only non-supporting isoform at a two-transcript locus).
    placed = {a["novel_match"]} if a["novel_match"] else set()
    art_rows = ([(a["novel_match"], a["art"][a["novel_match"]],
                  GREEN if a["novel_tx"] is None else DGREEN, a["novel_tx"] is not None)]
                if a["novel_match"] else [])
    for t in ([x for x in a["identical"] if x not in placed]
              + [x for x in a["site_shared"] if x not in placed])[:MAX_SUPPORT]:
        art_rows.append((t, a["art"][t], GREEN, False))
        placed.add(t)
    # Context rows: show an overlap-only isoform first when one exists, so a reader
    # sees that some AtRTD3 transcripts cross the interval without sharing the feature.
    pool = ([x for x in a["overlap_only"] if x not in placed][:1]
            + [x for x in a["other"] if x not in placed]
            + [x for x in a["overlap_only"] if x not in placed][1:])
    for t in pool[:MAX_CONTEXT]:
        art_rows.append((t, a["art"][t], PALE, False))
        placed.add(t)
    rows += art_rows
    bands.append((len(art_rows), "AtRTD3"))

    return rows, bands


def support_note(a: dict) -> str:
    n_id, n_ss, n_all = len(a["identical"]), len(a["site_shared"]), len(a["art"])
    n_tair = len(a["tair"])
    n_tair_chains = len({chain(r["cds"]) for r in a["tair"].values()})
    mode = a["mode"]
    if mode == "reproduced":
        return (f"TransGenic reproduced {a['n_reproduced']} of {n_tair_chains} distinct "
                f"TAIR10 chains and predicted nothing outside them   |   "
                f"AtRTD3 documents {a['n_shared']} of them "
                f"({n_id} of {n_all} transcripts)")
    if mode == "exon":
        n_ov = len(a["overlap_only"])
        return (f"TAIR10: 0 of {n_tair} transcripts carry this exon   |   "
                f"AtRTD3: {n_id} identical, {n_ss} splice-site-shared, "
                f"{n_ov} overlap-only (of {n_all} tx)")
    if mode == "junction":
        return (f"TAIR10: 0 of {n_tair} transcripts use this junction   |   "
                f"AtRTD3: {n_id} identical, {n_ss} splice-site-shared (of {n_all} tx)")
    if mode in ("retained", "unspliced_utr"):
        where = "in the CDS" if mode == "retained" else "in the UTR"
        return (f"TAIR10: 0 of {n_tair} transcripts leave this intron unspliced   |   "
                f"AtRTD3: {n_id} leave it unspliced {where}, {n_ss} splice it out "
                f"(of {n_all} tx)")
    return (f"TAIR10: 0 of {n_tair} transcripts have this chain   |   "
            f"AtRTD3: exact chain match to {a['novel_match']} (of {n_all} tx)")


def headline(a: dict) -> tuple[str, str]:
    f0 = a["feats"][0]
    mode = a["mode"]
    if mode == "reproduced":
        n = len(a["feats"])
        return ("alternatively spliced intron" + ("s" if n > 1 else ""),
                f"{f0[0]:,}-{f0[1]:,}" + (f" +{n - 1} more" if n > 1 else ""))
    if mode == "exon":
        return ("novel exon", f"{f0[0]:,}–{f0[1]:,} ({f0[1] - f0[0] + 1} bp)")
    if mode == "junction":
        return ("novel junction", f"{f0[0]:,}–{f0[1]:,}")
    if mode in ("retained", "unspliced_utr"):
        n = len(a["feats"])
        kind = ("retained intron" if mode == "retained"
                else "unspliced intron in predicted UTR")
        return (kind + ("s" if n > 1 and mode == "retained" else ""),
                f"{f0[0]:,}–{f0[1]:,}" + (f" +{n - 1} more" if n > 1 else ""))
    return ("novel intron-chain combination",
            "each junction occurs in TAIR10; this chain does not")


# --------------------------------------------------------------------------- drawing

def draw_tx(ax, y: float, rec: dict, color: str, lw: float) -> None:
    cds, thin = thick_thin(rec)
    if not (cds or thin):
        return
    x0, x1 = span(rec)
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=1, solid_capstyle="butt")
    for s, e in thin:
        ax.add_patch(Rectangle((s, y - H_UTR / 2), e - s, H_UTR, color=color, zorder=2))
    for s, e in cds:
        ax.add_patch(Rectangle((s, y - H_CDS / 2), e - s, H_CDS, color=color, zorder=3))


def render(ax, a: dict, *, compact: bool) -> None:
    rows, bands = build_rows(a)
    n = len(rows)
    spans = [span(r) for _, r, _, _ in rows if span(r) != (0, 0)]
    gx0, gx1 = min(s for s, _ in spans), max(e for _, e in spans)
    width = gx1 - gx0
    pad = width * 0.02
    kb = 1000.0

    label_fs = 5.4 if compact else 6.0
    band_fs = 7.0 if compact else 8.5

    # A 63 bp exon in a 2.4 kb window is a hairline; give every shaded feature a
    # floor of 1.5% of the window so the highlight reads at print size.
    for s, e in a["feats"]:
        half = max((e - s) / 2, width * 0.015)
        mid = (s + e) / 2
        ax.axvspan((mid - half) / kb, (mid + half) / kb,
                   color=SHADE, alpha=0.22, zorder=0, lw=0)

    y = n
    for label, rec, color, hl in rows:
        scaled = {**rec, "cds": [(s / kb, e / kb) for s, e in rec["cds"]],
                  "utr": [(s / kb, e / kb) for s, e in rec["utr"]],
                  "exon": [(s / kb, e / kb) for s, e in rec["exon"]]}
        draw_tx(ax, y, scaled, color, lw=1.2 if hl else 0.7)
        if hl:
            ax.plot((gx1 + pad * 1.2) / kb, y,
                    marker="<" if a["strand"] == "-" else ">", color=color, markersize=5)
        # transcript labels ride the axes edge so they line up across every panel
        ax.text(1.015, y, label, transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=label_fs, color=color, clip_on=False)
        y -= 1

    # Group labels sit horizontally outside the axes with a colour bracket. Rotated
    # labels do not fit: "TransGenic" set vertically is longer than its two-row band
    # and collides with the neighbouring group.
    ytop = n
    yaxt = ax.get_yaxis_transform()
    for (cnt, name), col in zip(bands, (GREY, ORANGE, DGREEN)):
        ax.plot([-0.018, -0.018], [ytop + 0.34, ytop - cnt + 0.66], transform=yaxt,
                color=col, lw=2.0, solid_capstyle="butt", clip_on=False)
        ax.text(-0.030, ytop - (cnt - 1) / 2, name, transform=yaxt,
                va="center", ha="right", fontsize=band_fs, fontweight="bold",
                color=col, clip_on=False)
        ytop -= cnt

    kind, where = headline(a)
    ax.text(0.5, 1.015, f"{kind}   {where}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=label_fs + 0.4, color="#7A5C00")

    ax.set_xlim((gx0 - pad * 1.5) / kb, (gx1 + pad * 2.5) / kb)
    ax.set_ylim(0.35, n + 0.9)
    ax.set_yticks([])
    for sp in ("top", "left", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="x", labelsize=label_fs + 0.6)
    ax.set_xlabel(f"{a['chrom']} position (kb), {a['strand']} strand\n{support_note(a)}",
                  fontsize=label_fs + 0.8, labelpad=8, linespacing=1.9)
    title = (f"{a['locus']}  —  TransGenic reproduces the alternative transcripts "
             f"shared by TAIR10 and AtRTD3" if a["mode"] == "reproduced" else
             f"{a['locus']}  —  TransGenic predicts an AtRTD3-supported isoform "
             f"absent from TAIR10")
    ax.set_title(title, fontsize=band_fs, pad=16 if compact else 20)


def legend_handles() -> list:
    return [Line2D([0], [0], color=GREY, lw=4, label="TAIR10 reference"),
            Line2D([0], [0], color=ORANGE, lw=4, label="TransGenic (novel isoform)"),
            Line2D([0], [0], color=BLUE, lw=4, label="TransGenic (reproduced)"),
            Line2D([0], [0], color=DGREEN, lw=4, label="AtRTD3 (exact chain match)"),
            Line2D([0], [0], color=GREEN, lw=4, label="AtRTD3 (supports highlighted feature)"),
            Line2D([0], [0], color=PALE, lw=4, label="AtRTD3 (other isoforms)")]


def draw_locus(locus: str) -> dict:
    a = analyse(locus)
    rows, _ = build_rows(a)
    fig, ax = plt.subplots(figsize=(FIG_W, ROW_H * len(rows) + 1.9))
    render(ax, a, compact=False)
    fig.legend(handles=legend_handles(), ncol=3, frameon=False, fontsize=6.2,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure4_panelC_{locus}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    novel_lbl = a["novel_tx"].split(".")[-1] if a["novel_tx"] else "-"
    print(f"{locus:10s} {a['mode']:12s} novel={novel_lbl:6s} "
          f"match={str(a['novel_match']):15s} rows={len(rows):2d} "
          f"AtRTD3 {len(a['identical'])} identical / {len(a['site_shared'])} site-shared "
          f"of {len(a['art'])}")
    return a


def draw_sheet(loci: list, name: str = "all", ncol: int = 2,
               panel_labels: list | None = None,
               prefix: str = "figure4_panelC_") -> None:
    """Multi-panel sheet. ncol=1 gives the single-column layout used for the
    manuscript figure; ncol=2 the wider supplementary sheet. panel_labels stamps
    (C), (D), ... on each panel so the legend can refer to them directly."""
    analyses = [analyse(x) for x in loci]
    heights = [len(build_rows(a)[0]) for a in analyses]
    nrow = (len(analyses) + ncol - 1) // ncol
    row_max = [max(heights[r * ncol:(r + 1) * ncol]) for r in range(nrow)]
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(FIG_W * (1.65 if ncol > 1 else 1.05),
                                      sum(ROW_H * h + 1.35 for h in row_max)),
                             gridspec_kw={"height_ratios": [h + 4.0 for h in row_max]})
    flat = axes.ravel()
    for i, (ax, a) in enumerate(zip(flat, analyses)):
        render(ax, a, compact=True)
        if panel_labels:
            ax.text(-0.085, 1.06, panel_labels[i], transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="left")
    for ax in flat[len(analyses):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), ncol=6 if ncol > 1 else 3, frameon=False,
               fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 0.002))
    fig.tight_layout(rect=(0, 0.035 if ncol > 1 else 0.055, 1, 1), h_pad=2.6, w_pad=3.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{prefix}{name}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"sheet -> {prefix}{name}.png ({len(analyses)} panels, {ncol} col)")


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(set(LOCI) | set(FIGURE4))
    for loc in targets:
        draw_locus(loc)
    if not sys.argv[1:]:
        print()
        # the rebuilt manuscript Figure 4
        draw_sheet(FIGURE4, name="figure4_example_loci_rebuilt", ncol=2, prefix="",
                   panel_labels=["(A)", "(B)", "(C)", "(D)", "(E)"])
        # manuscript Figure 4C-E only, if a three-panel version is wanted
        draw_sheet(STRONG, name="strong3", ncol=1, panel_labels=["(C)", "(D)", "(E)"])
        # supplementary: every Panel-C candidate, including the three weak ones
        draw_sheet(LOCI, name="all", ncol=2)
