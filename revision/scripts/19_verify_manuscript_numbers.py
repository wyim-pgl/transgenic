#!/usr/bin/env python3
"""Cross-check every quantitative claim in manuscript_v2.md against the result files.

Each check prints PASS/FAIL with the manuscript value, the value re-derived from
the source file, and the file the source value came from. Exit code is the
number of failures.

Usage:
    python 19_verify_manuscript_numbers.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "transgenic" / "revision" / "results"
CMP = ROOT / "transgenic_comparison"

TRAINING_SPECIES = [
    "A_thaliana", "B_distachyon", "G_max", "O_sativa", "P_patens",
    "P_trichocarpa", "S_bicolor", "S_italica", "V_vinifera",
]

results: list[tuple[bool, str, str, str, str]] = []


def check(name: str, claimed, derived, source: str, tol: float = 0.051) -> None:
    """Record a check. Numeric comparison uses an absolute tolerance."""
    if isinstance(claimed, (int, float)) and isinstance(derived, (int, float)):
        ok = abs(float(claimed) - float(derived)) <= tol
    else:
        ok = claimed == derived
    results.append((ok, name, str(claimed), str(derived), source))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def jload(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def stats_metrics(path: Path) -> dict[str, tuple[float, float]]:
    """Parse a gffcompare .stats file into {level: (sensitivity, precision)}."""
    out: dict[str, tuple[float, float]] = {}
    pat = re.compile(r"^\s*([A-Za-z ]+?) level:\s*([\d.-]+)\s*\|\s*([\d.-]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def stats_counts(path: Path) -> dict[str, int]:
    txt = path.read_text()
    out: dict[str, int] = {}
    m = re.search(r"Query mRNAs :\s*(\d+) in\s*(\d+) loci", txt)
    if m:
        out["query_mrna"], out["query_loci"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"Reference mRNAs :\s*(\d+) in\s*(\d+) loci", txt)
    if m:
        out["ref_mrna"], out["ref_loci"] = int(m.group(1)), int(m.group(2))
    return out


# ---------------------------------------------------------------- BUSCO ----
busco_path = CMP / "busco_summary_final.normalized.csv"
busco = load_csv(busco_path)
by_tool: dict[str, dict[str, float]] = {}
for r in busco:
    by_tool.setdefault(r["Tool"], {})[r["Species"]] = float(r["Complete (%)"])

for tool, label, lo, hi, mean in [
    ("transgenic400Mprompt", "prompted 400M", 78.1, 100.0, 95.2),
    ("transgenic400M", "de novo 400M", 48.0, 78.6, 62.4),
    ("transgenic160M", "de novo 160M", 35.3, 61.4, 49.7),
]:
    vals = list(by_tool[tool].values())
    check(f"BUSCO {label} min", lo, min(vals), busco_path.name)
    check(f"BUSCO {label} max", hi, max(vals), busco_path.name)
    check(f"BUSCO {label} mean", mean, sum(vals) / len(vals), busco_path.name)

check("BUSCO Z. mays prompted 400M", 78.1, by_tool["transgenic400Mprompt"]["Z_mays"], busco_path.name)
check("BUSCO Z. mays Helixer", 97.6, by_tool["helixer"]["Z_mays"], busco_path.name)

# ---------------------------------------------- transcript-level isoforms ----
p = RES / "prompted400Mbeam1_vs_TAIR10" / "summary_report.json"
j = jload(p)
t = j["transcript_level_metrics"]
src = "prompted400Mbeam1_vs_TAIR10/summary_report.json"
check("prompted vs TAIR10 isoform recall %", 61.9, t["isoform_recall"] * 100, src)
check("prompted vs TAIR10 isoform precision %", 74.4, t["isoform_precision"] * 100, src)
check("prompted vs TAIR10 isoform F1 %", 67.6, t["isoform_f1"] * 100, src)
check("prompted vs TAIR10 distinct ref matched", 21919, t["distinct_ref_matched"], src)
check("prompted vs TAIR10 total reference", 35386, t["total_reference"], src)
# summary_report.json's total_predicted is 29,923: the tmap header counted as a record.
# The manuscript quotes the true count, so derive it from the prediction file.
_n_mrna = sum(1 for line in (CMP / "standardized_results"
              / "A_thaliana_transgenic400Mprompt_beam1.gff3").open()
              if not line.startswith("#") and line.split("\t")[2:3] == ["mRNA"])
check("prompted vs TAIR10 total predicted (Methods)", 29922, _n_mrna,
      "standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3")
check("prompted vs TAIR10 exact matches", 22260, t["exact_matches"], src)
check("prompted vs TAIR10 duplicate queries", 341, t["duplicate_exact_matches"], src)

ic = j["isoform_count_analysis"]
denom = ic["exact_count_matches"] + ic["underpredictions"] + ic["overpredictions"]
check("isoform-count exact (n)", 22955, ic["exact_count_matches"], src)
check("isoform-count under (n)", 4419, ic["underpredictions"], src)
check("isoform-count over (n)", 39, ic["overpredictions"], src)
check("isoform-count denominator 27,413", 27413, denom, src + " (exact+under+over)")
check("isoform-count exact %", 83.7, 100 * ic["exact_count_matches"] / denom, src)

p = RES / "prompted400Mbeam1_vs_AtRTD3" / "summary_report.json"
j = jload(p)
t = j["transcript_level_metrics"]
src = "prompted400Mbeam1_vs_AtRTD3/summary_report.json"
check("prompted vs AtRTD3 recall %", 12.5, t["isoform_recall"] * 100, src)
check("prompted vs AtRTD3 precision %", 72.5, t["isoform_precision"] * 100, src)
check("prompted vs AtRTD3 distinct ref matched", 21252, t["distinct_ref_matched"], src)
check("prompted vs AtRTD3 total reference", 169499, t["total_reference"], src)
check("prompted vs AtRTD3 duplicate queries", 442, t["duplicate_exact_matches"], src)
ic = j["isoform_count_analysis"]
check("AtRTD3 isoform-count exact (n)", 8635, ic["exact_count_matches"], src)
check("AtRTD3 isoform-count under (n)", 18239, ic["underpredictions"], src)
check("AtRTD3 isoform-count over (n)", 52, ic["overpredictions"], src)
check("AtRTD3 missed genes (no prediction)", 14003, ic["missed_genes"], src)
check("AtRTD3 novel (outside documented loci)", 487, ic["novel_genes"], src)

p = RES / "denovo400M_vs_TAIR10" / "summary_report.json"
t = jload(p)["transcript_level_metrics"]
src = "denovo400M_vs_TAIR10/summary_report.json"
check("de novo vs TAIR10 recall %", 33.0, t["isoform_recall"] * 100, src)
check("de novo vs TAIR10 precision %", 42.4, t["isoform_precision"] * 100, src)
check("de novo vs TAIR10 distinct ref matched", 11681, t["distinct_ref_matched"], src)

# ------------------------------------------------------- splice events ----
p = RES / "prompted400Mbeam1_vs_TAIR10" / "splice_events_report.json"
ev = jload(p)["per_event_type"]
src = "prompted400Mbeam1_vs_TAIR10/splice_events_report.json"
for etype, rec, prec in [("SE", 6.9, 52.5), ("A5SS", 6.8, 28.3), ("A3SS", 6.6, 43.9), ("IR", 12.2, 41.8)]:
    check(f"prompted {etype} recall %", rec, ev[etype]["recall"] * 100, src)
    check(f"prompted {etype} precision %", prec, ev[etype]["precision"] * 100, src)

p = RES / "prompted400Mbeam1_vs_AtRTD3" / "splice_events_report.json"
ev = jload(p)["per_event_type"]
src = "prompted400Mbeam1_vs_AtRTD3/splice_events_report.json"
precs = [ev[e]["precision"] * 100 for e in ("SE", "A5SS", "A3SS", "IR")]
recs = [ev[e]["recall"] * 100 for e in ("SE", "A5SS", "A3SS", "IR")]
check("AtRTD3 event precision min %", 36.5, min(precs), src)
check("AtRTD3 event precision max %", 55.4, max(precs), src)
check("AtRTD3 event recall min %", 0.5, min(recs), src)
check("AtRTD3 event recall max %", 1.0, max(recs), src)

p = RES / "denovo400M_vs_TAIR10" / "splice_events_report.json"
dn = jload(p)
ev = dn["per_event_type"]
src = "denovo400M_vs_TAIR10/splice_events_report.json"
recs = [ev[e]["recall"] * 100 for e in ("SE", "A5SS", "A3SS", "IR")]
check("de novo event recall min %", 0.0, min(recs), src)
check("de novo event recall max %", 0.1, max(recs), src)
check("de novo events predicted (total)", 342, dn["summary"]["total_predicted_events"], src)
check("de novo events matched (total)", 7, dn["summary"]["total_matched_events"], src)
check("TAIR10 reference events (total)", 14643, dn["summary"]["total_reference_events"], src)

# ------------------------------------------------------------- alt-only ----
alt = RES / "altonly"
m = stats_metrics(alt / "A_thaliana_transgenic400Mprompt_beam1_vs_AtRTD3.stats")
src = "altonly/A_thaliana_transgenic400Mprompt_beam1_vs_AtRTD3.stats"
check("alt-only AtRTD3 prompted base Sn", 59.4, m["Base"][0], src)
check("alt-only AtRTD3 prompted base Pr", 77.3, m["Base"][1], src)
check("alt-only AtRTD3 prompted transcript Sn", 10.3, m["Transcript"][0], src)
check("alt-only AtRTD3 prompted transcript Pr", 47.3, m["Transcript"][1], src)
check("alt-only AtRTD3 prompted intron-chain Pr", 59.5, m["Intron chain"][1], src)

m = stats_metrics(alt / "A_thaliana_transgenic400M_vs_AtRTD3.stats")
src = "altonly/A_thaliana_transgenic400M_vs_AtRTD3.stats"
check("alt-only AtRTD3 de novo base Sn", 56.9, m["Base"][0], src)
check("alt-only AtRTD3 de novo base Pr", 75.5, m["Base"][1], src)

m = stats_metrics(alt / "A_thaliana_augustusSampling_vs_AtRTD3.stats")
src = "altonly/A_thaliana_augustusSampling_vs_AtRTD3.stats"
check("alt-only AtRTD3 AUGUSTUS transcript Pr", 16.3, m["Transcript"][1], src)
check("alt-only AtRTD3 AUGUSTUS intron-chain Pr", 18.0, m["Intron chain"][1], src)

m = stats_metrics(alt / "A_thaliana_augustusSampling_vs_TAIR10.stats")
c = stats_counts(alt / "A_thaliana_augustusSampling_vs_TAIR10.stats")
src = "altonly/A_thaliana_augustusSampling_vs_TAIR10.stats"
check("alt-only TAIR10 AUGUSTUS transcript Sn", 21.9, m["Transcript"][0], src)
check("alt-only TAIR10 AUGUSTUS transcript Pr", 2.5, m["Transcript"][1], src)
check("AUGUSTUS predicted transcripts", 69049, c["query_mrna"], src)
check("AUGUSTUS loci with a prediction", 25641, c["query_loci"], src)
check("AUGUSTUS transcripts per predicted locus", 2.7, c["query_mrna"] / c["query_loci"], src)
c2 = stats_counts(alt / "A_thaliana_transgenic400Mprompt_beam1_vs_TAIR10.stats")
# The alt-only run reports 29,922 query mRNAs vs 29,923 in the full-reference run:
# one transcript falls outside the alt-only reference's sequence set. The
# manuscript's precision denominator (29,923) comes from the full-reference run.
check("prompted predicted transcripts (alt-only run)", 29922, c2["query_mrna"], "altonly/...prompt_beam1_vs_TAIR10.stats")
check("prompted loci with a prediction", 27213, c2["query_loci"], "altonly/...prompt_beam1_vs_TAIR10.stats")
check("prompted transcripts per predicted locus", 1.1, c2["query_mrna"] / c2["query_loci"], "altonly/...prompt_beam1_vs_TAIR10.stats")
check("AUGUSTUS/TransGenic candidate ratio ~2.3x", 2.3, c["query_mrna"] / c2["query_mrna"], "derived", tol=0.05)
check("alt-only TAIR10 prompted transcript Sn", 9.3, m2 := stats_metrics(alt / "A_thaliana_transgenic400Mprompt_beam1_vs_TAIR10.stats")["Transcript"][0], "altonly/...prompt_beam1_vs_TAIR10.stats")

p = RES / "augustusSampling_vs_TAIR10" / "summary_report.json"
if p.exists():
    t = jload(p)["transcript_level_metrics"]
    src = "augustusSampling_vs_TAIR10/summary_report.json"
    check("AUGUSTUS full TAIR10 transcript recall %", 37.7, t["isoform_recall"] * 100, src)
    check("AUGUSTUS full TAIR10 transcript precision %", 25.7, t["isoform_precision"] * 100, src)
    check("AUGUSTUS full TAIR10 transcript F1 %", 30.6, t["isoform_f1"] * 100, src)
else:
    results.append((False, "AUGUSTUS full TAIR10 summary_report.json", "expected", "MISSING", str(p)))

p = RES / "augustusSampling_vs_AtRTD3" / "summary_report.json"
if p.exists():
    t = jload(p)["transcript_level_metrics"]
    src = "augustusSampling_vs_AtRTD3/summary_report.json"
    check("AUGUSTUS full AtRTD3 transcript recall %", 9.2, t["isoform_recall"] * 100, src)
    check("AUGUSTUS full AtRTD3 transcript precision %", 29.8, t["isoform_precision"] * 100, src)
else:
    results.append((False, "AUGUSTUS full AtRTD3 summary_report.json", "expected", "MISSING", str(p)))

# ------------------------------------------------------ self-consistency ----
sc_path = RES / "selfconsistency_summary.csv"
sc = {r["species_variant"]: r for r in load_csv(sc_path)}
src = sc_path.name
for suffix, label, lo, hi in [
    ("_transgenic400M", "de novo 400M", 8.9, 27.4),
    ("_transgenic400Mprompt", "prompted 400M", 85.4, 98.2),
    ("_REF", "reference", 92.9, 100.0),
]:
    vals = [float(v["pct_fully_consistent"]) for k, v in sc.items() if k.endswith(suffix)]
    check(f"self-consistency {label} min %", lo, min(vals), src)
    check(f"self-consistency {label} max %", hi, max(vals), src)
    if label == "prompted 400M":
        check("self-consistency prompted mean %", 94.1, sum(vals) / len(vals), src)

beam1 = sc.get("A_thaliana_transgenic400Mprompt_beam1")
if beam1:
    check("duplicate transcripts (n)", 311, int(beam1["duplicate_transcripts"]), src)
    check("beam1 transcripts checked", 29713, int(beam1["n_transcripts"]), src)
    check("duplicate %", 1.0, 100 * int(beam1["duplicate_transcripts"]) / int(beam1["n_transcripts"]), src)
else:
    results.append((False, "beam1 self-consistency row", "expected", "MISSING", src))

# --------------------------------------------------------------- TSS/TES ----
tss_path = RES / "tss_tes_summary.csv"
tss = load_csv(tss_path)
src = tss_path.name
den = [r for r in tss if r["species_variant"].endswith("_transgenic400M")]
pro = [r for r in tss if r["species_variant"].endswith("_transgenic400Mprompt")]
check("de novo TSS exact min %", 2.9, min(float(r["TSS_exact_pct"]) for r in den), src)
check("de novo TSS exact max %", 22.1, max(float(r["TSS_exact_pct"]) for r in den), src)
check("de novo TSS +-50 min %", 33.4, min(float(r["TSS_within50_pct"]) for r in den), src)
check("de novo TSS +-50 max %", 66.2, max(float(r["TSS_within50_pct"]) for r in den), src)
check("de novo TSS +-100 min %", 45.7, min(float(r["TSS_within100_pct"]) for r in den), src)
check("de novo TSS +-100 max %", 81.1, max(float(r["TSS_within100_pct"]) for r in den), src)
check("de novo TSS median offset min", 27, min(int(r["TSS_median_delta"]) for r in den), src, tol=0)
check("de novo TSS median offset max", 123, max(int(r["TSS_median_delta"]) for r in den), src, tol=0)
check("prompted TSS exact min %", 66.7, min(float(r["TSS_exact_pct"]) for r in pro), src)
check("prompted TSS exact max %", 98.0, max(float(r["TSS_exact_pct"]) for r in pro), src)
check("prompted TSS +-50 min %", 87.0, min(float(r["TSS_within50_pct"]) for r in pro), src)
check("prompted TSS +-50 max %", 99.5, max(float(r["TSS_within50_pct"]) for r in pro), src)

# ------------------------------------------------------------ vocabulary ----
fs_path = RES / "feature_stats_summary.csv"
fs = {r["species"]: r for r in load_csv(fs_path)}
src = fs_path.name
species13 = [k for k in fs if k != "AtRTD3"]
within = {k: float(fs[k]["pct_within_limits"]) for k in species13}
check("vocabulary min coverage % (13 refs)", 99.085, min(within.values()), src)
check("Z. mays genes over limits", 360, int(fs["Z_mays"]["over_any"]), src)
check("S. lycopersicum genes over limits", 2, int(fs["S_lycopersicum"]["over_any"]), src)
check("refs fully representable (all features)", 11, sum(1 for v in within.values() if v == 100.0), src, tol=0)
check("refs fully representable for CDS", 12, sum(1 for k in species13 if int(fs[k]["over_150_cds"]) == 0), src, tol=0)
cds_maxes = sorted((int(fs[k]["cds_max"]) for k in species13), reverse=True)
check("highest CDS segment count (S. lycopersicum)", 201, cds_maxes[0], src, tol=0)
check("next-highest CDS segment count (Z. mays)", 146, cds_maxes[1], src, tol=0)
check("max 5'UTR segments (Z. mays)", 232, int(fs["Z_mays"]["utr5_max"]), src, tol=0)
check("max 3'UTR segments (Z. mays)", 322, int(fs["Z_mays"]["utr3_max"]), src, tol=0)

# ------------------------------------------------------------- AS stats ----
as_path = RES / "asstats_summary.csv"
ast = {r["species"]: r for r in load_csv(as_path)}
src = as_path.name
pcts = {k: float(ast[k]["pct_multi_transcript"]) for k in TRAINING_SPECIES}
check("multi-transcript min % (training)", 14.2, min(pcts.values()), src)
check("multi-transcript max % (training)", 70.3, max(pcts.values()), src)
check("max transcripts per gene (training)", 26, max(int(ast[k]["max_transcripts_per_gene"]) for k in TRAINING_SPECIES), src, tol=0)

# --------------------------------------------------------- Chr4 pilot ----
p = ROOT / "transgenic" / "ath_chr4_comparison_FINAL.stats"
m, c = stats_metrics(p), stats_counts(p)
src = "ath_chr4_comparison_FINAL.stats"
check("Chr4 base Sn", 88.1, m["Base"][0], src)
check("Chr4 base Pr", 67.7, m["Base"][1], src)
check("Chr4 transcript Sn", 39.2, m["Transcript"][0], src)
check("Chr4 transcript Pr", 4.4, m["Transcript"][1], src)
check("Chr4 locus Sn", 51.2, m["Locus"][0], src)
check("Chr4 locus Pr", 51.9, m["Locus"][1], src)
check("Chr4 predicted transcripts", 47741, c["query_mrna"], src)
check("Chr4 predicted loci", 3914, c["query_loci"], src)
check("Chr4 reference transcripts", 5350, c["ref_mrna"], src)

# ---------------------------------------------------------- benchmark F1 ----
gc_path = CMP / "gffcompare_summary.csv"
gc = load_csv(gc_path)


def f1(sn: float, pr: float) -> float:
    return 0.0 if sn + pr == 0 else 2 * sn * pr / (sn + pr)


gmap = {(r["Species"], r["Tool"]): r for r in gc}
src = gc_path.name
# These are the 13-species benchmark values reported in Supplemental Table S2. They
# are NOT the 92.2% / 71.1% quoted in the abstract and first Results paragraph, which
# come from the original held-out test-split evaluation behind Figure 3.
r = gmap[("A_thaliana", "transgenic400M")]
check("A. thaliana 400M de novo base F1 (Table S2)", 92.4,
      f1(float(r["Base_Sensitivity"]), float(r["Base_Precision"])), src)
r = gmap[("Z_mays", "transgenic400M")]
check("Z. mays 400M de novo base F1 (Table S2)", 70.6,
      f1(float(r["Base_Sensitivity"]), float(r["Base_Precision"])), src)

# --------------------------------------------------- Figure 4 panels -----
# The Figure 4 loci are named in the legend with their support counts, so those
# counts are claims like any other. They are re-derived from the per-locus extracts
# of the original prompted prediction, which is the evaluation the submitted figure
# was drawn from - not from the second inference behind Figure S4, whose loci differ.
FIG4 = RES / "fig4_forensics" / "panelC_examples" / "original"


def _chains(path: Path, gtf: bool) -> dict[str, tuple]:
    """{transcript: CDS intron chain} for one locus extract."""
    cds: dict[str, list[tuple[int, int]]] = {}
    key = re.compile(r'transcript_id "([^"]+)"' if gtf else r"Parent=([^;]+)")
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        m = key.search(f[8])
        if m:
            cds.setdefault(m.group(1), []).append((int(f[3]), int(f[4])))
    out = {}
    for tx, segs in cds.items():
        s = sorted(segs)
        out[tx] = tuple((s[i][1], s[i + 1][0]) for i in range(len(s) - 1))
    return out


def panel_counts(locus: str) -> dict:
    """Distinct-chain bookkeeping for one Figure 4 locus."""
    tair = _chains(FIG4 / f"{locus}_tair.gff3", gtf=False)
    pred = _chains(FIG4 / f"{locus}_pred.gff3", gtf=False)
    art = _chains(FIG4 / f"{locus}_atrtd.gtf", gtf=True)
    tset, pset = set(tair.values()), set(pred.values())
    return {
        "tair_chains": len(tset),
        "reproduced": len(pset & tset),
        "outside": len(pset - tset),
        "art_tx": len(art),
        # AtRTD3 transcripts whose whole chain is one the model returned
        "art_supporting": sum(1 for c in art.values() if c in pset),
        "novel_match": sorted(t for t, c in art.items() if c in pset - tset),
        "tair_with_novel": sum(1 for c in tair.values() if c in pset - tset),
    }


for locus, chains, art_sup, art_tx in (("AT4G10840", 2, 2, 4), ("AT3G50550", 2, 4, 5)):
    p = panel_counts(locus)
    src = f"panelC_examples/original/{locus}_*"
    check(f"{locus} distinct TAIR10 chains (Fig. 4)", chains, p["tair_chains"], src)
    check(f"{locus} chains reproduced (Fig. 4)", chains, p["reproduced"], src)
    check(f"{locus} predicted outside TAIR10 (Fig. 4)", 0, p["outside"], src)
    check(f"{locus} AtRTD3 transcripts supporting (Fig. 4)", art_sup, p["art_supporting"], src)
    check(f"{locus} AtRTD3 transcripts at locus (Fig. 4)", art_tx, p["art_tx"], src)

p = panel_counts("AT1G19650")
src = "panelC_examples/original/AT1G19650_*"
check("AT1G19650 distinct TAIR10 chains (Fig. 4C)", 2, p["tair_chains"], src)
check("AT1G19650 TAIR10 transcripts with the novel chain (Fig. 4C)", 0, p["tair_with_novel"], src)
check("AT1G19650 AtRTD3 transcripts at locus (Fig. 4C)", 14, p["art_tx"], src)
check("AT1G19650 AtRTD3 exact chain match (Fig. 4C)", "AT1G19650.13",
      p["novel_match"][0] if p["novel_match"] else "-", src)

# Panel classification over the original evaluation, as stated in Methods.
rows = [r.split("\t") for r in
        (FIG4 / "panels_original.tsv").read_text().splitlines()[1:] if r.strip()]
panels = [r[1] for r in rows]
src = "panelC_examples/original/panels_original.tsv"
check("Fig. 4 panel A/B loci recovering two chains (Methods)", 25, panels.count("A"), src)
check("Fig. 4 panel A/B loci recovering three or more (Methods)", 2, panels.count("B"), src)
check("Fig. 4 panel C qualifying loci (Methods)", 1, panels.count("C"), src)

# --------------------------------------------------- Figure S4 panels ----
# Figure S4 comes from the prompted run over every evaluation locus, so its counts sit
# in the same file the transcript-level metrics do and can be checked the same way.
FIGS4 = RES / "fig4_forensics" / "panelC_examples" / "prompted_full"
scan = load_csv_tsv = [r.split("\t") for r in
                       (FIGS4 / "panelC_prompted.tsv").read_text().splitlines() if r.strip()]
hdr, rows = scan[0], scan[1:]
cat = hdr.index("category")
src = "panelC_examples/prompted_full/panelC_prompted.tsv"
check("Panel-C qualifying loci, full prompted run (Results, Methods)", 45, len(rows), src)
check("of those, chain uses a junction absent from TAIR10 (Results)", 13,
      sum(1 for r in rows if r[cat] == "junction"), src)
check("of those, novel only in chain combination (Fig. S4 legend)", 32,
      sum(1 for r in rows if r[cat] == "combination"), src)

by_locus = {r[0]: dict(zip(hdr, r)) for r in rows}
# AT4G30510 is quoted in the Results with both kinds of support, which are different
# numbers: seven transcripts use the junction, two reproduce the whole chain.
r = by_locus["AT4G30510"]
check("AT4G30510 AtRTD3 transcripts at locus (Results)", 9, int(r["atrtd_transcripts"]), src)
check("AT4G30510 AtRTD3 using the novel junction (Results)", 7,
      int(r["atrtd_carrying_novel_junction"]), src)
check("AT4G30510 AtRTD3 reproducing the full chain (Results)", 2,
      int(r["atrtd_sharing_chain"]), src)
check("AT4G30510 TAIR10 isoforms using the junction (Results)", 0,
      0 if r["category"] == "junction" else -1, src)

# The AT2G37450 cassette exon is quoted with its coordinates in the Results.
ex = [ln.split("\t") for ln in (FIGS4 / "AT2G37450_pred.gff3").read_text().splitlines()
      if ln and ln.split("\t")[2] == "CDS"]
cassette = [(int(f[3]), int(f[4])) for f in ex if int(f[4]) - int(f[3]) + 1 == 63]
check("AT2G37450 cassette exon length (Results)", 63,
      cassette[0][1] - cassette[0][0] + 1 if cassette else 0,
      "panelC_examples/prompted_full/AT2G37450_pred.gff3")
check("AT2G37450 cassette exon start (Results)", 15724032,
      cassette[0][0] if cassette else 0,
      "panelC_examples/prompted_full/AT2G37450_pred.gff3")

# ------------------------------------- prompt composition (Results) ------
# The Results now state what the scored prediction set contains, because the returned
# annotation retains the transcript it was given. Both numbers are re-derived here.
import collections as _c

def _cds_by_tx(path, gff3=True):
    d = _c.defaultdict(lambda: _c.defaultdict(list))
    lr = re.compile(r"GM=([^;\s]+)") if gff3 else re.compile(r'gene_id "([^"]+)"')
    tr = re.compile(r"Parent=([^;]+)") if gff3 else re.compile(r'transcript_id "([^"]+)"')
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, tx = lr.search(f[8]), tr.search(f[8])
            if g and tx:
                d[g.group(1).replace(".TAIR10", "")][tx.group(1)].append((int(f[3]), int(f[4])))
    return {g: {k: tuple(sorted(v)) for k, v in txs.items()} for g, txs in d.items()}

_pred = _cds_by_tx(CMP / "standardized_results" / "A_thaliana_transgenic400Mprompt_beam1.gff3")
_tair = _cds_by_tx(ROOT / "transgenic" / "revision" / "data" / "TAIR10" / "TAIR10.gtf", gff3=False)
_prim = {}
for _ln in (ROOT / "transgenic" / "revision" / "data" / "TAIR10"
            / "primary_transcript_ids.txt").read_text().splitlines():
    if _ln.strip():
        _prim[_ln.strip().split(".")[0]] = _ln.strip()
_total = sum(len(v) for v in _pred.values())
_supplied = sum(1 for g, txs in _pred.items()
                for s in txs.values()
                if s == _tair.get(g, {}).get(_prim.get(g)))
src = "standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3"
check("completed annotation, total transcripts (Results)", 29922, _total, src)
check("of those, reproducing the supplied model (Results)", 28536, _supplied, src)
check("of those, not the supplied model (Results)", 1386, _total - _supplied, src)
check("of those, distinct added structures (Results)", 1103,
      sum(len({s for s in txs.values() if s != _tair.get(g, {}).get(_prim.get(g))})
          for g, txs in _pred.items()), src)

# The added isoforms scored on their own, with the supplied transcript removed from both
# the prediction and the reference (28_score_added_isoforms.py).
_added = jload(RES / "added_isoforms_A_thaliana.json")
src2 = "added_isoforms_A_thaliana.json"
check("added isoforms matching a TAIR10 alternative, % (Results, Abstract)", 18.1,
      _added["precision_vs_TAIR10_alternatives_pct"], src2)
check("added isoforms matching an AtRTD3 transcript, % (Results)", 18.5,
      _added["precision_vs_AtRTD3_pct"], src2)
check("TAIR10 alternative transcripts recovered by additions, % (Results)", 3.6,
      _added["recall_of_TAIR10_alternatives_pct"], src2)
check("loci receiving an addition (Results)", 1076,
      _added["loci_with_at_least_one_addition"], src2)

# The AUGUSTUS baseline scored the same way (28_score_added_isoforms.py --augustus).
_aug = jload(RES / "added_isoforms_augustus.json")
src3 = "added_isoforms_augustus.json"
check("AUGUSTUS additions matching a TAIR10 alternative, % (Results)", 1.3,
      _aug["precision_vs_TAIR10_alternatives_pct"], src3)
check("AUGUSTUS additions matching an AtRTD3 transcript, % (Results)", 10.2,
      _aug["precision_vs_AtRTD3_pct"], src3)
check("AUGUSTUS TAIR10 alternatives recovered, % (Results)", 10.3,
      _aug["recall_of_TAIR10_alternatives_pct"], src3)
check("AUGUSTUS added structures (Results)", 43433, _aug["added_transcripts"], src3)
check("added-structure ratio, AUGUSTUS to TransGenic (Results)", 39,
      round(_aug["added_transcripts"] / _added["added_transcripts"]), src3)
check("supplied share of the returned annotation, % (Results)", 95.4,
      round(100 * _supplied / _total, 1), src, tol=0.051)


# --------------------------------------- BUSCO beam invariance (Table S3) ----
# Table S3 states that filtering to the top beam leaves completeness unchanged. That is a
# claim about a measurement, so it is pinned here rather than asserted.
_bp = jload(RES / "busco_beam_pilot.json")
src4 = "busco_beam_pilot.json"
check("BUSCO completeness, top beam (Table S3)", 97.6, _bp["top"]["complete"], src4)
check("BUSCO completeness, both beams (Table S3)", 97.6, _bp["both"]["complete"], src4)
check("BUSCO single-copy, top beam (Table S3)", 88.5, _bp["top"]["single"], src4)
check("BUSCO duplicated, both beams (Table S3)", 97.6, _bp["both"]["duplicated"], src4)

# ------------------------------------------- Table S2 sanity (Figure 5) ----
# A sequence-name mismatch makes GFFCompare score every level at 0.0, which reads as a
# result rather than as a failure; it silently zeroed two S. lycopersicum rows once.
_gc = load_csv(CMP / "gffcompare_summary.csv")
_zero = [f"{r['Species']}/{r['Tool']}" for r in _gc
         if float(r["Base_Sensitivity"]) == 0 and float(r["Base_Precision"]) == 0]
check("Table S2 rows scoring 0.0 at base level", 0, len(_zero),
      "gffcompare_summary.csv" + (f" ({', '.join(_zero[:3])})" if _zero else ""))
check("Table S2 rows total", 128, len(_gc), "gffcompare_summary.csv")

# ------------------------------- AUGUSTUS splice events (Results) ----------
# Quoted in the Results from Table S4c; pinned so the pair cannot drift apart.
_s4c = load_csv(ROOT / "supplementary" / "TableS4c_splice_events.csv")


def _events(pred_key: str) -> tuple[int, float]:
    tot = matched = 0
    for r in _s4c:
        if r.get("Reference", "").startswith("TAIR10") and pred_key in r.get("Prediction set", ""):
            tot += int(r["Predicted events (n)"])
            matched += int(r["Matched events (n)"])
    return tot, round(100 * matched / tot, 1) if tot else 0.0


_aug_n, _aug_p = _events("AUGUSTUS")
_cmp_n, _cmp_p = _events("reference-prompted")
src5 = "TableS4c_splice_events.csv"
check("AUGUSTUS splice events predicted (Results)", 41917, _aug_n, src5)
check("AUGUSTUS splice-event precision, % (Results)", 4.4, _aug_p, src5)
check("completion-mode splice events predicted (Results)", 3033, _cmp_n, src5)
check("completion-mode splice-event precision, % (Results)", 37.5, _cmp_p, src5)


def _matched(pred_key: str) -> int:
    return sum(int(r["Matched events (n)"]) for r in _s4c
               if r.get("Reference", "").startswith("TAIR10") and pred_key in r.get("Prediction set", ""))


check("AUGUSTUS splice events matched (Results)", 1855, _matched("AUGUSTUS"), src5)
check("completion-mode splice events matched (Results)", 1137, _matched("reference-prompted"), src5)

# ------------------------------- TAIR10 nuclear vs total (Table S8) --------
# Table S8 counts nuclear genes only while the Results quote the full annotation; the two
# were unreconciled and read as a contradiction. Both are re-derived here.
import collections as _c2
_genes: dict = {}
_tx = 0
with (ROOT / "transgenic" / "revision" / "data" / "TAIR10" / "TAIR10.gtf").open() as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "transcript":
            continue
        g = re.search(r'gene_id "([^"]+)"', f[8])
        if g:
            _genes[g.group(1)] = f[0]
            _tx += 1
_org_g = sum(1 for c in _genes.values() if c in ("ChrM", "ChrC"))
src6 = "TAIR10.gtf"
check("TAIR10 transcripts, full annotation (Results)", 35386, _tx, src6)
check("TAIR10 genes, full annotation", 27416, len(_genes), src6)
check("TAIR10 nuclear genes (Table S8)", 27206, len(_genes) - _org_g, src6)

# The Results state that organellar loci stay in the evaluation and 209 received
# predictions — a claim about the prediction file, not the reference.
_org_pred = 0
with (CMP / "standardized_results" / "A_thaliana_transgenic400Mprompt_beam1.gff3").open() as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        if f[0] in ("ChrM", "ChrC"):
            _org_pred += 1
check("organellar loci receiving predictions (Results)", 209, _org_pred,
      "standardized_results/A_thaliana_transgenic400Mprompt_beam1.gff3")

# --------------------------- self-prompted transcript counts (Fig. 5 legend) --
# The legend quotes 93,690 and points the reader at Table S5, which reports 93,063
# because it excludes organellar transcripts. Assert the relationship, don't assume it.
_sp = CMP / "standardized_results" / "A_thaliana_transgenic400M_prompt_denovo.gff3"
_sp_total = _sp_org = 0
with _sp.open() as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) > 2 and f[2] == "mRNA":
            _sp_total += 1
            if f[0] in ("ChrM", "ChrC"):
                _sp_org += 1
src7 = "standardized_results/A_thaliana_transgenic400M_prompt_denovo.gff3"
check("self-prompted transcripts, all loci (Fig. 5 legend)", 93690, _sp_total, src7)
check("self-prompted transcripts, nuclear only (Table S5)", 93063, _sp_total - _sp_org, src7)

# ------------------- claims aggregated over Table S2 (Results l.40) --------
# These four are rankings and counts computed across all 128 benchmark rows. They were
# checked by hand once each; one of them was wrong for two rounds. They also change
# wholesale whenever the benchmark is re-scored, so they are derived here, not asserted.
_bench = {(r["Species"], r["Tool"]): r for r in load_csv(CMP / "gffcompare_summary.csv")}
_species = sorted({s for s, _ in _bench})
_DE_NOVO_OTHER = ("annevo", "helixer", "tiberius", "tiberius_softmasked")


def _f1(species: str, tool: str, level: str) -> float | None:
    r = _bench.get((species, tool))
    if not r:
        return None
    sn, pr = float(r[f"{level}_Sensitivity"]), float(r[f"{level}_Precision"])
    return None if sn + pr == 0 else 2 * sn * pr / (sn + pr)


_wins = _within2 = 0
for sp in _species:
    tg = _f1(sp, "transgenic400M", "Base")
    best = max(v for v in (_f1(sp, t, "Base") for t in _DE_NOVO_OTHER) if v is not None)
    if tg >= best:
        _wins += 1
    elif best - tg <= 2.0:
        _within2 += 1
src8 = "gffcompare_summary.csv"
check("species where de novo 400M matches/exceeds others at base level (Results)", 8, _wins, src8)
check("species within 2 base-F1 points of the best (Results)", 3, _within2, src8)
check("wins and within-2 are disjoint and bounded (Results)", True, _wins + _within2 <= 13, src8)

_margins = []
for sp in _species:
    comp = _f1(sp, "transgenic400Mprompt", "Transcript")
    best_dn = max(v for v in (_f1(sp, t, "Transcript")
                              for t in _DE_NOVO_OTHER + ("transgenic400M", "transgenic160M"))
                  if v is not None)
    _margins.append(comp - best_dn)
check("completion-mode margin over best de novo, minimum (Results)", 17.1, min(_margins), src8)
check("completion-mode margin over best de novo, maximum (Results)", 47.8, max(_margins), src8)
check("species where completion beats the best de novo (Results)", 13,
      sum(1 for m in _margins if m > 0), src8)

_leader = {"tiberius": 0, "helixer": 0, "annevo": 0}
_tib_soft = 0
_tib_unmasked: list[str] = []
for sp in _species:
    cand = {t: _f1(sp, t, "Transcript") for t in _DE_NOVO_OTHER}
    # The Z. mays soft-masked run scores 0.0 and is disclosed as not a performance
    # measurement in three places; it must not count as a leader candidate.
    cand = {k: v for k, v in cand.items() if v is not None and v > 0}
    top = max(cand, key=cand.get)
    if top.startswith("tiberius"):
        _leader["tiberius"] += 1
        if top == "tiberius_softmasked":
            _tib_soft += 1
        else:
            _tib_unmasked.append(sp)
    else:
        _leader[top] += 1
check("species led by Tiberius at transcript level (Results)", 10, _leader["tiberius"], src8)
check("of those, with soft-masked input (Results)", 9, _tib_soft, src8)
check("unmasked-Tiberius leader species (Results)", "V_vinifera",
      ",".join(_tib_unmasked), src8)
check("species led by Helixer at transcript level (Results)", 2, _leader["helixer"], src8)
check("species led by ANNEVO at transcript level (Results)", 1, _leader["annevo"], src8)

# ------------------------ Figure 3 anchors and trends (Results l.34) -------
# Sourced outside revision/results/, so nothing else checks them. The .stats files carry
# Sn/Pr to one decimal, which is why the recomputed Z. mays F1 is 71.2 against the
# published 71.1 — inside the rounding of its own inputs. Tolerance reflects that.
_fig3 = RES / "fig3_original" / "denovo"
for _sp, _label, _claim in (("TAIR10", "A. thaliana", 92.2), ("Zm0", "Z. mays", 71.1)):
    _m = re.search(r"Base level:\s*([\d.]+)\s*\|\s*([\d.]+)",
                   (_fig3 / f"{_sp}_noPost.stats").read_text())
    _sn, _pr = float(_m.group(1)), float(_m.group(2))
    # Sn/Pr are printed to one decimal, so the true F1 lies in a computable interval.
    # Asserting the interval is stricter than a flat tolerance and says why it exists.
    _lo = 2 * (_sn - 0.05) * (_pr - 0.05) / ((_sn - 0.05) + (_pr - 0.05))
    _hi = 2 * (_sn + 0.05) * (_pr + 0.05) / ((_sn + 0.05) + (_pr + 0.05))
    check(f"{_label} anchor {_claim} within Sn/Pr rounding [{_lo:.2f}, {_hi:.2f}]",
          "yes", "yes" if _lo - 0.05 <= _claim <= _hi + 0.05 else f"NO ({_claim})",
          f"fig3_original/denovo/{_sp}_noPost.stats")


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman's rho as Pearson on ranks, with averaged ranks for ties.

    The d-squared shortcut is only valid when every rank is distinct. These inputs are
    base-level F1 carried to one decimal over dozens of bins, where ties are common.
    """
    def _ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return 0.0 if den == 0 else num / den


for _sub, _label, _rho, _nbins in (("binByLength", "gene length", -0.11, 27),
                                   ("binByCDS", "CDS count", -0.88, 44)):
    _xs, _ys, _with_query = [], [], 0
    for _f in sorted((_fig3 / _sub).glob("TAIR10-*.stats"),
                     key=lambda p: int(re.search(r"TAIR10-(\d+)", p.name).group(1))):
        _txt = _f.read_text()
        _m = re.search(r"Base level:\s*([\d.]+)\s*\|\s*([\d.]+)", _txt)
        _n = re.search(r"Reference mRNAs :\s*(\d+)", _txt)
        if not _m or not _n or int(_n.group(1)) == 0:
            continue
        _s, _p = float(_m.group(1)), float(_m.group(2))
        _q = re.search(r"Query mRNAs :\s*(\d+)", _txt)
        if _q and int(_q.group(1)) > 0:
            _with_query += 1
        _xs.append(int(re.search(r"TAIR10-(\d+)", _f.name).group(1)))
        _ys.append(0.0 if _s + _p == 0 else 2 * _s * _p / (_s + _p))
    check(f"Spearman rho, base F1 vs {_label} (Results)", _rho, _spearman(_xs, _ys),
          f"fig3_original/denovo/{_sub}/", tol=0.006)
    check(f"populated bins, {_label} (Results)", _nbins, len(_xs), f"fig3_original/denovo/{_sub}/")
    # A bin scoring F1 = 0 is a real result only if predictions were made there; if the
    # run produced nothing, the decline would be missing data rather than accuracy.
    check(f"bins with predictions, {_label} (Results)", _nbins, _with_query,
          f"fig3_original/denovo/{_sub}/")

# ------------------------------------------------------------- report ----
fails = [r for r in results if not r[0]]
width = max(len(r[1]) for r in results) + 2
print(f"{'CHECK':<{width}} {'MANUSCRIPT':>14}  {'SOURCE VALUE':>14}   FILE")
print("-" * (width + 50))
for ok, name, claimed, derived, source in results:
    flag = "  " if ok else "**"
    print(f"{flag}{name:<{width - 2}} {claimed:>14}  {derived[:14]:>14}   {source}")
print("-" * (width + 50))
print(f"{len(results) - len(fails)}/{len(results)} checks passed, {len(fails)} MISMATCH")
if fails:
    print("\nMISMATCHES:")
    for _, name, claimed, derived, source in fails:
        print(f"  - {name}: manuscript={claimed}  source={derived}  ({source})")
sys.exit(len(fails))
