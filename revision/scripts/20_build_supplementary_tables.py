#!/usr/bin/env python3
"""Build submission-ready supplemental tables from the result files.

Writes an English markdown document with every table fully populated, plus one
CSV per table under `supplementary/` for upload as separate supplemental files.

Usage:
    python 20_build_supplementary_tables.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "transgenic" / "revision" / "results"
CMP = ROOT / "transgenic_comparison"
PB = ROOT / "polishing_benchmark" / "results"

# Table S11. (json stem, prompt label, CDS source, UTR source, gene boundary). The filtered
# row reuses the arm above it, so its stem carries the `_filtered` suffix rather than naming
# a separate run.
PROMPT_TRANSFER_ARMS = [
    ("tair10selfutr", "Reference annotation", "TAIR10", "TAIR10", "TAIR10"),
    ("tair10helixerframeutr", "Reference features, predicted frame", "TAIR10", "TAIR10", "Helixer"),
    ("tair10helixerframeutr_filtered", "Reference features, predicted frame, after ORF and "
     "splice-site filter", "TAIR10", "TAIR10", "Helixer"),
    ("tair10self", "Reference CDS, no UTR", "TAIR10", "none", "TAIR10"),
    ("tair10helixerframe", "Reference CDS, no UTR, predicted frame", "TAIR10", "none", "Helixer"),
    ("helixertairutr", "Predicted CDS, reference UTR", "Helixer", "TAIR10", "CDS+UTR span"),
    ("annevotairutr", "Predicted CDS, reference UTR", "ANNEVO", "TAIR10", "CDS+UTR span"),
    ("helixer", "Helixer output as supplied", "Helixer", "Helixer", "Helixer"),
    ("annevo", "ANNEVO output as supplied", "ANNEVO", "none", "ANNEVO"),
    ("braker3", "BRAKER3 output as supplied", "BRAKER3", "BRAKER3", "BRAKER3"),
    ("gemoma", "GeMoMa output as supplied", "GeMoMa", "GeMoMa", "GeMoMa"),
    ("egapx", "EGAPx output as supplied", "EGAPx", "EGAPx", "EGAPx"),
]

SPECIES_NAME = {
    "A_thaliana": "Arabidopsis thaliana",
    "B_distachyon": "Brachypodium distachyon",
    "B_rapa": "Brassica rapa",
    "G_max": "Glycine max",
    "L_sativa": "Lactuca sativa",
    "O_sativa": "Oryza sativa",
    "P_patens": "Physcomitrium patens",
    "P_trichocarpa": "Populus trichocarpa",
    "S_bicolor": "Sorghum bicolor",
    "S_italica": "Setaria italica",
    "S_lycopersicum": "Solanum lycopersicum",
    "V_vinifera": "Vitis vinifera",
    "Z_mays": "Zea mays",
    "AtRTD3": "AtRTD3 (A. thaliana long-read transcriptome)",
}
SPECIES_ORDER = [k for k in SPECIES_NAME if k != "AtRTD3"]

TOOL_NAME = {
    "annevo": "ANNEVO v2.2",
    "helixer": "Helixer (land_plant v0.3)",
    "tiberius": "Tiberius v1.1.7",
    "tiberius_softmasked": "Tiberius v1.1.7 (soft-masked input)",
    "transgenic160M": "TransGenic 160M, de novo",
    "transgenic160Mprompt": "TransGenic 160M, reference-prompted",
    "transgenic160M_prompt_denovo": "TransGenic 160M, self-prompted",
    "transgenic400M": "TransGenic 400M, de novo",
    "transgenic400Mprompt": "TransGenic 400M, reference-prompted",
    "transgenic400M_prompt_denovo": "TransGenic 400M, self-prompted",
}
TOOL_ORDER = list(TOOL_NAME)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def jload(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def stats_metrics(path: Path) -> dict[str, tuple[float, float]]:
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


def f1(sn: float, pr: float) -> float:
    return 0.0 if sn + pr == 0 else 2 * sn * pr / (sn + pr)


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


class Out:
    """Collects markdown sections and per-table CSVs."""

    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.md: list[str] = []

    def table(self, slug: str, title: str, note: str, header: list[str], rows: list[list]) -> None:
        self.md.append(f"## {title}\n")
        if note:
            self.md.append(note + "\n")
        self.md.append(md_table(header, rows) + "\n")
        with (self.outdir / f"{slug}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    def text(self, s: str) -> None:
        self.md.append(s + "\n")


def supplemental_figure_legends() -> str:
    """Lift the Figure S1-S3 legends out of the manuscript so that the supplemental
    document is self-contained."""
    text = (ROOT / "manuscript_v2.md").read_text()
    start = text.index("**Figure S1.")
    end = text.index("**Table S1.")
    return text[start:end].rstrip()


def build(out: Out) -> None:
    out.text("# Supplemental Information\n")
    out.text(
        "**TransGenic: a transformer-based framework for direct DNA-to-annotation "
        "translation.** Lomas, Ramazan, Cushman, Tang, and Yim.\n\n"
        "All values in the tables below are generated directly from the analysis outputs "
        "deposited with the software repository by "
        "`transgenic/revision/scripts/20_build_supplementary_tables.py`, and each table is "
        "also provided as a machine-readable CSV file. The source file for each table is "
        "named in its legend. Sensitivity (Sn) and precision (Pr) follow the GFFCompare "
        "definitions; F1 is the harmonic mean of the two.\n"
    )
    out.text("## Supplemental figure legends\n")
    out.text(supplemental_figure_legends() + "\n")
    out.text("## Supplemental tables\n")

    # ------------------------------------------------------ Table S1 ----
    s1_rows = [
        ["Arabidopsis thaliana", "TAIR10", "Lamesch et al., 2012", "Training / test"],
        ["Vitis vinifera", "PN_T2T", "Shi et al., 2023", "Training"],
        ["Glycine max", "Wm82 ISU-01 v2.1", "Espina et al., 2024", "Training"],
        ["Populus trichocarpa", "v4", "Tuskan et al., 2006", "Training"],
        ["Sorghum bicolor", "v5", "McCormick et al., 2018", "Training"],
        ["Brachypodium distachyon", "v3", "The International Brachypodium Initiative, 2010", "Training"],
        ["Setaria italica", "v2", "Bennetzen et al., 2012", "Training"],
        ["Oryza sativa", "MSU v7.0", "Ouyang et al., 2007", "Training"],
        ["Physcomitrium patens", "v3", "Lang et al., 2018", "Training"],
        ["Zea mays", "RefGen_V4", "Jiao et al., 2017", "Held-out test"],
        ["Brassica rapa", "BrapaO_302V_711 v1.1", "Wang et al., 2011", "Extended benchmark"],
        ["Lactuca sativa", "v5 (Lsativa_467)", "Reyes-Chin-Wo et al., 2017", "Extended benchmark"],
        ["Solanum lycopersicum", "ITAG5.0 (SL5.0)", "The Tomato Genome Consortium, 2012", "Extended benchmark"],
    ]
    out.table(
        "TableS1_genome_versions",
        "Table S1. Genome assemblies and annotations used for training, testing, and the extended benchmark",
        "All genomes and annotations were obtained from Phytozome (Goodstein et al., 2012) "
        "except *S. lycopersicum* (ITAG5.0) and *L. sativa* v5. *Zea mays* was withheld from "
        "training entirely. The moss species is given under its current accepted name, "
        "*Physcomitrium patens*; the assembly and its describing publication use the earlier "
        "name *Physcomitrella patens*.",
        ["Species", "Assembly / annotation version", "Reference", "Role"],
        s1_rows,
    )

    # ------------------------------------------------------ Table S2 ----
    gc = {(r["Species"], r["Tool"]): r for r in load_csv(CMP / "gffcompare_summary.csv")}
    rows = []
    for sp in SPECIES_ORDER:
        for tool in TOOL_ORDER:
            r = gc.get((sp, tool))
            if r is None:
                continue
            bs, bp = float(r["Base_Sensitivity"]), float(r["Base_Precision"])
            es, ep = float(r["Exon_Sensitivity"]), float(r["Exon_Precision"])
            ts, tp = float(r["Transcript_Sensitivity"]), float(r["Transcript_Precision"])
            rows.append([
                SPECIES_NAME[sp], TOOL_NAME[tool],
                f"{bs:.1f}", f"{bp:.1f}", f"{f1(bs, bp):.1f}",
                f"{es:.1f}", f"{ep:.1f}", f"{f1(es, ep):.1f}",
                f"{ts:.1f}", f"{tp:.1f}", f"{f1(ts, tp):.1f}",
            ])
    out.table(
        "TableS2_gffcompare_benchmark",
        "Table S2. GFFCompare benchmark across 13 plant species",
        "Source: `transgenic_comparison/gffcompare_summary.csv`, produced by "
        "`revision/scripts/29_rescore_benchmark_v0126.py` with **GFFCompare v0.12.6**, the same "
        "release used for the isoform analyses in Table S4 and Figure 6, so the two are directly "
        "comparable. An earlier version of this table used v0.12.10, which scores transcript-level "
        "matches more permissively; base-level values are identical between the releases. The "
        "*TransGenic 400M, reference-prompted* rows are additionally filtered to the top-ranked "
        "beam, because the source GFF3 for that configuration exported both beam hypotheses as "
        "separate gene records (54,826 gene records for 27,413 A. thaliana loci). The "
        "*V. vinifera* Tiberius soft-masked row is absent because that run did not complete "
        "(see Table S3). "
        "ANNEVO, Helixer, and Tiberius were run on whole-genome sequences; TransGenic was "
        "run on individual reference-annotated single-gene loci, a setting that does not "
        "require gene-boundary resolution and therefore favours TransGenic (see Discussion). "
        "Reference-prompted TransGenic predictions are conditioned on the reference primary "
        "transcript and are not de novo performance. The *L. sativa* TransGenic 400M "
        "self-prompted cell is absent: it was seeded by a de novo run that terminated early "
        "and was therefore invalidated and withdrawn (see Table S3 footnote). "
        "Figure 5 displays the seven primary configurations from this table.",
        ["Species", "Tool / mode",
         "Base Sn (%)", "Base Pr (%)", "Base F1 (%)",
         "Exon Sn (%)", "Exon Pr (%)", "Exon F1 (%)",
         "Transcript Sn (%)", "Transcript Pr (%)", "Transcript F1 (%)"],
        rows,
    )

    # ------------------------------------------------------ Table S3 ----
    bu = {(r["Species"], r["Tool"]): r for r in
          load_csv(CMP / "busco_summary_final.normalized.csv")}
    rows = []
    for sp in SPECIES_ORDER:
        for tool in TOOL_ORDER:
            r = bu.get((sp, tool))
            if r is None:
                continue
            rows.append([
                SPECIES_NAME[sp], TOOL_NAME[tool], r["Complete (%)"],
                r["Single (S)"], r["Duplicated (D)"], r["Fragmented (F)"],
                r["Missing (M)"], r["Total BUSCOs"],
            ])
    out.table(
        "TableS3_busco",
        "Table S3. BUSCO functional completeness across 13 plant species",
        "Source: `transgenic_comparison/busco_summary_final.csv`, normalized by "
        "`18_normalize_busco_summary.py` (BUSCO v6.0.0, viridiplantae_odb10, n = 425, "
        "protein mode; proteins extracted with PASA v2.5.3 `gff3_file_to_proteins.pl`). "
        "Complete (%) = (single-copy + duplicated) / 425. The reference-prompted rows here were scored on the unfiltered "
        "export, which holds every locus twice (see Methods), so most conserved genes "
        "appear as duplicated rather than single-copy. Table S2 and Figure 5 report the "
        "deduplicated file for the same configuration; scoring it alone for A. thaliana "
        "returns the identical completeness (C 97.6% either way, F 2.1%, M 0.2%) and changes only "
        "the single-copy/duplicated split, from S 0.0/D 97.6 to S 88.5/D 9.2, so the completeness "
        "percentages in this table are comparable with Table S2 "
        "(`revision/results/busco_beam_pilot.json`). As in Table S2, TransGenic was scored on "
        "individual reference-annotated single-gene loci whereas the other tools were run on "
        "whole genomes, a setting that favours TransGenic. The *Z. mays* Tiberius soft-masked row is "
        "present but reflects a run that produced almost no output overlapping the reference, "
        "as described in the Table S2 legend, and should not be read as Tiberius performance. "
        "Two cells are absent: *L. sativa* TransGenic 400M "
        "self-prompted (invalidated — its de novo prompt source terminated after 1,792 of "
        "38,910 loci; all affected artefacts are quarantined under "
        "`transgenic_comparison/invalidated_stall/`) and *V. vinifera* Tiberius with "
        "soft-masked input (run not completed).",
        ["Species", "Tool / mode", "Complete (%)", "Single-copy (n)",
         "Duplicated (n)", "Fragmented (n)", "Missing (n)", "Total BUSCOs"],
        rows,
    )

    # ------------------------------------------------------ Table S4 ----
    alt = RES / "altonly"
    alt_cfg = [
        ("A_thaliana_transgenic400M_vs_TAIR10", "TransGenic 400M, de novo", "TAIR10"),
        ("A_thaliana_transgenic400Mprompt_beam1_vs_TAIR10", "TransGenic 400M, reference-prompted (deduplicated)", "TAIR10"),
        ("A_thaliana_augustusSampling_vs_TAIR10", "AUGUSTUS v3.5.0 posterior sampling", "TAIR10"),
        ("A_thaliana_transgenic400M_vs_AtRTD3", "TransGenic 400M, de novo", "AtRTD3"),
        ("A_thaliana_transgenic400Mprompt_beam1_vs_AtRTD3", "TransGenic 400M, reference-prompted (deduplicated)", "AtRTD3"),
        ("A_thaliana_augustusSampling_vs_AtRTD3", "AUGUSTUS v3.5.0 posterior sampling", "AtRTD3"),
    ]
    levels = ["Base", "Exon", "Intron", "Intron chain", "Transcript", "Locus"]
    rows = []
    for stem, label, ref in alt_cfg:
        p = alt / f"{stem}.stats"
        m, c = stats_metrics(p), stats_counts(p)
        row = [ref, label, c.get("query_mrna"), c.get("ref_mrna")]
        for lv in levels:
            sn, pr = m[lv]
            row += [f"{sn:.1f}", f"{pr:.1f}"]
        rows.append(row)
    header = ["Reference (alternative transcripts only)", "Prediction set",
              "Predicted transcripts (n)", "Reference transcripts (n)"]
    for lv in levels:
        header += [f"{lv} Sn (%)", f"{lv} Pr (%)"]
    out.table(
        "TableS4a_altonly_gffcompare",
        "Table S4a. Alternative-transcript-only evaluation (GFFCompare v0.12.6)",
        "Source: `transgenic/revision/results/altonly/*.stats`. The primary (first annotated) "
        "transcript of every gene was removed from the reference before comparison, so these "
        "metrics score only the alternative isoforms. All three prediction sets were scored "
        "with identical commands. AUGUSTUS parameters for *A. thaliana* are estimated from "
        "TAIR annotations, so agreement with the TAIR10-derived reference is partly circular; "
        "AtRTD3 provides independent long-read evidence.",
        header,
        rows,
    )

    full_cfg = [
        ("denovo400M_vs_TAIR10", "TransGenic 400M, de novo", "TAIR10"),
        ("prompted400Mbeam1_vs_TAIR10", "TransGenic 400M, reference-prompted (deduplicated)", "TAIR10"),
        ("augustusSampling_vs_TAIR10", "AUGUSTUS v3.5.0 posterior sampling", "TAIR10"),
        ("denovo400M_vs_AtRTD3", "TransGenic 400M, de novo", "AtRTD3"),
        ("prompted400Mbeam1_vs_AtRTD3", "TransGenic 400M, reference-prompted (deduplicated)", "AtRTD3"),
        ("augustusSampling_vs_AtRTD3", "AUGUSTUS v3.5.0 posterior sampling", "AtRTD3"),
    ]
    rows = []
    for d, label, ref in full_cfg:
        p = RES / d / "summary_report.json"
        if not p.exists():
            continue
        t = jload(p)["transcript_level_metrics"]
        rows.append([
            ref, label,
            t["total_reference"], predicted_transcripts(d), t["exact_matches"],
            t["duplicate_exact_matches"], t["distinct_ref_matched"],
            f"{t['isoform_recall'] * 100:.1f}",
            f"{t['isoform_precision'] * 100:.1f}",
            f"{t['isoform_f1'] * 100:.1f}",
        ])
    out.table(
        "TableS4b_transcript_level",
        "Table S4b. Transcript-level isoform accuracy against the complete references",
        "Source: `transgenic/revision/results/*/summary_report.json`, except the *Predicted "
        "transcripts* column, which counts mRNA records in each prediction file directly; "
        "`summary_report.json` reports one more per prediction set, a header row counted as "
        "a record. A transcript counts as an exact match when its intron chain is identical to a "
        "reference transcript (GFFCompare class code '='). "
        "Isoform "
        "recall = distinct reference transcripts matched / reference transcripts; isoform "
        "precision = predicted transcripts with class code '=' / predicted transcripts "
        "(duplicate matches to the same reference transcript are counted once in the recall "
        "numerator but individually in the precision numerator).",
        ["Reference", "Prediction set", "Reference transcripts (n)",
         "Predicted transcripts (n)", "Exact matches (n)", "Duplicate matches (n)",
         "Distinct reference transcripts matched (n)",
         "Isoform recall (%)", "Isoform precision (%)", "Isoform F1 (%)"],
        rows,
    )

    rows = []
    for d, label, ref in full_cfg:
        p = RES / d / "splice_events_report.json"
        if not p.exists():
            continue
        rep = jload(p)
        for et in ("SE", "A5SS", "A3SS", "IR"):
            e = rep["per_event_type"][et]
            rows.append([
                ref, label, et, e["reference_count"], e["predicted_count"],
                e["matched_count"], f"{e['recall'] * 100:.2f}",
                f"{e['precision'] * 100:.2f}", f"{e['f1'] * 100:.2f}",
            ])
    out.table(
        "TableS4c_splice_events",
        "Table S4c. Alternative splicing event recovery by event type",
        "Source: `transgenic/revision/results/*/splice_events_report.json`. Events were "
        "classified from exon coordinates using rMATS-style definitions (SE, exon skipping; "
        "A5SS/A3SS, alternative 5′/3′ splice site; IR, intron retention) and matched between "
        "reference and prediction within the same locus. De novo generation emits "
        "approximately one transcript per locus and therefore produces almost no alternative "
        "splicing events, which is why its event recall is near zero.",
        ["Reference", "Prediction set", "Event type", "Reference events (n)",
         "Predicted events (n)", "Matched events (n)", "Recall (%)",
         "Precision (%)", "F1 (%)"],
        rows,
    )

    # ------------------------------------------------------ Table S5 ----
    sc = load_csv(RES / "selfconsistency_summary.csv")
    order = {v: i for i, v in enumerate(
        ["REF", "transgenic160M", "transgenic160Mprompt", "transgenic160M_prompt_denovo",
         "transgenic400M", "transgenic400Mprompt", "transgenic400Mprompt_beam1",
         "transgenic400M_prompt_denovo"])}

    def split_variant(key: str) -> tuple[str, str]:
        for sp in SPECIES_ORDER:
            if key.startswith(sp + "_"):
                return sp, key[len(sp) + 1:]
        raise ValueError(key)

    parsed = []
    for r in sc:
        try:
            sp, var = split_variant(r["species_variant"])
        except ValueError:
            continue
        parsed.append((sp, var, r))
    parsed.sort(key=lambda x: (SPECIES_ORDER.index(x[0]), order.get(x[1], 99)))
    label_of = dict(TOOL_NAME)
    label_of["REF"] = "Reference annotation"
    label_of["transgenic400Mprompt_beam1"] = "TransGenic 400M, reference-prompted (deduplicated)"
    rows = [[
        SPECIES_NAME[sp], label_of.get(var, var), r["n_transcripts"], r["checked"],
        r["frame_fail"], r["no_start_atg"], r["no_terminal_stop"], r["internal_stop"],
        r["fully_consistent"], r["pct_fully_consistent"], r["duplicate_transcripts"],
        r["mean_isoforms_per_gene"],
    ] for sp, var, r in parsed]
    out.table(
        "TableS5_self_consistency",
        "Table S5. Structural self-consistency of predicted and reference transcripts",
        "Source: `transgenic/revision/results/selfconsistency_summary.csv`. Each transcript's "
        "CDS was assembled from the genome in transcription order (minus-strand fragments "
        "concatenated in ascending genomic order and then reverse-complemented) and tested "
        "for divisibility by three, an ATG start, a terminal stop codon, and the absence of "
        "internal stop codons. Organellar genes were excluded because RNA editing invalidates "
        "genomic translation checks. Duplicate transcripts are identical exon chains within "
        "the same locus.",
        ["Species", "Prediction set", "Transcripts (n)", "Checked (n)",
         "Frame failure (n)", "No ATG start (n)", "No terminal stop (n)",
         "Internal stop (n)", "Fully consistent (n)", "Fully consistent (%)",
         "Duplicate transcripts (n)", "Mean isoforms per gene"],
        rows,
    )

    # ------------------------------------------------------ Table S6 ----
    tss = load_csv(RES / "tss_tes_summary.csv")
    parsed = []
    for r in tss:
        try:
            sp, var = split_variant(r["species_variant"])
        except ValueError:
            continue
        if r["transcripts_compared"] in ("0", ""):
            continue
        parsed.append((sp, var, r))
    parsed.sort(key=lambda x: (SPECIES_ORDER.index(x[0]), order.get(x[1], 99)))
    rows = [[
        SPECIES_NAME[sp], label_of.get(var, var), r["transcripts_compared"],
        r["TSS_exact_pct"], r["TSS_within50_pct"], r["TSS_within100_pct"],
        r["TSS_median_delta"], r["TES_exact_pct"], r["TES_within50_pct"],
        r["TES_within100_pct"], r["TES_median_delta"],
    ] for sp, var, r in parsed]
    out.table(
        "TableS6_tss_tes",
        "Table S6. Transcription start site (TSS) and transcription end site (TES) positional accuracy",
        "Source: `transgenic/revision/results/tss_tes_summary.csv`. For each predicted "
        "transcript the distance to the nearest same-strand reference transcript terminus at "
        "the matched locus was measured; median offsets are in nucleotides. Self-prompted "
        "rows are omitted because their transcripts could not be linked to reference loci by "
        "identifier.",
        ["Species", "Prediction set", "Transcripts compared (n)",
         "TSS exact (%)", "TSS ±50 nt (%)", "TSS ±100 nt (%)", "TSS median offset (nt)",
         "TES exact (%)", "TES ±50 nt (%)", "TES ±100 nt (%)", "TES median offset (nt)"],
        rows,
    )

    # ------------------------------------------------------ Table S7 ----
    fs = load_csv(RES / "feature_stats_summary.csv")
    fs_by = {r["species"]: r for r in fs}
    rows = []
    for sp in SPECIES_ORDER + ["AtRTD3"]:
        r = fs_by.get(sp)
        if r is None:
            continue
        rows.append([
            SPECIES_NAME[sp], r["n_genes"], r["cds_max"], r["cds_p99"],
            r["utr5_max"], r["utr5_p99"], r["utr3_max"], r["utr3_p99"],
            r["over_150_cds"], r["over_50_utr5"], r["over_50_utr3"],
            r["over_any"], r["pct_within_limits"],
        ])
    out.table(
        "TableS7_vocabulary_coverage",
        "Table S7. Per-gene feature counts and GSF vocabulary coverage",
        "Source: `transgenic/revision/results/feature_stats_summary.csv`. Counts are unique "
        "feature segments per gene, pooled over all annotated transcripts of that gene. The "
        "GSF vocabulary allows 150 CDS, 50 five-prime UTR, and 50 three-prime UTR segments "
        "per gene. p99 is the 99th percentile. Genes over any limit is not the sum of the "
        "three preceding columns because a gene may exceed more than one limit.",
        ["Annotation", "Genes (n)", "Max CDS segments", "CDS p99",
         "Max 5′-UTR segments", "5′-UTR p99", "Max 3′-UTR segments", "3′-UTR p99",
         "Genes over 150 CDS (n)", "Genes over 50 5′-UTR (n)",
         "Genes over 50 3′-UTR (n)", "Genes over any limit (n)",
         "Genes within all limits (%)"],
        rows,
    )

    # ------------------------------------------------------ Table S8 ----
    ast = {r["species"]: r for r in load_csv(RES / "asstats_summary.csv")}
    rows = []
    for sp in SPECIES_ORDER + ["AtRTD3"]:
        r = ast.get(sp)
        if r is None:
            continue
        rows.append([
            SPECIES_NAME[sp], r["n_genes"], r["n_transcripts"],
            r["multi_transcript_genes"], r["pct_multi_transcript"],
            r["mean_transcripts_per_gene"], r["max_transcripts_per_gene"],
        ])
    out.table(
        "TableS8_as_statistics",
        "Table S8. Alternative splicing content of the reference annotations",
        "Source: `transgenic/revision/results/asstats_summary.csv`. Counts here are nuclear "
        "only: the *A. thaliana* row gives 27,206 genes and 35,176 transcripts, against the "
        "27,416 and 35,386 of the full TAIR10 annotation, the difference being 210 organellar "
        "genes excluded because RNA editing invalidates genomic translation checks. The nine training "
        "annotations are *A. thaliana*, *B. distachyon*, *G. max*, *O. sativa*, "
        "*P. patens*, *P. trichocarpa*, *S. bicolor*, *S. italica*, and *V. vinifera*; among "
        "these the maximum number of annotated transcripts at any single gene is 26 "
        "(*P. patens*). *Z. mays* was withheld from training.",
        ["Annotation", "Genes (n)", "Transcripts (n)", "Multi-transcript genes (n)",
         "Multi-transcript genes (%)", "Mean transcripts per gene",
         "Max transcripts per gene"],
        rows,
    )

    # ------------------------------------------------------ Table S9 ----
    p = ROOT / "transgenic" / "ath_chr4_comparison_FINAL.stats"
    m, c = stats_metrics(p), stats_counts(p)
    rows = [[lv, f"{m[lv][0]:.1f}", f"{m[lv][1]:.1f}", f"{f1(*m[lv]):.1f}"] for lv in levels]
    out.table(
        "TableS9_chr4_pilot",
        "Table S9. Whole-chromosome pilot: unconstrained scan of A. thaliana chromosome 4",
        f"Source: `transgenic/ath_chr4_comparison_FINAL.stats` (GFFCompare v0.12.10). "
        f"TransGenic 400M was run across the entire chromosome without locus constraints, "
        f"predicting {c['query_mrna']:,} transcripts in {c['query_loci']:,} loci "
        f"(~{c['query_mrna'] / c['query_loci']:.1f} per locus) against "
        f"{c['ref_mrna']:,} reference transcripts in {c['ref_loci']:,} loci. Base-level "
        f"sensitivity remains high while transcript-level precision collapses, quantifying "
        f"the isoform over-prediction that motivates the single-locus scope of the tool.",
        ["GFFCompare level", "Sensitivity (%)", "Precision (%)", "F1 (%)"],
        rows,
    )

    # ----------------------------------------------------- Table S10 ----
    s10 = [
        ["GFFCompare", "v0.12.6", "13-species benchmark (Table S2, Figure 5)",
         "`gffcompare -r <reference.gff3> -o <prefix> <prediction.gff3>`"],
        ["GFFCompare", "v0.12.6", "Isoform, alternative-transcript-only, and splice-event analyses (Tables S4a–c)",
         "`gffcompare -r <reference.gtf> -o <prefix> <prediction.gff3>`"],
        ["BUSCO", "v6.0.0", "Functional completeness (Table S3, Figure S1)",
         "`busco -i <proteins.fa> -m prot -l viridiplantae_odb10 -c 2 --offline`"],
        ["PASA", "v2.5.3", "Protein extraction for BUSCO",
         "`gff3_file_to_proteins.pl <annotation.gff3> <genome.fa> prot`"],
        ["ANNEVO", "v2.2", "Whole-genome ab initio benchmark",
         "`annevo predict --genome <genome.fa> --lineage plant`"],
        ["Helixer", "land_plant_v0.3 checkpoint (`land_plant_v0.3_a_0080.h5`)",
         "Whole-genome ab initio benchmark",
         "`Helixer.py --lineage land_plant --subsequence-length 64152 --fasta-path <genome.fa> --gff-output-path <out.gff3>`"],
        ["Tiberius", "v1.1.7", "Whole-genome ab initio benchmark, with and without soft-masked input",
         "`tiberius --genome <genome.fa> --out <out.gtf>` (soft-masked run uses the soft-masked assembly)"],
        ["AUGUSTUS", "v3.5.0", "Posterior-sampling isoform baseline (Table S4)",
         "`augustus --species=arabidopsis --sample=100 --alternatives-from-sampling=true --noInFrameStop=true <locus.fa>`"],
        ["AGAT", "v1.6.1", "Annotation cleaning, sorting, intron addition",
         "`agat_convert_sp_gxf2gxf.pl`; `agat_sp_add_introns.pl`"],
        ["HISAT2 (modified extraction script)", "—",
         "Intron-interval records added during dataset construction",
         "`hisat2_extract_splice_sites_gff3.py <annotation.gff3>`"],
        ["rMATS-style event classifier", "this study (`03_splice_event_detection.py`)",
         "Splice-event classification (Table S4c)",
         "`python 03_splice_event_detection.py --reference <ref.gtf> --prediction <pred.gff3>`"],
        ["TransGenic", "this study (400M and 160M checkpoints)", "De novo and prompted annotation",
         "`python src/run_genome_annotation.py <genome.fa> <loci.gff3> -o <out.gff3> --device cuda`"],
        ["AtRTD3", "atRTD3_TS_21Feb22_transfix", "Long-read reference transcriptome",
         "https://ics.hutton.ac.uk/atRTD/RTD3/"],
    ]
    out.table(
        "TableS10_software_versions",
        "Table S10. Software versions and command lines",
        "Every tool that produced a number reported in the manuscript. All GFFCompare scoring "
        "in the main analyses uses v0.12.6; the chromosome-4 pilot (Table S9) was run earlier "
        "with v0.12.10 and is reported separately. Transcript-level definitions differ between "
        "the two releases — v0.12.10 is the more permissive — while base-level values are "
        "identical, so values from the pilot should not be pooled with the rest.",
        ["Tool", "Version", "Use", "Command line"],
        s10,
    )

    # ----------------------------------------------------- Table S11 ----
    # Every completion-mode figure in the manuscript comes from prompting with TAIR10's own
    # primary transcripts. This table is what happens when the prompt comes from somewhere
    # else, and it exists because the submitted Discussion used to say the question was not
    # addressed. Rows are ordered as an argument rather than by name: the manuscript's own
    # condition, then the same features inside a predicted gene boundary with and without
    # the post-hoc filter, then the two-factor breakdown, then each external annotator's
    # output supplied unchanged.
    rows = []
    for stem, prompt, cds, utr, boundary in PROMPT_TRANSFER_ARMS:
        d = jload(PB / f"{stem}_as_additions.json")
        rows.append([
            prompt, cds, utr, boundary,
            f"{d['added_structures']:,}",
            f"{d['added_matching_TAIR10_alternative_exact_CDS']:,}",
            f"{d['precision_vs_TAIR10_alternatives_pct']:.1f}",
            f"{d['precision_vs_AtRTD3_pct']:.1f}",
            f"{d['recall_of_TAIR10_alternatives_pct']:.1f}",
        ])

    filt = jload(PB / "tair10helixerframeutr_filter.json")
    audit = jload(PB / "tair10_primary_orf_audit.json")
    out.table(
        "TableS11_prompt_transfer",
        "Table S11. Accuracy of added structures as a function of the supplied prompt",
        f"Source: `polishing_benchmark/results/*_as_additions.json`, scored with "
        f"`34_score_as_additions.py`, the script used for the AUGUSTUS comparison in Table S4. "
        f"Additions are scored with each tool's own primary structure removed, against TAIR10 "
        f"alternative transcripts (exact CDS match) and against the AtRTD3 long-read "
        f"transcriptome; all conditions use the same A. thaliana loci. The first row is the "
        f"condition under which every completion-mode figure in the manuscript was obtained. "
        f"The structural filter (`36_filter_additions_structurally.py`) keeps only additions "
        f"encoding a complete open reading frame whose introns all carry canonical GT..AG "
        f"termini, discarding {filt['additions_discarded_pct']:.1f}% of them "
        f"({filt['failed_orf']:,} for the reading frame and {filt['failed_splice']:,} for the "
        f"splice sites); it reads the genome and the predicted coordinates alone and does not "
        f"consult any reference annotation. Both criteria were checked against TAIR10's own "
        f"prompt transcripts before use: {audit['complete_orf_pct']:.1f}% encode a complete "
        f"reading frame and {audit['both_pct']:.1f}% satisfy both criteria "
        f"({audit['transcripts_scored']:,} transcripts), consistent with the reference "
        f"consistency reported in Table S5.",
        ["Prompt", "CDS source", "UTR source", "Gene boundary", "Added", "Matched",
         "Precision vs TAIR10 alt (%)", "Precision vs AtRTD3 (%)", "Recall of TAIR10 alt (%)"],
        rows,
    )


# `summary_report.json`'s total_predicted counts the tmap header as a record, so every
# value is one too high. Count mRNA rows in the prediction file instead — the same source
# 19_verify_manuscript_numbers.py uses, and the one the main text quotes.
PRED_FILE = {
    "denovo400M": "A_thaliana_transgenic400M.gff3",
    "prompted400Mbeam1": "A_thaliana_transgenic400Mprompt_beam1.gff3",
    "augustusSampling": "A_thaliana_augustusSampling.gff3",
}


def predicted_transcripts(result_dir: str) -> int:
    stem = result_dir.split("_vs_")[0]
    fn = PRED_FILE.get(stem)
    if not fn:
        raise KeyError(f"no prediction file mapped for {result_dir}")
    path = CMP / "standardized_results" / fn
    n = 0
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) > 2 and f[2] == "mRNA":
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "supplementary")
    args = ap.parse_args()

    out = Out(args.outdir)
    build(out)
    doc = args.outdir / "supplemental_information.md"
    doc.write_text("\n".join(out.md))
    csvs = sorted(args.outdir.glob("*.csv"))
    print(f"wrote {doc}")
    for c in csvs:
        with c.open() as fh:
            n = sum(1 for _ in fh) - 1
        print(f"  {c.name:<40} {n:>5} data rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
