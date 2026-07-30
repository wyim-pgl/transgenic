#!/usr/bin/env python3
"""Aggregate revision result JSONs into summary CSVs for the manuscript."""

import json
import glob
import csv
from pathlib import Path

REV = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision/results")

# ---------- 1. Self-consistency ----------
rows = []
for f in sorted(glob.glob(str(REV / "selfconsistency" / "*.json"))):
    d = json.load(open(f))
    name = Path(f).stem
    rows.append({
        "species_variant": name,
        "n_genes": d["n_genes"],
        "n_transcripts": d["n_transcripts"],
        "no_cds": d["n_transcripts_no_cds"],
        "checked": d["n_transcripts_checked"],
        "frame_fail": d["frame_len_mod3_fail"],
        "no_start_atg": d["missing_start_atg"],
        "no_terminal_stop": d["missing_terminal_stop"],
        "internal_stop": d["internal_stop_codons"],
        "fully_consistent": d["fully_consistent_transcripts"],
        "pct_fully_consistent": d["pct_fully_consistent"],
        "duplicate_transcripts": d["duplicate_transcripts"],
        "genes_with_duplicates": d["genes_with_duplicate_transcripts"],
        "mean_isoforms_per_gene": d["mean_isoforms_per_gene_pred"],
    })
with open(REV / "selfconsistency_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"selfconsistency_summary.csv: {len(rows)} rows")

# ---------- 2. Feature stats ----------
rows = []
for f in sorted(glob.glob(str(REV / "feature_tss" / "features_*.json"))):
    d = json.load(open(f))
    sp = Path(f).stem.replace("features_", "")
    rows.append({
        "species": sp,
        "n_genes": d["n_genes"],
        "has_utr": d["has_utr_annotation"],
        "cds_max": d["unique_cds_per_gene"]["max"],
        "cds_p99": d["unique_cds_per_gene"]["p99"],
        "utr5_max": d["unique_utr5_per_gene"]["max"],
        "utr5_p99": d["unique_utr5_per_gene"]["p99"],
        "utr3_max": d["unique_utr3_per_gene"]["max"],
        "utr3_p99": d["unique_utr3_per_gene"]["p99"],
        "tx_max": d["transcripts_per_gene"]["max"],
        "tx_p99": d["transcripts_per_gene"]["p99"],
        "over_150_cds": d["genes_over_150_cds"],
        "over_50_utr5": d["genes_over_50_utr5"],
        "over_50_utr3": d["genes_over_50_utr3"],
        "over_any": d["genes_over_any_limit"],
        "pct_within_limits": d["pct_genes_within_limits"],
    })
with open(REV / "feature_stats_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"feature_stats_summary.csv: {len(rows)} rows")

# ---------- 3. TSS/TES ----------
rows = []
for f in sorted(glob.glob(str(REV / "feature_tss" / "tsstes_*.json"))):
    d = json.load(open(f))
    name = Path(f).stem.replace("tsstes_", "")
    rows.append({
        "species_variant": name,
        "genes_linked": d["n_genes_linked"],
        "genes_unlinked": d["n_genes_unlinked"],
        "transcripts_compared": d["n_transcripts_compared"],
        "TSS_exact_pct": d["TSS"]["exact_pct"],
        "TSS_within50_pct": d["TSS"]["within_50nt_pct"],
        "TSS_within100_pct": d["TSS"]["within_100nt_pct"],
        "TSS_median_delta": d["TSS"]["median_delta"],
        "TES_exact_pct": d["TES"]["exact_pct"],
        "TES_within50_pct": d["TES"]["within_50nt_pct"],
        "TES_within100_pct": d["TES"]["within_100nt_pct"],
        "TES_median_delta": d["TES"]["median_delta"],
    })
with open(REV / "tss_tes_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"tss_tes_summary.csv: {len(rows)} rows")

# ---------- 4. AS stats of references ----------
rows = []
for f in sorted(glob.glob(str(REV / "feature_tss" / "asstats_*.json"))):
    d = json.load(open(f))
    sp = Path(f).stem.replace("asstats_", "")
    rows.append({
        "species": sp,
        "n_genes": d["n_genes"],
        "n_transcripts": d["n_transcripts"],
        "multi_transcript_genes": d["multi_transcript_genes"],
        "pct_multi_transcript": d["pct_multi_transcript"],
        "mean_transcripts_per_gene": d["mean_transcripts_per_gene"],
        "max_transcripts_per_gene": d["max_transcripts_per_gene"],
    })
with open(REV / "asstats_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"asstats_summary.csv: {len(rows)} rows")
