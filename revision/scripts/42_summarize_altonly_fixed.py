#!/usr/bin/env python3
"""Emit results/altonly_fixed/altonly_atrtd3_comparison.json: the alt-only AtRTD3
reference audit plus old-vs-new GFFCompare metrics for the three affected runs."""

import json
import re
from pathlib import Path

RES = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision/results")
LEVELS = ("Base", "Exon", "Intron", "Intron chain", "Transcript", "Locus")


def parse_stats(path):
    txt = Path(path).read_text()
    out = {}
    m = re.search(r"# Reference mRNAs\s*:\s*(\d+) in\s+(\d+) loci", txt)
    out["reference_mrnas"], out["reference_loci"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"# +Query mRNAs\s*:\s*(\d+) in\s+(\d+) loci", txt)
    out["query_mrnas"], out["query_loci"] = int(m.group(1)), int(m.group(2))
    for lvl in LEVELS:
        m = re.search(rf"^\s*{re.escape(lvl)} level:\s*([\d.]+)\s*\|\s*([\d.]+)", txt, re.M)
        sn, pr = float(m.group(1)), float(m.group(2))
        out[lvl.lower().replace(" ", "_")] = {
            "sn": sn, "pr": pr, "f1": round(0.0 if sn + pr == 0 else 2 * sn * pr / (sn + pr), 1)}
    out["matching_transcripts"] = int(re.search(r"Matching transcripts:\s*(\d+)", txt).group(1))
    return out


RUNS = {
    "transgenic400M_prompted_beam1": (
        "altonly/A_thaliana_transgenic400Mprompt_beam1_vs_AtRTD3.stats",
        "altonly_fixed/A_thaliana_transgenic400Mprompt_beam1_vs_AtRTD3altfix.stats"),
    "transgenic400M_denovo": (
        "altonly/A_thaliana_transgenic400M_vs_AtRTD3.stats",
        "altonly_fixed/A_thaliana_transgenic400M_vs_AtRTD3altfix.stats"),
    "augustus_sampling": (
        "altonly/A_thaliana_augustusSampling_vs_AtRTD3.stats",
        "altonly_fixed/A_thaliana_augustusSampling_vs_AtRTD3altfix.stats"),
}

doc = {
    "audit": {
        "atrtd3_full": {"genes": 40929, "transcripts": 169499},
        "old_reference": {
            "path": "revision/data/AtRTD3/AtRTD3.altonly.gtf",
            "rule": "drop transcripts whose ID matches the TAIR10 first-transcript set",
            "genes": 32465, "transcripts": 142635, "removed": 26864,
            "genes_retaining_their_primary": 14065,
            "genes_whose_first_transcript_survives": 16519,
            "single_transcript_genes_retained_intact": 12071,
            "genuine_alternatives_wrongly_removed": 2454,
        },
        "new_reference": {
            "path": "revision/results/altonly_fixed/AtRTD3.altonly_firsttx.gtf",
            "rule": "drop the first transcript of each gene (file order), as Methods states",
            "genes": 20394, "transcripts": 128570, "removed": 40929,
            "genes_retaining_their_primary": 0,
            "removed_ids_ending_in_dot1": 40901,
        },
        "tair10_altonly_control": {
            "path": "revision/data/TAIR10/TAIR10.altonly.gtf",
            "genes": 5804, "transcripts": 7970, "removed_per_gene": 1,
            "genes_retaining_their_primary": 0, "verdict": "correct, unchanged",
        },
    },
    "gffcompare": {
        "binary": "/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare",
        "version": "v0.12.6",
        "command": "gffcompare -r <AtRTD3.altonly_firsttx.gtf> -o <prefix> <prediction.gff3>",
    },
    "runs": {},
}

for key, (old, new) in RUNS.items():
    doc["runs"][key] = {"old": parse_stats(RES / old), "new": parse_stats(RES / new),
                        "old_stats": old, "new_stats": new}

out = RES / "altonly_fixed/altonly_atrtd3_comparison.json"
out.write_text(json.dumps(doc, indent=1) + "\n")
print(f"wrote {out}")
