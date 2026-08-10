#!/usr/bin/env python3
"""Re-score the reference-prompted 400M predictions on one record per locus.

Why this exists. `gffcompare_summary.csv` — the source of Table S2 and Figure 5 — scored
`*_transgenic400Mprompt.gff3`, which holds every locus twice as separate gene records:
54,826 genes for 27,413 A. thaliana loci. Each extra record is a false positive against the
reference, so the 400M completion row carries roughly half the precision it should, while
the 160M row (27,411 genes, one record per locus) does not. The two were compared as if
scored alike.

CORRECTED 2026-08-10 — what the second record is. This docstring said the file "exports
both beam hypotheses per locus", and the manuscript Methods said the same. It does not.
Generation requests one sequence per locus (`examples/prompt_mode.py` passes
`num_return_sequences=1`), the per-locus transcript count is even at every one of the
27,413 loci, and all 25,252 loci holding exactly two transcripts hold two structurally
identical copies — CDS, exon and both UTRs. Two beam hypotheses would differ from each
other; an exactly doubled, wholly identical, all-even set is the same prediction written
twice, which is what an append to an existing prediction database produces on a re-run.

Nothing below changes as a result. A duplicate scores against GFFCompare exactly as a
rival beam would, so the filter and every number it produces stand; only the stated reason
was wrong. The direction also stands: the error understates the 400M model. A. thaliana
transcript precision is 46.8% in Table S2 and 74.4% in the Results, which score the same
predictions with and without the duplicate.

What this does: filters each species' prompted GFF3 to the first gene record per GM value
(the same rule as `13_beam1_filter.py`), runs GFFCompare v0.12.10 against the same
reference annotation used for the rest of the benchmark, and writes a replacement row set.
It does not touch any other tool or mode.

Usage:
    python 27_rescore_prompted_topbeam.py [--out rescored_topbeam.csv] [--keep-tmp]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
STD = CMP / "standardized_results"
REF = CMP / "reference_annotations"
GFFCOMPARE = Path("/data/gpfs/assoc/pgl/bin/conda/conda_envs/AnnotationSplitter/bin/gffcompare")

# Species -> reference annotation, matching the existing benchmark.
REFERENCES = {
    "A_thaliana": "Athaliana_167_TAIR10.gene.clean.gff3",
    "B_distachyon": "Bdistachyon_314_v3.1.gene_exons.clean.gff3",
    "B_rapa": "BrapaO_302V_711_v1.1.gene.gff3",
    "G_max": "Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
    "L_sativa": "Lsativa_467_v5.gene_exons.gff3",
    "O_sativa": "Osativa_323_v7.0.gene_exons.exon.gff3",
    "P_patens": "Ppatens_318_v3.3.gene_exons.clean.gff3",
    "P_trichocarpa": "Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
    "S_bicolor": "Sbicolor_730_v5.1.gene_exons.clean.gff3",
    "S_italica": "Sitalica_312_v2.2.gene_exons.clean.gff3",
    "S_lycopersicum": "Slycopersicum_796_ITAG5.0.gene.gff3",
    "V_vinifera": "Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3",
    "Z_mays": "Zmays_493_RefGen_V4.gene_exons.exon.gff3",
}

TOOL = "transgenic400Mprompt"


def top_beam(src: Path, dst: Path) -> tuple[int, int]:
    """Keep the first gene record per GM value, with all of its children."""
    gm_re = re.compile(r"GM=([^;\s]+)")
    seen: set[str] = set()
    keep_ids: set[str] = set()
    kept = total = 0
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            m = gm_re.search(f[8])
            if not m:
                continue
            if f[2] == "gene":
                total += 1
                gm = m.group(1)
                if gm in seen:
                    keep = False
                else:
                    seen.add(gm)
                    keep = True
                    kept += 1
                gid = re.search(r"ID=([^;]+)", f[8])
                if keep and gid:
                    keep_ids.add(gid.group(1))
                if keep:
                    out.write(line)
                continue
            # child rows: keep if their gene (or their mRNA's gene) was kept
            parent = re.search(r"Parent=([^;]+)", f[8])
            ident = re.search(r"ID=([^;]+)", f[8])
            if not parent:
                continue
            root = parent.group(1).split(".t")[0]
            if root in keep_ids or parent.group(1) in keep_ids:
                if ident:
                    keep_ids.add(ident.group(1))
                out.write(line)
    return kept, total


def parse_stats(path: Path) -> dict[str, float]:
    """Pull the four levels Table S2 reports out of a GFFCompare .stats file."""
    out: dict[str, float] = {}
    pat = re.compile(r"^\s*([A-Za-z ]+?) level:\s*([\d.-]+)\s*\|\s*([\d.-]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1).strip() + "_Sn"] = float(m.group(2))
            out[m.group(1).strip() + "_Pr"] = float(m.group(3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CMP / "rescored_topbeam.csv")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    assert GFFCOMPARE.exists(), f"gffcompare not found at {GFFCOMPARE}"
    tmp = Path(tempfile.mkdtemp(prefix="topbeam_", dir=str(CMP)))
    rows = []
    for species, ref_name in REFERENCES.items():
        src = STD / f"{species}_{TOOL}.gff3"
        ref = REF / ref_name
        if not src.exists() or not ref.exists():
            print(f"  skip {species}: missing input", file=sys.stderr)
            continue
        filtered = tmp / f"{species}_topbeam.gff3"
        kept, total = top_beam(src, filtered)
        prefix = tmp / f"{species}_cmp"
        subprocess.run(
            [str(GFFCOMPARE), "-r", str(ref), "-o", str(prefix), str(filtered)],
            check=True, capture_output=True,
        )
        st = parse_stats(prefix.with_suffix(".stats"))
        rows.append({
            "Species": species, "Tool": TOOL,
            "Base_Sensitivity": st.get("Base_Sn"), "Base_Precision": st.get("Base_Pr"),
            "Exon_Sensitivity": st.get("Exon_Sn"), "Exon_Precision": st.get("Exon_Pr"),
            "Transcript_Sensitivity": st.get("Transcript_Sn"),
            "Transcript_Precision": st.get("Transcript_Pr"),
            "genes_kept": kept, "genes_in_source": total,
        })
        print(f"  {species:<16} genes {total:>6,} -> {kept:>6,}   "
              f"base F1 inputs {st.get('Base_Sn')}/{st.get('Base_Pr')}   "
              f"tx {st.get('Transcript_Sn')}/{st.get('Transcript_Pr')}", file=sys.stderr)

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written: {args.out} ({len(rows)} species)", file=sys.stderr)
    if not args.keep_tmp:
        for p in sorted(tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
