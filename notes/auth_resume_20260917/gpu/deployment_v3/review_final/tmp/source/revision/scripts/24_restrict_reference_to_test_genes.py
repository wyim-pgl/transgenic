#!/usr/bin/env python3
"""Restrict each reference annotation to the genes in the Figure 3 evaluation set.

Figure 3A scores de novo predictions over the held-out test split, which is 15% of
the genes for the nine training species (all genes for the withheld Z. mays). Scoring
those predictions against the complete genome annotation makes sensitivity
meaningless — it counts the 85% of genes that were never predicted as missed. The
first run of `fig3_evaluate.sh` did exactly that and gave A. thaliana base
sensitivity 14.0% against precision 92.5%.

This script writes a per-species reference containing only the evaluated genes and
their descendant features, so sensitivity and precision are measured over the same
gene set.

Inputs:
  fig3_regen/fig3_test_genes.tsv   species<TAB>geneModel, from the training database
  genomes/<reference>.gff3         the full annotations

Output:
  fig3_regen/refs/<species>.testset.gff3

Usage:
    python 24_restrict_reference_to_test_genes.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GENOMES = ROOT / "genomes"
REGEN = ROOT / "transgenic" / "revision" / "results" / "fig3_regen"
OUTDIR = REGEN / "refs"

REFS = {
    "A_thaliana": "Athaliana_167_TAIR10.gene.clean.gff3",
    "G_max": "Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
    "P_patens": "Ppatens_318_v3.3.gene_exons.clean.gff3",
    "P_trichocarpa": "Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
    "S_bicolor": "Sbicolor_730_v5.1.gene_exons.clean.gff3",
    "B_distachyon": "Bdistachyon_314_v3.1.gene_exons.clean.gff3",
    "S_italica": "Sitalica_312_v2.2.gene_exons.clean.gff3",
    "V_vinifera": "Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3",
    "O_sativa": "Osativa_323_v7.0.gene_exons.exon.gff3",
    "Z_mays": "Zmays_493_RefGen_V4.gene_exons.exon.gff3",
}

ID_RE = re.compile(r"(?:^|;)ID=([^;]+)")
PARENT_RE = re.compile(r"(?:^|;)Parent=([^;]+)")


def load_test_genes() -> dict[str, set[str]]:
    path = REGEN / "fig3_test_genes.tsv"
    if not path.exists():
        sys.exit(f"missing {path}; generate it from the training database first")
    genes: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sp, gm = line.split("\t")
        genes[sp].add(gm)
    return genes


def restrict(ref_path: Path, keep_genes: set[str], out_path: Path) -> tuple[int, int, int]:
    """Two passes: resolve which transcripts belong to the kept genes, then emit."""
    keep_tx: set[str] = set()
    with ref_path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] not in ("mRNA", "transcript"):
                continue
            mi, mp = ID_RE.search(f[8]), PARENT_RE.search(f[8])
            if mi and mp and mp.group(1) in keep_genes:
                keep_tx.add(mi.group(1))

    n_gene = n_tx = n_feat = 0
    with ref_path.open() as fh, out_path.open("w") as out:
        out.write("##gff-version 3\n")
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9:
                continue
            feat, attrs = f[2], f[8]
            if feat == "gene":
                m = ID_RE.search(attrs)
                if m and m.group(1) in keep_genes:
                    out.write(line)
                    n_gene += 1
            elif feat in ("mRNA", "transcript"):
                m = ID_RE.search(attrs)
                if m and m.group(1) in keep_tx:
                    out.write(line)
                    n_tx += 1
            else:
                m = PARENT_RE.search(attrs)
                if not m:
                    continue
                # a feature may list several parents
                if any(p in keep_tx or p in keep_genes for p in m.group(1).split(",")):
                    out.write(line)
                    n_feat += 1
    return n_gene, n_tx, n_feat


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    test_genes = load_test_genes()

    print(f"{'species':16s} {'wanted':>8s} {'genes':>8s} {'tx':>9s} {'features':>10s}  status")
    problems = 0
    for sp, refname in REFS.items():
        want = test_genes.get(sp, set())
        if not want:
            print(f"{sp:16s} {'-':>8s}  no test genes listed")
            continue
        ref = GENOMES / refname
        out = OUTDIR / f"{sp}.testset.gff3"
        n_gene, n_tx, n_feat = restrict(ref, want, out)
        # Every evaluated gene should be found; a shortfall means an ID mismatch
        # between the training database and the annotation, which would silently
        # depress sensitivity exactly as the unrestricted reference did.
        missing = len(want) - n_gene
        status = "ok" if missing == 0 else f"MISSING {missing} genes"
        if missing:
            problems += 1
        print(f"{sp:16s} {len(want):8d} {n_gene:8d} {n_tx:9d} {n_feat:10d}  {status}")

    print()
    print(f"wrote restricted references to {OUTDIR}")
    if problems:
        print(f"WARNING: {problems} species have genes that could not be matched by ID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
