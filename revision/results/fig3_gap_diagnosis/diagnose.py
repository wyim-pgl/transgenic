#!/usr/bin/env python3
"""Split fig3_regen A. thaliana predictions by row orientation (forward vs -rc) and
by train-exposure, and repair the -rc coordinate bug in fig3_infer.py:flip_rc.

flip_rc(lines, fin) is applied to lines whose coordinates gffString2GFF3 has ALREADY
offset by region_start, so the flipped coordinate is short by exactly region_start:

    emitted   e = region_start + 1 + p        (p = 0-based offset in the RC sequence)
    flip_rc  ns = fin - e + 1 = L - p         (L = region_end - region_start)
    correct      region_end - p = ns + region_start

region_start is deterministic from the reference annotation (preprocess.py:305-329),
so the repair needs nothing from the training database.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/gpfs/assoc/pgl/data/Transgenic")
REGEN = ROOT / "transgenic/revision/results/fig3_regen"
OUT = ROOT / "transgenic/revision/results/fig3_gap_diagnosis"
SPLIT_TSV = ROOT / "transgenic/revision/results/heldout_additions/at_gene_split_20260811.tsv"
PRED = REGEN / "preds/A_thaliana_test400M.gff3"
REF = REGEN / "refs/A_thaliana.testset.gff3"

STATIC = 6144
ID_RE = re.compile(r"(?:^|;)ID=([^;]+)")
PARENT_RE = re.compile(r"(?:^|;)Parent=([^;]+)")
GM_RE = re.compile(r"(?:^|;)GM=([^;\s]+)")


def region_start_of(start: int, fin: int) -> int:
    """Replicate preprocess.py:305-329 exactly (0-based region start)."""
    gene_length = fin - start + 1
    if gene_length <= STATIC:
        additional = STATIC - (gene_length % STATIC)
    else:
        additional = ((gene_length // STATIC) + 1) * STATIC - gene_length
    five = additional // 2 + (1 if additional % 2 else 0)
    if (start - five - 1) < 0:
        five = start - 1
    return start - five - 1


def load_ref_genes() -> dict[str, tuple[str, int, int]]:
    genes = {}
    with REF.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = ID_RE.search(f[8])
            if m:
                genes[m.group(1)] = (f[0], int(f[3]), int(f[4]))
    return genes


def write_ref_subset(keep_genes: set[str], out_path: Path) -> int:
    keep_tx: set[str] = set()
    with REF.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] not in ("mRNA", "transcript"):
                continue
            mi, mp = ID_RE.search(f[8]), PARENT_RE.search(f[8])
            if mi and mp and mp.group(1) in keep_genes:
                keep_tx.add(mi.group(1))
    n_gene = 0
    with REF.open() as fh, out_path.open("w") as out:
        out.write("##gff-version 3\n")
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9:
                continue
            if f[2] == "gene":
                m = ID_RE.search(f[8])
                if m and m.group(1) in keep_genes:
                    out.write(line)
                    n_gene += 1
            elif f[2] in ("mRNA", "transcript"):
                m = ID_RE.search(f[8])
                if m and m.group(1) in keep_tx:
                    out.write(line)
            else:
                m = PARENT_RE.search(f[8])
                if m and any(p in keep_tx or p in keep_genes for p in m.group(1).split(",")):
                    out.write(line)
    return n_gene


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_genes = load_ref_genes()
    print(f"reference genes: {len(ref_genes)}")

    # ---- categorise every predicted row ------------------------------------
    fwd_lines: list[str] = []
    rc_lines: list[str] = []
    rc_fixed: list[str] = []
    fwd_genes: set[str] = set()
    rc_genes: set[str] = set()
    unmapped_rc = 0
    offset_votes: dict[int, int] = defaultdict(int)

    # reference CDS boundary set, for validating the repair offset
    ref_cds: set[tuple[str, int, int]] = set()
    with REF.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) >= 9 and f[2] == "CDS":
                ref_cds.add((f[0], int(f[3]), int(f[4])))

    with PRED.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            m = GM_RE.search(f[8])
            if not m:
                continue
            gm = m.group(1)
            if gm.endswith("-rc"):
                gene = gm[:-3]
                rc_genes.add(gene)
                rc_lines.append(line)
                info = ref_genes.get(gene)
                if info is None:
                    unmapped_rc += 1
                    continue
                _, gs, ge = info
                rs = region_start_of(gs, ge)
                g = list(f)
                g[3], g[4] = str(int(f[3]) + rs), str(int(f[4]) + rs)
                rc_fixed.append("\t".join(g) + "\n")
                if f[2] == "CDS":
                    for d in (-1, 0, 1):
                        if (f[0], int(f[3]) + rs + d, int(f[4]) + rs + d) in ref_cds:
                            offset_votes[d] += 1
            else:
                fwd_genes.add(gm)
                fwd_lines.append(line)

    print(f"forward lines {len(fwd_lines)}  rc lines {len(rc_lines)}  "
          f"rc repaired {len(rc_fixed)}  rc unmapped {unmapped_rc}")
    print(f"forward genes {len(fwd_genes)}  rc genes {len(rc_genes)}  "
          f"rc-only {len(rc_genes - fwd_genes)}  both {len(rc_genes & fwd_genes)}")
    print(f"repair offset validation (exact CDS matches by extra shift): "
          f"{dict(sorted(offset_votes.items()))}")

    # ---- exposure categories ------------------------------------------------
    cat = {}
    for line in SPLIT_TSV.read_text().splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 2:
            cat[p[0]] = p[1]
    heldout = {g for g in ref_genes if cat.get(g) == "test"}
    exposed = {g for g in ref_genes if cat.get(g) in ("train", "validation")}
    print(f"pure held-out genes {len(heldout)}  train/val-exposed genes {len(exposed)}  "
          f"uncategorised {len(ref_genes) - len(heldout) - len(exposed)}")

    # ---- write prediction subsets ------------------------------------------
    (OUT / "pred_forward.gff3").write_text("".join(fwd_lines))
    (OUT / "pred_rc_raw.gff3").write_text("".join(rc_lines))
    (OUT / "pred_rc_repaired.gff3").write_text("".join(rc_fixed))
    (OUT / "pred_all_repaired.gff3").write_text("".join(fwd_lines) + "".join(rc_fixed))

    fwd_heldout = [l for l in fwd_lines if (GM_RE.search(l).group(1) in heldout)]
    fwd_exposed = [l for l in fwd_lines if (GM_RE.search(l).group(1) in exposed)]
    (OUT / "pred_forward_heldout.gff3").write_text("".join(fwd_heldout))
    (OUT / "pred_forward_exposed.gff3").write_text("".join(fwd_exposed))

    # ---- write reference subsets -------------------------------------------
    subsets = {
        "ref_forward": fwd_genes & set(ref_genes),
        "ref_rc": rc_genes & set(ref_genes),
        "ref_rc_only": (rc_genes - fwd_genes) & set(ref_genes),
        "ref_forward_heldout": fwd_genes & heldout,
        "ref_forward_exposed": fwd_genes & exposed,
    }
    for name, genes in subsets.items():
        n = write_ref_subset(genes, OUT / f"{name}.gff3")
        print(f"{name}: {n} genes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
