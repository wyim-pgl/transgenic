#!/usr/bin/env python3
"""Repair the fig3_infer.py flip_rc offset bug in every species' Figure 3 predictions.

flip_rc subtracts the region start from -rc rows (see diagnose.py). region_start is
deterministic from the reference gene span (preprocess.py:305-329), so the repair is
exact and needs nothing from the training database.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/data/gpfs/assoc/pgl/data/Transgenic")
REGEN = ROOT / "transgenic/revision/results/fig3_regen"
OUT = ROOT / "transgenic/revision/results/fig3_gap_diagnosis/preds_repaired"

STATIC = 6144
ID_RE = re.compile(r"(?:^|;)ID=([^;]+)")
GM_RE = re.compile(r"(?:^|;)GM=([^;\s]+)")

SPECIES = ["A_thaliana", "B_distachyon", "G_max", "O_sativa", "P_patens",
           "P_trichocarpa", "S_bicolor", "S_italica", "V_vinifera", "Z_mays"]


def region_start_of(start: int, fin: int) -> int:
    gene_length = fin - start + 1
    if gene_length <= STATIC:
        additional = STATIC - (gene_length % STATIC)
    else:
        additional = ((gene_length // STATIC) + 1) * STATIC - gene_length
    five = additional // 2 + (1 if additional % 2 else 0)
    if (start - five - 1) < 0:
        five = start - 1
    return start - five - 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'species':16s} {'rows':>7s} {'rc_rows':>8s} {'rc_fixed':>9s} {'unmapped':>9s}")
    for sp in SPECIES:
        ref = REGEN / "refs" / f"{sp}.testset.gff3"
        pred = REGEN / "preds" / f"{sp}_test400M.gff3"
        if not ref.exists() or not pred.exists():
            print(f"{sp:16s} SKIP (missing input)")
            continue
        genes = {}
        with ref.open() as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.split("\t")
                if len(f) >= 9 and f[2] == "gene":
                    m = ID_RE.search(f[8])
                    if m:
                        genes[m.group(1)] = (int(f[3]), int(f[4]))
        rows, rc_rows, rc_fixed, unmapped = set(), set(), 0, 0
        out_lines = []
        with pred.open() as fh:
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
                rows.add(gm)
                if not gm.endswith("-rc"):
                    out_lines.append(line)
                    continue
                rc_rows.add(gm)
                info = genes.get(gm[:-3])
                if info is None:
                    unmapped += 1
                    out_lines.append(line)
                    continue
                rs = region_start_of(*info)
                f[3], f[4] = str(int(f[3]) + rs), str(int(f[4]) + rs)
                out_lines.append("\t".join(f) + "\n")
                rc_fixed += 1
        (OUT / f"{sp}_test400M.repaired.gff3").write_text("".join(out_lines))
        print(f"{sp:16s} {len(rows):7d} {len(rc_rows):8d} {rc_fixed:9d} {unmapped:9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
