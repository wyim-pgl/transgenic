#!/usr/bin/env python3
"""Split a prompt/completion GFF3 pair by the strand of each locus, keeping whole gene blocks.

Used to rescore `helixertairutr` (arm G) on the half of the genome where its UTR donation
actually happened: the arm's donation logic filtered strand-labelled 5'/3' lists by raw
coordinate, so minus-strand loci received essentially no UTR at all.
"""
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision/scripts")
SCRATCH = Path(__file__).resolve().parent


def split(path: Path, want: str, out: Path) -> int:
    """Keep every row belonging to a locus whose top-level gene row is on strand `want`."""
    rows = []
    keep_ids: set[str] = set()
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            rows.append((f, line))
            if f[2] == "gene":
                fid = re.search(r"ID=([^;]+)", f[8])
                if fid and f[6] == want:
                    keep_ids.add(fid.group(1))
    # Transitive closure over Parent chains (gene -> mRNA -> CDS/UTR).
    changed = True
    while changed:
        changed = False
        for f, _ in rows:
            par = re.search(r"Parent=([^;]+)", f[8])
            fid = re.search(r"ID=([^;]+)", f[8])
            if par and fid and par.group(1) in keep_ids and fid.group(1) not in keep_ids:
                keep_ids.add(fid.group(1))
                changed = True
    kept, genes = ["##gff-version 3"], 0
    for f, line in rows:
        fid = re.search(r"ID=([^;]+)", f[8])
        par = re.search(r"Parent=([^;]+)", f[8])
        owner = fid.group(1) if fid else (par.group(1) if par else None)
        if owner in keep_ids or (par and par.group(1) in keep_ids):
            kept.append(line.rstrip("\n"))
            genes += f[2] == "gene"
    out.write_text("\n".join(kept) + "\n")
    return genes


def main() -> int:
    tool, in_gff, out_gff = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    for strand, tag in (("+", "plus"), ("-", "minus")):
        i = SCRATCH / f"{tool}_{tag}_input.gff3"
        o = SCRATCH / f"{tool}_{tag}_output.gff3"
        ni, no = split(in_gff, strand, i), split(out_gff, strand, o)
        print(f"{tool} {tag}: input {ni} loci, output {no} loci", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / "34_score_as_additions.py"),
                        "--input", str(i), "--output", str(o),
                        "--tool", f"{tool}_{tag}",
                        "--json", str(SCRATCH / f"{tool}_{tag}_as_additions.json")],
                       check=True, capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
