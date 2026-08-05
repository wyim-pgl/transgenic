#!/usr/bin/env python3
"""Stage external A. thaliana annotations into the polishing benchmark.

GeMoMa and BRAKER3 write sequence names as `Ath_Chr1`; EGAPx writes `Chr1`; the TAIR10
genome and reference use `Chr1`. GFFCompare scores every level at 0.0 when names do not
meet, without saying so, so normalisation happens here and overlap is asserted rather
than assumed.

Usage:
    python 30_stage_external_annotations.py [--out polishing_benchmark/inputs]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SYLVAN = Path("/data/gpfs/assoc/pgl/data/Sylvan")

SOURCES = {
    "gemoma": SYLVAN / "gemoma_output/Ath/final_annotation.gff",
    "braker3": SYLVAN / "braker3_output/Ath/braker.gff3",
    "egapx": SYLVAN / "egapx_results/Arabidopsis/Arabidopsis_out/complete.genomic.gff",
}

TAIR_SEQIDS = {"Chr1", "Chr2", "Chr3", "Chr4", "Chr5", "ChrC", "ChrM"}


def normalise_seqid(name: str) -> str:
    """Map a source sequence name onto TAIR10's."""
    n = re.sub(r"^[A-Za-z]{3}_", "", name)          # Ath_Chr1 -> Chr1
    if n in TAIR_SEQIDS:
        return n
    if re.fullmatch(r"\d+", n):                      # 1 -> Chr1
        return f"Chr{n}"
    if n.lower() in {"chrc", "c", "pltd", "chloroplast"}:
        return "ChrC"
    if n.lower() in {"chrm", "m", "mito", "mitochondrion"}:
        return "ChrM"
    return n


def stage(tool: str, src: Path, dst: Path) -> dict:
    """Normalise `src` into `dst`, dropping anything not on a TAIR10 sequence."""
    genes = transcripts = 0
    seqids: set[str] = set()
    dropped: set[str] = set()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seq = normalise_seqid(f[0])
            if seq not in TAIR_SEQIDS:
                dropped.add(f[0])
                continue
            f[0] = seq
            seqids.add(seq)
            if f[2] == "gene":
                genes += 1
            elif f[2] in ("mRNA", "transcript"):
                transcripts += 1
            out.write("\t".join(f) + "\n")
    return {
        "tool": tool,
        "source": str(src),
        "genes": genes,
        "transcripts": transcripts,
        "seqids": sorted(seqids),
        "dropped_seqids": sorted(dropped),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "polishing_benchmark" / "inputs")
    args = ap.parse_args()

    meta = []
    for tool, src in SOURCES.items():
        if not src.exists():
            print(f"  MISSING {tool}: {src}", file=sys.stderr)
            return 1
        dst = args.out / f"{tool}_Athaliana.gff3"
        m = stage(tool, src, dst)
        assert m["genes"] > 20000, f"{tool}: only {m['genes']} genes staged"
        assert set(m["seqids"]) & TAIR_SEQIDS, f"{tool}: no TAIR10 sequence names after normalisation"
        meta.append(m)
        print(f"  {tool:<8} {m['genes']:>6,} genes  {m['transcripts']:>6,} transcripts  "
              f"seqids={','.join(m['seqids'][:3])}…  dropped={len(m['dropped_seqids'])}",
              file=sys.stderr)

    (args.out / "provenance.json").write_text(json.dumps(meta, indent=1))
    print(f"written: {args.out}/provenance.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
