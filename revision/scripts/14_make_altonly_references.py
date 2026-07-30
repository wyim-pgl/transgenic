#!/usr/bin/env python3
"""Create 'alternative-transcript-only' references (primary transcript removed),
mirroring the manuscript's isoform evaluation: the first transcript of each
gene model is removed before comparison.

- TAIR10.gtf: remove the FIRST transcript feature block per gene (file order).
- AtRTD3.gtf: remove transcripts whose IDs match the TAIR10 primary set
  (AtRTD3 reuses TAIR IDs for known isoforms).
Also emits the list of removed primary transcript IDs.
"""

import re
import sys
from pathlib import Path

REV = Path("/data/gpfs/assoc/pgl/data/Transgenic/transgenic/revision")


def tair10_primary_ids(tair_gtf):
    """First transcript ID per gene (file order) from TAIR10 GTF."""
    seen = set()
    primary = []
    for line in open(tair_gtf):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "transcript":
            continue
        m = re.search(r'transcript_id "([^"]+)"', f[8])
        g = re.search(r'gene_id "([^"]+)"', f[8])
        if m and g and g.group(1) not in seen:
            seen.add(g.group(1))
            primary.append(m.group(1))
    return primary


def filter_tair10(tair_gtf, out_gtf, primary):
    """Drop transcript feature blocks (transcript line + children) for primary IDs."""
    pset = set(primary)
    skip_tx = None
    kept = dropped = 0
    with open(out_gtf, "w") as out:
        for line in open(tair_gtf):
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                out.write(line)
                continue
            mt = re.search(r'transcript_id "([^"]+)"', f[8])
            tx = mt.group(1) if mt else None
            if f[2] == "transcript":
                skip_tx = tx if tx in pset else None
                if skip_tx:
                    dropped += 1
                    continue
                kept += 1
            elif skip_tx and tx == skip_tx:
                continue
            out.write(line)
    return kept, dropped


def filter_atrtd3(atrtd_gtf, out_gtf, primary):
    """Drop all features whose transcript_id is in the primary set."""
    # AtRTD3 transcript IDs look like AT1G01010.1; TAIR10 primary IDs look
    # like AT1G01010.1.TAIR10 -> match on the shared prefix.
    pset = set(p.replace(".TAIR10", "") for p in primary)
    kept = dropped = 0
    with open(out_gtf, "w") as out:
        for line in open(atrtd_gtf):
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                out.write(line)
                continue
            mt = re.search(r'transcript_id "([^"]+)"', f[8])
            if mt and mt.group(1) in pset:
                dropped += 1
                continue
            kept += 1
            out.write(line)
    return kept, dropped


def main():
    tair = REV / "data/TAIR10/TAIR10.gtf"
    atrtd = REV / "data/AtRTD3/AtRTD3.gtf"
    primary = tair10_primary_ids(tair)
    print(f"primary transcripts: {len(primary)}")
    (REV / "data/TAIR10/primary_transcript_ids.txt").write_text(
        "\n".join(primary) + "\n")
    k, d = filter_tair10(tair, REV / "data/TAIR10/TAIR10.altonly.gtf", primary)
    print(f"TAIR10 altonly: kept {k} transcripts, dropped {d}")
    k, d = filter_atrtd3(atrtd, REV / "data/AtRTD3/AtRTD3.altonly.gtf", primary)
    print(f"AtRTD3 altonly: kept {k} feature lines, dropped {d}")


if __name__ == "__main__":
    main()
