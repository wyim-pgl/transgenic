#!/usr/bin/env python3
"""VecScreen-style vector/adapter trimming of EST FASTA from a BLAST+ hit table (protocol A21).

Input: the EST FASTA(.gz) and a BLAST tabular file produced with the VecScreen parameters
  blastn -task blastn -db UniVec_Core -reward 1 -penalty -5 -gapopen 3 -gapextend 3 -dust yes \
         -soft_masking true -evalue 700 -searchsp 1750000000000 -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue score qlen"
Categories (NCBI VecScreen, raw score, terminal = match within 25 nt of either end):
  strong   terminal >= 24, internal >= 30
  moderate terminal >= 19, internal >= 25
  weak     terminal >= 16, internal >= 23      (weak hits are reported, not acted on)
Actions: terminal strong/moderate -> trim the matched end (plus anything outboard of it);
         internal strong -> the record is a suspected chimera: split at the match and keep every
         piece >= --min-len (default 100 nt); internal moderate -> flagged only;
         records shorter than --min-len after trimming are dropped.
Outputs: trimmed FASTA(.gz), a per-record report TSV (accession, qlen, action, kept ranges, categories)
and a JSON summary. Pure Python; no external dependencies.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

TERMINAL_NT = 25
THRESH = {"terminal": {"strong": 24, "moderate": 19, "weak": 16}, "internal": {"strong": 30, "moderate": 25, "weak": 23}}


def open_any(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def read_fasta(path: str) -> Iterator[Tuple[str, str, str]]:
    """Yield (id, header_rest, seq)."""
    name, rest, chunks = None, "", []
    with open_any(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, rest, "".join(chunks)
                parts = line[1:].rstrip("\n").split(" ", 1)
                name, rest, chunks = parts[0], (parts[1] if len(parts) > 1 else ""), []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, rest, "".join(chunks)


def classify(qstart: int, qend: int, qlen: int, score: int) -> Tuple[str, str]:
    """Return (position, category) with position in {terminal, internal} and category in {strong, moderate, weak, none}."""
    lo, hi = min(qstart, qend), max(qstart, qend)
    position = "terminal" if lo <= TERMINAL_NT or hi >= qlen - TERMINAL_NT + 1 else "internal"
    t = THRESH[position]
    cat = "strong" if score >= t["strong"] else "moderate" if score >= t["moderate"] else "weak" if score >= t["weak"] else "none"
    return position, cat


def read_hits(path: str) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """qseqid -> list of (qstart, qend, score, qlen) from the 11-column table."""
    hits: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 11:
                raise ValueError(f"expected 11 columns (… score qlen), got {len(c)}: {line[:80]}")
            hits[c[0]].append((int(c[4]), int(c[5]), int(float(c[9])), int(c[10])))
    return hits


def plan_record(qlen: int, rec_hits: Iterable[Tuple[int, int, int, int]], min_len: int) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """Decide what to keep. Returns (action, kept 1-based inclusive ranges, category tags)."""
    tags: List[str] = []
    cut_left, cut_right = 0, qlen + 1           # keep (cut_left, cut_right) exclusive bounds
    internal_strong: List[Tuple[int, int]] = []
    for qs, qe, score, _ in rec_hits:
        lo, hi = min(qs, qe), max(qs, qe)
        pos, cat = classify(qs, qe, qlen, score)
        if cat == "none":
            continue
        tags.append(f"{pos}:{cat}:{lo}-{hi}:{score}")
        if pos == "terminal" and cat in ("strong", "moderate"):
            if lo <= TERMINAL_NT:
                cut_left = max(cut_left, hi)
            if hi >= qlen - TERMINAL_NT + 1:
                cut_right = min(cut_right, lo)
        elif pos == "internal" and cat == "strong":
            internal_strong.append((lo, hi))
    if not tags:
        return "keep", [(1, qlen)], tags
    pieces: List[Tuple[int, int]] = [(cut_left + 1, cut_right - 1)]
    if internal_strong:
        out = []
        for a, b in pieces:
            cur = a
            for lo, hi in sorted(internal_strong):
                if lo > cur:
                    out.append((cur, min(b, lo - 1)))
                cur = max(cur, hi + 1)
            if cur <= b:
                out.append((cur, b))
        pieces = out
    pieces = [(a, b) for a, b in pieces if b - a + 1 >= min_len]
    if not pieces:
        return "drop", [], tags
    if internal_strong:
        return "split", pieces, tags
    return "trim" if pieces != [(1, qlen)] else "keep", pieces, tags


def run(fasta: str, hits_path: str, out_fasta: str, report: str, summary: str, min_len: int = 100) -> Dict:
    hits = read_hits(hits_path)
    counts = defaultdict(int)
    with open_any(out_fasta, "wt") as fo, open(report, "w") as fr:
        fr.write("accession\tqlen\taction\tkept\tcategories\n")
        for acc, rest, seq in read_fasta(fasta):
            qlen = len(seq)
            action, pieces, tags = plan_record(qlen, hits.get(acc, []), min_len) if acc in hits else ("keep", [(1, qlen)], [])
            if action == "keep" and qlen < min_len:
                action, pieces = "drop", []
            counts[action] += 1
            counts["bases_in"] += qlen
            for i, (a, b) in enumerate(pieces, 1):
                name = acc if len(pieces) == 1 else f"{acc}_part{i}"
                fo.write(f">{name} {rest} univec={action}:{a}-{b}/{qlen}\n" if rest else f">{name} univec={action}:{a}-{b}/{qlen}\n")
                fo.write(seq[a - 1:b] + "\n")
                counts["bases_out"] += b - a + 1
                counts["records_out"] += 1
            fr.write(f"{acc}\t{qlen}\t{action}\t{';'.join(f'{a}-{b}' for a, b in pieces)}\t{';'.join(tags)}\n")
    counts["records_in"] = counts["keep"] + counts["trim"] + counts["split"] + counts["drop"]
    with open(summary, "w") as fs:
        json.dump(dict(counts), fs, indent=1)
    return dict(counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--hits", required=True, help="blastn outfmt 6 with 'qseqid sseqid pident length qstart qend sstart send evalue score qlen'")
    ap.add_argument("--out", required=True, help="trimmed FASTA (.gz allowed)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--min-len", type=int, default=100)
    a = ap.parse_args(argv)
    c = run(a.fasta, a.hits, a.out, a.report, a.summary, a.min_len)
    print(json.dumps(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
