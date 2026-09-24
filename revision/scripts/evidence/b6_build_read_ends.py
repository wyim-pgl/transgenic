#!/usr/bin/env python3
"""WP2c step 1 read-end supplement (plan v4.1 §6; ``transgenic.evidence.b6_read_ends``).

    b6_build_read_ends.py --merged evidence_c0.db --species Athaliana --reference <fasta>
        --ont-sheet data/manifests/ONT_PROTOCOL.b6_v1.tsv --pacbio-sheet data/manifests/PACBIO_LIBRARY.b6_v1.tsv
        --out read_ends_b6_v1.Athaliana.db [--alignment DATASET/RUN=PATH ...] [--samtools samtools]

    # validator: re-derive a deterministic 0.1 % sample (blake2b of alignment_id) from the raw alignment + FASTA
    b6_build_read_ends.py --validate read_ends_b6_v1.Athaliana.db --reference <fasta> [--fraction 0.001]

The validator's parser is written independently of the builder's (no shared parsing function) and compares every
column; it always checks at least one row per run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_read_ends as RE  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def _rc(s):
    return "".join(_COMP.get(c, "N") for c in reversed(s.upper()))


def _independent(line, strand, genome, adapter):
    fields = line.rstrip("\n").split("\t")
    pos, cigar, seq = int(fields[3]), fields[5], fields[9]
    ops = re.findall(r"(\d+)([MIDNSHP=X])", cigar)
    ops = [(int(n), o) for n, o in ops if o != "H"]
    lead = ops[0][0] if ops[0][1] == "S" else 0
    trail = ops[-1][0] if len(ops) > 1 and ops[-1][1] == "S" else 0
    span = sum(n for n, o in ops if o in "MDN=X")
    start0, end0 = pos - 1, pos - 1 + span
    chrom = genome[fields[2]]
    has_seq = seq != "*"
    if strand == "+":
        five = seq[:lead] if has_seq else None
        three = (seq[len(seq) - trail:] if trail else "") if has_seq else None
        window = chrom[end0:end0 + 20].upper()
    else:
        five = _rc(seq[len(seq) - trail:] if trail else "") if has_seq else None
        three = _rc(seq[:lead]) if has_seq else None
        window = _rc(chrom[max(0, start0 - 20):start0])
    tail = 0
    for c in (three or ""):
        if c.upper() != "A":
            break
        tail += 1
    runs = [len(m) for m in re.findall(r"A+", window)] or [0]

    def hit(s):
        s = (s or "").upper()
        for L in range(len(adapter), 11, -1):
            w = adapter[-L:]
            if any(sum(a == b for a, b in zip(s[i:i + L], w)) >= 0.8 * L for i in range(len(s) - L + 1)):
                return True
        return False

    return {"start0": start0, "end0": end0, "clip5_seq": five, "clip3_seq": three,
            "polyA_untemplated_len": tail if has_seq else None, "downstream20_A": window.count("A"),
            "downstream_max_consecutive_A": max(runs), "ssp_adapter_hit": hit(five) if has_seq else None,
            "clip5_len": (lead if strand == "+" else trail), "clip3_len": (trail if strand == "+" else lead)}


def validate(db, reference, fraction, samtools="samtools"):
    import duckdb

    sha, prov = RE.open_read_ends(db)
    require_ref = LK.sha256(reference) == prov["inputs"]["reference"]["sha256"]
    if not require_ref:
        raise RE.ReadEndError("reference sha256 != the one the read-end supplement recorded")
    genome = LK.load_build_b5().load_fasta(str(reference))
    con = duckdb.connect(str(db), read_only=True)
    cols = list(RE.COLUMN_NAMES)
    rows = con.execute(f"SELECT {', '.join(cols)} FROM read_end ORDER BY source, dataset, run, record_index").fetchall()
    con.close()
    picked, by_run = {}, {}
    for r in rows:
        d = dict(zip(cols, r))
        h = int.from_bytes(hashlib.blake2b(d["alignment_id"].encode(), digest_size=8).digest(), "big") / 2 ** 64
        key = f"{d['source']}:{d['dataset']}/{d['run']}"
        by_run.setdefault(key, d)
        if h < fraction:
            picked[d["alignment_id"]] = d
    for d in by_run.values():                         # at least one row per run
        picked.setdefault(d["alignment_id"], d)
    ES = RE._es()
    checked = 0
    for key, rec in prov["inputs"]["alignments"].items():
        want = {(d["read_name"], d["record_index"]): d for d in picked.values()
                if f"{d['source']}:{d['dataset']}/{d['run']}" == key}
        if not want:
            continue
        if LK.sha256(rec["path"]) != rec["sha256"]:
            raise RE.ReadEndError(f"{rec['path']}: sha256 changed since the build")
        for i, line in enumerate(ES.sam_lines(rec["path"], samtools)):
            if line.startswith("@"):
                continue
            d = want.get((line.split("\t", 1)[0], i))
            if d is None:
                continue
            if d["transcript_strand"] not in ("+", "-"):
                checked += 1
                continue
            got = _independent(line, d["transcript_strand"], genome, RE.ADAPTERS["ssp"])
            for k, v in got.items():
                if d[k] != v:
                    raise RE.ReadEndError(f"{key} {d['read_name']}: {k} = {d[k]!r} in the DB, re-derived {v!r}")
            checked += 1
    if checked != len(picked):
        raise RE.ReadEndError(f"validator re-derived {checked} of {len(picked)} sampled rows")
    return {"status": "PASS", "checked": checked, "rows": len(rows), "fraction": fraction}


def _pairs(values):
    out = {}
    for v in values or []:
        k, sep, p = v.partition("=")
        if not sep:
            raise RE.ReadEndError(f"--alignment expects DATASET/RUN=PATH, got {v!r}")
        out[k] = p
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate")
    ap.add_argument("--fraction", type=float, default=0.001)
    ap.add_argument("--merged")
    ap.add_argument("--species")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ont-sheet")
    ap.add_argument("--pacbio-sheet")
    ap.add_argument("--alignment", action="append")
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.validate:
            out = validate(a.validate, a.reference, a.fraction, a.samtools)
        else:
            missing = [k for k in ("merged", "species", "ont_sheet", "pacbio_sheet", "out") if not getattr(a, k)]
            if missing:
                raise RE.ReadEndError(f"build mode requires {missing}")
            prov = RE.build(merged=a.merged, species=a.species, reference=a.reference, ont_sheet=a.ont_sheet,
                            pacbio_sheet=a.pacbio_sheet, out=a.out, samtools=a.samtools,
                            alignment_paths=_pairs(a.alignment), argv=argv)
            out = {"output_sha256": prov["output_sha256"], "qc": prov["qc"]}
    except (RE.ReadEndError, ValueError, FileExistsError, OSError) as exc:
        print(f"b6_build_read_ends failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
