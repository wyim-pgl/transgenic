"""WP2c step 1 (plan v4.1 §6): read-end supplement -- the read-end measurements C0 does not store.

One row per C0 ACCEPTED read alignment (EST / ONT / PacBio), joined uniquely on (dataset, run, read_name,
record_index); ``record_index`` is the 0-based line of ``samtools view -h`` output exactly as ``b6_build_c0`` counts
it.  Inputs: the merged C0 snapshot (sidecar-verified; its ``c0_provenance`` names every frozen primary alignment and
its sha256, which the file must equal), the reference FASTA (= the C0 recorded reference) and the
``ONT_PROTOCOL`` / ``PACBIO_LIBRARY`` b6 sheets (sha recorded; every long-read run must have a row).

Columns (transcript orientation; NULL when the SAM record has no SEQ or the transcript strand is unknown):
``clip5_len``, ``clip3_len`` / ``clip5_seq``, ``clip3_seq`` (soft clips at the transcript 5' / 3' ends);
``tss0`` / ``tes0`` (genomic 0-based transcript 5' / 3' end base); ``polyA_untemplated_len`` (the A-run that starts at
the first 3'-clipped base, ``POLYA_RULE``); ``downstream20_A`` and ``downstream_max_consecutive_A`` (the 20 genomic nt
3' of the transcript end); ``primer5_hit`` (only where the protocol sheet names a 5' primer: a cap-trapped/TeloPrime
library); ``ssp_adapter_hit`` (the ONT strand-switch primer in the 5' clip, >= 80 % identity over >= 12 nt);
``adapter_5c_enabled`` (the sheet row carries ``adapter_5c_approved = true``: author approval; absent -> false).
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transgenic.evidence import leakage as LK

VERSION = "read_ends_b6_v1"
POLYA_RULE = "leading_A_run_at_first_transcript_3prime_softclip_base_v1"
ADAPTERS = {"ssp": "TTTCTGTTGGTGCTGATATTGC", "vnp": "ACTTGCCTGTCGCTCTATCTTC", "tso": "AAGCAGTGGTATCAACGCAGAG"}
ADAPTER_MIN_LEN = 12
ADAPTER_MIN_IDENTITY = 0.80
DOWNSTREAM = 20
READ_SOURCES = ("est", "ont", "pacbio")
_CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
COLUMNS = (
    ("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("dataset", "VARCHAR NOT NULL"),
    ("run", "VARCHAR NOT NULL"), ("alignment_id", "VARCHAR NOT NULL PRIMARY KEY"), ("read_name", "VARCHAR NOT NULL"),
    ("record_index", "BIGINT NOT NULL"), ("chrom", "VARCHAR NOT NULL"), ("start0", "BIGINT NOT NULL"),
    ("end0", "BIGINT NOT NULL"), ("transcript_strand", "VARCHAR NOT NULL"), ("seq_available", "BOOLEAN NOT NULL"),
    ("tss0", "BIGINT"), ("tes0", "BIGINT"), ("clip5_len", "INTEGER"), ("clip3_len", "INTEGER"),
    ("clip5_seq", "VARCHAR"), ("clip3_seq", "VARCHAR"), ("polyA_untemplated_len", "INTEGER"),
    ("downstream20_A", "INTEGER"), ("downstream_max_consecutive_A", "INTEGER"), ("primer5_hit", "BOOLEAN"),
    ("ssp_adapter_hit", "BOOLEAN"), ("protocol", "VARCHAR"), ("adapter_5c_enabled", "BOOLEAN NOT NULL"),
)
COLUMN_NAMES = tuple(c for c, _ in COLUMNS)


class ReadEndError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ReadEndError(message)


def revcomp(s):
    return s.translate(_COMP)[::-1]


def _es():
    root = Path(__file__).resolve().parents[3] / "revision" / "scripts"
    import importlib.util

    spec = importlib.util.spec_from_file_location("evidence_support_59_read_ends", root / "59_evidence_support.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ record geometry
def parse_line(line):
    """(read, flag, chrom, start0, end0, left_clip, right_clip, left_seq, right_seq, seq_available) of a SAM line."""
    f = line.rstrip("\n").split("\t")
    read, flag, chrom, pos, cigar, seq = f[0], int(f[1]), f[2], int(f[3]), f[5], f[9]
    ops = [(int(n), op) for n, op in _CIGAR.findall(cigar)]
    require(ops and "".join(f"{n}{op}" for n, op in ops) == cigar, f"{read}: bad CIGAR {cigar!r}")
    inner = [o for o in ops if o[1] != "H"]
    left = inner[0][0] if inner and inner[0][1] == "S" else 0
    right = inner[-1][0] if len(inner) > 1 and inner[-1][1] == "S" else 0
    ref_len = sum(n for n, op in ops if op in "MDN=X")
    seq_ok = seq != "*"
    lseq = seq[:left] if seq_ok else None
    rseq = seq[len(seq) - right:] if seq_ok and right else ("" if seq_ok else None)
    return read, flag, chrom, pos - 1, pos - 1 + ref_len, left, right, lseq, rseq, seq_ok


def leading_run(s, base="A"):
    n = 0
    for c in s or "":
        if c.upper() != base:
            break
        n += 1
    return n


def max_run(s, base="A"):
    best = cur = 0
    for c in s:
        cur = cur + 1 if c.upper() == base else 0
        best = max(best, cur)
    return best


def adapter_hit(seq, adapter, min_len=ADAPTER_MIN_LEN, min_identity=ADAPTER_MIN_IDENTITY):
    """Ungapped match of any >= ``min_len`` suffix window of ``adapter`` anywhere in ``seq`` at >= ``min_identity``."""
    if not seq:
        return False
    s, a = seq.upper(), adapter.upper()
    for L in range(len(a), min_len - 1, -1):
        w = a[len(a) - L:]
        need = min_identity * L
        for i in range(0, len(s) - L + 1):
            if sum(1 for x, y in zip(s[i:i + L], w) if x == y) >= need:
                return True
    return False


def measure(line, tstrand, genome, *, primer5=None):
    """The read-end fields of one SAM record given the C0 transcript strand."""
    read, flag, chrom, s0, e0, left, right, lseq, rseq, seq_ok = parse_line(line)
    out = {"chrom": chrom, "start0": s0, "end0": e0, "seq_available": seq_ok}
    if tstrand not in ("+", "-"):
        out.update(tss0=None, tes0=None, clip5_len=None, clip3_len=None, clip5_seq=None, clip3_seq=None,
                   polyA_untemplated_len=None, downstream20_A=None, downstream_max_consecutive_A=None,
                   primer5_hit=None, ssp_adapter_hit=None)
        return out
    g = genome[chrom]
    if tstrand == "+":
        c5, c3 = left, right
        s5 = lseq if seq_ok else None
        s3 = rseq if seq_ok else None
        down = g[e0:e0 + DOWNSTREAM].upper()
        tss0, tes0 = s0, e0 - 1
    else:
        c5, c3 = right, left
        s5 = revcomp(rseq) if seq_ok else None
        s3 = revcomp(lseq) if seq_ok else None
        down = revcomp(g[max(0, s0 - DOWNSTREAM):s0]).upper()
        tss0, tes0 = e0 - 1, s0
    out.update(tss0=tss0, tes0=tes0, clip5_len=c5, clip3_len=c3, clip5_seq=s5, clip3_seq=s3,
               polyA_untemplated_len=leading_run(s3) if seq_ok else None,
               downstream20_A=down.count("A"), downstream_max_consecutive_A=max_run(down),
               primer5_hit=(adapter_hit(s5, primer5) if (seq_ok and primer5) else None),
               ssp_adapter_hit=adapter_hit(s5, ADAPTERS["ssp"]) if seq_ok else None)
    return out


# ------------------------------------------------------------------------------------------ inputs
def _sidecar_sha(path):
    sha = LK.sha256(path)
    side = Path(str(path) + ".sha256")
    require(side.is_file() and side.read_text().split()[:1] == [sha], f"{path}: sha256 sidecar missing or != {sha}")
    require(not Path(str(path) + ".wal").exists(), f"{path} has a WAL")
    return sha


def read_sheet(path, required=("dataset", "run", "protocol")):
    import csv

    rows = {}
    with open(path, newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in required if c not in (r.fieldnames or [])]
        require(not missing, f"{path}: sheet lacks columns {missing}")
        for row in r:
            key = (row["dataset"], row["run"])
            require(key not in rows, f"{path}: duplicate sheet row {key}")
            rows[key] = row
    return rows


def c0_alignment_inputs(con, species):
    """(source, dataset, run) -> {path, sha256} from the merged snapshot's c0_provenance ``inputs``."""
    out = {}
    for source, value in con.execute("SELECT source, value FROM c0_provenance WHERE species = ? AND key = 'inputs' "
                                     "ORDER BY source", [species]).fetchall():
        inputs = json.loads(value)
        for k, v in inputs.items():
            if k.startswith("alignment:") and source in READ_SOURCES:
                out[(source, k[len("alignment:"):])] = v
        if source in READ_SOURCES:
            out[(source, "reference")] = inputs.get("reference")
    return out


# ------------------------------------------------------------------------------------------ build
def build(*, merged, species, reference, ont_sheet, pacbio_sheet, out, samtools="samtools", alignment_paths=None,
          argv=()):
    import duckdb

    out = Path(out).absolute()
    for p in (out, Path(str(out) + ".sha256"), Path(str(out) + ".PROVENANCE.json")):
        require(not os.path.lexists(p), f"{p} exists; refusing to overwrite")
    merged_sha = _sidecar_sha(merged)
    ref_sha = LK.sha256(reference)
    sheets = {"ont": (ont_sheet, read_sheet(ont_sheet)),
              "pacbio": (pacbio_sheet, read_sheet(pacbio_sheet, required=("dataset", "run")))}
    ES = _es()
    genome = LK.load_build_b5().load_fasta(str(reference))
    con = duckdb.connect(str(merged), read_only=True)
    try:
        inputs = c0_alignment_inputs(con, species)
        rows = con.execute("SELECT source, dataset, run, read_name, record_index, alignment_id, chrom, start0, end0, "
                           "transcript_strand FROM evidence_alignment WHERE species = ? AND accepted AND source IN "
                           "('est', 'ont', 'pacbio') ORDER BY source, dataset, run, record_index", [species]).fetchall()
    finally:
        con.close()
    by_run = defaultdict(dict)
    for source, ds, run, read, idx, aid, chrom, s0, e0, ts in rows:
        key = (read, idx)
        require(key not in by_run[(source, ds, run)], f"C0: two accepted alignments share {(ds, run, read, idx)}")
        by_run[(source, ds, run)][key] = (aid, chrom, s0, e0, ts)
    tmp = Path(str(out) + ".partial")
    require(not os.path.lexists(tmp), f"{tmp} exists; remove it")
    qc = Counter()
    used_inputs = {}
    dst = duckdb.connect(str(tmp))
    try:
        dst.execute("SET threads = 1")
        dst.execute("CREATE TABLE read_end (" + ", ".join(f"{c} {t}" for c, t in COLUMNS) + ")")
        dst.execute("CREATE TABLE read_end_meta (key VARCHAR NOT NULL PRIMARY KEY, value VARCHAR NOT NULL)")
        for (source, ds, run), want in sorted(by_run.items()):
            rec = inputs.get((source, f"{ds}/{run}"))
            require(rec is not None, f"merged c0_provenance names no alignment for {source} {ds}/{run}")
            require((inputs.get((source, "reference")) or {}).get("sha256") == ref_sha,
                    f"{reference}: sha256 != the reference the {source} C0 recorded")
            path = (alignment_paths or {}).get(f"{ds}/{run}", rec["path"])
            sha = LK.sha256(path)
            require(sha == rec["sha256"], f"{path}: sha256 {sha} != the frozen alignment the {source} C0 recorded")
            used_inputs[f"{source}:{ds}/{run}"] = {"path": str(Path(path).absolute()), "sha256": sha}
            sheet_row = None
            if source in sheets:
                sheet_row = sheets[source][1].get((ds, run))
                require(sheet_row is not None, f"{sheets[source][0]}: no row for {ds}/{run} (every long-read run needs one)")
            protocol = sheet_row.get("protocol") if sheet_row else None
            primer5 = (sheet_row.get("primer5_sequence") or None) if sheet_row else None
            approved = bool(sheet_row) and str(sheet_row.get("adapter_5c_approved", "")).lower() == "true"
            batch, seen = [], set()
            for i, line in enumerate(ES.sam_lines(str(path), samtools)):
                if line.startswith("@") or not line.strip():
                    continue
                read = line.split("\t", 1)[0]
                hit = want.get((read, i))
                if hit is None:
                    continue
                aid, chrom, s0, e0, ts = hit
                m = measure(line, ts, genome, primer5=primer5)
                require((m["chrom"], m["start0"], m["end0"]) == (chrom, s0, e0),
                        f"{ds}/{run} {read} record {i}: SAM geometry {m['chrom']}:{m['start0']}-{m['end0']} != C0 "
                        f"{chrom}:{s0}-{e0}")
                m.update(species=species, source=source, dataset=ds, run=run, alignment_id=aid, read_name=read,
                         record_index=i, transcript_strand=ts, protocol=protocol, adapter_5c_enabled=approved)
                batch.append(tuple(m[c] for c in COLUMN_NAMES))
                seen.add((read, i))
                qc[f"{source}:rows"] += 1
                qc[f"{source}:seq_available"] += m["seq_available"]
                qc[f"{source}:polyA_ge10"] += (m["polyA_untemplated_len"] or 0) >= 10
                qc[f"{source}:ssp_adapter_hit"] += bool(m["ssp_adapter_hit"])
                if len(batch) >= 4096:
                    dst.executemany(f"INSERT INTO read_end VALUES ({','.join('?' * len(COLUMNS))})", batch)
                    batch = []
            if batch:
                dst.executemany(f"INSERT INTO read_end VALUES ({','.join('?' * len(COLUMNS))})", batch)
            missing = sorted(set(want) - seen)
            require(not missing, f"{ds}/{run}: {len(missing)} C0 accepted alignment(s) not found in {path}: {missing[:3]}")
        prov = {"version": VERSION, "species": species, "polya_rule": POLYA_RULE, "adapters": ADAPTERS,
                "adapter_min_len": ADAPTER_MIN_LEN, "adapter_min_identity": ADAPTER_MIN_IDENTITY,
                "downstream_window": DOWNSTREAM,
                "inputs": {"merged": {"path": str(Path(merged).absolute()), "sha256": merged_sha},
                           "reference": {"path": str(Path(reference).absolute()), "sha256": ref_sha},
                           "ont_sheet": {"path": str(Path(ont_sheet).absolute()), "sha256": LK.sha256(ont_sheet)},
                           "pacbio_sheet": {"path": str(Path(pacbio_sheet).absolute()),
                                            "sha256": LK.sha256(pacbio_sheet)},
                           "alignments": used_inputs},
                "code_sha256": {Path(__file__).name: LK.sha256(__file__)}, "qc": dict(sorted(qc.items())),
                "argv": list(argv), "duckdb_version": duckdb.__version__}
        dst.executemany("INSERT INTO read_end_meta VALUES (?, ?)",
                        sorted((k, json.dumps(v, sort_keys=True)) for k, v in prov.items()))
        dst.execute("CHECKPOINT")
    finally:
        dst.close()
    require(not os.path.exists(str(tmp) + ".wal"), f"{tmp}.wal remains; refusing to publish")
    os.link(tmp, out)
    tmp.unlink()
    digest = LK.sha256(out)
    with open(str(out) + ".sha256", "x") as fh:
        fh.write(f"{digest}  {out.name}\n")
    prov["output"], prov["output_sha256"] = str(out), digest
    with open(str(out) + ".PROVENANCE.json", "x") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return prov


def open_read_ends(path):
    path = Path(path)
    sha = _sidecar_sha(path)
    prov = json.loads(Path(str(path) + ".PROVENANCE.json").read_text())
    require(prov.get("output_sha256") == sha and prov.get("version") == VERSION, f"{path}: PROVENANCE does not bind it")
    return sha, prov
