"""WP2-prot (plan v4.1 §4): protein-placement supplement from the raw miniprot GFF (##PAF lines) and the reference.

The protein C0 keeps only CDS intervals and writes ``query_length`` / ``aligned_query`` NULL; A50/A50.1 (UTR from
reads) and A19.3 (protein CDS) need the query extent, the GFF phase and the codon context.  This module re-reads the
file the protein C0 recorded and writes one row per ACCEPTED PRIMARY placement:

* selection exactly as ``b6_build_c0.load_protein``: records in file order (``gidx``); per protein the primary is the
  highest score, ties by ``gidx``; a protein whose maximum score is shared is ``mapping_ambiguous`` (no accepted
  primary); accepted = ``protein_accept.judge`` ok and not ambiguous.  Streaming, two passes over the GFF;
* ``alignment_id`` = the unique C0 row joined on (protein_id, chrom, start0, end0, strand); every C0 accepted primary
  protein alignment must be joined exactly once, and its C0 ``aligned_block`` rows must equal the CDS blocks;
* ``organism`` = ``protein_accept.organism_of``; ``qlen, qstart0, qend`` = ``protein_accept.parse_paf``;
* ``cds_blocks`` = JSON ``[[start0, end0, gff_phase], ...]`` in genomic order;
* ``start_codon_ok`` (strand-aware ATG at the first CDS base), ``stop_codon_ok`` (TAA/TAG/TGA under the frozen stop
  convention: ``inside_last_block`` = the last three CDS bases, ``after_last_block`` = the three bases after the last
  block, transcript orientation), ``orf_complete`` = qstart0 = 0 and qend = qlen and start_ok and stop_ok;
* ``frameshift_n`` (PAF ``fs:i``) and ``internal_stop_n`` (in-frame stops before the terminal codon) -- recorded only;
* the inherited C0 flags (molecule id/state, train_tiles_excluded, exclusion_cause, role, mapping_ambiguous, ...).

The stop convention is an input: it is measured (``measure_stop_convention``) on one miniprot GFF per species, frozen
once if all agree, per species otherwise; the measurement on the file being processed is recorded in provenance.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

VERSION = "protein_placement_b6_v1"
STOP_CONVENTIONS = ("inside_last_block", "after_last_block")
STOPS = {"TAA", "TAG", "TGA"}
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
COLUMNS = (
    ("species", "VARCHAR NOT NULL"), ("dataset", "VARCHAR NOT NULL"), ("run", "VARCHAR NOT NULL"),
    ("alignment_id", "VARCHAR NOT NULL PRIMARY KEY"), ("member_id", "VARCHAR NOT NULL"),
    ("molecule_id", "VARCHAR NOT NULL"), ("protein_id", "VARCHAR NOT NULL"), ("organism", "VARCHAR NOT NULL"),
    ("chrom", "VARCHAR NOT NULL"), ("strand", "VARCHAR NOT NULL CHECK (strand IN ('+', '-'))"),
    ("start0", "BIGINT NOT NULL"), ("end0", "BIGINT NOT NULL"), ("score", "BIGINT NOT NULL"),
    ("qlen", "BIGINT NOT NULL"), ("qstart0", "BIGINT NOT NULL"), ("qend", "BIGINT NOT NULL"),
    ("n_blocks", "INTEGER NOT NULL"), ("cds_blocks", "VARCHAR NOT NULL"),
    ("start_codon_ok", "BOOLEAN NOT NULL"), ("stop_codon_ok", "BOOLEAN NOT NULL"),
    ("stop_convention", "VARCHAR NOT NULL"), ("orf_complete", "BOOLEAN NOT NULL"),
    ("frameshift_n", "INTEGER"), ("internal_stop_n", "INTEGER NOT NULL"),
    ("molecule_state", "VARCHAR NOT NULL"), ("train_tiles_excluded", "BOOLEAN NOT NULL"),
    ("exclusion_cause", "VARCHAR"), ("role", "VARCHAR NOT NULL"), ("validation_only", "BOOLEAN NOT NULL"),
    ("mapping_ambiguous", "BOOLEAN NOT NULL"), ("c0_provenance_sha256", "VARCHAR NOT NULL"),
)
COLUMN_NAMES = tuple(c for c, _ in COLUMNS)


class PlacementError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise PlacementError(message)


def _scripts():
    root = Path(__file__).resolve().parents[3] / "revision" / "scripts" / "evidence"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import protein_accept as PA  # noqa: E402

    return PA


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def revcomp(s):
    return s.translate(_COMP)[::-1]


# ------------------------------------------------------------------------------------------ miniprot records
def iter_records(path):
    """Every mRNA record of a miniprot GFF in file order, with its ##PAF record and CDS (start1, end1, phase)."""
    PA = _scripts()
    paf, cur, cds = None, None, []

    def record():
        attrs, score, chrom, strand, s1, e1 = cur
        prot = (attrs.get("Target", "").split() or [""])[0]
        require(paf is not None and paf["qname"] == prot, f"{attrs.get('ID')}: no matching ##PAF record")
        ok, reasons, introns, identity, coverage = PA.judge(paf, attrs, [(s, e) for s, e, _ in cds])
        return {"protein": prot, "organism": PA.organism_of(prot), "chrom": chrom, "strand": strand, "score": score,
                "start1": s1, "end1": e1, "cds": sorted(cds), "ok": ok, "reasons": reasons, "paf": paf}

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##PAF\t"):
                if cur is not None:
                    yield record()
                    cur, cds = None, []
                paf = PA.parse_paf(line)
                continue
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] == "mRNA":
                if cur is not None:
                    yield record()
                cur, cds = (dict(PA.ATTR.findall(f[8])), int(f[5]), f[0], f[6], int(f[3]), int(f[4])), []
            elif f[2] == "CDS" and cur is not None:
                phase = f[7]
                require(phase in ("0", "1", "2"), f"{f[8]}: CDS phase {phase!r} is not 0/1/2")
                cds.append((int(f[3]), int(f[4]), int(phase)))
    if cur is not None:
        yield record()


def primary_index(path):
    """Pass 1: protein -> (max score, n records at max, gidx of the rank-0 record)."""
    best = {}
    for gidx, r in enumerate(iter_records(path)):
        p = r["protein"]
        if p not in best or r["score"] > best[p][0]:
            best[p] = (r["score"], 1, gidx)
        elif r["score"] == best[p][0]:
            best[p] = (best[p][0], best[p][1] + 1, best[p][2])
    return best


def accepted_primaries(path):
    """Pass 2: (gidx, record) of every accepted primary placement, in file order."""
    best = primary_index(path)
    for gidx, r in enumerate(iter_records(path)):
        score, n_max, g = best[r["protein"]]
        if gidx == g and n_max == 1 and r["ok"]:
            yield gidx, r


# ------------------------------------------------------------------------------------------ codon context
def _seq(genome, chrom, a, b):
    s = genome.get(chrom)
    if s is None or a < 0 or b > len(s) or a >= b:
        return ""
    return s[a:b].upper()


def codon_context(genome, chrom, strand, blocks0, first_phase, convention):
    """(start_codon_ok, stop_codon_ok, internal_stop_n) for 0-based genomic CDS blocks [(start0, end0), ...].

    The start codon and an ``inside_last_block`` stop are read from the transcript-oriented SPLICED CDS, so a terminal
    codon split across CDS blocks is read from its exonic bases, never from intronic ones.  The
    ``after_last_block`` stop is the three genomic bases 3' of the last block (transcript orientation)."""
    require(convention in STOP_CONVENTIONS, f"stop convention {convention!r} not in {STOP_CONVENTIONS}")
    blocks0 = sorted(blocks0)
    first, last = blocks0[0], blocks0[-1]
    spliced = "".join(_seq(genome, chrom, a, b) for a, b in blocks0)
    if strand == "+":
        exterior = _seq(genome, chrom, last[1], last[1] + 3)
    else:
        spliced = revcomp(spliced)
        exterior = revcomp(_seq(genome, chrom, first[0] - 3, first[0]))
    start = spliced[:3]
    stop = spliced[-3:] if convention == "inside_last_block" else exterior
    start_ok, stop_ok = len(start) == 3 and start == "ATG", len(stop) == 3 and stop in STOPS
    codons = [spliced[i:i + 3] for i in range(first_phase, len(spliced) - 2, 3)]
    if convention == "inside_last_block" and stop_ok and codons and codons[-1] in STOPS:
        codons = codons[:-1]
    return start_ok, stop_ok, sum(1 for c in codons if c in STOPS)


def placement_fields(genome, r, convention):
    """The computed fields of one accepted primary record (tested for both stop conventions)."""
    paf = r["paf"]
    blocks = [(s1 - 1, e1, ph) for s1, e1, ph in r["cds"]]
    tx_first = blocks[0] if r["strand"] == "+" else blocks[-1]
    start_ok, stop_ok, internal = codon_context(genome, r["chrom"], r["strand"], [(a, b) for a, b, _ in blocks],
                                                tx_first[2], convention)
    qlen, qstart0, qend = paf["qlen"], paf["qstart"], paf["qend"]
    return {"protein_id": r["protein"], "organism": r["organism"], "chrom": r["chrom"], "strand": r["strand"],
            "start0": r["start1"] - 1, "end0": r["end1"], "score": r["score"], "qlen": qlen, "qstart0": qstart0,
            "qend": qend, "n_blocks": len(blocks), "cds_blocks": json.dumps([list(b) for b in blocks], separators=(",", ":")),
            "start_codon_ok": start_ok, "stop_codon_ok": stop_ok, "stop_convention": convention,
            "orf_complete": qstart0 == 0 and qend == qlen and start_ok and stop_ok,
            "frameshift_n": paf["fs"], "internal_stop_n": internal}


def measure_stop_convention(gff, genome):
    """Counts over accepted primaries: is the stop codon the last three CDS bases, the three after, both, neither?"""
    out = Counter()
    for _, r in accepted_primaries(gff):
        blocks = [(s1 - 1, e1) for s1, e1, _ in r["cds"]]
        ph = r["cds"][0][2] if r["strand"] == "+" else r["cds"][-1][2]
        inside = codon_context(genome, r["chrom"], r["strand"], blocks, ph, "inside_last_block")[1]
        after = codon_context(genome, r["chrom"], r["strand"], blocks, ph, "after_last_block")[1]
        out["both" if inside and after else "inside_last_block" if inside else "after_last_block" if after
            else "neither"] += 1
        out["accepted_primaries"] += 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------------ build
def _load_build_c0():
    root = Path(__file__).resolve().parents[3] / "revision" / "scripts" / "evidence"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import b6_build_c0 as B  # noqa: E402

    return B


def _verify_merged(merged, species, protein_sha):
    import duckdb

    merged = Path(merged)
    require(merged.is_file() and not Path(str(merged) + ".wal").exists(), f"{merged}: missing or has a WAL")
    side = Path(str(merged) + ".sha256")
    sha = sha256(merged)
    require(side.is_file() and side.read_text().split()[:1] == [sha], f"{merged}: sha256 sidecar missing or != {sha}")
    con = duckdb.connect(str(merged), read_only=True)
    try:
        rows = con.execute("SELECT input_sha256 FROM merge_manifest WHERE species = ? AND source = 'protein'",
                           [species]).fetchall()
    finally:
        con.close()
    require(rows == [(protein_sha,)], f"{merged}: merge_manifest does not bind the protein C0 {protein_sha} ({rows})")
    return sha


def _c0_primaries(con):
    rows = con.execute(
        "SELECT a.read_name, a.chrom, a.start0, a.end0, a.alignment_strand, a.alignment_id, a.member_id, a.molecule_id, "
        "a.dataset, a.run, a.role, a.validation_only, a.mapping_ambiguous, a.provenance_sha256, m.molecule_state, "
        "m.train_tiles_excluded, m.exclusion_cause FROM evidence_alignment a JOIN molecule_member m USING (member_id) "
        "WHERE a.source = 'protein' AND a.accepted AND a.placement_kind = 'primary' ORDER BY a.alignment_id").fetchall()
    out = {}
    for r in rows:
        key = tuple(r[:5])
        require(key not in out, f"protein C0: two accepted primary rows share {key}; the join is not unique")
        out[key] = r[5:]
    return out


def build(*, species, gff, reference, protein_c0, merged, out, stop_convention, argv=()):
    import duckdb

    B = _load_build_c0()
    b5 = B.LK.load_build_b5()
    out = Path(out).absolute()
    for p in (out, Path(str(out) + ".sha256"), Path(str(out) + ".PROVENANCE.json")):
        require(not os.path.lexists(p), f"{p} exists; refusing to overwrite")
    require(stop_convention in STOP_CONVENTIONS, f"--stop-convention must be one of {STOP_CONVENTIONS}")
    try:
        c0_sha, c0_prov = B.open_published_c0(protein_c0, species, "protein")
    except B.C0Error as exc:
        raise PlacementError(str(exc)) from exc
    inputs = json.loads(c0_prov["inputs"])
    gff_sha, ref_sha = sha256(gff), sha256(reference)
    require(inputs.get("alignment:protein", {}).get("sha256") == gff_sha,
            f"{gff}: sha256 {gff_sha} != the miniprot GFF the protein C0 recorded")
    require(inputs.get("reference", {}).get("sha256") == ref_sha,
            f"{reference}: sha256 {ref_sha} != the reference the protein C0 recorded (= corpus build_manifest)")
    merged_sha = _verify_merged(merged, species, c0_sha)
    genome = b5.load_fasta(str(reference))
    con = duckdb.connect(str(protein_c0), read_only=True)
    try:
        c0 = _c0_primaries(con)
    finally:
        con.close()
    measured = measure_stop_convention(gff, genome)
    tmp = Path(str(out) + ".partial")
    require(not os.path.lexists(tmp), f"{tmp} exists; remove it")
    qc = Counter()
    dst = duckdb.connect(str(tmp))
    try:
        dst.execute("SET threads = 1")
        dst.execute("CREATE TABLE protein_placement (" + ", ".join(f"{c} {t}" for c, t in COLUMNS) + ")")
        dst.execute("CREATE TABLE protein_placement_meta (key VARCHAR NOT NULL PRIMARY KEY, value VARCHAR NOT NULL)")
        batch, used = [], set()
        for _, r in accepted_primaries(gff):
            f = placement_fields(genome, r, stop_convention)
            key = (f["protein_id"], f["chrom"], f["start0"], f["end0"], f["strand"])
            require(key in c0, f"accepted primary {key} of the miniprot GFF has no accepted primary row in the protein C0")
            require(key not in used, f"two accepted primaries map to {key}")
            used.add(key)
            (alignment_id, member_id, molecule_id, dataset, run, role, validation_only, ambiguous, prov, state, tte,
             cause) = c0[key]
            f.update(species=species, dataset=dataset, run=run, alignment_id=alignment_id, member_id=member_id,
                     molecule_id=molecule_id, molecule_state=state, train_tiles_excluded=tte, exclusion_cause=cause,
                     role=role, validation_only=validation_only, mapping_ambiguous=ambiguous, c0_provenance_sha256=prov)
            batch.append(tuple(f[c] for c in COLUMN_NAMES))
            qc["rows"] += 1
            qc["orf_complete"] += f["orf_complete"]
            qc["start_codon_ok"] += f["start_codon_ok"]
            qc["stop_codon_ok"] += f["stop_codon_ok"]
            qc["internal_stop_placements"] += f["internal_stop_n"] > 0
            qc["state:" + state] += 1
            if len(batch) >= 2048:
                dst.executemany(f"INSERT INTO protein_placement VALUES ({','.join('?' * len(COLUMNS))})", batch)
                batch = []
        if batch:
            dst.executemany(f"INSERT INTO protein_placement VALUES ({','.join('?' * len(COLUMNS))})", batch)
        missing = sorted(set(c0) - used)
        require(not missing, f"{len(missing)} accepted primary protein C0 row(s) have no miniprot record: {missing[:3]}")
        # the C0 aligned_block rows of each joined alignment must equal its CDS blocks (both directions)
        dst.execute(f"ATTACH '{Path(protein_c0).absolute()}' AS c0 (READ_ONLY)")
        diff = dst.execute("""
            WITH sup AS (SELECT p.alignment_id, CAST(b.value[1] AS BIGINT) AS start0, CAST(b.value[2] AS BIGINT) AS end0
                         FROM protein_placement p, unnest(CAST(p.cds_blocks AS BIGINT[][])) AS b(value)),
                 ref AS (SELECT alignment_id, start0, end0 FROM c0.aligned_block
                         WHERE alignment_id IN (SELECT alignment_id FROM protein_placement))
            SELECT (SELECT count(*) FROM (SELECT * FROM sup EXCEPT ALL SELECT * FROM ref)),
                   (SELECT count(*) FROM (SELECT * FROM ref EXCEPT ALL SELECT * FROM sup))""").fetchone()
        dst.execute("DETACH c0")
        require(diff == (0, 0), f"CDS blocks differ from the protein C0 aligned_block rows: {diff}")
        prov = {"version": VERSION, "species": species, "stop_convention": stop_convention,
                "stop_convention_measurement": measured,
                "inputs": {"miniprot_gff": {"path": str(Path(gff).absolute()), "sha256": gff_sha},
                           "reference": {"path": str(Path(reference).absolute()), "sha256": ref_sha},
                           "protein_c0": {"path": str(Path(protein_c0).absolute()), "sha256": c0_sha,
                                          "provenance_sha256": c0_prov.get("provenance_sha256")},
                           "merged": {"path": str(Path(merged).absolute()), "sha256": merged_sha}},
                "code_sha256": {Path(__file__).name: sha256(__file__)}, "qc": dict(sorted(qc.items())),
                "argv": list(argv), "duckdb_version": duckdb.__version__}
        dst.executemany("INSERT INTO protein_placement_meta VALUES (?, ?)",
                        sorted((k, json.dumps(v, sort_keys=True)) for k, v in prov.items()))
        dst.execute("CHECKPOINT")
    finally:
        dst.close()
    require(not os.path.exists(str(tmp) + ".wal"), f"{tmp}.wal remains; refusing to publish")
    os.link(tmp, out)
    tmp.unlink()
    digest = sha256(out)
    with open(str(out) + ".sha256", "x") as fh:
        fh.write(f"{digest}  {out.name}\n")
    prov["output"], prov["output_sha256"] = str(out), digest
    with open(str(out) + ".PROVENANCE.json", "x") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return prov


def open_supplement(path, *, expected_sha256=None):
    """Verify a published supplement (sidecar, PROVENANCE output_sha256, no WAL); returns (sha256, provenance)."""
    path = Path(path)
    require(path.is_file() and not Path(str(path) + ".wal").exists(), f"{path}: missing or has a WAL")
    sha = sha256(path)
    side = Path(str(path) + ".sha256")
    require(side.is_file() and side.read_text().split()[:1] == [sha], f"{path}: sidecar missing or != {sha}")
    prov = json.loads(Path(str(path) + ".PROVENANCE.json").read_text())
    require(prov.get("output_sha256") == sha and prov.get("version") == VERSION, f"{path}: PROVENANCE does not bind it")
    if expected_sha256 is not None:
        require(sha == expected_sha256, f"{path}: sha256 {sha} != expected {expected_sha256}")
    return sha, prov
