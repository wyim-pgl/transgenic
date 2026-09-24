#!/usr/bin/env python3
"""Merge per species x source C0 DBs into the one evidence DB whose sha256 A46 rule 9 records (SPEC-1 step 6).

    b6_merge_evidence_db.py --inputs Athaliana.est.db Athaliana.ont.db ... --out evidence_c0.db

Fail-closed: every input must carry schema ``c0-v1`` with the exact column list, a single (species, source) in
``c0_meta``, a zero leakage QC count (step 7 gate re-read) and a ``<input>.PROVENANCE.txt`` (REQUIRED, as
b6_build_c0.open_published_c0 requires it: only a published C0 DB is merged) whose output_sha256 matches; a (species, source) may appear once; any duplicate primary key across inputs
is refused (no silent de-duplication).  Step-7 gate on the merged rows (independent of the per-DB builds): every
``est_protein_hsp.molecule_id`` must be a molecule of the same species' EST ``molecule_member`` (no orphans), an HSP
of an ``excluded_heldout`` EST molecule must be ``leakage_excluded`` and an HSP of a train-tiles-excluded EST
molecule must be ``train_tiles_excluded``; the same two flags are inherited from the HSP's protein query where it is a
miniprot-aligned molecule of the merged protein C0 (rule 3: one unit everywhere it appears; rule 6: protein queries
touching excluded intervals are removed molecule-wide); and every HSP interval must lie inside an ACCEPTED primary
placement of a member of its EST molecule with its ``est_accession`` (rule 6 / A18.5: projected through a retained
observation); an HSP of a ``never`` (validation-only) EST molecule must be ``leakage_excluded``; and among retained
HSPs of one molecule and protein source no two overlapping intervals may disagree on strand or codon frame (rule 6
MULTI_HSP_RULE, re-tested independently of the builder's resolution).  Provenance gate across inputs: every input of
one species must record the same ``leakage_intervals_sha256``, ``seed`` and corpus / splits / flags / build_flags /
reference / gff input sha256 values (compared over the UNION of both inputs' keys, so an omitted or extra
``build_flags_*`` entry is a difference), and a protein C0 is merged only together with the very EST C0 (file sha256)
it recorded as ``inputs.est_c0`` -- an EST C0 rebuilt after the projection is refused, whatever its row content.
Species-level rule 4(b) exhaustiveness ("reported per species and required to be >= 99.9 %"): every audited source
(est / ont / pacbio) must carry ``exhaustiveness_below_cap`` / ``exhaustiveness_molecules`` / ``audit_n`` in ``c0_qc``;
each source's own fraction must reach the per-source floor ``leakage.MIN_SOURCE_EXHAUSTIVENESS`` (author decision
2026-09-23, re-checked here from the recorded counters so an older builder cannot bypass it) and their species sums
must reach ``leakage.MIN_EXHAUSTIVENESS`` under one N, or the merge is refused.  The result is written to ``c0_meta``
key ``exhaustiveness``.  Rule 6 MULTI_HSP_RULE is re-tested within each population (retained HSPs; leakage-excluded
HSPs), never across (fix round 6, finding 2).  A16 completeness codes:
every input must record a known ``completeness_codes.policy`` in its provenance and its accepted rows must obey it
(the NULL policy is explicit and re-counted here, never silent).  The output is created exclusively (temp file + hard
link), and its sha256 is written once to ``<out>.sha256`` -- that value is the evidence DB sha256 of rule 9.

Row order: the merged tables have NO guaranteed physical row order (DuckDB inserts are not order-preserving across
inputs and threads); every reader must ``ORDER BY`` a key when order matters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_EVIDENCE = Path(__file__).resolve().parent
if str(_EVIDENCE) not in sys.path:
    sys.path.insert(0, str(_EVIDENCE))

from transgenic.evidence import c0_schema as S  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402
import b6_build_c0 as B  # noqa: E402  (HSP_CONFLICT_SQL, COMPLETENESS_* -- one definition, re-tested here)

AUDITED_SOURCES = ("est", "ont", "pacbio")
EXHAUSTIVENESS_METRICS = ("exhaustiveness_below_cap", "exhaustiveness_molecules", "audit_n")


class MergeError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise MergeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _recorded_output_sha(path):
    prov = Path(str(path) + ".PROVENANCE.txt")
    require(prov.is_file(), f"{prov} missing (only a published C0 DB is merged)")
    for line in prov.read_text().splitlines():
        key, _, value = line.partition("\t")
        if key == "output_sha256":
            return value.strip()
    return None


def hsp_gate(con):
    """Route-B rows against the merged EST and protein molecule_member: orphans, inherited exclusion flags (EST
    molecule and protein query) and the accepted-primary placement of every interval."""
    rows = con.execute("""
        WITH e AS (SELECT species, molecule_id, bool_or(molecule_state = 'excluded_heldout') AS excl,
                          bool_or(train_tiles_excluded) AS tte, bool_and(molecule_state = 'never') AS never
                   FROM molecule_member WHERE source = 'est' GROUP BY species, molecule_id),
             p AS (SELECT species, read_name AS protein, bool_or(molecule_state = 'excluded_heldout') AS excl,
                          bool_or(train_tiles_excluded) AS tte
                   FROM molecule_member WHERE source = 'protein' GROUP BY species, read_name)
        SELECT count(*) FILTER (WHERE e.molecule_id IS NULL),
               count(*) FILTER (WHERE e.excl AND NOT h.leakage_excluded),
               count(*) FILTER (WHERE e.tte AND NOT h.train_tiles_excluded),
               count(*) FILTER (WHERE p.excl AND NOT h.leakage_excluded),
               count(*) FILTER (WHERE p.tte AND NOT h.train_tiles_excluded),
               count(*) FILTER (WHERE e.never AND NOT h.leakage_excluded),
               count(*) FILTER (WHERE NOT EXISTS (
                   SELECT 1 FROM molecule_member m JOIN evidence_alignment a USING (member_id)
                   WHERE m.source = 'est' AND m.species = h.species AND m.molecule_id = h.molecule_id
                     AND m.accession = h.est_accession AND a.accepted AND a.placement_kind = 'primary'
                     AND a.chrom = h.chrom AND a.start0 <= h.genomic_start0 AND h.genomic_end0 <= a.end0)),
               count(*) FILTER (WHERE NOT EXISTS (
                   SELECT 1 FROM molecule_member m WHERE m.source = 'est' AND m.species = h.species
                     AND m.molecule_id = h.molecule_id AND m.accession = h.est_accession AND m.library = h.library))
        FROM est_protein_hsp h LEFT JOIN e ON e.species = h.species AND e.molecule_id = h.molecule_id
                               LEFT JOIN p ON p.species = h.species AND p.protein = h.protein_id""").fetchone()
    orphans, excl, tte, p_excl, p_tte, never, unplaced, stale_library = rows
    require(orphans == 0, f"{orphans} est_protein_hsp row(s) reference no EST molecule of their species (orphan "
                          "molecule_id; merge the species' EST C0 DB, built by the same molecule map producer)")
    require(excl == 0, f"{excl} est_protein_hsp row(s) of an excluded_heldout EST molecule are not leakage_excluded")
    require(tte == 0, f"{tte} est_protein_hsp row(s) of a train-tiles-excluded EST molecule lack train_tiles_excluded")
    require(p_excl == 0, f"{p_excl} est_protein_hsp row(s) of an excluded_heldout protein query are not leakage_excluded")
    require(p_tte == 0, f"{p_tte} est_protein_hsp row(s) of a train-tiles-excluded protein query lack train_tiles_excluded")
    require(never == 0, f"{never} est_protein_hsp row(s) of a never-training (validation-only) EST molecule are not "
                        "leakage_excluded")
    require(unplaced == 0, f"{unplaced} est_protein_hsp interval(s) lie in no accepted primary placement of their EST "
                           "member (rule 6 / A18.5: projected through a rejected or foreign alignment)")
    require(stale_library == 0, f"{stale_library} est_protein_hsp row(s) carry a library / est_accession that no EST "
                                "member of their molecule has (stale projection after a metadata correction)")
    conflicts = con.execute(B.HSP_CONFLICT_SQL).fetchone()[0]
    require(conflicts == 0, f"{conflicts} retained est_protein_hsp pair(s) of one molecule and protein source overlap with "
                            "different strand or codon frame (rule 6 MULTI_HSP_RULE not applied to the retained set)")
    conflicts = con.execute(B.HSP_CONFLICT_EXCLUDED_SQL).fetchone()[0]
    require(conflicts == 0, f"{conflicts} leakage-excluded est_protein_hsp pair(s) of one molecule and protein source "
                            "overlap with different strand or codon frame (rule 6 MULTI_HSP_RULE not applied to the "
                            "excluded population; fix round 6)")


IDENTITY_INPUTS = ("corpus", "splits", "flags", "reference", "gff")


def _provenance(con):
    """c0_provenance of a per species x source C0 DB as {key: decoded value} (values are JSON as the builder wrote)."""
    out = {}
    for key, value in con.execute("SELECT key, value FROM c0_provenance ORDER BY key").fetchall():
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def build_identity(prov):
    """The exclusion/corpus identity every input of one species must share."""
    inputs = prov.get("inputs") or {}
    ident = {"leakage_intervals_sha256": prov.get("leakage_intervals_sha256"), "seed": prov.get("seed"),
             # plan v4.1 §2.5: one b6-intervals-v2 manifest per species (absent on every input of an older C0)
             "interval_manifest_sha256": prov.get("interval_manifest_sha256")}
    for k in IDENTITY_INPUTS:
        ident[k] = (inputs.get(k) or {}).get("sha256")
    for k in sorted(inputs):
        if k.startswith("build_flags_"):
            ident[k] = inputs[k].get("sha256")
    return ident


def identity_diff(a, b):
    """Keys on which two identity dicts differ, over the UNION of their keys (an omitted key is a difference)."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def provenance_gate(manifest):
    """manifest rows: (path, digest, species, source, prov, ...).  Fails closed on any cross-input inconsistency."""
    by_species = {}
    for path, digest, species, source, prov, *_ in manifest:
        by_species.setdefault(species, {})[source] = (path, digest, prov)
    for species, sources in sorted(by_species.items()):
        first = None
        for source, (path, _, prov) in sorted(sources.items()):
            ident = build_identity(prov)
            require(ident["leakage_intervals_sha256"] and ident["seed"] is not None,
                    f"{path}: c0_provenance lacks leakage_intervals_sha256/seed")
            if first is None:
                first = (path, ident)
            else:
                diff = identity_diff(ident, first[1])
                require(not diff, f"{species}: {path} and {first[0]} disagree on {diff} (different excluded intervals, "
                                  "seed or corpus inputs cannot be merged into one evidence DB)")
        if "protein" in sources:
            ppath, _, pprov = sources["protein"]
            dep = ((pprov.get("inputs") or {}).get("est_c0") or {}).get("sha256")
            require(dep, f"{ppath}: protein C0 records no inputs.est_c0 sha256")
            require("est" in sources, f"{species}: {ppath} was projected against EST C0 sha256 {dep}, but no {species} "
                                      "EST C0 is among the inputs")
            epath, edigest, _ = sources["est"]
            require(edigest == dep, f"{species}: {ppath} was projected against EST C0 sha256 {dep}, but {epath} "
                                    f"(sha256 {edigest}) is being merged -- the EST C0 was rebuilt after the projection")
    return {species: build_identity(next(iter(sources.values()))[2]) for species, sources in sorted(by_species.items())}


def exhaustiveness_gate(manifest):
    """A46 rule 4(b) exhaustiveness over every audited source of each species (manifest rows carry the input's c0_qc
    as their 6th item): first the per-source floor (leakage.MIN_SOURCE_EXHAUSTIVENESS, author 2026-09-23) re-applied
    to the counters each input recorded, then the species sum (leakage.MIN_EXHAUSTIVENESS).  Fails closed when a
    source lacks the counters, a source is below its floor, the sources disagree on N, or the species fraction is
    below the threshold."""
    parts = {}
    for path, _digest, species, source, _prov, qc in manifest:
        if source not in AUDITED_SOURCES:
            continue
        missing = [m for m in EXHAUSTIVENESS_METRICS if m not in qc]
        require(not missing, f"{path}: c0_qc lacks {missing} (per-source exhaustiveness counters of A46 rule 4(b)); "
                             "rebuild with the current b6_build_c0.py")
        try:
            LK.source_exhaustiveness(f"{species}/{source} ({path})", qc["exhaustiveness_below_cap"],
                                     qc["exhaustiveness_molecules"], qc["audit_n"])
        except LK.LeakageError as exc:
            raise MergeError(f"per-source exhaustiveness floor: {exc}") from exc
        parts.setdefault(species, []).append((source, qc["exhaustiveness_below_cap"], qc["exhaustiveness_molecules"],
                                              qc["audit_n"]))
    out = {}
    for species, rows in sorted(parts.items()):
        try:
            out[species] = LK.species_exhaustiveness(species, rows)
        except LK.LeakageError as exc:
            raise MergeError(f"species exhaustiveness gate: {exc}") from exc
        out[species]["min_source_exhaustiveness"] = LK.MIN_SOURCE_EXHAUSTIVENESS
    return out


def completeness_gate(path, prov, qc):
    """The input must record a known A16 completeness-code policy and its accepted rows must obey it."""
    policy = (prov.get("completeness_codes") or {}).get("policy")
    require(policy in B.COMPLETENESS_CODES_POLICIES,
            f"{path}: c0_provenance records no known completeness_codes.policy ({policy!r}); a NULL A16 code column "
            "must be declared, never silent (rebuild with the current b6_build_c0.py)")
    counts = [qc.get(k) for k in ("completeness_codes_accepted_rows", "completeness_codes_null_accepted_rows",
                                  "completeness_codes_coded_accepted_rows")]
    require(all(c is not None for c in counts), f"{path}: c0_qc lacks the completeness_codes_* counters")
    return policy


def merge(inputs, out):
    import duckdb

    out = Path(out).absolute()
    if os.path.lexists(out) or os.path.lexists(str(out) + ".sha256"):
        raise FileExistsError(out)
    require(inputs, "no input DBs")
    seen, manifest = {}, []
    for path in inputs:
        path = Path(path).absolute()
        require(path.is_file() and not Path(str(path) + ".wal").exists(), f"{path}: missing or has a WAL")
        digest = sha256(path)
        recorded = _recorded_output_sha(path)
        require(recorded == digest, f"{path}: sha256 {digest} != its PROVENANCE output_sha256 {recorded}")
        con = duckdb.connect(str(path), read_only=True)
        try:
            meta = S.verify_schema(con)
            key = (meta.get("species"), meta.get("source"))
            require(all(key), f"{path}: c0_meta lacks species/source")
            require(key not in seen, f"duplicate species x source {key}: {seen.get(key)} and {path}")
            leak = con.execute("SELECT value FROM c0_qc WHERE species = ? AND source = ? AND metric = 'leakage_nonzero'",
                               list(key)).fetchall()
            require(leak == [(0.0,)], f"{path}: leakage QC gate not passed ({leak})")
            for table in S.DATA_TABLES:
                bad = con.execute(f"SELECT count(*) FROM {table} WHERE species <> ? OR source <> ?", list(key)).fetchone()[0]
                require(bad == 0, f"{path}: {table} has {bad} rows outside {key}")
            prov = _provenance(con)
            qc = dict(con.execute("SELECT metric, value FROM c0_qc WHERE species = ? AND source = ? ORDER BY metric",
                                  list(key)).fetchall())
            policy = completeness_gate(path, prov, qc)
            counts = con.execute(B.COMPLETENESS_COUNTS_SQL).fetchone()
            violation = B.completeness_policy_violation(policy, *counts)
            require(violation == 0, f"{path}: {violation} accepted alignment row(s) contradict the recorded A16 "
                                    f"completeness-code policy {policy!r}")
        finally:
            con.close()
        seen[key] = path
        manifest.append((str(path), digest, key[0], key[1], prov, qc))
    identity = provenance_gate(manifest)
    exhaustiveness = exhaustiveness_gate(manifest)
    manifest = [m[:4] for m in manifest]

    out.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=out.name + ".partial-", dir=out.parent)
    os.close(handle)
    os.unlink(tmp)
    con = None
    try:
        con = duckdb.connect(tmp)
        S.create_schema(con, merged=True)
        for i, (path, _, _, _) in enumerate(manifest):
            con.execute(f"ATTACH '{path}' AS src{i} (READ_ONLY)")
        for table in S.ALL_TABLES:
            if table in ("merge_manifest", "c0_meta"):
                continue
            cols = ", ".join(c for c, _ in S.table_columns(table))
            pk = S.primary_key(table)
            union = " UNION ALL ".join(f"SELECT {cols} FROM src{i}.{table}" for i in range(len(manifest)))
            if pk:
                keys = ", ".join(pk)
                dup = con.execute(f"SELECT {keys}, count(*) n FROM ({union}) GROUP BY {keys} HAVING n > 1 LIMIT 3").fetchall()
                require(not dup, f"duplicate {table} key(s) {list(pk)} across inputs, e.g. {dup}")
            con.execute(f"INSERT INTO {table} {union}")
        hsp_gate(con)
        con.executemany("INSERT INTO merge_manifest VALUES (?, ?, ?, ?)", manifest)
        con.execute("INSERT INTO c0_meta VALUES ('merged_inputs', ?)", [json.dumps(sorted(m[1] for m in manifest))])
        con.execute("INSERT INTO c0_meta VALUES ('exhaustiveness', ?)", [json.dumps(exhaustiveness, sort_keys=True)])
        for i in range(len(manifest)):
            con.execute(f"DETACH src{i}")
        S.verify_schema(con, merged=True)
        con.execute("CHECKPOINT")
        con.close()
        con = None
        os.link(tmp, out)
    finally:
        if con is not None:
            con.close()
        for p in (Path(tmp), Path(tmp + ".wal")):
            p.unlink(missing_ok=True)
    digest = sha256(out)
    with open(str(out) + ".sha256", "x") as fh:
        fh.write(f"{digest}  {out.name}\n")
    return {"output": str(out), "sha256": digest, "inputs": [{"path": m[0], "sha256": m[1], "species": m[2], "source": m[3]}
                                                             for m in manifest], "identity": identity,
            "exhaustiveness": exhaustiveness}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        result = merge(args.inputs, args.out)
    except (MergeError, S.SchemaError, FileExistsError, OSError) as exc:
        print(f"b6_merge_evidence_db failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
