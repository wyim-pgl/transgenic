"""B6 observation cache (plan §7.5): C0 rows -> one parquet shard per (species, chrom) + a sha-bound manifest.

Families
--------
* ``train``: built from a ``training``-family C0 (merged snapshot with a leakage model).  Holds the train_track
  molecules (train and valid tiles) and the A. thaliana ``excluded_heldout`` molecules of ``b6_input`` datasets
  (their evaluation input lives in this DB).
* ``eval``: built from an ``eval_only`` C0 (``b6_build_eval_c0.py``; Z. mays).  Holds ``b6_input`` rows only.

Row types and kinds (one table; ``row_type`` / ``kind``)
--------------------------------------------------------
* ``track/alignment``  accepted primary placement extent (panel-locus admission at evaluation)
* ``track/block``      aligned block (coverage), strand = C0 block strand (+/-/?)
* ``track/junction``   accepted corrected junction, strand +/- only; start0/end0 = intron [start0, end0)
* ``track/protein``    accepted primary protein CDS block (channel 24), ``organism`` = OrthoDB organism,
                       ``query_id`` = the protein query id (the D7 thinning key; molecule_id = exclusion/QC unit)
* ``track/phase``      route-B ``est_protein_hsp`` interval of an EST molecule, ``phase`` = codon position at start0
* ``chain/chain``      one C0 partial_chain (accepted primary, strand +/-, >= 1 intron); ``introns`` =
                       "donor0-acceptor0,..." in transcription order, ``introns_ok`` = "1,0,..." accepted flags
* ``xjunc/donor|acceptor``  rule 4(c) excluded junction coordinate (training family only), start0 = coordinate
* ``xint/<cause>``     C0 ``excluded_interval`` row (training family only); strand = scope (all|train_tiles),
                       tile_rn = the tile of a tile-specific interval (edge_partial ...) or -1; units_key = gene_id

Build-time filtering (counted per species x source x cause, at molecule level): ``never`` molecules, ``b6_score``
datasets, non-input ``validation_only`` rows, ``excluded_heldout`` molecules outside ``b6_input`` datasets, non-primary
arms (the >= 121 arm), non-primary/unaccepted placements (never selected), excluded protein queries, route-B rows
of train_track ESTs whose HSP is leakage-excluded (the protein query was excluded).

Roles are selected by (dataset, run) only (plan §3); a dataset absent from DATASET_ROLES is neither input nor score.

b6_build_obs_cache extracts and writes one (species, chrom) at a time (``extract_rows(species=, chrom=)`` +
``CacheWriter``) to bound memory; the published cache is byte-identical to the one-shot
``write_cache(extract_rows(...))``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from transgenic.evidence import b6_track_spec as TS
from transgenic.evidence import c0_schema as CS

CACHE_VERSION = "b6-obs-cache-v1"
FAMILIES = ("train", "eval")
POPULATIONS = ("train_track", "valid_track", "eval_input")
TRACK_KINDS = ("alignment", "block", "junction", "protein", "phase")

SCHEMA = pa.schema([
    ("row_type", pa.string()), ("kind", pa.string()), ("species", pa.string()), ("chrom", pa.string()),
    ("source", pa.string()), ("dataset", pa.string()), ("run", pa.string()), ("library", pa.string()),
    ("molecule_id", pa.string()), ("molecule_state", pa.string()), ("train_tiles_excluded", pa.bool_()),
    ("b6_input", pa.bool_()), ("b6_score", pa.bool_()), ("b6_input_scope", pa.string()),
    ("strand", pa.string()), ("phase", pa.int8()), ("start0", pa.int64()), ("end0", pa.int64()),
    ("organism", pa.string()), ("introns", pa.string()), ("introns_ok", pa.string()), ("units_key", pa.string()),
    ("tile_rn", pa.int64()), ("query_id", pa.string()),
])
COLUMNS = tuple(SCHEMA.names)


class CacheError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise CacheError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ roles
def roles_index(roles):
    """{(dataset, run): (b6_input, b6_score, scope, set_label)} from a b6_roles.Roles-like object."""
    return {(r.dataset, r.run): (bool(r.b6_input), bool(r.b6_score), r.b6_input_scope, r.set_label)
            for r in roles.rows}


# ------------------------------------------------------------------------------------------ extraction
_MOL = """(SELECT species, source, member_id, any_value(molecule_id) AS molecule_id,
                  any_value(molecule_state) AS molecule_state, bool_or(train_tiles_excluded) AS train_tiles_excluded
           FROM molecule_member GROUP BY species, source, member_id)"""

_ALN = f"""SELECT a.alignment_id, a.species, a.source, a.dataset, a.run, a.library, a.role, a.chrom, a.start0, a.end0,
                  a.transcript_strand, a.arm, a.read_name, m.molecule_id, m.molecule_state, m.train_tiles_excluded
           FROM evidence_alignment a JOIN {_MOL} m
             ON m.species = a.species AND m.source = a.source AND m.member_id = a.member_id
           WHERE a.placement_kind = 'primary' AND a.accepted AND a.chrom IS NOT NULL"""


def check_molecule_states(con):
    """Molecule-level state consistency: a molecule is one unit everywhere (rule 3)."""
    bad = con.execute("""SELECT species, source, molecule_id FROM molecule_member
                         GROUP BY species, source, molecule_id
                         HAVING count(DISTINCT molecule_state) > 1 OR count(DISTINCT train_tiles_excluded) > 1
                         LIMIT 5""").fetchall()
    _require(not bad, f"molecule_member: molecules with inconsistent state across members: {bad}")


def check_species_coverage(con, species):
    """Refuse a DB holding rows of a species outside ``species`` in any table with a species column (a per-species
    build over the covered species would otherwise drop them silently, masks included)."""
    allowed = sorted(set(species))
    _require(allowed, "no covered species: an empty coverage would match no rows and hide every species")
    found = []
    for table in CS.ALL_TABLES:
        if table in ("c0_meta", "merge_manifest") or "species" not in dict(CS.table_columns(table)):
            continue
        marks = ", ".join("?" * len(allowed))
        extra = [r[0] for r in con.execute(f"SELECT DISTINCT species FROM {table} WHERE species NOT IN ({marks}) "
                                           "ORDER BY 1", allowed).fetchall()]
        if extra:
            found.append(f"{table}: {extra}")
    _require(not found, f"rows outside the covered species {allowed}: {'; '.join(found)}")


def species_chroms(con, species):
    """Sorted chromosomes holding rows of ``species`` in any extracted table (the per-chrom build's shard keys): accepted
    primary placements (the only alignments extraction reads, so a rejected- or secondary-only scaffold costs no pass),
    HSPs and exclusions; a NULL chrom outside evidence_alignment (where it marks an unplaced record) is refused."""
    found = set()
    for table, cond in (("evidence_alignment", " AND placement_kind = 'primary' AND accepted AND chrom IS NOT NULL"),
                        ("est_protein_hsp", ""),
                        ("excluded_junction_coordinate", ""), ("excluded_interval", "")):
        found.update(c for (c,) in con.execute(f"SELECT DISTINCT chrom FROM {table} WHERE species = ?{cond}",
                                               [species]).fetchall())
    _require(None not in found, f"{species}: rows with a NULL chrom outside evidence_alignment")
    return sorted(found)


def est_molecules(con, species=None):
    """{(species, molecule_id): (dataset, run, role, state, train_tiles_excluded, arm)} of EST molecules (all species
    or one); route-B rows are admitted under it.  A molecule with members in two (dataset, run, role) is refused."""
    params, sp_filter = ([], "") if species is None else ([species], " AND a.species = ?")
    est_mol = {}
    for sp, mid, ds, run, role, state, tte, arm, n_drr in con.execute(f"""
            SELECT a.species, m.molecule_id, min(a.dataset), min(a.run), min(a.role),
                   min(m.molecule_state), bool_or(m.train_tiles_excluded), min(a.arm),
                   count(DISTINCT (a.dataset, a.run, a.role))
            FROM evidence_alignment a JOIN {_MOL} m
              ON m.species = a.species AND m.source = a.source AND m.member_id = a.member_id
            WHERE a.source = 'est'{sp_filter} GROUP BY a.species, m.molecule_id""", params).fetchall():
        _require(n_drr == 1, f"EST molecule {mid} ({sp}) has members in more than one (dataset, run, role); its "
                             "route-B rows have no single role to be admitted under")
        est_mol[(sp, mid)] = (ds, run, role, state, tte, arm)
    return est_mol


def extract_rows(con, roles, family, *, species=None, chrom=None, check_states=True, est_mol=None, raw_drops=False):
    """Return (rows: list[dict], drops: {(species, source, cause): n_molecules}, libraries).

    ``libraries`` = {population: {species: sorted [(source, library), ...]}} over track sources (est/ont/pacbio)
    of admitted rows; it defines the per-source masks (plan §7.1).
    ``species``: every query is restricted to that species in SQL (the per-species build of b6_build_obs_cache);
    the rows are then exactly the one-shot rows of that species, in the same relative order.  ``chrom`` (with
    ``species``): likewise restricted to that chromosome (block/junction/chain rows by an alignment semi-join); EST
    molecules stay species-level (``est_mol``: a precomputed ``est_molecules(con, species)``).  ``raw_drops``: drops
    as {key: set of molecule ids}, to be unioned (not summed) across chromosomes.  ``check_states``: run the global
    molecule-state check (``check_molecule_states``); a per-species build runs it once, not per species.
    """
    _require(family in FAMILIES, f"family {family!r} not in {FAMILIES}")
    _require(chrom is None or species is not None, f"chrom {chrom!r} given without a species")
    if check_states:
        check_molecule_states(con)
    ridx = roles_index(roles)
    sp_filter = ""
    params = []
    if species is not None:
        sp_filter = " AND a.species = ?"
        params = [species]
    # per-table species clauses: empty (the unchanged one-shot SQL) unless ``species`` is given
    where_sp, and_sp, and_pc = ("", "", "") if species is None else (
        " WHERE species = ?", " AND species = ?", " AND pc.species = ? AND i.species = ?")
    pc_params = [] if species is None else [species, species]
    aln_params, hsp_params = list(params), list(params)
    if chrom is not None:                  # alignment-joined tables: semi-join on the species+chrom alignments
        sp_filter += " AND a.chrom = ?"
        aln_params.append(chrom)
        semi = " AND alignment_id IN (SELECT alignment_id FROM evidence_alignment WHERE species = ? AND chrom = ?)"
        where_sp, and_sp = where_sp + semi, and_sp + semi
        and_pc += semi.replace("alignment_id", "pc.alignment_id", 1)
        params = [species, species, chrom]
        pc_params = [species, species, species, chrom]
        hsp_params.append(chrom)
    drops = {}

    def drop(sp, src, cause, mid):
        drops.setdefault((sp, src, cause), set()).add(mid)

    def admit(sp, src, dataset, run, role, state, arm, mid):
        """-> (b6_input, b6_score, scope) or None (dropped, cause recorded)."""
        b_in, b_sc, scope, _label = ridx.get((dataset, run), (False, False, "-", ""))
        if arm != "primary":
            return drop(sp, src, "non_primary_arm", mid)
        if b_sc:
            return drop(sp, src, "b6_score", mid)
        if family == "eval":                     # eval-only C0: no leakage model, states carry no meaning
            if not b_in:
                return drop(sp, src, "eval_non_input", mid)
            return (b_in, b_sc, scope)
        if state == "never":
            return drop(sp, src, "never", mid)
        if role == "validation_only" and not b_in:
            return drop(sp, src, "validation_only_non_input", mid)
        if state == "excluded_heldout" and not b_in:
            return drop(sp, src, "excluded_heldout_non_input", mid)
        if src == "protein" and state != "train_track":
            return drop(sp, src, "protein_query_excluded", mid)
        return (b_in, b_sc, scope)

    rows = []
    aln = {}
    for (aid, sp, src, ds, run, lib, role, chrom_, s0, e0, tstrand, arm, read, mid, state, tte) in con.execute(
            _ALN + sp_filter + " ORDER BY a.species, a.source, a.alignment_id", aln_params).fetchall():
        ok = admit(sp, src, ds, run, role, state, arm, mid)
        if ok is None:
            continue
        b_in, b_sc, scope = ok
        base = dict(species=sp, chrom=chrom_, source=src, dataset=ds, run=run, library=lib, molecule_id=mid,
                    molecule_state=state, train_tiles_excluded=bool(tte), b6_input=b_in, b6_score=b_sc,
                    b6_input_scope=scope)
        organism = ""
        if src == "protein":
            organism = read.partition(":")[0]
            _require(bool(organism) and ":" in read, f"protein query id {read!r} is not '<organism>:<serial>'")
            base["query_id"] = read          # D7: thinning rank key (molecule_id stays the exclusion/QC unit)
        aln[aid] = (base, organism)
        if src != "protein":
            rows.append(dict(base, row_type="track", kind="alignment", strand=tstrand, phase=-1, start0=s0, end0=e0,
                             organism="", introns="", introns_ok="", units_key=""))
    if aln:                                  # blocks/junctions/chains only extend admitted alignments
        for aid, strand, s0, e0 in con.execute("SELECT alignment_id, strand, start0, end0 FROM aligned_block" + where_sp +
                                               " ORDER BY alignment_id, alignment_block_index", params).fetchall():
            if aid not in aln:
                continue
            base, organism = aln[aid]
            kind = "protein" if base["source"] == "protein" else "block"
            rows.append(dict(base, row_type="track", kind=kind, strand=strand, phase=-1, start0=s0, end0=e0,
                             organism=organism, introns="", introns_ok="", units_key=""))
        for aid, strand, s0, e0 in con.execute(
                "SELECT alignment_id, strand, start0, end0 FROM junction_observation WHERE accepted AND strand IN ('+','-')"
                " AND start0 IS NOT NULL" + and_sp + " ORDER BY alignment_id, intron_index", params).fetchall():
            if aid not in aln or aln[aid][0]["source"] == "protein":
                continue
            rows.append(dict(aln[aid][0], row_type="track", kind="junction", strand=strand, phase=-1, start0=s0, end0=e0,
                             organism="", introns="", introns_ok="", units_key=""))
        # chains: accepted primary placement, strand +/-, >= 1 intron; introns in transcription order
        chains = con.execute(f"""
            SELECT pc.chain_id, pc.alignment_id, pc.strand, pc.start0, pc.end0,
                   string_agg(CAST(i.donor0 AS VARCHAR) || '-' || CAST(i.acceptor0 AS VARCHAR), ',' ORDER BY i.tx_order),
                   string_agg(CASE WHEN i.accepted AND i.donor0 IS NOT NULL THEN '1' ELSE '0' END, ',' ORDER BY i.tx_order),
                   count(*), count(i.donor0)
            FROM partial_chain pc JOIN partial_chain_intron i ON i.chain_id = pc.chain_id
            WHERE pc.strand IN ('+', '-') AND pc.n_introns >= 1{and_pc}
            GROUP BY pc.chain_id, pc.alignment_id, pc.strand, pc.start0, pc.end0
            ORDER BY pc.chain_id""", pc_params).fetchall()
        for cid, aid, strand, s0, e0, introns, oks, n, n_coord in chains:
            if aid not in aln or aln[aid][0]["source"] == "protein":
                continue
            _require(n == n_coord, f"chain {cid}: strand {strand} chain has an intron without donor0/acceptor0")
            base = aln[aid][0]
            rows.append(dict(base, row_type="chain", kind="chain", strand=strand, phase=-1, start0=s0, end0=e0,
                             organism="", introns=introns, introns_ok=oks,
                             units_key=f"{base['source']}\t{strand}\t{introns}"))
    # route-B phase rows, keyed to their EST molecule (dataset/run/state of the EST molecule)
    # One (dataset, run, role) per EST molecule, or refuse: the representative decides route-B admission, so it is
    # never picked arbitrarily (any_value) when a molecule's members disagree (none do in the B6 EST C0s, 09-30).
    # (species-level even under ``chrom``: a molecule's EST members may align on another chromosome)
    if est_mol is None:
        est_mol = est_molecules(con, species)
    hsp_filter = "" if species is None else " WHERE species = ?" + ("" if chrom is None else " AND chrom = ?")
    for sp, mid, lib, chrom_, g0, g1, strand, phase, lexc, ttexc in con.execute(
            "SELECT species, molecule_id, library, chrom, genomic_start0, genomic_end0, strand, phase, "
            "leakage_excluded, train_tiles_excluded FROM est_protein_hsp" + hsp_filter +
            " ORDER BY species, hsp_id, interval_index", hsp_params).fetchall():
        _require((sp, mid) in est_mol, f"est_protein_hsp molecule {mid} has no EST molecule in {sp}")
        ds, run, role, state, tte, arm = est_mol[(sp, mid)]
        ok = admit(sp, "est", ds, run, role, state, arm, mid)
        if ok is None:
            continue
        if family == "train" and state == "train_track" and lexc:
            drop(sp, "est", "phase_hsp_leakage_excluded", mid)
            continue
        b_in, b_sc, scope = ok
        rows.append(dict(species=sp, chrom=chrom_, source="est", dataset=ds, run=run, library=lib, molecule_id=mid,
                         molecule_state=state, train_tiles_excluded=bool(tte or ttexc), b6_input=b_in, b6_score=b_sc,
                         b6_input_scope=scope, row_type="track", kind="phase", strand=strand, phase=int(phase),
                         start0=g0, end0=g1, organism="", introns="", introns_ok="", units_key=""))
    if family == "train":
        xwhere, xparams = hsp_filter, hsp_params
        for sp, chrom_, strand, kind, coord in con.execute(
                "SELECT DISTINCT species, chrom, strand, kind, coordinate0 FROM excluded_junction_coordinate" +
                xwhere + " ORDER BY 1, 2, 3, 4, 5", xparams).fetchall():
            rows.append(dict(species=sp, chrom=chrom_, source="", dataset="", run="", library="", molecule_id="",
                             molecule_state="", train_tiles_excluded=False, b6_input=False, b6_score=False,
                             b6_input_scope="-", row_type="xjunc", kind=kind, strand=strand, phase=-1,
                             start0=coord, end0=coord + 1, organism="", introns="", introns_ok="", units_key=""))
        for sp, chrom_, s0, e0, cause, scope, gene, tile_rn in con.execute(
                "SELECT DISTINCT species, chrom, start0, end0, cause, scope, coalesce(gene_id, ''), "
                "coalesce(tile_rn, -1) FROM excluded_interval" + xwhere + " ORDER BY 1, 2, 3, 4, 5, 6, 7, 8",
                xparams).fetchall():
            rows.append(dict(species=sp, chrom=chrom_, source="", dataset="", run="", library="", molecule_id="",
                             molecule_state="", train_tiles_excluded=False, b6_input=False, b6_score=False,
                             b6_input_scope="-", row_type="xint", kind=cause, strand=scope, phase=-1,
                             start0=s0, end0=e0, organism="", introns="", introns_ok="", units_key=gene,
                             tile_rn=int(tile_rn)))
    libraries = population_libraries(rows, family)
    return rows, {k: (v if raw_drops else len(v)) for k, v in sorted(drops.items())}, libraries


def _pop_ok(population, r):
    if r["row_type"] == "xjunc" or r["source"] not in TS.TRACK_SOURCES:
        return False
    if population == "train_track":
        return r["molecule_state"] == "train_track" and not r["train_tiles_excluded"]
    if population == "valid_track":
        return r["molecule_state"] == "train_track"
    return r["b6_input"] and not r["b6_score"]


def population_libraries(rows, family):
    pops = ("eval_input",) if family == "eval" else POPULATIONS
    out = {}
    for pop in pops:
        per = {}
        for r in rows:
            if r["kind"] in ("alignment",) and _pop_ok(pop, r) and (pop != "eval_input" or family == "eval"
                                                                    or r["molecule_state"] == "excluded_heldout"):
                per.setdefault(r["species"], set()).add((r["source"], r["library"]))
        out[pop] = {sp: sorted(v) for sp, v in sorted(per.items())}
    return out


# ------------------------------------------------------------------------------------------------ writer
def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CacheWriter:
    """Incremental ``write_cache``: ``add_rows`` writes the shards of the (species, chrom) keys in ``rows`` (each key
    once, so a key's rows must come in one call), ``finish`` writes the zero lists and the manifest.  Output bytes equal
    ``write_cache`` over the concatenated rows: shard files depend only on their own rows (stable per-shard sort) and
    the manifest lists shards sorted by (species, chrom) whatever the order of the ``add_rows`` calls."""

    def __init__(self, out_dir, *, family):
        out = Path(out_dir)
        _require(not out.exists() or not any(out.iterdir()), f"{out} exists and is not empty; caches are immutable")
        out.mkdir(parents=True, exist_ok=True)
        _require(family in FAMILIES, f"bad family {family!r}")
        self.out, self.family = out, family
        self._shards = {}                  # (species, chrom) -> shard manifest entry
        self._finished = False

    def add_rows(self, rows):
        """Write one parquet shard per (species, chrom) of ``rows``; a key already written is refused."""
        _require(not self._finished, f"{self.out}: cache already finished")
        out = self.out
        shards = {}
        for r in rows:
            shards.setdefault((r["species"], r["chrom"]), []).append(r)
        again = sorted(set(shards) & set(self._shards))
        _require(not again, f"shards {again} already written; a (species, chrom) shard is written once")
        for (sp, chrom), rs in sorted(shards.items()):
            rs.sort(key=lambda r: (r["row_type"], r["start0"], r["end0"], r["kind"], r["source"], r["molecule_id"],
                                   r["strand"], r["phase"], r["introns"]))
            table = pa.Table.from_pylist([{c: r.get(c, -1 if c == "tile_rn" else ("" if c == "query_id" else None))
                                           for c in COLUMNS} for r in rs], schema=SCHEMA)
            name = f"shard.{sp}.{chrom}.parquet"
            _require("/" not in sp and "/" not in chrom, f"bad shard key {(sp, chrom)}")
            pq.write_table(table, out / name, compression="zstd", use_dictionary=True, write_statistics=False)
            self._shards[(sp, chrom)] = {"species": sp, "chrom": chrom, "file": name, "rows": len(rs),
                                         "sha256": sha256_file(out / name)}

    def finish(self, *, libraries, drops, bindings, zero_lists=None):
        """Write zero lists + manifest.json (+ sidecar); returns the manifest dict."""
        _require(not self._finished, f"{self.out}: cache already finished")
        self._finished = True
        out = self.out
        shard_meta = [self._shards[k] for k in sorted(self._shards)]
        zero_meta = {}
        for pop, per in sorted((zero_lists or {}).items()):
            _require(pop in POPULATIONS, f"zero list population {pop!r}")
            for sp, (tsv, qc) in sorted(per.items()):
                qc_obj = json.loads(Path(qc).read_text())
                _require(qc_obj.get("population") == pop, f"zero-list QC {qc} names population "
                                                          f"{qc_obj.get('population')!r}, expected {pop!r}")
                dst_t, dst_q = out / f"zero.{pop}.{sp}.tsv", out / f"zero.{pop}.{sp}.qc.json"
                dst_t.write_bytes(Path(tsv).read_bytes())
                dst_q.write_bytes(Path(qc).read_bytes())
                zero_meta.setdefault(pop, {})[sp] = {"tsv": dst_t.name, "tsv_sha256": sha256_file(dst_t),
                                                     "qc": dst_q.name, "qc_sha256": sha256_file(dst_q),
                                                     "upstream": qc_obj.get("upstream", {})}
        manifest = {"cache_version": CACHE_VERSION, "family": self.family, "spec": dict(TS.SPEC_SHAS),
                    "shards": shard_meta, "zero_lists": zero_meta, "libraries": libraries,
                    "drops": [[sp, src, cause, n] for (sp, src, cause), n in sorted(drops.items())],
                    "bindings": dict(sorted(bindings.items()))}
        (out / "manifest.json").write_text(_canon(manifest) + "\n")
        (out / "manifest.json.sha256").write_text(f"{sha256_file(out / 'manifest.json')}  manifest.json\n")
        return manifest


def write_cache(out_dir, rows, *, family, libraries, drops, bindings, zero_lists=None):
    """Write shards + zero lists + manifest.json; returns the manifest dict (one-shot ``CacheWriter``).

    ``bindings``: {name: sha256} of the upstream artefacts (evidence, corpus/reference, roles, code, ...).
    ``zero_lists``: {population: {species: (tsv_path, qc_json_path)}} copied into the cache and hashed.
    """
    writer = CacheWriter(out_dir, family=family)
    writer.add_rows(rows)
    return writer.finish(libraries=libraries, drops=drops, bindings=bindings, zero_lists=zero_lists)


# ------------------------------------------------------------------------------------------------ loader
# Shards are read through a derived, uncompressed Arrow IPC file per shard (``<mmap_dir>/<shard sha256>.arrow``),
# built once from the sha-verified parquet and memory-mapped zero-copy: its pages live in the page cache, shared by
# every process on the node and reclaimable, instead of one private object-array copy per process.  The cache
# directory itself is never written; the derived files are a disposable runtime sidecar keyed by the verified shard
# sha (and MMAP_VERSION), so they cannot go stale.
MMAP_VERSION = "b6-obs-cache-mmap-v1"
MMAP_DIR_ENV = "B6_OBS_CACHE_MMAP_DIR"
MAX_SHARDS_ENV = "B6_OBS_CACHE_MAX_SHARDS"
REQUIRE_PREBUILT_ENV = "B6_OBS_CACHE_REQUIRE_PREBUILT"
DEFAULT_MAX_SHARDS = 512       # open shards kept per process: above the 455 train_v1 shards, so steady state never evicts
TMP_SWEEP_AGE_S = 3600         # prebuild removes abandoned ``.<name>.*.tmp`` files older than this
DICT_VOCAB_MAX = 1 << 16       # a string column with at most this many distinct values keeps a Python vocab
_ORDER, _ORDER_START = "_order", "_order_start"   # track row indices sorted by start0, then chain rows; padded -1


def default_mmap_dir(cache_path):
    """``<cache parent>/<cache name>.mmap_v1``: next to the cache directory, never inside it."""
    p = Path(os.path.abspath(cache_path))
    return p.parent / f"{p.name}.mmap_v1"


class StrColumn:
    """A shard string column without one Python str per row: integer codes into the column's distinct values.

    ``col[i]`` -> the value (str or None) and ``col[idx]`` (integer or boolean array) -> an object ndarray, equal to
    indexing ``ChunkedArray.to_numpy(zero_copy_only=False)``.  Up to ``DICT_VOCAB_MAX`` distinct values the vocab is
    an object ndarray, decoded on first use; beyond that it stays an Arrow array and only the selected rows are
    decoded."""

    __slots__ = ("codes", "_dictionary", "_vocab")

    def __init__(self, codes, dictionary):
        self.codes = codes
        self._dictionary = dictionary
        self._vocab = None

    @property
    def _arrow(self):
        return self._dictionary if len(self._dictionary) > DICT_VOCAB_MAX else None

    @property
    def vocab(self):
        """Object ndarray of the distinct values (None for a large vocabulary, which stays in Arrow)."""
        if self._vocab is None and len(self._dictionary) <= DICT_VOCAB_MAX:
            self._vocab = self._dictionary.to_numpy(zero_copy_only=False)
        return self._vocab

    def __len__(self):
        return len(self.codes)

    def mask(self, value):
        """Boolean mask of the rows equal to ``value`` (whole column, decoded through the vocab only)."""
        arrow = self._arrow
        hit = np.nonzero(self.vocab == value)[0] if arrow is None else \
            np.nonzero(pc.equal(arrow, value).fill_null(False).to_numpy(zero_copy_only=False))[0]
        return np.isin(self.codes, hit) if len(hit) else np.zeros(len(self.codes), bool)

    def __getitem__(self, key):
        arrow = self._arrow
        if isinstance(key, (int, np.integer)):
            code = int(self.codes[key])
            return self.vocab[code] if arrow is None else arrow[code].as_py()
        codes = self.codes[key]
        if arrow is None:
            return self.vocab[codes]
        return arrow.take(pa.array(codes.astype(np.int64, copy=False))).to_numpy(zero_copy_only=False)


@dataclass
class Shard:
    cols: dict                 # column -> numpy array (fixed width, mmap-backed) or StrColumn (strings)
    track_idx: np.ndarray      # row indices of track rows sorted by start0
    track_start: np.ndarray
    track_maxlen: int
    chain_idx: np.ndarray
    chain_start: np.ndarray
    chain_maxlen: int
    _xjunc: list | None = None     # [(strand, kind, coord)]; None = built from cols on first use
    _xint: list | None = None      # [(start0, end0, cause, scope, tile_rn)]; None = built from cols on first use

    @property
    def xjunc(self):
        if self._xjunc is None:
            cols, start = self.cols, self.cols["start0"]
            self._xjunc = [(cols["strand"][i], cols["kind"][i], int(start[i]))
                           for i in np.nonzero(cols["row_type"].mask("xjunc"))[0]]
        return self._xjunc

    @property
    def xint(self):
        if self._xint is None:
            cols, start, end = self.cols, self.cols["start0"], self.cols["end0"]
            self._xint = [(int(start[i]), int(end[i]), cols["kind"][i], cols["strand"][i], int(cols["tile_rn"][i]))
                          for i in np.nonzero(cols["row_type"].mask("xint"))[0]]
        return self._xint


def _env_limit(name, value, default):
    if value is None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        _require(raw.isdecimal(), f"{name}={raw!r}: expected a positive integer")
        value = int(raw)
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 1,
             f"{name} / max_shards = {value!r}: expected a positive integer")
    return value


def _env_flag(name, value):
    if value is not None:
        return bool(value)
    raw = os.environ.get(name, "").strip()
    _require(raw in ("", "0", "1"), f"{name}={raw!r}: expected 0 or 1")
    return raw == "1"


def derive_table(f, source_sha256):
    """Parquet shard -> the derived table: strings as dictionary<int8|16|32, large_string> (nulls encoded), numeric
    columns as the exact numpy arrays the reader exposes (bool stored as uint8), plus the start-sorted track/chain
    index columns.  Schema metadata binds it to ``source_sha256`` and ``MMAP_VERSION``."""
    t = pq.read_table(f)
    _require(tuple(t.schema.names) == COLUMNS, f"{f}: columns {t.schema.names} != {COLUMNS}")
    n = t.num_rows
    fields, arrays, np_cols = [], [], {}
    for c in COLUMNS:
        col = t.column(c)
        if pa.types.is_string(col.type):
            enc = col.dictionary_encode(null_encoding="encode").chunks
            dictionary = enc[0].dictionary if enc else pa.array([], pa.string())
            _require(all(ch.dictionary.equals(dictionary) for ch in enc[1:]), f"{f}: {c} chunk dictionaries differ")
            codes = np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in enc]) if enc \
                else np.zeros(0, np.int32)
            width = np.int8 if len(dictionary) <= 1 << 7 else np.int16 if len(dictionary) <= 1 << 15 else np.int32
            arr = pa.DictionaryArray.from_arrays(pa.array(codes.astype(width, copy=False)),
                                                 dictionary.cast(pa.large_string()))
            kind = "str"
            if c == "row_type":
                rt = StrColumn(codes, dictionary)
        else:
            a = col.to_numpy(zero_copy_only=False)
            if c in ("start0", "end0"):
                a = a.astype(np.int64, copy=False)
            np_cols[c] = a
            if a.dtype == object:                      # a boolean column with nulls: converted when opened
                arr, kind = col.combine_chunks(), "object"
            elif a.dtype == bool:
                arr, kind = pa.array(a.view(np.uint8)), "bool"
            else:
                arr, kind = pa.array(a), a.dtype.str
        fields.append(pa.field(c, arr.type, metadata={"np": kind}))
        arrays.append(arr)
    start, end = np_cols["start0"], np_cols["end0"]

    def index(mask):
        idx = np.nonzero(mask)[0]
        idx = idx[np.argsort(start[idx], kind="stable")]
        ml = int((end[idx] - start[idx]).max()) if len(idx) else 0
        return idx, start[idx], ml

    ti, ts, tml = index(rt.mask("track"))
    ci, cs, cml = index(rt.mask("chain"))
    del rt, t
    pad = n - len(ti) - len(ci)
    order = np.concatenate([ti, ci, np.full(pad, -1, np.int64)]).astype(np.int64)
    order_start = np.concatenate([ts, cs, np.zeros(pad, np.int64)]).astype(np.int64)
    for name, a in ((_ORDER, order), (_ORDER_START, order_start)):
        fields.append(pa.field(name, pa.int64(), metadata={"np": "<i8"}))
        arrays.append(pa.array(a))
    meta = {"mmap_version": MMAP_VERSION, "source_sha256": source_sha256, "rows": str(n), "n_track": str(len(ti)),
            "n_chain": str(len(ci)), "track_maxlen": str(tml), "chain_maxlen": str(cml)}
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=meta))


def _check_derived(table, path, shard_meta):
    m = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    _require(m.get("mmap_version") == MMAP_VERSION and m.get("source_sha256") == shard_meta["sha256"]
             and m.get("rows") == str(shard_meta["rows"]) and table.num_rows == shard_meta["rows"]
             and tuple(table.schema.names) == COLUMNS + (_ORDER, _ORDER_START),
             f"{path}: derived shard does not match manifest shard {shard_meta['file']} "
             f"(sha {shard_meta['sha256']}, {shard_meta['rows']} rows, {MMAP_VERSION}); delete it and rerun the prebuild")
    return m


def _single(col):
    if col.num_chunks == 1:
        return col.chunk(0)
    if pa.types.is_dictionary(col.type):
        _require(col.num_chunks == 0, "derived shard: dictionary column in several chunks")
        return pa.DictionaryArray.from_arrays(pa.array([], pa.int32()), pa.array([], pa.large_string()))
    return col.combine_chunks()


def shard_from_table(table, path, shard_meta):
    """Shard over a derived table (memory-mapped or in memory).  Numeric columns and the index arrays are zero-copy
    numpy views.  Nothing is decoded at open: each string vocabulary (<= DICT_VOCAB_MAX values) on its first use,
    xjunc/xint on the first ``excluded_junctions`` / ``excluded_intervals`` call."""
    m = _check_derived(table, path, shard_meta)
    cols = {}
    for c, field in zip(COLUMNS, table.schema):
        a, kind = _single(table.column(c)), field.metadata[b"np"].decode()
        if kind == "str":
            cols[c] = StrColumn(a.indices.to_numpy(zero_copy_only=True), a.dictionary)
        elif kind == "object":
            cols[c] = a.to_numpy(zero_copy_only=False)
        elif kind == "bool":
            cols[c] = a.to_numpy(zero_copy_only=True).view(np.bool_)
        else:
            cols[c] = a.to_numpy(zero_copy_only=True)
            _require(cols[c].dtype.str == kind, f"{path}: column {c} dtype {cols[c].dtype.str} != {kind}")
    order = _single(table.column(_ORDER)).to_numpy(zero_copy_only=True)
    order_start = _single(table.column(_ORDER_START)).to_numpy(zero_copy_only=True)
    nt, nc = int(m["n_track"]), int(m["n_chain"])
    return Shard(cols, order[:nt], order_start[:nt], int(m["track_maxlen"]), order[nt:nt + nc],
                 order_start[nt:nt + nc], int(m["chain_maxlen"]))


def open_derived(path):
    """Memory-map a derived shard file (zero-copy); None if it does not exist.  An unreadable, truncated or malformed
    file is a CacheError (only the footer, schema and metadata are checked here; the content hash is checked by
    ``prebuild_mmap`` against ``<path>.sha256``)."""
    try:
        return pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()
    except FileNotFoundError:
        return None
    except (pa.ArrowException, OSError, ValueError) as e:
        raise CacheError(f"{path}: unreadable derived shard ({e}); delete it and rerun the prebuild") from e


def write_derived(path, table):
    """Atomic write (tmp + fsync + rename).  Concurrent writers each write their own tmp file; the last rename wins
    and every candidate holds the same derived content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    try:
        with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, table.schema) as w:
            w.write_table(table)
        _fsync(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync(path.parent)


def _fsync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sidecar(path):
    return Path(f"{path}.sha256")


def write_sidecar(path):
    """``<path>.sha256`` = sha256 of the derived file (tmp + fsync + rename + directory fsync)."""
    side = _sidecar(path)
    tmp = side.parent / f".{side.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    try:
        tmp.write_text(f"{sha256_file(path)}  {Path(path).name}\n")
        _fsync(tmp)
        os.replace(tmp, side)
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync(side.parent)


def build_derived(f, shard_meta, path):
    """Write the derived file of parquet ``f`` (its sha256 already verified against ``shard_meta``).  An OSError
    (unwritable directory, ENOSPC, ...) is a CacheError."""
    try:
        write_derived(path, derive_table(f, shard_meta["sha256"]))
    except OSError as e:
        raise CacheError(f"cannot write derived shard {path}: {e}") from e
    finally:
        pa.default_memory_pool().release_unused()


class _BuildLock:
    """Advisory flock on ``<derived>.lock`` so one process builds a shard while the others wait for it.  Where the
    filesystem has no flock (some Lustre/NFS mounts) it is a no-op and the atomic rename keeps concurrent builds safe."""

    def __init__(self, path):
        self.path = Path(f"{path}.lock")
        self.fd = None

    def __enter__(self):
        import fcntl
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            raise CacheError(f"cannot create derived shard dir / lock {self.path}: {e}") from e
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        except OSError:
            pass
        return self

    def __exit__(self, *exc):
        os.close(self.fd)          # releases the flock


class ObsCache:
    """Read-only cache.  Verifies manifest sidecar, spec shas and (lazily, on first use) each shard's sha256.

    A shard is served from its derived memory-mapped file ``<mmap_dir>/<shard sha256>.arrow`` (``mmap_dir``: argument,
    env ``B6_OBS_CACHE_MMAP_DIR``, default ``default_mmap_dir(path)``).  A missing derived file is built from the parquet
    after its sha256 is verified against the manifest (under ``_BuildLock``); an existing one is accepted when its
    recorded source sha256, row count and MMAP_VERSION match the manifest entry.  ``verify_shards=False`` never writes
    a derived file: an unverified parquet is converted in memory.  Open shards are kept in an LRU of ``max_shards``
    (env ``B6_OBS_CACHE_MAX_SHARDS``, default 512); the most recently used shard is always kept.

    ``require_prebuilt`` (env ``B6_OBS_CACHE_REQUIRE_PREBUILT=1``; the production setting): nothing is built in the
    process; a missing derived file or a missing ``<derived>.sha256`` sidecar (written by ``prebuild_mmap``) is a
    CacheError.  The derived file's content hash is checked by ``prebuild_mmap`` only, not per process; opening checks
    the footer, schema and the recorded source sha256 / rows / MMAP_VERSION, not the file size or content.  The
    sidecar is written after its derived file, so a crash between the two leaves a derived file without a sidecar
    (rebuilt by the next prebuild) but a derived file replaced after its sidecar is not noticed until a prebuild
    re-hashes it: rerun the prebuild check right before each training submission."""

    def __init__(self, path, *, verify_shards=True, max_shards=None, mmap_dir=None, require_prebuilt=None):
        self.path = Path(path)
        mf = self.path / "manifest.json"
        side = self.path / "manifest.json.sha256"
        _require(mf.is_file() and side.is_file(), f"{self.path}: manifest or sidecar missing")
        self.manifest_sha256 = sha256_file(mf)
        _require(side.read_text().split()[0] == self.manifest_sha256, f"{mf}: sha256 != sidecar")
        self.manifest = json.loads(mf.read_text())
        _require(self.manifest.get("cache_version") == CACHE_VERSION, "cache version mismatch")
        _require(self.manifest.get("spec") == TS.SPEC_SHAS,
                 f"cache built under spec {self.manifest.get('spec')} != current {TS.SPEC_SHAS}")
        self.family = self.manifest["family"]
        self._shards = OrderedDict()       # (species, chrom) -> Shard, least recently used first
        self._empty = {}                   # keys without a shard (not counted by the LRU)
        self.max_shards = _env_limit(MAX_SHARDS_ENV, max_shards, DEFAULT_MAX_SHARDS)
        env_dir = os.environ.get(MMAP_DIR_ENV, "").strip()
        self.mmap_dir = Path(mmap_dir) if mmap_dir is not None else Path(env_dir) if env_dir else \
            default_mmap_dir(self.path)
        self._meta = {(s["species"], s["chrom"]): s for s in self.manifest["shards"]}
        self.verify_shards = verify_shards
        self.require_prebuilt = _env_flag(REQUIRE_PREBUILT_ENV, require_prebuilt)
        self._zero = {}

    # -- zero lists / libraries
    def owner_libraries(self, population, species):
        per = self.manifest["libraries"].get(population)
        _require(per is not None, f"cache family {self.family} has no population {population}")
        return [tuple(p) for p in per.get(species, [])]

    def zero_intervals(self, population, species, chrom, ws, we):
        """A48 zero-list intervals clipped to [ws, we), genomic coordinates.  A missing list fails closed."""
        key = (population, species)
        if key not in self._zero:
            meta = self.manifest["zero_lists"].get(population, {}).get(species)
            _require(meta is not None, f"no zero list for population {population}, species {species} in cache")
            t, q = self.path / meta["tsv"], self.path / meta["qc"]
            _require(sha256_file(t) == meta["tsv_sha256"] and sha256_file(q) == meta["qc_sha256"],
                     f"zero list {t} or its QC changed since the cache was built")
            by = {}
            lines = t.read_text().splitlines()
            _require(bool(lines) and lines[0].split("\t")[:4] == ["species", "chrom", "start0", "end0"],
                     f"{t}: bad header")
            for line in lines[1:]:
                f = line.split("\t")
                _require(f[0] == species, f"{t}: species {f[0]!r} != {species!r}")
                by.setdefault(f[1], []).append((int(f[2]), int(f[3])))
            self._zero[key] = {c: sorted(v) for c, v in by.items()}
        return [(max(s, ws), min(e, we)) for s, e in self._zero[key].get(chrom, []) if s < we and e > ws]

    # -- shards
    def derived_path(self, species, chrom):
        return self.mmap_dir / f"{self._meta[(species, chrom)]['sha256']}.arrow"

    def shard(self, species, chrom):
        key = (species, chrom)
        sh = self._shards.get(key)
        if sh is not None:
            self._shards.move_to_end(key)
            return sh
        meta = self._meta.get(key)
        if meta is None:
            if key not in self._empty:
                empty = np.zeros(0, np.int64)
                self._empty[key] = Shard({c: np.zeros(0, object) for c in COLUMNS}, empty, empty, 0, empty, empty, 0,
                                         [], [])
            return self._empty[key]
        f, derived = self.path / meta["file"], self.derived_path(*key)
        if self.require_prebuilt:
            _require(derived.is_file() and _sidecar(derived).is_file(),
                     f"derived shard {derived} or its .sha256 sidecar is missing (require_prebuilt): run "
                     f"b6_prebuild_obs_mmap.py on {self.path} with mmap dir {self.mmap_dir}")
            sh = shard_from_table(open_derived(derived), derived, meta)
        elif not self.verify_shards:
            table = open_derived(derived)
            sh = shard_from_table(table if table is not None else derive_table(f, meta["sha256"]), derived, meta)
        else:
            table = open_derived(derived)
            if table is None:
                with _BuildLock(derived):
                    table = open_derived(derived)          # built by another process meanwhile
                    if table is None:
                        _require(sha256_file(f) == meta["sha256"], f"shard {f} sha256 changed since the cache was built")
                        build_derived(f, meta, derived)
                        table = open_derived(derived)
            sh = shard_from_table(table, derived, meta)
        self._shards[key] = sh
        while len(self._shards) > self.max_shards:
            self._shards.popitem(last=False)
        return sh

    @staticmethod
    def _overlap(idx, starts, maxlen, end, ws, we):
        lo = np.searchsorted(starts, ws - maxlen, side="left")
        hi = np.searchsorted(starts, we, side="left")
        cand = idx[lo:hi]
        return cand[end[cand] > ws]

    def track_rows(self, species, chrom, ws, we):
        sh = self.shard(species, chrom)
        return sh, self._overlap(sh.track_idx, sh.track_start, sh.track_maxlen, sh.cols["end0"], ws, we)

    def chain_rows(self, species, chrom, ws, we):
        sh = self.shard(species, chrom)
        return sh, self._overlap(sh.chain_idx, sh.chain_start, sh.chain_maxlen, sh.cols["end0"], ws, we)

    def excluded_junctions(self, species, chrom):
        return frozenset(self.shard(species, chrom).xjunc)

    def excluded_intervals(self, species, chrom):
        return list(self.shard(species, chrom).xint)


_TMP_NAME = re.compile(r"\.[0-9a-f]{64}\.arrow(\.sha256)?\.[0-9]+\.[0-9a-f]{8}\.tmp")   # write_derived / write_sidecar


def _sweep_tmp(mmap_dir, now=None):
    """Remove abandoned write_derived / write_sidecar tmp files (``.<sha>.arrow[.sha256].<pid>.<rand>.tmp``) older than
    TMP_SWEEP_AGE_S; other files are left alone.  Returns the removed names."""
    now = time.time() if now is None else now
    gone = []
    for q in Path(mmap_dir).glob(".*.tmp") if Path(mmap_dir).is_dir() else ():
        if not _TMP_NAME.fullmatch(q.name):
            continue
        try:
            if now - q.stat().st_mtime > TMP_SWEEP_AGE_S:
                q.unlink()
                gone.append(q.name)
        except FileNotFoundError:
            continue
    return sorted(gone)


def prebuild_mmap(cache_path, *, mmap_dir=None, log=None):
    """Build, or check, the derived file of every manifest shard of a cache.  Run serially, once, before training.

    Abandoned tmp files older than an hour are removed first.  Per shard, the parquet sha256 is verified against the
    manifest (hashed once).  A missing derived file, or one without a ``.sha256`` sidecar, is (re)built from that
    verified parquet and its sidecar written; an existing one is re-hashed against its sidecar and its recorded source
    sha256 / rows / MMAP_VERSION checked.  Returns one record per shard."""
    cache = ObsCache(cache_path, max_shards=1, mmap_dir=mmap_dir, require_prebuilt=False)
    swept = _sweep_tmp(cache.mmap_dir)
    out = []
    for meta in cache.manifest["shards"]:
        key = (meta["species"], meta["chrom"])
        f, derived = cache.path / meta["file"], cache.derived_path(*key)
        _require(sha256_file(f) == meta["sha256"], f"shard {f} sha256 changed since the cache was built")
        side = _sidecar(derived)
        build = not (derived.is_file() and side.is_file())
        if build:
            build_derived(f, meta, derived)
            try:
                write_sidecar(derived)
            except OSError as e:
                raise CacheError(f"cannot write {side}: {e}") from e
        else:
            _require(side.read_text().split()[:1] == [sha256_file(derived)],
                     f"{derived}: sha256 != {side.name}; delete both and rerun the prebuild")
        _check_derived(open_derived(derived), derived, meta)
        rec = {"species": key[0], "chrom": key[1], "rows": meta["rows"], "source_sha256": meta["sha256"],
               "derived": str(derived), "derived_bytes": derived.stat().st_size, "built": build}
        out.append(rec)
        if log is not None:
            log(rec)
    if swept and log is not None:
        log({"swept_tmp": swept})
    return out
