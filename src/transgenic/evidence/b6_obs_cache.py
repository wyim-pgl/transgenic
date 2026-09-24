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
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transgenic.evidence import b6_track_spec as TS

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


def _mol_states(con):
    """Molecule-level state consistency: a molecule is one unit everywhere (rule 3)."""
    bad = con.execute("""SELECT species, source, molecule_id FROM molecule_member
                         GROUP BY species, source, molecule_id
                         HAVING count(DISTINCT molecule_state) > 1 OR count(DISTINCT train_tiles_excluded) > 1
                         LIMIT 5""").fetchall()
    _require(not bad, f"molecule_member: molecules with inconsistent state across members: {bad}")


def extract_rows(con, roles, family, *, species=None):
    """Return (rows: list[dict], drops: {(species, source, cause): n_molecules}, libraries).

    ``libraries`` = {population: {species: sorted [(source, library), ...]}} over track sources (est/ont/pacbio)
    of admitted rows; it defines the per-source masks (plan §7.1).
    """
    _require(family in FAMILIES, f"family {family!r} not in {FAMILIES}")
    _mol_states(con)
    ridx = roles_index(roles)
    sp_filter = ""
    params = []
    if species is not None:
        sp_filter = " AND a.species = ?"
        params = [species]
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
    for (aid, sp, src, ds, run, lib, role, chrom, s0, e0, tstrand, arm, read, mid, state, tte) in con.execute(
            _ALN + sp_filter + " ORDER BY a.species, a.source, a.alignment_id", params).fetchall():
        ok = admit(sp, src, ds, run, role, state, arm, mid)
        if ok is None:
            continue
        b_in, b_sc, scope = ok
        base = dict(species=sp, chrom=chrom, source=src, dataset=ds, run=run, library=lib, molecule_id=mid,
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
    for aid, strand, s0, e0 in con.execute("SELECT alignment_id, strand, start0, end0 FROM aligned_block "
                                           "ORDER BY alignment_id, alignment_block_index").fetchall():
        if aid not in aln:
            continue
        base, organism = aln[aid]
        kind = "protein" if base["source"] == "protein" else "block"
        rows.append(dict(base, row_type="track", kind=kind, strand=strand, phase=-1, start0=s0, end0=e0,
                         organism=organism, introns="", introns_ok="", units_key=""))
    for aid, strand, s0, e0 in con.execute(
            "SELECT alignment_id, strand, start0, end0 FROM junction_observation WHERE accepted AND strand IN ('+','-')"
            " AND start0 IS NOT NULL ORDER BY alignment_id, intron_index").fetchall():
        if aid not in aln or aln[aid][0]["source"] == "protein":
            continue
        rows.append(dict(aln[aid][0], row_type="track", kind="junction", strand=strand, phase=-1, start0=s0, end0=e0,
                         organism="", introns="", introns_ok="", units_key=""))
    # chains: accepted primary placement, strand +/-, >= 1 intron; introns in transcription order
    chains = con.execute("""
        SELECT pc.chain_id, pc.alignment_id, pc.strand, pc.start0, pc.end0,
               string_agg(CAST(i.donor0 AS VARCHAR) || '-' || CAST(i.acceptor0 AS VARCHAR), ',' ORDER BY i.tx_order),
               string_agg(CASE WHEN i.accepted AND i.donor0 IS NOT NULL THEN '1' ELSE '0' END, ',' ORDER BY i.tx_order),
               count(*), count(i.donor0)
        FROM partial_chain pc JOIN partial_chain_intron i ON i.chain_id = pc.chain_id
        WHERE pc.strand IN ('+', '-') AND pc.n_introns >= 1
        GROUP BY pc.chain_id, pc.alignment_id, pc.strand, pc.start0, pc.end0
        ORDER BY pc.chain_id""").fetchall()
    for cid, aid, strand, s0, e0, introns, oks, n, n_coord in chains:
        if aid not in aln or aln[aid][0]["source"] == "protein":
            continue
        _require(n == n_coord, f"chain {cid}: strand {strand} chain has an intron without donor0/acceptor0")
        base = aln[aid][0]
        rows.append(dict(base, row_type="chain", kind="chain", strand=strand, phase=-1, start0=s0, end0=e0,
                         organism="", introns=introns, introns_ok=oks,
                         units_key=f"{base['source']}\t{strand}\t{introns}"))
    # route-B phase rows, keyed to their EST molecule (dataset/run/state of the EST molecule)
    est_mol = {}
    for sp, mid, ds, run, role, state, tte, arm in con.execute(f"""
            SELECT a.species, m.molecule_id, any_value(a.dataset), any_value(a.run), any_value(a.role),
                   any_value(m.molecule_state), bool_or(m.train_tiles_excluded), min(a.arm)
            FROM evidence_alignment a JOIN {_MOL} m
              ON m.species = a.species AND m.source = a.source AND m.member_id = a.member_id
            WHERE a.source = 'est' GROUP BY a.species, m.molecule_id""").fetchall():
        est_mol[(sp, mid)] = (ds, run, role, state, tte, arm)
    hsp_params, hsp_filter = [], ""
    if species is not None:
        hsp_filter, hsp_params = " WHERE species = ?", [species]
    for sp, mid, lib, chrom, g0, g1, strand, phase, lexc, ttexc in con.execute(
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
        rows.append(dict(species=sp, chrom=chrom, source="est", dataset=ds, run=run, library=lib, molecule_id=mid,
                         molecule_state=state, train_tiles_excluded=bool(tte or ttexc), b6_input=b_in, b6_score=b_sc,
                         b6_input_scope=scope, row_type="track", kind="phase", strand=strand, phase=int(phase),
                         start0=g0, end0=g1, organism="", introns="", introns_ok="", units_key=""))
    if family == "train":
        for sp, chrom, strand, kind, coord in con.execute(
                "SELECT DISTINCT species, chrom, strand, kind, coordinate0 FROM excluded_junction_coordinate "
                "ORDER BY 1, 2, 3, 4, 5").fetchall():
            if species is not None and sp != species:
                continue
            rows.append(dict(species=sp, chrom=chrom, source="", dataset="", run="", library="", molecule_id="",
                             molecule_state="", train_tiles_excluded=False, b6_input=False, b6_score=False,
                             b6_input_scope="-", row_type="xjunc", kind=kind, strand=strand, phase=-1,
                             start0=coord, end0=coord + 1, organism="", introns="", introns_ok="", units_key=""))
        for sp, chrom, s0, e0, cause, scope, gene, tile_rn in con.execute(
                "SELECT DISTINCT species, chrom, start0, end0, cause, scope, coalesce(gene_id, ''), "
                "coalesce(tile_rn, -1) FROM excluded_interval ORDER BY 1, 2, 3, 4, 5, 6, 7, 8").fetchall():
            if species is not None and sp != species:
                continue
            rows.append(dict(species=sp, chrom=chrom, source="", dataset="", run="", library="", molecule_id="",
                             molecule_state="", train_tiles_excluded=False, b6_input=False, b6_score=False,
                             b6_input_scope="-", row_type="xint", kind=cause, strand=scope, phase=-1,
                             start0=s0, end0=e0, organism="", introns="", introns_ok="", units_key=gene,
                             tile_rn=int(tile_rn)))
    libraries = population_libraries(rows, family)
    return rows, {k: len(v) for k, v in sorted(drops.items())}, libraries


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


def write_cache(out_dir, rows, *, family, libraries, drops, bindings, zero_lists=None):
    """Write shards + zero lists + manifest.json; returns the manifest dict.

    ``bindings``: {name: sha256} of the upstream artefacts (evidence, corpus/reference, roles, code, ...).
    ``zero_lists``: {population: {species: (tsv_path, qc_json_path)}} copied into the cache and hashed.
    """
    out = Path(out_dir)
    _require(not out.exists() or not any(out.iterdir()), f"{out} exists and is not empty; caches are immutable")
    out.mkdir(parents=True, exist_ok=True)
    _require(family in FAMILIES, f"bad family {family!r}")
    shards = {}
    for r in rows:
        shards.setdefault((r["species"], r["chrom"]), []).append(r)
    shard_meta = []
    for (sp, chrom), rs in sorted(shards.items()):
        rs.sort(key=lambda r: (r["row_type"], r["start0"], r["end0"], r["kind"], r["source"], r["molecule_id"],
                               r["strand"], r["phase"], r["introns"]))
        table = pa.Table.from_pylist([{c: r.get(c, -1 if c == "tile_rn" else ("" if c == "query_id" else None))
                                       for c in COLUMNS} for r in rs], schema=SCHEMA)
        name = f"shard.{sp}.{chrom}.parquet"
        _require("/" not in sp and "/" not in chrom, f"bad shard key {(sp, chrom)}")
        pq.write_table(table, out / name, compression="zstd", use_dictionary=True, write_statistics=False)
        shard_meta.append({"species": sp, "chrom": chrom, "file": name, "rows": len(rs),
                           "sha256": sha256_file(out / name)})
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
    manifest = {"cache_version": CACHE_VERSION, "family": family, "spec": dict(TS.SPEC_SHAS),
                "shards": shard_meta, "zero_lists": zero_meta, "libraries": libraries,
                "drops": [[sp, src, cause, n] for (sp, src, cause), n in sorted(drops.items())],
                "bindings": dict(sorted(bindings.items()))}
    (out / "manifest.json").write_text(_canon(manifest) + "\n")
    (out / "manifest.json.sha256").write_text(f"{sha256_file(out / 'manifest.json')}  manifest.json\n")
    return manifest


# ------------------------------------------------------------------------------------------------ loader
@dataclass
class Shard:
    cols: dict                 # column -> numpy array (object for strings)
    track_idx: np.ndarray      # row indices of track rows sorted by start0
    track_start: np.ndarray
    track_maxlen: int
    chain_idx: np.ndarray
    chain_start: np.ndarray
    chain_maxlen: int
    xjunc: list                # [(strand, kind, coord)]
    xint: list                 # [(start0, end0, cause, scope, tile_rn)]


class ObsCache:
    """Read-only cache.  Verifies manifest sidecar, spec shas and (lazily, on first use) each shard's sha256."""

    def __init__(self, path, *, verify_shards=True):
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
        self._shards = {}
        self._meta = {(s["species"], s["chrom"]): s for s in self.manifest["shards"]}
        self.verify_shards = verify_shards
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
    def shard(self, species, chrom):
        key = (species, chrom)
        if key in self._shards:
            return self._shards[key]
        meta = self._meta.get(key)
        if meta is None:
            empty = np.zeros(0, np.int64)
            sh = Shard({c: np.zeros(0, object) for c in COLUMNS}, empty, empty, 0, empty, empty, 0, [], [])
            self._shards[key] = sh
            return sh
        f = self.path / meta["file"]
        if self.verify_shards:
            _require(sha256_file(f) == meta["sha256"], f"shard {f} sha256 changed since the cache was built")
        t = pq.read_table(f)
        _require(tuple(t.schema.names) == COLUMNS, f"{f}: columns {t.schema.names} != {COLUMNS}")
        cols = {}
        for c in COLUMNS:
            arr = t.column(c).to_numpy(zero_copy_only=False)
            cols[c] = arr
        rt = cols["row_type"]
        start, end = cols["start0"].astype(np.int64), cols["end0"].astype(np.int64)
        cols["start0"], cols["end0"] = start, end

        def index(mask):
            idx = np.nonzero(mask)[0]
            idx = idx[np.argsort(start[idx], kind="stable")]
            ml = int((end[idx] - start[idx]).max()) if len(idx) else 0
            return idx, start[idx], ml

        ti, ts, tml = index(rt == "track")
        ci, cs, cml = index(rt == "chain")
        xj = [(cols["strand"][i], cols["kind"][i], int(start[i])) for i in np.nonzero(rt == "xjunc")[0]]
        xi = [(int(start[i]), int(end[i]), cols["kind"][i], cols["strand"][i], int(cols["tile_rn"][i]))
              for i in np.nonzero(rt == "xint")[0]]
        sh = Shard(cols, ti, ts, tml, ci, cs, cml, xj, xi)
        self._shards[key] = sh
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
