"""WP2b ``evlabel`` (plan v4.1 §5; protocol A18.4, A19.3, A20, A50, A50.1, A50.2): evidence-derived segmentation labels.

A50.2 binds: this is a SEPARATE deliverable for the segmentation work.  It is never a B6 input, target, gradient,
checkpoint selector, control or gate; no B6 module imports this module, and ``refuse_if_evlabel`` lets the B6 trainer
refuse an evlabel artefact passed to it (the DB carries ``consumer_policy`` = ``CONSUMER_POLICY``).

Population (per species, from the merged C0 snapshot): accepted PRIMARY placements whose molecule is
``train_track`` and ``training_eligible`` (EST from the A37 primary arm only); no (dataset, run) marked ``b6_score`` in
the roles manifest; scope ``train`` (train tiles) additionally requires ``train_tiles_excluded = false``, scope
``valid`` (valid tiles) admits it; test tiles are never labelled; excluded protein queries are dropped; cells on N
bases of the B6 corpus tile are dropped (``cells_dropped_masked_dna``).

Cells (tile-local, 0-based half-open; unsupported cells are weight 0 and absent), one row per source:

* intron family -- accepted, canonical ``junction_observation`` on strand +/- of an eligible primary placement, joined
  to its ``junction_support`` row (``excluded_4c`` false) and to no ``excluded_junction_coordinate``; ``intron`` over
  [min, max] of (donor0, acceptor0), ``splice_donor`` at donor0, ``splice_acceptor`` at acceptor0 (own
  ``junction_cells``; never ``gsf_contract.add_junction_evidence``).  ``n`` per (junction, source, scope) = distinct
  eligible molecule ids (deduplicated across datasets/runs), counted per library, capped at 10 per library for EST
  as the C0 producer does, then summed; weight = ``gsf_contract.junction_weight`` (EST 1.0 / PacBio 0.9 / ONT 0.7).
  Protein junction cells need >= 2 organisms (``n`` = distinct organisms) on the exact boundary (A19.3).
* exon -- ``aligned_block`` bases, per-base units, ``evidence_weight(family="exon")`` (PacBio 1.0 / ONT 0.8 /
  EST 0.6); retained-intron-like bases (A18.4: a block fully spanning an intron with >= 3 junction molecules of the
  scope, spanned by < 3 molecules of the source) get 0.25.
* protein CDS (A19.3) -- ``protein_coding_gene`` and ``exon`` over the aligned CDS blocks of the WP2-prot supplement,
  source weight 1.0, n = distinct organisms covering the base; the codon position goes to ``evlabel_phase``;
  protein never labels UTR.
* UTR (A50/A50.1) -- candidates of a read: eligible protein placements on the same strand whose every CDS block lies
  inside a read block and whose CDS introns equal the read's introns over the CDS span.  Different CDS starts or
  stops among the candidates make the read neutral (``*_utr_blocked_conflicting_cds``); a side whose start (or stop)
  is not ORF-complete is neutral (``*_utr_blocked_incomplete_orf``); otherwise read bases 5' of the start codon are
  ``5UTR`` and 3' of the stop codon ``3UTR`` (stop codons never UTR), both also ``protein_coding_gene`` at the read's
  weight.  ESTs contribute interior UTR bases only; v1 emits no boundary cell at all
  (``BOUNDARY_POLICY`` = ``boundary_cells_not_emitted_v1``; NOT a final decision -- the A50.2 consumer amendment must
  re-decide boundary cells and the WP2c gate, and a consumer must not read the absence of boundary cells as one).

Merging: max across sources (``to_dense``), weight = min(4, 1 + sw * g * log1p(n)); genotype multiplier g = 1.0 when
every contributing unit is ``reference`` (protein ``not_applicable`` -> 1.0, D10 default) and 0.5 otherwise
(fail-closed reading for mixed strata, recorded as ``GENOTYPE_POLICY``).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from transgenic.evidence import leakage as LK

VERSION = "evlabel_b6_v1"
CONSUMER_POLICY = "evlabel_not_a_b6_input_A50.2"
BOUNDARY_POLICY = "boundary_cells_not_emitted_v1"
GENOTYPE_POLICY = "g1_iff_all_units_reference_or_not_applicable_else_0.5_v1"
UNIT_POLICY = "est_library_cap_10_as_c0_producer_longread_distinct_protein_distinct_organisms_v1"
PHASE_CONVENTION = "codon_position_of_tile_local_start0_in_tile_orientation_v1"
EST_LIBRARY_CAP = 10
RETAINED_MIN_JUNCTION = 3
RETAINED_MAX_SPANNING = 3
READ_SOURCES = ("est", "ont", "pacbio")
SCOPES = {"train": "train", "valid": "valid"}      # tile split -> scope
CELL_DDL = ("CREATE TABLE evlabel_cell (rn INTEGER NOT NULL, class_index INTEGER NOT NULL, start0 INTEGER NOT NULL, "
            "end0 INTEGER NOT NULL, weight DOUBLE NOT NULL, source VARCHAR NOT NULL, family VARCHAR NOT NULL, "
            "rule VARCHAR NOT NULL, CHECK (end0 > start0 AND start0 >= 0), CHECK (weight > 0 AND weight <= 4))")
TILE_DDL = ("CREATE TABLE evlabel_tile (rn INTEGER NOT NULL PRIMARY KEY, species VARCHAR NOT NULL, chrom VARCHAR NOT NULL, "
            "ws BIGINT NOT NULL, we BIGINT NOT NULL, L INTEGER NOT NULL, is_rc BOOLEAN NOT NULL, split VARCHAR NOT NULL, "
            "scope VARCHAR NOT NULL)")
PHASE_DDL = ("CREATE TABLE evlabel_phase (rn INTEGER NOT NULL, start0 INTEGER NOT NULL, end0 INTEGER NOT NULL, "
             "strand VARCHAR NOT NULL, codon_position INTEGER NOT NULL, alignment_id VARCHAR NOT NULL)")
JUNCTION_DDL = ("CREATE TABLE evlabel_junction (species VARCHAR NOT NULL, source VARCHAR NOT NULL, scope VARCHAR NOT NULL, "
                "junction_key VARCHAR NOT NULL, chrom VARCHAR NOT NULL, strand VARCHAR NOT NULL, donor0 BIGINT NOT NULL, "
                "acceptor0 BIGINT NOT NULL, n BIGINT NOT NULL, genotype_multiplier DOUBLE NOT NULL, weight DOUBLE NOT NULL, "
                "PRIMARY KEY (species, source, scope, junction_key))")
QC_DDL = ("CREATE TABLE evlabel_qc (species VARCHAR NOT NULL, source VARCHAR NOT NULL, metric VARCHAR NOT NULL, "
          "value DOUBLE NOT NULL, PRIMARY KEY (species, source, metric))")
PROV_DDL = "CREATE TABLE evlabel_provenance (key VARCHAR NOT NULL PRIMARY KEY, value VARCHAR NOT NULL)"


class EvlabelError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise EvlabelError(message)


def _gc():
    return LK.load_build_b5().gc


def class_index(name):
    return _gc().SEG_CLASSES.index(name)


def family_of(cls):
    return {"intron": "intron", "splice_donor": "intron", "splice_acceptor": "intron", "exon": "exon",
            "protein_coding_gene": "cds", "5UTR": "utr", "3UTR": "utr"}[cls]


def genotype_arg(nonref):
    return "unknown" if nonref else "reference"


def refuse_if_evlabel(path):
    """For B6 consumers (trainer, caches): raise if ``path`` is an evlabel artefact (A50.2)."""
    import duckdb

    p = Path(path)
    if not p.is_file():
        return
    try:
        con = duckdb.connect(str(p), read_only=True)
    except Exception:
        return
    try:
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if "evlabel_provenance" in tables or "evlabel_cell" in tables:
            raise EvlabelError(f"{path} is an evlabel artefact ({CONSUMER_POLICY}); B6 refuses it (A50.2)")
    finally:
        con.close()


# ------------------------------------------------------------------------------------------ inputs
def _sidecar_sha(path):
    sha = LK.sha256(path)
    side = Path(str(path) + ".sha256")
    require(side.is_file() and side.read_text().split()[:1] == [sha], f"{path}: sha256 sidecar missing or != {sha}")
    require(not Path(str(path) + ".wal").exists(), f"{path} has a WAL")
    return sha


AUDITED_SOURCES = ("est", "ont", "pacbio")


def _is_count(v):
    """A JSON integer counter: int, not bool, >= 0 (no float or string coercion)."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


TRAINING_PROVENANCE_KEYS = ("inputs", "leakage_intervals_sha256", "interval_manifest_sha256", "completeness_codes")
COMPLETENESS_DEFAULT = "null_not_computed_at_c0"


def _open_snapshot(path):
    """§8 common 1: refuse a WAL, recomputed sha == sidecar, open read-only, re-check no WAL appeared."""
    import duckdb

    sha = _sidecar_sha(path)
    con = duckdb.connect(str(path), read_only=True)
    if Path(str(path) + ".wal").exists():
        con.close()
        raise EvlabelError(f"{path}.wal appeared after opening")
    return sha, con


def verify_training_binding(*, merged, corpus, freeze, roles_obj):
    """Plan v4.1 §8 ``training`` family, implemented locally with the plan's key set (not imported from WP3):
    for every (species, source) of the merged snapshot's provenance -- ``inputs.corpus.sha256`` = freeze
    ``base_sha256``; ``leakage_intervals_sha256`` present; ``interval_manifest_sha256`` = the freeze entry for that
    species (a C0 without it fails: §8.3 revalidation is not offered here); B6 corpus sha = ``out_sha256``; roles sha
    = its sidecar.  Common: snapshot checks, ``verify_schema``, every ``leakage*`` c0_qc counter 0, A49.2
    exhaustiveness recomputed from ``c0_meta.exhaustiveness`` integers, completeness policy
    ``null_not_computed_at_c0``.  An eval_only snapshot (policy eval_only_no_leakage_model) is refused."""
    from transgenic.evidence import b6_corpus_repair as CR
    from transgenic.evidence import c0_schema as CS

    fz = json.loads(Path(freeze).read_text())
    try:
        CR.verify_freeze(fz)
    except CR.RepairError as exc:
        raise EvlabelError(f"freeze: {exc}") from exc
    corpus_sha = LK.sha256(corpus)
    require(not Path(str(corpus) + ".wal").exists(), f"{corpus} has a WAL")
    require(corpus_sha == fz["out_sha256"], f"B6 corpus sha256 {corpus_sha} != freeze out_sha256")
    merged_sha, con = _open_snapshot(merged)
    try:
        try:
            CS.verify_schema(con, merged=True)
        except CS.SchemaError as exc:
            raise EvlabelError(f"{merged}: not a merged c0-v1 snapshot: {exc}") from exc
        prov = defaultdict(dict)
        for sp, src, key, value in con.execute("SELECT species, source, key, value FROM c0_provenance "
                                               "ORDER BY species, source, key").fetchall():
            try:
                prov[(sp, src)][key] = json.loads(value)
            except (TypeError, ValueError):
                prov[(sp, src)][key] = value
        require(prov, f"{merged}: c0_provenance is empty")
        # review 2 H2: coverage is exactly the merge_manifest (species, source) set; provenance, leakage QC and
        # exhaustiveness must each cover every pair -- a pair absent from provenance fails, it is never skipped.
        cover = {(sp, src) for sp, src in con.execute("SELECT species, source FROM merge_manifest "
                                                      "ORDER BY species, source").fetchall()}
        require(cover, f"{merged}: merge_manifest is empty: not a merged snapshot")
        require(set(prov) == cover, f"c0_provenance coverage {sorted(prov)} != merge_manifest {sorted(cover)}")
        for (sp, src), p in sorted(prov.items()):
            where = f"c0_provenance {sp}/{src}"
            missing = [k for k in TRAINING_PROVENANCE_KEYS if k not in p]
            require(not missing, f"{where}: required key(s) {missing} missing (training family, §8)")
            require((p.get("policy") or "") != "eval_only_no_leakage_model", f"{where}: eval_only snapshot refused")
            corpus_in = (p["inputs"].get("corpus") or {}).get("sha256")
            require(corpus_in == fz["base_sha256"], f"{where}: inputs.corpus {corpus_in} != freeze base_sha256")
            try:
                want = CR.freeze_manifest_sha256(fz, sp)
            except CR.RepairError as exc:
                raise EvlabelError(f"{where}: {exc}") from exc
            require(p["interval_manifest_sha256"] == want,
                    f"{where}: interval_manifest_sha256 != freeze entry for {sp}")
            pol = (p["completeness_codes"] or {}).get("policy")
            require(pol == COMPLETENESS_DEFAULT, f"{where}: completeness policy {pol!r} without a bound WP2c DB")
        qc = defaultdict(dict)
        for sp, src, metric, value in con.execute("SELECT species, source, metric, value FROM c0_qc "
                                                  "ORDER BY ALL").fetchall():
            qc[(sp, src)][metric] = value
        for key in sorted(cover):
            require(key in qc, f"{key}: no c0_qc record")
            require("leakage_nonzero" in qc[key], f"{key}: c0_qc lacks leakage_nonzero")
            bad = {m: v for m, v in qc[key].items() if m.startswith("leakage") and v != 0}
            require(not bad, f"{key}: leakage QC counters not zero {bad}")
        extra = sorted(set(qc) - cover)
        require(not extra, f"c0_qc records outside the merged coverage: {extra}")
        meta = dict(con.execute("SELECT key, value FROM c0_meta ORDER BY key").fetchall())
        require("exhaustiveness" in meta, f"{merged}: c0_meta lacks exhaustiveness")
        ex = json.loads(meta["exhaustiveness"])
        require(isinstance(ex, dict), "c0_meta.exhaustiveness is not a mapping")
        audited = defaultdict(set)
        for sp, src in cover:
            if src in AUDITED_SOURCES:
                audited[sp].add(src)
        require(audited, "no audited (species, source) in the merged coverage")
        for sp, sources in sorted(audited.items()):
            rec = ex.get(sp)
            require(isinstance(rec, dict) and isinstance(rec.get("by_source"), dict),
                    f"{sp}: no exhaustiveness by_source record in c0_meta")
            by = rec["by_source"]
            require(set(by) == sources, f"A49.2: {sp} exhaustiveness sources {sorted(by)} != audited {sorted(sources)}")
            tot_c = tot_m = 0
            for src in sorted(sources):
                part = by[src]
                m, below = part.get("molecules"), part.get("below_cap")
                require(_is_count(m) and _is_count(below) and below <= m,
                        f"A49.2: {sp}/{src} counters must be nonnegative integers with below_cap <= molecules, "
                        f"got {part}")
                c = m - below
                require(c <= m // 100, f"A49.2: {sp}/{src} capped {c} > floor(0.01 x {m})")
                tot_c, tot_m = tot_c + c, tot_m + m
            require(tot_c * 1000 <= tot_m, f"A49.2: {sp} capped {tot_c} / {tot_m} > 0.001")
        protein_species = {r[0] for r in con.execute(
            "SELECT DISTINCT species FROM evidence_alignment WHERE source = 'protein' AND accepted AND "
            "placement_kind = 'primary' ORDER BY species").fetchall()}
    finally:
        con.close()
    return {"family": "training", "merged_sha256": merged_sha, "corpus_sha256": corpus_sha,
            "base_sha256": fz["base_sha256"], "interval_manifest_sha256": dict(fz["interval_manifest_sha256"]),
            "roles_sha256": roles_obj.sha256, "species": sorted({k[0] for k in prov}),
            "protein_species": sorted(protein_species)}, fz


def verify_inputs(*, merged, supplements, corpus, freeze, roles):
    from transgenic.evidence import b6_protein_placement as PP
    from transgenic.evidence import b6_roles as R

    roles_obj = R.load_roles(roles)
    binding, fz = verify_training_binding(merged=merged, corpus=corpus, freeze=freeze, roles_obj=roles_obj)
    merged_sha = binding["merged_sha256"]
    sups = {}
    for p in supplements:
        sha, prov = PP.open_supplement(p)
        require(prov["inputs"]["merged"]["sha256"] == merged_sha,
                f"{p}: protein supplement was built against another merged snapshot")
        require(prov["species"] not in sups, f"two protein supplements for {prov['species']}")
        sups[prov["species"]] = {"path": str(Path(p).absolute()), "sha256": sha, "stop_convention": prov["stop_convention"]}
    # review 1 M4: every species with accepted primary protein placements needs its bound supplement; a species
    # without protein evidence is recorded explicitly as absent, never silently dropped.
    missing = sorted(set(binding["protein_species"]) - set(sups))
    require(not missing, f"no protein-placement supplement bound for {missing}, which have accepted protein "
                         "placements in the merged snapshot (their CDS / phase / UTR labels would be dropped)")
    protein_evidence = {sp: ("bound" if sp in sups else "absent") for sp in binding["species"]}
    return {"binding": binding, "protein_evidence": protein_evidence,
            "merged": {"path": str(Path(merged).absolute()), "sha256": merged_sha},
            "corpus": {"path": str(Path(corpus).absolute()), "sha256": binding["corpus_sha256"]},
            "freeze": {"path": str(Path(freeze).absolute()), "sha256": LK.sha256(freeze), "rule_id": fz["rule_id"]},
            "roles": {"path": roles_obj.path, "sha256": roles_obj.sha256},
            "protein_supplements": sups}, roles_obj


# ------------------------------------------------------------------------------------------ population
def eligible_molecules(con, species, score_runs):
    """(source, molecule_id) -> (library, nonref, tte) for eligible molecules (train_track, training_eligible, no
    b6_score dataset/run)."""
    out = {}
    rows = con.execute(
        "SELECT source, molecule_id, min(library), bool_or(genotype_stratum NOT IN ('reference', 'not_applicable')), "
        "bool_or(train_tiles_excluded), bool_and(molecule_state = 'train_track'), bool_and(role = 'training_eligible'), "
        "list(DISTINCT dataset || chr(9) || run ORDER BY dataset || chr(9) || run) "
        "FROM molecule_member WHERE species = ? GROUP BY source, molecule_id ORDER BY source, molecule_id",
        [species]).fetchall()
    dropped = Counter()
    for source, mid, lib, nonref, tte, tt, te, runs in rows:
        if not (tt and te):
            continue
        if any(tuple(r.split("\t", 1)) in score_runs for r in runs):
            dropped[source] += 1
            continue
        out[(source, mid)] = (lib, bool(nonref), bool(tte))
    return out, dropped


def in_scope(elig, scope):
    return elig is not None and (scope == "valid" or not elig[2])


# ------------------------------------------------------------------------------------------ sweeps
def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def sweep(units, source):
    """``units``: {unit_key: (library, nonref, [(start0, end0), ...])} on one chrom -> segments (a, b, n, nonref)
    with n = sum over libraries of (distinct units, capped at 10 per library for EST)."""
    events = []
    for key, (lib, nonref, iv) in units.items():
        for a, b in _merge(iv):
            events.append((a, 1, lib, nonref))
            events.append((b, -1, lib, nonref))
    events.sort(key=lambda e: (e[0], e[1]))
    cap = EST_LIBRARY_CAP if source == "est" else None
    per_lib, nonref_n, out = Counter(), 0, []
    i, prev = 0, None
    while i < len(events):
        pos = events[i][0]
        if prev is not None and pos > prev:
            n = sum(min(cap, c) if cap else c for c in per_lib.values() if c > 0)
            if n > 0:
                seg = (prev, pos, n, nonref_n > 0)
                if out and out[-1][1] == prev and out[-1][2:] == seg[2:]:
                    out[-1] = (out[-1][0], pos) + seg[2:]
                else:
                    out.append(seg)
        while i < len(events) and events[i][0] == pos:
            _, d, lib, nr = events[i]
            per_lib[lib] += d
            nonref_n += d if nr else 0
            i += 1
        prev = pos
    return out


class SegmentIndex:
    """Overlap queries over (start0, end0, payload) segments that may overlap."""

    def __init__(self, segs):
        self.segs = sorted(segs, key=lambda s: (s[0], s[1]))
        self.starts = [s[0] for s in self.segs]
        self.prefix, m = [], -1
        for s in self.segs:
            m = max(m, s[1])
            self.prefix.append(m)

    def overlapping(self, a, b):
        i = bisect.bisect_left(self.starts, b) - 1
        out = []
        while i >= 0 and self.prefix[i] > a:
            if self.segs[i][1] > a:
                out.append(self.segs[i])
            i -= 1
        return out[::-1]


# ------------------------------------------------------------------------------------------ builder
class SpeciesLabels:
    """Genome-level labelled segments of one species, per scope: {scope: [(chrom, a, b, class, source, rule, w)]},
    junction rows and phase segments."""

    def __init__(self):
        self.cells = {s: defaultdict(list) for s in SCOPES.values()}      # scope -> chrom -> [(a, b, cls, src, rule, w)]
        # scope -> chrom -> [(a, b, strand, cp_at_a, gff_phase, alignment_id, block_end)]; filled by _protein_cds with
        # the same in_scope() filter as the CDS cells (review 1 H1: valid-only phase never reaches a train tile)
        self.phase = {s: defaultdict(list) for s in SCOPES.values()}
        self.junctions = []                                              # evlabel_junction rows
        self.qc = defaultdict(Counter)                                   # source -> metric -> n
        self.molecules_used = defaultdict(set)


def junction_cells(donor0, acceptor0):
    """(class, start0, end0) genomic cells of one junction: intron over [min, max], donor at donor0, acceptor at
    acceptor0 (strand-aware C0 coordinates, c0_schema.py:13-20)."""
    lo, hi = min(donor0, acceptor0), max(donor0, acceptor0)
    return [("intron", lo, hi + 1), ("splice_donor", donor0, donor0 + 1), ("splice_acceptor", acceptor0, acceptor0 + 1)]


def _read_placements(con, species, elig):
    """Accepted primary read placements of eligible molecules: alignment_id -> (source, molecule_id, chrom, strand,
    [blocks]); EST from the A37 primary arm only."""
    aln = {}
    for aid, source, mid, chrom, strand, arm in con.execute(
            "SELECT alignment_id, source, molecule_id, chrom, transcript_strand, arm FROM evidence_alignment WHERE species = ? "
            "AND accepted AND placement_kind = 'primary' AND source IN ('est', 'ont', 'pacbio') ORDER BY alignment_id",
            [species]).fetchall():
        if source == "est" and arm != "primary":
            continue
        if (source, mid) in elig:
            aln[aid] = [source, mid, chrom, strand, []]
    cur = con.cursor()
    cur.execute("SELECT alignment_id, start0, end0 FROM aligned_block WHERE species = ? AND source IN ('est', 'ont', 'pacbio') "
                "ORDER BY alignment_id, alignment_block_index", [species])
    while True:
        rows = cur.fetchmany(8192)
        if not rows:
            break
        for aid, a, b in rows:
            if aid in aln:
                aln[aid][4].append((a, b))
    cur.close()
    return aln


def _junction_units(con, species, elig, placements_primary):
    """Eligible junction support: [(source, junction_key, chrom, strand, start0, end0, donor0, acceptor0, molecule_id,
    dataset, run)] after the 4(c) flag and the excluded-coordinate join (counted)."""
    rows = con.execute("""
        SELECT o.source, o.junction_key, o.chrom, o.strand, o.start0, o.end0, o.donor0, o.acceptor0, o.molecule_id,
               o.dataset, o.run, o.alignment_id, s.excluded_4c,
               (cd.chrom IS NOT NULL OR ca.chrom IS NOT NULL) AS coord_hit
        FROM junction_observation o
        JOIN junction_support s ON s.species = o.species AND s.source = o.source AND s.junction_key = o.junction_key
             AND s.dataset = o.dataset AND s.run = o.run AND s.library = o.library AND s.role = o.role
             AND s.molecule_id = o.molecule_id
        LEFT JOIN (SELECT DISTINCT chrom, strand, coordinate0 FROM excluded_junction_coordinate
                   WHERE species = ? AND kind = 'donor') cd
               ON cd.chrom = o.chrom AND cd.strand = o.strand AND cd.coordinate0 = o.donor0
        LEFT JOIN (SELECT DISTINCT chrom, strand, coordinate0 FROM excluded_junction_coordinate
                   WHERE species = ? AND kind = 'acceptor') ca
               ON ca.chrom = o.chrom AND ca.strand = o.strand AND ca.coordinate0 = o.acceptor0
        WHERE o.species = ? AND o.accepted AND o.canonical AND o.strand IN ('+', '-')
        ORDER BY o.source, o.junction_key, o.molecule_id, o.dataset, o.run, o.alignment_id""",
                       [species, species, species]).fetchall()
    out, dropped = [], Counter()
    for (source, jk, chrom, strand, s0, e0, d0, a0, mid, ds, run, aid, ex4c, hit) in rows:
        if aid not in placements_primary or (source, mid) not in elig:
            continue
        if ex4c or hit:
            dropped[source] += 1
            continue
        out.append((source, jk, chrom, strand, s0, e0, d0, a0, mid, ds, run))
    return out, dropped


def _protein_rows(sup_path, species, elig):
    import duckdb

    con = duckdb.connect(str(sup_path), read_only=True)
    try:
        rows = con.execute("SELECT alignment_id, molecule_id, organism, chrom, strand, cds_blocks, qlen, qstart0, qend, "
                           "start_codon_ok, stop_codon_ok, stop_convention FROM protein_placement WHERE species = ? "
                           "ORDER BY alignment_id", [species]).fetchall()
    finally:
        con.close()
    out = []
    for aid, mid, org, chrom, strand, blocks, qlen, qs, qe, sok, tok, conv in rows:
        if ("protein", mid) not in elig:
            continue
        out.append({"alignment_id": aid, "molecule_id": mid, "organism": org, "chrom": chrom, "strand": strand,
                    "blocks": [tuple(b) for b in json.loads(blocks)], "start_complete": qs == 0 and bool(sok),
                    "stop_complete": qe == qlen and bool(tok), "stop_convention": conv})
    return out


def _cds_span(p):
    bl = p["blocks"]
    return bl[0][0], bl[-1][1]


def _stop_codon(p):
    a, b = _cds_span(p)
    inside = p["stop_convention"] == "inside_last_block"
    if p["strand"] == "+":
        return (b - 3, b) if inside else (b, b + 3)
    return (a, a + 3) if inside else (a - 3, a)


def utr_for_read(read_blocks, strand, candidates):
    """(utr5 intervals, utr3 intervals, status) of one read; status in {'ok', 'no_candidate', 'conflicting_cds',
    'incomplete_orf_5', 'incomplete_orf_3', 'incomplete_orf_both'}."""
    rb = sorted(read_blocks)
    read_introns = [(rb[i][1], rb[i + 1][0]) for i in range(len(rb) - 1)]
    fits = []
    for p in candidates:
        if p["strand"] != strand:
            continue
        cds = [(a, b) for a, b, _ in p["blocks"]]
        if not all(any(ra <= a and b <= rbb for ra, rbb in rb) for a, b in cds):
            continue
        cs, ce = cds[0][0], cds[-1][1]
        cds_introns = [(cds[i][1], cds[i + 1][0]) for i in range(len(cds) - 1)]
        if [i for i in read_introns if cs <= i[0] and i[1] <= ce] != cds_introns:
            continue
        fits.append(p)
    if not fits:
        return [], [], "no_candidate"
    spans = {(_cds_span(p), _stop_codon(p)) for p in fits}
    if len(spans) > 1:
        return [], [], "conflicting_cds"
    (cs, ce), (ss, se) = next(iter(spans))
    five_ok = all(p["start_complete"] for p in fits)
    three_ok = all(p["stop_complete"] for p in fits)
    u5, u3 = [], []
    for a, b in rb:
        if strand == "+":
            if five_ok and a < cs:
                u5.append((a, min(b, cs)))
            if three_ok and b > se:
                u3.append((max(a, se), b))
        else:
            if five_ok and b > ce:
                u5.append((max(a, ce), b))
            if three_ok and a < ss:
                u3.append((a, min(b, ss)))
    status = "ok" if five_ok and three_ok else ("incomplete_orf_both" if not five_ok and not three_ok
                                                else "incomplete_orf_5" if not five_ok else "incomplete_orf_3")
    return u5, u3, status


def build_species(con, species, elig, sup_rows):
    gc = _gc()
    lab = SpeciesLabels()
    reads = _read_placements(con, species, elig)
    junits, dropped4c = _junction_units(con, species, elig, reads)
    prot_primary = {r["alignment_id"] for r in sup_rows}
    # protein junction observations come from the protein C0 rows of the supplement's alignments
    prot_junits = []
    for (source, jk, chrom, strand, s0, e0, d0, a0, mid, ds, run, aid, ex4c, hit) in con.execute("""
            SELECT o.source, o.junction_key, o.chrom, o.strand, o.start0, o.end0, o.donor0, o.acceptor0, o.molecule_id,
                   o.dataset, o.run, o.alignment_id, s.excluded_4c,
                   EXISTS (SELECT 1 FROM excluded_junction_coordinate x WHERE x.species = o.species AND x.chrom = o.chrom
                           AND x.strand = o.strand AND ((x.kind = 'donor' AND x.coordinate0 = o.donor0)
                                                        OR (x.kind = 'acceptor' AND x.coordinate0 = o.acceptor0)))
            FROM junction_observation o
            JOIN junction_support s ON s.species = o.species AND s.source = o.source AND s.junction_key = o.junction_key
                 AND s.dataset = o.dataset AND s.run = o.run AND s.library = o.library AND s.role = o.role
                 AND s.molecule_id = o.molecule_id
            WHERE o.species = ? AND o.source = 'protein' AND o.accepted AND o.canonical AND o.strand IN ('+', '-')
            ORDER BY o.junction_key, o.molecule_id, o.alignment_id""", [species]).fetchall():
        if aid not in prot_primary or ("protein", mid) not in elig:
            continue
        if ex4c or hit:
            dropped4c["protein"] += 1
            continue
        prot_junits.append((source, jk, chrom, strand, s0, e0, d0, a0, mid, ds, run))
    for src, n in sorted(dropped4c.items()):
        lab.qc[src]["junctions_dropped_excluded_4c"] += n
    for scope in SCOPES.values():
        _junctions(lab, species, scope, junits + prot_junits, elig)
        _exons(lab, scope, reads, elig, junits)
        _protein_cds(lab, scope, sup_rows, elig)
        _utr(lab, scope, reads, sup_rows, elig)
    return lab


def _junctions(lab, species, scope, units, elig):
    gc = _gc()
    by = defaultdict(lambda: {"libs": defaultdict(set), "runs": defaultdict(set), "nonref": False, "geo": None})
    for (source, jk, chrom, strand, s0, e0, d0, a0, mid, ds, run) in units:
        e = elig[(source, mid)]
        if not in_scope(e, scope):
            continue
        rec = by[(source, jk)]
        rec["libs"][e[0]].add(mid)
        rec["runs"][mid].add((ds, run))
        rec["nonref"] |= e[1]
        rec["geo"] = (chrom, strand, d0, a0)
    for (source, jk), rec in sorted(by.items()):
        if source == "protein":
            n = len(rec["libs"])                      # library = organism: distinct organisms on the exact boundary
            if n < 2:
                lab.qc[source]["protein_junctions_below_2_organisms"] += 1
                continue
        else:
            cap = EST_LIBRARY_CAP if source == "est" else None
            n = sum(min(cap, len(m)) if cap else len(m) for m in rec["libs"].values())
        lab.qc[source]["junction_n_crossrun_dedup"] += sum(1 for runs in rec["runs"].values() if len(runs) > 1)
        chrom, strand, d0, a0 = rec["geo"]
        g = 0.5 if rec["nonref"] and source != "protein" else 1.0
        w = gc.junction_weight(source, n, genotype_arg(rec["nonref"] and source != "protein"))
        lab.junctions.append((species, source, scope, jk, chrom, strand, d0, a0, n, g, w))
        for cls, a, b in junction_cells(d0, a0):
            lab.cells[scope][chrom].append((a, b, cls, source, "junction", w))
        for libset in rec["libs"].values():
            lab.molecules_used[source] |= libset


def _exons(lab, scope, reads, elig, junits):
    gc = _gc()
    # retained-intron-like (A18.4): introns with >= 3 junction molecules (all read sources) in this scope
    jmol = defaultdict(set)
    for (source, jk, chrom, strand, s0, e0, d0, a0, mid, ds, run) in junits:
        if in_scope(elig[(source, mid)], scope):
            jmol[(chrom, s0, e0)].add((source, mid))
    supported = sorted(k for k, v in jmol.items() if len(v) >= RETAINED_MIN_JUNCTION)
    for source in READ_SOURCES:
        per_chrom = defaultdict(dict)
        for aid, (src, mid, chrom, strand, blocks) in reads.items():
            if src != source or not in_scope(elig[(src, mid)], scope):
                continue
            lib, nonref, _ = elig[(src, mid)]
            u = per_chrom[chrom].setdefault(mid, (lib, nonref, []))
            u[2].extend(blocks)
            lab.molecules_used[source].add(mid)
        for chrom, units in sorted(per_chrom.items()):
            segs = sweep(units, source)
            retained = []
            for (c, s0, e0) in supported:
                if c != chrom:
                    continue
                spanning = {mid for mid, (_, _, iv) in units.items() if any(a <= s0 and e0 <= b for a, b in iv)}
                if spanning and len(spanning) < RETAINED_MAX_SPANNING:
                    retained.append((s0, e0))
            for a, b, n, nonref in segs:
                w = gc.evidence_weight(source, n, genotype_arg(nonref), family="exon")
                for x, y, is_ret in _split_by(a, b, retained):
                    lab.cells[scope][chrom].append((x, y, "exon", source,
                                                    "retained_intron" if is_ret else "aligned_block",
                                                    gc.RETAINED_INTRON_WEIGHT if is_ret else w))
                    if is_ret:
                        lab.qc[source]["retained_intron_bases"] += y - x


def _split_by(a, b, intervals):
    """Split [a, b) at the union of ``intervals``: [(x, y, inside)]."""
    cuts = []
    for s, e in _merge([iv for iv in intervals if iv[0] < b and iv[1] > a]):
        cuts.append((max(a, s), min(b, e)))
    out, p = [], a
    for s, e in cuts:
        if s > p:
            out.append((p, s, False))
        out.append((s, e, True))
        p = e
    if p < b:
        out.append((p, b, False))
    return out


def _protein_cds(lab, scope, sup_rows, elig):
    gc = _gc()
    per_chrom = defaultdict(dict)
    for r in sup_rows:
        if not in_scope(elig[("protein", r["molecule_id"])], scope):
            continue
        u = per_chrom[r["chrom"]].setdefault(r["organism"], (r["organism"], False, []))
        u[2].extend((a, b) for a, b, _ in r["blocks"])
        lab.molecules_used["protein"].add(r["molecule_id"])
        for a, b, ph in r["blocks"]:
            cp_a = (0 - ph) % 3 if r["strand"] == "+" else ((b - 1 - a) - ph) % 3
            lab.phase[scope][r["chrom"]].append((a, b, r["strand"], cp_a, ph, r["alignment_id"], b))
    for chrom, units in sorted(per_chrom.items()):
        for a, b, n, _ in sweep(units, "protein"):
            w = gc.evidence_weight("protein", n, "reference", family="exon")        # D10: not_applicable -> 1.0
            lab.cells[scope][chrom].append((a, b, "protein_coding_gene", "protein", "protein_cds", w))
            lab.cells[scope][chrom].append((a, b, "exon", "protein", "protein_cds", w))


def _utr(lab, scope, reads, sup_rows, elig):
    gc = _gc()
    prot = defaultdict(list)
    for r in sup_rows:
        if in_scope(elig[("protein", r["molecule_id"])], scope):
            prot[r["chrom"]].append(r)
    index = {c: SegmentIndex([(_cds_span(p)[0], _cds_span(p)[1], i) for i, p in enumerate(ps)]) for c, ps in prot.items()}
    units = {s: {"5UTR": defaultdict(dict), "3UTR": defaultdict(dict)} for s in READ_SOURCES}
    for aid, (source, mid, chrom, strand, blocks) in sorted(reads.items()):
        if not in_scope(elig[(source, mid)], scope) or strand not in ("+", "-") or not blocks:
            continue
        a, b = min(x for x, _ in blocks), max(y for _, y in blocks)
        cands = [prot[chrom][i] for _, _, i in index[chrom].overlapping(a, b)] if chrom in index else []
        u5, u3, status = utr_for_read(blocks, strand, cands)
        tag = "est" if source == "est" else "longread"
        if scope == "train":
            if status == "conflicting_cds":
                lab.qc[source][f"{tag}_utr_blocked_conflicting_cds"] += 1
            elif status.startswith("incomplete_orf"):
                lab.qc[source][f"{tag}_utr_blocked_incomplete_orf"] += 1
        lib, nonref, _ = elig[(source, mid)]
        for cls, iv in (("5UTR", u5), ("3UTR", u3)):
            if iv:
                u = units[source][cls][chrom].setdefault(mid, (lib, nonref, []))
                u[2].extend(iv)
    for source in READ_SOURCES:
        tag = "est" if source == "est" else "longread"
        for cls in ("5UTR", "3UTR"):
            for chrom, us in sorted(units[source][cls].items()):
                for a, b, n, nonref in sweep(us, source):
                    w = gc.evidence_weight(source, n, genotype_arg(nonref), family="exon")
                    lab.cells[scope][chrom].append((a, b, cls, source, "utr_" + cls.lower(), w))
                    lab.cells[scope][chrom].append((a, b, "protein_coding_gene", source, "utr_pcg", w))
                    if scope == "train":
                        lab.qc[source][f"{tag}_utr{cls[0]}_bases"] += b - a


# ------------------------------------------------------------------------------------------ tile projection
def _n_runs(seq):
    out, i, L = [], 0, len(seq)
    while i < L:
        j = seq.find("N", i)
        if j < 0:
            break
        k = j
        while k < L and seq[k] in "Nn":
            k += 1
        out.append((j, k))
        i = k
    return out


def _minus(a, b, holes):
    out, p = [], a
    for s, e in holes:
        if e <= p or s >= b:
            continue
        if s > p:
            out.append((p, s))
        p = max(p, e)
    if p < b:
        out.append((p, b))
    return out


def project_tile(lab, scope, chrom, ws, we, is_rc, seq, qc_counter):
    """Tile-local cells (class_index, start0, end0, weight, source, family, rule), N bases dropped, RC mirrored."""
    L = we - ws
    idx = lab._index.setdefault((scope, chrom), SegmentIndex([(a, b, (cls, src, rule, w))
                                                             for a, b, cls, src, rule, w in lab.cells[scope][chrom]]))
    n_runs = _n_runs(seq)          # in the tile's own orientation (the RC row stores the RC sequence)
    out = []
    for a, b, (cls, src, rule, w) in idx.overlapping(ws, we):
        x, y = max(a, ws) - ws, min(b, we) - ws
        if is_rc:
            x, y = L - y, L - x
        kept = _minus(x, y, n_runs)
        dropped = (y - x) - sum(e - s for s, e in kept)
        if dropped:
            qc_counter[src]["cells_dropped_masked_dna"] += dropped
        for s, e in kept:
            out.append((class_index(cls), s, e, w, src, family_of(cls), rule))
    return out


def project_phase(lab, scope, chrom, ws, we, is_rc, seq):
    L = we - ws
    idx = lab._pindex.setdefault((scope, chrom), SegmentIndex([(a, b, (st, cp, ph, aid, e_last)) for a, b, st, cp, ph,
                                                               aid, e_last in lab.phase[scope][chrom]]))
    n_runs = _n_runs(seq)
    out = []
    for a, b, (strand, cp_a, ph, aid, blk_end) in idx.overlapping(ws, we):
        x, y = max(a, ws), min(b, we)
        for s, e in _minus(x - ws, y - ws, n_runs if not is_rc else [(L - q, L - p) for p, q in n_runs][::-1]):
            gs, ge = s + ws, e + ws            # genomic sub-block
            if not is_rc:
                cp = (gs - a - ph) % 3 if strand == "+" else ((blk_end - 1 - gs) - ph) % 3
                out.append((s, e, strand, cp, aid))
            else:
                # tile orientation: local start L-(ge-ws) is genomic base ge-1; strand swapped
                cp = (ge - 1 - a - ph) % 3 if strand == "+" else ((blk_end - 1 - (ge - 1)) - ph) % 3
                out.append((L - (ge - ws), L - (gs - ws), "-" if strand == "+" else "+", cp, aid))
    return out


# ------------------------------------------------------------------------------------------ build
def build(*, merged, supplements, corpus, freeze, roles, out, argv=()):
    import duckdb

    out = Path(out).absolute()
    for p in (out, Path(str(out) + ".sha256"), Path(str(out) + ".PROVENANCE.json")):
        require(not os.path.lexists(p), f"{p} exists; refusing to overwrite")
    inputs, roles_obj = verify_inputs(merged=merged, supplements=supplements, corpus=corpus, freeze=freeze, roles=roles)
    score_runs = {(r.dataset, r.run) for r in roles_obj.rows if r.b6_score}
    gc = _gc()
    mcon = duckdb.connect(str(merged), read_only=True)
    ccon = duckdb.connect(str(corpus), read_only=True)
    tmp = Path(str(out) + ".partial")
    require(not os.path.lexists(tmp), f"{tmp} exists; remove it")
    dst = duckdb.connect(str(tmp))
    qc_all = defaultdict(lambda: defaultdict(Counter))
    try:
        dst.execute("SET threads = 1")
        for ddl in (TILE_DDL, CELL_DDL, PHASE_DDL, JUNCTION_DDL, QC_DDL, PROV_DDL):
            dst.execute(ddl)
        species_list = [r[0] for r in ccon.execute("SELECT species_id FROM build_manifest ORDER BY species_id").fetchall()]
        ev_species = {r[0] for r in mcon.execute("SELECT DISTINCT species FROM molecule_member").fetchall()}
        for species in species_list:
            elig, dropped_score = eligible_molecules(mcon, species, score_runs) if species in ev_species else ({}, {})
            sup = inputs["protein_supplements"].get(species)
            sup_rows = _protein_rows(sup["path"], species, elig) if sup else []
            lab = build_species(mcon, species, elig, sup_rows) if elig else SpeciesLabels()
            lab._index, lab._pindex = {}, {}
            qc = qc_all[species]
            for src, n in dropped_score.items():
                qc[src]["molecules_dropped_b6_score"] += n
            for src, metrics in lab.qc.items():
                qc[src].update(metrics)
            for src, mols in lab.molecules_used.items():
                qc[src]["molecules_used"] = len(mols)
            if lab.junctions:
                dst.executemany("INSERT INTO evlabel_junction VALUES (?,?,?,?,?,?,?,?,?,?,?)", sorted(lab.junctions))
            blocks = defaultdict(list)
            for chrom, a, b, sp in ccon.execute("SELECT chromosome, start0, end0, split FROM tile_blocks WHERE species_id = ? "
                                                "ORDER BY ALL", [species]).fetchall():
                blocks[chrom].append((a, b, sp))
            cur = ccon.cursor()
            cur.execute("SELECT rn, chromosome, start, fin, is_rc, split, sequence FROM geneList WHERE species_id = ? "
                        "ORDER BY rn", [species])
            while True:
                rows = cur.fetchmany(256)
                if not rows:
                    break
                tiles, cells, phases = [], [], []
                for rn, chrom, ws, we, is_rc, split, seq in rows:
                    split = split or gc.tile_split(blocks[chrom], ws, we)
                    if split not in SCOPES:
                        qc["_all"]["tiles_test_unlabelled"] += 1
                        continue
                    scope = SCOPES[split]
                    tiles.append((rn, species, chrom, ws, we, we - ws, bool(is_rc), split, scope))
                    for c in project_tile(lab, scope, chrom, ws, we, is_rc, seq, qc):
                        cells.append((rn,) + c)
                        if not is_rc:
                            qc[c[4]][f"cells_{c[5]}"] += c[2] - c[1]
                            if c[4] == "protein" and c[0] == class_index("exon"):
                                qc["protein"]["protein_exon_cells"] += c[2] - c[1]
                    for s, e, strand, cp, aid in project_phase(lab, scope, chrom, ws, we, is_rc, seq):
                        phases.append((rn, s, e, strand, cp, aid))
                if tiles:
                    dst.executemany("INSERT INTO evlabel_tile VALUES (?,?,?,?,?,?,?,?,?)", tiles)
                if cells:
                    dst.executemany("INSERT INTO evlabel_cell VALUES (?,?,?,?,?,?,?,?)", sorted(cells))
                if phases:
                    dst.executemany("INSERT INTO evlabel_phase VALUES (?,?,?,?,?,?)", sorted(phases))
            cur.close()
        qc_rows = sorted((sp, src, m, float(v)) for sp, d in qc_all.items() for src, c in d.items() for m, v in c.items())
        if qc_rows:
            dst.executemany("INSERT INTO evlabel_qc VALUES (?,?,?,?)", qc_rows)
        prov = {"version": VERSION, "consumer_policy": CONSUMER_POLICY, "boundary_policy": BOUNDARY_POLICY,
                "boundary_note": "not final: the A50.2 consumer amendment must re-decide boundary cells and the WP2c "
                                 "gate before the segmentation experiment trains; absence of boundary cells is not a "
                                 "decision",
                "genotype_policy": GENOTYPE_POLICY, "unit_policy": UNIT_POLICY, "phase_convention": PHASE_CONVENTION,
                "classes": list(gc.SEG_CLASSES), "inputs": inputs, "argv": list(argv),
                "code_sha256": {Path(__file__).name: LK.sha256(__file__)}, "duckdb_version": duckdb.__version__}
        dst.executemany("INSERT INTO evlabel_provenance VALUES (?, ?)",
                        sorted((k, json.dumps(v, sort_keys=True)) for k, v in prov.items()))
        dst.execute("CHECKPOINT")
    finally:
        mcon.close()
        ccon.close()
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


def to_dense(con, rn, unit_weight=False):
    """(labels f32[L, 14], weights f32[L, 14]) of one tile: max across sources; ``unit_weight`` sets every positive
    weight to 1 (the mandatory A18.4 unit-weight ablation)."""
    import numpy as np

    row = con.execute("SELECT L FROM evlabel_tile WHERE rn = ?", [rn]).fetchone()
    require(row is not None, f"rn {rn} is not an evlabel tile (test tiles are never labelled)")
    L = row[0]
    n_cls = len(_gc().SEG_CLASSES)
    labels = np.zeros((L, n_cls), dtype=np.float32)
    weights = np.zeros((L, n_cls), dtype=np.float32)
    for cls, a, b, w in con.execute("SELECT class_index, start0, end0, weight FROM evlabel_cell WHERE rn = ? "
                                    "ORDER BY class_index, start0, end0, weight", [rn]).fetchall():
        labels[a:b, cls] = 1.0
        weights[a:b, cls] = np.maximum(weights[a:b, cls], 1.0 if unit_weight else w)
    return labels, weights


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
