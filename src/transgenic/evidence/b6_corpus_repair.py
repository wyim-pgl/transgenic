"""WP2 rule 5 repair of the A40 tile corpus (plan v4.1 §2): mask edge-crossing leak/hard genes by the C0 interval.

The A40 build (``build_b5._build_species_tiles``) masks only genes fully inside a tile; a held-out, test/valid or
hard-flagged gene that crosses a kept train/valid tile's edge stays as plain sequence, and its overlap partners stay
labelled.  This module repairs that without re-tiling: rows keep their ``rn``, ``geneModel`` (wid), order and
coordinates; only sequence, label and the derived columns change.

1. Edge crossers are enumerated explicitly with the predicate of ``gsf_contract.py:541`` / ``leakage.py``:
   ``g.end0 > ws and g.start0 < we and not (g.start0 >= ws and g.end0 <= we)`` over the build's genes.
2. A crosser is a seed if the build's ``leak`` predicate (gene split more restrictive than the tile split) or
   ``hard`` predicate (weight 0 under the build flags: A22 or A30) holds.  Decoy is not applied.
3. Geometry (D1 default): the seed's genome-level component interval from the ``b6-intervals-v2`` manifest
   (component span +- the largest historical member flank, else 150), clipped to the tile.  No RNG; the manifest is
   read, never recomputed.  The mask is component-level and equals C0's exclusion geometry for the shared causes.
4. Closure (A33.3): every labelled gene of the tile intersecting a final repair span is unlabelled
   (``closure_member`` if it belongs to the masked component, ``closure_flank`` if only the flank reaches it).
   ``gff`` is regenerated with ``window_to_gsf_v3``; tokens, ``n_transcripts``, ``qc_flags``, ``strict_holdout`` and
   ``window_genes`` follow.
5. The RC twin is ``revcomp`` of the repaired forward tile with ``reverse_complement_v3`` of the regenerated GSF.
6. Two events would change the row set -- a train/valid tile above masked fraction 0.60, and an RC twin whose tile
   loses its last labelled gene (or, under ``isoform-only``, its last multi-isoform gene) -- and ``apply`` refuses
   if either is non-zero (N_pilot = 4,142 rests on 198,828 train rows; a further amendment is then required).

Every tile that is repaired is first verified against the replayed A40 masking (``leakage.verify_tile``), its
labelled genes against ``window_genes`` and its stored GSF, and its RC twin against the forward row.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from transgenic.evidence import leakage as LK

RULE_ID = "rule5-leak-hard-c0interval-v1"
CORPUS_VERSION = "b5_full_b6_v1"
DROP_CAUSES = ("closure_member", "closure_flank")
EVENTS = ("tiles_over_mask_fraction", "rc_twin_orphaned")
B6_QC_SPANS = "b6_rule5_masked"
B6_QC_DROPS = "b6_label_drop"
REPAIR_COLUMNS = ("rn", "wid", "species_id", "chromosome", "start0", "end0", "component_id", "cause", "scope", "flank",
                  "flank_source", "is_rc")
REPAIR_DDL = ("CREATE TABLE b6_dna_repair (rn INTEGER NOT NULL, wid VARCHAR NOT NULL, species_id VARCHAR NOT NULL, "
              "chromosome VARCHAR NOT NULL, start0 BIGINT NOT NULL, end0 BIGINT NOT NULL, component_id VARCHAR NOT NULL, "
              "cause VARCHAR NOT NULL, scope VARCHAR NOT NULL, flank INTEGER NOT NULL, flank_source VARCHAR NOT NULL, "
              "is_rc BOOLEAN NOT NULL)")
LABEL_DROP_DDL = "CREATE TABLE b6_label_drop (rn INTEGER NOT NULL, gene_id VARCHAR NOT NULL, cause VARCHAR NOT NULL)"
META_DDL = "CREATE TABLE b6_corpus_meta (key VARCHAR NOT NULL PRIMARY KEY, value VARCHAR NOT NULL)"
BATCH = 256


class RepairError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise RepairError(message)


def _gc():
    return LK.load_build_b5().gc


# ----------------------------------------------------------------------------------------- manifest components
@dataclass(frozen=True)
class Component:
    species: str
    chrom: str
    component_id: str
    members: tuple
    start0: int
    end0: int
    flank: int
    flank_source: str
    causes: tuple            # ((cause, scope, c0_exclusion), ...) sorted


class ComponentIndex:
    """Manifest rows grouped into components; gene -> component per species.  Fails closed on a gene in two
    components or a component whose rows disagree on geometry."""

    def __init__(self, rows):
        groups = defaultdict(list)
        for r in rows:
            groups[(r["species"], r["chrom"], r["component_id"])].append(r)
        self.components, self.by_gene = {}, {}
        for key, rs in sorted(groups.items()):
            geo = {(r["start0"], r["end0"], tuple(r["members"]), r["flank"], r["flank_source"]) for r in rs}
            require(len(geo) == 1, f"interval manifest component {key} has rows with different geometry: {sorted(geo)}")
            (a, b, members, flank, source), = geo
            comp = Component(key[0], key[1], key[2], members, a, b, flank, source,
                             tuple(sorted((r["cause"], r["scope"], r["c0_exclusion"]) for r in rs)))
            self.components[key] = comp
            for gid in members:
                gk = (key[0], gid)
                require(gk not in self.by_gene, f"gene {gk} is a member of two manifest components")
                self.by_gene[gk] = comp

    def of(self, species, gene_id):
        return self.by_gene.get((species, gene_id))

    def species(self):
        return sorted({k[0] for k in self.components})


def load_manifests(paths, *, relative_to=None):
    """Read every ``b6-intervals-v2`` file (canonical, sidecar-verified); one file per species.

    ``relative_to``: record each path relative to this directory (the freeze's, decision 14) so the freeze and its
    manifests can move hosts together; consumers resolve it against the freeze's actual location
    (``b6_bindings.freeze_manifest_path``).  Otherwise the absolute path is recorded (measure report)."""
    rows, shas = [], {}
    for p in paths:
        part = LK.read_interval_manifest(p)
        species = sorted({r["species"] for r in part})
        require(len(species) == 1, f"{p}: an interval manifest must hold exactly one species, found {species}")
        require(species[0] not in shas, f"two interval manifests for {species[0]}")
        path = Path(p).absolute()
        if relative_to is not None:
            path = Path(os.path.relpath(path, Path(relative_to).absolute()))
        shas[species[0]] = {"path": path.as_posix(), "sha256": LK.interval_manifest_sha256(part)}
        rows += part
    return ComponentIndex(rows), shas


# ----------------------------------------------------------------------------------------- per-tile repair
@dataclass
class TileRepair:
    rn: int
    wid: str
    species: str
    chrom: str
    ws: int
    we: int
    split: str
    components: list                      # Component, sorted by (start0, component_id)
    spans: list                           # (start0, end0, Component) genomic, clipped to the tile
    labelled_before: list
    labelled_after: list
    drops: list                           # (gene_id, cause)
    seq: str = ""
    gsf: str = ""
    tokens: int = 0
    n_transcripts: int = 0
    qc_flags: str | None = None
    strict_holdout: bool = False
    masked_fraction: float = 0.0
    over_mask: bool = False
    rc: dict | None = None                # {"rn", "wid", "seq", "gsf", "tokens"} of the RC twin (new values)
    rc_orphan: str | None = None          # 'last_labelled_gene' | 'isoform_condition'
    edge_unmasked: int = 0

    def repair_rows(self):
        out = []
        for a, b, comp in self.spans:
            for cause, scope, _ in comp.causes:
                out.append((self.rn, self.wid, self.species, self.chrom, a, b, comp.component_id, cause, scope,
                            comp.flank, comp.flank_source, False))
                if self.rc is not None:
                    out.append((self.rc["rn"], self.rc["wid"], self.species, self.chrom, a, b, comp.component_id,
                                cause, scope, comp.flank, comp.flank_source, True))
        return out

    def drop_rows(self):
        out = [(self.rn, g, c) for g, c in self.drops]
        if self.rc is not None:
            out += [(self.rc["rn"], g, c) for g, c in self.drops]
        return out


def edge_crossers(genes, ws, we):
    return [g for g in genes if g.end0 > ws and g.start0 < we and not (g.start0 >= ws and g.end0 <= we)]


def is_seed(gene_meta, gene, tile_split):
    gc = _gc()
    m = gene_meta[gene.gene_id]
    leak = bool(m["split"]) and gc.SPLIT_RANK.get(m["split"], 0) > gc.SPLIT_RANK[tile_split]
    return leak or m["weight"] == 0


def _qc_parts(qc_flags):
    parts = qc_flags.split(";") if qc_flags else []
    return [p for p in parts if "=" not in p], [p for p in parts if "=" in p]


def _rc_condition(rc_mode, labelled):
    if not labelled:
        return False
    if rc_mode == "all":
        return True
    if rc_mode == "isoform-only":
        return any(len(g.transcripts) >= 2 for g in labelled)
    return False


class SpeciesContext:
    """Everything the repair needs for one species, from the frozen corpus and verified inputs."""

    def __init__(self, con, species, *, splits, flags, build_flags, seed, components, reference=None, gff=None):
        b5 = LK.load_build_b5()
        self.species, self.seed, self.components = species, seed, components
        self.manifest = LK.corpus_manifest(con, species)
        gc = b5.gc
        require(self.manifest["window_policy"] == gc.WINDOW_POLICY_V3,
                f"{species}: corpus window_policy {self.manifest['window_policy']!r} is not {gc.WINDOW_POLICY_V3!r}")
        require(LK.sha256(splits) == self.manifest["split_file_sha256"],
                f"{species}: --splits sha256 differs from the split file the corpus was built from")
        self.gff = gff or self.manifest["gff"]
        self.fasta = reference or self.manifest["fasta"]
        require(os.path.isfile(self.gff) and LK.sha256(self.gff) == self.manifest["gff_sha256"],
                f"{species}: GFF {self.gff} missing or sha256 != corpus build_manifest")
        require(os.path.isfile(self.fasta) and LK.sha256(self.fasta) == self.manifest["fasta_sha256"],
                f"{species}: reference {self.fasta} missing or sha256 != corpus build_manifest")
        rc_modes = con.execute("SELECT DISTINCT rc_mode FROM build_manifest WHERE species_id = ?", [species]).fetchall()
        require(len(rc_modes) == 1, f"{species}: build_manifest rc_mode is not unique: {rc_modes}")
        self.rc_mode = rc_modes[0][0]
        self.genome = b5.load_fasta(self.fasta)
        self.prep = LK.prepare_genes(species=species, gff_path=self.gff, splits_path=splits, flags_path=flags,
                                     build_flags=build_flags, genome=self.genome)
        self.gene_by_id = {g.gene_id: g for gs in self.prep.by_chrom.values() for g in gs}
        self.blocks = defaultdict(list)
        for chrom, a, b, sp in con.execute("SELECT chromosome, start0, end0, split FROM tile_blocks WHERE species_id = ? "
                                           "ORDER BY chromosome, start0, end0", [species]).fetchall():
            self.blocks[chrom].append((a, b, sp))
        require(self.blocks, f"{species}: corpus has no tile_blocks")


def _expected_split(ctx, rn, chrom, ws, we, split):
    gc = _gc()
    expected = gc.tile_split(ctx.blocks[chrom], ws, we)
    if split is None:
        inside, _ = gc.genes_in_window(ctx.prep.by_chrom.get(chrom, []), ws, we)
        explicit = any(a < we and b > ws for a, b, _ in ctx.blocks[chrom])
        require(explicit and not inside, f"tile {rn} ({chrom}:{ws}-{we}) has split NULL and no explicit expected split")
    require(split in (expected, None), f"tile {rn}: stored split {split} != recomputed {expected}")
    return expected


def candidate_tiles(con, ctx, qc):
    """Pass 1 (no sequences): kept train/valid forward tiles with >= 1 seed edge crosser, in rn order."""
    out = []
    cur = con.cursor()
    try:
        cur.execute("SELECT rn, geneModel, chromosome, start, fin, split FROM geneList WHERE species_id = ? AND NOT is_rc "
                    "ORDER BY rn", [ctx.species])
        while True:
            rows = cur.fetchmany(4096)
            if not rows:
                break
            for rn, wid, chrom, ws, we, split in rows:
                qc["tiles_forward"] += 1
                expected = _expected_split(ctx, rn, chrom, ws, we, split)
                if expected not in ("train", "valid"):
                    qc["tiles_test_untouched"] += 1
                    continue
                edge = edge_crossers(ctx.prep.by_chrom.get(chrom, []), ws, we)
                seeds = [g for g in edge if is_seed(ctx.prep.gene_meta, g, expected)]
                if not seeds:
                    qc["edge_partial_unmasked"] += len(edge)
                    continue
                comps = {}
                for g in seeds:
                    comp = ctx.components.of(ctx.species, g.gene_id)
                    require(comp is not None, f"tile {rn} ({wid}): seed edge crosser {g.gene_id} has no component in "
                                              "the interval manifest (manifest from another corpus or build?)")
                    comps[comp.component_id] = comp
                seed_ids = {g.gene_id for g in seeds}
                unmasked = [g for g in edge if g.gene_id not in seed_ids
                            and not any(g.gene_id in c.members for c in comps.values())]
                out.append((rn, wid, chrom, ws, we, expected,
                            sorted(comps.values(), key=lambda c: (c.start0, c.component_id)), len(unmasked)))
    finally:
        cur.close()
    return out


def repair_tile(ctx, row, rc_row, window_genes, cand):
    """Pass 2 for one candidate: verify the A40 tile, compute the repaired forward tile and its RC twin."""
    gc = _gc()
    b5 = LK.load_build_b5()
    rn, wid, chrom, ws, we, split, comps, edge_unmasked = cand
    _, seq, gsf, qc_flags, strict = row
    L = we - ws
    tm = LK.tile_masks(ctx.prep, seed=ctx.seed, species=ctx.species, chrom=chrom, ws=ws, we=we, split=split)
    try:
        LK.verify_tile(tm, ctx.genome, rn=rn, chrom=chrom, ws=ws, we=we, seq=seq, qc_flags=qc_flags)
    except LK.LeakageError as exc:
        raise RepairError(f"A40 tile does not replay: {exc}") from exc
    gm = ctx.prep.gene_meta
    labelled, _ = gc.collapse_duplicate_genes([g for g in tm.inside if gm[g.gene_id]["labelable"]
                                               and g.gene_id not in tm.masked_ids])
    fwd_ids = sorted(g for g, is_rc in window_genes if not is_rc)
    require(sorted(g.gene_id for g in labelled) == fwd_ids,
            f"tile {rn} ({wid}): window_genes {fwd_ids} != replayed labelled genes {sorted(g.gene_id for g in labelled)}")
    require(gc.window_to_gsf_v3(labelled, ws) == gsf, f"tile {rn} ({wid}): stored gff != regenerated A40 GSF")
    flag_parts, count_parts = _qc_parts(qc_flags)
    require(flag_parts == sorted({f for g in labelled for f in gm[g.gene_id]["qc"]}),
            f"tile {rn} ({wid}): qc_flags gene flags {flag_parts} do not replay")
    require(bool(strict) == any(gm[g.gene_id]["strict"] for g in labelled), f"tile {rn} ({wid}): strict_holdout drift")

    spans = []
    chars = list(seq)
    for comp in comps:
        a, b = max(ws, comp.start0), min(we, comp.end0)
        require(a < b, f"tile {rn} ({wid}): component {comp.component_id} interval does not reach the tile")
        spans.append((a, b, comp))
        chars[a - ws:b - ws] = "N" * (b - a)
    new_seq = "".join(chars)
    drops = []
    for g in labelled:
        hits = [c for a, b, c in spans if g.start0 < b and g.end0 > a]
        if hits:
            drops.append((g.gene_id, "closure_member" if any(g.gene_id in c.members for c in hits) else "closure_flank"))
    dropped = {g for g, _ in drops}
    after = [g for g in labelled if g.gene_id not in dropped]
    new_gsf = gc.window_to_gsf_v3(after, ws)
    try:
        gc.check_caps_v3(new_gsf, window_len=L)
    except gc.CapError as exc:
        raise RepairError(f"tile {rn} ({wid}): regenerated GSF breaks the v3 caps: {exc}") from exc
    new_flags = sorted({f for g in after for f in gm[g.gene_id]["qc"]})
    extra = [f"{B6_QC_SPANS}={len(spans)}"] + ([f"{B6_QC_DROPS}={len(drops)}"] if drops else [])
    rep = TileRepair(rn, wid, ctx.species, chrom, ws, we, split, comps, spans, labelled, after, drops)
    rep.seq, rep.gsf, rep.tokens = new_seq, new_gsf, gc.count_tokens_v3(new_gsf)
    rep.n_transcripts = sum(len(g.transcripts) for g in after)
    rep.qc_flags = ";".join(new_flags + count_parts + extra) or None
    rep.strict_holdout = any(gm[g.gene_id]["strict"] for g in after)
    rep.masked_fraction = gc.masked_fraction(new_seq)
    rep.over_mask = rep.masked_fraction > gc.MASK_FRACTION_MAX
    rep.edge_unmasked = edge_unmasked

    had_twin = _rc_condition(ctx.rc_mode, labelled)
    require((rc_row is not None) == had_twin,
            f"tile {rn} ({wid}): RC twin present={rc_row is not None} but the build's rc condition gives {had_twin}")
    if rc_row is not None:
        rc_rn, rc_wid, rc_seq, rc_gsf = rc_row
        require(rc_seq == b5.revcomp(seq), f"tile {rn}: RC twin {rc_rn} sequence != revcomp(forward)")
        require(rc_gsf == gc.reverse_complement_v3(gsf, L), f"tile {rn}: RC twin {rc_rn} gff != reverse_complement_v3")
        rc_ids = sorted(g for g, is_rc in window_genes if is_rc)
        require(rc_ids == fwd_ids, f"tile {rn}: RC window_genes {rc_ids} != forward {fwd_ids}")
        rgsf = gc.reverse_complement_v3(new_gsf, L)
        rep.rc = {"rn": rc_rn, "wid": rc_wid, "seq": b5.revcomp(new_seq), "gsf": rgsf, "tokens": gc.count_tokens_v3(rgsf)}
        if not _rc_condition(ctx.rc_mode, after):
            rep.rc_orphan = "last_labelled_gene" if not after else "isoform_condition"
    return rep


def iter_species_repairs(con, ctx, qc):
    """Yield TileRepair in rn order for one species (streaming: BATCH candidates at a time)."""
    cands = candidate_tiles(con, ctx, qc)
    look = con.cursor()
    try:
        for i in range(0, len(cands), BATCH):
            batch = cands[i:i + BATCH]
            rns = [c[0] for c in batch]
            wids = [c[1] for c in batch]
            rows = {r[0]: r for r in look.execute(
                "SELECT rn, sequence, gff, qc_flags, strict_holdout FROM geneList WHERE list_contains(?, rn) ORDER BY rn",
                [rns]).fetchall()}
            rc_rows = {r[1][:-3]: r for r in look.execute(
                "SELECT rn, geneModel, sequence, gff FROM geneList WHERE species_id = ? AND is_rc "
                "AND list_contains(?, geneModel) ORDER BY rn", [ctx.species, [w + "-rc" for w in wids]]).fetchall()}
            wg = defaultdict(list)
            for window_id, gene_id, is_rc in look.execute(
                    "SELECT window_id, gene_id, is_rc FROM window_genes WHERE species_id = ? AND list_contains(?, window_id) "
                    "ORDER BY window_id, is_rc, gene_id", [ctx.species, wids]).fetchall():
                wg[window_id].append((gene_id, is_rc))
            for cand in batch:
                require(cand[0] in rows, f"tile {cand[0]} vanished between passes")
                yield repair_tile(ctx, rows[cand[0]], rc_rows.get(cand[1]), wg.get(cand[1], []), cand)
    finally:
        look.close()


# ----------------------------------------------------------------------------------------- measure / apply
@dataclass
class Report:
    species: dict = field(default_factory=dict)       # species -> qc counters
    counts: dict = field(default_factory=dict)        # species -> split -> cause -> {"spans", "label_drops"}
    events: dict = field(default_factory=lambda: {e: 0 for e in EVENTS})
    event_tiles: list = field(default_factory=list)   # (species, rn, wid, event, detail)
    declared_differences: dict = field(default_factory=dict)

    def add(self, rep, qc):
        c = self.counts.setdefault(rep.species, {}).setdefault(rep.split, {})
        for _, _, comp in rep.spans:
            for cause, _, _ in comp.causes:
                c.setdefault(cause, {"spans": 0, "label_drops": 0})["spans"] += 1
        for _, cause in rep.drops:
            c.setdefault(cause, {"spans": 0, "label_drops": 0})["label_drops"] += 1
        qc["tiles_repaired"] += 1
        qc["repair_spans"] += len(rep.spans)
        qc["label_drops"] += len(rep.drops)
        qc["rc_twins_repaired"] += rep.rc is not None
        qc["edge_partial_unmasked"] += rep.edge_unmasked
        for _, _, comp in rep.spans:
            if all(not c0 for _, _, c0 in comp.causes):
                qc["a30_only_spans"] += 1
        if rep.over_mask:
            self.events["tiles_over_mask_fraction"] += 1
            self.event_tiles.append((rep.species, rep.rn, rep.wid, "tiles_over_mask_fraction",
                                     f"{rep.masked_fraction:.4f}"))
        if rep.rc_orphan:
            self.events["rc_twin_orphaned"] += 1
            self.event_tiles.append((rep.species, rep.rn, rep.wid, "rc_twin_orphaned", rep.rc_orphan))

    def as_dict(self):
        return {"rule_id": RULE_ID, "species_qc": self.species, "counts": self.counts, "events": self.events,
                "event_tiles": [list(t) for t in self.event_tiles], "declared_differences": self.declared_differences}


def _open_ro(path):
    import duckdb

    require(os.path.isfile(path), f"corpus missing: {path}")
    require(not os.path.exists(str(path) + ".wal"), f"{path} has a WAL; supply a closed/checkpointed snapshot")
    return duckdb.connect(str(path), read_only=True)


def _species(con):
    return [r[0] for r in con.execute("SELECT species_id FROM build_manifest ORDER BY species_id").fetchall()]


def _contexts(con, *, splits, flags, build_flags, seed, components, references, gffs):
    for species in _species(con):
        yield SpeciesContext(con, species, splits=splits, flags=flags, build_flags=build_flags, seed=seed,
                             components=components, reference=(references or {}).get(species),
                             gff=(gffs or {}).get(species))


def _require_manifest_coverage(con, components):
    have = set(components.species())
    for species in _species(con):
        n = con.execute("SELECT count(*) FROM geneList WHERE species_id = ? AND split IN ('train', 'valid')",
                        [species]).fetchone()[0]
        require(species in have or n == 0, f"{species}: no interval manifest supplied for a species with train/valid tiles")


def measure(base, *, splits, flags, build_flags, seed, manifests, references=None, gffs=None):
    components, shas = load_manifests(manifests)
    report = Report()
    con = _open_ro(base)
    try:
        _require_manifest_coverage(con, components)
        for ctx in _contexts(con, splits=splits, flags=flags, build_flags=build_flags, seed=seed, components=components,
                             references=references, gffs=gffs):
            qc = defaultdict(int)
            for rep in iter_species_repairs(con, ctx, qc):
                report.add(rep, qc)
            report.species[ctx.species] = dict(sorted(qc.items()))
            report.declared_differences[ctx.species] = {
                "a30_only_spans_masked_not_c0_excluded": qc["a30_only_spans"],
                "edge_partial_c0_excluded_not_masked": qc["edge_partial_unmasked"]}
    finally:
        con.close()
    out = report.as_dict()
    out["interval_manifests"] = shas
    return report, out


def _code_shas():
    here = Path(__file__).resolve()
    root = here.parents[3]
    files = [here, Path(LK.__file__).resolve(), root / "src" / "transgenic" / "datasets" / "build_b5.py",
             root / "src" / "transgenic" / "utils" / "gsf_contract.py",
             root / "revision" / "scripts" / "evidence" / "b6_build_corpus.py"]
    return {str(p.relative_to(root)): LK.sha256(p) for p in files if p.is_file()}


def _row_counts(con):
    rows = con.execute("SELECT species_id, coalesce(split, 'null'), is_rc, count(*) FROM geneList "
                       "GROUP BY ALL ORDER BY ALL").fetchall()
    out = {"total": sum(r[3] for r in rows), "by_species_split_rc": {}}
    for sp, split, is_rc, n in rows:
        out["by_species_split_rc"].setdefault(sp, {}).setdefault(split, {})["rc" if is_rc else "forward"] = n
    out["train"] = sum(r[3] for r in rows if r[1] == "train")
    return out


def apply(base, out, freeze, *, splits, flags, build_flags, seed, manifests, references=None, gffs=None, argv=()):
    """Refuse on any row-set event (measured first), then copy A40 -> ``out`` and update rows in place."""
    import duckdb

    out, freeze = Path(out).absolute(), Path(freeze).absolute()
    for p in (out, freeze):
        require(not os.path.lexists(p), f"{p} exists; refusing to overwrite")
    report, measured = measure(base, splits=splits, flags=flags, build_flags=build_flags, seed=seed,
                               manifests=manifests, references=references, gffs=gffs)
    bad = {k: v for k, v in report.events.items() if v}
    require(not bad, f"--apply refused: row-set events {bad} (tiles {report.event_tiles[:10]}); the train rows would "
                     "leave 198,828 (the N_pilot = 4,142 basis, P:999) -- a further amendment is required")
    base_sha = LK.sha256(base)
    components, shas = load_manifests(manifests, relative_to=freeze.parent)
    tmp = Path(str(out) + ".partial")
    require(not os.path.lexists(tmp) and not os.path.lexists(str(tmp) + ".wal"), f"{tmp} exists; remove it")
    shutil.copyfile(base, tmp)
    require(LK.sha256(tmp) == base_sha, "copy of the base corpus does not hash to the base sha256")
    src = _open_ro(base)
    dst = duckdb.connect(str(tmp))
    try:
        dst.execute("SET threads = 1")
        dst.execute(REPAIR_DDL)
        dst.execute(LABEL_DROP_DDL)
        dst.execute(META_DDL)
        repair_rows, drop_rows = [], []
        for ctx in _contexts(src, splits=splits, flags=flags, build_flags=build_flags, seed=seed, components=components,
                             references=references, gffs=gffs):
            qc = defaultdict(int)
            pending = []
            for rep in iter_species_repairs(src, ctx, qc):
                require(not rep.over_mask and not rep.rc_orphan, f"tile {rep.rn}: row-set event appeared between passes")
                pending.append(rep)
                if len(pending) >= BATCH:
                    _write_batch(dst, pending, repair_rows, drop_rows)
                    pending = []
            _write_batch(dst, pending, repair_rows, drop_rows)
        repair_rows.sort()
        drop_rows.sort()
        if repair_rows:
            dst.executemany("INSERT INTO b6_dna_repair VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", repair_rows)
        if drop_rows:
            dst.executemany("INSERT INTO b6_label_drop VALUES (?,?,?)", drop_rows)
        meta = {"rule_id": RULE_ID, "corpus_version": CORPUS_VERSION, "base_sha256": base_sha, "seed": str(seed),
                "interval_manifest_sha256": json.dumps({k: v["sha256"] for k, v in sorted(shas.items())},
                                                       sort_keys=True)}
        dst.executemany("INSERT INTO b6_corpus_meta VALUES (?, ?)", sorted(meta.items()))
        row_counts = _row_counts(dst)
        dst.execute("CHECKPOINT")
    finally:
        src.close()
        dst.close()
    require(not os.path.exists(str(tmp) + ".wal"), f"{tmp}.wal remains after CHECKPOINT; refusing to publish")
    base_counts = _row_counts_path(base)
    require(base_counts == row_counts, f"row counts changed: {base_counts} -> {row_counts}")
    os.link(tmp, out)
    tmp.unlink()
    out_sha = LK.sha256(out)
    content = content_sha256(out)
    record = {"rule_id": RULE_ID, "corpus_version": CORPUS_VERSION, "base": str(Path(base).absolute()),
              "base_sha256": base_sha, "out": str(out), "out_sha256": out_sha, "content_sha256": content,
              "interval_manifest_sha256": {k: v["sha256"] for k, v in sorted(shas.items())},
              "interval_manifests": shas, "seed": seed, "code_sha256": _code_shas(), "argv": list(argv),
              "inputs": _input_shas(splits, flags, build_flags),
              "species_inputs": _species_inputs(base, references, gffs),
              "counts": measured["counts"], "species_qc": measured["species_qc"],
              "declared_differences": measured["declared_differences"], "events": measured["events"],
              "row_counts": row_counts, "rows_repaired": len({r[0] for r in repair_rows}),
              "b6_dna_repair_rows": len(repair_rows), "b6_label_drop_rows": len(drop_rows),
              "duckdb_version": duckdb.__version__,
              "geometry": "D1 default: C0 rule 4(a) component interval (max historical member flank, else 150), no RNG",
              "grandfathered": "A40 per-gene flanks (build_b5.py:600, leak_mask :654-663) stay as built; not claimed "
                               "as literal A33.3 compliance"}
    with open(freeze, "x") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return record


def _row_counts_path(path):
    con = _open_ro(path)
    try:
        return _row_counts(con)
    finally:
        con.close()


def _input_shas(splits, flags, build_flags):
    out = {"splits": {"path": str(Path(splits).absolute()), "sha256": LK.sha256(splits)},
           "flags": {"path": str(Path(flags).absolute()), "sha256": LK.sha256(flags)}}
    for i, p in enumerate(build_flags or []):
        out[f"build_flags_{i}"] = {"path": str(Path(p).absolute()), "sha256": LK.sha256(p)}
    return out


def _species_inputs(base, references, gffs):
    con = _open_ro(base)
    try:
        out = {}
        for species in _species(con):
            m = LK.corpus_manifest(con, species)
            out[species] = {"fasta": (references or {}).get(species, m["fasta"]), "fasta_sha256": m["fasta_sha256"],
                            "gff": (gffs or {}).get(species, m["gff"]), "gff_sha256": m["gff_sha256"],
                            "split_file_sha256": m["split_file_sha256"], "build_version": m["build_version"]}
        return out
    finally:
        con.close()


def _write_batch(dst, reps, repair_rows, drop_rows):
    if not reps:
        return
    upd = []
    dels = []
    for rep in reps:
        upd.append((rep.seq, rep.gsf, rep.tokens, rep.n_transcripts, rep.qc_flags, rep.strict_holdout, rep.rn))
        if rep.rc is not None:
            upd.append((rep.rc["seq"], rep.rc["gsf"], rep.rc["tokens"], rep.n_transcripts, rep.qc_flags,
                        rep.strict_holdout, rep.rc["rn"]))
        dels += [(rep.species, rep.wid, g) for g, _ in rep.drops]
        repair_rows += rep.repair_rows()
        drop_rows += rep.drop_rows()
    dst.executemany("UPDATE geneList SET sequence = ?, gff = ?, gsf_token_count = ?, n_transcripts = ?, qc_flags = ?, "
                    "strict_holdout = ? WHERE rn = ?", upd)
    if dels:
        dst.executemany("DELETE FROM window_genes WHERE species_id = ? AND window_id = ? AND gene_id = ?", dels)


FREEZE_REQUIRED_KEYS = ("rule_id", "base_sha256", "out_sha256", "interval_manifest_sha256")


def verify_freeze(freeze):
    """The corpus freeze contract consumers rely on: the required keys, this rule id, and
    ``interval_manifest_sha256`` as the canonical species-indexed mapping {species: sha256}."""
    missing = [k for k in FREEZE_REQUIRED_KEYS if k not in freeze]
    require(not missing, f"corpus freeze lacks required key(s) {missing}")
    require(freeze["rule_id"] == RULE_ID, f"freeze rule_id {freeze['rule_id']!r} != {RULE_ID!r}")
    for k in ("base_sha256", "out_sha256"):
        require(isinstance(freeze[k], str) and len(freeze[k]) == 64, f"freeze {k} is not a sha256")
    man = freeze["interval_manifest_sha256"]
    require(isinstance(man, dict) and man, "freeze interval_manifest_sha256 must be a non-empty {species: sha256} map")
    for sp, sha in man.items():
        require(isinstance(sha, str) and len(sha) == 64, f"freeze interval_manifest_sha256[{sp!r}] is not a sha256")
    return freeze


def freeze_manifest_sha256(freeze, species):
    """Per-species consumer lookup: the interval-manifest sha the freeze records for ``species`` (fails closed)."""
    man = verify_freeze(freeze)["interval_manifest_sha256"]
    require(species in man, f"freeze records no interval manifest for {species} (species {sorted(man)})")
    return man[species]


def file_sha256(path):
    return LK.sha256(path)


def content_sha256(path):
    """Canonical content hash of a corpus DB: every main-schema table (sorted by name) as its column (name, type)
    list and its rows in a total order (``geneList`` by rn, others ORDER BY ALL), one canonical JSON line per row.
    DuckDB does not write byte-identical files for identical content (block layout/header), so rebuild identity is
    asserted on this hash; ``out_sha256`` (the file bytes) remains the binding every consumer verifies."""
    con = _open_ro(path)
    h = hashlib.sha256()
    try:
        tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' "
                                            "AND table_type = 'BASE TABLE' ORDER BY table_name").fetchall()]
        for t in tables:
            cols = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
                               "ORDER BY ordinal_position", [t]).fetchall()
            h.update(json.dumps(["table", t, cols]).encode() + b"\n")
            order = "rn" if t == "geneList" else "ALL"
            cur = con.cursor()
            cur.execute(f'SELECT * FROM "{t}" ORDER BY {order}')
            while True:
                rows = cur.fetchmany(1024)
                if not rows:
                    break
                for r in rows:
                    h.update(json.dumps(list(r), default=str, separators=(",", ":")).encode() + b"\n")
            cur.close()
    finally:
        con.close()
    return h.hexdigest()


def canonical_json_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
