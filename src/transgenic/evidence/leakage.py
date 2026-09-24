"""A46 rule 4 leakage closure for the B6 C0 build (observation level, before any raster).

(a) ``build_leakage_model`` -- excluded intervals per species, all 0-based half-open:

    * ``heldout`` / ``test_orthogroup`` / ``a22_hard`` (scope ``all``) and
      ``valid_orthogroup`` (scope ``train_tiles``): the genome-level A33 overlap
      component of every seed gene (strict held-out locus, gene of a test / valid
      orthogroup, A22 hard-flagged gene), padded by the seeded flank.  The flank
      is the build's own draw: ``build_b5.py:577-601`` draws one flank per masked
      gene per tile from ``random.Random(f"{seed}:{species}:{chrom}:{ws}-{we}:flank")``
      (in ``inside`` order); the genome-level interval uses the largest flank the
      build drew for any gene of the component, and ``FLANK_RANGE[1]`` (150) when
      no tile masked it (conservative upper bound of the seeded range).
    * ``test_block`` (scope ``all``): A29 test blocks from ``tile_blocks``.
    * ``tile_leak`` / ``tile_hard`` / ``tile_decoy`` / ``tile_component``: every
      gene the corpus build N-masked in a kept train/valid tile because of a
      rule-4(a) seed (leak, A22-hard, decoy, and their overlap components; an
      A30-only mask is counted but yields no interval), padded by its exact
      per-tile seeded flank and clipped to the tile.  Scope ``train_tiles`` for
      a train tile, ``all`` for a valid tile (``edge_partial`` likewise).  The
      per-tile masking is recomputed with the build's own code path and seeds and
      is VERIFIED against the stored tile (qc_flags counts and the exact N-masked
      sequence); any mismatch fails closed.
    * ``edge_partial``: the part inside a kept train/valid tile of every gene
      crossing that tile's edge.

  Scope ``train_tiles`` intervals do not make a molecule ``excluded_heldout``;
  they set ``train_tiles_excluded`` (the molecule may be rastered only into
  valid tiles).  Every ``all``-scope hit excludes the molecule from all
  training-side tracks.

  A22 hard-flagged means ``build_b5.loss_mask_decision`` -> weight 0 on the
  ``--flags`` (GeenuFF) file only; A30 is NOT a rule-4(a) cause.  The corpus
  build itself may have used A22 + A30 flags; ``build_flags`` names the files it
  used, only to reproduce its masking and decoy rate.

(b) ``decide_molecule`` / ``decide_molecules`` -- molecule-wide exclusion
    (placements of every member, frozen primary/supplementary + audit
    primary/supplementary/secondary): cause (a) interval overlap on either
    strand; unauditable if the MOLECULE has >= N (=50) secondary placements
    (summed over its members, the frozen wording "a molecule with >= N
    placements"), or any placement has MAPQ < 5 and overlaps no excluded
    interval.  Exhaustiveness = fraction of molecules with fewer than N
    secondary placements; it must be >= 99.9 % or ``ExhaustivenessError`` is
    raised (no tracks).  Two behaviours were settled by author ruling 2026-09-23 (both = the literal
    rule text) and are single named constants: ``LOW_MAPQ_APPLIES_TO_AUDIT_SECONDARIES`` and
    ``EDGE_PARTIAL_MOLECULE_WIDE``; QC counts their effect.

(c) ``coordinate_exclusions`` -- intron boundary bases shared by a seed-excluded
    gene and a non-seed neighbour; junction support at those coordinates is
    excluded (flagged in ``junction_support.excluded_4c``).

(d) ``heldout_qc`` -- independent recount of molecules overlapping strict
    held-out loci: overlapping == excluded and retained == 0.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import os
import random
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

AUDIT_N = 50
AUDIT_MAPQ_MIN = 5
# AUTHOR RULING 2026-09-23 (a), "적용" — the literal rule 4(b) reading: the MAPQ < 5 unauditability test is applied to audit SECONDARY placements too
# (not only to primary/supplementary ones).  QC ``excluded_b_cause_secondary_mapq`` counts the molecules whose
# MAPQ flag comes from secondary placements alone, i.e. the molecules this ruling decides.
LOW_MAPQ_APPLIES_TO_AUDIT_SECONDARIES = True
# AUTHOR RULING 2026-09-23 (b), "분자 전체 적용" — molecule-wide, the literal rule 4(a) reading: an ``edge_partial`` interval (part of a gene crossing a kept tile's edge) is applied
# MOLECULE-WIDE like every other rule-4(a) interval (any member placement touching it flags the whole molecule).
# False would drop edge_partial from the molecule decision (left to the raster step).  QC
# ``excluded_a_cause_edge_partial`` / ``train_tiles_cause_edge_partial`` count its effect.
EDGE_PARTIAL_MOLECULE_WIDE = True
MIN_EXHAUSTIVENESS = 0.999
# Per (species x source) exhaustiveness floor: author decision 2026-09-23 (A49 decision 2) -- BOTH the species-level
# rule 4(b) gate (0.999) AND a per-source floor apply.  The value 0.99 is A49.2 (the author delegated the number to a
# Codex consultation, 2026-09-23): up to 1 % source-specific capped loss before an audit rerun, with the unchanged
# species gate still bounding aggregate loss at 0.1 %; no small-denominator exemption.  Enforced at build time
# (Exhaustiveness.check, nothing published below it) and re-checked by the merge from the recorded per-source
# counters (source_exhaustiveness).
MIN_SOURCE_EXHAUSTIVENESS = 0.99
TILE_QC_KEYS = ("leak_masked", "hard_masked", "decoy_masked", "component_masked", "edge_partial")


class LeakageError(ValueError):
    pass


class ExhaustivenessError(LeakageError):
    pass


def require(ok, message):
    if not ok:
        raise LeakageError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_B5 = None


def load_build_b5():
    """``build_b5`` without the torch-importing ``transgenic.datasets`` package (as tests/test_build_b5.py loads it)."""
    global _B5
    if _B5 is None:
        path = Path(__file__).resolve().parents[1] / "datasets" / "build_b5.py"
        mod = types.ModuleType("_c0_build_b5")
        mod.__file__ = str(path)
        sys.modules[mod.__name__] = mod
        exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
        _B5 = mod
    return _B5


@dataclass(frozen=True)
class ExcludedInterval:
    chrom: str
    start0: int
    end0: int
    cause: str
    scope: str  # 'all' | 'train_tiles'
    gene_id: str | None = None
    tile_rn: int | None = None


class IntervalIndex:
    """Overlap queries over excluded intervals, per (chrom, scope); strand-blind by construction (rule 4(b))."""

    def __init__(self, intervals):
        groups = defaultdict(list)
        for iv in intervals:
            require(iv.scope in ("all", "train_tiles"), f"invalid scope {iv.scope}")
            require(0 <= iv.start0 < iv.end0, f"invalid excluded interval {iv}")
            groups[(iv.chrom, iv.scope)].append((iv.start0, iv.end0, iv.cause))
        self._g = {}
        for key, rows in groups.items():
            rows.sort()
            starts = [r[0] for r in rows]
            prefix, m = [], -1
            for r in rows:
                m = max(m, r[1])
                prefix.append(m)
            self._g[key] = (starts, rows, prefix)

    def causes(self, chrom, start0, end0, scope="all"):
        found = set()
        g = self._g.get((chrom, scope))
        if not g:
            return found
        starts, rows, prefix = g
        i = bisect.bisect_left(starts, end0) - 1
        while i >= 0 and prefix[i] > start0:
            if rows[i][1] > start0:
                found.add(rows[i][2])
            i -= 1
        return found

    def overlaps_any(self, chrom, start0, end0):
        return bool(self.causes(chrom, start0, end0, "all") or self.causes(chrom, start0, end0, "train_tiles"))


@dataclass
class LeakageModel:
    species: str
    intervals: list
    index: IntervalIndex
    heldout_loci: list          # (gene_id, chrom, start0, end0) of strict held-out genes, unpadded
    coordinates: dict           # (chrom, strand, kind, coord0) -> (excluded_gene, neighbour_gene)
    genome: dict
    manifest: dict
    qc: dict

    def excluded_4c(self, chrom, strand, donor0, acceptor0):
        return ((chrom, strand, "donor", donor0) in self.coordinates
                or (chrom, strand, "acceptor", acceptor0) in self.coordinates)


def corpus_manifest(con, species):
    rows = con.execute("SELECT fasta, fasta_sha256, gff, gff_sha256, split_file_sha256, window_policy, build_version "
                       "FROM build_manifest WHERE species_id = ?", [species]).fetchall()
    require(len(rows) == 1, f"corpus build_manifest has {len(rows)} rows for {species}, expected 1")
    keys = ("fasta", "fasta_sha256", "gff", "gff_sha256", "split_file_sha256", "window_policy", "build_version")
    return dict(zip(keys, rows[0]))


def _parse_tile_qc(value):
    out = {k: 0 for k in TILE_QC_KEYS}
    for part in (value or "").split(";"):
        name, sep, n = part.partition("=")
        if sep and name in out:
            out[name] = int(n)
    return out


def gene_introns(gene):
    """(start0, end0, donor0, acceptor0) of every distinct intron of a gc.Gene (exons = merged UTR/CDS features)."""
    out = set()
    for feats in gene.transcripts.values():
        spans = sorted((f.start1, f.end1) for f in feats)
        exons = []
        for s, e in spans:
            if exons and s <= exons[-1][1] + 1:
                exons[-1][1] = max(exons[-1][1], e)
            else:
                exons.append([s, e])
        for (_, e_prev), (s_next, _) in zip(exons, exons[1:]):
            start0, end0 = e_prev, s_next - 1        # 1-based intron [e_prev+1, s_next-1] -> 0-based half-open
            if end0 <= start0:
                continue
            if gene.strand == "-":
                out.add((start0, end0, end0 - 1, start0))
            else:
                out.add((start0, end0, start0, end0 - 1))
    return out


def coordinate_exclusions(genes, seed_ids):
    """Rule 4(c): boundary bases shared by a seed-excluded gene and a non-seed neighbour (same chrom and strand)."""
    seed_coords, other_coords = {}, {}
    for g in genes:
        target = seed_coords if g.gene_id in seed_ids else other_coords
        for _, _, d, a in gene_introns(g):
            target.setdefault((g.chrom, g.strand, "donor", d), g.gene_id)
            target.setdefault((g.chrom, g.strand, "acceptor", a), g.gene_id)
    return {k: (seed_coords[k], other_coords[k]) for k in sorted(seed_coords.keys() & other_coords.keys())}


def build_leakage_model(*, corpus_db, species, splits_path, flags_path, build_flags=None, gff_path=None,
                        fasta_path=None, seed, verify_tiles=True):
    """Rule 4(a) excluded intervals and rule 4(c) coordinates for one species, from the frozen corpus DB.

    ``seed`` (the corpus build seed) is required: build_manifest does not record it; a wrong seed fails closed in the
    per-tile masking recomputation."""
    import duckdb

    b5 = load_build_b5()
    gc = b5.gc
    require(os.path.isfile(corpus_db), f"corpus DB missing: {corpus_db}")
    require(not os.path.exists(str(corpus_db) + ".wal"), "corpus has a WAL; supply a closed/checkpointed snapshot")
    con = duckdb.connect(str(corpus_db), read_only=True)
    try:
        manifest = corpus_manifest(con, species)
        require(manifest["window_policy"] == gc.WINDOW_POLICY_V3,
                f"corpus window_policy {manifest['window_policy']!r} is not the tile policy {gc.WINDOW_POLICY_V3!r}")
        require(sha256(splits_path) == manifest["split_file_sha256"],
                "--splits sha256 differs from the split file the corpus was built from")
        gff = gff_path or manifest["gff"]
        require(os.path.isfile(gff) and sha256(gff) == manifest["gff_sha256"], f"GFF {gff} missing or sha256 != corpus build_manifest")
        fasta = fasta_path or manifest["fasta"]
        require(os.path.isfile(fasta) and sha256(fasta) == manifest["fasta_sha256"],
                f"reference {fasta} missing or sha256 != corpus build_manifest")
        genome = b5.load_fasta(fasta)
        split_rows, _ = b5.read_split_table(splits_path)
        rule_flags = b5.read_qc_flags(flags_path) if flags_path else {}
        build_q = b5.read_qc_flags(list(build_flags) if build_flags else flags_path) if (build_flags or flags_path) else None

        # --- the build's gene preparation, verbatim in effect (build_b5.py _build_species_tiles) ---
        by_chrom, gene_meta, all_genes = {}, {}, []
        with open(gff) as fh:
            for gene in gc.parse_gff3(fh, species_code=gc.species_code(species)):
                key = (species, gene.gene_id)
                require(key in split_rows, f"{species}:{gene.gene_id} has no split entry")
                srow = split_rows[key]
                strict = str(srow.get("strict_holdout", "")).lower() in ("1", "true", "yes")
                no_cds = not any(f.type == "CDS" for tx in gene.transcripts.values() for f in tx)
                all_tx = list(gene.transcripts.keys())
                rflags = b5.flags_for_gene(rule_flags, species, gene)
                a22_hard = bool(rflags) and b5.loss_mask_decision(rflags, all_tx)[0] == 0
                all_genes.append((gene, srow["split"], strict, a22_hard))
                weight = 1.0
                gflags = b5.flags_for_gene(build_q, species, gene)
                if gflags:
                    weight, keep_tx, _ = b5.loss_mask_decision(gflags, gene.transcripts.keys())
                    if weight > 0 and len(keep_tx) < len(gene.transcripts):
                        gene = gc.Gene(gene.gene_id, gene.chrom, gene.strand, gene.start0, gene.end0,
                                       {t: gene.transcripts[t] for t in keep_tx}, gene.gene_id_original, gene.name_original)
                if gene.chrom not in genome:
                    continue
                try:
                    gc.check_caps(gc.gene_to_gsf(gene, gene.start0))
                except gc.CapError:
                    if not no_cds:
                        continue
                by_chrom.setdefault(gene.chrom, []).append(gene)
                gene_meta[gene.gene_id] = {"split": srow["split"], "weight": weight, "labelable": not no_cds}
        a22_ids = {gene.gene_id for gene, _, _, hard_ in all_genes if hard_}
        n_genes = sum(len(v) for v in by_chrom.values())
        n_masked = sum(1 for v in by_chrom.values() for g in v
                       if gene_meta[g.gene_id]["weight"] == 0 or gc.SPLIT_RANK.get(gene_meta[g.gene_id]["split"] or "train", 0) > 0)
        decoy_rate = min(gc.DECOY_MAX, (n_masked / n_genes) * gc.DECOY_RATIO) if n_genes else 0.0

        blocks = defaultdict(list)
        intervals = []
        for chrom, a, b, sp in con.execute("SELECT chromosome, start0, end0, split FROM tile_blocks WHERE species_id = ?",
                                           [species]).fetchall():
            blocks[chrom].append((a, b, sp))
            if sp == "test":
                intervals.append(ExcludedInterval(chrom, a, b, "test_block", "all"))
        require(blocks, f"corpus has no tile_blocks for {species}")

        # --- per kept tile: recompute and verify the build's masking (seeds and order exactly as build_b5) ---
        flank_seen = defaultdict(int)
        qc = defaultdict(int)
        cur = con.cursor()
        cur.execute("SELECT rn, chromosome, start, fin, split, sequence, qc_flags FROM geneList "
                    "WHERE species_id = ? AND NOT is_rc ORDER BY rn", [species])
        while True:
            rows = cur.fetchmany(256)
            if not rows:
                break
            for rn, chrom, ws, we, split, seq, qc_flags in rows:
                qc["tiles"] += 1
                inside, partial = gc.genes_in_window(by_chrom.get(chrom, []), ws, we)
                expected_split = gc.tile_split(blocks[chrom], ws, we)
                if split is None:
                    # build_b5 writes split NULL only for a gene-free tile under allow_missing_split.  Fail closed
                    # unless an explicit split can be derived: the tile must be gene-free AND overlap at least one
                    # tile_block (tile_split's 'train' default for a block-less tile is not explicit).  The tile is
                    # then treated as that split (its edge_partial intervals are emitted), never skipped.
                    explicit = any(a < we and b > ws for a, b, _ in blocks[chrom])
                    require(explicit and not inside,
                            f"tile {rn} ({chrom}:{ws}-{we}) has split NULL and no explicit expected split "
                            f"(overlaps a tile_block: {explicit}; genes inside: {len(inside)}); refusing")
                    qc["tiles_split_null_resolved"] += 1
                require(split in (expected_split, None), f"tile {rn}: stored split {split} != recomputed {expected_split}")
                leak = {g.gene_id for g in inside if gene_meta[g.gene_id]["split"]
                        and gc.SPLIT_RANK.get(gene_meta[g.gene_id]["split"], 0) > gc.SPLIT_RANK[expected_split]}
                hard = {g.gene_id for g in inside if g.gene_id not in leak and gene_meta[g.gene_id]["weight"] == 0}
                decoy = set()
                if expected_split == "train" and decoy_rate > 0:
                    drng = random.Random(f"{seed}:{species}:{chrom}:{ws}-{we}:decoy")
                    decoy = {g.gene_id for g in inside if gene_meta[g.gene_id]["labelable"] and g.gene_id not in leak
                             and g.gene_id not in hard and drng.random() < decoy_rate}
                seed_ids = leak | hard | decoy
                masked_ids = set()
                for comp in gc.overlap_components(inside):
                    if any(g.gene_id in seed_ids for g in comp):
                        masked_ids |= {g.gene_id for g in comp}
                masked = [g for g in inside if g.gene_id in masked_ids]
                flanks = []
                if masked:
                    frng = random.Random(f"{seed}:{species}:{chrom}:{ws}-{we}:flank")
                    flanks = [frng.randint(*gc.FLANK_RANGE) for _ in masked]
                if verify_tiles:
                    got = _parse_tile_qc(qc_flags)
                    want = {"leak_masked": len(leak), "hard_masked": len(hard), "decoy_masked": len(decoy),
                            "component_masked": len(masked_ids) - len(seed_ids), "edge_partial": partial}
                    require(got == want, f"tile {rn} ({chrom}:{ws}-{we}): recomputed masking {want} != stored qc_flags {got}; "
                                         "wrong --build-flags, seed or corpus")
                    ref = genome[chrom][ws:we]
                    if len(ref) < we - ws:
                        ref = ref + "N" * (we - ws - len(ref))
                    if masked:
                        ref = gc.leak_mask(ref, ws, masked, flanks=flanks)
                    require(ref == seq, f"tile {rn} ({chrom}:{ws}-{we}): recomputed N-mask differs from the stored sequence")
                    qc["tiles_verified"] += 1
                if expected_split not in ("train", "valid"):
                    continue
                # Rule 4(a) seeds only: an A30-only hard mask (and its component) is corpus masking, not a
                # rule-4(a) cause, so it yields no interval (counted).
                rule_seed = leak | (hard & a22_ids) | decoy
                rule_genes = set()
                for comp in gc.overlap_components(inside):
                    if any(g.gene_id in rule_seed for g in comp):
                        rule_genes |= {g.gene_id for g in comp}
                scope = "train_tiles" if expected_split == "train" else "all"
                for g, f in zip(masked, flanks):
                    flank_seen[g.gene_id] = max(flank_seen[g.gene_id], f)
                    if g.gene_id not in rule_genes:
                        qc["tile_masks_not_rule4a_a30_only"] += 1
                        continue
                    kind = ("tile_leak" if g.gene_id in leak else "tile_hard" if g.gene_id in hard
                            else "tile_decoy" if g.gene_id in decoy else "tile_component")
                    a, b = max(ws, g.start0 - f), min(we, g.end0 + f)
                    intervals.append(ExcludedInterval(chrom, a, b, kind, scope, g.gene_id, rn))
                for g in by_chrom.get(chrom, []) if EDGE_PARTIAL_MOLECULE_WIDE else ():
                    if g.end0 > ws and g.start0 < we and not (g.start0 >= ws and g.end0 <= we):
                        intervals.append(ExcludedInterval(chrom, max(ws, g.start0), min(we, g.end0), "edge_partial", scope,
                                                          g.gene_id, rn))
        cur.close()
    finally:
        con.close()

    # --- genome-level seed genes and their overlap components ---
    seeds = {}
    per_chrom = defaultdict(list)
    for gene, split, strict, a22_hard in all_genes:
        if gene.chrom not in genome:
            continue
        per_chrom[gene.chrom].append(gene)
        causes = []
        if strict:
            causes.append(("heldout", "all"))
        if split == "test":
            causes.append(("test_orthogroup", "all"))
        if split == "valid":
            causes.append(("valid_orthogroup", "train_tiles"))
        if a22_hard:
            causes.append(("a22_hard", "all"))
        if causes:
            seeds[gene.gene_id] = causes
    heldout = [(g.gene_id, g.chrom, g.start0, g.end0) for g, _, strict, _ in all_genes if strict and g.chrom in genome]
    for chrom, genes in per_chrom.items():
        L = len(genome[chrom])
        for comp in gc.overlap_components(genes):
            comp_causes = {c for g in comp for c in seeds.get(g.gene_id, ())}
            if not comp_causes:
                continue
            flank = max((flank_seen[g.gene_id] for g in comp if g.gene_id in flank_seen), default=gc.FLANK_RANGE[1])
            a = max(0, min(g.start0 for g in comp) - flank)
            b = min(L, max(g.end0 for g in comp) + flank)
            label = ",".join(sorted(g.gene_id for g in comp if g.gene_id in seeds))
            for cause, scope in sorted(comp_causes):
                intervals.append(ExcludedInterval(chrom, a, b, cause, scope, label))
    coords = coordinate_exclusions([g for g, *_ in all_genes if g.chrom in genome], set(seeds))
    for iv in intervals:
        qc["intervals_" + iv.cause] += 1
    qc.update(genes=len(all_genes), seed_genes=len(seeds), heldout_loci=len(heldout), decoy_rate_ppm=round(decoy_rate * 1e6),
              coordinates_4c=len(coords))
    return LeakageModel(species, intervals, IntervalIndex(intervals), heldout, coords, genome, manifest, dict(qc))


@dataclass(frozen=True)
class Placement:
    member_id: str
    origin: str      # 'frozen' | 'audit'
    kind: str        # primary | supplementary | secondary
    chrom: str
    start0: int
    end0: int
    mapq: int | None


@dataclass
class MoleculeDecision:
    molecule_id: str
    n_members: int
    n_secondary: int            # audit secondary placements summed over the molecule's members
    n_placements: int
    cause_a: tuple
    cause_a_direct: bool
    cause_a_train_tiles: tuple
    unauditable_secondary: bool
    unauditable_low_mapq: bool
    audited: bool
    low_mapq_secondary_only: bool = False   # the MAPQ flag rests on secondary placements alone (ruling (a))

    @property
    def excluded(self):
        return bool(self.cause_a) or self.unauditable_secondary or self.unauditable_low_mapq

    @property
    def train_tiles_excluded(self):
        return bool(self.cause_a_train_tiles)

    @property
    def primary_cause(self):
        """Exclusive precedence for the QC table: (a) before (b)-secondary before (b)-MAPQ."""
        if self.cause_a:
            return "a"
        if self.unauditable_secondary:
            return "b_secondary"
        if self.unauditable_low_mapq:
            return "b_low_mapq"
        return None


def decide_molecule(molecule_id, n_members, placements, n_secondary, index, *, audited=True, n_cap=AUDIT_N,
                    mapq_min=AUDIT_MAPQ_MIN):
    """Rule 4(b) for ONE molecule.  ``placements``: iterable of (origin, kind, chrom, start0, end0, mapq) or
    ``Placement`` over every member; ``n_secondary``: the molecule's audit secondary placements (sum over members).

    Frozen wording: a molecule with >= N secondary placements is unauditable; -N caps secondaries per query, so
    the molecule total is compared with N (a multi-member molecule reaching N in total is unauditable too)."""
    causes, direct, tt, n_pl = set(), False, set(), 0
    low_other = low_secondary = False
    for p in placements:
        if isinstance(p, Placement):
            origin, kind, chrom, start0, end0, mapq = p.origin, p.kind, p.chrom, p.start0, p.end0, p.mapq
        else:
            origin, kind, chrom, start0, end0, mapq = p
        n_pl += 1
        c = index.causes(chrom, start0, end0, "all")
        t = index.causes(chrom, start0, end0, "train_tiles")
        causes |= c
        tt |= t
        if c and origin == "frozen" and kind in ("primary", "supplementary"):
            direct = True
        if audited and mapq is not None and mapq < mapq_min and not (c or t):
            if kind == "secondary":
                low_secondary = True
            else:
                low_other = True
    low_mapq = low_other or (low_secondary and LOW_MAPQ_APPLIES_TO_AUDIT_SECONDARIES)
    return MoleculeDecision(molecule_id, n_members, n_secondary, n_pl, tuple(sorted(causes)), direct, tuple(sorted(tt)),
                            audited and n_secondary >= n_cap, low_mapq, audited,
                            low_mapq and not low_other)


class Exhaustiveness:
    """Streaming counter: fraction of molecules with fewer than N secondary placements (frozen wording)."""

    def __init__(self, n_cap=AUDIT_N, min_exhaustiveness=MIN_SOURCE_EXHAUSTIVENESS, label="source"):
        self.n_cap, self.min, self.total, self.below, self.label = n_cap, min_exhaustiveness, 0, 0, label

    def add(self, decision):
        self.total += 1
        self.below += decision.n_secondary < self.n_cap

    def report(self):
        """Per-source numerator / denominator for the species-level rule (no threshold applied here).

        A46 rule 4(b): "Exhaustiveness = the fraction of molecules with fewer than N = 50 placements, reported per
        species and required to be >= 99.9 %".  A species x source build cannot see the other sources, so it records
        ``exhaustiveness_below_cap`` (numerator) and ``exhaustiveness_molecules`` (denominator) and its own fraction
        for reporting; ``species_exhaustiveness`` sums them over the species' sources where they meet (the merge)."""
        qc = {"molecules": self.total, "audit_n": self.n_cap, "exhaustiveness_molecules": self.total,
              "exhaustiveness_below_cap": self.below}
        if self.total:
            qc["exhaustiveness"] = self.below / self.total
        return qc

    def check(self):
        """report() plus the per-source floor (author 2026-09-23): ExhaustivenessError below ``self.min``."""
        qc = self.report()
        source_exhaustiveness(self.label, self.below, self.total, self.n_cap, self.min)
        return qc


def source_exhaustiveness(label, below, total, audit_n, min_exhaustiveness=MIN_SOURCE_EXHAUSTIVENESS):
    """Per (species x source) floor (author decision 2026-09-23, in addition to the species-level rule 4(b) gate):
    ``below / total`` must be >= ``min_exhaustiveness`` or ``ExhaustivenessError`` is raised.  Used by the build
    (``Exhaustiveness.check``) and re-applied by the merge to the counters each input recorded, so a DB built by an
    older builder cannot bypass the floor.  An empty source (``total == 0``) has no fraction and passes."""
    below, total = int(below), int(total)
    require(0 <= below <= total, f"{label}: exhaustiveness counters below_cap={below} > molecules={total}")
    if not total:
        return None
    fraction = below / total
    if fraction < min_exhaustiveness:
        raise ExhaustivenessError(
            f"{label}: per-source audit exhaustiveness {fraction:.6f} < {min_exhaustiveness} ({total - below} of "
            f"{total} molecules have >= N={int(audit_n)} secondary placements): raise N and rerun the audit "
            "alignment before any track is built (A46 rule 4(b); per-source floor, author 2026-09-23)")
    return fraction


def species_exhaustiveness(species, parts, min_exhaustiveness=MIN_EXHAUSTIVENESS):
    """Species-level rule 4(b) exhaustiveness over every audited source of one species.

    ``parts``: iterable of (source, below_cap, molecules, audit_n) as the per-source builds recorded them in
    ``c0_qc`` (``exhaustiveness_below_cap`` / ``exhaustiveness_molecules`` / ``audit_n``).  All sources must share
    one N (the fraction is defined for one N); the summed fraction must be >= ``min_exhaustiveness`` or
    ``ExhaustivenessError`` is raised ("N is raised and the audit rerun before any track is built")."""
    parts = sorted(parts)
    require(parts, f"{species}: no audited source carries exhaustiveness counters")
    ns = sorted({int(p[3]) for p in parts})
    require(len(ns) == 1, f"{species}: the audited sources record different audit N {ns}; the species fraction is "
                          "defined for one N (A46 rule 4(b))")
    below = sum(int(p[1]) for p in parts)
    total = sum(int(p[2]) for p in parts)
    out = {"species": species, "audit_n": ns[0], "molecules": total, "below_cap": below,
           "by_source": {p[0]: {"below_cap": int(p[1]), "molecules": int(p[2])} for p in parts},
           "min_exhaustiveness": min_exhaustiveness}
    if not total:
        out["exhaustiveness"] = None
        return out
    out["exhaustiveness"] = below / total
    if out["exhaustiveness"] < min_exhaustiveness:
        raise ExhaustivenessError(
            f"{species}: species-level audit exhaustiveness {out['exhaustiveness']:.6f} < {min_exhaustiveness} "
            f"({total - below} of {total} molecules over sources {sorted(out['by_source'])} have >= N={ns[0]} secondary "
            "placements): raise N and rerun the audit alignment before any track is built (A46 rule 4(b))")
    return out


def decide_molecules(molecules, placements, secondary_counts, index, *, audited=True, n_cap=AUDIT_N,
                     mapq_min=AUDIT_MAPQ_MIN, min_exhaustiveness=MIN_EXHAUSTIVENESS):
    """In-memory wrapper over ``decide_molecule``.  ``molecules``: molecule_id -> member ids; ``placements``:
    member_id -> [Placement]; ``secondary_counts``: member_id -> secondary records in the audit alignment
    (summed per molecule)."""
    decisions = {}
    ex = Exhaustiveness(n_cap, min_exhaustiveness)
    for mid, members in molecules.items():
        n_sec = sum(secondary_counts.get(m, 0) for m in members)
        d = decide_molecule(mid, len(members), (p for m in members for p in placements.get(m, ())), n_sec, index,
                            audited=audited, n_cap=n_cap, mapq_min=mapq_min)
        decisions[mid] = d
        ex.add(d)
    qc = {"molecules": len(decisions)}
    if audited and decisions:
        qc = ex.check()
    return decisions, qc


class HeldoutLoci:
    """Rule 4(d): the unpadded strict held-out loci as an independent index (not the excluded-interval index)."""

    def __init__(self, heldout_loci):
        self.index = IntervalIndex([ExcludedInterval(chrom, a, b, "heldout_locus", "all", gid)
                                    for gid, chrom, a, b in heldout_loci])

    def hits(self, chrom, start0, end0):
        return bool(self.index.causes(chrom, start0, end0))


def heldout_qc(decisions, molecules, placements, heldout_loci):
    """Rule 4(d): an independent recount against the unpadded strict held-out loci (not the excluded-interval index)."""
    loci = HeldoutLoci(heldout_loci)

    def hits(p):
        return loci.hits(p.chrom, p.start0, p.end0)

    overlapping = {mid for mid, members in molecules.items()
                   if any(hits(p) for m in members for p in placements.get(m, ()))}
    excluded = {mid for mid in overlapping if decisions[mid].excluded}
    return {"heldout_overlapping_molecules": len(overlapping), "heldout_overlapping_excluded": len(excluded),
            "heldout_retained": len(overlapping - excluded)}


def write_intervals_tsv(intervals, path):
    """Optional human-readable dump (b6_leakage_intervals.py)."""
    with open(path, "x", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["chrom", "start0", "end0", "cause", "scope", "gene_id", "tile_rn"])
        for iv in sorted(intervals, key=lambda i: (i.chrom, i.start0, i.end0, i.cause, i.scope, i.gene_id or "",
                                                  -1 if i.tile_rn is None else i.tile_rn)):
            w.writerow([iv.chrom, iv.start0, iv.end0, iv.cause, iv.scope, iv.gene_id or "", "" if iv.tile_rn is None else iv.tile_rn])

