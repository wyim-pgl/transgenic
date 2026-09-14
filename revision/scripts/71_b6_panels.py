#!/usr/bin/env python3
"""B6 eligible locus pool for A. thaliana (protocol amendment A46; work plan B6 §2 item 10 and MVP pilot §1.2).

Builds the eligible A. thaliana held-out locus pool from which the disjoint panels that A46 freezes before WP5 are drawn
(B6_EVIDENCE_MODE_WORKPLAN_v1.md §2 item 10 [C3-4]: "the pilot panel (WP5) and the
final scoring panel are drawn once, disjoint, from the held-out loci with seed
20260914 and their ids frozen in A46 before WP5"; B6_MVP_PILOT_v1.md §1.2 for the
MVP core/extended panel) from the frozen B5 corpus `b5_full_a40_v1.db`.

Corpus layout that the eligibility rules are applied to
-------------------------------------------------------
`geneList` in the a40 corpus holds 30,720-nt TILES (`window_policy = tile6144-v3`),
not genes: `gene_id` there is a window id, `orthogroup_id` is NULL and
`strict_holdout / contig_boundary / n_transcripts / train_weight` describe the tile.
Gene-level facts live in `gene_key_map` (coordinates, original id), `gene_split`
(orthogroup, split, strict_holdout) and `window_genes` (which labelled tiles carry
the gene). The per-gene rules of the brief are therefore applied as follows:

  strict_holdout = TRUE   gene_split.strict_holdout (species_id = Athaliana)
  one row per gene        gene_key_map has one row per gene_id; RC copies exist only
                          at tile level (window_genes.is_rc) and are not loci
  multi-exon              max exon count over the gene's transcripts >= --min-exons,
                          read from the source GFF3 named in build_manifest (its
                          sha256 is checked against build_manifest.gff_sha256); the
                          gene must have >= 1 transcript (n_transcripts >= 1)
  contig_boundary = FALSE the 30,720-nt evaluation window centred on the gene
                          (B6_MVP_PILOT_v1.md §1.2) fits inside the chromosome
                          (lengths from the manifest FASTA .fai), AND no labelled
                          tile carrying the gene has geneList.contig_boundary = TRUE
  no overlap              half-open [start0, end0) interval test against every
                          other A. thaliana gene in gene_key_map, either strand,
                          optionally widened by --overlap-flank nt
  train_weight > 0        no labelled tile carrying the gene has train_weight = 0
                          (genes with no labelled tile, i.e. N-masked everywhere,
                          pass this test)

  nuclear only            chromosome not in {ChrC, ChrM} (organellar genes are dropped
                          and counted)
  no A22 hard flag        the gene carries no GeenuFF hard flag (qc_flags.HARD_FLAG_PATTERNS,
                          qc_flags.py sha256 29a52907...) in the A22 flag file the a40 build
                          consumed (--a22-flags, md5 frozen in b5_full_a40_v1.freeze.json
                          `qc_flag_files`); flags are matched by builder key, GFF ID or Name
                          like build_b5.flags_for_gene, on any transcript or the gene itself
  no A30 hard flag        the gene is neither in the Swiss-Prot caution list
                          evidence/swissprot_hard_masked_genes_20260902.tsv (--a30-list,
                          sha256 e191f619..., 567 A. thaliana genes; byte-identical copies on
                          pronghorn, on the GPU host and in
                          revision/supplemental_tables/TableS_swissprot_hard_masked_genes.tsv)
                          nor carries a `swissprot_caution_*` flag in the A30 flag file the
                          build consumed (--a30-flags, md5 frozen). Tile train_weight cannot
                          express these: A32 masks hard-flagged genes at gene level and keeps
                          tile weights at 1 (Codex finding), so the gene-level lists are used.

Output: EVERY eligible locus, one row each, ordered by `shuffle_rank` = the 1-based
position of the gene_id in one `random.Random(--seed)` (default 20260914) shuffle of
the eligible gene_ids sorted lexicographically. This script assigns NO panels: the
MVP / pilot / final split is made downstream by `revision/scripts/72_mvp_select_loci.py`,
which walks `shuffle_rank` order with the A-EST molecule counts that
`72a_est_molecules_per_locus.sh` produces on pronghorn. No `hash()` is used anywhere;
the output is byte-identical across runs.

Coordinates in the TSV: `start`/`fin` are 1-based inclusive (GFF convention, i.e.
gene_key_map.start0 + 1 and end0), so `length = fin - start + 1` is the gene length in
nt. `n_transcripts`, `exon_count` (max exons over the gene's transcripts) and
`intron_count` (= exon_count - 1) come from the GFF3. `orthogroup_id` is the empty
string for singleton genes.

Usage (the corpus lives on the lab GPU host; the script ships itself over ssh,
opens the DuckDB read-only there and brings back only the TSV):

  python revision/scripts/71_b6_panels.py --remote gpu \
      --db /home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1/merged/b5_full_a40_v1.db \
      --out revision/protocols/b6_eligible_loci_v1.tsv

Run directly on the host that holds the DB with the same arguments minus --remote.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_DB = "/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1/merged/b5_full_a40_v1.db"
DEFAULT_DB_MD5 = "dda7e78c880c993435654b57dcd1c498"
DEFAULT_REMOTE_PYTHON = "/home/wyim/micromamba/envs/transgenic/bin/python"
SPECIES = "Athaliana"
WINDOW = 30720  # nt, evaluation window centred on the gene (B6_MVP_PILOT_v1.md §1.2)
ORGANELLES = ("ChrC", "ChrM")
DEFAULT_A22_FLAGS = "/home/pgl/scratch1/wyim/transgenic_data/qc/Athaliana.geenuff_flags.tsv"
DEFAULT_A22_MD5 = "5ab607c50e37af0591494cfb1f5b0b05"      # b5_full_a40_v1.freeze.json qc_flag_files
DEFAULT_A30_FLAGS = "/home/pgl/scratch1/wyim/transgenic_data/swissprot_v3/Athaliana.swissprot_flags.tsv"
DEFAULT_A30_MD5 = "1bf39ce9ae07324ad9a9924cbc6b4f2d"      # b5_full_a40_v1.freeze.json qc_flag_files
DEFAULT_A30_LIST = "/home/pgl/scratch1/wyim/transgenic_data/swissprot_v3/swissprot_hard_masked_genes.tsv"
DEFAULT_A30_LIST_SHA256 = "e191f61982e4b435d680bc704bb4d8b49d0d7ac097dd9b7e8b4bfbc9c9206903"  # == evidence/swissprot_hard_masked_genes_20260902.tsv
# Copied verbatim from src/transgenic/datasets/qc_flags.py (sha256 29a52907b72e66ab2d0c1d7cd64883542789d3f475a399590d17efe2ef4797ff,
# the version frozen with the a40 corpus); tests/test_b6_panels.py asserts the copy has not drifted.
HARD_FLAG_PATTERNS = ("missing_start", "missing_stop", "wrong_starting_phase", "mismatched_ending_phase", "mismatched_phase",
                      "overlapping_exon", "too_short_intron", "empty_transcript", "empty_super_locus", "wrong_phase",
                      "super_loci_overlap", "missmatching_strand", "truncated_intron",
                      "swissprot_caution")
COLUMNS = ("shuffle_rank", "species_id", "gene_id", "chromosome", "start", "fin", "strand", "length",
           "orthogroup_id", "n_transcripts", "exon_count", "intron_count")


class PanelError(RuntimeError):
    """Any inconsistency between the corpus tables and the source GFF aborts the draw."""


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- GFF3
def parse_gff_genes(path: Path) -> dict:
    """Return {gene ID: {"chrom", "start0", "end0", "strand", "n_transcripts", "exon_count"}}.

    A transcript is any feature whose Parent is a gene; exons are counted per transcript
    (falling back to CDS segments when a transcript has no exon lines); exon_count is the
    maximum over the gene's transcripts."""
    genes: dict = {}
    tx_parent: dict = {}
    exons_per_tx: dict = defaultdict(int)
    cds_per_tx: dict = defaultdict(int)
    pending: list = []  # (type, attrs) for non-gene features, resolved after the pass
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                raise PanelError(f"malformed GFF line: {line[:80]!r}")
            chrom, _, ftype, start, end, _, strand, _, attrs = cols[:9]
            kv = dict(a.split("=", 1) for a in attrs.split(";") if "=" in a)
            if ftype == "gene":
                gid = kv.get("ID")
                if not gid:
                    raise PanelError(f"gene without ID: {line[:80]!r}")
                if gid in genes:
                    raise PanelError(f"duplicate gene ID in GFF: {gid}")
                genes[gid] = {"chrom": chrom, "start0": int(start) - 1, "end0": int(end), "strand": strand,
                              "n_transcripts": 0, "exon_count": 0}
            else:
                pending.append((ftype, kv.get("ID"), kv.get("Parent", "")))
    # transcripts first (any feature whose Parent is a gene), then exon/CDS lines
    for ftype, fid, parent in pending:
        for p in parent.split(","):
            if p in genes and fid:
                tx_parent[fid] = p
    for ftype, fid, parent in pending:
        for p in parent.split(","):
            if p in tx_parent:
                if ftype == "exon":
                    exons_per_tx[p] += 1
                elif ftype == "CDS":
                    cds_per_tx[p] += 1
    for tx, gid in tx_parent.items():
        g = genes[gid]
        g["n_transcripts"] += 1
        n = exons_per_tx[tx] if exons_per_tx[tx] else cds_per_tx[tx]
        g["exon_count"] = max(g["exon_count"], n)
    return genes


def read_fai(path: Path) -> dict:
    lengths = {}
    with open(path) as fh:
        for line in fh:
            name, length = line.split("\t")[:2]
            lengths[name] = int(length)
    return lengths


def is_hard(flag: str) -> bool:
    f = flag.lower()
    return any(p in f for p in HARD_FLAG_PATTERNS)


def read_hard_flags(path: Path, species: str = SPECIES) -> dict:
    """{gene id as written in the A22-schema flag file: {transcript_id or "*": set(hard flags)}} for one species.
    Same file contract as build_b5.read_qc_flags (species_id, gene_id, transcript_id, flag, start, end); soft flags dropped."""
    out: dict = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for col in ("species_id", "gene_id", "transcript_id", "flag"):
            if col not in idx:
                raise PanelError(f"{path}: missing column {col}")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4 or c[idx["species_id"]] != species or not is_hard(c[idx["flag"]]):
                continue
            out.setdefault(c[idx["gene_id"]], {}).setdefault(c[idx["transcript_id"]] or "*", set()).add(c[idx["flag"]])
    return out


def read_gene_list(path: Path, species: str = SPECIES) -> set:
    """gene_id column of a TSV with species_id and gene_id columns (the A30 caution list)."""
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        if "gene_id" not in idx or "species_id" not in idx:
            raise PanelError(f"{path}: needs species_id and gene_id columns, got {header}")
        return {c[idx["gene_id"]] for c in (l.rstrip("\n").split("\t") for l in fh) if len(c) > idx["gene_id"] and c[idx["species_id"]] == species}


def verify_hash(path: Path, expected: str, algorithm: str, label: str) -> str:
    got = digest(path, algorithm)
    if expected and got != expected:
        raise PanelError(f"{label} {path}: {algorithm} {got} != expected {expected}")
    return got


# ----------------------------------------------------------------------------- corpus
def load_corpus(db: str, expect_md5: str | None):
    import duckdb  # noqa: WPS433 (only needed on the host that holds the DB)

    if expect_md5:
        got = digest(Path(db), "md5")
        if got != expect_md5:
            raise PanelError(f"DB md5 {got} != expected {expect_md5}: refusing to draw panels from an unverified corpus")
    con = duckdb.connect(db, read_only=True)
    manifest = con.execute("SELECT gff, gff_sha256, fasta, fasta_sha256, window_policy, build_version FROM build_manifest "
                           "WHERE species_id = ?", [SPECIES]).fetchall()
    if len(manifest) != 1:
        raise PanelError(f"expected exactly one build_manifest row for {SPECIES}, found {len(manifest)}")
    gff, gff_sha, fasta, fasta_sha, policy, build_version = manifest[0]
    genes = con.execute("SELECT k.gene_id, k.gene_id_original, k.chromosome, k.start0, k.end0, s.orthogroup_id, s.split, "
                        "s.strict_holdout, k.name_original FROM gene_key_map k JOIN gene_split s USING (species_id, gene_id) "
                        "WHERE k.species_id = ? ORDER BY k.gene_id", [SPECIES]).fetchall()
    n_key = con.execute("SELECT count(*), count(DISTINCT gene_id) FROM gene_key_map WHERE species_id = ?", [SPECIES]).fetchone()
    if n_key[0] != n_key[1] or n_key[0] != len(genes):
        raise PanelError(f"gene_key_map/gene_split mismatch for {SPECIES}: {n_key} vs joined {len(genes)}")
    tiles = con.execute("SELECT w.gene_id, g.contig_boundary, g.train_weight FROM window_genes w JOIN geneList g "
                        "ON g.species_id = w.species_id AND g.gene_id = w.window_id AND g.is_rc = w.is_rc "
                        "WHERE w.species_id = ?", [SPECIES]).fetchall()
    n_windows = con.execute("SELECT count(*) FROM window_genes WHERE species_id = ?", [SPECIES]).fetchone()[0]
    if n_windows != len(tiles):
        raise PanelError(f"window_genes rows without a geneList tile: {n_windows - len(tiles)}")
    rejected = {r[0] for r in con.execute("SELECT gene_id FROM rejected_records WHERE species_id = ?", [SPECIES]).fetchall()}
    duck_version = duckdb.__version__
    con.close()
    return {"gff": gff, "gff_sha256": gff_sha, "fasta": fasta, "fasta_sha256": fasta_sha, "window_policy": policy,
            "build_version": build_version, "genes": genes, "tiles": tiles, "rejected": rejected,
            "duckdb_version": duck_version}


def overlapping_genes(genes, flank: int) -> set:
    """gene_ids that overlap another gene on either strand (half-open intervals widened by `flank`)."""
    by_chrom = defaultdict(list)
    for gene_id, _, chrom, start0, end0, *_ in genes:
        by_chrom[chrom].append((start0 - flank, end0 + flank, gene_id))
    hits = set()
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        active: list = []
        for start0, end0, gene_id in ivs:
            active = [a for a in active if a[1] > start0]
            for a in active:
                hits.add(a[2])
                hits.add(gene_id)
            active.append((start0, end0, gene_id))
    return hits


def draw_panels(args, report: list) -> list:
    corpus = load_corpus(args.db, None if args.skip_md5 else args.expect_md5)
    gff_path, fai_path = Path(corpus["gff"]), Path(corpus["fasta"] + ".fai")
    gff_sha_now = digest(gff_path)
    if gff_sha_now != corpus["gff_sha256"]:
        raise PanelError(f"source GFF sha256 {gff_sha_now} != build_manifest {corpus['gff_sha256']}")
    if not fai_path.exists():
        raise PanelError(f"FASTA index missing: {fai_path}")
    gff_genes = parse_gff_genes(gff_path)
    chrom_len = read_fai(fai_path)
    a22_md5 = verify_hash(Path(args.a22_flags), args.a22_md5, "md5", "A22 flag file")
    a30_md5 = verify_hash(Path(args.a30_flags), args.a30_md5, "md5", "A30 flag file")
    a30_sha = verify_hash(Path(args.a30_list), args.a30_list_sha256, "sha256", "A30 caution list")
    a22_hard = read_hard_flags(Path(args.a22_flags))
    a30_hard = read_hard_flags(Path(args.a30_flags))
    a30_list = read_gene_list(Path(args.a30_list))

    report.append(f"db={args.db}")
    report.append(f"db_md5={'skipped' if args.skip_md5 else args.expect_md5 + ' (verified)'}")
    report.append(f"duckdb={corpus['duckdb_version']} window_policy={corpus['window_policy']} build_version={corpus['build_version']}")
    report.append(f"gff={gff_path} sha256={gff_sha_now} (matches build_manifest)")
    report.append(f"a22_flags={args.a22_flags} md5={a22_md5} (matches freeze qc_flag_files); hard-flagged {SPECIES} gene ids in file: {len(a22_hard)}")
    report.append(f"a30_flags={args.a30_flags} md5={a30_md5} (matches freeze qc_flag_files); swissprot_caution gene ids in file: {len(a30_hard)}")
    report.append(f"a30_list={args.a30_list} sha256={a30_sha} (== evidence/swissprot_hard_masked_genes_20260902.tsv); {SPECIES} gene ids: {len(a30_list)}")
    report.append(f"seed={args.seed} min_exons={args.min_exons} overlap_flank={args.overlap_flank} window={WINDOW}")

    genes = corpus["genes"]
    tile_bad_contig = {g for g, cb, _ in corpus["tiles"] if cb}
    tile_zero_weight = {g for g, _, tw in corpus["tiles"] if not tw or tw <= 0}
    overlap = overlapping_genes(genes, args.overlap_flank)

    # funnel, in the order of the brief; every line reports what the filter dropped
    report.append(f"[filter] {SPECIES} genes in gene_key_map: {len(genes)}")
    rows = [g for g in genes if g[7]]
    total_holdout = len(rows)
    report.append(f"[filter] strict_holdout = TRUE: {total_holdout} (total strict_holdout loci)")
    seen, dedup = set(), []
    for g in rows:
        if g[0] not in seen:
            seen.add(g[0]); dedup.append(g)
    report.append(f"[filter] one row per gene / non-RC (RC exists only per tile): kept {len(dedup)}, dropped {len(rows) - len(dedup)}")
    rows = dedup
    before = len(rows)
    rows = [g for g in rows if g[2] not in ORGANELLES]
    report.append(f"[filter] nuclear chromosomes only (not in {'/'.join(ORGANELLES)}): kept {len(rows)}, dropped {before - len(rows)}")

    def keys(g):  # builder key, GFF ID, Name attribute — the three ids build_b5.flags_for_gene consults
        return {k for k in (g[0], g[1], g[8]) if k}

    def hard_of(g, table):
        merged: dict = {}
        for k in keys(g):
            for tx, flags in table.get(k, {}).items():
                merged.setdefault(tx, set()).update(flags)
        return merged

    before = len(rows)
    a22_hit = {g[0] for g in rows if hard_of(g, a22_hard)}
    a22_full = {g[0] for g in rows if g[0] in a22_hit and ("*" in hard_of(g, a22_hard)
                or len(hard_of(g, a22_hard)) >= gff_genes.get(g[1], {"n_transcripts": 0})["n_transcripts"])}
    rows = [g for g in rows if g[0] not in a22_hit]
    report.append(f"[filter] (a) no A22 GeenuFF hard flag on the gene or any transcript: kept {len(rows)}, dropped {before - len(rows)} "
                  f"(of which fully masked, i.e. train_weight 0 by loss_mask_decision: {len(a22_full)})")
    before = len(rows)
    in_list = {g[0] for g in rows if keys(g) & a30_list}
    in_flags = {g[0] for g in rows if hard_of(g, a30_hard)}
    rows = [g for g in rows if g[0] not in in_list and g[0] not in in_flags]
    report.append(f"[filter] (b) not on the A30 Swiss-Prot caution list and no swissprot_caution flag: kept {len(rows)}, dropped {before - len(rows)} "
                  f"(caution list: {len(in_list)}; build-consumed A30 flag file: {len(in_flags)}; flag file only: {len(in_flags - in_list)}; list only: {len(in_list - in_flags)})")

    missing = [g[0] for g in rows if g[1] not in gff_genes]
    if missing:
        raise PanelError(f"{len(missing)} held-out genes absent from the GFF, e.g. {missing[:5]}")
    for g in rows:
        ref = gff_genes[g[1]]
        if (ref["chrom"], ref["start0"], ref["end0"]) != (g[2], g[3], g[4]):
            raise PanelError(f"coordinate mismatch for {g[0]} ({g[1]}): DB {g[2]}:{g[3]}-{g[4]} vs GFF "
                             f"{ref['chrom']}:{ref['start0']}-{ref['end0']}")
    before = len(rows)
    rows = [g for g in rows if gff_genes[g[1]]["n_transcripts"] >= 1 and gff_genes[g[1]]["exon_count"] >= args.min_exons]
    report.append(f"[filter] multi-exon (n_transcripts >= 1 and exon_count >= {args.min_exons}, from GFF): kept {len(rows)}, dropped {before - len(rows)}")

    def window_fits(g):
        length = g[4] - g[3]
        if length > WINDOW:
            return False
        ws = g[3] - (WINDOW - length) // 2
        return ws >= 0 and ws + WINDOW <= chrom_len[g[2]]

    before = len(rows)
    n_long = sum(1 for g in rows if g[4] - g[3] > WINDOW)
    n_tile_flag = sum(1 for g in rows if g[0] in tile_bad_contig)
    rows = [g for g in rows if window_fits(g) and g[0] not in tile_bad_contig]
    report.append(f"[filter] contig_boundary = FALSE ({WINDOW}-nt centred window inside chromosome; carrying tile not flagged): "
                  f"kept {len(rows)}, dropped {before - len(rows)} (genes longer than the window: {n_long}; tile-flagged: {n_tile_flag})")
    before = len(rows)
    rows = [g for g in rows if g[0] not in overlap]
    report.append(f"[filter] no overlap with another gene on either strand (flank {args.overlap_flank} nt): kept {len(rows)}, dropped {before - len(rows)}")
    before = len(rows)
    rows = [g for g in rows if g[0] not in tile_zero_weight]
    report.append(f"[filter] train_weight > 0 on every carrying tile (train_weight <= 0 = hard-flagged): kept {len(rows)}, dropped {before - len(rows)}")
    eligible = rows
    report.append(f"[result] strict_holdout loci: {total_holdout}; eligible pool: {len(eligible)}; dropped in total: {total_holdout - len(eligible)}")

    # descriptive counts that other drafts of the rules would change (not applied)
    in_rejected = sum(1 for g in eligible if g[0] in corpus["rejected"])
    ge3 = sum(1 for g in eligible if gff_genes[g[1]]["exon_count"] >= 3)
    le20k = sum(1 for g in eligible if g[4] - g[3] <= 20000)
    both = sum(1 for g in eligible if gff_genes[g[1]]["exon_count"] >= 3 and g[4] - g[3] <= 20000)
    report.append(f"[info] eligible genes listed in rejected_records: {in_rejected}")
    report.append(f"[info] eligible with >= 2 introns (exon_count >= 3, MVP §1.2): {ge3}; gene length <= 20 kb (MVP §1.2): {le20k}; "
                  f"both: {both}. The MVP §1.2 '>= 3 A-EST molecules' rule needs the EST BAM and is not applied here.")

    ids = sorted(g[0] for g in eligible)
    if len(ids) != len(set(ids)):
        raise PanelError("duplicate gene_id among eligible loci")
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    by_id = {g[0]: g for g in eligible}
    out = []
    for shuffle_rank, gene_id in enumerate(ids, start=1):
        g = by_id[gene_id]
        ref = gff_genes[g[1]]
        start, fin = g[3] + 1, g[4]  # 1-based inclusive
        out.append({"shuffle_rank": shuffle_rank, "species_id": SPECIES, "gene_id": gene_id, "chromosome": g[2],
                    "start": start, "fin": fin, "strand": ref["strand"], "length": fin - start + 1,
                    "orthogroup_id": g[5] or "", "n_transcripts": ref["n_transcripts"],
                    "exon_count": ref["exon_count"], "intron_count": ref["exon_count"] - 1})
    return out


def rows_to_tsv(rows) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


def write_outputs(tsv: bytes, out: Path, report: list) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tsv)
    sha = sha256_bytes(tsv)
    Path(str(out) + ".sha256").write_text(f"{sha}  {out.name}\n")
    report.append(f"wrote {out} ({len(tsv)} bytes)")
    report.append(f"tsv_sha256={sha} -> {out}.sha256")


def run_remote(args) -> tuple[bytes, str]:
    """Execute this script on `args.remote` via ssh (source on stdin), TSV back on stdout, report on stderr."""
    source = Path(__file__).read_bytes()
    remote_args = ["--db", args.db, "--seed", str(args.seed),
                   "--min-exons", str(args.min_exons), "--overlap-flank", str(args.overlap_flank),
                   "--a22-flags", args.a22_flags, "--a22-md5", args.a22_md5, "--a30-flags", args.a30_flags,
                   "--a30-md5", args.a30_md5, "--a30-list", args.a30_list, "--a30-list-sha256", args.a30_list_sha256,
                   "--expect-md5", args.expect_md5, "--emit-stdout"]
    if args.skip_md5:
        remote_args.append("--skip-md5")
    cmd = ["ssh", "-o", "BatchMode=yes", args.remote, args.remote_python, "-", *remote_args]
    proc = subprocess.run(cmd, input=source, capture_output=True)
    if proc.returncode != 0:
        raise PanelError(f"remote run failed (exit {proc.returncode}):\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout, proc.stderr.decode()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help="frozen B5 corpus DuckDB (opened read-only)")
    ap.add_argument("--out", default="revision/protocols/b6_eligible_loci_v1.tsv", help="output TSV; <out>.sha256 is written beside it")
    ap.add_argument("--seed", type=int, default=20260914, help="seed of the single shuffle that defines shuffle_rank")
    ap.add_argument("--min-exons", type=int, default=2, help="minimum max-per-transcript exon count")
    ap.add_argument("--overlap-flank", type=int, default=0, help="nt added on both sides for the overlap test")
    ap.add_argument("--expect-md5", default=DEFAULT_DB_MD5, help="md5 the DB must have")
    ap.add_argument("--a22-flags", default=DEFAULT_A22_FLAGS, help="A22 GeenuFF flag file consumed by the a40 build")
    ap.add_argument("--a22-md5", default=DEFAULT_A22_MD5, help="expected md5 of --a22-flags ('' to skip)")
    ap.add_argument("--a30-flags", default=DEFAULT_A30_FLAGS, help="A30 Swiss-Prot flag file consumed by the a40 build")
    ap.add_argument("--a30-md5", default=DEFAULT_A30_MD5, help="expected md5 of --a30-flags ('' to skip)")
    ap.add_argument("--a30-list", default=DEFAULT_A30_LIST, help="A30 caution list (swissprot_hard_masked_genes_20260902.tsv)")
    ap.add_argument("--a30-list-sha256", default=DEFAULT_A30_LIST_SHA256, help="expected sha256 of --a30-list ('' to skip)")
    ap.add_argument("--skip-md5", action="store_true", help="do not hash the 23 GB DB (not for the frozen draw)")
    ap.add_argument("--remote", default=None, help="ssh host holding the DB; the script runs itself there")
    ap.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON, help="python with duckdb on --remote")
    ap.add_argument("--emit-stdout", action="store_true", help="write the TSV to stdout and the report to stderr (remote side)")
    args = ap.parse_args(argv)

    report: list = []
    try:
        if args.remote:
            tsv, remote_report = run_remote(args)
            report.extend(l for l in remote_report.splitlines() if l)
        else:
            rows = draw_panels(args, report)
            tsv = rows_to_tsv(rows)
            if args.emit_stdout:
                sys.stdout.buffer.write(tsv)
                sys.stdout.flush()
                print("\n".join(report), file=sys.stderr)
                return 0
        write_outputs(tsv, Path(args.out), report)
    except PanelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(io.StringIO(tsv.decode()), delimiter="\t"))
    report.append("eligible_rows=" + json.dumps({"rows": len(rows), "max_shuffle_rank": len(rows)}, sort_keys=True))
    report.append(f"script_sha256={digest(Path(__file__))} ({Path(__file__).name})")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
