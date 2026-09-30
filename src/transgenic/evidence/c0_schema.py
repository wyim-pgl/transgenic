"""Full C0 observation schema for B6 evidence mode (A46 rule 2), version ``c0-v1``.

Column-level DuckDB DDL for the nine C0 tables named in
``revision/protocols/IMPLEMENTATION_ORDER_B5_C0_C2_v1.md`` line 34, the route-B
``est_protein_hsp`` table (A46 rule 2/6) and the QC/provenance tables that the
builder (``revision/scripts/evidence/b6_build_c0.py``) and the merger
(``revision/scripts/evidence/b6_merge_evidence_db.py``) write.

Coordinate conventions (one convention everywhere)
--------------------------------------------------
* Intervals (alignments, blocks, introns, excluded intervals, HSP projections)
  are 0-based half-open ``[start0, end0)``.
* Junction boundary bases are also stored as ``donor0`` / ``acceptor0``:
  0-based, INCLUSIVE, the first and last base of the intron in transcript
  order, i.e. strand-aware.  ``+`` strand: ``donor0 = start0``,
  ``acceptor0 = end0 - 1``.  ``-`` strand: ``donor0 = end0 - 1``,
  ``acceptor0 = start0`` (so ``donor0 > acceptor0``).  Unknown strand (``?``):
  both NULL.  This is the ``est_junction`` definition of
  ``mvp_build_c0lite.py:266-268`` that SPEC-2c's chain cache consumes; it is
  NOT the coordinate-ordered pair ``gsf_contract.add_junction_evidence`` wants.
* ``59_evidence_support.parse_sam`` returns 1-based inclusive coordinates; the
  builder converts them exactly once, when a row is materialised.

Row identity columns (``COMMON_COLUMNS``) are carried by every data table:
species, source (est|ont|pacbio|protein), dataset, run, library,
genotype_stratum, role (training_eligible|validation_only), validation_only
(must equal ``role = 'validation_only'``) and provenance_sha256 (sha256 of the
canonical JSON build provenance of the per-species x source DB).  QC and
provenance tables (``c0_meta``, ``c0_provenance``, ``c0_qc``,
``excluded_interval``, ``heldout_locus``, ``molecule_exclusion``,
``audit_placement``, ``excluded_junction_coordinate``, ``merge_manifest``) are
bookkeeping, not observations, and carry species/source only.

Tables (grain -> key)
---------------------
======================  =======================================================  ==========================================
table                   grain                                                    primary key
======================  =======================================================  ==========================================
evidence_alignment      one SAM/GFF alignment record of one member (every kind,  alignment_id
                        accepted or rejected; unmapped kept for accounting)
molecule_member         one member (EST record / read / protein query) and its   member_id
                        A11/A16/A19 molecule unit and rule-10 molecule_state
aligned_block           one aligned genomic block of an accepted primary         (alignment_id, alignment_block_index)
                        alignment; block_index orders blocks within a molecule
junction_observation    one intron of an accepted primary alignment: raw and     observation_id
                        corrected coordinates + correction status (A18.2/A20)
partial_chain           the ordered intron chain of one member placement         chain_id
partial_chain_intron    one intron of a partial chain (genomic and tx order)     (chain_id, intron_index)
junction                one corrected accepted junction as seen in one           (junction_key, dataset, run, library, role)
                        dataset/run/library; molecule-unit counts
junction_support        one supporting molecule unit of a junction row;          (junction_key, dataset, run, library, role,
                        rule 4(c) flag                                            molecule_id)
chain_junction          chain -> accepted junction, transcript order             (chain_id, tx_order)
est_protein_hsp         one projected genomic interval of a route-B HSP (part B  (hsp_id, interval_index)
                        contract, 23 fixed columns first, C0 identity columns
                        appended; an HSP spanning introns has several rows)
======================  =======================================================  ==========================================

``est_protein_hsp`` keeps the 23 part-B columns in their fixed order
(``HSP_COLUMNS``).  Its own ``provenance_sha256`` is part B's (diamond run);
the C0 build provenance goes to the appended ``c0_provenance_sha256``.
``molecule_id`` must be a molecule of the same species' EST ``molecule_member``
(``b6_est_molecule_map.py`` is the single producer of those ids); the C0 build
and the merge gate refuse orphans.  ``leakage_excluded`` is true when that EST
molecule is ``excluded_heldout`` (or the interval itself hits an ``all``-scope
excluded interval), and ``train_tiles_excluded`` carries the EST molecule's flag
(or a ``train_tiles`` hit of the interval).  ``phase`` follows
``PHASE_CONVENTION`` (recorded in ``c0_provenance``).
"""
from __future__ import annotations

import re

import hashlib
import json

C0_SCHEMA_VERSION = "c0-v1"
PHASE_CONVENTION = "codon_position_at_genomic_start0_v1"

SOURCES = ("est", "ont", "pacbio", "protein")
ROLES = ("training_eligible", "validation_only")
MOLECULE_STATES = ("train_track", "excluded_heldout", "never")
GENOTYPE_STRATA = ("reference", "known_nonreference", "hybrid_pooled", "unknown", "not_applicable")
PLACEMENT_KINDS = ("primary", "supplementary", "secondary", "unmapped")
STRANDS = ("+", "-", "?")
CORRECTION_STATUSES = ("uncorrected", "canonical", "corrected", "noncanonical", "ambiguous_correction",
                       "anchor_failed", "strand_ambiguous", "training_noncanonical")


def _in(values):
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + ")"


COMMON_COLUMNS = (
    ("species", "VARCHAR NOT NULL"),
    ("source", f"VARCHAR NOT NULL CHECK (source IN {_in(SOURCES)})"),
    ("dataset", "VARCHAR NOT NULL"),
    ("run", "VARCHAR NOT NULL"),
    ("library", "VARCHAR NOT NULL"),
    ("genotype_stratum", f"VARCHAR NOT NULL CHECK (genotype_stratum IN {_in(GENOTYPE_STRATA)})"),
    ("role", f"VARCHAR NOT NULL CHECK (role IN {_in(ROLES)})"),
    ("validation_only", "BOOLEAN NOT NULL"),
    ("provenance_sha256", "VARCHAR NOT NULL CHECK (length(provenance_sha256) = 64)"),
)
COMMON_CHECKS = ("CHECK (validation_only = (role = 'validation_only'))",)

_STRAND = f"VARCHAR NOT NULL CHECK ({{c}} IN {_in(STRANDS)})"


def _strand(col):
    return _STRAND.format(c=col)


# Part B contract: exactly these 23 columns, in this order.
HSP_COLUMNS = (
    ("hsp_id", "VARCHAR NOT NULL"),
    ("species", "VARCHAR NOT NULL"),
    ("library", "VARCHAR NOT NULL"),
    ("molecule_id", "VARCHAR NOT NULL"),
    ("est_accession", "VARCHAR NOT NULL"),
    ("protein_id", "VARCHAR NOT NULL"),
    ("protein_source", "VARCHAR NOT NULL CHECK (protein_source IN ('orthodb', 'swissprot'))"),
    ("protein_species", "VARCHAR NOT NULL"),
    ("est_start", "BIGINT NOT NULL"),
    ("est_end", "BIGINT NOT NULL"),
    ("frame", "INTEGER NOT NULL"),
    ("evalue", "DOUBLE NOT NULL"),
    ("bitscore", "DOUBLE NOT NULL"),
    ("interval_index", "INTEGER NOT NULL"),
    ("chrom", "VARCHAR NOT NULL"),
    ("genomic_start0", "BIGINT NOT NULL"),
    ("genomic_end0", "BIGINT NOT NULL"),
    ("strand", "VARCHAR NOT NULL CHECK (strand IN ('+', '-'))"),
    ("phase", "INTEGER NOT NULL CHECK (phase IN (0, 1, 2))"),
    ("diamond_version", "VARCHAR NOT NULL"),
    ("database_sha256", "VARCHAR NOT NULL"),
    ("command_sha256", "VARCHAR NOT NULL"),
    ("provenance_sha256", "VARCHAR NOT NULL"),
)
HSP_COLUMN_NAMES = tuple(c for c, _ in HSP_COLUMNS)

# A11 / A49.1 decision 4: the biological library of a LONG-READ run is its ENA/SRA experiment accession, or
# '<BioSample>:<ENA library_name>' where one (sample, library_name) spans several experiments.  A run accession or any
# other text is not a library.  Shared by the metadata producer (b6_c0_metadata) and the C0 builder's ingestion, so
# checksum-valid metadata written under the old run-as-library rule cannot reach molecule assignment.
LONGREAD_LIBRARY_EXPERIMENT = re.compile(r"[SED]RX\d+")
LONGREAD_LIBRARY_SAMPLE_NAME = re.compile(r"SAM[NED][A-Z]?\d+:[^\t\r\n]+")


def is_longread_library(value):
    """True iff ``value`` is an A49.1 long-read library key (experiment accession or '<BioSample>:<library_name>')."""
    return bool(value) and bool(LONGREAD_LIBRARY_EXPERIMENT.fullmatch(value)
                                or LONGREAD_LIBRARY_SAMPLE_NAME.fullmatch(value))

# name -> (columns, primary key, extra table checks); COMMON_COLUMNS are prepended to data tables.
DATA_TABLES = {
    "evidence_alignment": ((
        ("alignment_id", "VARCHAR NOT NULL"),
        ("member_id", "VARCHAR NOT NULL"),
        ("molecule_id", "VARCHAR NOT NULL"),
        ("read_name", "VARCHAR NOT NULL"),
        ("record_index", "BIGINT NOT NULL"),
        ("placement_kind", f"VARCHAR NOT NULL CHECK (placement_kind IN {_in(PLACEMENT_KINDS)})"),
        ("flag", "INTEGER"),
        ("chrom", "VARCHAR"),
        ("start0", "BIGINT"),
        ("end0", "BIGINT"),
        ("alignment_strand", "VARCHAR"),
        ("transcript_strand", _strand("transcript_strand")),
        ("ts_tag", "VARCHAR"),
        ("mapq", "INTEGER"),
        ("cigar", "VARCHAR"),
        ("query_length", "BIGINT"),
        ("aligned_query", "BIGINT"),
        ("aligned_fraction", "DOUBLE"),
        ("identity", "DOUBLE"),
        ("accepted", "BOOLEAN NOT NULL"),
        ("rejection_reason", "VARCHAR"),
        ("mapping_ambiguous", "BOOLEAN NOT NULL"),
        ("arm", "VARCHAR NOT NULL"),
        ("complete_5p_code", "VARCHAR"),
        ("complete_3p_code", "VARCHAR"),
        ("read_internal_code", "VARCHAR"),
    ), ("alignment_id",), ("CHECK (start0 IS NULL OR (start0 >= 0 AND end0 > start0))",)),
    "molecule_member": ((
        ("member_id", "VARCHAR NOT NULL"),
        ("molecule_id", "VARCHAR NOT NULL"),
        ("read_name", "VARCHAR NOT NULL"),
        ("accession", "VARCHAR NOT NULL"),
        ("accession_stem", "VARCHAR NOT NULL"),
        ("accession_version", "INTEGER"),
        ("part_index", "INTEGER"),
        ("clone_key", "VARCHAR"),
        ("bioproject", "VARCHAR"),
        ("unit_rule", "VARCHAR NOT NULL"),
        ("superseded_version", "BOOLEAN NOT NULL"),
        ("molecule_state", f"VARCHAR NOT NULL CHECK (molecule_state IN {_in(MOLECULE_STATES)})"),
        ("train_tiles_excluded", "BOOLEAN NOT NULL"),
        ("exclusion_cause", "VARCHAR"),
    ), ("member_id",), ()),
    "aligned_block": ((
        ("alignment_id", "VARCHAR NOT NULL"),
        ("molecule_id", "VARCHAR NOT NULL"),
        ("alignment_block_index", "INTEGER NOT NULL"),
        ("block_index", "INTEGER NOT NULL"),
        ("chrom", "VARCHAR NOT NULL"),
        ("start0", "BIGINT NOT NULL"),
        ("end0", "BIGINT NOT NULL"),
        ("strand", _strand("strand")),
    ), ("alignment_id", "alignment_block_index"), ("CHECK (start0 >= 0 AND end0 > start0)",)),
    "junction_observation": ((
        ("observation_id", "VARCHAR NOT NULL"),
        ("alignment_id", "VARCHAR NOT NULL"),
        ("molecule_id", "VARCHAR NOT NULL"),
        ("intron_index", "INTEGER NOT NULL"),
        ("chrom", "VARCHAR NOT NULL"),
        ("strand", _strand("strand")),
        ("raw_start0", "BIGINT NOT NULL"),
        ("raw_end0", "BIGINT NOT NULL"),
        ("raw_donor0", "BIGINT"),
        ("raw_acceptor0", "BIGINT"),
        ("start0", "BIGINT"),
        ("end0", "BIGINT"),
        ("donor0", "BIGINT"),
        ("acceptor0", "BIGINT"),
        ("correction_status", f"VARCHAR NOT NULL CHECK (correction_status IN {_in(CORRECTION_STATUSES)})"),
        ("correction_radius", "INTEGER NOT NULL"),
        ("accepted", "BOOLEAN NOT NULL"),
        ("canonical", "BOOLEAN NOT NULL"),
        ("junction_key", "VARCHAR"),
    ), ("observation_id",), ("CHECK (raw_end0 > raw_start0)",)),
    "partial_chain": ((
        ("chain_id", "VARCHAR NOT NULL"),
        ("alignment_id", "VARCHAR NOT NULL"),
        ("molecule_id", "VARCHAR NOT NULL"),
        ("member_id", "VARCHAR NOT NULL"),
        ("placement_rank", "INTEGER NOT NULL"),
        ("chrom", "VARCHAR NOT NULL"),
        ("strand", _strand("strand")),
        ("start0", "BIGINT NOT NULL"),
        ("end0", "BIGINT NOT NULL"),
        ("n_introns", "INTEGER NOT NULL"),
        ("intron_chain_key", "VARCHAR NOT NULL"),
        ("all_junctions_accepted", "BOOLEAN NOT NULL"),
    ), ("chain_id",), ()),
    "partial_chain_intron": ((
        ("chain_id", "VARCHAR NOT NULL"),
        ("intron_index", "INTEGER NOT NULL"),
        ("tx_order", "INTEGER NOT NULL"),
        ("observation_id", "VARCHAR NOT NULL"),
        ("chrom", "VARCHAR NOT NULL"),
        ("strand", _strand("strand")),
        ("start0", "BIGINT NOT NULL"),
        ("end0", "BIGINT NOT NULL"),
        ("donor0", "BIGINT"),
        ("acceptor0", "BIGINT"),
        ("correction_status", "VARCHAR NOT NULL"),
        ("accepted", "BOOLEAN NOT NULL"),
    ), ("chain_id", "intron_index"), ()),
    "junction": ((
        ("junction_key", "VARCHAR NOT NULL"),
        ("chrom", "VARCHAR NOT NULL"),
        ("strand", "VARCHAR NOT NULL CHECK (strand IN ('+', '-'))"),
        ("start0", "BIGINT NOT NULL"),
        ("end0", "BIGINT NOT NULL"),
        ("donor0", "BIGINT NOT NULL"),
        ("acceptor0", "BIGINT NOT NULL"),
        ("n_molecules", "BIGINT NOT NULL"),
        ("n_molecules_train_track", "BIGINT NOT NULL"),
        ("n_molecules_library_capped", "BIGINT NOT NULL"),
        ("n_observations", "BIGINT NOT NULL"),
    ), ("junction_key", "dataset", "run", "library", "role"), ()),
    "junction_support": ((
        ("junction_key", "VARCHAR NOT NULL"),
        ("molecule_id", "VARCHAR NOT NULL"),
        ("n_observations", "INTEGER NOT NULL"),
        ("molecule_state", f"VARCHAR NOT NULL CHECK (molecule_state IN {_in(MOLECULE_STATES)})"),
        ("excluded_4c", "BOOLEAN NOT NULL"),
    ), ("junction_key", "dataset", "run", "library", "role", "molecule_id"), ()),
    "chain_junction": ((
        ("chain_id", "VARCHAR NOT NULL"),
        ("tx_order", "INTEGER NOT NULL"),
        ("junction_key", "VARCHAR NOT NULL"),
    ), ("chain_id", "tx_order"), ()),
}

HSP_EXTRA_COLUMNS = (
    ("source", "VARCHAR NOT NULL CHECK (source = 'protein')"),
    ("dataset", "VARCHAR NOT NULL"),
    ("run", "VARCHAR NOT NULL"),
    ("genotype_stratum", f"VARCHAR NOT NULL CHECK (genotype_stratum IN {_in(GENOTYPE_STRATA)})"),
    ("role", f"VARCHAR NOT NULL CHECK (role IN {_in(ROLES)})"),
    ("validation_only", "BOOLEAN NOT NULL"),
    ("c0_provenance_sha256", "VARCHAR NOT NULL CHECK (length(c0_provenance_sha256) = 64)"),
    ("leakage_excluded", "BOOLEAN NOT NULL"),
    ("train_tiles_excluded", "BOOLEAN NOT NULL"),
    ("exclusion_cause", "VARCHAR"),
)
HSP_PRIMARY_KEY = ("hsp_id", "interval_index")

QC_TABLES = {
    "c0_meta": ((("key", "VARCHAR NOT NULL"), ("value", "VARCHAR NOT NULL")), ("key",)),
    "c0_provenance": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("key", "VARCHAR NOT NULL"),
                       ("value", "VARCHAR NOT NULL")), ("species", "source", "key")),
    "c0_qc": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("metric", "VARCHAR NOT NULL"),
               ("value", "DOUBLE NOT NULL")), ("species", "source", "metric")),
    "excluded_interval": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("chrom", "VARCHAR NOT NULL"),
                           ("start0", "BIGINT NOT NULL"), ("end0", "BIGINT NOT NULL"), ("cause", "VARCHAR NOT NULL"),
                           ("scope", "VARCHAR NOT NULL CHECK (scope IN ('all', 'train_tiles'))"), ("gene_id", "VARCHAR"),
                           ("tile_rn", "BIGINT")), ()),
    "heldout_locus": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("gene_id", "VARCHAR NOT NULL"),
                       ("chrom", "VARCHAR NOT NULL"), ("start0", "BIGINT NOT NULL"), ("end0", "BIGINT NOT NULL")),
                      ("species", "source", "gene_id")),
    "molecule_exclusion": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("molecule_id", "VARCHAR NOT NULL"),
                            ("n_members", "INTEGER NOT NULL"), ("n_secondary", "INTEGER NOT NULL"),
                            ("n_placements", "INTEGER NOT NULL"), ("cause_a", "VARCHAR"), ("cause_a_direct", "BOOLEAN NOT NULL"),
                            ("cause_a_train_tiles", "VARCHAR"), ("unauditable_secondary", "BOOLEAN NOT NULL"),
                            ("unauditable_low_mapq", "BOOLEAN NOT NULL"), ("excluded", "BOOLEAN NOT NULL"),
                            ("train_tiles_excluded", "BOOLEAN NOT NULL"), ("audited", "BOOLEAN NOT NULL"),
                            ("low_mapq_secondary_only", "BOOLEAN NOT NULL")),
                           ("species", "source", "molecule_id")),
    "audit_placement": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"), ("molecule_id", "VARCHAR NOT NULL"),
                         ("member_id", "VARCHAR NOT NULL"), ("origin", "VARCHAR NOT NULL CHECK (origin IN ('frozen', 'audit'))"),
                         ("placement_kind", "VARCHAR NOT NULL"), ("chrom", "VARCHAR NOT NULL"), ("start0", "BIGINT NOT NULL"),
                         ("end0", "BIGINT NOT NULL"), ("mapq", "INTEGER")), ()),
    "excluded_junction_coordinate": ((("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"),
                                      ("chrom", "VARCHAR NOT NULL"), ("strand", "VARCHAR NOT NULL"),
                                      ("kind", "VARCHAR NOT NULL CHECK (kind IN ('donor', 'acceptor'))"),
                                      ("coordinate0", "BIGINT NOT NULL"), ("excluded_gene_id", "VARCHAR NOT NULL"),
                                      ("neighbour_gene_id", "VARCHAR NOT NULL")), ()),
    "merge_manifest": ((("input_path", "VARCHAR NOT NULL"), ("input_sha256", "VARCHAR NOT NULL"),
                        ("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL")), ("species", "source")),
}


def table_columns(name):
    """Ordered (column, type) pairs of any C0 table."""
    if name in DATA_TABLES:
        return COMMON_COLUMNS + DATA_TABLES[name][0]
    if name == "est_protein_hsp":
        return HSP_COLUMNS + HSP_EXTRA_COLUMNS
    return QC_TABLES[name][0]


ALL_TABLES = tuple(DATA_TABLES) + ("est_protein_hsp",) + tuple(QC_TABLES)


def ddl(name, *, primary_key=True):
    """CREATE TABLE for ``name``; ``primary_key=False`` omits the PRIMARY KEY clause only (columns and CHECKs kept)."""
    cols = table_columns(name)
    parts = [f"{c} {t}" for c, t in cols]
    if name in DATA_TABLES:
        _, pk, checks = DATA_TABLES[name]
        parts += list(COMMON_CHECKS) + list(checks)
    elif name == "est_protein_hsp":
        pk = HSP_PRIMARY_KEY
        parts += ["CHECK (validation_only = (role = 'validation_only'))",
                  "CHECK (genomic_start0 >= 0 AND genomic_end0 > genomic_start0)", "CHECK (est_end > est_start)"]
    else:
        pk = QC_TABLES[name][1]
    if pk and primary_key:
        parts.append("PRIMARY KEY (" + ", ".join(pk) + ")")
    return f"CREATE TABLE {name} (\n  " + ",\n  ".join(parts) + "\n)"


def primary_key(name):
    if name in DATA_TABLES:
        return DATA_TABLES[name][1]
    if name == "est_protein_hsp":
        return HSP_PRIMARY_KEY
    return QC_TABLES[name][1]


def create_schema(con, *, merged=False, data_primary_keys=True):
    """Create every C0 table and stamp the schema version; ``merge_manifest`` only in merged DBs.

    ``data_primary_keys=False`` (the merged evidence DB) creates the data tables and ``est_protein_hsp`` without their
    PRIMARY KEY: DuckDB builds a key's ART index in memory at commit, which does not fit for the 9-species merge;
    the merge enforces the same keys with its own duplicate check.  The logical keys (``primary_key``) and the
    schema hash are unchanged."""
    for name in ALL_TABLES:
        if name == "merge_manifest" and not merged:
            continue
        big = name in DATA_TABLES or name == "est_protein_hsp"
        con.execute(ddl(name, primary_key=data_primary_keys or not big))
    con.execute("INSERT INTO c0_meta VALUES ('schema_version', ?), ('schema_sha256', ?)",
                [C0_SCHEMA_VERSION, schema_sha256()])


def schema_sha256():
    """Hash of the full column-level schema; a drifted DDL cannot share a version string silently."""
    payload = {name: [list(c) for c in table_columns(name)] for name in ALL_TABLES}
    payload["_pk"] = {name: list(primary_key(name)) for name in ALL_TABLES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SchemaError(ValueError):
    pass


def verify_schema(con, *, merged=False):
    """Fail closed unless the DB carries this exact schema version and column list."""
    try:
        meta = dict(con.execute("SELECT key, value FROM c0_meta").fetchall())
    except Exception as exc:  # duckdb.CatalogException
        raise SchemaError(f"not a C0 database (no c0_meta): {exc}") from exc
    if meta.get("schema_version") != C0_SCHEMA_VERSION:
        raise SchemaError(f"schema version {meta.get('schema_version')!r} != {C0_SCHEMA_VERSION!r}")
    if meta.get("schema_sha256") != schema_sha256():
        raise SchemaError("schema_sha256 mismatch: DDL drifted under an unchanged version string")
    for name in ALL_TABLES:
        if name == "merge_manifest" and not merged:
            continue
        actual = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [name]).fetchall()]
        expected = [c for c, _ in table_columns(name)]
        if actual != expected:
            raise SchemaError(f"{name}: columns {actual} != schema {expected}")
    return meta
