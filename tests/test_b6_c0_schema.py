"""C0 schema c0-v1: column-level DDL, fixed est_protein_hsp contract, identity columns, version pinning."""
import pytest

import test_b6_c0_world  # noqa: F401  (sys.path)
from transgenic.evidence import c0_schema as S

duckdb = pytest.importorskip("duckdb")

NINE = ("evidence_alignment", "molecule_member", "junction_observation", "partial_chain", "partial_chain_intron",
        "aligned_block", "junction", "junction_support", "chain_junction")


def test_schema_version_and_all_tables_created():
    con = duckdb.connect()
    S.create_schema(con)
    assert S.C0_SCHEMA_VERSION == "c0-v1"
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert set(NINE) | {"est_protein_hsp", "c0_qc", "c0_provenance", "excluded_interval", "molecule_exclusion"} <= names
    assert "merge_manifest" not in names
    assert S.verify_schema(con)["schema_version"] == "c0-v1"


def test_every_data_table_carries_identity_columns_and_role_flag_consistency():
    ident = ["species", "source", "dataset", "run", "library", "genotype_stratum", "role", "validation_only",
             "provenance_sha256"]
    for name in NINE:
        assert [c for c, _ in S.table_columns(name)][:9] == ident, name
    con = duckdb.connect()
    S.create_schema(con)
    row = ["Athaliana", "est", "d", "r", "L1", "reference", "training_eligible", True, "a" * 64, "c", 0, "m"]
    with pytest.raises(duckdb.ConstraintException):     # validation_only must equal role = 'validation_only'
        con.execute("INSERT INTO chain_junction VALUES (" + ",".join("?" * len(row)) + ")", row)
    row[7] = False
    con.execute("INSERT INTO chain_junction VALUES (" + ",".join("?" * len(row)) + ")", row)


def test_est_protein_hsp_keeps_the_part_b_contract_first_and_in_order():
    expected = ("hsp_id species library molecule_id est_accession protein_id protein_source protein_species est_start "
                "est_end frame evalue bitscore interval_index chrom genomic_start0 genomic_end0 strand phase "
                "diamond_version database_sha256 command_sha256 provenance_sha256").split()
    assert list(S.HSP_COLUMN_NAMES) == expected
    cols = [c for c, _ in S.table_columns("est_protein_hsp")]
    assert cols[:23] == expected and "role" in cols[23:] and "c0_provenance_sha256" in cols[23:]


def test_junction_observation_keeps_raw_and_corrected_coordinates_and_status():
    cols = [c for c, _ in S.table_columns("junction_observation")]
    for c in ("raw_start0", "raw_end0", "raw_donor0", "raw_acceptor0", "start0", "end0", "donor0", "acceptor0",
              "correction_status", "correction_radius"):
        assert c in cols
    assert [c for c, _ in S.table_columns("partial_chain_intron")].count("donor0") == 1


def test_verify_schema_refuses_other_version_and_drifted_columns():
    con = duckdb.connect()
    S.create_schema(con)
    con.execute("UPDATE c0_meta SET value = 'c0-v0' WHERE key = 'schema_version'")
    with pytest.raises(S.SchemaError, match="schema version"):
        S.verify_schema(con)
    con = duckdb.connect()
    S.create_schema(con)
    con.execute("ALTER TABLE junction ADD COLUMN extra INTEGER")
    with pytest.raises(S.SchemaError, match="junction: columns"):
        S.verify_schema(con)


def test_est_protein_hsp_key_is_hsp_and_interval_and_carries_train_tiles_flag():
    assert S.primary_key("est_protein_hsp") == ("hsp_id", "interval_index")
    assert "train_tiles_excluded" in [c for c, _ in S.table_columns("est_protein_hsp")][23:]
    assert "PRIMARY KEY (hsp_id, interval_index)" in S.ddl("est_protein_hsp")
    assert S.PHASE_CONVENTION == "codon_position_at_genomic_start0_v1"
