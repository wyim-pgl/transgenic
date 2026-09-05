"""Unit tests for the per-species B5 merge (issue #50): rn uniqueness, every table, one gene_split, refusals."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
duckdb = pytest.importorskip("duckdb")
SCRIPT = ROOT / "scripts" / "merge_b5_databases.py"


def _load(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def b5():
    return _load(ROOT / "src" / "transgenic" / "datasets" / "build_b5.py", "build_b5")


MANIFEST = ("split_file_sha256", "rc_mode", "build_version", "ordering_version", "window_policy", "duckdb_version")


def _species_db(path, b5, species, n_rows, *, git_commit="abc1234", split_sha="904c6265", window_policy="tile6144-v3", seq_len=30720):
    """A minimal stand-in for one per-species build: rn restarts at 1, gene_split holds the whole frozen table."""
    con = duckdb.connect(str(path))
    b5.ensure_schema(con)
    # Model the pre-#57 frozen source schema; merge must accept missing diagnostics.
    con.sql("ALTER TABLE build_manifest DROP COLUMN tier_margin_unguaranteed")
    con.sql("CREATE TABLE IF NOT EXISTS window_genes (species_id VARCHAR, window_id VARCHAR, gene_id VARCHAR, is_rc BOOLEAN)")
    con.sql("CREATE TABLE IF NOT EXISTS tile_blocks (species_id VARCHAR, chromosome VARCHAR, start0 INT, end0 INT, split VARCHAR)")
    con.sql("CREATE TABLE IF NOT EXISTS rejected_records (species_id VARCHAR, gene_id VARCHAR, reason VARCHAR)")
    for i in range(n_rows):
        wid = f"{species}:Chr1:{i * 30720}-{(i + 1) * 30720}"
        con.execute(
            "INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, species_id, gene_id, split, "
            "is_rc, strict_holdout, ordering_version, build_version, split_file_sha256, window_policy, train_weight, gene_id_original) "
            "VALUES (nextval('row_id'), ?, ?, ?, '+', 'Chr1', ?, ?, ?, ?, 'train', false, false, 'gsf-order-v1', 'gsf-contract-v1', ?, ?, 1.0, ?)",
            [wid, i * 30720, (i + 1) * 30720, "A" * seq_len, "0|CDS1|300|+|A>CDS1", species, wid, split_sha, window_policy, wid])
    con.executemany("INSERT INTO window_genes VALUES (?,?,?,?)", [[species, f"{species}:Chr1:0-30720", f"{species}_g{i}", False] for i in range(3)])
    con.execute("INSERT INTO tile_blocks VALUES (?,?,?,?,?)", [species, "Chr1", 0, 100000, "train"])
    con.execute("INSERT INTO rejected_records VALUES (?,?,?)", [species, "gx", "masked fraction 0.7 > 0.6"])
    con.execute("INSERT INTO gene_key_map VALUES (?,?,?,?,?,?,?)", [species, "g1", "G1", "G1", "Chr1", 0, 100])
    # the frozen split table is loaded into every per-species build, identically
    con.executemany("INSERT INTO gene_split VALUES (?,?,?,?,?,?,?)",
                    [[sp, f"{sp}_g{i}", f"OG{i}", "train", False, 123, "v1"] for sp in ("Athaliana", "Gmax") for i in range(5)])
    con.execute("INSERT INTO build_manifest VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [species, "g.fa", "fa" * 32, "g.gff", "gf" * 32, split_sha, "isoform-only", n_rows, 0, 1, "{}",
                 "gsf-contract-v1", "gsf-order-v1", window_policy, git_commit, "2026-09-03T00:00:00", duckdb.__version__])
    con.close()


def _run(src, out, manifest, expect=2):
    return subprocess.run([sys.executable, str(SCRIPT), "--src-dir", str(src), "--out", str(out),
                           "--manifest", str(manifest), "--expect-species", str(expect)], capture_output=True, text=True)


@pytest.fixture
def two_species(tmp_path, b5):
    src = tmp_path / "full"
    src.mkdir()
    _species_db(src / "Athaliana.db", b5, "Athaliana", 4)
    _species_db(src / "Gmax.db", b5, "Gmax", 6)
    for s in ("Athaliana", "Gmax"):
        (src / f"{s}.DONE").write_text("2026-09-03T00:00:00")
    return src


def test_merge_renumbers_rn_and_keeps_every_row(two_species, tmp_path):
    out, man = tmp_path / "b5.db", tmp_path / "freeze.json"
    r = _run(two_species, out, man)
    assert r.returncode == 0, r.stderr
    con = duckdb.connect(str(out), read_only=True)
    assert con.sql("SELECT count(*), count(DISTINCT rn) FROM geneList").fetchone() == (10, 10)
    assert con.sql("SELECT min(rn), max(rn) FROM geneList").fetchone() == (1, 10)
    # species order is the sorted file order, and each species keeps a contiguous rn block
    assert con.sql("SELECT min(rn), max(rn) FROM geneList WHERE species_id = 'Athaliana'").fetchone() == (1, 4)
    assert con.sql("SELECT min(rn), max(rn) FROM geneList WHERE species_id = 'Gmax'").fetchone() == (5, 10)
    con.close()


def test_merge_copies_gene_split_once_and_other_tables_per_species(two_species, tmp_path):
    out, man = tmp_path / "b5.db", tmp_path / "freeze.json"
    assert _run(two_species, out, man).returncode == 0
    con = duckdb.connect(str(out), read_only=True)
    assert con.sql("SELECT count(*) FROM gene_split").fetchone()[0] == 10          # not 20: copied once
    assert con.sql("SELECT count(*) FROM build_manifest").fetchone()[0] == 2       # one row per species
    assert con.sql("SELECT count(*) FROM window_genes").fetchone()[0] == 6
    assert con.sql("SELECT count(*) FROM tile_blocks").fetchone()[0] == 2
    assert con.sql("SELECT count(*) FROM rejected_records").fetchone()[0] == 2
    assert con.sql("SELECT count(*) FROM gene_key_map").fetchone()[0] == 2
    con.close()


def test_freeze_manifest_records_content_hash_and_commits(two_species, tmp_path):
    out, man = tmp_path / "b5.db", tmp_path / "freeze.json"
    assert _run(two_species, out, man).returncode == 0
    m = json.loads(man.read_text())
    assert m["geneList_rows"] == 10
    assert len(m["geneList_content_sha256"]) == 64
    assert m["rows_by_species"] == {"Athaliana": 4, "Gmax": 6}
    assert m["git_commits"] == {"abc1234": ["Athaliana", "Gmax"]}
    assert len(m["file_md5"]) == 32
    assert m["frozen_inputs"]["window_policy"] == "tile6144-v3"


def test_content_hash_is_independent_of_the_duckdb_file(two_species, tmp_path):
    """Two merges of the same inputs agree on the content hash even if the file bytes differ."""
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    ma, mb = tmp_path / "a.json", tmp_path / "b.json"
    assert _run(two_species, a, ma).returncode == 0
    assert _run(two_species, b, mb).returncode == 0
    ja, jb = json.loads(ma.read_text()), json.loads(mb.read_text())
    assert ja["geneList_content_sha256"] == jb["geneList_content_sha256"]
    assert ja["table_content_sha256"] == jb["table_content_sha256"]


def test_merge_refuses_a_frozen_input_mismatch(tmp_path, b5):
    src = tmp_path / "full"
    src.mkdir()
    _species_db(src / "Athaliana.db", b5, "Athaliana", 3)
    _species_db(src / "Gmax.db", b5, "Gmax", 3, window_policy="tier6144-v2")   # different recipe
    for s in ("Athaliana", "Gmax"):
        (src / f"{s}.DONE").write_text("x")
    r = _run(src, tmp_path / "b5.db", tmp_path / "f.json")
    assert r.returncode != 0 and "disagree on a frozen input" in r.stdout + r.stderr
    assert not (tmp_path / "b5.db").exists()


def test_merge_refuses_a_missing_done_marker_and_an_existing_output(two_species, tmp_path):
    (two_species / "Gmax.DONE").unlink()
    r = _run(two_species, tmp_path / "b5.db", tmp_path / "f.json")
    assert r.returncode != 0 and "DONE" in r.stdout + r.stderr
    (two_species / "Gmax.DONE").write_text("x")
    out = tmp_path / "b5.db"
    assert _run(two_species, out, tmp_path / "f.json").returncode == 0
    r = _run(two_species, out, tmp_path / "f2.json")
    assert r.returncode != 0 and "refusing to overwrite" in r.stdout + r.stderr


def test_merge_refuses_a_wrong_species_count(two_species, tmp_path):
    r = _run(two_species, tmp_path / "b5.db", tmp_path / "f.json", expect=9)
    assert r.returncode != 0 and "expected 9" in r.stdout + r.stderr


def _big_species_db(path, b5, species, n_rows):
    """A wide-but-trivial source built entirely in SQL: 300k rows through executemany would dominate the suite."""
    con = duckdb.connect(str(path))
    b5.ensure_schema(con)
    # Model the pre-#57 frozen source schema; merge must accept missing diagnostics.
    con.sql("ALTER TABLE build_manifest DROP COLUMN tier_margin_unguaranteed")
    for ddl in ("window_genes (species_id VARCHAR, window_id VARCHAR, gene_id VARCHAR, is_rc BOOLEAN)",
                "tile_blocks (species_id VARCHAR, chromosome VARCHAR, start0 INT, end0 INT, split VARCHAR)",
                "rejected_records (species_id VARCHAR, gene_id VARCHAR, reason VARCHAR)"):
        con.sql(f"CREATE TABLE IF NOT EXISTS {ddl}")
    con.sql(f"""INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, species_id, gene_id,
                                      split, is_rc, strict_holdout, ordering_version, build_version, split_file_sha256,
                                      window_policy, train_weight, gene_id_original)
                SELECT i + 1, '{species}:Chr1:' || i, i * 100, (i + 1) * 100, '+', 'Chr1', 'A',
                       '0|CDS1|300|+|A>CDS1', '{species}', '{species}:Chr1:' || i, 'train', false, false,
                       'gsf-order-v1', 'gsf-contract-v1', '904c6265', 'tile6144-v3', 1.0, '{species}:Chr1:' || i
                FROM range({n_rows}) t(i)""")
    con.sql("DROP SEQUENCE IF EXISTS row_id")
    con.sql(f"CREATE SEQUENCE row_id START {n_rows + 1}")
    con.execute("INSERT INTO build_manifest VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [species, "g.fa", "fa" * 32, "g.gff", "gf" * 32, "904c6265", "isoform-only", n_rows, 0, 0, "{}",
                 "gsf-contract-v1", "gsf-order-v1", "tile6144-v3", "abc1234", "2026-09-03T00:00:00", duckdb.__version__])
    con.close()


def test_merged_rn_follows_the_source_rn_order(tmp_path, b5):
    """A sequence in the SELECT list is evaluated during the scan, not in ORDER BY order.

    The scramble appears once the sort exceeds one sort block: measured with duckdb 1.5.5, an
    `INSERT ... SELECT nextval(...) ... ORDER BY rn` is exact at 100,000 rows and wrong for 262,144 of
    300,000 (2 threads) and for 300,000 of 300,000 (16 threads). So the source here has to be larger than
    that boundary for the test to be able to fail at all; at 30,000 rows nextval looks perfectly correct.
    """
    src = tmp_path / "full"
    src.mkdir()
    n = 300000
    _big_species_db(src / "Athaliana.db", b5, "Athaliana", n)
    _big_species_db(src / "Gmax.db", b5, "Gmax", 1000)
    for s in ("Athaliana", "Gmax"):
        (src / f"{s}.DONE").write_text("x")
    out, man = tmp_path / "b5.db", tmp_path / "f.json"
    assert _run(src, out, man).returncode == 0
    con = duckdb.connect(str(out), read_only=True)
    con.sql(f"ATTACH '{src / 'Athaliana.db'}' AS a (READ_ONLY)")
    wrong = con.sql("SELECT count(*) FROM geneList m JOIN a.geneList s ON s.geneModel = m.geneModel "
                    "WHERE m.species_id = 'Athaliana' AND m.rn <> s.rn").fetchone()[0]
    assert wrong == 0, f"{wrong} of {n} Athaliana rows are not in source rn order"
    con.sql(f"ATTACH '{src / 'Gmax.db'}' AS g (READ_ONLY)")
    wrong_g = con.sql("SELECT count(*) FROM geneList m JOIN g.geneList s ON s.geneModel = m.geneModel "
                      "WHERE m.species_id = 'Gmax' AND m.rn <> s.rn + ?", params=[n]).fetchone()[0]
    assert wrong_g == 0
    con.close()


def test_merge_leaves_the_sequence_past_the_last_row(two_species, tmp_path):
    out = tmp_path / "b5.db"
    assert _run(two_species, out, tmp_path / "f.json").returncode == 0
    con = duckdb.connect(str(out))
    assert con.sql("SELECT nextval('row_id')").fetchone()[0] == 11   # 10 rows merged
    con.close()


def test_merge_refuses_an_excluded_species(tmp_path, b5):
    """--src-dir is globbed: a stray Zmays.db must not be merged in (docs/gsf_spec_v1.md 7-8)."""
    src = tmp_path / "full"
    src.mkdir()
    _species_db(src / "Athaliana.db", b5, "Athaliana", 3)
    _species_db(src / "Zmays.db", b5, "Zmays", 3)
    for s in ("Athaliana", "Zmays"):
        (src / f"{s}.DONE").write_text("x")
    r = _run(src, tmp_path / "b5.db", tmp_path / "f.json")
    assert r.returncode != 0 and "excluded species" in r.stdout + r.stderr
    assert not (tmp_path / "b5.db").exists()



def test_merge_preserves_new_margin_metadata_and_accepts_legacy_sources(two_species):
    src = two_species
    con = duckdb.connect(str(src / "Athaliana.db"))
    con.sql("ALTER TABLE build_manifest ADD COLUMN tier_margin_unguaranteed VARCHAR")
    value = json.dumps({"not_covered_with_margin": {"30720": 1}})
    con.execute("UPDATE build_manifest SET tier_margin_unguaranteed = ?", [value])
    con.close()
    out = src.parent / "with_margin.db"
    run = _run(src, out, src.parent / "with_margin.json")
    assert run.returncode == 0, run.stderr
    con = duckdb.connect(str(out), read_only=True)
    recorded = dict(con.sql("SELECT species_id, tier_margin_unguaranteed FROM build_manifest").fetchall())
    assert recorded == {"Athaliana": value, "Gmax": None}
    con.close()
