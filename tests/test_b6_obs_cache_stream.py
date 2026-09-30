"""Per-species observation-cache build (b6_build_obs_cache): SQL-level species filter in extract_rows, the
incremental CacheWriter, and byte identity of the streamed cache with the one-shot write_cache(extract_rows(...))."""
import hashlib
import json
import shutil
import sys

import duckdb
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import c0_schema as CS

sys.path.insert(0, str(FX.ROOT / "revision" / "scripts" / "evidence"))
import b6_build_obs_cache as BOC  # noqa: E402

OTHER = "Bdistachyon"          # second species of the two-species DB; its rows sit on OTHER_CHROM
OTHER_CHROM = "Chr2"
ID_COLUMNS = ("alignment_id", "member_id", "chain_id", "observation_id", "hsp_id", "junction_key")
COPIED = tuple(CS.DATA_TABLES) + ("est_protein_hsp", "excluded_junction_coordinate", "excluded_interval")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"))


def two_species_db(src, dst):
    """Copy of the merged fixture C0 plus every observation row re-labelled to OTHER (chrom OTHER_CHROM, ids
    suffixed so the merged primary keys stay unique; molecule ids are kept, they are only unique per species)."""
    shutil.copyfile(src, dst)
    con = duckdb.connect(str(dst))
    try:
        for table in COPIED:
            cols = [c for c, _ in CS.table_columns(table)]
            exprs = []
            for c in cols:
                if c == "species":
                    exprs.append(f"'{OTHER}'")
                elif c == "chrom":
                    exprs.append(f"'{OTHER_CHROM}'")
                elif c in ID_COLUMNS:
                    exprs.append(f"{c} || '~{OTHER}'")
                else:
                    exprs.append(c)
            con.execute(f"INSERT INTO {table} SELECT {', '.join(exprs)} FROM {table} WHERE species = ?", [FX.SPECIES])
        n = dict(con.execute("SELECT species, count(*) FROM evidence_alignment GROUP BY species").fetchall())
    finally:
        con.close()
    assert n[FX.SPECIES] == n[OTHER] > 0
    return dst


@pytest.fixture(scope="module")
def two(world, tmp_path_factory):
    return two_species_db(world.c0, tmp_path_factory.mktemp("two") / "two_species.db")


def tree_bytes(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def one_shot(db, roles, out, *, bindings, zero_lists):
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows, drops, libs = OC.extract_rows(con, roles, "train")
    finally:
        con.close()
    return OC.write_cache(out, rows, family="train", libraries=libs, drops=drops, bindings=bindings,
                          zero_lists=zero_lists)


def builder_argv(world, tmp_path, out):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(world.freeze))
    zero = sum((["--zero", f"{pop}:{FX.SPECIES}:{t}:{q}"] for pop, per in world.zero.items()
                for t, q in [per[FX.SPECIES]]), [])
    return ["--evidence", str(world.c0), "--family", "train", "--roles", "roles.tsv",
            "--reference", f"{FX.SPECIES}={world.fasta}", "--corpus", str(world.corpus), "--freeze", str(freeze),
            "--out", str(out)] + zero


class _Counting:
    """Connection proxy recording how many rows each execute(...).fetchall() brought into Python."""

    def __init__(self, con):
        self.con, self.fetched = con, []

    def execute(self, *args):
        cur = self.con.execute(*args)
        outer = self

        class _Result:
            def fetchall(self):
                rows = cur.fetchall()
                outer.fetched.append(len(rows))
                return rows
        return _Result()


# -------------------------------------------------------------------------------- builder (single species)
def test_builder_cache_is_byte_identical_to_one_shot_write_cache(world, tmp_path):
    out = tmp_path / "streamed"
    assert BOC.main(builder_argv(world, tmp_path, out), roles_loader=lambda p: world.roles) == 0
    bindings = json.loads((out / "manifest.json").read_text())["bindings"]
    ref = tmp_path / "one_shot"
    one_shot(world.c0, world.roles, ref, bindings=bindings, zero_lists=world.zero)
    assert (out / "manifest.json").read_bytes() == (ref / "manifest.json").read_bytes()
    assert tree_bytes(out) == tree_bytes(ref)
    assert any(name.startswith("shard.") for name in tree_bytes(out))


def test_builder_reports_one_progress_line_per_species(world, tmp_path, capsys):
    out = tmp_path / "streamed"
    assert BOC.main(builder_argv(world, tmp_path, out), roles_loader=lambda p: world.roles) == 0
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("b6_build_obs_cache: species ")]
    n = sum(s["rows"] for s in OC.ObsCache(out).manifest["shards"])
    assert err == [f"b6_build_obs_cache: species {FX.SPECIES}: {n} rows"]


# -------------------------------------------------------------------------------- two-species evidence DB
def test_per_species_extraction_equals_one_shot_extraction_on_two_species(world, two):
    con = duckdb.connect(str(two), read_only=True)
    try:
        rows, drops, libs = OC.extract_rows(con, world.roles, "train")
        per = {sp: OC.extract_rows(con, world.roles, "train", species=sp) for sp in sorted({FX.SPECIES, OTHER})}
    finally:
        con.close()
    assert {r["species"] for r in rows} == {FX.SPECIES, OTHER}
    for sp, (rs, _d, _l) in per.items():           # same rows in the same relative order (stable shard sort)
        assert rs == [r for r in rows if r["species"] == sp]
    key = lambda r: json.dumps(r, sort_keys=True)  # noqa: E731
    assert sorted(map(key, sum((p[0] for p in per.values()), []))) == sorted(map(key, rows))
    merged_drops = {}
    for _rs, d, _l in per.values():
        assert not set(d) & set(merged_drops)
        merged_drops.update(d)
    assert merged_drops == drops and any(k[0] == OTHER for k in drops)
    merged_libs = {}
    for _rs, _d, lb in per.values():
        for pop, by in lb.items():
            merged_libs.setdefault(pop, {}).update(by)
    assert merged_libs == libs and set(libs) == set(OC.POPULATIONS)
    assert any(OTHER in by for by in libs.values())


def test_streamed_writer_is_byte_identical_to_one_shot_on_two_species(world, two, tmp_path):
    bindings = {"evidence_sha256": FX._sha(two), "roles_sha256": world.roles.sha256}
    con = duckdb.connect(str(two), read_only=True)
    try:
        writer = OC.CacheWriter(tmp_path / "streamed", family="train")
        drops, libs = BOC.extract_by_species(con, world.roles, "train", [OTHER, FX.SPECIES], writer)
    finally:
        con.close()
    man = writer.finish(libraries=libs, drops=drops, bindings=bindings, zero_lists=world.zero)
    ref = one_shot(two, world.roles, tmp_path / "one_shot", bindings=bindings, zero_lists=world.zero)
    assert man == ref
    assert [(s["species"], s["chrom"]) for s in man["shards"]] == [(FX.SPECIES, FX.CHROM), (OTHER, OTHER_CHROM)]
    assert tree_bytes(tmp_path / "streamed") == tree_bytes(tmp_path / "one_shot")
    OC.ObsCache(tmp_path / "streamed")               # the streamed cache opens and verifies


def test_species_filter_restricts_every_query_at_sql_level(world, two):
    single = _Counting(duckdb.connect(str(world.c0), read_only=True))
    double = _Counting(duckdb.connect(str(two), read_only=True))
    try:
        ref = OC.extract_rows(single, world.roles, "train")
        got = OC.extract_rows(double, world.roles, "train", species=FX.SPECIES)
    finally:
        single.con.close()
        double.con.close()
    assert got == ref
    # every query fetches exactly the single-species DB's rows: nothing of OTHER is brought into Python
    assert double.fetched == single.fetched and sum(single.fetched) > 0


def test_species_filter_never_returns_rows_of_another_species(world, two):
    con = duckdb.connect(str(two), read_only=True)
    try:
        rows, drops, libs = OC.extract_rows(con, world.roles, "train", species=OTHER)
    finally:
        con.close()
    assert rows and {r["species"] for r in rows} == {OTHER} and {r["chrom"] for r in rows} == {OTHER_CHROM}
    assert {k[0] for k in drops} == {OTHER}
    assert all(set(by) <= {OTHER} for by in libs.values())


def test_per_species_build_refuses_inconsistent_molecule_states(world, tmp_path):
    db = tmp_path / "bad.db"
    shutil.copyfile(world.c0, db)
    con = duckdb.connect(str(db))
    cols = [c for c, _ in CS.table_columns("molecule_member")]
    exprs = [("member_id || '~dup'" if c == "member_id" else
              "CASE WHEN molecule_state = 'never' THEN 'train_track' ELSE 'never' END" if c == "molecule_state"
              else c) for c in cols]
    con.execute(f"INSERT INTO molecule_member SELECT {', '.join(exprs)} FROM molecule_member LIMIT 1")
    con.close()
    con = duckdb.connect(str(db), read_only=True)
    try:
        with pytest.raises(OC.CacheError, match="inconsistent state"):
            OC.extract_rows(con, world.roles, "train", species=FX.SPECIES)
        with pytest.raises(OC.CacheError, match="inconsistent state"):
            BOC.extract_by_species(con, world.roles, "train", [FX.SPECIES], OC.CacheWriter(tmp_path / "c",
                                                                                          family="train"))
    finally:
        con.close()


# -------------------------------------------------------------------------------- writer
def test_add_rows_refuses_a_shard_written_twice(world, tmp_path):
    writer = OC.CacheWriter(tmp_path / "c", family="train")
    writer.add_rows(world.rows)
    with pytest.raises(OC.CacheError, match="already written"):
        writer.add_rows([r for r in world.rows if r["chrom"] == FX.CHROM][:1])


def test_cache_writer_refuses_a_non_empty_directory(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "x").write_text("x")
    with pytest.raises(OC.CacheError, match="immutable"):
        OC.CacheWriter(tmp_path / "c", family="train")
