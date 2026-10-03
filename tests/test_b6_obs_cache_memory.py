"""DuckDB resource options of b6_build_obs_cache (--memory-limit / --threads / --temp-dir): applied to the evidence
connection and read back to stderr, an owned spill directory beside --out removed afterwards, a malformed limit
reported on the CLI's normal failure path, and the cache bytes unchanged by any of it."""
import json

import duckdb
import pytest

import b6_fixtures as FX
from test_b6_obs_cache_stream import BOC, builder_argv, tree_bytes

SETTINGS_PREFIX = "b6_build_obs_cache: duckdb settings "
SPILL_PREFIX = "duckdb spill directory: "
SPILL_GLOB = "b6_build_obs_cache.spill-*"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"))


def _build(world, tmp_path, out, extra=()):
    return BOC.main(builder_argv(world, tmp_path, out) + list(extra), roles_loader=lambda p: world.roles)


def _settings_line(err):
    lines = [ln for ln in err.splitlines() if ln.startswith(SETTINGS_PREFIX)]
    assert len(lines) == 1, err
    return json.loads(lines[0][len(SETTINGS_PREFIX):])


def _spill_line(err):
    lines = [ln for ln in err.splitlines() if ln.startswith(SPILL_PREFIX)]
    assert len(lines) == 1, err
    return lines[0][len(SPILL_PREFIX):]


def _scratch_readback(memory_limit, threads):
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET threads = {threads}")
        return con.execute("SELECT current_setting('memory_limit'), current_setting('threads')").fetchone()
    finally:
        con.close()


def test_resource_options_are_applied_and_read_back_and_leave_the_cache_bytes_unchanged(world, tmp_path, capsys):
    """Same code, with vs without the options: every file under --out (manifest included) is byte-identical, and the
    stderr line carries DuckDB's read-back values.  (Builder-vs-one-shot equivalence, which the base commit also
    satisfies, is test_b6_obs_cache_stream.test_builder_cache_is_byte_identical_to_one_shot_write_cache.)"""
    plain, limited, spill = tmp_path / "plain" / "cache", tmp_path / "limited" / "cache", tmp_path / "spill"
    spill.mkdir()
    assert _build(world, tmp_path, plain) == 0
    capsys.readouterr()
    assert _build(world, tmp_path, limited, ["--memory-limit", "1GB", "--threads", "2", "--temp-dir", str(spill)]) == 0
    err = capsys.readouterr().err
    want_mem, want_threads = _scratch_readback("1GB", 2)
    applied = _settings_line(err)
    assert applied["memory_limit"] == want_mem
    assert applied["threads"] == want_threads == 2
    assert applied["temp_directory"] == str(spill)
    assert set(applied) == {"memory_limit", "threads", "temp_directory"}  # nothing else is SET (insertion order kept)
    assert _spill_line(err) == f"{spill} (--temp-dir; not removed)"
    assert spill.is_dir()                                            # a given --temp-dir is left alone
    assert list(limited.parent.glob(SPILL_GLOB)) == []               # ... and no owned one is created
    assert tree_bytes(limited) == tree_bytes(plain)                  # every file, manifest included
    assert (limited / "manifest.json").read_bytes() == (plain / "manifest.json").read_bytes()
    assert any(name.startswith("shard.") for name in tree_bytes(limited))


def test_without_temp_dir_an_owned_spill_directory_beside_out_exists_during_the_run_and_not_afterwards(
        world, tmp_path, capsys, monkeypatch):
    out = tmp_path / "cache_dir" / "cache"
    seen = []
    real = BOC.extract_by_species

    def spy(*args, **kwargs):
        seen.extend(p for p in out.parent.glob(SPILL_GLOB) if p.is_dir())
        return real(*args, **kwargs)

    monkeypatch.setattr(BOC, "extract_by_species", spy)
    assert _build(world, tmp_path, out) == 0
    err = capsys.readouterr().err
    assert len(seen) == 1 and seen[0].parent == out.parent
    assert _spill_line(err) == f"{seen[0]} (removed on normal exit; delete it by hand after a kill)"
    assert _settings_line(err)["temp_directory"] == str(seen[0])
    assert not seen[0].exists()
    assert list(out.parent.glob(SPILL_GLOB)) == []
    assert (out / "manifest.json").is_file()


def test_malformed_memory_limit_fails_on_the_normal_failure_path_and_leaves_no_spill_directory(world, tmp_path):
    out = tmp_path / "cache_dir" / "cache"
    with pytest.raises(SystemExit) as info:
        _build(world, tmp_path, out, ["--memory-limit", "bogus"])
    assert isinstance(info.value.code, str)                          # the CLI's message path: exit 1, one line
    assert info.value.code.startswith("b6_build_obs_cache: ")
    assert "bogus" in info.value.code and "Traceback" not in info.value.code
    assert not out.parent.exists()                                   # nothing created before the settings held:
    assert list(tmp_path.glob("*/" + SPILL_GLOB)) == []              # no parent, no spill directory, no cache


def test_memory_limit_below_one_gb_is_refused_from_its_read_back_value(world, tmp_path):
    out = tmp_path / "cache_dir" / "cache"
    with pytest.raises(SystemExit) as info:
        _build(world, tmp_path, out, ["--memory-limit", "512MB"])     # SET accepts it; the floor check does not
    assert isinstance(info.value.code, str) and info.value.code.startswith("b6_build_obs_cache: ")
    assert "'512MB'" in info.value.code and "1 GB floor" in info.value.code
    assert not out.parent.exists()


def test_unwritable_temp_dir_fails_up_front_on_the_normal_failure_path(world, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    out = tmp_path / "cache_dir" / "cache"
    with pytest.raises(SystemExit) as info:
        _build(world, tmp_path, out, ["--temp-dir", str(blocker / "spill")])
    assert isinstance(info.value.code, str)
    assert info.value.code.startswith(f"b6_build_obs_cache: --temp-dir {blocker / 'spill'} is not a writable directory")
    assert not out.parent.exists()


def test_parse_memory_reads_duckdb_units():
    assert BOC.parse_memory("953.6 MiB") == pytest.approx(953.6 * 2**20)
    assert BOC.parse_memory("1.0 GiB") == 2**30
    assert BOC.parse_memory("12 bytes") == 12
    assert BOC.parse_memory("80%") is None and BOC.parse_memory("1GB") is None


def test_threads_one_builds_and_is_read_back(world, tmp_path, capsys):
    out = tmp_path / "cache_dir" / "cache"
    assert _build(world, tmp_path, out, ["--threads", "1"]) == 0
    assert _settings_line(capsys.readouterr().err)["threads"] == 1
    assert (out / "manifest.json").is_file()
