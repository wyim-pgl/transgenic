"""ObsCache reader memory: compact string columns, a bounded shard LRU and sha256 verification once per file identity.

Every accessor must return exactly what the object-array reader of 9f003e1 returned (``_reference_shard`` below is a
verbatim copy of its ``ObsCache.shard`` body), and the rasterised tracks / chains of the fixture world must not change.
"""
import os
import tracemalloc

import numpy as np
import pyarrow.parquet as pq
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_aug as AG
from transgenic.evidence import b6_chains as CH
from transgenic.evidence import b6_context as CX
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import b6_tracks as TR

KINDS = {"track": ("alignment", "block", "junction", "protein", "phase"), "chain": ("chain",),
         "xjunc": ("donor", "acceptor"), "xint": ("heldout", "edge_partial")}
CHROMS = ("Chr1", "Chr2", "Chr3", "Chr4")


def _reference_shard(path, meta):
    """The 9f003e1 ``ObsCache.shard`` load (object arrays for strings), returned as an ``OC.Shard``."""
    f = path / meta["file"]
    t = pq.read_table(f)
    cols = {}
    for c in OC.COLUMNS:
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
    return OC.Shard(cols, ti, ts, tml, ci, cs, cml, xj, xi)


class ReferenceCache(OC.ObsCache):
    """ObsCache whose shards come from the 9f003e1 reader (unbounded, as before)."""

    def shard(self, species, chrom):
        key = (species, chrom)
        if key not in self._shards:
            meta = self._meta.get(key)
            if meta is None:
                return super().shard(species, chrom)
            self._shards[key] = _reference_shard(self.path, meta)
        return self._shards[key]


def synthetic_rows(rng, n, chrom, *, species="Athaliana"):
    rows = []
    for k in range(n):
        rt = rng.choice(["track"] * 6 + ["chain", "chain", "xjunc", "xint"])
        s = int(rng.integers(0, 50_000))
        r = {"row_type": rt, "kind": str(rng.choice(KINDS[rt])), "species": species, "chrom": chrom,
             "source": str(rng.choice(["est", "ont", "pacbio", "protein"])), "dataset": f"ds{rng.integers(5)}",
             "run": f"SRR{rng.integers(7)}", "library": str(rng.choice(["L1", "L2", "", None])),
             "molecule_id": f"m{chrom}_{k // 3}_{rng.integers(1_000_000)}",   # high cardinality
             "molecule_state": str(rng.choice(["train_track", "excluded_heldout"])),
             "train_tiles_excluded": bool(rng.integers(2)), "b6_input": bool(rng.integers(2)), "b6_score": False,
             "b6_input_scope": str(rng.choice(["tracks", "tracks+chains"])),
             "strand": str(rng.choice(["+", "-", "?", "all", "train_tiles"])),
             "phase": int(rng.integers(3)) if rng.random() < 0.5 else None,
             "start0": s, "end0": s + int(rng.integers(1, 3000)),
             "organism": str(rng.choice(["3701:0001", "4577:0003"])) if rng.random() < 0.3 else None,
             "introns": f"{s + 10}-{s + 90},{s + 200}-{s + 260}" if rt == "chain" else None,
             "introns_ok": "1,0" if rt == "chain" else None, "units_key": f"g{rng.integers(50)}",
             "tile_rn": int(rng.integers(-1, 9)), "query_id": f"q{rng.integers(400)}" if rng.random() < 0.5 else ""}
        rows.append(r)
    return rows


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    rng = np.random.default_rng(20261005)
    out = tmp_path_factory.mktemp("synth") / "cache"
    rows = [r for i, c in enumerate(CHROMS) for r in synthetic_rows(rng, 1500 + 700 * i, c)]
    OC.write_cache(out, rows, family="train", libraries={}, drops={}, bindings={})
    return out


WINDOWS = [(0, 1), (0, 600), (100, 5000), (4999, 5001), (20_000, 20_600), (30_000, 60_000), (-100, 10), (52_999, 60_000)]


def _canon(values):
    return [("nan",) if isinstance(x, float) and x != x else x for x in values]   # NaN (null phase) != NaN


def _assert_same_values(a, b):
    assert type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape
    assert [type(x) for x in a.tolist()] == [type(x) for x in b.tolist()]
    assert _canon(a.tolist()) == _canon(b.tolist())


@pytest.mark.parametrize("vocab_max", [OC.DICT_VOCAB_MAX, 4])
def test_accessors_equal_the_object_array_reader(synth, monkeypatch, vocab_max):
    monkeypatch.setattr(OC, "DICT_VOCAB_MAX", vocab_max)      # 4 forces the Arrow-dictionary path for most columns
    new, ref = OC.ObsCache(synth, max_shards=2), ReferenceCache(synth)
    for chrom in CHROMS + ("ChrNone",):
        for ws, we in WINDOWS:
            for acc in ("track_rows", "chain_rows"):
                (sh_n, idx_n), (sh_r, idx_r) = getattr(new, acc)("Athaliana", chrom, ws, we), \
                    getattr(ref, acc)("Athaliana", chrom, ws, we)
                assert idx_n.dtype == idx_r.dtype and idx_n.tolist() == idx_r.tolist()
                for c in OC.COLUMNS:
                    _assert_same_values(sh_n.cols[c][idx_n], sh_r.cols[c][idx_r])
                    for i in idx_n[:25]:
                        a, b = sh_n.cols[c][i], sh_r.cols[c][i]
                        assert type(a) is type(b) and (a == b or (a != a and b != b))
                    mask = np.zeros(len(idx_n), bool)
                    mask[::2] = True
                    if len(idx_n):
                        _assert_same_values(sh_n.cols[c][idx_n[mask]], sh_r.cols[c][idx_r[mask]])
        assert new.excluded_junctions("Athaliana", chrom) == ref.excluded_junctions("Athaliana", chrom)
        assert new.excluded_intervals("Athaliana", chrom) == ref.excluded_intervals("Athaliana", chrom)
        sn, sr = new.shard("Athaliana", chrom), ref.shard("Athaliana", chrom)
        assert [type(x) for v in sn.xjunc + sn.xint for x in v] == [type(x) for v in sr.xjunc + sr.xint for x in v]
        for a in ("track_idx", "track_start", "chain_idx", "chain_start"):
            assert getattr(sn, a).dtype == getattr(sr, a).dtype and np.array_equal(getattr(sn, a), getattr(sr, a))
        assert (sn.track_maxlen, sn.chain_maxlen) == (sr.track_maxlen, sr.chain_maxlen)


def test_lru_keeps_at_most_the_bound_and_reloads_identically(synth):
    cache = OC.ObsCache(synth, max_shards=2)
    first = cache.track_rows("Athaliana", "Chr1", 0, 60_000)
    before = {c: _canon(first[0].cols[c][first[1]].tolist()) for c in OC.COLUMNS}
    for chrom in CHROMS:
        cache.shard("Athaliana", chrom)
        assert len(cache._shards) <= 2
    assert ("Athaliana", "Chr1") not in cache._shards
    sh, idx = cache.track_rows("Athaliana", "Chr1", 0, 60_000)
    assert sh is not first[0] and idx.tolist() == first[1].tolist()
    assert {c: _canon(sh.cols[c][idx].tolist()) for c in OC.COLUMNS} == before
    assert list(cache._shards) == [("Athaliana", "Chr4"), ("Athaliana", "Chr1")]


def test_lru_default_and_env(synth, monkeypatch):
    monkeypatch.delenv(OC.MAX_SHARDS_ENV, raising=False)
    assert OC.ObsCache(synth).max_shards == OC.DEFAULT_MAX_SHARDS == 512            # > the 455 train_v1 shards
    monkeypatch.setenv(OC.MAX_SHARDS_ENV, "3")
    assert OC.ObsCache(synth).max_shards == 3
    assert OC.ObsCache(synth, max_shards=5).max_shards == 5          # an explicit argument wins over the env
    cache = OC.ObsCache(synth, max_shards=1)
    for chrom in CHROMS:
        cache.shard("Athaliana", chrom)
        assert list(cache._shards) == [("Athaliana", chrom)]          # the newest shard is always kept
    for bad in ("0", "abc", "1.5", "-3", "\u00b2"):                       # "\u00b2" (superscript 2): isdigit, not decimal
        monkeypatch.setenv(OC.MAX_SHARDS_ENV, bad)
        with pytest.raises(OC.CacheError, match="positive integer"):
            OC.ObsCache(synth)
    monkeypatch.delenv(OC.MAX_SHARDS_ENV)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(OC.CacheError, match="positive integer"):
            OC.ObsCache(synth, max_shards=bad)


def _hash_calls(monkeypatch):
    calls = []
    real = OC.sha256_file
    monkeypatch.setattr(OC, "sha256_file", lambda p: calls.append(os.path.basename(p)) or real(p))
    return calls


def test_derived_files_sit_next_to_the_cache_and_leave_it_untouched(tmp_path):
    rng = np.random.default_rng(3)
    out = tmp_path / "train_v1"
    OC.write_cache(out, synthetic_rows(rng, 300, "Chr1") + synthetic_rows(rng, 200, "Chr2"), family="train",
                   libraries={}, drops={}, bindings={})
    before = {q.name: q.read_bytes() for q in out.iterdir()}
    cache = OC.ObsCache(out)
    assert cache.mmap_dir == tmp_path / "train_v1.mmap_v1"
    for chrom in ("Chr1", "Chr2"):
        cache.shard("Athaliana", chrom)
    assert {q.name: q.read_bytes() for q in out.iterdir()} == before
    shas = sorted(m["sha256"] for m in cache.manifest["shards"])
    assert sorted(q.name for q in cache.mmap_dir.glob("*.arrow")) == [f"{h}.arrow" for h in shas]
    assert not list(cache.mmap_dir.glob("*.tmp"))


def test_mmap_dir_env_and_argument(synth, tmp_path, monkeypatch):
    monkeypatch.setenv(OC.MMAP_DIR_ENV, str(tmp_path / "env"))
    assert OC.ObsCache(synth).mmap_dir == tmp_path / "env"
    assert OC.ObsCache(synth, mmap_dir=tmp_path / "arg").mmap_dir == tmp_path / "arg"


def test_columns_are_zero_copy_views_of_the_mapped_file(synth, tmp_path):
    sh = OC.ObsCache(synth, mmap_dir=tmp_path / "m").shard("Athaliana", "Chr4")
    n = len(sh.cols["start0"])
    for c, f in zip(OC.SCHEMA.names, OC.SCHEMA):
        col = sh.cols[c]
        arr = col.codes if f.type == "string" else col
        if f.type == "string":
            assert isinstance(col, OC.StrColumn) and col.codes.dtype in (np.int8, np.int16, np.int32) and len(col) == n
            assert col.vocab is None or len(col.vocab) <= OC.DICT_VOCAB_MAX   # Python strs bounded by the vocab
        elif c == "phase":                                             # nulls -> float64 NaN, as to_numpy gives
            assert col.dtype == np.float64
        assert isinstance(arr, np.ndarray) and len(arr) == n
        assert not arr.flags.writeable and not arr.flags.owndata, c    # a view of the mapped Arrow buffer
    for a in (sh.track_idx, sh.chain_idx, sh.track_start, sh.chain_start):
        assert a.dtype == np.int64 and not a.flags.owndata


def test_parquet_hashed_once_then_derived_file_reused(synth, tmp_path, monkeypatch):
    calls = _hash_calls(monkeypatch)
    mm = tmp_path / "m"
    cache = OC.ObsCache(synth, max_shards=1, mmap_dir=mm)
    calls.clear()
    for chrom in ("Chr1", "Chr2", "Chr1", "Chr2"):
        cache.shard("Athaliana", chrom)
    assert calls == ["shard.Athaliana.Chr1.parquet", "shard.Athaliana.Chr2.parquet"]
    calls.clear()
    other = OC.ObsCache(synth, mmap_dir=mm)                             # another process: derived files exist
    calls.clear()
    other.shard("Athaliana", "Chr1")
    assert calls == []


def test_changed_parquet_refused_before_any_derived_file_is_built(synth, tmp_path):
    import shutil
    copy = tmp_path / "cache"
    shutil.copytree(synth, copy)
    f = copy / "shard.Athaliana.Chr1.parquet"
    data = f.read_bytes()
    f.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    cache = OC.ObsCache(copy)
    with pytest.raises(OC.CacheError, match="sha256 changed"):
        cache.shard("Athaliana", "Chr1")
    assert not cache.derived_path("Athaliana", "Chr1").exists()


def test_derived_file_of_another_shard_is_refused(synth, tmp_path):
    mm = tmp_path / "m"
    cache = OC.ObsCache(synth, mmap_dir=mm)
    cache.shard("Athaliana", "Chr1")
    (mm / cache.derived_path("Athaliana", "Chr1").name).replace(cache.derived_path("Athaliana", "Chr2"))
    with pytest.raises(OC.CacheError, match="does not match manifest shard"):
        OC.ObsCache(synth, mmap_dir=mm).shard("Athaliana", "Chr2")


def test_unverified_reader_writes_no_derived_file(synth, tmp_path):
    cache = OC.ObsCache(synth, verify_shards=False, mmap_dir=tmp_path / "m")
    ref = ReferenceCache(synth)
    sh, idx = cache.track_rows("Athaliana", "Chr2", 0, 60_000)
    rsh, ridx = ref.track_rows("Athaliana", "Chr2", 0, 60_000)
    assert idx.tolist() == ridx.tolist()
    for c in OC.COLUMNS:
        _assert_same_values(sh.cols[c][idx], rsh.cols[c][ridx])
    assert not (tmp_path / "m").exists()


def _build_in_child(args):
    path, mm, chrom = args
    sh, idx = OC.ObsCache(path, mmap_dir=mm).track_rows("Athaliana", chrom, 0, 60_000)
    return [_canon(sh.cols[c][idx].tolist()) for c in OC.COLUMNS]


def test_concurrent_processes_build_one_valid_derived_file(synth, tmp_path):
    import multiprocessing as mp
    mm = tmp_path / "m"
    with mp.get_context("fork").Pool(4) as pool:
        results = pool.map(_build_in_child, [(synth, mm, "Chr3")] * 8)
    assert all(r == results[0] for r in results)
    assert _build_in_child((synth, tmp_path / "fresh", "Chr3")) == results[0]
    assert len(list(mm.glob("*.arrow"))) == 1 and not list(mm.glob("*.tmp"))


def test_prebuild_builds_then_checks_every_shard(synth, tmp_path, monkeypatch):
    import b6_prebuild_obs_mmap as PB
    mm = tmp_path / "m"
    recs = OC.prebuild_mmap(synth, mmap_dir=mm)
    assert [(r["chrom"], r["built"]) for r in recs] == [(c, True) for c in CHROMS]
    assert all(r["source_sha256"] in r["derived"] for r in recs)
    calls = _hash_calls(monkeypatch)
    assert PB.main([str(synth), "--mmap-dir", str(mm)]) == 0
    # each parquet re-verified once against the manifest; the existing derived files are checked, not rebuilt
    assert [c for c in calls if c.endswith(".parquet")] == [f"shard.Athaliana.{c}.parquet" for c in CHROMS]
    assert [r["built"] for r in OC.prebuild_mmap(synth, mmap_dir=mm)] == [False] * len(CHROMS)


def test_compact_shard_is_much_smaller_than_object_arrays(tmp_path):
    rng = np.random.default_rng(7)
    out = tmp_path / "cache"
    OC.write_cache(out, synthetic_rows(rng, 30_000, "Chr1"), family="train", libraries={}, drops={}, bindings={})
    meta = OC.ObsCache(out)._meta[("Athaliana", "Chr1")]

    def traced(load):
        tracemalloc.start()
        keep = load()
        size = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        del keep
        return size

    old = traced(lambda: _reference_shard(out, meta))
    new = traced(lambda: OC.ObsCache(out).shard("Athaliana", "Chr1"))
    assert new * 3 < old, (new, old)


# ------------------------------------------------------------------------------------------- fixture world tensors
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"), molecules=FX.MOLECULES + FX.cap_molecules())


def test_world_tracks_and_chains_unchanged(world):
    new, ref = OC.ObsCache(world.cache_dir, max_shards=1), ReferenceCache(world.cache_dir)
    corpora = CX.B6Corpus(world.corpus, new), CX.B6Corpus(world.corpus, ref)
    n = 0
    for rn, _start, _rc, split in FX.TILES:
        if split not in ("train", "valid"):
            continue
        mode = f"{split}_track"
        for epoch in range(4):
            out = []
            for cache, corpus in zip((new, ref), corpora):
                ctx = CX.corpus_context(corpus, rn, mode)
                owner = AG.Owner.from_pairs(ctx.species, cache.owner_libraries(mode, ctx.species))
                resolved = AG.resolve(AG.make_draw(20260914, ctx.row, epoch), owner)
                tracks, qc = TR.rasterise_with_qc(cache, ctx, resolved)
                tile = CH.build_chains(cache, ctx, resolved)
                out.append((tracks, qc, tile))
            (t1, q1, c1), (t2, q2, c2) = out
            assert t1.dtype == t2.dtype and t1.tobytes() == t2.tobytes() and q1 == q2
            for a in ("chain_pos", "chain_feat", "chain_mask"):
                assert getattr(c1, a).tobytes() == getattr(c2, a).tobytes()
            assert c1.qc == c2.qc
            n += 1
    assert n >= 20


# ------------------------------------------------------------------------------------------- review follow-ups
def test_open_decodes_nothing_and_xj_xi_are_lazy_and_equal(synth, tmp_path):
    cache, ref = OC.ObsCache(synth, mmap_dir=tmp_path / "m"), ReferenceCache(synth)
    for chrom in CHROMS:
        sh = cache.shard("Athaliana", chrom)
        assert sh._xjunc is None and sh._xint is None
        assert all(c._vocab is None for c in sh.cols.values() if isinstance(c, OC.StrColumn))
        assert cache.excluded_junctions("Athaliana", chrom) == ref.excluded_junctions("Athaliana", chrom)
        assert cache.excluded_intervals("Athaliana", chrom) == ref.excluded_intervals("Athaliana", chrom)
        assert sh._xjunc is not None and sh._xint is not None and cache.shard("Athaliana", chrom) is sh
        assert sh.cols["molecule_id"]._vocab is None                      # untouched columns stay undecoded


def _one_shard_cache(tmp_path, rows):
    out = tmp_path / "c"
    OC.write_cache(out, rows, family="train", libraries={}, drops={}, bindings={})
    return out


def _assert_shard_equal(cache, ref, chrom="Chr1"):
    for acc in ("track_rows", "chain_rows"):
        (sn, i_n), (sr, i_r) = getattr(cache, acc)("Athaliana", chrom, -10, 10**9), \
            getattr(ref, acc)("Athaliana", chrom, -10, 10**9)
        assert i_n.dtype == i_r.dtype and i_n.tolist() == i_r.tolist()
        for c in OC.COLUMNS:
            _assert_same_values(sn.cols[c][i_n], sr.cols[c][i_r])
    assert cache.excluded_junctions("Athaliana", chrom) == ref.excluded_junctions("Athaliana", chrom)
    assert cache.excluded_intervals("Athaliana", chrom) == ref.excluded_intervals("Athaliana", chrom)


def test_bool_column_with_nulls_matches_the_object_reader(tmp_path):
    rows = synthetic_rows(np.random.default_rng(11), 400, "Chr1")
    for r in rows[::3]:
        r["train_tiles_excluded"] = None
    out = _one_shard_cache(tmp_path, rows)
    cache, ref = OC.ObsCache(out), ReferenceCache(out)
    assert cache.shard("Athaliana", "Chr1").cols["train_tiles_excluded"].dtype == object
    _assert_shard_equal(cache, ref)
    _assert_shard_equal(OC.ObsCache(out), ref)                               # reopened from the derived file


def test_zero_row_shard(tmp_path):
    import json
    out = _one_shard_cache(tmp_path, synthetic_rows(np.random.default_rng(1), 5, "Chr1"))
    pq.write_table(OC.SCHEMA.empty_table(), out / "shard.Athaliana.Chr1.parquet")
    m = json.loads((out / "manifest.json").read_text())
    m["shards"][0].update(rows=0, sha256=OC.sha256_file(out / "shard.Athaliana.Chr1.parquet"))
    (out / "manifest.json").write_text(OC._canon(m) + "\n")
    (out / "manifest.json.sha256").write_text(f"{OC.sha256_file(out / 'manifest.json')}  manifest.json\n")
    _assert_shard_equal(OC.ObsCache(out), ReferenceCache(out))
    _assert_shard_equal(OC.ObsCache(out), ReferenceCache(out))               # reopened from the derived file
    assert OC.prebuild_mmap(out)[0]["rows"] == 0


@pytest.mark.parametrize("damage", ["truncate", "garbage", "empty"])
def test_damaged_derived_file_is_a_cache_error(synth, tmp_path, damage):
    mm = tmp_path / "m"
    OC.ObsCache(synth, mmap_dir=mm).shard("Athaliana", "Chr1")
    d = OC.ObsCache(synth, mmap_dir=mm).derived_path("Athaliana", "Chr1")
    data = d.read_bytes()
    d.write_bytes({"truncate": data[: len(data) // 2], "garbage": os.urandom(len(data)), "empty": b""}[damage])
    with pytest.raises(OC.CacheError, match="delete it and rerun the prebuild"):
        OC.ObsCache(synth, mmap_dir=mm).shard("Athaliana", "Chr1")


def test_require_prebuilt_never_builds(synth, tmp_path, monkeypatch):
    mm = tmp_path / "m"
    monkeypatch.setenv(OC.REQUIRE_PREBUILT_ENV, "1")
    cache = OC.ObsCache(synth, mmap_dir=mm)
    assert cache.require_prebuilt
    with pytest.raises(OC.CacheError, match="sidecar is missing"):
        cache.shard("Athaliana", "Chr1")
    assert not mm.exists()                                                  # no build, no lock file
    OC.ObsCache(synth, mmap_dir=mm, require_prebuilt=False).shard("Athaliana", "Chr1")   # build-on-miss, no sidecar
    with pytest.raises(OC.CacheError, match="sidecar is missing"):
        OC.ObsCache(synth, mmap_dir=mm).shard("Athaliana", "Chr1")
    OC.prebuild_mmap(synth, mmap_dir=mm)
    ref = ReferenceCache(synth)
    for chrom in CHROMS:
        _assert_shard_equal(OC.ObsCache(synth, mmap_dir=mm), ref, chrom)
    monkeypatch.setenv(OC.REQUIRE_PREBUILT_ENV, "yes")
    with pytest.raises(OC.CacheError, match="expected 0 or 1"):
        OC.ObsCache(synth)


def test_unwritable_mmap_dir_is_a_cache_error(synth, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        for mm in (ro, ro / "sub"):
            with pytest.raises(OC.CacheError, match="cannot"):
                OC.ObsCache(synth, mmap_dir=mm).shard("Athaliana", "Chr1")
            with pytest.raises(OC.CacheError, match="cannot"):
                OC.prebuild_mmap(synth, mmap_dir=mm)
    finally:
        ro.chmod(0o700)


def test_prebuild_sidecar_rehash_and_tmp_sweep(synth, tmp_path, monkeypatch):
    mm = tmp_path / "m"
    OC.prebuild_mmap(synth, mmap_dir=mm)
    d = OC.ObsCache(synth, mmap_dir=mm).derived_path("Athaliana", "Chr2")
    assert OC._sidecar(d).read_text().split()[0] == OC.sha256_file(d)
    old_tmp, new_tmp = mm / f".{d.name}.1.deadbeef.tmp", mm / f".{d.name}.2.cafef00d.tmp"
    old_side_tmp, unrelated = mm / f".{d.name}.sha256.3.0badf00d.tmp", mm / ".foo.tmp"
    for q in (old_tmp, new_tmp, old_side_tmp, unrelated):
        q.write_bytes(b"x")
    for q in (old_tmp, old_side_tmp, unrelated):
        os.utime(q, (1, 1))
    calls = _hash_calls(monkeypatch)
    recs = OC.prebuild_mmap(synth, mmap_dir=mm)
    assert not old_tmp.exists() and not old_side_tmp.exists() and new_tmp.exists() and unrelated.exists()
    assert [r["built"] for r in recs] == [False] * len(CHROMS)
    parquet = [c for c in calls if c.endswith(".parquet")]
    assert parquet == [f"shard.Athaliana.{c}.parquet" for c in CHROMS]       # each parquet hashed once
    assert [c for c in calls if c.endswith(".arrow")] == [OC.ObsCache(synth, mmap_dir=mm).derived_path(
        "Athaliana", c).name for c in CHROMS]                                # each derived file re-hashed once
    data = bytearray(d.read_bytes())
    data[len(data) // 3] ^= 1                                                # content damage the footer check misses
    d.write_bytes(bytes(data))
    with pytest.raises(OC.CacheError, match="sha256 != "):
        OC.prebuild_mmap(synth, mmap_dir=mm)
    OC._sidecar(d).unlink()                                                  # no sidecar: rebuilt from the parquet
    assert [r["built"] for r in OC.prebuild_mmap(synth, mmap_dir=mm)] == [c == "Chr2" for c in CHROMS]
    _assert_shard_equal(OC.ObsCache(synth, mmap_dir=mm, require_prebuilt=True), ReferenceCache(synth), "Chr2")


def test_sweep_tolerates_a_tmp_file_vanishing(tmp_path, monkeypatch):
    gone = tmp_path / f".{'a' * 64}.arrow.7.deadbeef.tmp"
    kept = tmp_path / f".{'b' * 64}.arrow.8.deadbeef.tmp"
    for q in (gone, kept):
        q.write_bytes(b"x")
        os.utime(q, (1, 1))
    real_stat = type(gone).stat

    def stat(self, *a, **kw):
        if self.name == gone.name:                                         # removed by someone else meanwhile
            self.unlink(missing_ok=True)
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(type(gone), "stat", stat)
    assert OC._sweep_tmp(tmp_path) == [kept.name]


def test_unreadable_derived_file_is_a_cache_error(synth, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    mm = tmp_path / "m"
    OC.prebuild_mmap(synth, mmap_dir=mm)
    d = OC.ObsCache(synth, mmap_dir=mm).derived_path("Athaliana", "Chr1")
    d.chmod(0o000)
    try:
        with pytest.raises(OC.CacheError, match="delete it and rerun the prebuild"):
            OC.ObsCache(synth, mmap_dir=mm, require_prebuilt=True).shard("Athaliana", "Chr1")
    finally:
        d.chmod(0o644)
    d.unlink()
    d.mkdir()                                                              # IsADirectoryError
    with pytest.raises(OC.CacheError, match="delete it and rerun the prebuild"):
        OC.ObsCache(synth, mmap_dir=mm).shard("Athaliana", "Chr1")
