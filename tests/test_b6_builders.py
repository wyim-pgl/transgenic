"""CLI builders: observation cache, dose-response archive, loader throughput bench (plan §7.5, §7.8, §7.9)."""
import json
import sys
from pathlib import Path

import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_context as CX
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import b6_track_spec as TS

sys.path.insert(0, str(FX.ROOT / "revision" / "scripts" / "b6"))
import b6_build_obs_cache as BOC  # noqa: E402
import b6_build_tracks as BTR  # noqa: E402
import bench_throughput as BT  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"))


def zero_args(w):
    return sum((["--zero", f"{pop}:{FX.SPECIES}:{t}:{q}"] for pop, per in w.zero.items()
                for t, q in [per[FX.SPECIES]]), [])


def test_obs_cache_cli_binds_inputs_and_matches_library(world, tmp_path):
    out = tmp_path / "cache"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(world.freeze))
    argv = ["--evidence", str(world.c0), "--family", "train", "--roles", "roles.tsv",
            "--reference", f"{FX.SPECIES}={world.fasta}", "--corpus", str(world.corpus), "--freeze", str(freeze),
            "--out", str(out)] + zero_args(world)
    assert BOC.main(argv, roles_loader=lambda p: world.roles) == 0
    c = OC.ObsCache(out)
    b = c.manifest["bindings"]
    assert b["evidence_sha256"] == FX._sha(world.c0) and b["corpus_sha256"] == FX._sha(world.corpus)
    assert b["reference_sha256"] == {FX.SPECIES: FX._sha(world.fasta)} and b["freeze_sha256"] == FX._sha(freeze)
    assert set(b["code_sha256"]) >= {"b6_obs_cache.py", "b6_track_spec.py"}
    ref = OC.ObsCache(world.cache_dir)
    assert [s["rows"] for s in c.manifest["shards"]] == [s["rows"] for s in ref.manifest["shards"]]
    with pytest.raises(OC.CacheError, match="immutable"):
        BOC.main(argv, roles_loader=lambda p: world.roles)


def test_obs_cache_cli_publishes_nothing_on_a_wrong_reference(world, tmp_path):
    other = tmp_path / "other.fa"
    other.write_text(Path(world.fasta).read_text().replace("A", "C", 1))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(world.freeze))
    out = tmp_path / "cache"
    with pytest.raises(SystemExit, match="inputs.reference"):
        BOC.main(["--evidence", str(world.c0), "--family", "train", "--roles", "r", "--reference",
                  f"{FX.SPECIES}={other}", "--corpus", str(world.corpus), "--freeze", str(freeze),
                  "--out", str(out)] + zero_args(world), roles_loader=lambda p: world.roles)
    assert not out.exists()
    bad = FX.write_zero_lists(tmp_path, upstream={"est_c0": "7" * 64})
    with pytest.raises(SystemExit, match="upstream"):
        BOC.main(["--evidence", str(world.c0), "--family", "train", "--roles", "r", "--reference",
                  f"{FX.SPECIES}={world.fasta}", "--corpus", str(world.corpus), "--freeze", str(freeze),
                  "--out", str(out)] + sum((["--zero", f"{p}:{FX.SPECIES}:{v[FX.SPECIES][0]}:{v[FX.SPECIES][1]}"]
                                            for p, v in bad.items()), []), roles_loader=lambda p: world.roles)
    assert not out.exists()


def test_obs_cache_cli_refuses_family_mismatch_and_missing_zero(tmp_path):
    w = FX.make_world(tmp_path, c0_kw={"provenance_extra": {s: {"policy": "eval_only_no_leakage_model"}
                                                            for s in ("est", "ont", "pacbio", "protein")}})
    base = ["--evidence", str(w.c0), "--roles", "r", "--reference", f"{FX.SPECIES}={w.fasta}"]
    fz = tmp_path / "fz.json"
    fz.write_text(json.dumps(w.freeze))
    with pytest.raises(SystemExit, match="not accepted"):
        BOC.main(base + ["--family", "train", "--corpus", str(w.corpus), "--freeze", str(fz),
                         "--out", str(tmp_path / "c1")] + zero_args(w), roles_loader=lambda p: w.roles)
    with pytest.raises(SystemExit, match="no --corpus"):
        BOC.main(base + ["--family", "eval", "--corpus", str(w.corpus), "--out", str(tmp_path / "c2")],
                 roles_loader=lambda p: w.roles)
    (tmp_path / "w2").mkdir()
    w2 = FX.make_world(tmp_path / "w2")
    fz2 = tmp_path / "fz2.json"
    fz2.write_text(json.dumps(w2.freeze))
    with pytest.raises(SystemExit, match="zero list"):
        BOC.main(["--evidence", str(w2.c0), "--roles", "r", "--reference", f"{FX.SPECIES}={w2.fasta}",
                  "--family", "train", "--corpus", str(w2.corpus), "--freeze", str(fz2), "--out", str(tmp_path / "c3"),
                  "--zero", "train_track:{}:{}:{}".format(FX.SPECIES, *w2.zero["train_track"][FX.SPECIES])],
                 roles_loader=lambda p: w2.roles)


def test_dose_response_archive_nested_deterministic(world, tmp_path):
    wins = [CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, rc, "gH", "centred30720-v1") for rc in (False, True)]
    CX.WindowManifest.write(tmp_path / "win.tsv", wins)
    fz = world.freeze_path                    # the freeze file the world's cache is bound to (verify_cache checks it)
    argv = ["--cache", str(world.cache_dir), "--evidence", str(world.c0), "--freeze", str(fz), "--corpus",
            str(world.corpus), "--reference", str(world.fasta), "--panel", str(world.panel),
            "--panel-sha", world.panel_sha, "--panel-id", "pilot", "--windows", str(tmp_path / "win.tsv"),
            "--generator", "centred30720-v1", "--roles", "r"]
    assert BTR.main(argv + ["--out", str(tmp_path / "a.jsonl"), "--arrays", str(tmp_path / "arr")],
                    roles_loader=lambda p: world.roles) == 0
    assert BTR.main(argv + ["--out", str(tmp_path / "b.jsonl")], roles_loader=lambda p: world.roles) == 0
    a = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (tmp_path / "b.jsonl").read_text().splitlines()]
    assert len(a) == 1 + 2 * (5 * 5 + 1)
    strip = lambda r: {k: v for k, v in r.items() if k != "arrays"}                      # noqa: E731
    assert [strip(r) for r in a[1:]] == [strip(r) for r in b[1:]]
    ev = [r for r in a[1:] if r["mode"] == "evidence" and not r["is_rc"]]
    for d in TS.DOSE_DRAWS:
        per = {r["fraction"]: set().union(*map(set, r["retained"].values())) for r in ev if r["draw"] == d}
        assert per[0.0] == set() and per[0.01] <= per[0.1] <= per[0.5] <= per[1.0] == {"eh", "oh"}
    assert Path(str(tmp_path / "a.jsonl") + ".sha256").is_file() and len(list((tmp_path / "arr").glob("*.npz"))) == 50
    assert a[0]["identity"]["family"] == "training" and a[0]["identity"]["reference_sha256"]
    other = tmp_path / "other.fa"
    other.write_text(Path(world.fasta).read_text().replace("A", "C", 1))
    bad = [x if x != str(world.fasta) else str(other) for x in argv]
    with pytest.raises(Exception, match="reference"):
        BTR.main(bad + ["--out", str(tmp_path / "c.jsonl")], roles_loader=lambda p: world.roles)
    assert not (tmp_path / "c.jsonl").exists()


def test_throughput_bench_smoke(world, tmp_path):
    rec = BT.run(world.corpus, world.cache_dir, tier=FX.TILE, workers=2, samples=8)
    assert rec["samples"] == 8 and rec["loader_samples_per_s"] > 0
    out = tmp_path / "bench.json"
    assert BT.main(["--corpus", str(world.corpus), "--cache", str(world.cache_dir), "--tier", str(FX.TILE),
                    "--workers", "1", "--samples", "4", "--model-samples-per-s", "0.001", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["pass"] is True
