"""Relocatable interval-manifest paths in the B6 corpus freeze (author decision 14, 2026-09-26, option b).

The freeze and its interval manifests are built on one host (pronghorn, /data/gpfs/...) and consumed on another
(DeltaAI, inside a container, /work/...).  The freeze therefore records each manifest path relative to the freeze
file's own directory; consumers resolve it against the freeze's actual location.  Hash verification is unchanged:
a manifest that does not resolve, or whose bytes differ, fails closed.  Legacy absolute-path freezes still verify
where the file exists.

The merged evidence DB likewise records each C0 input path relative to its own directory (b6_merge_evidence_db);
verify_upstream resolves it against the merged DB's actual location, so the C0 bundle (DBs + .PROVENANCE.txt)
travels with it in the same relative layout."""
import json
import os
import shutil
from pathlib import Path

import pytest

import b6_fixtures as FX
import test_b6_bindings as TB
from transgenic.evidence import b6_bindings as BD
from transgenic.evidence import b6_obs_cache as OC

MANIFEST = "intervals.Athaliana.jsonl"


def _verify_at(root, w, **kw):
    """verify_inputs on the bundle copied to ``root`` (same relative layout as the world ``w`` built in w.tmp)."""
    rel = lambda p: Path(root) / Path(p).relative_to(w.tmp)
    args = dict(accept={"training"}, evidence=rel(w.c0), roles=w.roles, cache=OC.ObsCache(rel(w.cache_dir)),
                freeze=rel(w.freeze_path), corpus=rel(w.corpus), references={FX.SPECIES: FX.Ref(rel(w.fasta))},
                aug_seed=20261037, draws=None)
    args.update(kw)
    return BD.verify_inputs(**args)


def _move_bundle(w, dst):
    """Copy the bundle to ``dst`` and remove the manifest, sidecar and freeze from the build location, so a pass at
    ``dst`` cannot come from the original files."""
    shutil.copytree(w.tmp, dst)
    for name in (MANIFEST, MANIFEST + ".sha256", w.freeze_path.name):
        (w.tmp / name).unlink()
    return dst


def _world(tmp_path, **kw):
    (tmp_path / "A").mkdir()
    return FX.make_world(tmp_path / "A", **kw)


@pytest.fixture
def relocatable(tmp_path):
    return _world(tmp_path, relative_manifest=True)


def test_freeze_records_the_manifest_path_relative_to_its_directory(relocatable):
    rec = json.loads(relocatable.freeze_path.read_text())["interval_manifests"][FX.SPECIES]
    assert rec["path"] == MANIFEST and not os.path.isabs(rec["path"])


def test_bundle_moved_to_another_directory_verifies_there(relocatable, tmp_path):
    dst = _move_bundle(relocatable, tmp_path / "B")
    ident = _verify_at(dst, relocatable)
    assert ident["family"] == "training"
    assert ident["interval_manifest_sha256"] == {FX.SPECIES: FX.IMAN_SHA}
    assert ident["provenance"][f"interval_manifest:{FX.SPECIES}"] == FX.IMAN_SHA


def test_tampered_manifest_in_the_moved_bundle_fails_closed(relocatable, tmp_path):
    dst = _move_bundle(relocatable, tmp_path / "B")
    m = dst / MANIFEST
    m.write_text(m.read_text() + m.read_text().splitlines(keepends=True)[0])
    with pytest.raises(BD.BindingError, match=r"sha256 != freeze\[Athaliana\]"):
        _verify_at(dst, relocatable)


def test_missing_manifest_in_the_moved_bundle_names_the_relative_path(relocatable, tmp_path):
    dst = _move_bundle(relocatable, tmp_path / "B")
    (dst / MANIFEST).unlink()
    with pytest.raises(BD.BindingError, match=rf"'{MANIFEST}'.*{dst.name}.*missing"):
        _verify_at(dst, relocatable)


def test_missing_sidecar_in_the_moved_bundle_fails_closed(relocatable, tmp_path):
    dst = _move_bundle(relocatable, tmp_path / "B")
    (dst / (MANIFEST + ".sha256")).unlink()
    with pytest.raises(BD.BindingError, match="missing"):
        _verify_at(dst, relocatable)


def test_legacy_absolute_path_freeze_still_verifies_where_the_file_exists(tmp_path):
    w = _world(tmp_path)
    assert os.path.isabs(json.loads(w.freeze_path.read_text())["interval_manifests"][FX.SPECIES]["path"])
    assert _verify_at(w.tmp, w)["interval_manifest_sha256"] == {FX.SPECIES: FX.IMAN_SHA}


def test_legacy_absolute_path_freeze_fails_closed_when_the_file_is_gone(tmp_path):
    w = _world(tmp_path)
    (w.tmp / MANIFEST).unlink()
    with pytest.raises(BD.BindingError, match=rf"{MANIFEST}.*missing"):
        _verify_at(w.tmp, w)


def test_legacy_c0_revalidation_reads_a_relative_manifest_after_the_move(tmp_path):
    a = tmp_path / "A"
    w = TB.legacy_world(a, TB.RECOMPUTED, TB.ROWS)
    msha = w.freeze["interval_manifest_sha256"][FX.SPECIES]
    w.freeze = FX.writer_a_freeze(w.corpus, {FX.SPECIES: {"path": "legacy_intervals.Athaliana.jsonl",
                                                         "sha256": msha}})
    w.freeze_path = FX.freeze_file(a, w.freeze, name="freeze_rel.json")
    w.cache_dir = TB._rebound_cache(w, a, "cache_rel", freeze_sha256=FX._sha(w.freeze_path)).path
    shutil.copytree(a, tmp_path / "B")
    for name in ("legacy_intervals.Athaliana.jsonl", "legacy_intervals.Athaliana.jsonl.sha256", "freeze_rel.json"):
        (a / name).unlink()
    ident = _verify_at(tmp_path / "B", w, leakage_rerun=lambda p, sp: TB.RECOMPUTED)
    assert len(ident["revalidated"]) == 4
    assert all(v["interval_manifest_sha256"] == msha for v in ident["revalidated"].values())


# ------------------------------------------------------------------------------ writer: b6_corpus_repair.apply
def test_apply_writes_manifest_paths_relative_to_the_freeze_directory(tmp_path):
    import test_b6_repair_world as RW
    from transgenic.evidence import b6_corpus_repair as CR
    from transgenic.evidence import leakage as LK

    pytest.importorskip("duckdb")
    world = RW.build_world(tmp_path / "world")
    (tmp_path / "bundle" / "intervals").mkdir(parents=True)
    (tmp_path / "bundle" / "freeze").mkdir()
    manifest = tmp_path / "bundle" / "intervals" / "Athaliana.b6-intervals-v2.jsonl"
    LK.write_interval_manifest(RW.leakage_model(world).interval_manifest, manifest)
    freeze = tmp_path / "bundle" / "freeze" / "freeze.json"
    CR.apply(world["corpus"], tmp_path / "bundle" / "b6.db", freeze, splits=world["split"], flags=world["a22"],
             build_flags=[world["a22"], world["a30"]], seed=RW.SEED, manifests=[manifest])
    rec = json.loads(freeze.read_text())["interval_manifests"][RW.SPECIES]
    assert rec["path"] == "../intervals/Athaliana.b6-intervals-v2.jsonl"
    assert (freeze.parent / rec["path"]).resolve() == manifest.resolve()
    assert rec["sha256"] == LK.sha256(manifest)


# ------------------------------------------------------------------------------ merged C0 input paths
def _c0_provenance_files(root):
    return sorted(Path(root).glob("c0/*.PROVENANCE.txt"))


@pytest.fixture
def relocatable_c0(tmp_path):
    return _world(tmp_path, relative_manifest=True, c0_kw={"relative_c0_paths": True})


def _move_c0_bundle(w, dst):
    """Copy the whole bundle and remove the build location's C0 PROVENANCE files (and the freeze manifest), so a
    pass at ``dst`` cannot read them from the original directory."""
    _move_bundle(w, dst)
    files = _c0_provenance_files(w.tmp)
    assert len(files) == 4
    for f in files:
        f.unlink()
    return dst


def test_merged_db_records_c0_paths_relative_to_its_directory(relocatable_c0):
    import duckdb

    con = duckdb.connect(str(relocatable_c0.c0), read_only=True)
    paths = [r[0] for r in con.execute("SELECT input_path FROM merge_manifest ORDER BY ALL").fetchall()]
    con.close()
    assert paths and all(p.startswith("c0/") and not os.path.isabs(p) for p in paths)


def test_merged_db_and_c0_bundle_moved_together_verify_there(relocatable_c0, tmp_path):
    dst = _move_c0_bundle(relocatable_c0, tmp_path / "B")
    ident = _verify_at(dst, relocatable_c0)
    assert ident["family"] == "training"
    c0_entries = {k for k in ident["provenance"] if k.startswith("c0:")}
    assert len(c0_entries) == 4


def test_tampered_c0_provenance_in_the_moved_bundle_fails_closed(relocatable_c0, tmp_path):
    dst = _move_c0_bundle(relocatable_c0, tmp_path / "B")
    pv = _c0_provenance_files(dst)[0]
    pv.write_text(pv.read_text().replace("output_sha256\t", "output_sha256\t0"))
    with pytest.raises(BD.BindingError, match="PROVENANCE output_sha256 != merge_manifest input sha256"):
        _verify_at(dst, relocatable_c0)


def test_missing_c0_provenance_in_the_moved_bundle_names_the_relative_path(relocatable_c0, tmp_path):
    dst = _move_c0_bundle(relocatable_c0, tmp_path / "B")
    _c0_provenance_files(dst)[0].unlink()
    with pytest.raises(BD.BindingError, match=r"input_path 'c0/Athaliana\.\w+\.c0\.db' \(relative to the merged "
                                              r"evidence directory"):
        _verify_at(dst, relocatable_c0)


def test_legacy_absolute_c0_path_fails_closed_when_the_file_is_gone(tmp_path):
    w = _world(tmp_path, relative_manifest=True)            # merge_manifest records absolute C0 paths (legacy)
    dst = _move_bundle(w, tmp_path / "B")
    for f in sorted(w.tmp.glob("*.c0.db.PROVENANCE.txt")):
        f.unlink()
    with pytest.raises(BD.BindingError, match="absolute input_path recorded by the merge"):
        _verify_at(dst, w)
