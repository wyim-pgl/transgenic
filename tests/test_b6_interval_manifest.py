"""b6-intervals-v2 interval manifest (plan v4.1 §2.5, §8.3 (b)-(c)) produced by leakage.build_leakage_model."""
import hashlib
import json
from pathlib import Path

import pytest

import test_b6_repair_world as RW
import b6_leakage_intervals as LI
from transgenic.evidence import leakage as LK


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return RW.build_world(tmp_path_factory.mktemp("manifest_world"))


@pytest.fixture(scope="module")
def model(world):
    return RW.leakage_model(world)


def _rows(model, component_id):
    return [r for r in model.interval_manifest if r["component_id"] == component_id]


def test_held_out_component_rows_carry_the_c0_interval_and_its_historical_flank(model):
    rows = _rows(model, "gHE")
    assert sorted((r["cause"], r["scope"]) for r in rows) == [("heldout", "all"), ("test_orthogroup", "all")]
    for r in rows:
        assert list(r) == list(LK.MANIFEST_FIELDS)
        assert r["members"] == ["gHE", "gM"] and r["species"] == RW.SPECIES and r["chrom"] == "Chr1"
        assert (r["start0"], r["end0"]) == (RW.HE_COMPONENT[0] - RW.HE_FLANK, RW.HE_COMPONENT[1] + RW.HE_FLANK)
        assert r["flank"] == RW.HE_FLANK and r["flank_source"] == "historical_max" and r["c0_exclusion"] is True
        assert r["manifest_version"] == "b6-intervals-v2"


def test_valid_component_has_train_tiles_scope_and_a30_only_components_are_explicit(model):
    assert [(r["cause"], r["scope"]) for r in _rows(model, "gV3")] == [("valid_orthogroup", "train_tiles")]
    for gid in ("gW", "gW2"):
        rows = _rows(model, gid)
        assert [(r["cause"], r["scope"], r["c0_exclusion"]) for r in rows] == [("a30_only", "all", False)]
    # A30-only components never become C0 exclusions; gW2 crosses a tile edge, so only its A48 edge_partial applies
    hits = model.index.causes("Chr1", 156100, 156900, "all") | model.index.causes("Chr1", 156100, 156900, "train_tiles")
    assert hits == {"edge_partial"}


def test_never_masked_component_uses_the_fallback_flank(model):
    rows = _rows(model, "gQ")
    assert rows and all(r["flank"] == 150 and r["flank_source"] == "fallback_150" for r in rows)
    assert (rows[0]["start0"], rows[0]["end0"]) == (40000 - 150, 40600 + 150)


def test_manifest_projection_verifies_against_the_legacy_genome_level_rows(model):
    assert LK.verify_interval_manifest(model.interval_manifest, model.intervals)
    bad = [dict(r) for r in model.interval_manifest]
    i = next(k for k, r in enumerate(bad) if r["component_id"] == "gHE")
    bad[i]["end0"] += 1
    with pytest.raises(LK.LeakageError, match="projection"):
        LK.verify_interval_manifest(bad, model.intervals)
    dropped = [r for r in model.interval_manifest if r["component_id"] != "gV3"]
    with pytest.raises(LK.LeakageError, match="projection"):
        LK.verify_interval_manifest(dropped, model.intervals)
    flipped = [dict(r) for r in model.interval_manifest]
    j = next(k for k, r in enumerate(flipped) if r["cause"] == "a30_only")
    flipped[j]["c0_exclusion"] = True
    with pytest.raises(LK.LeakageError):
        LK.verify_interval_manifest(flipped, model.intervals)


def test_canonical_hash_file_and_sidecar_agree_and_are_order_independent(model, tmp_path):
    rows = model.interval_manifest
    digest = LK.interval_manifest_sha256(rows)
    assert digest == LK.interval_manifest_sha256(list(reversed(rows)))
    path = tmp_path / "m.jsonl"
    assert LK.write_interval_manifest(rows, path) == digest
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    expected = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in LK.sort_manifest_rows(rows))
    assert path.read_text() == expected
    assert LK.read_interval_manifest(path, expected_sha256=digest) == LK.sort_manifest_rows(rows)
    with pytest.raises(FileExistsError):
        LK.write_interval_manifest(rows, path)


def test_reader_fails_closed_on_tampering(model, tmp_path):
    path = tmp_path / "m.jsonl"
    LK.write_interval_manifest(model.interval_manifest, path)
    lines = path.read_text().splitlines(keepends=True)
    spaced = tmp_path / "spaced.jsonl"
    spaced.write_text(lines[0].replace(",", ", ", 1) + "".join(lines[1:]))
    Path(str(spaced) + ".sha256").write_text(hashlib.sha256(spaced.read_bytes()).hexdigest() + "  spaced.jsonl\n")
    with pytest.raises(LK.LeakageError, match="canonical"):
        LK.read_interval_manifest(spaced)
    unordered = tmp_path / "unordered.jsonl"
    unordered.write_text("".join(reversed(lines)))
    Path(str(unordered) + ".sha256").write_text(hashlib.sha256(unordered.read_bytes()).hexdigest() + "  x\n")
    with pytest.raises(LK.LeakageError, match="order"):
        LK.read_interval_manifest(unordered)
    with pytest.raises(LK.LeakageError, match="expected"):
        LK.read_interval_manifest(path, expected_sha256="0" * 64)
    Path(str(path) + ".sha256").write_text("0" * 64 + "  m.jsonl\n")
    with pytest.raises(LK.LeakageError, match="sidecar"):
        LK.read_interval_manifest(path)


def test_row_validation_rejects_bad_versions_and_flags(model):
    row = dict(model.interval_manifest[0])
    for key, value, pattern in (("manifest_version", "b6-intervals-v1", "version"), ("flank_source", "fresh", "flank_source"),
                                ("c0_exclusion", "true", "boolean"), ("members", ["z", "a"], "sorted")):
        with pytest.raises(LK.LeakageError, match=pattern):
            LK.interval_manifest_sha256([dict(row, **{key: value})])
    with pytest.raises(LK.LeakageError, match="keys"):
        LK.interval_manifest_sha256([{k: v for k, v in row.items() if k != "flank"}])


def test_cli_writes_the_manifest_beside_the_tsv(world, model, tmp_path):
    out, man = tmp_path / "iv.tsv", tmp_path / "iv.jsonl"
    rc = LI.main(["--corpus", str(world["corpus"]), "--species", RW.SPECIES, "--splits", str(world["split"]),
                  "--flags", str(world["a22"]), "--build-flags", str(world["a22"]), str(world["a30"]),
                  "--seed", str(RW.SEED), "--out", str(out), "--manifest-out", str(man)])
    assert rc == 0
    assert LK.read_interval_manifest(man) == model.interval_manifest
