"""b6_bindings: typed required keys per verified family (plan §8), Writer-A freeze shape, in-verifier §8.3
revalidation, coverage-complete leakage/A49.2 gates (review round 1: H2, M3, M4)."""
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_bindings as BD
from transgenic.evidence import b6_obs_cache as OC

ALL = ("est", "ont", "pacbio", "protein")


def verify(w, **kw):
    args = dict(accept={"training"}, evidence=w.c0, roles=w.roles, cache=OC.ObsCache(w.cache_dir), freeze=w.freeze,
                corpus=w.corpus, references={FX.SPECIES: FX.Ref(w.fasta)}, aug_seed=20261037, draws=None)
    args.update(kw)
    if isinstance(args["freeze"], dict):
        args["freeze"] = FX.freeze_file(w.tmp, args["freeze"])
    return BD.verify_inputs(**args)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"))


def test_training_world_with_writer_a_freeze_passes(world):
    assert isinstance(world.freeze["interval_manifest_sha256"], dict)          # b6_corpus_repair record shape
    ident = verify(world)
    assert ident["family"] == "training" and ident["evlabel"] == "absent"
    assert ident["interval_manifest_sha256"] == {FX.SPECIES: FX.IMAN_SHA} and ident["revalidated"] == {}
    assert ident["corpus_sha256"] == world.freeze["out_sha256"] and ident["reference_sha256"]


def test_freeze_manifest_must_be_species_indexed_and_complete(world):
    with pytest.raises(BD.BindingError, match="species-indexed"):
        verify(world, freeze=dict(world.freeze, interval_manifest_sha256=FX.IMAN_SHA))
    with pytest.raises(BD.BindingError, match="lacks species"):
        verify(world, freeze=dict(world.freeze, interval_manifest_sha256={"Osativa": FX.IMAN_SHA}))
    with pytest.raises(BD.BindingError, match="sha != mapping"):
        verify(world, freeze=dict(world.freeze, interval_manifest_sha256={FX.SPECIES: "2" * 64}))
    locs = {FX.SPECIES: dict(world.freeze["interval_manifests"][FX.SPECIES], sha256="2" * 64)}
    with pytest.raises(BD.BindingError, match=r"!= freeze\[Athaliana\]"):
        verify(world, freeze=dict(world.freeze, interval_manifest_sha256={FX.SPECIES: "2" * 64},
                                  interval_manifests=locs))


@pytest.mark.parametrize("key", ["inputs", "leakage_intervals_sha256", "completeness_codes"])
def test_each_required_training_key_fails_when_removed(tmp_path, key):
    w = FX.make_world(tmp_path, c0_kw={"drop_provenance_keys": (key,)})
    with pytest.raises(BD.BindingError, match=key):
        verify(w)


def test_freeze_keys_required(world):
    for k in ("base_sha256", "out_sha256", "interval_manifest_sha256"):
        fr = {x: v for x, v in world.freeze.items() if x != k}
        with pytest.raises(BD.BindingError, match=k):
            verify(world, freeze=fr)
    with pytest.raises(BD.BindingError, match="out_sha256"):
        verify(world, freeze=dict(world.freeze, out_sha256="0" * 64))


def test_reference_must_equal_c0_inputs_reference(world, tmp_path):
    other = tmp_path / "other.fa"
    other.write_text(Path(world.fasta).read_text().replace("A", "C", 1))
    with pytest.raises(BD.BindingError, match="inputs.reference"):
        verify(world, references={FX.SPECIES: FX.Ref(other)})
    with pytest.raises(BD.BindingError, match="outside the evidence coverage"):
        verify(world, references={FX.SPECIES: FX.Ref(world.fasta), "Osativa": FX.Ref(world.fasta)})


# ------------------------------------------------------------------------------ M3: legacy-C0 revalidation
def _iv(chrom, s, e, cause, scope, gene, tile_rn=None):
    return SimpleNamespace(chrom=chrom, start0=s, end0=e, cause=cause, scope=scope, gene_id=gene, tile_rn=tile_rn)


def _mrow(**k):
    base = dict(manifest_version="b6-intervals-v2", species=FX.SPECIES, chrom=FX.CHROM, component_id="gH",
                members=["gH"], start0=16000, end0=16250, cause="heldout", scope="all", flank=60,
                flank_source="historical_max", c0_exclusion=True)
    base.update(k)
    return base


def legacy_world(tmp_path, recomputed, rows):
    legacy = BD.legacy_intervals_sha256(recomputed)
    tmp_path.mkdir(parents=True, exist_ok=True)
    mpath = tmp_path / "legacy_intervals.Athaliana.jsonl"
    msha = FX.write_interval_manifest(mpath, rows)
    w = FX.make_world(tmp_path, c0_kw={"drop_provenance_keys": ("interval_manifest_sha256",),
                                       "provenance_extra": {s: {"leakage_intervals_sha256": legacy} for s in ALL}})
    w.freeze = FX.writer_a_freeze(w.corpus, {FX.SPECIES: {"path": str(mpath), "sha256": msha}})
    # verify(w) consumes this freeze as FX.freeze_file(w.tmp, w.freeze); the cache must be bound to those bytes.
    w.cache_dir = _rebound_cache(w, tmp_path, "cache_legacy",
                                 freeze_sha256=FX._sha(FX.freeze_file(tmp_path, w.freeze))).path
    return w


RECOMPUTED = [_iv(FX.CHROM, 16000, 16250, "heldout", "all", "gH"), _iv(FX.CHROM, 580, 620, "edge_partial", "all",
                                                                        "gP", 1)]
ROWS = [_mrow(), _mrow(component_id="gW", members=["gW"], start0=5000, end0=5100, cause="a30_only",
                       c0_exclusion=False)]


def test_legacy_c0_revalidated_inside_the_verifier(tmp_path):
    w = legacy_world(tmp_path, RECOMPUTED, ROWS)
    calls = []

    def rerun(p, sp):
        calls.append((sp, p["leakage_intervals_sha256"]))
        return RECOMPUTED
    ident = verify(w, leakage_rerun=rerun)
    assert len(ident["revalidated"]) == 4 and len(calls) == 4
    assert all(v["interval_manifest_sha256"] == w.freeze["interval_manifest_sha256"][FX.SPECIES]
               for v in ident["revalidated"].values())


def test_legacy_revalidation_fails_on_rerun_or_projection_mismatch(tmp_path):
    w = legacy_world(tmp_path / "a", RECOMPUTED, ROWS)
    with pytest.raises(BD.BindingError, match=r"\(a\)"):
        verify(w, leakage_rerun=lambda p, sp: RECOMPUTED[:1])                    # another C0's intervals
    w2 = legacy_world(tmp_path / "b", RECOMPUTED, [_mrow(start0=16001)])
    with pytest.raises(BD.BindingError, match=r"\(b\)"):
        verify(w2, leakage_rerun=lambda p, sp: RECOMPUTED)
    w3 = legacy_world(tmp_path / "c", RECOMPUTED, ROWS)
    Path(w3.freeze["interval_manifests"][FX.SPECIES]["path"]).write_text("")    # tampered manifest file
    with pytest.raises(BD.BindingError):
        verify(w3, leakage_rerun=lambda p, sp: RECOMPUTED)


def test_default_rerun_requires_recorded_inputs_unchanged(tmp_path):
    w = legacy_world(tmp_path, RECOMPUTED, ROWS)
    with pytest.raises(BD.BindingError, match="recorded input corpus"):
        verify(w)                                           # the recorded a40.db does not exist: fail closed


# ------------------------------------------------------------------------------ M4: coverage-complete gates
def test_empty_c0_qc_fails(tmp_path):
    w = FX.make_world(tmp_path, c0_kw={"qc_rows": False})
    with pytest.raises(BD.BindingError, match="no c0_qc record"):
        verify(w)


@pytest.mark.parametrize("ex, msg", [
    ({FX.SPECIES: {"by_source": {}}}, "sources"),
    ({FX.SPECIES: {"by_source": {"est": {"below_cap": 98, "molecules": 100}, "ont": {"below_cap": 5, "molecules": 5},
                                 "pacbio": {"below_cap": 5, "molecules": 5}}}}, "A49.2"),
    ({FX.SPECIES: {"by_source": {s: {"below_cap": 5.0, "molecules": 5} for s in ("est", "ont", "pacbio")}}},
     "integers"),
    ({}, "exhaustiveness"),
])
def test_exhaustiveness_requires_every_audited_source_with_integer_counters(tmp_path, ex, msg):
    w = FX.make_world(tmp_path, c0_kw={"exhaustiveness": ex})
    with pytest.raises(BD.BindingError, match=msg):
        verify(w)


def test_eval_only_evidence_rejected_by_training_even_with_corpus_key(tmp_path):
    w = FX.make_world(tmp_path, c0_kw={"provenance_extra": {s: {"policy": BD.EVAL_ONLY_POLICY} for s in ALL}})
    with pytest.raises(BD.BindingError, match="not accepted"):
        verify(w)


def test_mixed_policy_fails(tmp_path):
    w = FX.make_world(tmp_path, c0_kw={"provenance_extra": {"est": {"policy": BD.EVAL_ONLY_POLICY}}})
    with pytest.raises(BD.BindingError, match="mixed"):
        verify(w, accept={"training", "eval_only"})


def test_wal_and_evlabel(world):
    with pytest.raises(BD.BindingError, match="evlabel"):
        verify(world, evlabel="evlabel_b6_v1.db")
    wal = Path(str(world.c0) + ".wal")
    wal.write_text("x")
    try:
        with pytest.raises(BD.BindingError, match="unsealed"):
            verify(world)
    finally:
        wal.unlink()


def test_zero_list_upstream_must_be_a_merged_input(tmp_path):
    w = FX.make_world(tmp_path)
    zl = FX.write_zero_lists(tmp_path, upstream={"protein_c0": "7" * 64})
    with pytest.raises(BD.BindingError, match="upstream"):
        verify(w, zero_lists={p: {sp: tuple(map(Path, v)) for sp, v in per.items()} for p, per in zl.items()})
    bad = FX.write_zero_lists(tmp_path)
    (Path(bad["valid_track"][FX.SPECIES][1])).write_text(json.dumps({"population": "train_track",
                                                                     "upstream": {"e": FX.FAKE_C0_SHA["est"]}}))
    with pytest.raises(BD.BindingError, match="population"):
        verify(w, zero_lists=bad)


def test_interval_manifest_hash_and_revalidation_steps():
    rows = ROWS
    h = BD.interval_manifest_sha256(rows)
    assert h == BD.interval_manifest_sha256(rows[::-1])
    rec = BD.legacy_intervals_sha256(RECOMPUTED)
    assert BD.revalidate_interval_manifest(rec, RECOMPUTED, rows, BD.legacy_intervals_sha256) == h
    hit = [rows[0], _mrow(component_id="gW", members=["gW"], cause="a30_only", c0_exclusion=False)]
    with pytest.raises(BD.BindingError, match=r"\(c\)"):
        BD.revalidate_interval_manifest(rec, RECOMPUTED, hit, BD.legacy_intervals_sha256)


def test_legacy_hash_equals_b6_build_c0():
    import b6_build_c0 as B
    assert BD.legacy_intervals_sha256(RECOMPUTED) == B.intervals_sha256(RECOMPUTED)


# ------------------------------------------------------------------------------ rule 9 provenance map
TRAINING_KEYS = {"family", "evidence_sha256", "roles_sha256", "coverage", "spec", "aug_seed", "draws", "evlabel",
                 "reference_sha256", "exhaustiveness", "corpus_sha256", "base_sha256", "interval_manifest_sha256",
                 "revalidated", "zero_lists", "provenance", "cache_manifest_sha256", "record_sha256"}


def test_record_carries_rehashed_provenance_and_record_sha(world):
    ident = verify(world)
    assert set(ident) == TRAINING_KEYS
    sp = FX.SPECIES
    want = ({"evidence.sha256", "freeze", f"interval_manifest:{sp}", f"interval_manifest:{sp}.sha256",
             "cache_manifest", "cache_manifest.sha256"}
            | {f"c0:{sp}/{s}.PROVENANCE.txt" for s in ALL}
            | {f"{k}:{p}/{sp}" for k in ("zero_list", "zero_qc") for p in ("train_track", "valid_track", "eval_input")})
    assert set(ident["provenance"]) == want and list(ident["provenance"]) == sorted(want)
    assert ident["provenance"]["freeze"] == FX._sha(world.freeze_path)
    assert ident["provenance"][f"interval_manifest:{sp}"] == FX.IMAN_SHA
    assert ident["provenance"][f"c0:{sp}/est.PROVENANCE.txt"] == FX._sha(world.tmp / f"{sp}.est.c0.db.PROVENANCE.txt")
    assert ident["record_sha256"] == BD.canonical_record_sha256(ident)
    assert verify(world)["record_sha256"] == ident["record_sha256"]          # deterministic


def test_missing_or_mismatched_provenance_files_fail(tmp_path):
    w = FX.make_world(tmp_path)
    pv = w.tmp / f"{FX.SPECIES}.ont.c0.db.PROVENANCE.txt"
    text = pv.read_text()
    pv.write_text(text.replace(FX.FAKE_C0_SHA["ont"], "9" * 64))
    with pytest.raises(BD.BindingError, match="output_sha256"):
        verify(w)
    pv.unlink()
    with pytest.raises(BD.BindingError, match="PROVENANCE file"):
        verify(w)
    pv.write_text(text)
    side = Path(str(w.freeze["interval_manifests"][FX.SPECIES]["path"]) + ".sha256")
    side.unlink()
    with pytest.raises(BD.BindingError, match="missing"):
        verify(w)


def test_freeze_must_be_a_file(world):
    with pytest.raises(BD.BindingError, match="path of the frozen"):
        BD.verify_inputs(accept={"training"}, evidence=world.c0, roles=world.roles,
                         cache=OC.ObsCache(world.cache_dir), freeze=object(), corpus=world.corpus)


def _rebound_cache(world, tmp_path, name, **over):
    """The world's training cache republished with some bindings replaced (value) or removed (None)."""
    b = dict(OC.ObsCache(world.cache_dir).manifest["bindings"])
    for k, v in over.items():
        if v is None:
            b.pop(k, None)
        else:
            b[k] = v
    out = tmp_path / name
    OC.write_cache(out, world.rows, family="train", libraries=world.libraries, drops=world.drops, bindings=b,
                   zero_lists=world.zero)
    return OC.ObsCache(out)


def test_training_cache_binds_the_verified_corpus_and_freeze(world, tmp_path):
    """Integration review 1 MEDIUM: verify_cache enforces the corpus and freeze bindings the cache records."""
    b = OC.ObsCache(world.cache_dir).manifest["bindings"]
    assert b["corpus_sha256"] == FX._sha(world.corpus) and b["freeze_sha256"] == FX._sha(world.freeze_path)
    ident = verify(world, cache=_rebound_cache(world, tmp_path, "same"))        # republished unchanged: passes
    assert ident["provenance"]["freeze"] == b["freeze_sha256"]


@pytest.mark.parametrize("over, pattern", [
    ({"corpus_sha256": None}, "binds no corpus_sha256"),
    ({"corpus_sha256": "0" * 64}, "different B6 corpus"),
    ({"corpus_sha256": "not-a-sha"}, "binds no corpus_sha256"),
    ({"freeze_sha256": None}, "binds no freeze_sha256"),
    ({"freeze_sha256": "0" * 64}, "different corpus freeze"),
])
def test_training_cache_with_a_missing_or_mismatched_corpus_or_freeze_binding_fails(world, tmp_path, over, pattern):
    with pytest.raises(BD.BindingError, match=pattern):
        verify(world, cache=_rebound_cache(world, tmp_path, "bad", **over))


def test_stale_cache_refused_with_another_valid_freeze_or_corpus(world, tmp_path):
    """The reviewer's case: upstream alone verifies with another freeze (or corpus + freeze) on the same C0, but the
    cache was built against the original pair."""
    other_freeze = tmp_path / "freeze_reformatted.json"          # same record, different bytes: a different freeze
    other_freeze.write_text(json.dumps(world.freeze, indent=2, sort_keys=True) + "\n")
    with pytest.raises(BD.BindingError, match="different corpus freeze"):
        verify(world, freeze=other_freeze)
    corpus2 = tmp_path / "b5_full_b6_v1.db"                      # another corpus, with its own valid freeze
    corpus2.write_bytes(Path(world.corpus).read_bytes())
    con = duckdb.connect(str(corpus2))
    con.execute("UPDATE geneList SET gene_id = gene_id || '_v2' WHERE rn = 3")
    con.close()
    assert FX._sha(corpus2) != FX._sha(world.corpus)
    freeze2 = FX.writer_a_freeze(corpus2, world.freeze["interval_manifests"])
    with pytest.raises(BD.BindingError, match="different B6 corpus"):
        verify(world, corpus=corpus2, freeze=FX.freeze_file(tmp_path, freeze2, "freeze2.json"))


def test_zero_upstream_needs_both_producer_keys_each_matching_its_own_source(tmp_path):
    """Codex round 2 reproduction: {"protein_c0": <EST hash>} without est_c0 was accepted."""
    w = FX.make_world(tmp_path)
    cases = [({"protein_c0": FX.FAKE_C0_SHA["est"]}, "upstream keys"),
             ({"est_c0": FX.FAKE_C0_SHA["est"]}, "upstream keys"),
             ({"est_c0": FX.FAKE_C0_SHA["est"], "protein_c0": FX.FAKE_C0_SHA["est"]}, "protein"),
             ({"est_c0": FX.FAKE_C0_SHA["protein"], "protein_c0": FX.FAKE_C0_SHA["protein"]}, "est"),
             ({"est_c0": FX.FAKE_C0_SHA["est"], "protein_c0": FX.FAKE_C0_SHA["protein"], "x": "1" * 64},
              "upstream keys")]
    for i, (up, msg) in enumerate(cases):
        d = tmp_path / f"z{i}"
        d.mkdir()
        zl = FX.write_zero_lists(d, upstream=up)
        with pytest.raises(BD.BindingError, match=msg):
            verify(w, zero_lists=zl)
    (tmp_path / "ok").mkdir()
    ok = FX.write_zero_lists(tmp_path / "ok")               # default upstream: est_c0 + protein_c0, both correct
    assert verify(w, zero_lists=ok)["family"] == "training"


def test_protein_c0_present_makes_protein_c0_upstream_mandatory(tmp_path):
    w = FX.make_world(tmp_path)                                   # merged DB holds a protein C0 for Athaliana
    (tmp_path / "z").mkdir()
    zl = FX.write_zero_lists(tmp_path / "z", upstream={"est_c0": FX.FAKE_C0_SHA["est"]})
    with pytest.raises(BD.BindingError, match="upstream keys"):
        verify(w, zero_lists=zl)


def test_training_family_species_without_protein_c0_fails(tmp_path):
    no_protein = [m for m in FX.MOLECULES if m[1] != "protein"]
    w = FX.make_world(tmp_path, molecules=no_protein)
    with pytest.raises(BD.BindingError, match="without a merged protein C0"):
        verify(w)
