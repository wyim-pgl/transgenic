"""WP2 rule 5 repair (plan v4.1 §2) on the variant world (tests/test_b6_repair_world.py): geometry = the C0 rule
4(a) component interval, closure, RC twin, row-set refusal, freeze, independent validation and C0 consequences."""
import json
import shutil
from pathlib import Path

import pytest

import test_b6_repair_world as RW
import b6_build_corpus as BC
import b6_validate_corpus as VC
from transgenic.evidence import b6_corpus_repair as CR
from transgenic.evidence import leakage as LK

duckdb = pytest.importorskip("duckdb")
EDGE_WID = f"{RW.SPECIES}:Chr1:{RW.EDGE_TILE[0]}-{RW.EDGE_TILE[1]}"


def q(db, sql, params=()):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql, list(params)).fetchall()
    finally:
        con.close()


def _kw(world, manifest):
    return dict(splits=world["split"], flags=world["a22"], build_flags=[world["a22"], world["a30"]], seed=RW.SEED,
                manifests=[manifest])


def _cli(world, manifest, *extra):
    return ["--base", str(world["corpus"]), "--splits", str(world["split"]), "--flags", str(world["a22"]),
            "--build-flags", str(world["a22"]), str(world["a30"]), "--seed", str(RW.SEED),
            "--interval-manifest", str(manifest), *extra]


def _validate_args(world, manifest, corpus, freeze, *extra):
    return ["--base", str(world["corpus"]), "--corpus", str(corpus), "--freeze", str(freeze),
            "--splits", str(world["split"]), "--flags", str(world["a22"]), "--build-flags", str(world["a22"]),
            str(world["a30"]), "--seed", str(RW.SEED), "--interval-manifest", str(manifest), *extra]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return RW.build_world(tmp_path_factory.mktemp("repair_world"))


@pytest.fixture(scope="module")
def manifest(world):
    model = RW.leakage_model(world)
    path = world["tmp"] / "Athaliana.b6-intervals-v2.jsonl"
    LK.write_interval_manifest(model.interval_manifest, path)
    return path


@pytest.fixture(scope="module")
def applied(world, manifest):
    out, freeze = world["tmp"] / "b5_full_b6_v1.db", world["tmp"] / "freeze.json"
    assert BC.main(_cli(world, manifest, "--apply", "--out", str(out), "--freeze", str(freeze))) == 0
    return out, freeze, json.loads(freeze.read_text())


@pytest.fixture(scope="module")
def c0(world, tmp_path_factory):
    return RW.build_ont_c0(world, tmp_path_factory.mktemp("repair_c0"))


def _row(db, wid):
    return q(db, "SELECT rn, start, fin, sequence, gff, gsf_token_count, n_transcripts, qc_flags FROM geneList "
                 "WHERE geneModel = ?", [wid])[0]


# ------------------------------------------------------------------------------ geometry
def test_shared_cause_span_equals_the_c0_interval_clipped_to_the_tile(world, applied):
    out, _, _ = applied
    rows = q(out, "SELECT start0, end0, component_id, cause, scope, flank, flank_source FROM b6_dna_repair "
                  "WHERE wid = ? ORDER BY ALL", [EDGE_WID])
    he_end = RW.HE_COMPONENT[1] + RW.HE_FLANK
    assert (RW.EDGE_TILE[0], he_end, "gHE", "heldout", "all", RW.HE_FLANK, "historical_max") in rows
    assert (RW.EDGE_TILE[0], he_end, "gHE", "test_orthogroup", "all", RW.HE_FLANK, "historical_max") in rows
    assert [r for r in rows if r[2] == "gV3"][0][3:5] == ("valid_orthogroup", "train_tiles")
    # the left neighbour tile holds gHE's other half: component start - flank up to the tile edge
    left = q(out, "SELECT start0, end0 FROM b6_dna_repair WHERE component_id = 'gHE' AND wid = ? AND cause = 'heldout'",
             [f"{RW.SPECIES}:Chr1:33580-64300"])
    assert left == [(RW.HE_COMPONENT[0] - RW.HE_FLANK, 64300)]


def test_molecule_at_40_nt_is_masked_and_at_100_nt_is_not(world, applied):
    out, _, _ = applied
    rn, ws, we, seq, *_ = _row(out, EDGE_WID)
    ref = world["genome"]["Chr1"]
    end = RW.HE_COMPONENT[1]
    assert set(seq[end + 40 - ws:end + RW.HE_FLANK - ws]) == {"N"}           # mA starts inside the mask
    assert seq[end + RW.HE_FLANK - ws] == ref[end + RW.HE_FLANK] != "N"      # the mask stops at the historical flank
    assert seq[end + 100 - ws:end + 150 - ws] == ref[end + 100:end + 150]    # mB unmasked
    assert "N" not in seq[end + 100 - ws:end + 150 - ws]
    # the counterexample the test guards: a fresh flank draw of 140 would have masked mB while C0 admits it
    assert end + 140 > end + 100


def test_a30_only_component_is_masked_with_the_same_formula(world, applied, manifest):
    out, _, _ = applied
    rows = q(out, "SELECT start0, end0, cause, scope FROM b6_dna_repair WHERE component_id = 'gW2' AND NOT is_rc "
                  "ORDER BY ALL")
    m = [r for r in LK.read_interval_manifest(manifest) if r["component_id"] == "gW2"][0]
    assert rows == [(m["start0"], 156460, "a30_only", "all"), (156460, m["end0"], "a30_only", "all")]


def test_non_seed_edge_crosser_gL_stays_unmasked(world, applied):
    out, _, _ = applied
    for wid in (f"{RW.SPECIES}:Chr1:187180-217900", f"{RW.SPECIES}:Chr1:163886-225326"):
        assert q(out, "SELECT count(*) FROM b6_dna_repair WHERE wid = ?", [wid]) == [(0,)]
        assert _row(out, wid) == _row(world["corpus"], wid)


# ------------------------------------------------------------------------------ closure and derived columns
def test_closure_drops_member_and_flank_genes_and_regenerates_the_label(world, applied):
    out, _, _ = applied
    rn = _row(out, EDGE_WID)[0]
    assert q(out, "SELECT gene_id, cause FROM b6_label_drop WHERE rn = ? ORDER BY gene_id", [rn]) == [
        ("gM", "closure_member"), ("gN", "closure_flank")]
    assert [g for (g,) in q(out, "SELECT gene_id FROM window_genes WHERE window_id = ? ORDER BY is_rc, gene_id",
                              [EDGE_WID])] == ["gK", "gK"]
    b5 = LK.load_build_b5()
    gk = [g for g in b5.gc.parse_gff3(open(world["gff"]), species_code="Ath") if g.gene_id == "gK"]
    _, ws, we, _, gff, tokens, ntx, qc = _row(out, EDGE_WID)
    assert gff == b5.gc.window_to_gsf_v3(gk, ws) and tokens == b5.gc.count_tokens_v3(gff) and ntx == 1
    assert "b6_rule5_masked=2" in qc and "b6_label_drop=2" in qc and "leak_masked=1" in qc


def test_rc_twin_is_the_reverse_complement_of_the_repaired_forward_tile(applied):
    out, _, _ = applied
    b5 = LK.load_build_b5()
    _, ws, we, seq, gff, *_ = _row(out, EDGE_WID)
    _, _, _, rseq, rgff, rtok, *_ = _row(out, EDGE_WID + "-rc")
    assert rseq == b5.revcomp(seq) and rgff == b5.gc.reverse_complement_v3(gff, we - ws)
    assert rtok == b5.gc.count_tokens_v3(rgff)
    assert q(out, "SELECT count(*) FROM b6_dna_repair WHERE wid = ? AND is_rc", [EDGE_WID + "-rc"])[0][0] == 3


def test_row_set_order_coordinates_and_test_tiles_are_unchanged(world, applied):
    out, _, _ = applied
    ids = "SELECT rn, geneModel, chromosome, start, fin, split, is_rc FROM geneList ORDER BY rn"
    assert q(out, ids) == q(world["corpus"], ids)
    test = "SELECT * FROM geneList WHERE split = 'test' ORDER BY rn"
    assert q(out, test) == q(world["corpus"], test)
    untouched = ("SELECT * FROM geneList WHERE rn NOT IN (SELECT rn FROM b6_dna_repair) ORDER BY rn")
    base = {r[11]: r for r in q(world["corpus"], "SELECT * FROM geneList ORDER BY rn")}
    for r in q(out, untouched):
        assert r == base[r[11]]


# ------------------------------------------------------------------------------ freeze and rebuild
def test_freeze_records_the_bindings(world, manifest, applied):
    out, freeze, rec = applied
    assert rec["rule_id"] == "rule5-leak-hard-c0interval-v1"
    assert rec["base_sha256"] == LK.sha256(world["corpus"]) and rec["out_sha256"] == LK.sha256(out)
    assert rec["interval_manifest_sha256"] == {RW.SPECIES: LK.sha256(manifest)}
    assert rec["seed"] == RW.SEED and rec["events"] == {"tiles_over_mask_fraction": 0, "rc_twin_orphaned": 0}
    assert rec["counts"][RW.SPECIES]["train"]["closure_member"]["label_drops"] == 1
    assert rec["row_counts"] == CR._row_counts_path(world["corpus"])
    assert {"splits", "flags", "build_flags_0", "build_flags_1"} <= set(rec["inputs"])
    assert "src/transgenic/evidence/b6_corpus_repair.py" in rec["code_sha256"]
    assert not Path(str(out) + ".wal").exists() and not Path(str(out) + ".partial").exists()


def test_rebuild_is_content_identical(world, manifest, applied, tmp_path):
    out, _, rec = applied
    out2, freeze2 = tmp_path / "again.db", tmp_path / "again.json"
    rec2 = CR.apply(world["corpus"], out2, freeze2, **_kw(world, manifest))
    assert CR.content_sha256(out2) == CR.content_sha256(out) == rec["content_sha256"] == rec2["content_sha256"]
    assert rec2["counts"] == rec["counts"]


def test_apply_refuses_to_overwrite(world, manifest, applied):
    out, freeze, _ = applied
    assert BC.main(_cli(world, manifest, "--apply", "--out", str(out), "--freeze", str(freeze) + ".2")) == 1


# ------------------------------------------------------------------------------ row-set events
def test_apply_refuses_when_a_tile_crosses_the_mask_fraction_or_loses_its_last_label(tmp_path, capsys):
    world = RW.build_world(tmp_path / "w", refuse=True)
    man = tmp_path / "m.jsonl"
    LK.write_interval_manifest(RW.leakage_model(world).interval_manifest, man)
    report = tmp_path / "measure.json"
    assert BC.main(_cli(world, man, "--measure-only", "--report", str(report))) == 0
    events = json.loads(report.read_text())["events"]
    assert events["tiles_over_mask_fraction"] == 1 and events["rc_twin_orphaned"] == 2
    out, freeze = tmp_path / "b6.db", tmp_path / "f.json"
    assert BC.main(_cli(world, man, "--apply", "--out", str(out), "--freeze", str(freeze))) == 1
    assert "further amendment" in capsys.readouterr().err
    assert not out.exists() and not freeze.exists() and not Path(str(out) + ".partial").exists()


# ------------------------------------------------------------------------------ fail-closed inputs
def test_wrong_seed_or_foreign_manifest_fails_closed(world, manifest, tmp_path):
    with pytest.raises(CR.RepairError, match="replay"):
        CR.measure(world["corpus"], **dict(_kw(world, manifest), seed=123))
    rows = [r for r in LK.read_interval_manifest(manifest) if r["component_id"] != "gV3"]
    partial = tmp_path / "partial.jsonl"
    LK.write_interval_manifest(rows, partial)
    with pytest.raises(CR.RepairError, match="no component"):
        CR.measure(world["corpus"], **_kw(world, partial))
    assert BC.main(_cli(world, manifest, "--measure-only", "--out", str(tmp_path / "x.db"))) == 1


# ------------------------------------------------------------------------------ independent validation
def test_validator_passes_on_the_applied_corpus_with_c0_consequences(world, manifest, applied, c0, tmp_path):
    out, freeze, _ = applied
    c0_db, _ = c0
    report = tmp_path / "validate.json"
    assert VC.main(_validate_args(world, manifest, out, freeze, "--c0", str(c0_db), "--report", str(report))) == 0
    qc = json.loads(report.read_text())["qc"]
    assert qc["manifests_recomputed"] == 1 and qc["interval_equal_shared"] > 0 and qc["interval_equal_a30_only"] > 0
    assert qc["c0_molecules_consistent"] >= 2                   # mA (heldout, all) and mV (valid, train_tiles)
    assert qc["declared_a30_only_admitted_molecules"] == 1      # mW
    assert qc["declared_edge_partial_intervals_in_c0"] > 0      # A48 edge_partial incl. the unmasked gL


def test_c0_molecule_states_match_the_repair_geometry(c0):
    db, _ = c0
    state = dict((r, (s, t)) for r, s, t in q(db, "SELECT read_name, molecule_state, train_tiles_excluded "
                                                  "FROM molecule_member ORDER BY read_name"))
    assert state["mA"][0] == "excluded_heldout"
    assert state["mB"] == ("train_track", False)
    assert state["mV"] == ("train_track", True)
    assert state["mW"] == ("train_track", False)


def _tamper(applied, tmp_path, sql, params=()):
    out, freeze, rec = applied
    bad = tmp_path / "bad.db"
    shutil.copyfile(out, bad)
    con = duckdb.connect(str(bad))
    con.execute(sql, list(params))
    con.execute("CHECKPOINT")
    con.close()
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(dict(rec, out_sha256=LK.sha256(bad))))
    return bad, f


def test_validator_rejects_a_base_masked_outside_the_repair_spans(world, manifest, applied, tmp_path, capsys):
    out, _, _ = applied
    rn, ws, _, seq, *_ = _row(out, EDGE_WID)
    p = 20000
    bad, f = _tamper(applied, tmp_path, "UPDATE geneList SET sequence = ? WHERE rn = ?",
                     [seq[:p] + "N" + seq[p + 1:], rn])
    assert VC.main(_validate_args(world, manifest, bad, f, "--no-recompute-manifest")) == 1
    assert "outside the repair spans" in capsys.readouterr().err


def test_validator_rejects_a_missing_repair_row(world, manifest, applied, tmp_path, capsys):
    bad, f = _tamper(applied, tmp_path, "DELETE FROM b6_dna_repair WHERE component_id = 'gV3' AND wid LIKE '%95020-125740'")
    assert VC.main(_validate_args(world, manifest, bad, f, "--no-recompute-manifest")) == 1
    assert "FAILED" in capsys.readouterr().err


def test_validator_rejects_a_label_kept_on_a_repair_span(world, manifest, applied, tmp_path, capsys):
    bad, f = _tamper(applied, tmp_path, "DELETE FROM b6_label_drop WHERE gene_id = 'gN'")
    assert VC.main(_validate_args(world, manifest, bad, f, "--no-recompute-manifest")) == 1
    assert "FAILED" in capsys.readouterr().err


def test_validator_rejects_a_freeze_that_does_not_bind_the_corpus(world, manifest, applied, tmp_path, capsys):
    out, freeze, rec = applied
    f = tmp_path / "f.json"
    f.write_text(json.dumps(dict(rec, out_sha256="0" * 64)))
    assert VC.main(_validate_args(world, manifest, out, f, "--no-recompute-manifest")) == 1
    assert "out_sha256" in capsys.readouterr().err


# ------------------------------------------------------------------------------ review 1 H3: the freeze contract
def test_a_wp2_freeze_feeds_a_per_species_consumer(world, applied, c0):
    """The canonical freeze format is species-indexed; a consumer looks up each C0 provenance species."""
    _, _, rec = applied
    c0_db, _ = c0
    prov = dict(q(c0_db, "SELECT key, value FROM c0_provenance"))
    species = json.loads(prov["species"])
    assert CR.freeze_manifest_sha256(rec, species) == json.loads(prov["interval_manifest_sha256"])
    with pytest.raises(CR.RepairError, match="species"):
        CR.freeze_manifest_sha256(dict(rec, interval_manifest_sha256=rec["interval_manifest_sha256"][species]), species)
    with pytest.raises(CR.RepairError, match="no interval manifest"):
        CR.freeze_manifest_sha256(rec, "Gmax")
    with pytest.raises(CR.RepairError, match="required key"):
        CR.verify_freeze({k: v for k, v in rec.items() if k != "base_sha256"})
