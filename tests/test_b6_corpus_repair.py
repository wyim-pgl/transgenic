"""WP2 rule 5 repair (plan v4.1 §2) on the variant world (tests/test_b6_repair_world.py): geometry = the C0 rule
4(a) component interval, closure, RC twin, row-set refusal, freeze, independent validation and C0 consequences."""
import json
import shutil
import sys
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
    assert rec["row_counts"] == CR._row_counts_path(world["corpus"]) == rec["base_row_counts"]
    assert rec["row_set_policy"] == CR.ROW_SET_REFUSE and rec["row_set_drop"]["tiles"] == []
    assert rec["n_pilot"] == CR.n_pilot_record(rec["train_rows"]) and rec["counts"] == rec["measured_counts"]
    assert set(rec["inputs"]) == {"splits"}
    sf = rec["species_flags"][RW.SPECIES]
    assert sf["flags"]["sha256"] == LK.sha256(world["a22"])
    assert [b["sha256"] for b in sf["build_flags"]] == [LK.sha256(world["a22"]), LK.sha256(world["a30"])]
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


# ------------------------------------------------------------------------------ decision 18: drop the event tiles
def _trainer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_b6_corpus_rows",
                                                  Path(__file__).resolve().parents[1] / "train" /
                                                  "train_HyenaTransgenic_B6.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def evworld(tmp_path_factory):
    """The refuse world applied with --drop-row-set-events, and the same repair with the events not acted on."""
    tmp = tmp_path_factory.mktemp("event_world")
    w = RW.build_world(tmp / "w", refuse=True)
    man = tmp / "m.jsonl"
    LK.write_interval_manifest(RW.leakage_model(w).interval_manifest, man)
    report = tmp / "measure.json"
    assert BC.main(_cli(w, man, "--measure-only", "--report", str(report))) == 0
    out, freeze = tmp / "b6_drop.db", tmp / "drop_freeze.json"
    assert BC.main(_cli(w, man, "--apply", "--drop-row-set-events", "--out", str(out), "--freeze", str(freeze))) == 0
    real = CR.repair_tile

    def events_ignored(*a, **k):
        rep = real(*a, **k)
        rep.over_mask, rep.rc_orphan = False, None
        return rep

    mp = pytest.MonkeyPatch()
    mp.setattr(CR, "repair_tile", events_ignored)
    try:
        ref = tmp / "b6_keep.db"
        CR.apply(w["corpus"], ref, tmp / "keep_freeze.json", **_kw(w, man))
    finally:
        mp.undo()
    return {"world": w, "manifest": man, "report": json.loads(report.read_text()), "out": out, "freeze": freeze,
            "rec": json.loads(freeze.read_text()), "ref": ref}


def _dropped(ev):
    return q(ev["out"], "SELECT " + ", ".join(CR.ROW_DROP_COLUMNS) + " FROM b6_row_drop ORDER BY rn")


def test_drop_mode_deletes_exactly_the_event_tiles_and_their_rc_twins(evworld):
    base, out, rep = evworld["world"]["corpus"], evworld["out"], evworld["report"]
    fwd = {rn for _, rn, _, _, _ in rep["event_tiles"]}
    wids = {wid for _, _, wid, _, _ in rep["event_tiles"]}
    twins = {rn for (rn,) in q(base, "SELECT rn FROM geneList WHERE list_contains(?, geneModel)",
                               [[w + "-rc" for w in wids]])}
    assert rep["events"] == {"tiles_over_mask_fraction": 1, "rc_twin_orphaned": 2} and twins
    gone = {r[0] for r in q(base, "SELECT rn FROM geneList")} - {r[0] for r in q(out, "SELECT rn FROM geneList")}
    assert gone == fwd | twins == {r[0] for r in _dropped(evworld)}
    rows = [dict(zip(CR.ROW_DROP_COLUMNS, r)) for r in _dropped(evworld)]
    assert rows == evworld["rec"]["row_set_drop"]["tiles"] == rep["row_set_drop"]["tiles"]
    assert {r["split"] for r in rows} <= {"train", "valid"}
    assert all(r["forward_rn"] in fwd for r in rows)
    assert "tiles_over_mask_fraction;rc_twin_orphaned" in {r["events"] for r in rows}      # gBIG's tile: both events
    assert not q(out, "SELECT * FROM window_genes WHERE list_contains(?, window_id)", [sorted(wids)])
    test = "SELECT * FROM geneList WHERE split = 'test' ORDER BY rn"
    assert q(out, test) == q(base, test)
    assert evworld["rec"]["row_set_policy"] == CR.ROW_SET_DROP
    assert dict(q(out, "SELECT key, value FROM b6_corpus_meta"))["row_set_policy"] == CR.ROW_SET_DROP


def test_surviving_rows_are_byte_identical_to_the_non_dropping_repair(evworld):
    out, ref = evworld["out"], evworld["ref"]
    rns = sorted(r[0] for r in _dropped(evworld))
    windows = sorted({r[1] for r in _dropped(evworld) if not r[4]})
    assert q(out, "SELECT * FROM geneList ORDER BY rn") == q(
        ref, "SELECT * FROM geneList WHERE NOT list_contains(?, rn) ORDER BY rn", [rns])
    assert q(out, "SELECT * FROM window_genes ORDER BY ALL") == q(
        ref, "SELECT * FROM window_genes WHERE NOT list_contains(?, window_id) ORDER BY ALL", [windows])
    for t in ("b6_dna_repair", "b6_label_drop"):
        assert q(out, f"SELECT * FROM {t} ORDER BY ALL") == q(
            ref, f"SELECT * FROM {t} WHERE NOT list_contains(?, rn) ORDER BY ALL", [rns])
    for t in ("tile_blocks", "gene_key_map", "gene_split", "build_manifest", "rejected_records"):
        assert q(out, f"SELECT * FROM {t} ORDER BY ALL") == q(ref, f"SELECT * FROM {t} ORDER BY ALL"), t


def test_freeze_and_loader_see_the_reduced_train_rows_and_n_pilot(evworld):
    base, out, rec = evworld["world"]["corpus"], evworld["out"], evworld["rec"]
    rows = [dict(zip(CR.ROW_DROP_COLUMNS, r)) for r in _dropped(evworld)]
    trainer = _trainer()
    base_train, base_valid, _ = trainer.corpus_rows(base)
    train, valid, c3 = trainer.corpus_rows(out)
    dropped_train = {r["rn"] for r in rows if r["split"] == "train"} & set(base_train)
    assert dropped_train and len(train) == len(base_train) - len(dropped_train) == rec["train_rows"]
    assert set(valid) == set(base_valid) - {r["rn"] for r in rows}
    assert not {r["rn"] for r in rows} & {r[0] for r in c3}                          # no C3 donor from a dropped tile
    assert rec["n_pilot"] == {"value": len(train) * 2 // 96, "formula": "floor(train_rows * epochs / effective_batch)",
                              "train_rows": len(train), "epochs": 2, "effective_batch": 96,
                              "train_rows_predicate": CR.LOADER_TRAIN_PREDICATE}
    drop = rec["row_set_drop"]
    assert (drop["train_rows_before"], drop["train_rows_after"]) == (len(base_train), len(train))
    assert drop["n_pilot"] == rec["n_pilot"] == evworld["report"]["row_set_drop"]["n_pilot"]
    per = {}
    for r in rows:
        c = per.setdefault(r["species_id"], {}).setdefault(r["split"], {"forward": 0, "rc": 0})
        c["rc" if r["is_rc"] else "forward"] += 1
    assert drop["counts"] == per and drop["rows"] == len(rows) and drop["window_genes_dropped"] > 0
    assert rec["row_counts"]["train"] == rec["base_row_counts"]["train"] - sum(r["split"] == "train" for r in rows)
    rt = trainer.rt
    assert rt.bind_n_pilot(rec, len(train), rec["n_pilot"]["value"]) == rec["n_pilot"]["value"]
    with pytest.raises(ValueError, match="config n_pilot 4142 != the corpus freeze N_pilot"):
        rt.bind_n_pilot(rec, len(train), 4142)
    with pytest.raises(ValueError, match="train_rows"):
        rt.bind_n_pilot(rec, len(base_train), rec["n_pilot"]["value"])
    with pytest.raises(ValueError, match="no n_pilot"):
        rt.bind_n_pilot({k: v for k, v in rec.items() if k != "n_pilot"}, len(train), 0)
    value, verified = rec["n_pilot"]["value"], {"provenance": {"freeze": LK.sha256(evworld["freeze"])}}
    assert trainer.bind_freeze_n_pilot(evworld["freeze"], verified, len(train), value) == value     # main's binding
    with pytest.raises(ValueError, match="changed after the bindings verified it"):
        trainer.bind_freeze_n_pilot(evworld["freeze"], {"provenance": {"freeze": "0" * 64}}, len(train), value)


def test_n_pilot_formula_is_the_trainer_rule_and_gives_4142_on_the_a46_basis_and_4099_repaired():
    from transgenic.training import b6_runtime as rt

    assert (CR.PILOT_EPOCHS, CR.PILOT_EFFECTIVE_BATCH, CR.N_PILOT_FORMULA) == (
        rt.PILOT_EPOCHS, rt.EFFECTIVE_BATCH, rt.N_PILOT_FORMULA)
    assert CR.n_pilot_record(198828)["value"] == rt.n_pilot_from_train_rows(198828) == 4142
    assert CR.n_pilot_record(196774)["value"] == rt.n_pilot_from_train_rows(196774) == rt.N_PILOT == 4099


def test_bind_n_pilot_accepts_the_repaired_corpus_freeze_with_the_updated_configs_and_refuses_4142():
    """Decision 18: the 9-species measure_v2.json of b6_prepare_20260925 gives train_rows 196,774 -> N_pilot 4,099."""
    from transgenic.training import b6_runtime as rt

    freeze = {"n_pilot": {"value": 4099, "formula": rt.N_PILOT_FORMULA, "train_rows": 196774, "epochs": 2,
                          "effective_batch": 96, "train_rows_predicate": CR.LOADER_TRAIN_PREDICATE}}
    assert freeze["n_pilot"] == CR.n_pilot_record(196774)
    for name in ("b6_400m_win_v3.json", "b6_400m_win_v3_wp5_pilot.json"):
        cfg = rt.load_b6_config(str(Path(__file__).resolve().parents[1] / "configs" / name))
        assert rt.bind_n_pilot(freeze, 196774, cfg["n_pilot"]) == 4099
        assert cfg.get("wp5_pilot", {"stop_after_updates": 4099})["stop_after_updates"] == 4099
    with pytest.raises(ValueError, match="config n_pilot 4142 != the corpus freeze N_pilot 4099"):
        rt.bind_n_pilot(freeze, 196774, 4142)
    with pytest.raises(ValueError, match="train_rows"):
        rt.bind_n_pilot(freeze, 198828, 4099)


def test_validator_confirms_the_dropped_rows_and_rejects_a_hidden_drop(evworld, tmp_path, capsys):
    w, man, out, freeze, rec = (evworld[k] for k in ("world", "manifest", "out", "freeze", "rec"))
    report = tmp_path / "v.json"
    assert VC.main(_validate_args(w, man, out, freeze, "--report", str(report))) == 0
    qc = json.loads(report.read_text())["qc"]
    assert qc["tiles_dropped_events_confirmed"] == len({r[5] for r in _dropped(evworld)})
    assert qc["train_rows"] == rec["train_rows"]
    fwd = next(r for r in _dropped(evworld) if not r[4])
    bad, f = _tamper((out, freeze, rec), tmp_path, "DELETE FROM b6_row_drop WHERE rn = ?", [fwd[0]])
    f.write_text(json.dumps(dict(json.loads(f.read_text()), row_set_drop=dict(
        rec["row_set_drop"], tiles=[t for t in rec["row_set_drop"]["tiles"] if t["rn"] != fwd[0]]))))
    assert VC.main(_validate_args(w, man, bad, f, "--no-recompute-manifest")) == 1
    assert "FAILED" in capsys.readouterr().err


def test_drop_flag_is_an_apply_option(world, manifest, tmp_path):
    assert BC.main(_cli(world, manifest, "--measure-only", "--drop-row-set-events")) == 1


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


# ------------------------------------------------------------------------------ C0 audit memory (source_v14)
def _normalised_memory(value):
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{value}'")
        return con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    finally:
        con.close()


def _tamper_c0(c0, tmp_path, sql, params=()):
    db, _ = c0
    bad = tmp_path / "bad_c0.db"
    shutil.copyfile(db, bad)
    con = duckdb.connect(str(bad))
    con.execute(sql, list(params))
    con.execute("CHECKPOINT")
    con.close()
    return bad


def _molecule_audit_inputs(manifest, corpus):
    """``repair_rows`` and ``manifest_by_comp`` exactly as ``b6_validate_corpus.validate`` hands them to
    ``audit_molecules``."""
    from collections import defaultdict

    by_comp = defaultdict(list)
    for r in LK.read_interval_manifest(manifest):
        by_comp[(r["species"], r["chrom"], r["component_id"])].append(r)
    splits = dict(q(corpus, "SELECT rn, coalesce(split, 'train') FROM geneList ORDER BY rn"))
    repair_rows = []
    for row in q(corpus, "SELECT " + ", ".join(CR.REPAIR_COLUMNS) + " FROM b6_dna_repair ORDER BY ALL"):
        r = dict(zip(CR.REPAIR_COLUMNS, row))
        r["species"], r["chrom"], r["wid"], r["split"] = r["species_id"], r["chromosome"], r["wid"], splits[r["rn"]]
        repair_rows.append(r)
    return repair_rows, by_comp


def _old_audit_molecules(c0_paths, repair_rows, manifest_by_comp, qc):
    """The c8e01f4 ``audit_molecules`` verbatim (every molecule state materialised in a Python dict): the reference
    the streamed SQL lookup must reproduce."""
    from collections import defaultdict

    shared = defaultdict(list)
    a30 = defaultdict(list)
    for r in repair_rows:
        if r["is_rc"]:
            continue
        comp_rows = manifest_by_comp[(r["species"], r["chrom"], r["component_id"])]
        c0 = {(x["cause"], x["scope"]): x["c0_exclusion"] for x in comp_rows}
        target = shared if c0[(r["cause"], r["scope"])] else a30
        target[(r["species"], r["chrom"])].append((r["start0"], r["end0"], r["scope"], r["split"]))
    for path in c0_paths:
        con = duckdb.connect(str(path), read_only=True)
        try:
            states = {}
            for sp, src, mid, state, tte in con.execute(
                    "SELECT species, source, molecule_id, min(molecule_state), bool_or(train_tiles_excluded) "
                    "FROM molecule_member GROUP BY ALL ORDER BY ALL").fetchall():
                states[(sp, src, mid)] = (state, tte)
            cur = con.cursor()
            cur.execute("SELECT species, source, molecule_id, chrom, start0, end0 FROM audit_placement ORDER BY ALL")
            seen_bad, seen_ok, seen_a30 = set(), set(), set()
            while True:
                rows = cur.fetchmany(4096)
                if not rows:
                    break
                for sp, src, mid, chrom, s, e in rows:
                    key = (sp, src, mid)
                    state, tte = states.get(key, (None, None))
                    VC.require(state is not None, f"{path}: placement of unknown molecule {key}")
                    for a, b, scope, split in shared.get((sp, chrom), ()):
                        if s < b and e > a:
                            ok = state == "excluded_heldout" if scope == "all" else (
                                state == "excluded_heldout" or (split == "train" and bool(tte)) or split != "train")
                            (seen_ok if ok else seen_bad).add(key)
                    for a, b, _, _ in a30.get((sp, chrom), ()):
                        if s < b and e > a and state == "train_track" and not tte:
                            seen_a30.add(key)
            cur.close()
            VC.require(not seen_bad, f"{path}: molecules overlapping a shared-cause repair span are admitted: "
                                     f"{sorted(seen_bad)[:5]}")
            qc["c0_molecules_consistent"] += len(seen_ok)
            qc["declared_a30_only_admitted_molecules"] += len(seen_a30)
            ep = con.execute("SELECT count(*) FROM excluded_interval WHERE cause = 'edge_partial'").fetchone()[0]
            qc["declared_edge_partial_intervals_in_c0"] += ep
        finally:
            con.close()


def test_validator_applies_and_records_duckdb_memory_limit_and_threads(world, manifest, applied, c0, tmp_path):
    out, freeze, _ = applied
    c0_db, _ = c0
    report = tmp_path / "limited.json"
    assert VC.main(_validate_args(world, manifest, out, freeze, "--c0", str(c0_db), "--report", str(report),
                                  "--no-recompute-manifest", "--memory-limit", "1GB", "--threads", "2")) == 0
    settings = json.loads(report.read_text())["duckdb"]
    assert settings["threads"] == 2
    assert settings["memory_limit"] == _normalised_memory("1GB")
    assert settings["preserve_insertion_order"] is False
    spill = Path(settings["temp_directory"])
    assert spill.parent == tmp_path and not spill.exists()      # spill dir next to the report, removed afterwards
    assert sorted(p.name for p in tmp_path.iterdir()) == ["limited.json"]


def test_audit_molecules_rejects_a_placement_of_an_unknown_molecule(manifest, applied, c0, tmp_path):
    from collections import Counter

    out, _, _ = applied
    bad = _tamper_c0(c0, tmp_path, "INSERT INTO audit_placement SELECT species, source, 'ghost-molecule', 'ghost-member', "
                                   "origin, placement_kind, chrom, start0, end0, mapq FROM audit_placement LIMIT 1")
    repair_rows, by_comp = _molecule_audit_inputs(manifest, out)
    with pytest.raises(VC.ValidationError, match="placement of unknown molecule"):
        VC.audit_molecules([str(bad)], repair_rows, by_comp, Counter())
    with pytest.raises(VC.ValidationError, match="placement of unknown molecule"):
        _old_audit_molecules([str(bad)], repair_rows, by_comp, Counter())


def test_audit_molecules_equals_the_dict_based_lookup_with_and_without_memory_options(world, manifest, applied, c0,
                                                                                       tmp_path):
    from collections import Counter

    out, freeze, _ = applied
    c0_db, _ = c0
    keys = ("c0_molecules_consistent", "declared_a30_only_admitted_molecules", "declared_edge_partial_intervals_in_c0")
    plain, limited = tmp_path / "plain.json", tmp_path / "limited.json"
    assert VC.main(_validate_args(world, manifest, out, freeze, "--c0", str(c0_db), "--report", str(plain),
                                  "--no-recompute-manifest")) == 0
    assert VC.main(_validate_args(world, manifest, out, freeze, "--c0", str(c0_db), "--report", str(limited),
                                  "--no-recompute-manifest", "--memory-limit", "1GB", "--threads", "1")) == 0
    qc_plain, qc_limited = json.loads(plain.read_text())["qc"], json.loads(limited.read_text())["qc"]
    assert qc_plain == qc_limited
    repair_rows, by_comp = _molecule_audit_inputs(manifest, out)
    old = Counter()
    _old_audit_molecules([str(c0_db)], repair_rows, by_comp, old)
    assert old["c0_molecules_consistent"] >= 2 and old["declared_a30_only_admitted_molecules"] == 1
    assert {k: qc_plain[k] for k in keys} == {k: old[k] for k in keys}
    new = Counter()
    VC.audit_molecules([str(c0_db)], repair_rows, by_comp, new)
    assert new == old


def test_audit_molecules_still_rejects_an_admitted_molecule_on_a_shared_span(manifest, applied, c0, tmp_path):
    from collections import Counter

    out, _, _ = applied
    bad = _tamper_c0(c0, tmp_path, "UPDATE molecule_member SET molecule_state = 'train_track', train_tiles_excluded = false "
                                   "WHERE read_name = 'mA'")
    repair_rows, by_comp = _molecule_audit_inputs(manifest, out)
    with pytest.raises(VC.ValidationError, match="shared-cause repair span are admitted") as new:
        VC.audit_molecules([str(bad)], repair_rows, by_comp, Counter())
    with pytest.raises(VC.ValidationError, match="shared-cause repair span are admitted") as old:
        _old_audit_molecules([str(bad)], repair_rows, by_comp, Counter())
    assert str(new.value) == str(old.value)


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


# ------------------------------------------------------------------------------ per-species A22 / A30 flags
def _sp_kw(w, manifests, **over):
    kw = dict(splits=w["split"], seed=RW.SEED, manifests=list(manifests.values()),
              flags={sp: a22 for sp, (a22, _) in w["flags"].items()},
              build_flags={sp: [a22, a30] for sp, (a22, a30) in w["flags"].items()})
    kw.update(over)
    return kw


def _sp_cli(w, manifests, *extra):
    out = ["--base", str(w["corpus"]), "--splits", str(w["split"]), "--seed", str(RW.SEED),
           "--interval-manifest", *[str(p) for p in manifests.values()]]
    for sp, (a22, a30) in w["flags"].items():
        out += ["--flags", f"{sp}={a22}", "--build-flags", f"{sp}={a22}", f"{sp}={a30}"]
    return out + list(extra)


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    w = RW.build_two_species_world(tmp_path_factory.mktemp("two_species"))
    return w, RW.two_species_manifests(w)


def test_species_b_a22_hard_edge_crosser_is_masked_with_its_own_flags(two):
    w, manifests = two
    b_a22 = [r for r in LK.read_interval_manifest(manifests[RW.SPECIES_B]) if r["cause"] == "a22_hard"]
    assert [r["component_id"] for r in b_a22] == ["gW2"]            # B's C0 manifest names the A22-hard seed
    _, out = CR.measure(w["corpus"], **_sp_kw(w, manifests))
    assert out["counts"][RW.SPECIES_B]["train"]["a22_hard"]["spans"] > 0
    assert "a22_hard" not in out["counts"][RW.SPECIES]["train"]    # A's gW2 is A30-only, never an A22 cause
    for sp, (a22, a30) in w["flags"].items():
        rec = out["species_flags"][sp]
        assert rec["flags"] == {"path": str(Path(a22).absolute()), "sha256": LK.sha256(a22)}
        assert [b["sha256"] for b in rec["build_flags"]] == [LK.sha256(a22), LK.sha256(a30)]


def test_one_species_flag_file_is_refused_for_a_multi_species_corpus(two):
    """The old single --flags / --build-flags form gave species B no flags: its A22-hard edge crosser went unmasked
    without an error.  It is now refused unless the corpus has a single species."""
    w, manifests = two
    a22, a30 = w["flags"][RW.SPECIES]
    with pytest.raises(CR.RepairError, match="ambiguous"):
        CR.measure(w["corpus"], **_sp_kw(w, manifests, flags=a22, build_flags=[a22, a30]))


def test_a_species_without_its_flags_is_refused(two):
    w, manifests = two
    a22, a30 = w["flags"][RW.SPECIES]
    with pytest.raises(CR.RepairError, match=r"--flags: no flag file for species \['Gmax'\]"):
        CR.measure(w["corpus"], **_sp_kw(w, manifests, flags={RW.SPECIES: a22}))
    with pytest.raises(CR.RepairError, match=r"--build-flags: no flag file for species \['Gmax'\]"):
        CR.measure(w["corpus"], **_sp_kw(w, manifests, build_flags={RW.SPECIES: [a22, a30]}))
    with pytest.raises(CR.RepairError, match="not in the corpus build_manifest"):
        CR.measure(w["corpus"], **_sp_kw(w, manifests, flags=dict(_sp_kw(w, manifests)["flags"], Osativa=a22)))


def test_a_flag_file_naming_another_species_is_refused(two):
    w, manifests = two
    a_a22, a_a30 = w["flags"][RW.SPECIES]
    kw = _sp_kw(w, manifests)
    with pytest.raises(CR.RepairError, match="has rows for other species"):
        CR.measure(w["corpus"], **dict(kw, flags=dict(kw["flags"], Gmax=a_a22)))
    with pytest.raises(CR.RepairError, match="has rows for other species"):
        CR.measure(w["corpus"], **dict(kw, build_flags=dict(kw["build_flags"], Gmax=[a_a22, a_a30])))


def test_cli_per_species_flags_measure_and_mixed_forms(two, tmp_path, capsys):
    w, manifests = two
    report = tmp_path / "measure.json"
    assert BC.main(_sp_cli(w, manifests, "--measure-only", "--report", str(report))) == 0
    out = json.loads(report.read_text())
    assert set(out["species_flags"]) == {RW.SPECIES, RW.SPECIES_B}
    assert out["counts"][RW.SPECIES_B]["train"]["a22_hard"]["spans"] > 0
    a22, a30 = w["flags"][RW.SPECIES]
    mixed = [x for x in _sp_cli(w, manifests, "--measure-only")]
    mixed[mixed.index(f"{RW.SPECIES}={a22}")] = str(a22)             # one bare --flags among SPECIES=PATH ones
    assert BC.main(mixed) == 1
    assert "mix SPECIES=PATH and bare paths" in capsys.readouterr().err
    single = ["--base", str(w["corpus"]), "--splits", str(w["split"]), "--seed", str(RW.SEED), "--interval-manifest",
              *[str(p) for p in manifests.values()], "--flags", str(a22), "--build-flags", str(a22), str(a30),
              "--measure-only"]
    assert BC.main(single) == 1
    assert "ambiguous" in capsys.readouterr().err


def test_parse_flag_args_keeps_each_species_build_files_in_order():
    fl, bf = CR.parse_flag_args(["A=/q/a22", "B=/q/b22"], [["A=/q/a22", "A=/q/a30"], ["B=/q/b22", "B=/q/b30"]])
    assert fl == {"A": "/q/a22", "B": "/q/b22"} and bf == {"A": ["/q/a22", "/q/a30"], "B": ["/q/b22", "/q/b30"]}
    assert CR.parse_flag_args(["/q/x=1/a22"], [["/q/a22", "/q/a30"]]) == ("/q/x=1/a22", ["/q/a22", "/q/a30"])
    with pytest.raises(CR.RepairError, match="twice"):
        CR.parse_flag_args(["A=/q/a", "A=/q/b"], [["A=/q/a"]])


def test_two_species_apply_freeze_and_validator_bind_each_species_flags(two, tmp_path, capsys):
    w, manifests = two
    out, freeze = tmp_path / "b6.db", tmp_path / "freeze.json"
    assert BC.main(_sp_cli(w, manifests, "--apply", "--out", str(out), "--freeze", str(freeze))) == 0
    rec = json.loads(freeze.read_text())
    for sp, (a22, a30) in w["flags"].items():
        assert rec["species_flags"][sp]["flags"]["sha256"] == LK.sha256(a22)
        assert [b["sha256"] for b in rec["species_flags"][sp]["build_flags"]] == [LK.sha256(a22), LK.sha256(a30)]
    assert rec["counts"][RW.SPECIES_B]["train"]["a22_hard"]["spans"] > 0
    args = ["--base", str(w["corpus"]), "--corpus", str(out), "--freeze", str(freeze), "--splits", str(w["split"]),
            "--seed", str(RW.SEED), "--interval-manifest", *[str(p) for p in manifests.values()]]
    for sp, (a22, a30) in w["flags"].items():
        args += ["--flags", f"{sp}={a22}", "--build-flags", f"{sp}={a22}", f"{sp}={a30}"]
    assert VC.main(args + ["--report", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text())["qc"]["manifests_recomputed"] == 2
    bad = tmp_path / "freeze_bad.json"
    rec["species_flags"][RW.SPECIES_B]["flags"]["sha256"] = "0" * 64
    bad.write_text(json.dumps(rec))
    assert VC.main([x if x != str(freeze) else str(bad) for x in args] + ["--no-recompute-manifest"]) == 1
    assert "freeze species_flags" in capsys.readouterr().err
