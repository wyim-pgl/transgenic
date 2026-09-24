"""b6_merge_evidence_db.py: one merged DB, one sha256 (A46 rule 9), schema check, no duplicate keys."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import test_b6_c0_world as W
import b6_build_c0 as B
import b6_merge_evidence_db as M
import test_b6_c0_build as TB
from transgenic.evidence import c0_schema as S
from transgenic.evidence import leakage as LK

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def dbs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("merge")
    world = W.build_corpus(tmp)
    W.write_est(tmp)
    W.write_ont(tmp)
    out = []
    for source in ("est", "ont"):
        args = W.cli_args(world, tmp, source=source)
        assert B.main(args) == 0
        out.append(args[args.index("--out") + 1])
    est = (out[0], None)
    TB.write_protein(tmp)
    hsp = TB.write_hsp(tmp, TB._mols(est), state=TB.est_state(est))
    args = W.cli_args(world, tmp, source="protein", extra=TB._protein_extra(tmp, est, hsp))
    assert B.main(args) == 0
    out.append(args[args.index("--out") + 1])
    return out


def count(db, table):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_merge_records_a_single_sha256_and_keeps_every_row(dbs, tmp_path):
    out = tmp_path / "evidence_c0.db"
    result = M.merge(dbs, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert result["sha256"] == digest
    assert (tmp_path / "evidence_c0.db.sha256").read_text().split() == [digest, "evidence_c0.db"]
    for table in ("evidence_alignment", "junction", "molecule_member", "excluded_interval"):
        assert count(out, table) == sum(count(d, table) for d in dbs), table
    con = duckdb.connect(str(out), read_only=True)
    S.verify_schema(con, merged=True)
    assert con.execute("SELECT count(*) FROM merge_manifest").fetchone()[0] == 3
    con.close()
    with pytest.raises(FileExistsError):
        M.merge(dbs, out)


def test_same_species_source_twice_is_refused(dbs, tmp_path):
    with pytest.raises(M.MergeError, match="duplicate species x source"):
        M.merge([dbs[0], dbs[0]], tmp_path / "x.db")


def _copy(db, tmp_path, name):
    dst = tmp_path / name
    shutil.copy(db, dst)
    return dst


def _republish(db):
    """A deliberately altered DB re-published (PROVENANCE output_sha256 = its new sha256), so the later gates run."""
    Path(str(db) + ".PROVENANCE.txt").write_text(f"output\t{db}\noutput_sha256\t{W.sha(db)}\n")
    return db


def test_input_without_provenance_is_refused(dbs, tmp_path):
    bare = _copy(dbs[0], tmp_path, "bare.db")                 # identical bytes, but not published
    with pytest.raises(M.MergeError, match="PROVENANCE.txt missing"):
        M.merge([bare], tmp_path / "x.db")
    assert not (tmp_path / "x.db").exists()


def test_duplicate_primary_keys_are_refused(dbs, tmp_path):
    clone = _copy(dbs[0], tmp_path, "clone.db")
    con = duckdb.connect(str(clone))
    for table in S.DATA_TABLES:
        con.execute(f"UPDATE {table} SET source = 'pacbio'")
    for table in ("c0_qc", "c0_provenance", "excluded_interval", "heldout_locus", "molecule_exclusion", "audit_placement",
                  "excluded_junction_coordinate"):
        con.execute(f"UPDATE {table} SET source = 'pacbio'")
    con.execute("UPDATE c0_meta SET value = 'pacbio' WHERE key = 'source'")
    con.close()
    _republish(clone)
    with pytest.raises(M.MergeError, match="duplicate evidence_alignment key"):
        M.merge([dbs[0], clone], tmp_path / "x.db")
    assert not (tmp_path / "x.db").exists()


def test_schema_version_and_leakage_gate_are_rechecked(dbs, tmp_path):
    old = _copy(dbs[0], tmp_path, "old.db")
    con = duckdb.connect(str(old))
    con.execute("UPDATE c0_meta SET value = 'c0-v0' WHERE key = 'schema_version'")
    con.close()
    _republish(old)
    with pytest.raises(S.SchemaError, match="schema version"):
        M.merge([old], tmp_path / "x.db")
    leak = _copy(dbs[0], tmp_path, "leak.db")
    con = duckdb.connect(str(leak))
    con.execute("UPDATE c0_qc SET value = 1 WHERE metric = 'leakage_nonzero'")
    con.close()
    _republish(leak)
    with pytest.raises(M.MergeError, match="leakage QC gate"):
        M.merge([leak], tmp_path / "y.db")


def test_tampered_input_differs_from_its_provenance_record(dbs, tmp_path):
    src = _copy(dbs[0], tmp_path, "t.db")
    shutil.copy(str(dbs[0]) + ".PROVENANCE.txt", str(src) + ".PROVENANCE.txt")
    con = duckdb.connect(str(src))
    con.execute("UPDATE junction SET n_observations = n_observations + 1")
    con.close()
    with pytest.raises(M.MergeError, match="PROVENANCE output_sha256"):
        M.merge([src], tmp_path / "z.db")


def test_merge_gate_refuses_orphan_and_unflagged_hsp_rows(dbs, tmp_path):
    with pytest.raises(M.MergeError, match="no Athaliana EST C0 is among the inputs"):   # protein DB without its EST DB
        M.merge([dbs[2]], tmp_path / "a.db")
    bad = _copy(dbs[2], tmp_path, "bad.db")
    con = duckdb.connect(str(bad))
    con.execute("UPDATE est_protein_hsp SET leakage_excluded = false")
    con.close()
    _republish(bad)
    with pytest.raises(M.MergeError, match="not leakage_excluded"):
        M.merge([dbs[0], bad], tmp_path / "b.db")
    assert not (tmp_path / "b.db").exists()


def _protein_copy(dbs, tmp_path, name, sql):
    db = _copy(dbs[2], tmp_path, name)
    con = duckdb.connect(str(db))
    con.execute(sql)
    con.close()
    return _republish(db)


def test_merge_refuses_a_protein_c0_projected_against_another_est_c0(dbs, tmp_path):
    """Finding 4: the protein C0 records inputs.est_c0.sha256; an EST C0 with the same molecule ids / states but other
    bytes (rebuilt or edited after the projection) is refused, whatever its rows say."""
    e2 = _copy(dbs[0], tmp_path, "est_v2.db")
    con = duckdb.connect(str(e2))
    con.execute("UPDATE junction SET n_observations = n_observations + 1")
    con.close()
    _republish(e2)
    with pytest.raises(M.MergeError, match="was projected against EST C0 sha256 .* rebuilt after the projection"):
        M.merge([e2, dbs[1], dbs[2]], tmp_path / "x.db")
    assert not (tmp_path / "x.db").exists()
    result = M.merge(dbs, tmp_path / "ok.db")                    # the recorded EST C0 itself merges
    assert set(result["identity"]) == {"Athaliana"} and result["identity"]["Athaliana"]["seed"] == W.CORPUS_SEED


def test_merge_refuses_inputs_with_different_exclusion_identity(dbs, tmp_path):
    """Finding 4: every input of one species must share leakage_intervals_sha256, seed and corpus inputs."""
    ont = _copy(dbs[1], tmp_path, "ont_other.db")
    con = duckdb.connect(str(ont))
    con.execute("UPDATE c0_provenance SET value = '\"" + "0" * 64 + "\"' WHERE key = 'leakage_intervals_sha256'")
    con.close()
    _republish(ont)
    with pytest.raises(M.MergeError, match=r"disagree on \['leakage_intervals_sha256'\]"):
        M.merge([dbs[0], ont], tmp_path / "y.db")
    ont2 = _copy(dbs[1], tmp_path, "ont_seed.db")
    con = duckdb.connect(str(ont2))
    con.execute("UPDATE c0_provenance SET value = '999' WHERE key = 'seed'")
    con.close()
    _republish(ont2)
    with pytest.raises(M.MergeError, match=r"disagree on \['seed'\]"):
        M.merge([dbs[0], ont2], tmp_path / "z.db")


def test_merge_gate_rechecks_protein_query_flags_and_accepted_placements(dbs, tmp_path):
    """Findings 2/3 at merge time: an HSP of an excluded_heldout protein query must be leakage_excluded, and every HSP
    interval must lie inside an accepted primary placement of its EST member."""
    pq = _protein_copy(dbs, tmp_path, "pq.db", "UPDATE est_protein_hsp SET protein_id = '4577_0:000002', leakage_excluded = false "
                                               "WHERE hsp_id = 'h1'")             # 4577_0:000002 is excluded_heldout
    with pytest.raises(M.MergeError, match="excluded_heldout protein query are not leakage_excluded"):
        M.merge([dbs[0], pq], tmp_path / "p.db")
    moved = _protein_copy(dbs, tmp_path, "moved.db", "UPDATE est_protein_hsp SET genomic_start0 = 170000, genomic_end0 = 170100 "
                                                     "WHERE hsp_id = 'h1'")     # outside e1/e2's accepted primary
    with pytest.raises(M.MergeError, match="no accepted primary placement"):
        M.merge([dbs[0], moved], tmp_path / "m.db")
    assert not (tmp_path / "p.db").exists() and not (tmp_path / "m.db").exists()


# ------------------------------------------------------------------------------ fix round 5 (Codex re-review)
def _edit_inputs(db, fn):
    """Rewrite the c0_provenance 'inputs' JSON of a copied DB through fn(dict) -> dict."""
    con = duckdb.connect(str(db))
    inputs = json.loads(con.execute("SELECT value FROM c0_provenance WHERE key = 'inputs'").fetchone()[0])
    con.execute("UPDATE c0_provenance SET value = ? WHERE key = 'inputs'", [json.dumps(fn(inputs), sort_keys=True)])
    con.close()
    return _republish(db)


def test_build_identity_is_compared_over_the_union_of_keys(dbs, tmp_path):
    """Finding 4: an input that OMITS a build_flags_* entry (before: silently accepted, the loop ran over the other
    side's keys) and one that carries an EXTRA entry (before: KeyError, not a MergeError) are both refused."""
    fewer = _edit_inputs(_copy(dbs[1], tmp_path, "fewer.db"), lambda d: {k: v for k, v in d.items() if k != "build_flags_1"})
    with pytest.raises(M.MergeError, match=r"disagree on \['build_flags_1'\]"):
        M.merge([dbs[0], fewer], tmp_path / "f.db")
    with pytest.raises(M.MergeError, match=r"disagree on \['build_flags_1'\]"):
        M.merge([fewer, dbs[0]], tmp_path / "f2.db")
    more = _edit_inputs(_copy(dbs[1], tmp_path, "more.db"), lambda d: dict(d, build_flags_2={"path": "x", "sha256": "1" * 64}))
    with pytest.raises(M.MergeError, match=r"disagree on \['build_flags_2'\]"):
        M.merge([dbs[0], more], tmp_path / "m.db")
    with pytest.raises(M.MergeError, match=r"disagree on \['build_flags_2'\]"):
        M.merge([more, dbs[0]], tmp_path / "m2.db")
    assert M.identity_diff({"a": 1, "b": 2}, {"a": 1}) == ["b"] and M.identity_diff({"a": 1}, {"a": 1, "c": 3}) == ["c"]
    assert not any((tmp_path / n).exists() for n in ("f.db", "f2.db", "m.db", "m2.db"))


def _with_counters(dbs, tmp_path, name, index, below, molecules):
    """A copy of dbs[index] whose recorded per-source exhaustiveness counters are replaced (an older / other builder's
    record), re-published so the merge gates run over it."""
    db = _copy(dbs[index], tmp_path, name)
    con = duckdb.connect(str(db))
    con.execute("UPDATE c0_qc SET value = ? WHERE metric = 'exhaustiveness_below_cap'", [below])
    con.execute("UPDATE c0_qc SET value = ? WHERE metric = 'exhaustiveness_molecules'", [molecules])
    con.close()
    return _republish(db)


def test_per_source_floor_and_species_sum_are_rechecked_in_the_merge_from_recorded_counters(dbs, tmp_path):
    """Fix round 6, item 3 (author 2026-09-23: BOTH gates).  A46 rule 4(b): exhaustiveness is "reported per species and
    required to be >= 99.9 %"; the per-source floor is 0.99 (A49.2).  The merge re-applies
    the floor to the counters each input RECORDED, so an EST C0 recording 16/17 (a DB an older builder published;
    the current builder aborts on it, test_b6_c0_build.py) is refused even though the species sum with a large ONT
    source (16 + 1203) / (17 + 1203) = 99.92 % would pass the species gate alone (fix round 5 accepted exactly that).
    Sources at or above the floor merge, and the species result records both thresholds."""
    est = _with_counters(dbs, tmp_path, "est16.db", 0, 16, 17)
    ont = _with_counters(dbs, tmp_path, "ont1203.db", 1, 1203, 1203)
    with pytest.raises(M.MergeError, match=r"per-source exhaustiveness floor: Athaliana/est .*0\.941176 < 0\.99.*raise N"):
        M.merge([est, ont], tmp_path / "both.db")
    assert not (tmp_path / "both.db").exists()
    est = _with_counters(dbs, tmp_path, "est999.db", 0, 999, 1000)
    result = M.merge([est, ont], tmp_path / "ok.db")
    ex = result["exhaustiveness"]["Athaliana"]
    assert (ex["below_cap"], ex["molecules"], ex["audit_n"]) == (2202, 2203, 50) and ex["exhaustiveness"] >= 0.999
    assert ex["by_source"] == {"est": {"below_cap": 999, "molecules": 1000}, "ont": {"below_cap": 1203, "molecules": 1203}}
    assert (ex["min_exhaustiveness"], ex["min_source_exhaustiveness"]) == (0.999, 0.99)
    con = duckdb.connect(str(tmp_path / "ok.db"), read_only=True)
    stored = json.loads(con.execute("SELECT value FROM c0_meta WHERE key = 'exhaustiveness'").fetchone()[0])
    con.close()
    assert stored["Athaliana"]["molecules"] == 2203
    # The species-level sum is its own, stricter gate: a source at 0.99 passes the floor but not the species gate alone.
    with pytest.raises(LK.ExhaustivenessError, match="species-level"):
        LK.species_exhaustiveness("X", [("est", 998, 1000, 50)])
    with pytest.raises(LK.ExhaustivenessError, match=r"per-source audit exhaustiveness 0\.985000 < 0\.99"):
        LK.source_exhaustiveness("X/est", 985, 1000, 50)
    LK.source_exhaustiveness("X/est", 990, 1000, 50)                  # exactly at the floor: passes
    with pytest.raises(LK.ExhaustivenessError, match="species-level"):
        LK.species_exhaustiveness("X", [("est", 990, 1000, 50)])      # ...but 0.99 < 0.999 for the species
    assert LK.source_exhaustiveness("X/est", 0, 0, 50) is None          # an empty source has no fraction


def test_exhaustiveness_counters_and_one_audit_n_are_required(dbs, tmp_path):
    """Finding 5, fail closed: an audited source without the per-source counters, or sources disagreeing on N."""
    bare = _copy(dbs[1], tmp_path, "bare.db")
    con = duckdb.connect(str(bare))
    con.execute("DELETE FROM c0_qc WHERE metric = 'exhaustiveness_molecules'")
    con.close()
    _republish(bare)
    with pytest.raises(M.MergeError, match=r"lacks \['exhaustiveness_molecules'\]"):
        M.merge([dbs[0], bare], tmp_path / "b.db")
    other_n = _copy(dbs[1], tmp_path, "n60.db")
    con = duckdb.connect(str(other_n))
    con.execute("UPDATE c0_qc SET value = 60 WHERE metric = 'audit_n'")
    con.close()
    _republish(other_n)
    with pytest.raises(M.MergeError, match=r"different audit N \[50, 60\]"):
        M.merge([dbs[0], other_n], tmp_path / "n.db")
    with pytest.raises(LK.ExhaustivenessError, match="species-level"):
        LK.species_exhaustiveness("X", [("est", 998, 1000, 50)])
    assert LK.species_exhaustiveness("X", [("est", 998, 1000, 50), ("ont", 1000, 1000, 50)])["exhaustiveness"] == 0.999


def test_completeness_code_policy_is_required_and_rechecked(dbs, tmp_path):
    """Finding 6: an input without a declared A16 completeness-code policy, or whose accepted rows contradict the
    declared NULL policy, is refused (a NULL code column is never silent)."""
    nopol = _copy(dbs[1], tmp_path, "nopolicy.db")
    con = duckdb.connect(str(nopol))
    con.execute("DELETE FROM c0_provenance WHERE key = 'completeness_codes'")
    con.close()
    _republish(nopol)
    with pytest.raises(M.MergeError, match="no known completeness_codes.policy"):
        M.merge([dbs[0], nopol], tmp_path / "np.db")
    coded = _copy(dbs[1], tmp_path, "coded.db")
    con = duckdb.connect(str(coded))
    con.execute("UPDATE evidence_alignment SET complete_5p_code = '5I' WHERE read_name = 'o1'")
    con.close()
    _republish(coded)
    with pytest.raises(M.MergeError, match="1 accepted alignment row\\(s\\) contradict the recorded A16"):
        M.merge([dbs[0], coded], tmp_path / "c.db")


def test_merge_rechecks_phase_consistency_among_retained_hsps(dbs, tmp_path):
    """Finding 2 at merge time: two retained HSPs of one molecule and protein source that overlap in different codon
    frames (the rule-6 MULTI_HSP_RULE was not applied to the retained set) are refused independently of the builder."""
    clash = _protein_copy(dbs, tmp_path, "clash.db",
                          "INSERT INTO est_protein_hsp SELECT * REPLACE ('hx' AS hsp_id, 1 AS phase) FROM est_protein_hsp "
                          "WHERE hsp_id = 'h1' AND interval_index = 0")
    with pytest.raises(M.MergeError, match="1 retained est_protein_hsp pair\\(s\\) .* different strand or codon frame"):
        M.merge([dbs[0], clash], tmp_path / "x.db")
    same = _protein_copy(dbs, tmp_path, "same.db",
                         "INSERT INTO est_protein_hsp SELECT * REPLACE ('hy' AS hsp_id, 10003 AS genomic_start0, 0 AS phase) "
                         "FROM est_protein_hsp WHERE hsp_id = 'h1' AND interval_index = 0")   # same frame from 3 nt later
    M.merge([dbs[0], same], tmp_path / "y.db")
    assert (tmp_path / "y.db").exists() and not (tmp_path / "x.db").exists()


def test_merge_rechecks_phase_consistency_among_excluded_hsps_and_never_across(dbs, tmp_path):
    """Fix round 6, finding 2, at merge time: two leakage-excluded HSPs of one (excluded_heldout) molecule and protein
    source overlapping in different frames are refused (the excluded population is resolved on its own); a retained
    and an excluded HSP in different frames are NOT a conflict (populations are never tested against each other)."""
    con = duckdb.connect(str(dbs[2]), read_only=True)
    assert con.execute("SELECT leakage_excluded FROM est_protein_hsp WHERE hsp_id = 'h2'").fetchall() == [(True,)]
    con.close()
    clash = _protein_copy(dbs, tmp_path, "xclash.db",
                          "INSERT INTO est_protein_hsp SELECT * REPLACE ('hz' AS hsp_id, 0 AS phase) FROM est_protein_hsp "
                          "WHERE hsp_id = 'h2' AND interval_index = 0")            # h2 is phase 1 on the held-out molecule
    with pytest.raises(M.MergeError, match="1 leakage-excluded est_protein_hsp pair\\(s\\) .* different strand or codon frame"):
        M.merge([dbs[0], clash], tmp_path / "xz.db")
    keep = TB.mol(dbs[0], "e2")
    across = _protein_copy(dbs, tmp_path, "across.db",
                           "INSERT INTO est_protein_hsp SELECT * REPLACE ('hw' AS hsp_id, 1 AS phase, true AS leakage_excluded, "
                           "true AS train_tiles_excluded, 'protein_query' AS exclusion_cause, '4577_0:000002' AS protein_id) "
                           "FROM est_protein_hsp WHERE hsp_id = 'h1' AND interval_index = 0")   # excluded (its query is), other frame than retained h1
    con = duckdb.connect(str(across), read_only=True)
    assert con.execute("SELECT molecule_id FROM est_protein_hsp WHERE hsp_id = 'hw'").fetchall() == [(keep,)]
    con.close()
    M.merge([dbs[0], across], tmp_path / "across_ok.db")
    assert (tmp_path / "across_ok.db").exists() and not (tmp_path / "xz.db").exists()


# ------------------------------------------------------------------------------ fix round 7 (Codex review 4)
def test_merge_refuses_an_hsp_whose_library_no_est_member_has(dbs, tmp_path):
    """Codex review 4 MEDIUM at merge time: a projection made before a library correction keeps the old library; no
    EST member of its molecule carries it, so the merge refuses (library withholding would otherwise keep it)."""
    stale = _protein_copy(dbs, tmp_path, "stale.db", "UPDATE est_protein_hsp SET library = 'L-old' WHERE hsp_id = 'h1'")
    with pytest.raises(M.MergeError, match="stale projection after a metadata correction"):
        M.merge([dbs[0], stale], tmp_path / "s.db")
    assert not (tmp_path / "s.db").exists()
