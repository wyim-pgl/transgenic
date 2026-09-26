"""b6_build_c0.py end to end on the synthetic world: molecule units, rule 4 states, junction correction,
chain coordinates, rule 4(c), QC gate, fail-closed input gates and the protein / route-B path."""
import json
import shutil
from pathlib import Path

import pytest

import test_b6_c0_world as W
import b6_build_c0 as B
from transgenic.evidence import leakage as LK

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build_corpus(tmp_path_factory.mktemp("world"))


@pytest.fixture(scope="module")
def est_db(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("est")
    W.write_est(tmp)
    args = W.cli_args(world, tmp)
    assert B.main(args) == 0
    out = args[args.index("--out") + 1]
    return out, json.load(open(args[args.index("--qc") + 1]))


@pytest.fixture(scope="module")
def ont_db(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ont")
    W.write_ont(tmp)
    args = W.cli_args(world, tmp, source="ont")
    assert B.main(args) == 0
    return args[args.index("--out") + 1]


def q(db, sql, params=()):
    con = duckdb.connect(db, read_only=True)
    try:
        return con.execute(sql, list(params)).fetchall()
    finally:
        con.close()


def mol(db, read):
    return q(db, "SELECT molecule_id FROM molecule_member WHERE read_name = ?", [read])[0][0]


def member(db, read):
    cols = "molecule_state, train_tiles_excluded, exclusion_cause, superseded_version, unit_rule, part_index, accession_stem"
    return q(db, f"SELECT {cols} FROM molecule_member WHERE read_name = ?", [read])[0]


# ------------------------------------------------------------------------------ molecule units (A11 / A16)
def test_est_molecule_units_versions_parts_signature_and_clone(est_db):
    db, _ = est_db
    assert mol(db, "e1") == mol(db, "e2")                       # accession versions: one molecule
    assert member(db, "e1")[3] is True and member(db, "e2")[3] is False
    assert mol(db, "e3") == mol(db, "e4")                       # A21 _partN pieces: one molecule
    assert member(db, "e3")[5:] == (1, "EB2") and member(db, "e4")[5] == 2
    assert mol(db, "e5") == mol(db, "e6") and member(db, "e5")[4] == "signature"
    assert mol(db, "e7") != mol(db, "e5")                       # other library, same signature: never collapsed
    assert mol(db, "e8") == mol(db, "e9") and member(db, "e8")[4] == "clone"   # 5'/3' reads of one clone
    blocks = q(db, "SELECT count(*) FROM aligned_block b JOIN evidence_alignment a USING (alignment_id) "
                   "WHERE a.read_name = 'e1'")[0][0]
    assert blocks == 0                                           # superseded version supplies no observation


def test_ont_pcr_equivalence_units(ont_db):
    assert mol(ont_db, "o1") == mol(ont_db, "o2") == mol(ont_db, "o4")   # same chain, ends within 10 nt
    assert mol(ont_db, "o3") != mol(ont_db, "o1")                           # start 20 nt away
    assert member(ont_db, "o1")[4] == "pcr_equivalence"


def test_ont_nonclique_pcr_component_is_one_molecule(world, tmp_path):
    """A16, author decision 9(a) (2026-09-26): a connected component of the 10-nt PCR-equivalence graph within one
    library/contig/strand/chain key is ONE molecule whether or not it is a clique.  Before: RuleUnresolved
    'A16 non-transitive 10-nt PCR equivalence' aborted the whole build (pronghorn job 6172567)."""
    W.write_ont(tmp_path)
    W.write_ont_nonclique(tmp_path)
    roles = W.write_roles(tmp_path, extra_rows=[W.ONT_NONCLIQUE_ROLE_ROW])
    rc, args = _run(world, tmp_path, source="ont", roles=roles)
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert mol(db, "n1") == mol(db, "n2") == mol(db, "n3")
    assert mol(db, "n4") == mol(db, "n5") != mol(db, "n1")
    assert {member(db, r)[4] for r in ("n1", "n2", "n3", "n4", "n5")} == {"pcr_equivalence"}
    assert mol(db, "o1") == mol(db, "o2") == mol(db, "o4") != mol(db, "o3")     # existing cliques unchanged
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["pcr_nonclique_components"] == 1 and qc["pcr_nonclique_reads"] == 3
    prov = json.loads(q(db, "SELECT value FROM c0_provenance WHERE key = 'a16_pcr_linkage'")[0][0])
    assert prov["linkage"] == "connected_component" and prov["date"] == "2026-09-26"


def test_ont_clique_only_build_records_zero_nonclique_counters(ont_db):
    rows = dict(q(ont_db, "SELECT metric, value FROM c0_qc WHERE metric LIKE 'pcr_nonclique_%'"))
    assert rows == {"pcr_nonclique_components": 0.0, "pcr_nonclique_reads": 0.0}


# ------------------------------------------------------------------------------ rule 4 states and QC
def test_rule_4_states_per_cause(est_db):
    db, _ = est_db
    expect = {"h1": "heldout", "t1": "test_orthogroup", "f1": "a22_hard", "z1": "test_block", "s1": "heldout",
              "m1": "b_low_mapq"}
    for read, cause in expect.items():
        state, _, causes, *_ = member(db, read)
        assert state == "excluded_heldout" and cause in causes.split(","), (read, causes)
    for read in ("e2", "e5", "e7", "w1", "n1", "q1"):
        assert member(db, read)[:2] == ("train_track", False), read
    assert member(db, "v1")[:2] == ("train_track", True)        # valid orthogroup: train tiles only
    assert member(db, "g1")[:2] == ("train_track", True)        # edge-crossing gL span in a train tile
    direct = q(db, "SELECT cause_a_direct FROM molecule_exclusion WHERE molecule_id = ?", [mol(db, "s1")])[0][0]
    assert direct is True                                        # frozen supplementary placement at gH
    assert q(db, "SELECT mapping_ambiguous, rejection_reason FROM evidence_alignment "
                 "WHERE read_name = 'm1' AND placement_kind = 'primary'")[0] == (True, "mapping_ambiguous_family_only")


def test_qc_json_and_athaliana_heldout_identity(est_db):
    db, record = est_db
    qc = record["qc"]
    assert record["status"] == "PASS" and qc["leakage_nonzero"] == 0
    assert qc["heldout_overlapping_molecules"] == qc["heldout_overlapping_excluded"] == qc["gate_heldout_overlapping"] == 2
    assert qc["heldout_retained"] == qc["gate_heldout_retained"] == 0
    assert qc["molecules_in"] == qc["retained"] + qc["excluded_exclusive_a"] + qc["excluded_exclusive_b_secondary"] \
        + qc["excluded_exclusive_b_low_mapq"]
    assert qc["excluded_b_low_mapq"] == 1 and qc["excluded_c_molecules"] == 1 and qc["exhaustiveness"] == 1.0
    stored = dict(q(db, "SELECT metric, value FROM c0_qc"))
    assert stored["leakage_nonzero"] == 0 and stored["excluded_a_cause_heldout"] == 2


def test_rule_4c_flags_junction_support_of_a_retained_molecule(est_db):
    db, _ = est_db
    rows = q(db, "SELECT s.excluded_4c, s.molecule_state, j.n_molecules, j.n_molecules_train_track FROM junction_support s "
                 "JOIN junction j USING (junction_key, dataset, run, library, role) WHERE s.molecule_id = ?", [mol(db, "v1")])
    assert rows == [(True, "train_track", 1, 0)]
    assert q(db, "SELECT count(*) FROM excluded_junction_coordinate WHERE coordinate0 = 100300 AND kind = 'donor'")[0][0] == 1


def test_every_row_carries_role_and_the_build_provenance_hash(est_db):
    db, record = est_db
    prov = record["provenance_sha256"]
    for table in ("evidence_alignment", "molecule_member", "aligned_block", "junction_observation", "partial_chain",
                  "partial_chain_intron", "junction", "junction_support", "chain_junction"):
        rows = q(db, f"SELECT DISTINCT role, validation_only, provenance_sha256, genotype_stratum FROM {table}")
        assert rows == [("training_eligible", False, prov, "reference")], table
    assert q(db, "SELECT value FROM c0_provenance WHERE key = 'provenance_sha256'")[0][0] == prov


# ------------------------------------------------------------------------------ junctions and chains
def test_est_junction_statuses_uncorrected_noncanonical_and_unknown_strand(est_db):
    db, _ = est_db
    js = q(db, "SELECT o.correction_status, o.accepted, o.correction_radius, o.strand, o.donor0 FROM junction_observation o "
               "JOIN evidence_alignment a USING (alignment_id) WHERE a.read_name = ? ORDER BY o.intron_index", ["e2"])
    assert js == [("uncorrected", True, 0, "+", 10200), ("uncorrected", True, 0, "+", 10500)]
    assert q(db, "SELECT o.correction_status, o.accepted FROM junction_observation o JOIN evidence_alignment a "
                 "USING (alignment_id) WHERE a.read_name = 'n1'") == [("training_noncanonical", False)]
    assert q(db, "SELECT o.correction_status, o.strand, o.donor0, o.acceptor0, o.junction_key FROM junction_observation o "
                 "JOIN evidence_alignment a USING (alignment_id) WHERE a.read_name = 'q1'") == \
        [("strand_ambiguous", "?", None, None, None)]


def test_ont_correction_raw_and_corrected_and_ambiguous(ont_db):
    rows = q(ont_db, "SELECT a.read_name, o.raw_start0, o.raw_end0, o.start0, o.end0, o.correction_status, o.accepted, "
                     "o.correction_radius FROM junction_observation o JOIN evidence_alignment a USING (alignment_id) "
                     "WHERE a.read_name IN ('o1', 'o4', 'o5') ORDER BY a.read_name")
    assert rows == [("o1", 170200, 170300, 170200, 170300, "canonical", True, 3),
                    ("o4", 170202, 170302, 170200, 170300, "corrected", True, 3),
                    ("o5", 185200, 185300, None, None, "ambiguous_correction", False, 3)]
    # corrected o4 and canonical o1 are one junction; their PCR unit counts once
    assert q(ont_db, "SELECT n_molecules, n_observations FROM junction WHERE start0 = 170200 AND end0 = 170300") == [(2, 4)]


def test_chain_coordinates_are_converted_once_and_strand_aware(est_db):
    db, _ = est_db
    # minus-strand t1: 1-based intron 70301-70400 -> [70300, 70400), donor0 = last base 70399, acceptor0 = 70300
    assert q(db, "SELECT i.start0, i.end0, i.donor0, i.acceptor0, i.tx_order, c.strand, c.n_introns FROM partial_chain_intron i "
                 "JOIN partial_chain c USING (chain_id) JOIN evidence_alignment a USING (alignment_id) "
                 "WHERE a.read_name = 't1'") == [(70300, 70400, 70399, 70300, 0, "-", 1)]
    plus = q(db, "SELECT i.intron_index, i.tx_order, i.start0, i.end0, i.donor0, i.acceptor0 FROM partial_chain_intron i "
                 "JOIN partial_chain c USING (chain_id) JOIN evidence_alignment a USING (alignment_id) "
                 "WHERE a.read_name = 'e2' ORDER BY i.intron_index")
    assert plus == [(0, 0, 10200, 10300, 10200, 10299), (1, 1, 10500, 10600, 10500, 10599)]
    assert q(db, "SELECT start0, end0 FROM evidence_alignment WHERE read_name = 'e2'") == [(10000, 10800)]
    blocks = q(db, "SELECT block_index, start0, end0 FROM aligned_block WHERE molecule_id = ? ORDER BY block_index",
               [mol(db, "e2")])
    assert blocks == [(0, 10000, 10200), (1, 10300, 10500), (2, 10600, 10800)]
    assert q(db, "SELECT count(*) FROM chain_junction j JOIN partial_chain c USING (chain_id) JOIN evidence_alignment a "
                 "USING (alignment_id) WHERE a.read_name = 'e2'")[0][0] == 2


# ------------------------------------------------------------------------------ fail-closed gates
def _run(world, tmp, **kw):
    args = W.cli_args(world, tmp, **kw)
    return B.main(args), args


def test_missing_role_row_refused(world, tmp_path, capsys):
    W.write_est(tmp_path)
    rc, args = _run(world, tmp_path, roles=W.write_roles(tmp_path, drop_est=True))
    assert rc == 1 and "A18.3" in capsys.readouterr().err
    assert not (tmp_path / f"{W.SPECIES}.est.c0.db").exists()


def test_input_hash_mismatch_refused(world, tmp_path, capsys):
    W.write_est(tmp_path, done_md5_ok=False)
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "hash mismatch" in capsys.readouterr().err


def test_validation_only_rows_load_only_as_validation_only_without_audit(world, tmp_path):
    W.write_est(tmp_path, audit=False)
    rc, args = _run(world, tmp_path, roles=W.write_roles(tmp_path, est_role="b1_validation_only"))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT DISTINCT role, validation_only, molecule_state FROM molecule_member") == \
        [("validation_only", True, "never")]
    assert q(db, "SELECT count(*) FROM molecule_exclusion")[0][0] == 0


def test_never_training_species_and_excluded_roles(tmp_path):
    rows = [{"dataset": "est/Zmays", "run": "est.fa.gz", "species": "Zmays", "role": "c2_training_eligible"}]
    with pytest.raises(B.C0Error, match="rule 1"):
        B.select_runs(rows, "Zmays", "est")
    rows = [{"dataset": "est/Gmax", "run": "est.fa.gz", "species": "Gmax", "role": "excluded"}]
    with pytest.raises(B.C0Error, match="A18.3"):
        B.select_runs(rows, "Gmax", "est")


def test_unroled_alignment_directory_refused(world, tmp_path, capsys):
    W.write_ont(tmp_path)
    W.write_ont(tmp_path, run="SRR_NOROLE")
    rc, _ = _run(world, tmp_path, source="ont")
    assert rc == 1 and "no DATASET_ROLES row" in capsys.readouterr().err


def test_tool_identity_and_audit_command_checked(world, tmp_path, capsys):
    W.write_est(tmp_path, audit_cl=W.AUDIT_CL.replace("-N 50", "-N 5"))
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "-N 50" in capsys.readouterr().err
    shutil.rmtree(tmp_path / "align"), shutil.rmtree(tmp_path / "audit")
    W.write_est(tmp_path)
    args = W.cli_args(world, tmp_path, out_name="x.db")
    W.write_provenance(tmp_path, version="2.99")
    args[args.index("--provenance") + 1] = str(tmp_path / "PROVENANCE.cpu_tools_v1.json")
    assert B.main(args) == 1 and "minimap2 VN" in capsys.readouterr().err


def test_sensitivity_arm_is_never_c0_input(world, tmp_path, capsys):
    W.write_est(tmp_path)
    done = tmp_path / "align" / W.SPECIES / "DONE"
    done.write_text(done.read_text().replace("arm=primary", "arm=sensitivity"))
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "A37 primary arm" in capsys.readouterr().err


def test_capped_molecule_below_the_per_source_floor_aborts_the_build(world, tmp_path, capsys):
    """Fix round 6, item 3 (author decision 2026-09-23: BOTH the species-level rule 4(b) gate AND a per-source floor).
    One capped molecule among the 17 EST molecules (16/17 = 94.1 %) is below the per-source floor
    (leakage.MIN_SOURCE_EXHAUSTIVENESS = 0.99, A49.2), so the build
    aborts fail-closed: exit 1, ExhaustivenessError text, status ABORT, nothing published.  (Fix round 5 had made this
    a merge-only species check; the species sum is still enforced there.)"""
    extra = dict(W.EST_EXTRA_AUDIT)
    extra["e1"] = [W.sam("e1", "Chr1", [(180001 + 10 * i, 180050 + 10 * i)], flag=256, mapq=0) for i in range(50)]
    W.write_est(tmp_path, audit_extra=extra)
    rc, args = _run(world, tmp_path)
    err = capsys.readouterr().err
    assert rc == 1 and "per-source audit exhaustiveness 0.941176 < 0.99" in err and "raise N" in err
    assert "Athaliana/est" in err
    record = json.load(open(args[args.index("--qc") + 1]))
    assert record["status"] == "ABORT" and record["output"] is None
    assert not Path(args[args.index("--out") + 1]).exists()
    assert LK.MIN_SOURCE_EXHAUSTIVENESS == 0.99 and LK.MIN_EXHAUSTIVENESS == 0.999
    # Codex review 4 LOW: the counters survive the abort (report() is recorded before check() raises).
    qc = record["qc"]
    assert (qc["exhaustiveness_molecules"], qc["exhaustiveness_below_cap"], qc["audit_n"]) == (17, 16, 50)


def test_exhaustiveness_counters_and_both_rules_are_recorded(est_db):
    """Item 3: a passing build records the per-source numerator / denominator (for the merge's species sum and its
    per-source re-check) and the provenance names both rules."""
    db, record = est_db
    qc = record["qc"]
    assert qc["exhaustiveness_molecules"] == qc["exhaustiveness_below_cap"] == qc["molecules_in"] == 17
    assert qc["exhaustiveness"] == 1.0 and qc["audit_n"] == 50
    stored = dict(q(db, "SELECT metric, value FROM c0_qc"))
    assert (stored["exhaustiveness_below_cap"], stored["exhaustiveness_molecules"], stored["audit_n"]) == (17, 17, 50)
    rule = json.loads(q(db, "SELECT value FROM c0_provenance WHERE key = 'exhaustiveness_rule'")[0][0])
    assert rule["scope"] == "per_source_floor_and_species"
    assert rule["rule"] == "per-source floor 0.99 (A49 decision 2 / A49.2) + species >= 0.999 (rule 4(b))"
    assert (rule["min_exhaustiveness"], rule["min_source_exhaustiveness"]) == (0.999, 0.99)


def test_audit_must_cover_every_member(world, tmp_path, capsys):
    W.write_est(tmp_path)
    audit = W.est_audit_path(tmp_path / "audit")
    audit.write_text("\n".join(l for l in audit.read_text().splitlines() if not l.startswith("e5\t")) + "\n")
    (audit.parent / "DONE").write_text(f"sam_md5={W.md5(audit)}\nsource=est\nrun=est_primary\n")
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "audit alignment lacks 1 member" in capsys.readouterr().err


def test_output_is_exclusive(world, tmp_path):
    W.write_est(tmp_path)
    (tmp_path / f"{W.SPECIES}.est.c0.db").write_text("x")
    rc, _ = _run(world, tmp_path)
    assert rc == 1


# ------------------------------------------------------------------------------ protein (miniprot) + route B
def _paf(q, n_introns, qlen=200, qend=180):
    cs = ":50" + "".join("~gt100ag:50" for _ in range(n_introns))
    return f"##PAF\t{q}\t{qlen}\t0\t{qend}\t+\tChr1\t250000\t0\t1\t1\t1\t60\tfs:i:0\tcs:Z:{cs}"


def _mrna(i, q, chrom, strand, score, cds):
    lines = [f"{chrom}\tminiprot\tmRNA\t{cds[0][0]}\t{cds[-1][1]}\t{score}\t{strand}\t.\tID=MP{i};Identity=0.9;Target={q} 1 180"]
    lines += [f"{chrom}\tminiprot\tCDS\t{s}\t{e}\t{score}\t{strand}\t0\tParent=MP{i}" for s, e in cds]
    return lines


def write_protein(tmp):
    d = tmp / "align" / W.SPECIES
    d.mkdir(parents=True, exist_ok=True)
    lines = ["##gff-version 3"]
    lines += [_paf("4577_0:000001", 1)] + _mrna(1, "4577_0:000001", "Chr1", "+", 500, [(170001, 170200), (170301, 170500)])
    lines += [_paf("4577_0:000002", 1)] + _mrna(2, "4577_0:000002", "Chr1", "+", 400, [(40001, 40200), (40301, 40500)])
    lines += [_paf("39947_0:000003", 1)] + _mrna(3, "39947_0:000003", "Chr1", "+", 300, [(170001, 170200), (170301, 170500)])
    lines += [_paf("39947_0:000003", 0)] + _mrna(4, "39947_0:000003", "Chr1", "+", 300, [(10001, 10200)])
    gff = d / f"{W.SPECIES}.protein.gff"
    gff.write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text(f"gff_md5          {W.md5(gff)}\n")


def write_hsp(tmp, molecules, protein_species="4577", rows=None, state=None):
    """Route-B rows (part-B contract).  ``molecules``: (retained EST molecule, held-out EST molecule) of the EST C0 DB."""
    from transgenic.evidence import c0_schema as S

    keep, held = molecules
    rows = rows or [
        ["h1", W.SPECIES, "L1", keep, "EA1.2", "4577_0:000001", "orthodb", protein_species, "0", "150", "1", "1e-20",
         "200", "0", "Chr1", "10000", "10150", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64],
        ["h1", W.SPECIES, "L1", keep, "EA1.2", "4577_0:000001", "orthodb", protein_species, "0", "150", "1", "1e-20",
         "200", "1", "Chr1", "10300", "10400", "+", "2", "2.1.24", "d" * 64, "c" * 64, "p" * 64],
        ["h2", W.SPECIES, "L1", held, "EH1", "4577_0:000009", "swissprot", "39947", "0", "150", "1", "1e-20",
         "200", "0", "Chr1", "40000", "40150", "+", "1", "2.1.24", "d" * 64, "c" * 64, "p" * 64]]
    p = tmp / "hsp.tsv"
    p.write_text("\t".join(S.HSP_COLUMN_NAMES) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    if state is not None:          # b6_hsp_project.py's <out>.molecule_state.tsv (cross-check sidecar)
        seen = {}
        for r in rows:
            seen.setdefault(r[0], [r[0], r[3], state[r[3]][0], state[r[3]][1], state[r[3]][2]])
        (tmp / "hsp.tsv.molecule_state.tsv").write_text(
            "hsp_id\tmolecule_id\tqname\tmolecule_state\ttrain_tiles_excluded\n"
            + "".join("\t".join(v) + "\n" for v in seen.values()))
    return p


def est_state(est_db):
    """molecule_id -> (a member QNAME, molecule_state, 'True'/'False') from the EST C0 DB, as part B passes it through."""
    rows = q(est_db[0], "SELECT molecule_id, min(read_name), any_value(molecule_state), bool_or(train_tiles_excluded) "
                        "FROM molecule_member GROUP BY molecule_id")
    return {m: (r, st, str(t)) for m, r, st, t in rows}


def est_alignment_sha256(est_db):
    inputs = json.loads(q(est_db[0], "SELECT value FROM c0_provenance WHERE key = 'inputs'")[0][0])
    (sha,) = [v["sha256"] for k, v in inputs.items() if k.startswith("alignment:")]
    return sha


def write_hsp_provenance(hsp, est_db, **override):
    """b6_hsp_project.py's <out>.provenance.json (the fields b6_build_c0 verifies) and every row's provenance_sha256."""
    import hashlib
    from transgenic.evidence import c0_schema as S

    prov = dict(schema="est_protein_hsp_v1", species=W.SPECIES, phase_convention=S.PHASE_CONVENTION,
                alignment_sha256=est_alignment_sha256(est_db), multi_hsp_rule_applied_by=B.MULTI_HSP_APPLIED_BY)
    prov.update(override)
    blob = json.dumps(prov, sort_keys=True, separators=(",", ":")).encode()
    (hsp.parent / (hsp.name + ".provenance.json")).write_bytes(blob + b"\n")
    digest = hashlib.sha256(blob).hexdigest()
    lines = hsp.read_text().splitlines()
    hsp.write_text("\n".join([lines[0]] + [l.rsplit("\t", 1)[0] + "\t" + digest for l in lines[1:]]) + "\n")
    return digest


def _protein_extra(tmp, est_db, hsp, **override):
    write_hsp_provenance(hsp, est_db, **override)
    return ("--hsp", str(hsp), "--est-c0", est_db[0], "--lineage", str(W.write_lineage(tmp)))


def _mols(est_db):
    return mol(est_db[0], "e2"), mol(est_db[0], "h1")


def test_protein_source_and_route_b_hsp(world, est_db, tmp_path):
    write_protein(tmp_path)
    rc, args = _run(world, tmp_path, source="protein", extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), state=est_state(est_db))))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert member(db, "4577_0:000001")[0] == "train_track"
    assert member(db, "4577_0:000002")[0] == "excluded_heldout"
    amb = q(db, "SELECT placement_kind, accepted, rejection_reason FROM evidence_alignment WHERE read_name = '39947_0:000003' "
                "ORDER BY record_index")
    assert amb == [("primary", False, "mapping_ambiguous_family_only"), ("secondary", False, "mapping_ambiguous_family_only")]
    assert q(db, "SELECT DISTINCT library, genotype_stratum FROM molecule_member WHERE read_name = '4577_0:000001'") == \
        [("4577_0", "not_applicable")]
    assert q(db, "SELECT o.start0, o.end0, o.donor0, o.acceptor0, o.correction_status FROM junction_observation o JOIN evidence_alignment a "
                 "USING (alignment_id) WHERE a.read_name = '4577_0:000001'") == [(170200, 170300, 170200, 170299, "uncorrected")]
    hsp = q(db, "SELECT hsp_id, interval_index, phase, leakage_excluded, train_tiles_excluded, role FROM est_protein_hsp "
                "ORDER BY hsp_id, interval_index")
    assert hsp == [("h1", 0, 0, False, False, "training_eligible"), ("h1", 1, 2, False, False, "training_eligible"),
                   ("h2", 0, 1, True, True, "training_eligible")]   # gH is also tile-leak masked
    assert q(db, "SELECT value FROM c0_provenance WHERE key = 'phase_convention'") == [('"codon_position_at_genomic_start0_v1"',)]


def test_hsp_inherits_the_est_molecule_train_tiles_flag(world, est_db, tmp_path):
    write_protein(tmp_path)
    v1 = mol(est_db[0], "v1")                                   # valid-orthogroup EST: train_track, train tiles excluded
    rows = [["hv", W.SPECIES, "L1", v1, "EV1", "4577_0:000001", "orthodb", "4577", "0", "150", "1", "1e-20", "200", "0",
             "Chr1", "100000", "100100", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64]]   # inside v1's accepted primary
    rc, args = _run(world, tmp_path, source="protein",
                    extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    assert rc == 0
    assert q(args[args.index("--out") + 1], "SELECT leakage_excluded, train_tiles_excluded FROM est_protein_hsp") == [(False, True)]


def test_orphan_hsp_molecule_is_refused(world, est_db, tmp_path, capsys):
    write_protein(tmp_path)
    rc, _ = _run(world, tmp_path, source="protein",
                 extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, ("f" * 64, mol(est_db[0], "h1")),
                                                                  state=dict(est_state(est_db), **{"f" * 64: ("x", "train_track", "False")}))))
    assert rc == 1 and "reference no molecule" in capsys.readouterr().err


def test_hsp_state_sidecar_must_agree_with_the_est_c0(world, est_db, tmp_path, capsys):
    write_protein(tmp_path)
    state = est_state(est_db)
    keep = mol(est_db[0], "e2")
    state[keep] = (state[keep][0], "excluded_heldout", state[keep][2])        # sidecar claims a different state
    rc, _ = _run(world, tmp_path, source="protein",
                 extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), state=state)))
    assert rc == 1 and "disagrees with the EST C0 molecule_member" in capsys.readouterr().err


@pytest.mark.parametrize("override,message", [
    (dict(alignment_sha256="0" * 64), "alignment_sha256"),
    (dict(phase_convention="codon_position_at_genomic_end0_v0"), "phase_convention"),
    (dict(species="Osativa"), "species"),
])
def test_hsp_provenance_must_match_the_est_c0_and_the_schema(world, est_db, tmp_path, capsys, override, message):
    write_protein(tmp_path)
    hsp = write_hsp(tmp_path, _mols(est_db), state=est_state(est_db))
    rc, _ = _run(world, tmp_path, source="protein", extra=_protein_extra(tmp_path, est_db, hsp, **override))
    err = capsys.readouterr().err
    assert rc == 1 and message in err and "provenance.json" in err


def test_hsp_rows_must_cite_their_provenance_record(world, est_db, tmp_path, capsys):
    write_protein(tmp_path)
    hsp = write_hsp(tmp_path, _mols(est_db), state=est_state(est_db))
    extra = _protein_extra(tmp_path, est_db, hsp)
    prov = tmp_path / "hsp.tsv.provenance.json"
    prov.write_bytes(prov.read_bytes().replace(b"est_protein_hsp_v1", b"est_protein_hsp_v2"))   # rows now cite another record
    rc, _ = _run(world, tmp_path, source="protein", extra=extra)
    assert rc == 1 and "cite provenance" in capsys.readouterr().err
    prov.unlink()
    rc, _ = _run(world, tmp_path, source="protein", out_name="p2.db", extra=extra)
    assert rc == 1 and "provenance record" in capsys.readouterr().err


def test_est_done_without_arm_fails_closed(world, tmp_path, capsys):
    W.write_est(tmp_path)
    done = tmp_path / "align" / W.SPECIES / "DONE"
    done.write_text(done.read_text().replace("arm=primary\n", ""))
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "records no arm=" in capsys.readouterr().err


def test_audit_n_comes_from_the_audit_header_not_a_done_key(world, tmp_path, capsys):
    W.write_est(tmp_path, audit_cl=W.AUDIT_CL.replace("-N 50", "-N 60"))
    rc, _ = _run(world, tmp_path)                               # DONE has no audit_n; the header says N=60 != 50
    assert rc == 1 and "-N 50" in capsys.readouterr().err


def test_seed_is_required(world, tmp_path, capsys):
    W.write_est(tmp_path)
    args = W.cli_args(world, tmp_path)
    i = args.index("--seed")
    with pytest.raises(SystemExit):
        B.main(args[:i] + args[i + 2:])
    assert "--seed" in capsys.readouterr().err


def test_protein_requires_hsp_est_c0_and_lineage(world, tmp_path, capsys):
    write_protein(tmp_path)
    rc, _ = _run(world, tmp_path, source="protein")
    assert rc == 1 and "requires --hsp, --est-c0 and --lineage" in capsys.readouterr().err


def test_same_species_hsp_and_hsp_outside_protein_refused(world, est_db, tmp_path, capsys):
    write_protein(tmp_path)
    rc, _ = _run(world, tmp_path, source="protein",
                 extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), protein_species="3702", state=est_state(est_db))))
    assert rc == 1 and "same-species" in capsys.readouterr().err
    rc, _ = _run(world, tmp_path, out_name="descendant.db",           # a subtaxon of A. thaliana: refused by lineage
                 source="protein", extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), protein_species="999001", state=est_state(est_db))))
    assert rc == 1 and "same-species" in capsys.readouterr().err
    W.write_est(tmp_path)
    rc, _ = _run(world, tmp_path, out_name="e.db", extra=("--hsp", str(tmp_path / "hsp.tsv")))
    assert rc == 1 and "--hsp is only valid" in capsys.readouterr().err


def test_miniprot_query_of_the_athaliana_lineage_is_refused(world, est_db, tmp_path, capsys):
    write_protein(tmp_path)
    gff = tmp_path / "align" / W.SPECIES / f"{W.SPECIES}.protein.gff"
    gff.write_text(gff.read_text().replace("39947_0:000003", "3702_0:000003"))
    (gff.parent / "DONE").write_text(f"gff_md5          {W.md5(gff)}\n")
    rc, _ = _run(world, tmp_path, source="protein", extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), state=est_state(est_db))))
    assert rc == 1 and "miniprot query 3702_0:000003: same-species protein (A19)" in capsys.readouterr().err


def test_protein_at_the_miniprot_cap_is_unauditable(world, est_db, tmp_path):
    d = tmp_path / "align" / W.SPECIES
    d.mkdir(parents=True, exist_ok=True)
    lines = ["##gff-version 3"]
    lines += [_paf("4577_0:000001", 1)] + _mrna(1, "4577_0:000001", "Chr1", "+", 500, [(170001, 170200), (170301, 170500)])
    for k in range(21):                                          # 21 placements = the -N 20 cap
        lines += [_paf("4577_0:000050", 0)] + _mrna(100 + k, "4577_0:000050", "Chr1", "+", 400 - k, [(160001 + 100 * k, 160050 + 100 * k)])
    gff = d / f"{W.SPECIES}.protein.gff"
    gff.write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text(f"gff_md5          {W.md5(gff)}\n")
    rc, args = _run(world, tmp_path, source="protein", extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), state=est_state(est_db))))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert member(db, "4577_0:000050")[:3] == ("excluded_heldout", False, "b_secondary_cap")
    assert member(db, "4577_0:000001")[0] == "train_track"
    record = json.load(open(args[args.index("--qc") + 1]))
    assert record["qc"]["protein_excluded_b_miniprot_cap"] == 1 == record["qc"]["protein_alignments_at_miniprot_cap_21"]


def test_est_molecule_id_matches_the_c0lite_and_route_b_formula(est_db):
    import hashlib
    import json as _json

    db, _ = est_db
    want = hashlib.sha256(_json.dumps(["e1", "e2"], separators=(",", ":")).encode()).hexdigest()
    assert mol(db, "e1") == want


# ------------------------------------------------------------------------------ review fixes (no silent defaults, gates)
def test_read_missing_from_metadata_fails_closed(world, tmp_path, capsys):
    W.write_est(tmp_path)
    d = tmp_path / "align" / W.SPECIES
    meta = d / "c0_metadata.tsv"
    meta.write_text("\n".join(l for l in meta.read_text().splitlines() if not l.startswith("e7\t")) + "\n")
    (d / "c0_metadata.tsv.sha256").write_text(f"{W.sha(meta)}  c0_metadata.tsv\n")
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "lacks 1 read(s)" in capsys.readouterr().err and not (tmp_path / f"{W.SPECIES}.est.c0.db").exists()


def test_audit_cl_must_differ_from_the_frozen_cl_only_by_the_secondary_flags(world, tmp_path, capsys):
    W.write_est(tmp_path, audit_cl=W.AUDIT_CL.replace("-G 200000", "-G 100000"))
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "must equal the frozen primary CL" in capsys.readouterr().err


def test_legacy_est_audit_layout_is_not_accepted(world, tmp_path, capsys):
    W.write_est(tmp_path)
    audit = W.est_audit_path(tmp_path / "audit")
    legacy = tmp_path / "audit" / W.SPECIES / f"{W.SPECIES}.est.sam"
    legacy.write_bytes(audit.read_bytes())
    (legacy.parent / "DONE").write_text(f"sam_md5={W.md5(legacy)}\naudit_n=50\n")
    shutil.rmtree(audit.parent)
    rc, _ = _run(world, tmp_path)
    assert rc == 1 and "est_primary.audit" in capsys.readouterr().err


def test_step7_gate_retests_written_rows_independently(est_db, tmp_path):
    """The gate reads rows and intervals back from the DB: a train_track aligned block planted at the held-out
    locus is caught although no audit_placement row mentions it."""
    import collections

    db = tmp_path / "g.db"
    shutil.copy(est_db[0], db)
    con = duckdb.connect(str(db))
    con.execute("""INSERT INTO aligned_block SELECT * REPLACE (40050 AS start0, 40100 AS end0, 99 AS alignment_block_index)
                   FROM aligned_block WHERE molecule_id = (SELECT molecule_id FROM molecule_member WHERE read_name = 'e2') LIMIT 1""")
    ctx = {"qc": collections.Counter(est_db[1]["qc"])}
    B.sql_gate(con, ctx)
    con.close()
    assert ctx["qc"]["gate_train_track_block_rows_in_excluded_all"] == 1
    assert ctx["qc"]["gate_train_track_block_rows_at_heldout_locus"] == 1
    assert ctx["qc"]["gate_retained_overlapping_excluded_all"] == 0      # the audit_placement view would miss it
    assert ctx["qc"]["leakage_nonzero"] >= 2


def test_qc_counts_the_effect_of_the_pending_rulings(est_db):
    qc = est_db[1]["qc"]
    assert qc["excluded_b_cause_secondary_mapq"] == 1                  # m1: MAPQ 0 on an audit secondary only
    assert qc["excluded_a_cause_edge_partial"] == 0 and qc["train_tiles_cause_edge_partial"] == 1   # g1 in gL's span


# ------------------------------------------------------------------------------ fix round 4 (Codex WP1 findings)
def test_rejected_pacbio_member_closes_its_a16_molecule(world, tmp_path):
    """Finding 1 (A46 rule 4(b) 'any placement of any member', rule 3 / A16 'one source ZMW molecule = one unit'):
    p2 (MAPQ 10, rejected) shares the ZMW molecule of p1 (MAPQ 60, clean locus) and touches the held-out gene, so
    the whole molecule -- p1 included -- is excluded_heldout; the leakage QC gate still passes (count 0)."""
    W.write_pacbio(tmp_path)
    roles = W.write_roles(tmp_path, extra_rows=[W.PACBIO_ROLE_ROW])
    rc, args = _run(world, tmp_path, source="pacbio", roles=roles)
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert mol(db, "p1") == mol(db, "p2") != mol(db, "p3")
    assert member(db, "p1")[0] == member(db, "p2")[0] == "excluded_heldout"
    assert member(db, "p3")[0] == "train_track" and member(db, "p1")[4] == member(db, "p2")[4] == "source_molecule"
    assert q(db, "SELECT accepted, rejection_reason FROM evidence_alignment WHERE read_name = 'p2'") == [(False, "mapq")]
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["leakage_nonzero"] == 0 and qc["rejected_members_in_a16_unit"] == 1
    assert qc["heldout_overlapping_molecules"] == qc["heldout_overlapping_excluded"] == 1 and qc["heldout_retained"] == 0


def test_hsp_of_an_excluded_protein_query_is_leakage_excluded(world, est_db, tmp_path):
    """Finding 2 (A46 rule 3 'a molecule is one unit everywhere it appears'; rule 6 'protein queries touching excluded
    intervals are removed molecule-wide'): 4577_0:000002 is excluded_heldout in this protein C0 (its miniprot placement
    is the held-out gene), so its route-B HSP on the retained EST e2 at a clean locus inherits leakage_excluded."""
    write_protein(tmp_path)
    keep = mol(est_db[0], "e2")
    rows = [["hp", W.SPECIES, "L1", keep, "EA1.2", "4577_0:000002", "orthodb", "4577", "0", "150", "1", "1e-20", "200", "0",
             "Chr1", "10000", "10150", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64],
            ["hs", W.SPECIES, "L1", keep, "EA1.2", "SWISS_NOT_ALIGNED", "swissprot", "39947", "0", "150", "1", "1e-20", "200",
             "0", "Chr1", "10000", "10150", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64]]
    rc, args = _run(world, tmp_path, source="protein",
                    extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert member(db, "4577_0:000002")[0] == "excluded_heldout"
    assert q(db, "SELECT hsp_id, leakage_excluded, train_tiles_excluded, exclusion_cause FROM est_protein_hsp ORDER BY hsp_id") == \
        [("hp", True, True, "protein_query"), ("hs", False, False, None)]   # held-out gene = tile cause too; Swiss-Prot: never miniprot-aligned
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["hsp_rows_excluded_by_protein_query_only"] == 1 and qc["hsp_rows_excluded"] == 1
    assert qc["hsp_rows_train_tiles_excluded_by_protein_query_only"] == 1


def test_hsp_projected_through_a_rejected_est_placement_is_refused(world, est_db, tmp_path, capsys):
    """Finding 3 (A46 rule 6 'through the EST's spliced alignment ... from retained observations', A18.5): u1 is a
    member of a train_track molecule but its C0 primary was never accepted (unmapped); an HSP citing it fails closed."""
    write_protein(tmp_path)
    u1 = mol(est_db[0], "u1")
    assert member(est_db[0], "u1")[0] == "train_track"
    assert q(est_db[0], "SELECT accepted FROM evidence_alignment WHERE read_name = 'u1'") == [(False,)]
    rows = [["hu", W.SPECIES, "L1", u1, "EU1", "4577_0:000001", "orthodb", "4577", "0", "150", "1", "1e-20", "200", "0",
             "Chr1", "10000", "10150", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64]]
    rc, _ = _run(world, tmp_path, source="protein",
                 extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    err = capsys.readouterr().err
    assert rc == 1 and "ACCEPTED primary EST placement" in err and "hu" in err
    # An interval outside the accepted placement of an accepted qname is refused the same way.
    keep = mol(est_db[0], "e2")
    rows = [["ho", W.SPECIES, "L1", keep, "EA1.2", "4577_0:000001", "orthodb", "4577", "0", "150", "1", "1e-20", "200", "0",
             "Chr1", "170000", "170100", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64]]
    rc, _ = _run(world, tmp_path, source="protein", out_name="p2.db",
                 extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    assert rc == 1 and "ACCEPTED primary EST placement" in capsys.readouterr().err


# ------------------------------------------------------------------------------ fix round 5 (Codex re-review)
def test_unknown_strand_umi_siblings_share_one_molecule_and_close_together(world, tmp_path):
    """Finding 1.  A16 'UMI when present' = the unit; rule 4(b) 'any placement of any member'.  k1 (accepted, clean)
    and k2 (MAPQ 10 on the held-out gene, rejected) share library / sample / UMI / locus and carry no strand evidence:
    59.assign_molecules keys the UMI unit on the transcript strand, so the accepted path (strand None) and the rejected
    path (before: '?') hashed to two molecules and k1 stayed train_track.  Both must be one unit, both excluded."""
    W.write_ont(tmp_path)
    W.write_ont_umi(tmp_path)
    roles = W.write_roles(tmp_path, extra_rows=[W.ONT_UMI_ROLE_ROW])
    rc, args = _run(world, tmp_path, source="ont", roles=roles)
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT transcript_strand FROM evidence_alignment WHERE read_name IN ('k1', 'k2') ORDER BY read_name") == \
        [("?",), ("?",)]                                              # schema spelling of 'unknown' -- not the unit key
    assert q(db, "SELECT accepted FROM evidence_alignment WHERE read_name = 'k1'") == [(True,)]
    assert q(db, "SELECT accepted, rejection_reason FROM evidence_alignment WHERE read_name = 'k2'") == [(False, "mapq")]
    assert mol(db, "k1") == mol(db, "k2") and member(db, "k1")[4] == member(db, "k2")[4] == "umi"
    assert member(db, "k1")[0] == member(db, "k2")[0] == "excluded_heldout"
    assert member(db, "o1")[0] == "train_track"
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["leakage_nonzero"] == 0 and qc["rejected_members_in_a16_unit"] == 1 and qc["heldout_retained"] == 0


def test_hsp_phase_conflicts_are_resolved_after_protein_query_exclusion(world, est_db, tmp_path):
    """Finding 2.  Rule 6: "HSPs whose frame is inconsistent within one EST after projection are dropped"; rule 3/6:
    an excluded protein query's HSPs are "removed molecule-wide".  On the retained EST e2, P = 4577_0:000002
    (excluded_heldout: its miniprot placement is the held-out gene) has an HSP in phase 0 over [10000,10150) and
    Q = 4577_0:000001 (train_track) one in phase 1 over the same span.  Before: b6_hsp_project resolved the conflict
    first and dropped BOTH; now P's rows are written flagged (leakage_excluded, cause protein_query) and Q's survive.
    A third train_track query R in phase 2 over Q's span is a real conflict among eligible HSPs: Q and R are dropped
    and counted; P stays flagged."""
    write_protein(tmp_path)
    keep = mol(est_db[0], "e2")

    def row(hsp, protein, phase, s, e, source="orthodb"):
        return [hsp, W.SPECIES, "L1", keep, "EA1.2", protein, source, "4577", "0", "150", "1", "1e-20", "200", "0",
                "Chr1", str(s), str(e), "+", str(phase), "2.1.24", "d" * 64, "c" * 64, "p" * 64]

    rows = [row("hP", "4577_0:000002", 0, 10000, 10150), row("hQ", "4577_0:000001", 1, 10000, 10150)]
    rc, args = _run(world, tmp_path, source="protein",
                    extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT hsp_id, phase, leakage_excluded, exclusion_cause FROM est_protein_hsp ORDER BY hsp_id") == \
        [("hP", 0, True, "protein_query"), ("hQ", 1, False, None)]
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["hsp_dropped_phase_conflict_within_molecule"] == 0 and qc["hsp_rows_excluded_by_protein_query_only"] == 1
    assert qc["hsp_phase_conflict_among_retained_recheck"] == 0
    rows.append(row("hR", "39947_0:000003", 2, 10050, 10120))    # eligible (train_track query), conflicts with Q
    rc, args = _run(world, tmp_path, source="protein", out_name="p2.db",
                    extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT hsp_id, leakage_excluded FROM est_protein_hsp ORDER BY hsp_id") == [("hP", True)]
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["hsp_dropped_phase_conflict_within_molecule"] == 2 == qc["hsp_dropped_phase_conflict_within_molecule_orthodb"]
    assert qc["hsp_rows_dropped_phase_conflict_within_molecule"] == 2 and qc["hsp_rows"] == 1
    prov = json.loads(q(db, "SELECT value FROM c0_provenance WHERE key = 'multi_hsp_rule'")[0][0])
    assert prov["applied_by"] == "b6_build_c0.load_hsp"


def test_projection_that_resolved_conflicts_itself_is_refused(world, est_db, tmp_path, capsys):
    """Finding 2: a route-B provenance record without multi_hsp_rule_applied_by = b6_build_c0.load_hsp (an older
    projection that dropped conflicts before eligibility) fails closed."""
    write_protein(tmp_path)
    hsp = write_hsp(tmp_path, _mols(est_db), state=est_state(est_db))
    rc, _ = _run(world, tmp_path, source="protein",
                 extra=_protein_extra(tmp_path, est_db, hsp, multi_hsp_rule_applied_by="b6_hsp_project._emit"))
    err = capsys.readouterr().err
    assert rc == 1 and "multi_hsp_rule_applied_by" in err and "re-run b6_hsp_project.py" in err


# ------------------------------------------------------------------------------ fix round 6 (Codex review 3)
def test_two_runs_of_one_library_form_one_a16_molecule_and_close_together(world, tmp_path):
    """Finding 1.  A11 (protocol line 558): "A biological library is one independently prepared cDNA library; runs
    ... do not create libraries"; A16: "one source FLNC/ZMW molecule = one unit"; rule 4(b): "any placement of any
    member".  p1 (run SRR2) and p6 (run SRR5) share ZMW m64011/42 of ONE library (--library SRX1000201); p6's audit
    secondary (MAPQ 0) lies on the held-out gene.  Before (library = run): two molecules, p1 stayed train_track.
    Now: one molecule, both excluded_heldout; p3 (other ZMW) stays train_track."""
    W.write_pacbio(tmp_path, reads=[r for r in W.PACBIO_READS if r[0] != "p2"])
    W.write_pacbio(tmp_path, W.PACBIO_RUN2, reads=W.PACBIO_READS2, audit_extra=W.PACBIO_AUDIT_EXTRA2)
    roles = W.write_roles(tmp_path, extra_rows=[W.pacbio_role_row(W.PACBIO_RUN, 2), W.pacbio_role_row(W.PACBIO_RUN2, 1)])
    rc, args = _run(world, tmp_path, source="pacbio", roles=roles)
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT DISTINCT library FROM molecule_member") == [("SRX1000201",)]
    assert q(db, "SELECT read_name, run FROM molecule_member WHERE read_name IN ('p1', 'p6') ORDER BY read_name") == \
        [("p1", "SRR2"), ("p6", "SRR5")]                              # run is kept separately from the library
    assert q(db, "SELECT read_name, accepted FROM evidence_alignment WHERE read_name IN ('p1', 'p6') "
                 "AND placement_kind = 'primary' ORDER BY read_name") == [("p1", True), ("p6", True)]
    assert mol(db, "p1") == mol(db, "p6") != mol(db, "p3")
    assert member(db, "p1")[0] == member(db, "p6")[0] == "excluded_heldout" and member(db, "p3")[0] == "train_track"
    assert member(db, "p1")[2] == member(db, "p6")[2] and member(db, "p1")[2]      # one molecule-wide cause set
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["leakage_nonzero"] == 0 and qc["heldout_retained"] == 0
    assert qc["libraries"] == 1 and qc["libraries_spanning_runs"] == 1 and qc["molecules_spanning_runs"] == 1


def test_two_runs_of_different_libraries_are_the_a11_split(world, tmp_path):
    """Finding 1, the other side: the same reads with a DIFFERENT --library per run are two preparations (A11), so
    the shared ZMW string is two molecules; only p6's molecule closes and p1 stays train_track (documented split)."""
    W.write_pacbio(tmp_path, reads=[r for r in W.PACBIO_READS if r[0] != "p2"])
    W.write_pacbio(tmp_path, W.PACBIO_RUN2, library="SRX1000202", reads=W.PACBIO_READS2, audit_extra=W.PACBIO_AUDIT_EXTRA2)
    roles = W.write_roles(tmp_path, extra_rows=[W.pacbio_role_row(W.PACBIO_RUN, 2), W.pacbio_role_row(W.PACBIO_RUN2, 1)])
    rc, args = _run(world, tmp_path, source="pacbio", roles=roles)
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert mol(db, "p1") != mol(db, "p6")
    assert member(db, "p1")[0] == "train_track" and member(db, "p6")[0] == "excluded_heldout"
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["libraries"] == 2 and qc["libraries_spanning_runs"] == 0 and qc["molecules_spanning_runs"] == 0


def test_hsp_phase_conflicts_are_resolved_within_the_excluded_population_too(world, est_db, tmp_path):
    """Finding 2.  Rule 6: "HSPs whose frame is inconsistent within one EST after projection are dropped".  Since
    round 5 the resolution ran only among non-leakage-excluded HSPs, so an excluded_heldout molecule (h1, the
    held-out gene -- rule-10 evaluation evidence) with two overlapping Swiss-Prot HSPs in phase 0 and 1 had BOTH
    written.  Now each population is resolved separately (never across): hA/hB (excluded population) are both dropped
    and counted under the *_excluded keys; hQ on the retained e2 (training population) is untouched, and hP
    (protein-query-excluded on e2, alone in its population) stays flagged."""
    write_protein(tmp_path)
    keep, held = _mols(est_db)

    def row(hsp, molecule, acc, protein, source, phase, s, e):
        return [hsp, W.SPECIES, "L1", molecule, acc, protein, source, "39947", "0", "150", "1", "1e-20", "200", "0",
                "Chr1", str(s), str(e), "+", str(phase), "2.1.24", "d" * 64, "c" * 64, "p" * 64]

    rows = [row("hA", held, "EH1", "SWISS_A", "swissprot", 0, 40000, 40150),
            row("hB", held, "EH1", "SWISS_B", "swissprot", 1, 40050, 40120),
            row("hP", keep, "EA1.2", "4577_0:000002", "orthodb", 0, 10000, 10150),
            row("hQ", keep, "EA1.2", "4577_0:000001", "orthodb", 1, 10000, 10150)]
    rc, args = _run(world, tmp_path, source="protein",
                    extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=rows, state=est_state(est_db))))
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT hsp_id, phase, leakage_excluded FROM est_protein_hsp ORDER BY hsp_id") == \
        [("hP", 0, True), ("hQ", 1, False)]
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["hsp_dropped_phase_conflict_within_molecule_excluded"] == 2
    assert qc["hsp_dropped_phase_conflict_within_molecule_excluded_swissprot"] == 2
    assert qc["hsp_rows_dropped_phase_conflict_within_molecule_excluded"] == 2
    assert qc["hsp_dropped_phase_conflict_within_molecule"] == 0 == qc["hsp_dropped_phase_conflict_within_molecule_excluded_orthodb"]
    assert qc["hsp_phase_conflict_among_retained_recheck"] == 0 == qc["hsp_phase_conflict_among_excluded_recheck"]
    prov = json.loads(q(db, "SELECT value FROM c0_provenance WHERE key = 'multi_hsp_rule'")[0][0])
    assert "never across" in prov["scope"]


def test_a16_completeness_codes_null_policy_is_recorded_and_gated(est_db):
    """Finding 6.  A16: "Every accepted read alignment carries three codes stored in the C0 tables" -- this build
    cannot compute them (design §2 needs cap/CAGE TSS clusters, poly(A)/TES clusters and the admitted chain set), so
    the NULL is declared in c0_provenance (policy null_not_computed_at_c0, AUTHOR DECISION PENDING) and counted in
    c0_qc; the step-7 gate refuses any accepted row that contradicts the policy."""
    db, record = est_db
    policy = json.loads(q(db, "SELECT value FROM c0_provenance WHERE key = 'completeness_codes'")[0][0])
    assert policy["policy"] == "null_not_computed_at_c0" and policy["status"].startswith("AUTHOR_DECISION_PENDING")
    assert policy["columns"] == ["complete_5p_code", "complete_3p_code", "read_internal_code"] and "A16" in policy["clause"]
    qc = record["qc"]
    accepted = q(db, "SELECT count(*) FROM evidence_alignment WHERE accepted")[0][0]
    assert accepted > 0 and qc["completeness_codes_accepted_rows"] == qc["completeness_codes_null_accepted_rows"] == accepted
    assert qc["completeness_codes_coded_accepted_rows"] == 0 and qc["completeness_codes_policy_violation"] == 0
    assert B.completeness_policy_violation("null_not_computed_at_c0", 10, 9, 0) == 1     # one partially / fully coded row
    assert B.completeness_policy_violation("null_not_computed_at_c0", 10, 10, 0) == 0
    with pytest.raises(B.C0Error, match="unknown completeness-code policy"):
        B.completeness_policy_violation("computed_v0", 1, 0, 1)


# ------------------------------------------------------------------------------ fix round 7 (Codex review 4)
def test_legacy_run_as_library_metadata_is_refused_at_ingestion(world, tmp_path, capsys):
    """Codex review 4 HIGH.  A49.1 decision 4 / A11: a long-read library is an ENA experiment accession or
    '<BioSample>:<library_name>'.  Checksum-valid metadata written under the old rule (library = run, 'SRR2' / 'SRR5')
    would split ZMW m64011/42 into two molecules and keep p1 train_track; the builder now refuses it before molecule
    assignment, with the same validator the producer uses."""
    W.write_pacbio(tmp_path, library="SRR2", reads=[r for r in W.PACBIO_READS if r[0] != "p2"])
    W.write_pacbio(tmp_path, W.PACBIO_RUN2, library="SRR5", reads=W.PACBIO_READS2, audit_extra=W.PACBIO_AUDIT_EXTRA2)
    roles = W.write_roles(tmp_path, extra_rows=[W.pacbio_role_row(W.PACBIO_RUN, 2), W.pacbio_role_row(W.PACBIO_RUN2, 1)])
    rc, args = _run(world, tmp_path, source="pacbio", roles=roles)
    err = capsys.readouterr().err
    assert rc == 1 and "is not an A49.1 key" in err and "'SRR2'" in err
    assert not Path(args[args.index("--out") + 1]).exists()
    from transgenic.evidence import c0_schema as S
    assert S.is_longread_library("SRX1000201") and S.is_longread_library("SAMN49012086:HOTXU")
    for bad in ("SRR2", "SRX_PB1", "LIB1", "SAMN1:", "SAMN1:a\tb", ""):
        assert not S.is_longread_library(bad)


def test_hsp_with_a_stale_library_is_refused(world, est_db, tmp_path, capsys):
    """Codex review 4 MEDIUM.  An HSP projected before a metadata correction keeps the old library; library
    withholding would then keep phase evidence of a withheld library.  The builder compares the HSP library /
    est_accession with its EST member in --est-c0 and fails closed."""
    write_protein(tmp_path)
    keep, _ = _mols(est_db)
    stale = [["h1", W.SPECIES, "L1-old", keep, "EA1.2", "4577_0:000001", "orthodb", "4577", "0", "150", "1", "1e-20",
              "200", "0", "Chr1", "10000", "10150", "+", "0", "2.1.24", "d" * 64, "c" * 64, "p" * 64]]
    rc, _ = _run(world, tmp_path, source="protein",
                 extra=_protein_extra(tmp_path, est_db, write_hsp(tmp_path, _mols(est_db), rows=stale, state=est_state(est_db))))
    err = capsys.readouterr().err
    assert rc == 1 and "stale projection" in err


# ------------------------------------------------------------------------------ fix round 8 (Codex review 5)
def test_rejected_clone_less_est_twin_closes_its_signature_molecule(world, tmp_path, monkeypatch):
    """Codex review 5 HIGH.  A11: clone-less ESTs of one library with an identical exact alignment signature are one
    molecule; rule 4(b): "any placement of any member".  a1 (MAPQ 60, accepted) and b1 (MAPQ 10, rejected) share
    library L1 and contig / strand / blocks at a clean locus; b1's audit secondary (MAPQ 0) lies on the held-out gene
    gH.  Before: b1 had no signature (computed only for accepted members) -> two molecules, a1 stayed train_track.
    Now: one signature molecule, both excluded_heldout."""
    twins = [("a1", "EX1", "L1", "", dict(chrom="Chr1", exons=[(190001, 190100), (190201, 190300)])),
             ("b1", "EY1", "L1", "", dict(chrom="Chr1", exons=[(190001, 190100), (190201, 190300)], mapq=10))]
    monkeypatch.setattr(W, "EST_READS", W.EST_READS + twins)
    audit = dict(W.EST_EXTRA_AUDIT)
    audit["b1"] = [W.sam("b1", "Chr1", [(40001, 40200)], flag=256, mapq=0)]
    W.write_est(tmp_path, audit_extra=audit)
    rc, args = _run(world, tmp_path)
    assert rc == 0
    db = args[args.index("--out") + 1]
    assert q(db, "SELECT read_name, accepted FROM evidence_alignment WHERE read_name IN ('a1', 'b1') "
                 "AND placement_kind = 'primary' ORDER BY read_name") == [("a1", True), ("b1", False)]
    assert mol(db, "a1") == mol(db, "b1")
    assert member(db, "a1")[0] == member(db, "b1")[0] == "excluded_heldout"
    qc = json.load(open(args[args.index("--qc") + 1]))["qc"]
    assert qc["leakage_nonzero"] == 0 and qc["rejected_est_members_with_signature"] >= 1
