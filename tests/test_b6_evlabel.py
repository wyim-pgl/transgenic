"""WP2b evlabel (plan v4.1 §5; A18.4, A19.3, A20, A50, A50.1, A50.2) on the variant evidence world."""
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pytest

import test_b6_evidence_world as EW
import test_b6_repair_world as RW
import b6_build_evlabel as BE
import b6_validate_evlabel as VE
from transgenic.evidence import b6_corpus_repair as CR
from transgenic.evidence import b6_evlabel as EV
from transgenic.evidence import b6_protein_placement as PP
from transgenic.evidence import leakage as LK

duckdb = pytest.importorskip("duckdb")
ROOT = Path(__file__).resolve().parents[1]
GC = LK.load_build_b5().gc
CLS = {c: i for i, c in enumerate(GC.SEG_CLASSES)}


# ------------------------------------------------------------------------------ pure functions
def test_sweep_caps_est_per_library_but_not_long_reads():
    units = {f"m{i}": ("L1", False, [(0, 10)]) for i in range(12)}
    units.update({"x": ("L2", False, [(5, 10)])})
    assert EV.sweep(units, "est") == [(0, 5, 10, False), (5, 10, 11, False)]
    assert EV.sweep(units, "ont") == [(0, 5, 12, False), (5, 10, 13, False)]


def test_sweep_counts_one_unit_once_and_tracks_non_reference():
    units = {"a": ("L1", False, [(0, 10), (5, 15)]), "b": ("L1", True, [(10, 20)])}
    assert EV.sweep(units, "ont") == [(0, 10, 1, False), (10, 15, 2, True), (15, 20, 1, True)]


def test_junction_cells_are_strand_aware_c0_coordinates():
    assert EV.junction_cells(100, 199) == [("intron", 100, 200), ("splice_donor", 100, 101),
                                           ("splice_acceptor", 199, 200)]
    assert EV.junction_cells(199, 100) == [("intron", 100, 200), ("splice_donor", 199, 200),
                                           ("splice_acceptor", 100, 101)]           # '-' strand: donor at the high end


def _p(blocks, strand="+", start=True, stop=True, conv="inside_last_block"):
    return {"strand": strand, "blocks": blocks, "start_complete": start, "stop_complete": stop, "stop_convention": conv}


def test_utr_for_read_labels_5_and_3_prime_bases_and_never_the_stop_codon():
    read = [(100, 300), (400, 600)]
    cds = [(150, 300, 0), (400, 510, 0)]
    u5, u3, st = EV.utr_for_read(read, "+", [_p(cds)])
    assert (u5, u3, st) == ([(100, 150)], [(510, 600)], "ok")
    u5, u3, st = EV.utr_for_read(read, "+", [_p(cds, conv="after_last_block")])
    assert u3 == [(513, 600)]                                                    # stop codon 510-513 is not UTR
    u5, u3, st = EV.utr_for_read(read, "-", [_p(cds, strand="-")])
    assert (u5, u3) == ([(510, 600)], [(100, 150)])


def test_utr_incomplete_side_is_neutral_and_conflicts_block_the_read():
    read = [(100, 300), (400, 600)]
    cds = [(150, 300, 0), (400, 510, 0)]
    assert EV.utr_for_read(read, "+", [_p(cds, start=False)]) == ([], [(510, 600)], "incomplete_orf_5")
    other = [(153, 300, 0), (400, 510, 0)]
    assert EV.utr_for_read(read, "+", [_p(cds), _p(other)])[2] == "conflicting_cds"
    wrong_intron = [(150, 250, 0), (400, 510, 0)]                               # CDS intron != read intron
    assert EV.utr_for_read(read, "+", [_p(wrong_intron)])[2] == "no_candidate"
    assert EV.utr_for_read(read, "-", [_p(cds)])[2] == "no_candidate"           # other strand


# ------------------------------------------------------------------------------ separation from B6 (A50.2)
def test_no_b6_module_imports_evlabel():
    allowed = {"b6_evlabel.py", "b6_build_evlabel.py", "b6_validate_evlabel.py"}
    pat = re.compile(r"^\s*(from|import)\s+[\w.]*b6_evlabel|b6_evlabel\s+import|import_module\([^)]*b6_evlabel",
                     re.M)
    offenders = []
    for base in (ROOT / "src", ROOT / "revision" / "scripts", ROOT / "train", ROOT / "configs"):
        for p in base.rglob("*.py"):
            if p.name not in allowed and pat.search(p.read_text(errors="replace")):
                offenders.append(str(p.relative_to(ROOT)))
    assert offenders == []


def test_valid_only_protein_phase_never_reaches_train_scope():
    """Review 1 H1 (the reviewer's in-memory case): a train_tiles_excluded protein gives valid-scope phase only."""
    rows = [{"alignment_id": "x", "molecule_id": "m", "organism": "o", "chrom": "c", "strand": "+",
             "blocks": [(0, 30, 0)]}]
    elig = {("protein", "m"): ("o", False, True)}                 # (library, nonref, train_tiles_excluded)
    lab = EV.SpeciesLabels()
    EV._protein_cds(lab, "train", rows, elig)
    assert not lab.phase["train"]["c"] and not lab.cells["train"]["c"]
    EV._protein_cds(lab, "valid", rows, elig)
    assert lab.phase["valid"]["c"] == [(0, 30, "+", 0, 0, "x", 30)] and lab.cells["valid"]["c"]


# ------------------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("evlabel_world")
    ev = EW.build_evidence_world(tmp, variant=True, evlabel_extras=True)
    man = tmp / "manifest.jsonl"
    LK.write_interval_manifest(RW.leakage_model(ev).interval_manifest, man)
    corpus, freeze = tmp / "b5_full_b6_v1.db", tmp / "freeze.json"
    CR.apply(ev["corpus"], corpus, freeze, splits=ev["split"], flags=ev["a22"], build_flags=[ev["a22"], ev["a30"]],
             seed=RW.SEED, manifests=[man])
    sup = tmp / f"protein_placement_b6_v1.{EW.SPECIES}.db"
    PP.build(species=EW.SPECIES, gff=ev["protein_gff"], reference=ev["fasta"], protein_c0=ev["dbs"]["protein"],
             merged=ev["merged"], out=sup, stop_convention="inside_last_block")
    roles = tmp / "DATASET_ROLES.b6_v1.tsv"
    roles.write_text((ROOT / "data" / "manifests" / "DATASET_ROLES.b6_v1.tsv").read_text()
                     + "training/pacbio/Athaliana/proj2\tSRR2\tAthaliana\tDATASET_ROLES.longread_v1.tsv\t"
                       "b1_validation_only\tA-CCS23\tfalse\ttrue\t-\ttest: world PacBio run as a scoring set\n")
    Path(str(roles) + ".sha256").write_text(hashlib.sha256(roles.read_bytes()).hexdigest() + "  roles.tsv\n")
    out = tmp / "evlabel_b6_v1.db"
    assert BE.main(["--merged", str(ev["merged"]), "--protein-placement", str(sup), "--corpus", str(corpus),
                    "--freeze", str(freeze), "--roles", str(roles), "--out", str(out)]) == 0
    return dict(ev=ev, corpus=corpus, freeze=freeze, sup=sup, roles=roles, out=out)


def q(db, sql, params=()):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql, list(params)).fetchall()
    finally:
        con.close()


def qc(built, source, metric):
    r = q(built["out"], "SELECT value FROM evlabel_qc WHERE source = ? AND metric = ?", [source, metric])
    return r[0][0] if r else 0


def _validate(built, *extra):
    return VE.main(["--evlabel", str(built["out"]), "--merged", str(built["ev"]["merged"]), "--corpus",
                    str(built["corpus"]), "--roles", str(built["roles"]), "--protein-placement", str(built["sup"]),
                    *extra])


def test_validator_passes_independent_n_rc_mirror_and_n_base_checks(built, tmp_path):
    report = tmp_path / "r.json"
    assert _validate(built, "--report", str(report)) == 0
    got = json.loads(report.read_text())["qc"]
    assert got["junctions_n_verified"] > 0 and got["rc_tiles_mirrored"] > 0


def test_provenance_records_policies_and_every_input(built):
    prov = json.loads(Path(str(built["out"]) + ".PROVENANCE.json").read_text())
    assert prov["consumer_policy"] == EV.CONSUMER_POLICY and prov["boundary_policy"] == "boundary_cells_not_emitted_v1"
    assert "not final" in prov["boundary_note"]
    assert prov["inputs"]["corpus"]["sha256"] == LK.sha256(built["corpus"])
    assert prov["inputs"]["roles"]["sha256"] == LK.sha256(built["roles"])
    with pytest.raises(EV.EvlabelError, match="A50.2"):
        EV.refuse_if_evlabel(built["out"])
    EV.refuse_if_evlabel(built["corpus"])                     # a corpus is not refused


def test_est_junction_n_counts_distinct_molecules_and_4c_drops(built):
    rows = q(built["out"], "SELECT n, weight FROM evlabel_junction WHERE source = 'est' AND scope = 'train' "
                           "AND donor0 = 10200 AND acceptor0 = 10299")
    assert rows, "gA intron 1 must be an EST junction"
    n, w = rows[0]
    assert n >= 3 and w == pytest.approx(min(4.0, 1 + 1.0 * math.log1p(n)))
    assert qc(built, "est", "junctions_dropped_excluded_4c") >= 1       # gV / gV2 shared donor (rule 4(c))


def test_protein_junction_needs_two_organisms(built):
    rows = dict(((d, a), (n, w)) for d, a, n, w in q(
        built["out"], "SELECT donor0, acceptor0, n, weight FROM evlabel_junction WHERE source = 'protein' "
                      "AND scope = 'train' ORDER BY ALL"))
    assert rows[(10200, 10299)] == (2, pytest.approx(1 + 0.5 * math.log1p(2)))   # gA intron 1: 4577_0 + 39947_0
    b11 = _orf(built, "4577_0:000011")                                           # one organism only: no cell
    assert (b11[0][1], b11[1][0] - 1) not in rows
    assert all(n >= 2 for n, _ in rows.values())
    assert qc(built, "protein", "protein_junctions_below_2_organisms") >= 1


def test_scoring_dataset_never_shapes_a_label(built):
    assert qc(built, "pacbio", "molecules_dropped_b6_score") >= 1
    assert q(built["out"], "SELECT count(*) FROM evlabel_cell WHERE source = 'pacbio'") == [(0,)]


def _tile(built, ws, we, rc=False):
    return q(built["out"], "SELECT rn FROM evlabel_tile WHERE chrom = 'Chr1' AND ws = ? AND we = ? AND is_rc = ?",
             [ws, we, rc])[0][0]


def _cells(built, rn, cls, source=None):
    sql = "SELECT start0, end0, weight, source, rule FROM evlabel_cell WHERE rn = ? AND class_index = ?"
    params = [rn, CLS[cls]]
    if source:
        sql += " AND source = ?"
        params.append(source)
    return q(built["out"], sql + " ORDER BY ALL", params)


def _orf(built, protein):
    return json.loads(q(built["sup"], "SELECT cds_blocks FROM protein_placement WHERE protein_id = ?", [protein])[0][0])


def test_long_read_utr_5_and_3_prime_cells_on_both_strands(built):
    ws = 2860
    rn = _tile(built, ws, 33580)
    for prot, strand in (("4577_0:000011", "+"), ("4577_0:000012", "-")):
        blocks = _orf(built, prot)
        five, three = EW.UTR_EXT[prot]
        cs, ce = blocks[0][0], blocks[-1][1]
        u5 = [(s, e) for s, e, w, src, rule in _cells(built, rn, "5UTR", "ont")]
        u3 = [(s, e) for s, e, w, src, rule in _cells(built, rn, "3UTR", "ont")]
        if strand == "+":
            assert (cs - five - ws, cs - ws) in u5 and (ce - ws, ce + three - ws) in u3
        else:
            assert (ce - ws, ce + five - ws) in u5 and (cs - three - ws, cs - ws) in u3
    pcg = {(s, e, rule) for s, e, w, src, rule in _cells(built, rn, "protein_coding_gene", "ont")}
    assert all(rule == "utr_pcg" for *_, rule in pcg) and pcg
    assert qc(built, "ont", "longread_utr5_bases") >= 160 and qc(built, "ont", "longread_utr3_bases") >= 240


def test_conflicting_cds_and_incomplete_orf_block_utr(built):
    assert qc(built, "ont", "longread_utr_blocked_conflicting_cds") == 1        # ORF 14 vs 39947_0:000016
    gA = {r[0]: (r[1], r[2]) for r in q(built["sup"], "SELECT protein_id, start_codon_ok AND qstart0 = 0, "
                                                      "stop_codon_ok AND qend = qlen FROM protein_placement")}
    if not all(gA[p][0] and gA[p][1] for p in ("4577_0:000021", "39947_0:000022")):
        assert qc(built, "est", "est_utr_blocked_incomplete_orf") >= 1


def test_protein_cds_labels_cds_family_only_with_organism_units(built):
    rn = _tile(built, 2860, 33580)
    cells = _cells(built, rn, "protein_coding_gene", "protein") + _cells(built, rn, "exon", "protein")
    ga = [c for c in cells if c[0] == 10000 - 2860]
    assert ga and all(w == pytest.approx(1 + math.log1p(2)) for _, _, w, _, _ in ga)       # two organisms
    for cls in ("5UTR", "3UTR", "intron"):
        assert all(src != "protein" or rule == "junction" for *_, src, rule in _cells(built, rn, cls))
    phases = q(built["out"], "SELECT start0, codon_position FROM evlabel_phase WHERE rn = ? ORDER BY start0", [rn])
    b11 = _orf(built, "4577_0:000011")
    assert (b11[0][0] - 2860, 0) in phases and (b11[1][0] - 2860, 1) in phases     # GFF phase 2 -> codon pos 1


def test_retained_intron_bases_get_quarter_weight(built):
    rn = _tile(built, 2860, 33580)
    ret = [c for c in _cells(built, rn, "exon", "ont") if c[4] == "retained_intron"]
    assert ret == [(10200 - 2860, 10300 - 2860, 0.25, "ont", "retained_intron")]


def test_cells_on_n_and_test_tiles_are_dropped_and_rc_rows_are_explicit(built):
    assert q(built["out"], "SELECT count(*) FROM evlabel_tile WHERE split = 'test'") == [(0,)]
    assert q(built["out"], "SELECT count(*) FROM evlabel_tile WHERE is_rc") != [(0,)]
    rn = _tile(built, 64300, 95020)
    con = duckdb.connect(str(built["out"]), read_only=True)
    labels, weights = EV.to_dense(con, rn)
    ul, uw = EV.to_dense(con, rn, unit_weight=True)
    con.close()
    assert labels.shape == (30720, len(GC.SEG_CLASSES)) and labels.dtype.name == "float32"
    assert set(uw[uw > 0].tolist()) <= {1.0} and (ul == labels).all()


def test_to_dense_takes_the_max_across_sources(built):
    rn = _tile(built, 2860, 33580)
    con = duckdb.connect(str(built["out"]), read_only=True)
    labels, weights = EV.to_dense(con, rn)
    con.close()
    p = 10050 - 2860                                   # gA exon 1: EST exon and protein exon overlap
    est = max(w for s, e, w, src, _ in _cells(built, rn, "exon", "est") if s <= p < e)
    prot = max(w for s, e, w, src, _ in _cells(built, rn, "exon", "protein") if s <= p < e)
    assert weights[p, CLS["exon"]] == pytest.approx(max(est, prot))


def test_validator_rejects_a_tampered_junction_n(built, tmp_path, capsys):
    bad = tmp_path / "bad.db"
    shutil.copyfile(built["out"], bad)
    con = duckdb.connect(str(bad))
    con.execute("UPDATE evlabel_junction SET n = n + 1 WHERE source = 'est'")
    con.execute("CHECKPOINT")
    con.close()
    sha = LK.sha256(bad)
    Path(str(bad) + ".sha256").write_text(f"{sha}  bad.db\n")
    prov = json.loads(Path(str(built["out"]) + ".PROVENANCE.json").read_text())
    Path(str(bad) + ".PROVENANCE.json").write_text(json.dumps(dict(prov, output_sha256=sha)))
    assert VE.main(["--evlabel", str(bad), "--merged", str(built["ev"]["merged"]), "--corpus", str(built["corpus"]),
                    "--roles", str(built["roles"]), "--protein-placement", str(built["sup"])]) == 1
    assert "independent SQL" in capsys.readouterr().err


def test_build_refuses_unbound_inputs(built, tmp_path):
    fz = tmp_path / "f.json"
    fz.write_text(json.dumps(dict(json.loads(built["freeze"].read_text()), out_sha256="0" * 64)))
    with pytest.raises(EV.EvlabelError, match="freeze"):
        EV.build(merged=built["ev"]["merged"], supplements=[built["sup"]], corpus=built["corpus"], freeze=fz,
                 roles=built["roles"], out=tmp_path / "x.db")
    with pytest.raises(EV.EvlabelError, match="exists"):
        EV.build(merged=built["ev"]["merged"], supplements=[built["sup"]], corpus=built["corpus"],
                 freeze=built["freeze"], roles=built["roles"], out=built["out"])


# ------------------------------------------------------------------------------ review 1 fixes, end to end
def _aid(built, protein):
    return q(built["sup"], "SELECT alignment_id FROM protein_placement WHERE protein_id = ?", [protein])[0][0]


def test_train_tiles_excluded_protein_gives_no_train_phase_or_cds(built):
    aid = _aid(built, "4577_0:000017")
    assert q(built["sup"], "SELECT train_tiles_excluded FROM protein_placement WHERE alignment_id = ?", [aid]) == [(True,)]
    assert q(built["out"], "SELECT count(*) FROM evlabel_phase p JOIN evlabel_tile t USING (rn) "
                           "WHERE p.alignment_id = ? AND t.scope = 'train'", [aid]) == [(0,)]
    rows = q(built["out"], "SELECT count(*) FROM evlabel_cell c JOIN evlabel_tile t USING (rn) WHERE t.scope = 'train' "
                           "AND c.source = 'protein' AND NOT t.is_rc AND t.ws + c.start0 >= 100000 AND t.ws + c.end0 <= 100700")
    assert rows == [(0,)]


def test_validator_rejects_a_train_phase_row_of_a_valid_only_protein(built, tmp_path, capsys):
    aid = _aid(built, "4577_0:000017")
    rn = _tile(built, 95020, 125740)
    bad = tmp_path / "bad.db"
    shutil.copyfile(built["out"], bad)
    con = duckdb.connect(str(bad))
    con.execute("INSERT INTO evlabel_phase VALUES (?, 4980, 5280, '+', 0, ?)", [rn, aid])
    con.execute("CHECKPOINT")
    con.close()
    sha = LK.sha256(bad)
    Path(str(bad) + ".sha256").write_text(f"{sha}  bad.db\n")
    prov = json.loads(Path(str(built["out"]) + ".PROVENANCE.json").read_text())
    Path(str(bad) + ".PROVENANCE.json").write_text(json.dumps(dict(prov, output_sha256=sha)))
    assert VE.main(["--evlabel", str(bad), "--merged", str(built["ev"]["merged"]), "--corpus", str(built["corpus"]),
                    "--roles", str(built["roles"]), "--protein-placement", str(built["sup"])]) == 1
    assert "not eligible" in capsys.readouterr().err


def _freeze_variant(built, tmp_path, **over):
    fz = json.loads(built["freeze"].read_text())
    for k, v in over.items():
        if v is None:
            fz.pop(k)
        else:
            fz[k] = v
    p = tmp_path / "freeze.variant.json"
    p.write_text(json.dumps(fz))
    return p


def _roles_obj(built):
    from transgenic.evidence import b6_roles as R

    return R.load_roles(built["roles"])


def test_training_binding_passes_on_the_real_wp2_freeze_and_binds_per_species(built):
    rec, fz = EV.verify_training_binding(merged=built["ev"]["merged"], corpus=built["corpus"], freeze=built["freeze"],
                                         roles_obj=_roles_obj(built))
    assert rec["family"] == "training" and rec["species"] == [EW.SPECIES]
    assert rec["interval_manifest_sha256"] == {EW.SPECIES: CR.freeze_manifest_sha256(fz, EW.SPECIES)}


@pytest.mark.parametrize("over, pattern", [
    ({"base_sha256": None, "interval_manifest_sha256": None, "rule_id": None}, "required key"),
    ({"base_sha256": "1" * 64}, "base_sha256"),
    ({"interval_manifest_sha256": {EW.SPECIES: "2" * 64}}, "interval_manifest_sha256"),
    ({"interval_manifest_sha256": {"Other": "2" * 64}}, "no interval manifest for"),
    ({"interval_manifest_sha256": "2" * 64}, "species"),
])
def test_training_binding_refuses_a_freeze_that_does_not_bind_c0_exclusions(built, tmp_path, over, pattern):
    with pytest.raises(EV.EvlabelError, match=pattern):
        EV.verify_training_binding(merged=built["ev"]["merged"], corpus=built["corpus"],
                                   freeze=_freeze_variant(built, tmp_path, **over), roles_obj=_roles_obj(built))


def _tampered_merged(built, tmp_path, name, edit):
    bad = tmp_path / name
    shutil.copyfile(built["ev"]["merged"], bad)
    con = duckdb.connect(str(bad))
    edit(con)
    con.execute("CHECKPOINT")
    con.close()
    Path(str(bad) + ".sha256").write_text(f"{LK.sha256(bad)}  {name}\n")
    return bad


def _starve_exhaustiveness(con):
    ex = json.loads(con.execute("SELECT value FROM c0_meta WHERE key = 'exhaustiveness'").fetchone()[0])
    ex[EW.SPECIES]["by_source"]["est"]["below_cap"] = 0
    con.execute("UPDATE c0_meta SET value = ? WHERE key = 'exhaustiveness'", [json.dumps(ex)])


@pytest.mark.parametrize("name, edit, pattern", [
    ("leak.db", lambda c: c.execute("UPDATE c0_qc SET value = 1 WHERE metric = 'leakage_nonzero' AND source = 'est'"),
     "leakage"),
    ("exh.db", _starve_exhaustiveness, "A49.2"),
    ("pol.db", lambda c: c.execute("UPDATE c0_provenance SET value = '{\"policy\": \"computed_post_c0_v1\"}' "
                                   "WHERE key = 'completeness_codes' AND source = 'ont'"), "completeness"),
])
def test_training_binding_refuses_leakage_qc_exhaustiveness_and_policy_failures(built, tmp_path, name, edit, pattern):
    bad = _tampered_merged(built, tmp_path, name, edit)
    with pytest.raises(EV.EvlabelError, match=pattern):
        EV.verify_training_binding(merged=bad, corpus=built["corpus"], freeze=built["freeze"],
                                   roles_obj=_roles_obj(built))


def test_a_species_with_accepted_proteins_needs_its_supplement(built, tmp_path):
    with pytest.raises(EV.EvlabelError, match="no protein-placement supplement"):
        EV.build(merged=built["ev"]["merged"], supplements=[], corpus=built["corpus"], freeze=built["freeze"],
                 roles=built["roles"], out=tmp_path / "x.db")
    prov = json.loads(Path(str(built["out"]) + ".PROVENANCE.json").read_text())
    assert prov["inputs"]["protein_evidence"] == {EW.SPECIES: "bound"}


def _wp3_bindings():
    try:
        from transgenic.evidence import b6_bindings as BB   # WP3; present only after the b6-integration merge
        return BB
    except ImportError:
        return None


WP3_REASON = ("requires the b6-integration tree: WP3's b6_bindings.verify_inputs needs the freeze FILE PATH and a real "
              "observation cache built by WP3 (b6_build_obs_cache); neither exists in the WP2 worktree, so this case "
              "builds a valid WP3 binding first and runs only where both are available")


def _local(evidence, corpus, built):
    return EV.verify_training_binding(merged=evidence, corpus=corpus, freeze=built["freeze"],
                                      roles_obj=_roles_obj(built))


def test_training_input_verifier_rejects_an_evlabel_db_as_evidence_or_corpus(built):
    """Review M7: the B6 trainer (train_HyenaTransgenic_B6.py) passes its evidence and corpus through the §8
    training verifier; an evlabel DB in either slot must be rejected by that verifier (behaviour, not a name)."""
    _local(built["ev"]["merged"], built["corpus"], built)                  # the valid binding passes first
    with pytest.raises(EV.EvlabelError, match=r"(?i)c0|schema|sha256"):
        _local(built["out"], built["corpus"], built)
    with pytest.raises(EV.EvlabelError, match=r"out_sha256"):
        _local(built["ev"]["merged"], built["out"], built)


@pytest.mark.skipif(_wp3_bindings() is None, reason=WP3_REASON)
def test_wp3_training_verifier_rejects_an_evlabel_db_after_a_valid_binding(built):
    BB = _wp3_bindings()
    cache_path = built.get("cache")
    if cache_path is None:
        pytest.skip(WP3_REASON)
    from transgenic.evidence import b6_obs_cache as OC

    def verify(evidence, corpus):
        return BB.verify_inputs(accept=("training",), evidence=evidence, roles=_roles_obj(built),
                                cache=OC.ObsCache(cache_path), freeze=str(built["freeze"]), corpus=corpus)
    verify(built["ev"]["merged"], built["corpus"])                           # a valid binding first
    with pytest.raises(Exception):
        verify(built["out"], built["corpus"])
    with pytest.raises(Exception):
        verify(built["ev"]["merged"], built["out"])


# ------------------------------------------------------------------------------ review 2 H2: exact source coverage
def _drop_ont_everywhere_but_the_manifest(con):
    """The reviewer's reproduction: merge_manifest keeps EST + ONT, provenance and exhaustiveness keep EST only, and
    ONT's leakage_nonzero is 1."""
    con.execute("DELETE FROM c0_provenance WHERE source <> 'est'")
    con.execute("UPDATE c0_qc SET value = 1 WHERE source = 'ont' AND metric = 'leakage_nonzero'")
    ex = json.loads(con.execute("SELECT value FROM c0_meta WHERE key = 'exhaustiveness'").fetchone()[0])
    ex[EW.SPECIES]["by_source"] = {"est": ex[EW.SPECIES]["by_source"]["est"]}
    con.execute("UPDATE c0_meta SET value = ? WHERE key = 'exhaustiveness'", [json.dumps(ex)])
    con.execute("DELETE FROM merge_manifest WHERE source NOT IN ('est', 'ont')")


def _edit_exhaustiveness(fn):
    def edit(con):
        ex = json.loads(con.execute("SELECT value FROM c0_meta WHERE key = 'exhaustiveness'").fetchone()[0])
        fn(ex[EW.SPECIES]["by_source"])
        con.execute("UPDATE c0_meta SET value = ? WHERE key = 'exhaustiveness'", [json.dumps(ex)])
    return edit


@pytest.mark.parametrize("name, edit, pattern", [
    ("repro.db", _drop_ont_everywhere_but_the_manifest, "coverage"),
    ("noqc.db", lambda c: c.execute("DELETE FROM c0_qc WHERE source = 'ont'"), "no c0_qc record"),
    ("extraqc.db", lambda c: c.execute("INSERT INTO c0_qc VALUES ('Gmax', 'ont', 'leakage_nonzero', 0)"),
     "outside the merged coverage"),
    ("noexh.db", _edit_exhaustiveness(lambda by: by.pop("ont")), "exhaustiveness sources"),
    ("float.db", _edit_exhaustiveness(lambda by: by["est"].update(molecules=float(by["est"]["molecules"]))),
     "integers"),
    ("string.db", _edit_exhaustiveness(lambda by: by["ont"].update(below_cap=str(by["ont"]["below_cap"]))),
     "integers"),
    ("negative.db", _edit_exhaustiveness(lambda by: by["ont"].update(below_cap=-1)), "integers"),
])
def test_training_binding_requires_exact_merge_manifest_coverage_and_integer_counters(built, tmp_path, name, edit,
                                                                                      pattern):
    bad = _tampered_merged(built, tmp_path, name, edit)
    with pytest.raises(EV.EvlabelError, match=pattern):
        EV.verify_training_binding(merged=bad, corpus=built["corpus"], freeze=built["freeze"],
                                   roles_obj=_roles_obj(built))
