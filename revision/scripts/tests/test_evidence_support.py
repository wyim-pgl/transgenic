"""Synthetic B1 controls; no alignment, sequence or manifest from evidence/ is read."""
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "59_evidence_support.py"
m = types.ModuleType("evidence_support")
m.__file__ = str(SCRIPT)
sys.modules[m.__name__] = m
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), m.__dict__)


def genome(introns=((101, 200), (251, 350)), strand="+"):
    seq = list("C" * 1000)
    for d, a in introns:
        seq[d-1:d+1] = "GT" if strand == "+" else "CT"
        seq[a-2:a] = "AG" if strand == "+" else "AC"
    return {"chr1": "".join(seq)}


def sam(name="read1", start=51, cigar="50M100N50M100N50M",
        cs=":50~gt100ag:50~gt100ag:50", flag=0, ts="+", mapq=60, extra=""):
    tags = f"cs:Z:{cs}\tAS:i:150" + (f"\tts:A:{ts}" if ts is not None else "")
    return f"{name}\t{flag}\tchr1\t{start}\t{mapq}\t{cigar}\t*\t0\t0\t*\t*\t{tags}{extra}\n"


def addition(**kwargs):
    base = dict(addition_id="add1", locus_id="gene1", species="synthetic", assembly="fixture",
                contig="chr1", strand="+", cds_start=51, cds_end=400, locus_start=1, locus_end=500,
                introns=((101, 200), (251, 350)), novel_introns=((251, 350),),
                novelty="junction-novel", filter_pass=True, exact_cds_match=False)
    base.update(kwargs)
    return m.Addition(**base)


def obs(name="read1", source="ONT", library="lib", run="run1", g=None, **kwargs):
    a = m.parse_sam(sam(name=name, **kwargs))
    strand = m.transcript_strand(a)
    return m.Observation(a, "synthetic", "fixture", source, "dataset", run, library, "project",
                         "reference", "independent", True, strand,
                         m.call_junctions(a, g or genome(), strand, source), metadata={
                             "protocol": "direct_RNA", "umi_status": "absent", "arm": "primary",
                             "source_molecule": name})


def score(a, observations):
    return m.score_addition(a, m.assign_molecules(observations))


def test_positive_complete_chain_control():
    row = score(addition(), [obs(f"r{i}") for i in range(3)])
    assert row["chain_support_single_read"] == "complete"
    assert row["novel_junction_support"] == "all"
    assert row["chain_reading_report"]["chain_support_single_read"]["tier"] == "T1"
    assert row["chain_callable"] and row["junction_callable"]
    assert row["chain_reading_report"]["chain_support_single_read"]["chain_negative"] is False


def test_partial_control_does_not_assemble_a_combination():
    left = [obs(f"l{i}", cigar="50M100N30M", cs=":50~gt100ag:30") for i in range(3)]
    right = [obs(f"r{i}", start=221, cigar="30M100N50M", cs=":30~gt100ag:50") for i in range(3)]
    a = addition(novelty="combination-novel", novel_introns=())
    row = score(a, left + right)
    assert row["chain_support_single_read"] == "partial"
    assert row["constituent_junction_support"] is True
    assert row["chain_witness_reads"] == 0 and not row["chain_callable"]
    assert row["novel_junction_support"] == "N/A" and row["chain_reading_report"]["chain_support_single_read"]["tier"] == "T4"
    assert row["chain_reading_report"]["chain_support_single_read"]["chain_negative"] is None


def test_ambiguous_correction_control():
    g = genome(((101, 200), (103, 200), (251, 350)))
    row = score(addition(), [obs(f"r{i}", g=g) for i in range(3)])
    assert row["chain_support_single_read"] == "partial"
    assert row["chain_reading_report"]["chain_support_single_read"]["chain_negative"] is True
    assert m.call_junctions(m.parse_sam(sam()), g, "+", "ONT")[0].status == "ambiguous_correction"


def test_negative_callable_control_and_uncovered_are_different():
    a = addition()
    unspliced = [obs(cigar="350M", cs=":350")]
    row = score(a, unspliced)
    assert row["chain_reading_report"]["chain_support_single_read"]["tier"] == "T5" and row["chain_reading_report"]["chain_support_single_read"]["chain_negative"] is True
    row = score(a, [])
    assert row["chain_reading_report"]["chain_support_single_read"]["tier"] == "T6" and row["chain_reading_report"]["chain_support_single_read"]["chain_negative"] is None
    assert row["junction_negative"] is None and row["uncovered"]


@pytest.mark.parametrize("flag,ts,expected", [(0,"+","+"),(0,"-","-"),(16,"+","-"),(16,"-","+")])
def test_a39_ts_is_relative_to_original_read(flag, ts, expected):
    assert m.transcript_strand(m.parse_sam(sam(flag=flag, ts=ts))) == expected


def test_uf_fallback_and_unknown_strand():
    a = m.parse_sam(sam(flag=16, ts=None))
    assert m.transcript_strand(a, uf=True) == "-"
    assert m.transcript_strand(a) is None
    with pytest.raises(m.EvidenceError, match="contradicts"):
        m.transcript_strand(m.parse_sam(sam(ts="-")), uf=True)


def test_minus_strand_control():
    a = addition(strand="-")
    row = score(a, [obs(f"r{i}", flag=16, g=genome(strand="-")) for i in range(3)])
    assert row["chain_support_single_read"] == "complete"


def test_primary_later_supplementary_and_cross_contig_rejected():
    primary = m.parse_sam(sam())
    supplementary = m.parse_sam(sam(flag=2048))
    assert m.alignment_rejection([primary, supplementary]) == "supplementary_or_chimeric"
    assert m.alignment_rejection([supplementary, primary]) == "supplementary_or_chimeric"
    assert m.alignment_rejection([primary, replace(primary, contig="chr2")]) == "cross_contig_or_strand"
    assert m.alignment_rejection([replace(primary, mapq=19)]) == "mapq"
    assert m.alignment_rejection([primary, replace(primary, flag=256)]) == "equal_best_placement"


def test_malformed_or_missing_cs_is_not_negative():
    a = m.parse_sam(sam())
    with pytest.raises(m.EvidenceError, match="missing cs"):
        m.parse_cs(replace(a, tags={}))
    with pytest.raises(m.EvidenceError, match="cs/CIGAR"):
        m.parse_cs(replace(a, tags={"cs": ":149"}))


@pytest.mark.parametrize("cs,accepted", [
    (":19*ac~gt100ag:20", True),
    (":17*ac*ac*ac~gt100ag:20", False),
    (":20+a~gt100ag:20", False),
    (":19~gt100ag:20", False),
])
def test_ont_anchor_boundaries(cs, accepted):
    # Build matching CIGAR from cs so tests exercise junction filtering, not malformed input.
    left = cs.split("~")[0]
    qleft = sum(int(x[1:]) if x[0] == ":" else 1 for x in m.CS.findall(left) if x[0] != "+")
    cigar = f"{qleft}M" + ("1I" if "+" in left else "") + "100N20M"
    a = m.parse_sam(sam(start=101-qleft, cigar=cigar, cs=cs))
    assert m.call_junctions(a, genome(), "+", "ONT")[0].accepted is accepted


def test_source_specific_thresholds_never_pool_subthreshold_units():
    row = score(addition(), [obs("ont", cigar="50M100N30M", cs=":50~gt100ag:30"),
                             obs("pb", source="PacBio", cigar="50M100N30M", cs=":50~gt100ag:30")])
    assert row["chain_support_single_read"] == "none" and row["supported_junctions"] == 0
    assert score(addition(), [obs("p1", source="PacBio"), obs("p2", source="PacBio")])["chain_reading_report"]["chain_support_single_read"]["tier"] == "T1"


def test_est_signature_clone_and_library_units():
    observations = [obs("a.1", source="EST"), obs("b.1", source="EST"),
                    obs("c.1", source="EST", library="other")]
    m.assign_molecules(observations)
    assert m.support_counts(observations)["units"] == 2
    observations[0].metadata["clone"] = "CLONE-1-forward"
    observations[1].metadata["clone"] = "clone 1 reverse"
    m.assign_molecules(observations)
    assert observations[0].molecule == observations[1].molecule
    assert m.support_counts(observations)["units"] == 2


def test_est_cap_and_single_record_column():
    observations = [obs(f"r{i}", source="EST") for i in range(12)]
    for i, o in enumerate(observations):
        o.metadata["clone"] = f"clone{i}x"
    m.assign_molecules(observations)
    assert m.support_counts(observations)["units"] == 10
    assert m.support_counts(observations)["raw_reads"] == 12


def test_pcr_units_and_unresolved_nontransitive_group():
    observations = [obs(f"r{i}") for i in range(3)]
    for o in observations:
        o.metadata["protocol"] = "cDNA"
    m.assign_molecules(observations)
    assert m.support_counts(observations)["units"] == 1
    for offset, o in zip((0, 9, 18), observations):
        o.alignment = replace(o.alignment, start=51+offset, end=400+offset)
    with pytest.raises(m.RuleUnresolved, match="non-transitive"):
        m.assign_molecules(observations)


def test_unknown_umi_status_fails_closed():
    o = obs()
    o.metadata.update(protocol="cDNA", umi_status="unknown")
    with pytest.raises(m.EvidenceError, match="UMI"):
        m.assign_molecules([o])


def test_chain_threshold_disagreement_is_paired_and_flagged():
    observations = [obs("whole")]
    observations += [obs(f"l{i}", cigar="50M100N30M", cs=":50~gt100ag:30") for i in range(2)]
    observations += [obs(f"r{i}", start=221, cigar="30M100N50M", cs=":30~gt100ag:50") for i in range(2)]
    row = score(addition(), observations)
    assert row["chain_support_single_read"] == "complete"
    assert row["chain_support_threshold"] == "partial"
    assert row["chain_reading_report"]["chain_support_threshold"]["tier"] == "T3"
    assert row["tier_disagreement"] is True


def test_both_denominators_and_wilson_empty_are_explicit():
    rows = [score(addition(), [obs(f"r{i}") for i in range(3)]),
            score(addition(addition_id="uncovered"), [])]
    for row in rows:
        row.update(arm="primary", genotype="reference", scope="ONT", length_bin="all")
    table = m.table_inputs(rows)
    chain = [r for r in table if r["metric"] == "chain" and not r["filtered"]
             and r["chain_reading"] == "chain_support_single_read"]
    assert [(r["denominator"], r["numerator"], r["n"]) for r in chain] == [("all", 1, 2), ("callable", 1, 1)]
    assert chain[0]["callable_unsupported"] == 0
    assert m.wilson(0, 0) == (None, None, None)


def test_paf_alone_rejected(tmp_path):
    p = tmp_path / "raw.paf"
    p.write_text("")
    with pytest.raises(m.EvidenceError, match="PAF alone"):
        list(m.sam_lines(p))


def test_est_accession_versions_keep_newest():
    old, new = obs("A.1", source="EST"), obs("A.2", source="EST")
    old.alignment = replace(old.alignment, start=50)
    retained = m.assign_molecules([old, new])
    assert len(retained) == 1 and retained[0].alignment.read == "A.2"


def test_indel_outside_exclusion_zone_does_not_reject_anchor():
    a = m.parse_sam(sam(start=71, cigar="20M1I10M100N30M", cs=":20+a:10~gt100ag:30"))
    assert m.call_junctions(a, genome(), "+", "ONT")[0].accepted


def test_unique_shift_and_exact_est_coordinates():
    g = genome(((102, 201),))
    assert m.correct_junction(g, "chr1", (101, 200), "+", "ONT") == ((102, 201), "corrected")
    assert m.correct_junction(g, "chr1", (101, 200), "+", "PacBio") == ((102, 201), "corrected")
    assert m.correct_junction(g, "chr1", (101, 200), "+", "EST") == ((101, 200), "uncorrected")


def test_additional_failed_intron_cannot_be_deleted_to_create_chain():
    o = obs()
    extra = m.Junction((220, 225), None, "anchor_failed", False, False)
    o.junctions = (o.junctions[0], extra, o.junctions[1])
    assert not m.chain_witness(addition(), o)


def synthetic_config(tmp_path):
    def write(name, value):
        p = tmp_path / name
        p.write_text(value)
        return {"path": name, "sha256": m.digest(p)}
    reference = write("ref.fa", ">chr1\n" + genome()["chr1"] + "\n")
    roles = write("roles.tsv", "dataset\trun\tspecies\trole\n"
                  "dataset\trun1\tsynthetic\tb1_validation_only\n")
    alignment = write("run.sam", "".join(sam(name=f"r{i}") for i in range(3)))
    metadata = write("meta.json", json.dumps({f"r{i}": dict(
        library="lib", bioproject="project", genotype_stratum="reference", ingestion_qc_pass=True,
        mapping_ambiguous=False, protocol="direct_RNA", umi_status="absent") for i in range(3)}))
    additions = write("additions.json", json.dumps([m.asdict(addition())]))
    spec = dict(dataset="dataset", run="run1", species="synthetic", assembly="fixture", source="ONT",
                genotype_stratum="reference", uf=False, annotation_independence="independent",
                model_independent=True, status="complete", alignment=alignment, metadata=metadata)
    config = dict(schema="b1-evidence-v1", purpose="synthetic", assembly="fixture", genome=reference,
                  role_manifests=[roles], additions=additions, runs=[spec])
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    return p, config


def test_cli_reproducibility_and_hash_tampering(tmp_path):
    p, config = synthetic_config(tmp_path)
    assert m.main(["--config", str(p), "--out", str(tmp_path / "out1")]) == 0
    assert m.main(["--config", str(p), "--out", str(tmp_path / "out2")]) == 0
    one, two = tmp_path / "out1", tmp_path / "out2"
    assert {f.name: f.read_bytes() for f in one.iterdir()} == {f.name: f.read_bytes() for f in two.iterdir()}
    provenance = json.loads((one / "PROVENANCE.json").read_text())
    assert str(SCRIPT) in provenance["inputs_sha256"]
    (tmp_path / "run.sam").write_text(sam())
    assert m.main(["--config", str(p), "--out", str(tmp_path / "out3")]) == 2
    assert not (tmp_path / "out3").exists()


def test_missing_manifest_role_and_missing_metadata_stop(tmp_path):
    p, config = synthetic_config(tmp_path)
    config["runs"][0]["run"] = "absent"
    p.write_text(json.dumps(config))
    assert m.main(["--config", str(p), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    config["runs"][0]["run"] = "run1"
    (tmp_path / "meta.json").write_text("{}")
    config["runs"][0]["metadata"]["sha256"] = m.digest(tmp_path / "meta.json")
    p.write_text(json.dumps(config))
    assert m.main(["--config", str(p), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_b1_now_reaches_existing_qc_gate(tmp_path):
    p, config = synthetic_config(tmp_path)
    config["purpose"] = "b1"
    p.write_text(json.dumps(config))
    with pytest.raises(m.EvidenceError, match="frozen sha256"):
        m.execute(p, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_corrected_chain_uses_corrected_coordinates_without_rewriting_blocks():
    introns = ((102, 201), (252, 351))
    a = addition(introns=introns, novel_introns=(introns[1],))
    assert score(a, [obs(f"r{i}", g=genome(introns)) for i in range(3)])["chain_support_single_read"] == "complete"


@pytest.mark.parametrize("source,threshold,tier", [("ONT", 3, "T1"), ("PacBio", 2, "T1"),
                                                 ("EST", 2, "T2"), ("FL-cDNA", 2, "T2")])
def test_single_witness_needs_no_constituent_threshold(source, threshold, tier):
    reads = [obs(f"r{i}", source=source) for i in range(threshold)]
    for i, o in enumerate(reads):
        o.metadata["clone"] = f"clone{i}x"
    one = score(addition(), reads[:1])
    assert one["chain_support_single_read"] == "complete"
    assert one["chain_support_threshold"] == "none"
    assert one["chain_reading_report"]["chain_support_single_read"]["tier"] == tier
    assert one["chain_reading_report"]["chain_support_threshold"]["tier"] == "T5"
    assert one["tier_disagreement"]
    assert "tier" not in one and "chain_support" not in one
    enough = score(addition(), reads)
    assert enough["chain_support_threshold"] == "complete"
    assert not enough["tier_disagreement"]


def test_union_does_not_pool_chain_thresholds():
    row = score(addition(), [obs("ont"), obs("pb", source="PacBio")])
    assert row["chain_support_single_read"] == "complete"
    assert row["chain_support_threshold"] == "none"


def test_pcr_diagnostic_counts_candidates_without_mutating_or_raising():
    reads = [obs(f"r{i}", run="run1" if i < 2 else "run2") for i in range(3)]
    for offset, o in zip((0, 9, 18), reads):
        o.metadata["protocol"] = "cDNA"
        o.alignment = replace(o.alignment, start=51+offset, end=400+offset)
    # A separate clique in the same library/chain is another candidate.
    far = obs("far")
    far.metadata["protocol"] = "cDNA"
    far.alignment = replace(far.alignment, start=700, end=900)
    specs = [dict(dataset="dataset", run=r, arm="primary") for r in ("run1", "run2", "empty")]
    additions = [addition(), addition(addition_id="away", locus_start=910, locus_end=990,
                                     cds_start=920, cds_end=980, introns=(), novel_introns=(), novelty="reference-alt")]
    before = [m.asdict(o) for o in reads + [far]]
    d = m.pcr_diagnostics(reads + [far], additions, specs)
    assert (d["candidate_equivalence_groups"], d["non_clique_groups"]) == (2, 1)
    assert [r["non_clique_groups"] for r in d["per_run"]] == [1, 1, 0]
    assert [r["non_clique_groups"] for r in d["per_addition"]] == [1, 0]
    assert before == [m.asdict(o) for o in reads + [far]]
    assert d == m.pcr_diagnostics(list(reversed(reads + [far])), additions, specs)
    with pytest.raises(m.RuleUnresolved):
        m.assign_molecules(reads)


def test_est_arms_crossed_with_both_readings_in_every_rate():
    observations = [obs("primary", source="EST"), obs("sensitivity", source="EST")]
    observations[1].metadata["arm"] = "min121"
    specs = [dict(species="synthetic", source="EST", dataset="dataset", run="run1", arm=arm,
                  genotype_stratum="reference", annotation_independence="independent", model_independent=True)
             for arm in ("primary", "min121")]
    rows = m.score_scopes([addition()], m.assign_molecules(observations), specs)
    assert all(set(r["chain_reading_report"]) == set(m.CHAIN_READINGS) for r in rows)
    tables = m.table_inputs(rows)
    assert {(r["arm"], r["chain_reading"]) for r in tables} == {
        (a, c) for a in ("primary", "min121") for c in m.CHAIN_READINGS}
    for r in tables:
        assert r["chain_reading_label"] == m.CHAIN_READINGS[r["chain_reading"]]
    chain = [r for r in tables if r["metric"] == "chain" and r["scope"] == "EST"]
    assert all(r["numerator"] == (1 if r["chain_reading_label"] == "primary" else 0) for r in chain)


def make_b1_inputs(tmp_path):
    p, config = synthetic_config(tmp_path)
    def write(name, value):
        path = tmp_path / name
        path.write_text(value)
        return dict(path=name, sha256=m.digest(path))
    config["purpose"] = "b1"
    config["qc_gate"] = write("qc.json", json.dumps(dict(passed=True, seed=123, loci=2000, agreement=.99)))
    config["positive_control_seed"] = 123
    controls = [addition(addition_id=f"p{i}", control="positive") for i in range(500)]
    controls += [addition(), addition(addition_id="negative", control="negative")]
    config["additions"] = write("additions.json", json.dumps([m.asdict(a) for a in controls]))
    spec = config["runs"][0]
    spec["done"] = write("DONE", "bam_md5=" + m.digest(tmp_path / "run.sam", "md5") +
                         "\nuf=none\naudit_status=FAIL\n")
    spec["provenance"] = write("run.provenance", "fixture\n")
    spec["orientation_audit"] = write("audit.json", json.dumps(dict(status="FAIL")))
    p.write_text(json.dumps(config))
    return p, config


def test_b1_incomplete_work_still_refuses_without_scores(tmp_path):
    p, config = make_b1_inputs(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(m.EvidenceError, match="B1 incomplete implementation"):
        m.execute(p, out)
    report = json.loads((out / "PROVENANCE.json").read_text())
    assert report["status"] == "scoring_refused"
    assert report["incomplete_work"] == m.INCOMPLETE_WORK
    assert report["pcr_diagnostic"]["non_clique_groups"] == 0
    assert not (out / "DONE").exists() and not (out / "per_addition.csv").exists()


def test_b1_missing_orientation_audit_still_refuses(tmp_path):
    p, config = make_b1_inputs(tmp_path)
    del config["runs"][0]["orientation_audit"]
    p.write_text(json.dumps(config))
    with pytest.raises(m.EvidenceError, match="frozen sha256"):
        m.execute(p, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_nonclique_cli_keeps_diagnostics_but_no_scores(tmp_path):
    p, config = synthetic_config(tmp_path)
    alignment = tmp_path / "run.sam"
    alignment.write_text("".join(sam(name=f"r{i}", start=51+offset,
        cigar=f"{50-offset}M100N50M100N{50+offset}M",
        cs=f":{50-offset}~gt100ag:50~gt100ag:{50+offset}") for i, offset in enumerate((0, 9, 18))))
    config["runs"][0]["alignment"]["sha256"] = m.digest(alignment)
    meta_path = tmp_path / "meta.json"
    metadata = json.loads(meta_path.read_text())
    for v in metadata.values():
        v["protocol"] = "cDNA"
    meta_path.write_text(json.dumps(metadata))
    config["runs"][0]["metadata"]["sha256"] = m.digest(meta_path)
    p.write_text(json.dumps(config))
    out = tmp_path / "out"
    assert m.main(["--config", str(p), "--out", str(out)]) == 2
    report = json.loads((out / "PROVENANCE.json").read_text())
    assert report["pcr_diagnostic"]["non_clique_groups"] == 1
    assert report["pcr_diagnostic"]["per_addition"][0]["non_clique_groups"] == 1
    assert set(f.name for f in out.iterdir()) == {
        "PCR_diagnostic_runs.csv", "PCR_diagnostic_additions.csv", "PROVENANCE.json"}
    for name, sha in report["outputs_sha256"].items():
        assert m.digest(out / name) == sha



def test_tier_disagreement_flag_even_when_both_chains_complete():
    reads = [obs("ont"), obs("est1", source="EST", library="lib1"),
             obs("est2", source="EST", library="lib2")]
    row = score(addition(), reads)
    assert row["chain_support_single_read"] == row["chain_support_threshold"] == "complete"
    assert row["chain_reading_report"]["chain_support_single_read"]["tier"] == "T1"
    assert row["chain_reading_report"]["chain_support_threshold"]["tier"] == "T2"
    assert row["tier_disagreement"]
