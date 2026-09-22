"""CPU fixtures only: no measured biological outcomes."""
import importlib.util
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from transgenic.evidence import mvp_bootstrap as b, mvp_windows as w

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'revision/scripts/mvp' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def locus():
    return w.PanelLocus('mvp_core', 'g', 'Chr1', 30001, 31000, '+', 1000)


def record(rn, delta, denominator=1, evidence=True):
    return dict(rn=rn, has_evidence=evidence, loss_off=1 + delta, loss_on=1.,
                composite_loss_off=100., composite_loss_on=1.,
                ce_numerator_off=(1 + delta) * denominator, ce_numerator_on=denominator,
                ce_denominator_off=denominator, ce_denominator_on=denominator)


def test_centered_window_prefilter_and_distinct_scopes():
    x = locus()
    assert w.evaluation_window(x) == (15140, 45860)
    assert w.prefilter_interval(w.evaluation_window(x)) == (5140, 55860)
    assert w.junction_within_windows('Chr1', 16000, 16100, w.windows_by_chromosome([x]))
    assert not w.junction_within_genes('Chr1', 16000, 16100, w.genes_by_chromosome([x]))
    assert w.transcript_overlaps_gene('Chr1', 29990, 30000, w.genes_by_chromosome([x]))
    with pytest.raises(w.WindowError):
        w.evaluation_window(x, 45859)


def test_primary_equal_positive_tiles_not_tokens_or_composite():
    records = {s: [record(1, .01), record(2, .03, 100), record(3, 9, evidence=False)] for s in (1, 2, 3)}
    rns, delta = b.build_tiles(records)
    assert rns == (1, 2)
    result = b.tile_bootstrap(delta)
    assert result['pooled_point'] == pytest.approx(.02)
    assert result['pooled_lower95'] == pytest.approx(.01)
    assert result['pooled_upper95'] == pytest.approx(.03)
    assert result == b.tile_bootstrap(delta)
    assert result['replicates'] == 2000
    with pytest.raises(b.BootstrapError):
        b.build_tiles({1: records[1], 2: records[2]})


@pytest.mark.parametrize('updates,want', [
    ({}, 'STOP'),
    ({'structural_available': False}, 'INCONCLUSIVE'),
    ({'withholding_state': 'unassessable', 'withholding_primary_upper': None}, 'INCONCLUSIVE'),
    ({'withholding_primary_upper': .005}, 'GO'),
    ({'delta_f1_lower': .001}, 'GO'),
    ({'delta_r_lower': .001, 'delta_p_point': 0}, 'GO'),
    ({'delta_r_lower': .001, 'delta_p_point': -.001}, 'STOP'),
    ({'primary_lower': 0, 'delta_f1_lower': 1}, 'INCONCLUSIVE'),
    ({'seed_points': (.002, .002, 0), 'primary_upper': .006}, 'INCONCLUSIVE'),
])
def test_stop_first(updates, want):
    args = dict(primary_lower=.001, primary_upper=.004, seed_points=(.002,) * 3,
                delta_f1_lower=0, structural_available=True, delta_r_lower=0,
                delta_p_point=0, withholding_state='assessed', withholding_primary_upper=.004)
    args.update(updates)
    assert b.decide(b.DecisionInputs(**args))[0] == want


def test_junction_real_sam_threshold_strand_and_duplicate_inputs(tmp_path):
    m = script('mvp_junction_table')
    p = tmp_path / 'reads.sam'
    p.write_text(''.join(f'r{i}\t16\tChr1\t30001\t60\t50M100N50M\t*\t0\t0\t*\t*\tts:A:+\n' for i in range(3)))
    rows, qc = m.build_table([('ONT', False, p)], w.windows_by_chromosome([locus()]))
    assert rows == [dict(chrom='Chr1', intron_start0=30050, intron_end0=30149,
                         tx_strand='-', n_ONT=3, n_HiFi=0, supported=True)]
    with pytest.raises(ValueError, match='duplicate'):
        m.build_table([('ONT', False, p), ('ONT', False, p)], w.windows_by_chromosome([locus()]))


def test_junction_table_rejects_forged_support(tmp_path):
    m = script('mvp_junction_table')
    p = tmp_path / 'junctions.tsv'
    p.write_text('\t'.join(m.HEADER) + '\nChr1\t100\t199\t+\t1\t0\ttrue\n')
    with pytest.raises(ValueError):
        m.read_table(p)


def test_junction_hifi_threshold_and_chimeric_read_rejection(tmp_path):
    m = script('mvp_junction_table')
    p = tmp_path / 'reads.sam'
    lines = [f'r{i}\t0\tChr1\t30001\t60\t50M100N50M\t*\t0\t0\t*\t*\tts:A:+\n' for i in range(3)]
    lines += ['r2\t2048\tChr1\t40001\t60\t100M\t*\t0\t0\t*\t*\tts:A:+\n']
    p.write_text(''.join(lines))
    rows, qc = m.build_table([('HiFi', False, p)], w.windows_by_chromosome([locus()]))
    assert rows[0]['n_HiFi'] == 2
    assert rows[0]['supported'] is True
    rows, qc = m.build_table([('ONT', False, p)], w.windows_by_chromosome([locus()]))
    assert rows[0]['n_ONT'] == 2
    assert rows[0]['supported'] is False


def test_three_seed_contract_not_overrideable():
    with pytest.raises(b.BootstrapError):
        b.build_tiles({1: [record(1, .01)]}, expected_seeds=(1,))


def test_scoring_missing_independent_not_negative():
    m = script('mvp_score')
    assert m.junction_counts({('Chr1', 30050, 30149, '+')}, None, locus()) is None
    counts = m.junction_counts({('Chr1', 16000, 16100, '+'), ('Chr1', 30050, 30149, '+')},
                               {('Chr1', 16000, 16100, '+')}, locus())
    assert counts == dict(predicted=2, precision_tp=1, supported=0, recall_tp=0)
    with pytest.raises(ValueError, match='coverage policy'):
        m.require_coverage_policy(None)


def approved_coverage_policy(**overrides):
    m = script('mvp_score')
    policy = dict(policy_id=m.COVERAGE_POLICY_ID,
                  structural_sources=list(m.COVERAGE_STRUCTURAL_SOURCES),
                  source_sets={'ONT': 'A-ONT1', 'HiFi': 'A-CCS23'},
                  thresholds={'ONT': 3, 'HiFi': 2},
                  unavailable_sources=list(m.COVERAGE_UNAVAILABLE_SOURCES),
                  pooling=m.COVERAGE_POOLING,
                  per_run_retained=True,
                  excluded_judgements={'HiFi': ['TES']})
    policy.update(overrides)
    return policy


SHIPPED_POLICY = ROOT / 'revision/scripts/mvp/coverage_policy_a47_2.json'
SUPERSEDED_POLICY = ROOT / 'revision/scripts/mvp/coverage_policy_a47.json'


def test_coverage_policy_accepts_the_a47_2_contract_and_normalises_it():
    m = script('mvp_score')
    contract = m.require_coverage_policy(approved_coverage_policy())
    assert contract == dict(policy_id='B1-A47.2-v1.38/MVP-v2', structural_sources=['ONT', 'HiFi'],
                            source_sets={'ONT': 'A-ONT1', 'HiFi': 'A-CCS23'},
                            thresholds={'ONT': 3, 'HiFi': 2}, unavailable_sources=['A-HiFi'],
                            pooling='within_source', per_run_retained=True,
                            excluded_judgements={'HiFi': ['TES']})


def test_coverage_policy_rejects_the_superseded_a47_v1_contract():
    m = script('mvp_score')
    v1 = json.loads(SUPERSEDED_POLICY.read_text())
    assert v1['policy_id'] == 'B1-A47-v1.37/MVP-v1'
    with pytest.raises(ValueError, match='superseded by PROTOCOL_B1 A47.2'):
        m.require_coverage_policy(v1)


def test_coverage_policy_rejects_ont_only_structural_sources():
    m = script('mvp_score')
    with pytest.raises(ValueError, match='structural sources'):
        m.require_coverage_policy(approved_coverage_policy(
            structural_sources=['ONT'], thresholds={'ONT': 3}))


def test_coverage_policy_rejects_a_hifi_column_filled_by_the_unavailable_a_hifi():
    m = script('mvp_score')
    with pytest.raises(ValueError, match='source sets'):
        m.require_coverage_policy(approved_coverage_policy(
            source_sets={'ONT': 'A-ONT1', 'HiFi': 'A-HiFi'}))


@pytest.mark.parametrize('thresholds', [{'ONT': 1, 'HiFi': 2}, {'ONT': 3, 'HiFi': 3},
                                        {'ONT': 3}, {'ONT': 3, 'HiFi': 2, 'PacBio': 2}])
def test_coverage_policy_rejects_thresholds_that_contradict_the_frozen_scorer(thresholds):
    m = script('mvp_score')
    with pytest.raises(ValueError, match='frozen'):
        m.require_coverage_policy(approved_coverage_policy(thresholds=thresholds))


def test_coverage_policy_rejects_an_unavailable_source_recorded_as_available():
    m = script('mvp_score')
    with pytest.raises(ValueError, match='never as zero'):
        m.require_coverage_policy(approved_coverage_policy(unavailable_sources=[]))
    with pytest.raises(ValueError, match='never as zero'):
        m.require_coverage_policy(approved_coverage_policy(unavailable_sources=['HiFi']))


def test_coverage_policy_rejects_cross_source_pooling():
    m = script('mvp_score')
    with pytest.raises(ValueError, match='pooling'):
        m.require_coverage_policy(approved_coverage_policy(pooling='across_sources'))


def test_coverage_policy_requires_the_a_ccs23_tes_exclusion():
    m = script('mvp_score')
    for excluded in ({}, {'HiFi': []}, {'HiFi': ['TSS']}):
        with pytest.raises(ValueError, match='TES'):
            m.require_coverage_policy(approved_coverage_policy(excluded_judgements=excluded))


def test_shipped_coverage_policy_file_is_the_approved_contract():
    m = script('mvp_score')
    shipped = json.loads(SHIPPED_POLICY.read_text())
    assert m.require_coverage_policy(shipped)['policy_id'] == m.COVERAGE_POLICY_ID


def test_coverage_policy_requires_every_field_to_be_stated():
    m = script('mvp_score')
    partial = approved_coverage_policy()
    del partial['per_run_retained']
    with pytest.raises(ValueError, match='missing'):
        m.require_coverage_policy(partial)


def test_bootstrap_carries_seeds_together_and_matches_rng():
    deltas = np.array([[1., 2., -3.], [4., 5., -9.], [7., 8., -15.]])
    result = b.tile_bootstrap(deltas)
    assert result['pooled_lower95'] == result['pooled_upper95'] == 0
    rng = np.random.default_rng(20260914)
    reps = [deltas[rng.choice(3, 3, replace=True)].mean(axis=0) for _ in range(2000)]
    assert result['per_seed_lower95'] == pytest.approx(np.percentile(reps, 5, axis=0))
    assert result['per_seed_upper95'] == pytest.approx(np.percentile(reps, 95, axis=0))


def test_schema_mirrors_real_training_metrics():
    from transgenic.training import mvp_metrics
    assert b.PAIRED_NUMERIC_FIELDS == mvp_metrics._NUMERIC_FIELDS


@pytest.mark.parametrize('field', ['primary_lower', 'delta_f1_lower'])
def test_decision_rejects_boolean_statistics(field):
    args = dict(primary_lower=.001, primary_upper=.004, seed_points=(.002,) * 3,
                delta_f1_lower=0, structural_available=False)
    args[field] = True
    with pytest.raises(b.BootstrapError):
        b.decide(b.DecisionInputs(**args))


def test_prediction_selection_is_gene_overlap_not_junction_clipping():
    m = script('mvp_score')
    txs = [dict(chrom='Chr1', strand='+', exons=[(29000, 29100), (30000, 30100)]),
           dict(chrom='Chr1', strand='+', exons=[(16000, 16100), (16200, 16300)])]
    assert m.predicted_junctions(txs, locus()) == {('Chr1', 29100, 29999, '+')}


def test_real_score_cli_full_frozen_panel_missing_structural(tmp_path):
    panel = ROOT / 'revision/protocols/b6_panels_v1.tsv'
    loci = w.load_panel(panel)
    pairs = {}
    for seed in (1, 2, 3):
        pairfile = tmp_path / f'pairs{seed}.jsonl'
        pairfile.write_text(json.dumps(record(1, .002)) + '\n')
        pairs[str(seed)] = str(pairfile)
    counts = dict(f1_tp=1, f1_predicted=2, f1_reference=2)
    payload = dict(pairs=pairs,
                   loci=[dict(gene_id=l.gene_id, seeds={str(s): dict(off=counts, on=counts) for s in (1, 2, 3)}) for l in loci])
    inp = tmp_path / 'input.json'; inp.write_text(json.dumps(payload))
    out = tmp_path / 'out.json'
    proc = subprocess.run([sys.executable, str(ROOT / 'revision/scripts/mvp/mvp_score.py'),
                           '--input', str(inp), '--panels', str(panel), '--out', str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result['primary']['pooled_point'] == pytest.approx(.002)
    assert result['composite_delta_diagnostic'] == 99
    assert result['decision'] is None
    assert_unverified_runs(result)
    assert result['structural_available'] is False
    assert result['readouts']['all50']['F1']['pooled']['point'] == 0
    assert result['readouts']['core20']['F1']['pooled']['n_loci'] == 20
    blocked_out = tmp_path / 'blocked.json'
    blocked = subprocess.run([sys.executable, str(ROOT / 'revision/scripts/mvp/mvp_score.py'),
                              '--input', str(inp), '--panels', str(panel), '--out', str(blocked_out),
                              '--final-decision'], capture_output=True, text=True)
    assert blocked.returncode != 0
    assert 'coverage policy required' in blocked.stderr
    assert not blocked_out.exists()
    # the superseded A47 v1 contract no longer unblocks --final-decision
    v1_out = tmp_path / 'v1.json'
    rejected = subprocess.run([sys.executable, str(ROOT / 'revision/scripts/mvp/mvp_score.py'),
                               '--input', str(inp), '--panels', str(panel), '--out', str(v1_out),
                               '--final-decision', '--coverage-policy', str(SUPERSEDED_POLICY)],
                              capture_output=True, text=True)
    assert rejected.returncode != 0 and 'A47.2' in rejected.stderr
    assert not v1_out.exists()
    # the shipped A47.2 contract unblocks --final-decision and is recorded in the result
    policy_out = tmp_path / 'policy.json'
    accepted = subprocess.run([sys.executable, str(ROOT / 'revision/scripts/mvp/mvp_score.py'),
                               '--input', str(inp), '--panels', str(panel), '--out', str(policy_out),
                               '--final-decision', '--coverage-policy', str(SHIPPED_POLICY)],
                              capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    recorded = json.loads(policy_out.read_text())['coverage_policy']
    assert recorded['structural_sources'] == ['ONT', 'HiFi']
    assert recorded['source_sets'] == {'ONT': 'A-ONT1', 'HiFi': 'A-CCS23'}
    assert recorded['unavailable_sources'] == ['A-HiFi']
    assert recorded['excluded_judgements'] == {'HiFi': ['TES']}
    payload['pairs']['01'] = str(pairfile)
    with pytest.raises(ValueError, match='seeds'):
        script('mvp_score').score(payload, loci)
    del payload['pairs']['01']
    partial = payload['loci'][0]['seeds']['1']['off'].copy()
    partial.update(precision_tp=0, predicted=1, recall_tp=0, supported=1)
    payload['loci'][0]['seeds']['1']['off'] = partial
    with pytest.raises(ValueError, match='partially missing'):
        script('mvp_score').score(payload, loci)
    payload['loci'].pop()
    with pytest.raises(ValueError, match='inventory'):
        script('mvp_score').score(payload, loci)


def assert_unverified_runs(result):
    assert result['pair_run_provenance_status'] == 'UNVERIFIED'
    assert result['three_seed_identity_verified'] is False
    assert result['three_seed_policy_verified'] is False
    assert result['decision'] is None


def score_fixture(tmp_path):
    pairs = {}
    for seed in (1, 2, 3):
        path = tmp_path / f'seed{seed}.jsonl'
        path.write_text(json.dumps(record(1, .002)) + '\n')
        pairs[str(seed)] = str(path)
    counts = dict(f1_tp=1, f1_predicted=2, f1_reference=2,
                  precision_tp=0, predicted=1, recall_tp=0, supported=10)
    row = dict(gene_id='g', seeds={str(s): {f: dict(counts) for f in ('off', 'on')} for s in (1, 2, 3)})
    return dict(pairs=pairs, loci=[row])


def test_recall_true_positives_must_be_precision_subset(tmp_path):
    payload = score_fixture(tmp_path)
    payload['loci'][0]['seeds']['1']['on']['recall_tp'] = 1
    with pytest.raises(ValueError, match='subset'):
        script('mvp_score').score(payload, [locus()])


@pytest.mark.parametrize('field', ['f1_reference', 'supported'])
def test_frozen_denominator_cannot_change_across_seeds(tmp_path, field):
    payload = score_fixture(tmp_path)
    for fraction in ('off', 'on'):
        payload['loci'][0]['seeds']['2'][fraction][field] = 200
    with pytest.raises(ValueError, match='frozen'):
        script('mvp_score').score(payload, [locus()])


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_pair_inputs_cannot_relabel_one_file(tmp_path, alias):
    payload = score_fixture(tmp_path)
    original = Path(payload['pairs']['1'])
    if alias == 'same':
        path = original.parent / '.' / original.name
    else:
        path = tmp_path / 'alias.jsonl'
        if alias == 'symlink':
            path.symlink_to(original)
        else:
            import os
            os.link(original, path)
    payload['pairs']['2'] = str(path)
    with pytest.raises(ValueError, match='duplicate.*pair'):
        script('mvp_score').score(payload, [locus()])


def test_identical_numbers_distinct_files_not_verified_runs_and_zero_not_missing(tmp_path):
    payload = score_fixture(tmp_path)
    for pair in payload['loci'][0]['seeds'].values():
        for counts in pair.values():
            counts['supported'] = 0
    result = script('mvp_score').score(payload, [locus()])
    assert_unverified_runs(result)
    assert result['structural_available'] is True
    assert result['readouts']['all50']['R']['pooled']['status'] == 'UNASSESSABLE'
    assert result['primary']['per_seed_point'] == pytest.approx([.002] * 3)
