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
    pairfile = tmp_path / 'pairs.jsonl'
    pairfile.write_text(json.dumps(record(1, .002)) + '\n')
    counts = dict(f1_tp=1, f1_predicted=2, f1_reference=2)
    payload = dict(pairs={str(s): str(pairfile) for s in (1, 2, 3)},
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
    assert result['structural_available'] is False
    assert result['readouts']['all50']['F1']['pooled']['point'] == 0
    assert result['readouts']['core20']['F1']['pooled']['n_loci'] == 20
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
