"""CPU fixtures only: binds independent junction tables to structural counts.

No measured biological outcome, no coverage-policy approval and no decoded
model output is created or implied here.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from transgenic.evidence import mvp_windows as w

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'revision/scripts/mvp' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def locus():
    return w.PanelLocus('mvp_core', 'g', 'Chr1', 30001, 31000, '+', 1000)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_table(tmp_path, rows, name='junctions.tsv'):
    table = script('mvp_junction_table')
    path = tmp_path / name
    table.write_table(rows, path)
    return path


def row(start0, end0, n_ont=0, n_hifi=0, chrom='Chr1', strand='+'):
    return dict(chrom=chrom, intron_start0=start0, intron_end0=end0, tx_strand=strand,
                n_ONT=n_ont, n_HiFi=n_hifi, supported=(n_ont >= 3 or n_hifi >= 2))


def write_report(tmp_path, table_path, *, run='SRR28341943', source='ONT',
                 panel_sha256=None, name='alignment.json', extra=None):
    report = dict(run=run, source=source, use_uf=False,
                  panel_sha256=panel_sha256 or w.PANEL_SHA256,
                  coverage_policy='PENDING',
                  reference_identity_status='UNVERIFIED',
                  status='PREPARATORY_COMMANDS_COMPLETED',
                  outputs_md5={Path(table_path).name: md5(table_path)})
    report.update(extra or {})
    path = tmp_path / name
    path.write_text(json.dumps(report, indent=2))
    return path


# ---- binding ---------------------------------------------------------------

def test_binds_single_run_table_to_locus_structural_counts(tmp_path):
    m = script('mvp_junction_binding')
    score = script('mvp_score')
    # one supported junction inside the gene, one supported inside the window only
    table = write_table(tmp_path, [row(16000, 16100, n_ont=3), row(30050, 30149, n_ont=3)])
    report = write_report(tmp_path, table)
    run = m.load_run(table, report)
    assert run.run_id == 'SRR28341943' and run.source == 'ONT'
    supported = m.supported_set(m.bind([run]))
    predicted = {('Chr1', 30050, 30149, '+'), ('Chr1', 16000, 16100, '+')}
    counts = m.locus_counts({'g': predicted}, supported, [locus()])
    assert counts['g'] == score.junction_counts(predicted, supported, locus())
    assert counts['g'] == dict(predicted=2, precision_tp=2, supported=1, recall_tp=1)


def test_unsupported_junction_rows_are_excluded_from_the_supported_set(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=2), row(30300, 30399, n_ont=3)])
    run = m.load_run(table, write_report(tmp_path, table))
    assert m.supported_set(m.bind([run])) == frozenset({('Chr1', 30300, 30399, '+')})


def test_missing_independent_table_yields_none_counts_not_zero(tmp_path):
    m = script('mvp_junction_binding')
    counts = m.locus_counts({'g': {('Chr1', 30050, 30149, '+')}}, None, [locus()])
    assert counts == {'g': None}


# ---- fail-closed provenance ------------------------------------------------

def test_refuses_table_not_bound_to_its_alignment_report(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=3)])
    report = write_report(tmp_path, table)
    write_table(tmp_path, [row(30050, 30149, n_ont=3), row(30300, 30399, n_ont=3)])
    with pytest.raises(m.JunctionBindingError, match='md5'):
        m.load_run(table, report)


def test_refuses_panel_sha256_drift(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=3)])
    report = write_report(tmp_path, table, panel_sha256='0' * 64)
    with pytest.raises(m.JunctionBindingError, match='panel'):
        m.load_run(table, report)


def test_refuses_unknown_run_or_source_mismatch(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=3)])
    with pytest.raises(m.JunctionBindingError, match='run'):
        m.load_run(table, write_report(tmp_path, table, run='SRRnotareadrun'))
    with pytest.raises(m.JunctionBindingError, match='source'):
        m.load_run(table, write_report(tmp_path, table, run='SRR28341943', source='HiFi'))


def test_refuses_duplicate_run_identity(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=3)])
    run = m.load_run(table, write_report(tmp_path, table))
    with pytest.raises(m.JunctionBindingError, match='duplicate'):
        m.bind([run, run])


def test_refuses_pooling_runs_without_approved_coverage_policy(tmp_path):
    m = script('mvp_junction_binding')
    a = write_table(tmp_path, [row(30050, 30149, n_ont=2)], name='a.tsv')
    b = write_table(tmp_path, [row(30050, 30149, n_ont=2)], name='b.tsv')
    first = m.load_run(a, write_report(tmp_path, a, run='SRR28341943', name='a.json'))
    second = m.load_run(b, write_report(tmp_path, b, run='SRR28341944', name='b.json'))
    with pytest.raises(ValueError, match='coverage policy'):
        m.bind([first, second])


def approved_policy():
    return json.loads((ROOT / 'revision/scripts/mvp/coverage_policy_a47_2.json').read_text())


def two_ont_runs(tmp_path, n_ont=2):
    m = script('mvp_junction_binding')
    a = write_table(tmp_path, [row(30050, 30149, n_ont=n_ont)], name='a.tsv')
    b = write_table(tmp_path, [row(30050, 30149, n_ont=n_ont)], name='b.tsv')
    return (m.load_run(a, write_report(tmp_path, a, run='SRR28341943', name='a.json')),
            m.load_run(b, write_report(tmp_path, b, run='SRR28341944', name='b.json')))


def ccs_run(tmp_path, rows, run='SRR22719002', name='h'):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, rows, name=name + '.tsv')
    return m.load_run(table, write_report(tmp_path, table, run=run, source='HiFi',
                                          name=name + '.json'))


def ont_run(tmp_path, rows, run='SRR28341943', name='o'):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, rows, name=name + '.tsv')
    return m.load_run(table, write_report(tmp_path, table, run=run, name=name + '.json'))


def test_pools_read_counts_within_one_source_under_the_approved_policy(tmp_path):
    m = script('mvp_junction_binding')
    first, second = two_ont_runs(tmp_path, n_ont=2)
    # neither run clears the ONT threshold of 3 on its own
    assert m.supported_set(m.bind([first])) == frozenset()
    pooled = m.bind([first, second], policy=approved_policy())
    assert [r['n_ONT'] for r in pooled] == [4]
    assert m.supported_set(pooled) == frozenset({('Chr1', 30050, 30149, '+')})


def test_a_ccs23_runs_pool_within_the_hifi_column(tmp_path):
    m = script('mvp_junction_binding')
    first = ccs_run(tmp_path, [row(30050, 30149, n_hifi=1)], run='SRR22719002', name='h1')
    second = ccs_run(tmp_path, [row(30050, 30149, n_hifi=1)], run='SRR22719007', name='h2')
    pooled = m.bind([first, second], policy=approved_policy())
    assert [(r['n_ONT'], r['n_HiFi'], r['supported']) for r in pooled] == [(0, 2, True)]


def test_junction_with_two_a_ccs23_reads_and_under_three_ont_reads_is_supported(tmp_path):
    m = script('mvp_junction_binding')
    ont = ont_run(tmp_path, [row(30050, 30149, n_ont=2)])
    ccs = ccs_run(tmp_path, [row(30050, 30149, n_hifi=2)])
    bound = m.bind([ont, ccs], policy=approved_policy())
    assert [(r['n_ONT'], r['n_HiFi'], r['supported']) for r in bound] == [(2, 2, True)]
    assert m.supported_set(bound) == frozenset({('Chr1', 30050, 30149, '+')})
    predicted = {('Chr1', 30050, 30149, '+')}
    counts = m.locus_counts({'g': predicted}, m.supported_set(bound), [locus()])
    assert counts['g'] == dict(predicted=1, precision_tp=1, supported=1, recall_tp=1)


def test_one_ccs_plus_two_ont_reads_is_not_supported_no_cross_source_summation(tmp_path):
    m = script('mvp_junction_binding')
    ont = ont_run(tmp_path, [row(30050, 30149, n_ont=2)])
    ccs = ccs_run(tmp_path, [row(30050, 30149, n_hifi=1)])
    bound = m.bind([ont, ccs], policy=approved_policy())
    # 1 + 2 = 3 would clear the ONT threshold if sources were summed; they are not
    assert [(r['n_ONT'], r['n_HiFi'], r['supported']) for r in bound] == [(2, 1, False)]
    assert m.supported_set(bound) == frozenset()


def test_cross_source_counts_are_never_summed_into_one_column(tmp_path):
    m = script('mvp_junction_binding')
    first, second = two_ont_runs(tmp_path, n_ont=1)
    ccs1 = ccs_run(tmp_path, [row(30050, 30149, n_hifi=1), row(30300, 30399, n_hifi=5)], name='h1')
    ccs2 = ccs_run(tmp_path, [row(30300, 30399, n_hifi=1)], run='SRR22719003', name='h2')
    bound = m.bind([first, ccs1, second, ccs2], policy=approved_policy())
    by_key = {(r['intron_start0'], r['intron_end0']): (r['n_ONT'], r['n_HiFi']) for r in bound}
    assert by_key == {(30050, 30149): (2, 1), (30300, 30399): (0, 6)}
    assert m.supported_set(bound) == frozenset({('Chr1', 30300, 30399, '+')})
    with pytest.raises(ValueError, match='cross-source pooling'):
        m.pool([first, ccs1])
    # per-source pooled tables are retained separately, as are per-run rows
    per = m.per_source([first, ccs1, second, ccs2], approved_policy())
    assert [r['n_HiFi'] for r in per['ONT']] == [0] and [r['n_ONT'] for r in per['HiFi']] == [0, 0]
    assert first.rows[0]['n_ONT'] == 1


def test_run_table_with_counts_in_another_sources_column_is_refused(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=1, n_hifi=2)])
    with pytest.raises(m.JunctionBindingError, match="another source's column"):
        m.load_run(table, write_report(tmp_path, table))


def test_unavailable_a_hifi_run_is_refused(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_hifi=2)])
    with pytest.raises(m.JunctionBindingError, match='A-HiFi'):
        m.load_run(table, write_report(tmp_path, table, run='ERR13994458', source='HiFi'))


def test_superseded_v1_policy_cannot_pool(tmp_path):
    m = script('mvp_junction_binding')
    first, second = two_ont_runs(tmp_path)
    v1 = json.loads((ROOT / 'revision/scripts/mvp/coverage_policy_a47.json').read_text())
    with pytest.raises(ValueError, match='A47.2'):
        m.bind([first, second], policy=v1)


def test_receipt_records_the_validated_policy_instead_of_pending(tmp_path):
    m = script('mvp_junction_binding')
    first, second = two_ont_runs(tmp_path)
    policy = approved_policy()
    receipt = m.receipt([first, second], m.bind([first, second], policy=policy), policy=policy)
    assert receipt['coverage_policy']['policy_id'] == policy['policy_id']
    assert receipt['pooled_across_runs'] is True
    # A-CCS23 not bound here: recorded missing, never as zero support
    assert receipt['structural_sources_bound'] == ['ONT']
    assert receipt['structural_sources_missing'] == ['HiFi']


def test_receipt_retains_per_source_and_per_run_results(tmp_path):
    m = script('mvp_junction_binding')
    ont = ont_run(tmp_path, [row(30050, 30149, n_ont=2)])
    ccs = ccs_run(tmp_path, [row(30050, 30149, n_hifi=2)])
    policy = approved_policy()
    receipt = m.receipt([ont, ccs], m.bind([ont, ccs], policy=policy), policy=policy)
    assert receipt['structural_sources_missing'] == []
    assert receipt['per_source'] == {
        'HiFi': dict(dataset='A-CCS23', runs=['SRR22719002'], junctions=1, supported_junctions=1),
        'ONT': dict(dataset='A-ONT1', runs=['SRR28341943'], junctions=1, supported_junctions=0)}
    assert [(r['run'], r['dataset'], r['supported_junctions']) for r in receipt['runs']] == [
        ('SRR28341943', 'A-ONT1', 0), ('SRR22719002', 'A-CCS23', 1)]
    assert receipt['supported_junctions'] == 1


def test_forged_support_column_is_rejected_by_the_frozen_table_reader(tmp_path):
    m = script('mvp_junction_binding')
    path = tmp_path / 'junctions.tsv'
    table = script('mvp_junction_table')
    path.write_text('\t'.join(table.HEADER) + '\nChr1\t30050\t30149\t+\t1\t0\ttrue\n')
    with pytest.raises(ValueError):
        m.load_run(path, write_report(tmp_path, path))


# ---- receipt ---------------------------------------------------------------

def test_receipt_records_read_counts_run_identity_and_unverified_provenance(tmp_path):
    m = script('mvp_junction_binding')
    table = write_table(tmp_path, [row(30050, 30149, n_ont=3), row(30300, 30399, n_ont=1)])
    run = m.load_run(table, write_report(tmp_path, table))
    receipt = m.receipt([run], m.bind([run]))
    assert receipt['counts_are'] == 'reads (UMI status unknown; never molecules)'
    assert receipt['coverage_policy'] == 'NONE_SUPPLIED_SINGLE_RUN_ONLY'
    assert receipt['pooled_across_runs'] is False
    assert receipt['run_provenance_status'] == 'UNVERIFIED'
    assert receipt['panel_sha256'] == w.PANEL_SHA256
    assert receipt['junctions'] == 2 and receipt['supported_junctions'] == 1
    assert receipt['runs'] == [dict(run=run.run_id, source='ONT', dataset='A-ONT1', use_uf=False,
                                    table_sha256=run.table_sha256, supported_junctions=1)]
