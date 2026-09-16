"""Review R1/R2 CPU regressions; no actual SIF/CUDA certification."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys

import duckdb
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'revision/scripts/mvp'
sys.path.insert(0, str(TOOLS))
import real_probes as probes
from transgenic.training.mvp_cli_runtime import validate_decoder_dataset, _close_dataset
from transgenic.training.b5_runtime import load_b5_config


@pytest.fixture
def fixture():
    import test_mvp_dataset
    value = test_mvp_dataset.DatasetTests()
    value.setUp()
    yield value
    value.doCleanups()


def config():
    return load_b5_config(str(TOOLS.parents[2] / 'configs/b5_400m_win_v3.json'))


@pytest.mark.parametrize('label', [
    '<empty>',
    '1|CDS1|9|+|A>CDS1<gene>20|CDS1|29|+|A>CDS1',
    '<gene>'.join(['1|CDS1|9|+|A>CDS1'] * 300),
    '<gene>'.join(['1|CDS1|9|+|A>CDS1'] * 1000),
], ids=['empty', 'two-gene', 'over2048', 'cap8192'])
def test_bound_v3_real_gff_labels_not_default_v2(fixture, label):
    with duckdb.connect(fixture.db) as con:
        con.execute('UPDATE geneList SET gff=?', [label])
    cfg = config()
    ds = probes.datasets(SimpleNamespace(db=fixture.db, evidence_db=fixture.ev),
                         {'seed': 1}, cfg, SimpleNamespace(**cfg))
    try:
        tokens = ds.dt._tokenize(label)
        expected = [ds.dt._convert_token_to_id(x) for x in tokens]
        if len(expected) > 8192:
            expected = expected[:8191] + [ds.dt.vocab['</s>']]
        assert ds[0][2][0].tolist() == expected
        assert ds.gff_vocab_version == 'v3' and len(ds.dt) == 290 and ds.maxlength == 8192
        if label == '<empty>':
            assert expected == [0, 289, 2]
        if label.count('<gene>') > 100:
            assert len(expected) > 2048
        elif '<gene>' in label:
            assert expected.count(ds.dt.vocab['<gene>']) == 1
    finally:
        _close_dataset(ds)


@pytest.mark.parametrize('field,value', [('vocab_size', 288), ('max_label_tokens', 2048),
    ('max_decoder_position_embeddings', 2048), ('gff_vocab_version', 'v2')])
def test_decoder_binding_disagreement_rejected(fixture, field, value):
    cfg = config()
    ds = fixture.dataset(gff_vocab_version='v3')
    broken = dict(cfg, **{field: value})
    with pytest.raises(ValueError, match='decoder vocabulary/label cap'):
        validate_decoder_dataset(ds, broken, SimpleNamespace(**cfg))


def test_inherited_encoder_report_directory_refused(monkeypatch):
    monkeypatch.setattr(probes, 'require_cuda', lambda: None)
    monkeypatch.setattr(probes, 'require_allocation', lambda: None)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setenv('TRANSGENIC_RUN_DIR', '/B5/must/not/write')
    with pytest.raises(ValueError, match='override environment'):
        probes.setup(SimpleNamespace())
    from transgenic.training.mvp_cli_runtime import _train
    with pytest.raises(ValueError, match='TRANSGENIC_RUN_DIR'):
        _train(1, 0, None, None, None, None, None, None)


@pytest.mark.parametrize('case,message', [('loader', 'injected rank1 loader failure'),
    ('compiler', 'no observed Inductor kernel generation')])
def test_real_two_rank_local_failure_no_done(tmp_path, monkeypatch, case, message):
    import run_bounded
    from make_launch_readiness import tooling_identity, _canonical_bytes
    monkeypatch.setenv('MVP_DATA_MOVER_ALLOCATION_ID', 'CPU_TEST_NOT_GPU_CERTIFICATION')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(TOOLS.parents[2] / '.testdeps'),
                                                   str(TOOLS.parents[2] / 'src')]))
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    bindings = {'tooling': tooling_identity()}
    prepared = {'bindings': bindings, 'bindings_sha256': hashlib.sha256(_canonical_bytes(bindings)).hexdigest()}
    def put(name, value):
        path = tmp_path / name
        payload = _canonical_bytes(value)
        path.write_bytes(payload)
        return str(path), hashlib.sha256(payload).hexdigest()
    binding_path, binding_sha = put('bindings.json', prepared)
    command_path, command_sha = put('command.json', [sys.executable, '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=2', str(Path(__file__).with_name('mvp_real_probe_faults.py')),
        case, str(tmp_path)])
    out = tmp_path / 'bounded'
    assert run_bounded.main(['--name', 'four_rank_smoke', '--bindings', binding_path,
        '--bindings-file-sha256', binding_sha, '--command-file', command_path,
        '--command-sha256', command_sha, '--output-dir', str(out), '--timeout-seconds', '90']) == 2
    for rank in (0, 1):
        assert message in json.loads((tmp_path / f'rank{rank}.json').read_text())['error']
    assert not list(out.glob('*.DONE'))
    failed = json.loads((out / 'four_rank_smoke.FAILED.json').read_text())
    assert failed['exit_code'] != 0 and failed['timed_out'] is False
