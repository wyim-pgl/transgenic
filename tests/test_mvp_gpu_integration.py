"""CPU integration boundaries only; never produce a successful GPU receipt."""
import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'revision/scripts/mvp'
sys.path.insert(0, str(TOOLS))
import real_probes as p
import resource_probe as r


def test_resource_uses_bound_factory_and_validates_before_wrap():
    tree = ast.parse(inspect.getsource(r.run))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    factory = next(n for n in calls if ast.unparse(n.func) == 'p.datasets')
    assert [ast.unparse(a) for a in factory.args] == ['args', 'bindings', 'cfg', 'model.config']
    validators = [n for n in calls if ast.unparse(n.func) == 'validate_decoder_dataset']
    assert {ast.unparse(n.args[0]) for n in validators} == {'ds', 'valid'}
    prepare = next(n for n in calls if ast.unparse(n.func) == 'accelerator.prepare')
    assert factory.lineno < prepare.lineno
    assert all(ast.unparse(n.args[2]) == 'model_config' for n in validators)


def test_resource_dispatch_is_early_and_offline(monkeypatch):
    monkeypatch.setattr(p, 'parser', lambda: SimpleNamespace(parse_args=lambda argv: SimpleNamespace(mode='resource_budget')))
    def dispatched(args):
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
        raise RuntimeError('resource owner reached')
    monkeypatch.setattr(r, 'run', dispatched)
    monkeypatch.setattr(p, 'require_cuda', lambda: pytest.fail('normal main must not initialize resource'))
    with pytest.raises(RuntimeError, match='resource owner reached'):
        p.main([])


def test_compiled_main_passes_owned_accelerator():
    tree = ast.parse(inspect.getsource(p.main))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'compiled_update')
    assert [ast.unparse(a) for a in call.args] == ['args', 'bindings', 'model', 'ds', 'accelerator']
    assert list(inspect.signature(p.compiled_update).parameters) == ['args', 'bindings', 'model', 'ds', 'accelerator']


def test_resource_cpu_dispatch_refuses_cuda(monkeypatch, tmp_path):
    import torch
    from test_mvp_resource_probe import fixture
    budget, _ = fixture()
    monkeypatch.delenv('TRANSGENIC_RUN_DIR', raising=False)
    monkeypatch.setenv('MVP_BINDINGS_SHA256', 'a' * 64)
    monkeypatch.setenv('MVP_PROBE_TIMEOUT_SECONDS', str(budget['max_probe_wall_seconds']))
    monkeypatch.setattr(r, 'json_snapshot', lambda *args: budget)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    args = SimpleNamespace(mode='resource_budget', **{name: str(tmp_path / name) for name in (
        'budget_authorization', 'budget_sha256', 'profile_dir', 'acceptance_record',
        'inventory_record', 'receipts_record', 'panel', 'sif')})
    monkeypatch.setattr(p, 'parser', lambda: SimpleNamespace(parse_args=lambda argv: args))
    with pytest.raises(RuntimeError, match='CUDA-only'):
        p.main([])
    assert not list(tmp_path.iterdir())


def test_resource_real_factories_both_decoder_contracts(monkeypatch, tmp_path):
    import duckdb
    import torch
    from test_mvp_dataset import DatasetTests
    from transgenic.training import mvp_cli_runtime as runtime, mvp_runtime, mvp_finalize as f, mvp_cli
    from transgenic.training.b5_runtime import load_b5_config
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from test_mvp_resource_probe import fixture
    data = DatasetTests()
    data.setUp()
    budget, _ = fixture()
    cfg = load_b5_config(str(TOOLS.parents[2] / 'configs/b5_400m_win_v3.json'))
    model_config = HyenaTransgenicConfig(**cfg)
    class StopInventory(Exception):
        pass
    counts = dict(create=0, end=0, prepare=0)
    class Accelerator:
        process_index = 0
        is_main_process = True
        device = torch.device('cpu')
        def prepare(self, model, optimizer):
            counts['prepare'] += 1
            assert observed == ['train']
            return model, optimizer
        def wait_for_everyone(self):
            pass
        def end_training(self):
            counts['end'] += 1
    def make(*args, **kwargs):
        counts['create'] += 1
        return Accelerator()
    observed, datasets = [], []
    original = runtime.validate_decoder_dataset
    def validate(ds, bound, actual):
        assert bound is cfg and actual is model_config
        original(ds, bound, actual)
        assert ds[0][2][0].tolist() == [0, 289, 2]
        observed.append(ds.split)
        datasets.append(ds)
        if ds.split == 'valid':
            raise StopInventory('both actual factories reached')
    args = SimpleNamespace(db=data.db, evidence_db=data.ev, scratch_root=str(tmp_path),
        profile_dir=str(tmp_path / 'profile'), budget_authorization='budget', budget_sha256='b'*64,
        acceptance_record='accept', inventory_record='inventory', receipts_record='receipts', panel='panel', sif='sif')
    bindings = dict(seed=1, technical_acceptance_sha256='accept', inventory_sha256='inventory', receipts_sha256='receipts', sif_sha256='sif')
    monkeypatch.delenv('TRANSGENIC_RUN_DIR', raising=False)
    monkeypatch.setenv('MVP_BINDINGS_SHA256', 'a'*64)
    monkeypatch.setenv('MVP_PROBE_TIMEOUT_SECONDS', str(budget['max_probe_wall_seconds']))
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(r, 'json_snapshot', lambda *a: budget)
    monkeypatch.setattr(p, 'require_cuda', lambda: None)
    for name in ('set_device', 'reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *a: SimpleNamespace(total_memory=100))
    monkeypatch.setattr(mvp_runtime, 'make_accelerator', make)
    monkeypatch.setattr(f, '_collective_preflight', lambda accelerator, operation: operation())
    monkeypatch.setattr(runtime, '_collect_runtime_hardware', lambda *a: [])
    monkeypatch.setattr(runtime, '_validate_public_hardware', lambda x: x)
    monkeypatch.setattr(runtime, 'require_matching_identity', lambda *a: None)
    monkeypatch.setattr(runtime, 'validate_decoder_dataset', validate)
    monkeypatch.setattr(p, 'setup', lambda *a: (bindings, {}, {}))
    model = SimpleNamespace(config=model_config, gradient_checkpointing_enable=lambda **kw: None)
    monkeypatch.setattr(p, 'load_model', lambda *a: (model, cfg, {}))
    monkeypatch.setattr(mvp_runtime, 'build_optimizer', lambda *a: object())
    monkeypatch.setattr(runtime, 'compile_bound_forward', lambda *a: None)
    monkeypatch.setattr(r, 'strict_compile_limits', lambda *a: None)
    monkeypatch.setattr(mvp_cli, 'validate_candidate_records', lambda *a: SimpleNamespace(identities=dict(acceptance='accept', inventory='inventory', receipts='receipts')))
    monkeypatch.setattr(runtime, 'verify_candidate_artifacts', lambda *a: {})
    monkeypatch.setattr(r, '_sha256', lambda *a: 'sif')
    try:
        with duckdb.connect(data.db) as con:
            con.execute("UPDATE geneList SET gff='<empty>'")
        with pytest.raises(StopInventory, match='both actual factories reached'):
            r.run(args)
        assert observed == ['train', 'train', 'valid']  # shared factory + resource train/valid checks
        assert counts == dict(create=1, end=1, prepare=1)
        assert all(ds._con is None for ds in datasets) if hasattr(datasets[0], '_con') else True
        assert not (tmp_path / 'profile/PROBE_ONLY_CHECKPOINT').exists()
    finally:
        data.doCleanups()
