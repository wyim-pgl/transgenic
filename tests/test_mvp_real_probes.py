"""CPU contracts only: no test in this file certifies GPU readiness."""
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'revision/scripts/mvp'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('real_probes', TOOLS / 'real_probes.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_cpu_refused_before_assets(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA-only'):
        probe.require_cuda()


def test_native_bf16_must_be_observable(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda **kw: False)
    with pytest.raises(RuntimeError, match='native bf16'):
        probe.require_cuda()


def test_tensor_digest_binds_dtype_shape_and_bytes():
    import torch
    x = torch.tensor([1, 2], dtype=torch.int32)
    assert probe.tensor_digest(x) != probe.tensor_digest(x.reshape(1, 2))
    assert probe.tensor_digest(x) != probe.tensor_digest(x.to(torch.float32))
    assert probe.tensor_digest(x) == probe.tensor_digest(x.clone())


def test_bitwise_negative():
    import torch
    with pytest.raises(ValueError, match='bitwise'):
        probe.equal_tensors(torch.tensor([0.]), torch.tensor([-0.]))
    with pytest.raises(FloatingPointError):
        probe.tensor_digest(torch.tensor([float('nan')]))


def test_resource_not_exposed_without_measured_overheads(capsys):
    with pytest.raises(SystemExit):
        probe.parser().parse_args(['--mode', 'resource_budget'])
    assert "invalid choice: 'resource_budget'" in capsys.readouterr().err


def test_asset_inventory_changed_or_missing(tmp_path):
    p = tmp_path / 'asset'
    p.write_bytes(b'offline')
    inventory = probe.tree_identity(tmp_path)
    p.write_bytes(b'changed')
    assert inventory != probe.tree_identity(tmp_path)
    with pytest.raises(ValueError, match='empty'):
        probe.tree_identity(tmp_path / 'absent')


def test_bound_probe_is_part_of_tool_identity():
    from make_launch_readiness import tooling_identity
    assert str(TOOLS / 'real_probes.py') in tooling_identity()


def test_missing_probe_identity_rejected():
    with pytest.raises(ValueError, match='probe identity'):
        probe.validate_identity({})


def test_actual_runtime_rejects_wrong_versions():
    import torch
    if str(torch.__version__) == '2.14.0a0+4fdf77b940.nv26.08':
        pytest.skip('negative contract is for the local non-SIF test environment')
    with pytest.raises(ValueError, match='actual SIF import versions'):
        probe.runtime_contract({})


def test_identity_rechecks_import_bytes_and_assets(tmp_path):
    asset = tmp_path / 'cache'
    asset.mkdir()
    (asset / 'config.json').write_text('{}')
    imported = tmp_path / 'imported.py'
    imported.write_text('value = 1')
    executable = str(TOOLS / 'real_probes.py')
    value = {'probe_identity': {'schema_version': 1, 'probe_path': executable,
        'imports': {executable: probe._sha256(Path(executable)),
                    str(imported): probe._sha256(imported)},
        'asset_roots': {str(asset): probe.tree_identity(asset)}}}
    probe.validate_identity(value)
    imported.write_text('value = 2')
    with pytest.raises(ValueError, match='import identity changed'):
        probe.validate_identity(value)
    imported.write_text('value = 1')
    (asset / 'new_asset').write_text('added')
    with pytest.raises(ValueError, match='asset identity changed'):
        probe.validate_identity(value)


def test_real_receipt_mode_and_identity_cannot_be_omitted():
    from run_bounded import validate_measurement
    value = dict(device='cuda', real_runtime=True, mode='real_tokenizer',
                 tokenizer_roundtrip=True, hyena_loaded=True)
    with pytest.raises(ValueError, match='probe identity'):
        validate_measurement('real_tokenizer', value)
    with pytest.raises(ValueError, match='mode/receipt mismatch'):
        validate_measurement('bf16_compile', value)


@pytest.fixture
def toy_generation_model(monkeypatch):
    """CPU control-flow fixture only; explicitly NOT real Hyena/SIF validation."""
    import torch
    from types import SimpleNamespace
    from test_mvp_model import tiny_config, TinyHyena
    from transgenic.model import modeling_HyenaTransgenic as modeling
    monkeypatch.setattr(modeling.AutoConfig, 'from_pretrained', lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(modeling.AutoModel, 'from_config', lambda cfg, **k: TinyHyena(cfg))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = modeling.transgenicForConditionalGeneration(tiny_config(evidence_channels=8)).eval()
    ids = torch.tensor([[3, 4, 5, 6] * 6])
    tracks = torch.arange(192, dtype=torch.float32).reshape(1, 24, 8) / 192
    batch = (ids, torch.ones_like(ids), torch.tensor([[2, 7, 9]]), None, None,
             None, None, None, tracks, None)
    yield model, batch
    torch.set_num_threads(old_threads)


def test_cpu_generation_hook_contract_and_projection_restoration(toy_generation_model):
    import torch
    model, batch = toy_generation_model
    original = model.get_parameter(probe.PROJECTION).detach().clone()
    result = probe.generation(model, batch, torch.device('cpu'))
    assert result['encoder_activation_sha256']['on'] != result['encoder_activation_sha256']['off']
    assert len(result['generated']) == 8
    assert result['reorder_indices_sha256']
    assert torch.equal(original, model.get_parameter(probe.PROJECTION))
    assert '_reorder_cache' not in model.__dict__


def test_generation_dropped_est_is_fatal(toy_generation_model, monkeypatch):
    import torch
    model, batch = toy_generation_model
    original = model.generate
    def drop(**kwargs):
        kwargs.pop('evidence_tracks', None)
        return original(**kwargs)
    monkeypatch.setattr(model, 'generate', drop)
    with pytest.raises(ValueError, match='dropped EST tracks'):
        probe.generation(model, batch, torch.device('cpu'))
    assert torch.count_nonzero(model.get_parameter(probe.PROJECTION)) == 0


def test_generation_failure_restores_projection(toy_generation_model, monkeypatch):
    import torch
    model, batch = toy_generation_model
    original = model.get_parameter(probe.PROJECTION).detach().clone()
    def fail(**kwargs):
        raise RuntimeError('injected generation failure')
    monkeypatch.setattr(model, 'generate', fail)
    with pytest.raises(RuntimeError, match='injected generation failure'):
        probe.generation(model, batch, torch.device('cpu'))
    assert torch.equal(original, model.get_parameter(probe.PROJECTION))
    assert '_reorder_cache' not in model.__dict__
    assert not model._forward_pre_hooks
    assert not model.get_encoder()._forward_hooks


def test_alignment_masks_n_and_checks_channel_order():
    import torch
    good = torch.zeros((4, 8), dtype=torch.float16)
    good[[0, 2, 3], 7] = 1
    probe.check_tracks('ANCG', good)
    good[1, 0] = 1
    with pytest.raises(ValueError, match='N mask'):
        probe.check_tracks('ANCG', good)
