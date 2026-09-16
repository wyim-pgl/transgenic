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


def test_alignment_masks_n_and_checks_channel_order():
    import torch
    good = torch.zeros((4, 8), dtype=torch.float16)
    good[[0, 2, 3], 7] = 1
    probe.check_tracks('ANCG', good)
    good[1, 0] = 1
    with pytest.raises(ValueError, match='N mask'):
        probe.check_tracks('ANCG', good)
