"""Synthetic caller-boundary tests, not actual50 or real-data certification."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def case(tmp_path):
    request = tmp_path / 'request.json'
    intended = b'{"synthetic": true}\n'
    request.write_bytes(intended)
    # Pin the intended fixture bytes, never the mutable request on disk.
    pin = dict(path=str(request), sha256=hashlib.sha256(intended).hexdigest(), size=len(intended))
    return request, pin, tmp_path / 'receipt.json'


def invoke(case):
    request, pin, out = case
    caller = importlib.import_module('mvp_verify_pinned_panel')
    return caller.verify_pinned_panel(request, pin['sha256'], pin['size'], out)


def install_helper(monkeypatch, case, change=None):
    import mvp_panel_compatibility as compatibility
    request, pin, out = case
    calls = []

    def helper(input_path, output_path):
        calls.append((input_path, output_path))
        assert Path(input_path) == request and Path(output_path) == out
        assert len(calls) == 1, 'caller must not retry'
        report = {'verification_request': dict(pin), 'synthetic_only': True}
        serialized = json.loads(json.dumps(report))
        if change:
            change(report, serialized)
        out.write_text(json.dumps(serialized))
        return report

    monkeypatch.setattr(compatibility, 'verify_and_publish', helper)
    return calls


@pytest.mark.parametrize('change', ['replace', 'mutate', 'size', 'digest'])
def test_wrong_entry_pin_stops_before_helper_or_output(case, monkeypatch, change):
    request, pin, out = case
    calls = install_helper(monkeypatch, case)
    if change == 'replace':
        other = request.with_suffix('.replacement')
        other.write_bytes(b'{"synthetic":false}\n')
        other.replace(request)
    elif change == 'mutate':
        with request.open('ab') as handle:
            handle.write(b' ')
    elif change == 'size':
        pin['size'] += 1
    else:
        pin['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        invoke(case)
    assert calls == []
    assert not out.exists() and not Path(str(out) + '.pending').exists()


@pytest.mark.parametrize('change', ['replace', 'mutate', 'symlink'])
def test_late_request_change_rejects_and_preserves_output(case, monkeypatch, change):
    request, pin, out = case

    def alter(report, serialized):
        if change == 'mutate':
            request.write_bytes(b'changed')
        else:
            replacement = request.with_suffix('.replacement')
            replacement.write_bytes(b'changed')
            if change == 'replace':
                replacement.replace(request)
            else:
                request.unlink()
                request.symlink_to(replacement)

    calls = install_helper(monkeypatch, case, alter)
    with pytest.raises(ValueError):
        invoke(case)
    assert len(calls) == 1
    assert json.loads(out.read_text())['verification_request'] == pin


@pytest.mark.parametrize('field,value', [('path', '/another/request.json'), ('sha256', '0' * 64),
                                        ('size', 999), ('extra', True)])
def test_wrong_serialized_request_binding_rejects_even_matching_return(case, monkeypatch, field, value):
    def alter(report, serialized):
        report['verification_request'][field] = value
        serialized['verification_request'][field] = value

    calls = install_helper(monkeypatch, case, alter)
    with pytest.raises(ValueError, match='verification_request'):
        invoke(case)
    assert len(calls) == 1
    assert json.loads(case[2].read_text())['verification_request'][field] == value


def test_returned_report_must_match_serialized_output(case, monkeypatch):
    def alter(report, serialized):
        serialized['synthetic_only'] = False

    install_helper(monkeypatch, case, alter)
    with pytest.raises(ValueError, match='serialized'):
        invoke(case)
    assert json.loads(case[2].read_text())['synthetic_only'] is False


def test_valid_pin_calls_once_and_returns_serialized_report(case, monkeypatch):
    calls = install_helper(monkeypatch, case)
    result = invoke(case)
    assert len(calls) == 1
    assert result == json.loads(case[2].read_text())
    assert result['verification_request'] == case[1]


@pytest.mark.parametrize('path_kind', ['relative', 'symlink', 'parent'])
def test_noncanonical_request_fails_before_helper(case, monkeypatch, path_kind):
    request, pin, out = case
    calls = install_helper(monkeypatch, case)
    if path_kind == 'relative':
        monkeypatch.chdir(request.parent)
        request = Path(request.name)
    elif path_kind == 'parent':
        request = request.parent / '..' / request.parent.name / request.name
    else:
        link = request.with_suffix('.link')
        link.symlink_to(request)
        request = link
    with pytest.raises(ValueError):
        invoke((request, pin, out))
    assert calls == [] and not out.exists()


@pytest.mark.parametrize('digest,size', [('bad', 20), ('0' * 64, -1), ('0' * 64, True)])
def test_malformed_expected_pin_rejected(case, monkeypatch, digest, size):
    calls = install_helper(monkeypatch, case)
    case[1].update(sha256=digest, size=size)
    with pytest.raises(ValueError):
        invoke(case)
    assert calls == [] and not case[2].exists()


def test_helper_failure_preserves_pending_and_does_not_retry(case, monkeypatch):
    import mvp_panel_compatibility as compatibility
    calls = []
    pending = Path(str(case[2]) + '.pending')

    def fail(request, out):
        calls.append(request)
        pending.write_text('failure diagnostics')
        raise RuntimeError('helper failed')

    monkeypatch.setattr(compatibility, 'verify_and_publish', fail)
    with pytest.raises(RuntimeError, match='helper failed'):
        invoke(case)
    assert len(calls) == 1 and pending.read_text() == 'failure diagnostics'


def test_cli_forwards_independent_pin(case, monkeypatch, capsys):
    calls = install_helper(monkeypatch, case)
    caller = importlib.import_module('mvp_verify_pinned_panel')
    request, pin, out = case
    assert caller.main(['--input', str(request), '--expected-sha256', pin['sha256'],
                        '--expected-size', str(pin['size']), '--out', str(out)]) == 0
    assert len(calls) == 1 and out.exists()
    assert 'accepted' not in capsys.readouterr().out.lower()


def test_real_helper_allocation_guard_remains_active(case, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='CPU compute allocation'):
        invoke(case)
    assert not case[2].exists()
