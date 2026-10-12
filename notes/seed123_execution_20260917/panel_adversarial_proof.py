"""Reviewer-only synthetic tests; no accepted data or source edits."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, '/tmp/mvp-seed123-integration-20260917/tests')
from test_mvp_panel_compatibility import example, publish, consume, p

@pytest.mark.parametrize('change', ['candidate', 'acceptance_pin', 'panel_root', 'external_pin'])
def test_cross_binding(example, change):
    e = example
    receipt, _ = publish(e)
    if change == 'candidate':
        e.payload['candidate']['unexpected_binding'] = 'other request'
    elif change == 'acceptance_pin':
        e.payload['candidate']['acceptance_sha256'] = '0'*64
    elif change == 'panel_root':
        other = e.tmp/'other-panel'; other.mkdir()
        e.root = other
    else:
        e.payload['panel_compatibility']['sha256'] = '0'*64
    with pytest.raises(ValueError):
        consume(e)

@pytest.mark.parametrize('change', ['request', 'pending', 'destination'])
def test_mutation_after_serialization(example, monkeypatch, change):
    e = example
    original = e.c.producer.stage_json
    def stage(path, report):
        pending = original(path, report)
        if change == 'request':
            with (e.tmp/'current-request.json').open('a') as f: f.write(' ')
        elif change == 'pending':
            pending.write_text('{}')
        else:
            Path(path).write_text('occupied')
        return pending
    monkeypatch.setattr(e.c.producer, 'stage_json', stage)
    with pytest.raises((ValueError, FileExistsError)):
        publish(e)
    target = e.tmp/'compat.json'
    assert not target.exists() if change != 'destination' else target.read_text() == 'occupied'
    assert (e.tmp/'compat.json.pending').exists()
