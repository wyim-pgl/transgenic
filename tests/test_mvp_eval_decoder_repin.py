"""Eval9603 -> decoder consumption contract: the recorded decoder package must
cover the reviewed DECODER_ONLY source closure.

CPU only. These are byte/closure contract checks, not GPU receipts or any
scientific acceptance.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'revision/scripts/mvp'))
import mvp_produce_counts as p
import mvp_decode_panel as decoder
import mvp_decode_runtime as runtime
from transgenic.evidence import mvp_windows as w


def _full_package():
    """Decoder package exactly as the reviewed decoder records it."""
    files = {str(path): w.sha256_file(path) for path in decoder.decoder_code_paths()}
    return {'files': files, 'commands': [['synthetic-test-only']]}


def test_decoder_code_inventory_covers_reviewed_decoder_closure():
    recorded = {str(Path(path).resolve().relative_to(ROOT)) for path in decoder.decoder_code_paths()}
    missing = sorted(set(runtime.DECODER_ONLY) - recorded)
    assert missing == []


def test_producer_accepts_decoder_package_covering_reviewed_closure():
    package = _full_package()
    records = p.verify_decoder_package(package)
    assert {r['path'] for r in records} == set(package['files'])


def test_producer_rejects_decoder_package_missing_reviewed_closure_file():
    package = _full_package()
    dropped = str((ROOT / runtime.DECODER_ONLY[-1]).resolve())
    package['files'].pop(dropped)
    with pytest.raises(ValueError, match='reviewed decoder closure'):
        p.verify_decoder_package(package)


def test_producer_rejects_stale_decoder_package_recording_only_its_own_caller():
    package = {'files': {str(Path(__file__).resolve()): w.sha256_file(__file__)},
               'commands': [['synthetic-test-only']]}
    with pytest.raises(ValueError, match='reviewed decoder closure'):
        p.verify_decoder_package(package)


def test_producer_still_rejects_wrong_recorded_hash_inside_closure():
    package = _full_package()
    target = next(iter(package['files']))
    package['files'][target] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        p.verify_decoder_package(package)


def test_producer_still_rejects_empty_or_malformed_decoder_package():
    with pytest.raises(ValueError, match='decoder code/commands missing'):
        p.verify_decoder_package({'files': {}, 'commands': [['x']]})
    package = _full_package()
    package['commands'] = [[]]
    with pytest.raises(ValueError, match='decoder code/commands missing'):
        p.verify_decoder_package(package)
