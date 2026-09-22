"""One integrated source, CPU-only closure/semantic regressions, not readiness."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_mvp_score_producer import ROOT, finalized_fixture, p


def test_reviewed_a_implementation_bytes_unchanged():
    # Re-pinned 2026-09-22 to 426c328 (A47.2 coverage contract v2); before that eff505f
    # (2026-09-21, A47/A47.1), whose predecessor 772c38c had already drifted.
    # The pin records the tree, not an independent review.
    # Includes paired metric validator/math, v3 dataset runtime, finalizer, model,
    # config/tokenizer, B5 initialization, probes/resource/runner and recipes.
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '426c328',
        'src/transgenic', 'train/train_HyenaTransgenic_MVP.py', 'configs',
        'revision/scripts/mvp'], cwd=ROOT, text=True).splitlines()
    for relative in paths:
        expected = subprocess.check_output(['git', 'show', '426c328:' + relative], cwd=ROOT)
        assert (ROOT / relative).read_bytes() == expected, relative


def test_fresh_manifest_includes_decoder_and_training_closure(tmp_path):
    from make_launch_readiness import _consumed_code_manifest
    from transgenic.training.mvp_cli_runtime import _expected_code_files, _manifest_files
    source = _consumed_code_manifest()
    added = {'src/transgenic/evidence/' + name for name in
             ('mvp_bootstrap.py', 'mvp_decode.py', 'mvp_windows.py')}
    assert {str(ROOT / name) for name in added} <= set(source['files'])
    assert set(source['files']) == {str(path) for path in _expected_code_files()}
    manifest = tmp_path / 'fresh-code.json'
    manifest.write_text(json.dumps(source))
    _manifest_files(manifest)
    # Identity is content-based; future staging mtimes must not change it.
    os.utime(manifest, (2000000000, 2000000000))
    _manifest_files(manifest)
    del source['files'][str(ROOT / 'src/transgenic/evidence/mvp_decode.py')]
    manifest.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='exact executed source set'):
        _manifest_files(manifest)


def test_finalizer_and_scorer_share_integrated_training_closure(tmp_path):
    final, hashes, code = finalized_fixture(tmp_path)
    verified = p.verify_final(final, 1, [1], hashes, code)
    assert str(ROOT / 'src/transgenic/evidence/mvp_decode.py') in {
        record['path'] for record in verified['code_files']}
    os.utime(code, (2000000000, 2000000000))
    assert p.verify_final(final, 1, [1], hashes, code) == verified
    source = json.loads(code.read_text())
    metadata = json.loads((final / 'metadata.json').read_text())
    del source['files'][str(ROOT / 'src/transgenic/evidence/mvp_bootstrap.py')]
    with pytest.raises(ValueError, match='exact bound training source closure'):
        p.verify_training_code(source, metadata)
