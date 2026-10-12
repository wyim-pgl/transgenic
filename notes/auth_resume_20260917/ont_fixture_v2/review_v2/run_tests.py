"""Review-only runner: redirect temporary writes, retain original test assertions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(HERE)}

before = snapshot(ROOT)
original_before = snapshot(ROOT.parent/'ont_fixture')
original_temp = tempfile.TemporaryDirectory

def temporary(*args, **kwargs):
    kwargs['dir'] = HERE
    return original_temp(*args, **kwargs)

tempfile.TemporaryDirectory = temporary
suite = unittest.defaultTestLoader.loadTestsFromNames(['test_amendments', 'test_fixture'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
after = snapshot(ROOT)
original_after = snapshot(ROOT.parent/'ont_fixture')
record = dict(python=sys.version, before=before, after=after, unchanged=before == after,
              original_unchanged=original_before == original_after,
              original_matches_saved=original_after == json.loads((ROOT/'original_identities.json').read_text()))
(HERE/'identities.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
assert record['unchanged'] and record['original_unchanged'] and record['original_matches_saved']
sys.exit(not result.wasSuccessful())
