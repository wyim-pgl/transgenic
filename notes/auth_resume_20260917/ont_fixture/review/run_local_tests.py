"""Review-only runner: unchanged six tests; relocate temporary writes to review/."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
def snapshot():
    return {str(p.relative_to(CANDIDATE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CANDIDATE.rglob('*')) if p.is_file() and HERE not in p.parents}
before = snapshot()
original = tempfile.TemporaryDirectory
def under_review(*args, **kwargs):
    kwargs['dir'] = HERE
    return original(*args, **kwargs)
tempfile.TemporaryDirectory = under_review
spec = importlib.util.spec_from_file_location('reviewed_tests', CANDIDATE/'test_fixture.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
assert suite.countTestCases() == 6
result = unittest.TextTestRunner(verbosity=2).run(suite)
after = snapshot()
assert before == after, 'candidate changed during review tests'
(HERE/'identities.json').write_text(json.dumps({'candidate_files_sha256': before, 'unchanged_after_tests': True, 'python': sys.version, 'executable': sys.executable, 'tests_run': result.testsRun, 'success': result.wasSuccessful()}, indent=2)+'\n')
sys.exit(not result.wasSuccessful())
