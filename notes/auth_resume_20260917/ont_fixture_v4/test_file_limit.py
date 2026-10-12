"""Actual child-process limit tests; no scheduler or alignment is simulated."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent


class FileLimit(unittest.TestCase):
    def child(self, initial):
        code = ('import json,resource,sys;sys.path.insert(0,' + repr(str(HERE)) + ');'
                'import fixture;resource.setrlimit(resource.RLIMIT_FSIZE,' + initial + ');'
                'print(json.dumps(fixture.enforce_startup_file_limit()))')
        return subprocess.run([sys.executable, '-I', '-B', '-c', code],
                              capture_output=True, text=True, timeout=10)

    def test_unlimited_inheritance_is_lowered_and_recorded(self):
        result = self.child('(resource.RLIM_INFINITY,resource.RLIM_INFINITY)')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['before'], [-1, -1])
        self.assertEqual(report['after'], [4194304, 4194304])

    def test_different_soft_limit_is_set_inside_task(self):
        result = self.child('(1048576,resource.RLIM_INFINITY)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['after'], [4194304, 4194304])

    def test_insufficient_hard_limit_refuses_without_claiming_success(self):
        result = self.child('(1048576,1048576)')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('file_limit_before', result.stderr)


if __name__ == '__main__':
    unittest.main()
