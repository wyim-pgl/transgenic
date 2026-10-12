"""Retained Slurm format regression. All live observations here are local fixtures."""
import hashlib
from pathlib import Path
import re
import unittest
import admission as a
import test_admission_cases as cases

RAW = Path(__file__).with_name('scontrol-job-3163986.txt').read_bytes()


def current_raw():
    # The retained line is PENDING historical evidence, not a new live allocation.
    # Change only identity/state/placement/allocation fields required by the
    # existing synthetic RUNNING fixture. Keep its raw CPUs/Task spelling.
    _, current, jobs, _ = cases.fixture()
    line = RAW.decode().strip()
    for key in ('JobId', 'JobName', 'JobState', 'NodeList', 'AllocTRES', 'Command', 'WorkDir'):
        line, count = re.subn(r'(?<!\S)' + key + r'=\S*', key+'='+jobs[current][key], line)
        assert count == 1
    return line+'\n'


def observations(argv, output, commands):
    if argv[:4] == ['/usr/bin/scontrol', '-o', 'show', 'job'] and argv[-1] == '4000000':
        output = current_raw()
    return output, '', 0


class FieldAliasTests(unittest.TestCase):
    def test_exact_retained_line_normalizes(self):
        self.assertEqual(hashlib.sha256(RAW).hexdigest(),
                         '95d819c95d7665cb830f177c75f62db80585245d77dda26ac48159c80ad26679')
        parsed = a.fields(RAW.decode().strip())
        self.assertEqual(parsed.get('CPUsPerTask'), '64')
        self.assertNotIn('CPUs/Task', parsed)
        self.assertEqual(parsed['JobId'], '3163986')

    def test_aliases_are_ambiguous_even_when_equal(self):
        for first, second in [('CPUs/Task', 'CPUsPerTask'), ('CPUsPerTask', 'CPUs/Task')]:
            for value in ('64', '32'):
                with self.subTest(first=first, value=value), self.assertRaisesRegex(ValueError, 'duplicate scheduler field'):
                    a.fields(first+'=64 '+second+'='+value)

    def test_raw_format_full_admission(self):
        cases.LivePathTests().run_live(observations)

    def test_changed_normalized_value_rejected_in_final_snapshot(self):
        def changed(argv, output, commands):
            output, stderr, code = observations(argv, output, commands)
            if argv[-1] == '4000000' and 'job' in argv:
                if sum(c == argv for c in commands) == 2:
                    output = output.replace('CPUs/Task=64', 'CPUs/Task=32')
            return output, stderr, code
        with self.assertRaisesRegex(ValueError, 'allocation changed during admission'):
            cases.LivePathTests().run_live(changed)

    def test_only_observed_alias_and_no_default(self):
        self.assertEqual(a.fields('CPUsPerTask=32'), {'CPUsPerTask': '32'})
        self.assertEqual(a.fields('CPUs/Task=32'), {'CPUsPerTask': '32'})
        self.assertEqual(a.fields('CPUs_Task=64'), {'CPUs_Task': '64'})
        self.assertNotIn('CPUsPerTask', a.fields('JobId=1'))


if __name__ == '__main__':
    unittest.main()
