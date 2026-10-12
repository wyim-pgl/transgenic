"""Local-only release/argv checks: timeout is intercepted; no container executes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
OLD = HERE / 'final_release'
NEW = Path(os.environ.get('BIND_FIX_RELEASE', str(HERE / 'final_release_bindfix1')))
OLD_TRANSPORT = 'b4bc567d6bdae382318fd0061186bf83c5077b12ac024b5cc36a6dcf9bb54eef'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def argv(script):
    with tempfile.TemporaryDirectory(prefix='bindfix1-') as td:
        root = Path(td)
        release = root / 'release with spaces'
        (release / 'outputs').mkdir(parents=True)
        stub = root / 'timeout'
        stub.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        stub.chmod(0o755)
        env = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'],
                   PANEL_RELEASE=str(release), PANEL_CPU_SITE=str(release / 'cpu_site'),
                   SLURM_JOB_ID='999999', SLURM_CPUS_PER_TASK='4')
        result = subprocess.run(['bash', str(script)], env=env, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert not (release / 'outputs/job-999999').exists()
        return [s.replace(str(release), 'RELEASE') for s in json.loads(result.stdout)]


class BindFixTests(unittest.TestCase):
    def test_payloads_and_transport(self):
        self.assertEqual(sha(OLD / 'transport.json'), OLD_TRANSPORT)
        old = json.loads((OLD / 'transport.json').read_text())
        new = json.loads((NEW / 'transport.json').read_text())
        self.assertEqual({p.name for p in NEW.iterdir()}, set(old['files']) | {'transport.json'})
        for name, digest in old['files'].items():
            self.assertEqual(sha(OLD / name), digest)
            self.assertEqual(sha(NEW / name), new['files'][name])
            if name != 'panel_cpu.sbatch':
                self.assertEqual(new['files'][name], digest)
        self.assertNotEqual(new['files']['panel_cpu.sbatch'], old['files']['panel_cpu.sbatch'])
        new['files']['panel_cpu.sbatch'] = old['files']['panel_cpu.sbatch']
        self.assertEqual(new, old)

    def test_exact_one_option_change_and_syntax(self):
        old = (OLD / 'panel_cpu.sbatch').read_bytes()
        self.assertEqual(old.count(b'"$PARENT:$PARENT:rw"'), 1)
        self.assertEqual((NEW / 'panel_cpu.sbatch').read_bytes(),
                         old.replace(b'"$PARENT:$PARENT:rw"', b'"$PARENT:$PARENT"'))
        subprocess.run(['bash', '-n', str(NEW / 'panel_cpu.sbatch')], check=True)

    def test_actual_shell_argv_default_rw_only_for_output(self):
        old = argv(OLD / 'panel_cpu.sbatch')
        new = argv(NEW / 'panel_cpu.sbatch')
        expected = [s.replace('RELEASE/outputs:RELEASE/outputs:rw',
                              'RELEASE/outputs:RELEASE/outputs') for s in old]
        self.assertEqual(new, expected)
        binds = [new[i + 1] for i, s in enumerate(new) if s == '-B']
        # Python's -B is also present; only the two arguments before the SIF are binds.
        self.assertEqual(binds[:2], ['/data/gpfs:/data/gpfs:ro', 'RELEASE/outputs:RELEASE/outputs'])
        self.assertEqual(new[:4], ['--signal=TERM', '--kill-after=20s', '3500s',
                                  '/apps/singularity/3.6.1/bin/singularity'])
        self.assertEqual(new[4:6], ['exec', '--cleanenv'])
        self.assertFalse(any(':rw' in s for s in new))


if __name__ == '__main__':
    unittest.main(verbosity=2)
