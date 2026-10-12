"""Local metadata/synthetic-file tests. Never opens a historical DB."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('transport', HERE / 'transport.py')
t = importlib.util.module_from_spec(SPEC) if SPEC else None
if SPEC and SPEC.loader and (HERE / 'transport.py').exists():
    SPEC.loader.exec_module(t)


class TransportTests(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertTrue(hasattr(t, 'build'), 'metadata builder not implemented')

    def test_inventory_and_pins(self):
        m = t.build()
        self.assertEqual(len(m['historical_files']), 167)
        self.assertEqual(sum(x['size'] for x in m['historical_files'] if x['action'] == 'replica_after_review'), 212152146)
        self.assertEqual(sum(x['action'] == 'reuse_existing_no_copy' for x in m['historical_files']), 2)
        self.assertEqual(len(m['consumer_files']), 103)
        self.assertEqual(len(m['authority']['objects']), 5)
        self.assertEqual(m['request']['size'], 1264)
        self.assertEqual(m['runtime']['python'], '3.12.3')
        self.assertIsNone(m['p1_stage_receipt'])
        t.validate_manifest(m)

    def test_manifest_tampering(self):
        m = t.build()
        for mutate in [lambda x: x['historical_files'].pop(),
                       lambda x: x['historical_files'].append(x['historical_files'][0]),
                       lambda x: x['historical_files'][0].update(sha256='0'*64),
                       lambda x: x['request'].update(size=1265),
                       lambda x: x['runtime'].update(numpy='1.26.4'),
                       lambda x: x['consumer_files'][0].update(sha256='0'*64),
                       lambda x: x.update(p1_stage_receipt='/guessed/stage.json')]:
            altered = copy.deepcopy(m)
            mutate(altered)
            with self.assertRaises(ValueError):
                t.validate_manifest(altered)

    def test_paths(self):
        for name in ['../x', '/x', 'a//x', 'a/./x', 'a/../x', 'x:ro', 'x,y', 'x\\y', 'x\ny', 'x.wal', 'x-wal', '.', '']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                t.relative(name)
        with self.assertRaises(ValueError):
            t.unique_paths(['a', 'a/b'])
        with self.assertRaises(ValueError):
            t.unique_paths(['a', 'a'])

    def test_tree_tamper_extra_symlink_wal(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            p = root / 'a.py'
            p.write_bytes(b'pass\n')
            rows = [{'path': 'a.py', 'size': 5, 'sha256': hashlib.sha256(b'pass\n').hexdigest()}]
            t.validate_tree(root, rows)
            p.write_bytes(b'evil\n')
            with self.assertRaises(ValueError): t.validate_tree(root, rows)
            p.write_bytes(b'pass\n')
            for extra in ['extra.py', 'a.py.wal', 'a.py-wal', '__pycache__/a.pyc']:
                f = root / extra
                f.parent.mkdir(exist_ok=True)
                f.write_bytes(b'')
                with self.assertRaises(ValueError): t.validate_tree(root, rows)
                f.unlink()
                if f.parent != root:
                    f.parent.rmdir()
            t.validate_tree(root, rows)
            link = root / 'link'
            link.symlink_to(p)
            with self.assertRaises(ValueError): t.validate_tree(root, rows)
            link.unlink()
            p.unlink()
            p.symlink_to(HERE / 'test_transport.py')
            with self.assertRaises(ValueError): t.validate_tree(root, rows)

    def test_mounts(self):
        mounts = [{'source': '/host/tree', 'target': t.G, 'mode': 'ro'},
                  {'source': '/host/source', 'target': '/work/repo', 'mode': 'ro'},
                  {'source': '/host/output', 'target': '/panel-output', 'mode': 'rw'}]
        t.validate_mounts(mounts)
        for modify in [lambda x: x[0].update(mode='rw'),
                       lambda x: x[1].update(target=t.G + '/source'),
                       lambda x: x[2].update(source='/host/tree/output'),
                       lambda x: x[2].update(target='/work/repo/output'),
                       lambda x: x[0].update(source='/host/../tree')]:
            altered = copy.deepcopy(mounts)
            modify(altered)
            with self.assertRaises(ValueError): t.validate_mounts(altered)

    def test_launcher_is_preparation_only(self):
        proc = subprocess.run([sys.executable, '-B', str(HERE / 'candidate_launcher.py'), '--describe'], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        recipe = json.loads(proc.stdout)
        self.assertEqual(recipe['wrapper_argv'][4:8], ['--expected-sha256', t.REQUEST_SHA, '--expected-size', '1264'])
        self.assertIsNone(recipe['container_argv'])
        self.assertNotIn('--nv', recipe['container_options'])
        self.assertEqual(recipe['environment']['CUDA_VISIBLE_DEVICES'], '')
        proc = subprocess.run([sys.executable, '-B', str(HERE / 'candidate_launcher.py'), '--execute'], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('P1', proc.stderr)
        proc = subprocess.run(['bash', str(HERE / 'candidate.sbatch')], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 78)
        self.assertIn('BLOCKED', proc.stderr)

    def test_namespace_reuse_and_missing_receipt(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            fresh = Path(tmp) / 'fresh'
            t.require_fresh(fresh)
            fresh.mkdir()
            with self.assertRaises(ValueError): t.require_fresh(fresh)
            fresh.rmdir()
            fresh.symlink_to(Path(tmp) / 'missing')
            with self.assertRaises(ValueError): t.require_fresh(fresh)
        with self.assertRaisesRegex(ValueError, 'P1'):
            t.require_execution_ready(t.build())


if __name__ == '__main__':
    unittest.main()
