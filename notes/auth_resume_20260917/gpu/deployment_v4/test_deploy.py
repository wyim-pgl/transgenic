"""Stdlib-only local negative checks; no SIF, CUDA, SSH or DB claims."""
import ast
import io
import build_release
import json
import os
import subprocess
from unittest.mock import patch
import deploy
from pathlib import Path
import tarfile
import tempfile
import unittest
from deploy import parse_quota, pair_candidates, extract_archive, copy_cache, MODEL, REV, sha


class DeploymentTests(unittest.TestCase):
    def test_actual_quota_format_conservative(self):
        text = '| /work/nvme/bilv | 350G | 500G | 550G | 4482 | 2550000 | 2805000 |'
        value = parse_quota(text)
        self.assertEqual(value, 499 * 1000**3 - 351 * 1024**3)
        self.assertGreater(value, 45 * 2**30)
        self.assertLess(value, 150 * 2**30)

    def test_unknown_or_ambiguous_quota_fails(self):
        row = '| /work/nvme/bilv | 350G | 500G | 550G | 4482 | 2550000 | 2805000 |'
        for text in ('', row + '\n' + row, row.replace('350G', '350G*'),
                     row.replace('500G', '0'), row.replace('2550000', '4k')):
            with self.assertRaises(ValueError):
                parse_quota(text)

    def test_pair_identity_not_adjacent_rn_guess(self):
        rows = [{'rn': rn, 'chromosome': 'Chr1', 'start0': 10, 'end0': 30730, 'is_rc': rc}
                for rn, rc in ((99, True), (17, False))]
        pairs = pair_candidates(rows)
        self.assertEqual(pairs['30720'], [[17, 99]])
        self.assertEqual(pairs['61440'], [])
        with self.assertRaises(ValueError):
            pair_candidates(rows + rows[:1])
        rows[0]['end0'] += 1
        self.assertEqual(pair_candidates(rows)['30720'], [])

    def test_archive_rejects_link_and_escape(self):
        for name, link in (('../escape', False), ('evil', True)):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                archive = root / 'test.tar'
                with tarfile.open(archive, 'w') as t:
                    m = tarfile.TarInfo(name)
                    if link:
                        m.type = tarfile.SYMTYPE
                        m.linkname = '/tmp/unsafe'
                    else:
                        m.size = 1
                    t.addfile(m, None if link else io.BytesIO(b'x'))
                with self.assertRaises(ValueError):
                    extract_archive(archive, root / 'out', {})

    def test_cache_materializes_both_layouts_without_source_write(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / 'source'
            for layout, names in (('', ('special_tokens_map.json', 'tokenization_hyena.py', 'tokenizer_config.json')),
                                  ('hub', ('model.safetensors', 'config.json', 'configuration_hyena.py', 'modeling_hyena.py'))):
                model = src / layout / MODEL
                (model / 'refs').mkdir(parents=True)
                (model / 'refs/main').write_text(REV + '\n')
                (model / 'blobs').mkdir()
                snap = model / 'snapshots' / REV
                snap.mkdir(parents=True)
                for name in names:
                    blob = model / 'blobs' / name
                    blob.write_text(name)
                    (snap / name).symlink_to(blob)
            before = {str(p): sha(p) for p in src.rglob('*') if p.is_file()}
            dst = root / 'destination'
            copy_cache(src, dst)
            self.assertEqual(before, {str(p): sha(p) for p in src.rglob('*') if p.is_file()})
            self.assertFalse(any(p.is_symlink() for p in dst.rglob('*')))
            self.assertEqual(len([p for p in dst.rglob('*') if p.is_file()]), 9)
            with self.assertRaises(FileExistsError):
                copy_cache(src, dst)

    def test_host_per_job_mounts_and_collision(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            namespace = root / 'existing'
            release = namespace / 'staging/dev123_p1_release_v6_20260917'
            release.mkdir(parents=True)
            source = root / 'seed123'
            source.mkdir()
            (namespace / 'code').mkdir()  # existing namespace children are preserved
            (namespace / 'code/sentinel').write_text('untouched')
            files = {name: '' for name in ('code.tar', 'deploy.py', 'p1.sbatch',
                                         deploy.ACCEPT, deploy.INVENTORY, deploy.RECEIPTS)}
            for name in files:
                p = release / name
                p.parent.mkdir(exist_ok=True)
                p.write_text('fixture')
            manifest = {'release_files': files, 'source_files': {},
                        'accepted_companions_not_in_source_commit':
                        {n: '' for n in (deploy.ACCEPT, deploy.INVENTORY, deploy.RECEIPTS)}}
            (release / 'release.json').write_text(json.dumps(manifest))
            (release / 'release.sha256').write_text('fixture')
            (release / 'UNLISTED').write_text('must not copy')
            (namespace / 'staging/evidence').mkdir()
            origin = namespace / 'staging/evidence/fullc0lite.db'
            origin.write_text('fixture'); origin.chmod(0o444)
            with patch.multiple(deploy, ROOT=namespace, SOURCE=source, EVIDENCE_SOURCE=origin), \
                 patch.object(deploy.admission, 'UID', os.getuid()), \
                 patch.object(deploy.admission, 'NAMESPACE', str(release)), \
                 patch.object(deploy.admission, 'live'), \
                 patch.dict(os.environ, SLURM_JOB_ID='12345', SLURM_JOB_NUM_NODES='1'), \
                 patch.object(deploy, 'pinned'), patch.object(deploy, 'freeze'), \
                 patch.object(deploy, 'verify_source_manifest'), \
                 patch.object(deploy, 'controller_gate', return_value={}), \
                 patch.object(deploy, 'quota_record') as quota, \
                 patch.object(deploy, 'copy_cache'), \
                 patch.object(deploy, 'extract_archive', side_effect=lambda a, p, f: p.mkdir()), \
                 patch.object(deploy, 'run') as run:
                deploy.host()
                job = namespace / 'dev123-p1-v6-job-12345'
                self.assertFalse((job / 'release/UNLISTED').exists())
                self.assertEqual((namespace / 'code/sentinel').read_text(), 'untouched')
                self.assertTrue((job / 'inputs/evidence/fullc0lite.db').is_file())
                self.assertEqual(quota.call_count, 2)
                stage, probe = [list(map(str, c.args[0])) for c in run.call_args_list]
                for argv in (stage, probe):
                    self.assertIn(str(deploy.SIF), argv)
                    self.assertIn(str(deploy.SIF) + ':/image.sif:ro', argv)
                    self.assertIn(str(deploy.CORPUS) + ':/corpus.db:ro', argv)
                    self.assertIn(str(job / 'code/source') + ':/work/repo:ro', argv)
                    self.assertIn(str(job / 'cache/p1') + ':/scratch:rw', argv)
                    self.assertFalse(any('TRANSGENIC_RUN_DIR' in x for x in argv))
                self.assertIn(str(source) + ':/source/seed123:ro', stage)
                self.assertNotIn(str(source) + ':/source/seed123:ro', probe)
                self.assertIn(str(job / 'inputs') + ':/mvp/inputs:rw', stage)
                self.assertIn(str(job / 'inputs') + ':/mvp/inputs:ro', probe)
                self.assertEqual(origin.read_text(), 'fixture')
                with self.assertRaises(FileExistsError):
                    deploy.host()
                self.assertEqual(run.call_count, 2)

    def test_release_exact_candidate_bytes_and_no_generated_trees(self):
        release = Path(__file__).resolve().parent / 'release_v6'
        manifest = json.loads((release / 'release.json').read_text())
        self.assertEqual(manifest['source_provenance']['base_commit'], deploy.BASE_COMMIT)
        self.assertNotIn('commit', manifest)
        expected = set(manifest['release_files']) | {'release.json', 'release.sha256'}
        self.assertEqual({str(p.relative_to(release)) for p in release.rglob('*') if p.is_file()}, expected)
        for name, digest in manifest['release_files'].items():
            self.assertEqual(sha(release / name), digest)
        for name in ('deploy.py', 'p1.sbatch'):
            self.assertEqual((release / name).read_bytes(), (release.parent / name).read_bytes())
        with tarfile.open(release / 'code.candidate.tar') as tar:
            files = {m.name: m for m in tar.getmembers() if m.isfile()}
            self.assertEqual(set(files), set(manifest['source_files']))
            self.assertTrue(all(m.isdir() or m.isfile() for m in tar.getmembers()))
            for name, member in files.items():
                self.assertFalse(any(p in {'__pycache__', '.venv', 'site-packages', '.testdeps'} for p in Path(name).parts))
                payload = tar.extractfile(member).read()
                candidate = release.parent.parent / 'candidate/code.candidate.tar'
                with tarfile.open(candidate) as original:
                    self.assertEqual(payload, original.extractfile(name).read(), name)
                self.assertEqual(deploy.hashlib.sha256(payload).hexdigest(), manifest['source_files'][name])

    def test_final_candidate_closure_and_operator_pins(self):
        here = Path(__file__).resolve().parent
        manifest, payload = build_release.candidate_payload(here.parent / 'candidate')
        pins = build_release.operator_pins(manifest['source_files'], payload)
        self.assertEqual(pins, json.loads((here / 'release_v6/operator-pins.json').read_text()))
        self.assertEqual(pins['source_comparison_count'], 89)
        self.assertEqual(pins['decoder_file_count'], 14)
        self.assertNotEqual(pins['source_comparison_sha256'],
                            '555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44')
        self.assertNotEqual(pins['decoder_code_sha256'],
                            'b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb')
        for name in ('src/transgenic/training/mvp_profiles.py',
                     'revision/scripts/mvp/mvp_verify_pinned_panel.py',
                     'revision/scripts/mvp/mvp_panel_compatibility.py',
                     'revision/scripts/07_selfconsistency_stats.py',
                     'revision/scripts/24_restrict_reference_to_test_genes.py'):
            self.assertIn(name, payload)

    def test_old_p1_unchanged(self):
        here = Path(__file__).resolve().parent
        before = json.loads((here / 'originals-preservation.json').read_text())
        old = here.parents[2] / 'mvp_execution_20260916/final_p1'
        self.assertEqual(before, {str(p.relative_to(old)): sha(p) for p in old.rglob('*') if p.is_file()})

    def test_quota_reserve_and_warning_fail_closed(self):
        row = '| /work/nvme/bilv | 350G | 500G | 550G | 4482 | 2550000 | 2805000 |'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, (stderr, free, ok) in enumerate((('', 45 * 2**30, True),
                    ('', 45 * 2**30 - 1, False), ('warning', 100 * 2**30, False))):
                result = subprocess.CompletedProcess(['quota', '-s'], 0, row, stderr)
                with patch.object(deploy.subprocess, 'run', return_value=result), \
                     patch.object(deploy.shutil, 'disk_usage', return_value=type('Usage', (), {'free': free})()):
                    dest = root / ('quota%d.json' % i)
                    if ok:
                        deploy.quota_record(dest, root)
                        record = json.loads(dest.read_text())
                        self.assertEqual(record['reserve_bytes'], 40 * 2**30)
                        self.assertLess(abs(deploy.time.time() - record['measured_at_unix']), 5)
                    else:
                        with self.assertRaises(ValueError):
                            deploy.quota_record(dest, root)
                        self.assertFalse(dest.exists())
                    self.assertTrue(dest.with_suffix('.raw.json').exists())

    def test_cache_rejects_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / 'source'
            model = src / MODEL
            (model / 'refs').mkdir(parents=True)
            (model / 'refs/main').write_text(REV)
            snap = model / 'snapshots' / REV
            snap.mkdir(parents=True)
            outside = root / 'outside'
            outside.write_text('unapproved')
            for name in ('special_tokens_map.json', 'tokenization_hyena.py', 'tokenizer_config.json'):
                (snap / name).symlink_to(outside)
            with self.assertRaises(ValueError):
                copy_cache(src, root / 'destination')


if __name__ == '__main__':
    unittest.main()
