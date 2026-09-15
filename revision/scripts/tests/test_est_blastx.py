"""Launcher contracts, not DIAMOND engine tests."""
import gzip
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'evidence/est_blastx.py'


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('est_blastx', SCRIPT)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'reviewed EST launcher not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def version_engine(self, version, banner, fail_command=None, stderr=False):
        # CLI double only: never counts as real-DIAMOND alignment evidence.
        engine = self.put('diamond', f'''#!{sys.executable}
import sys
outputs = {{'version': {version!r}, 'help': {banner!r}}}
command = sys.argv[1]
print(outputs[command], file=sys.stderr if {stderr!r} else sys.stdout)
raise SystemExit(7 if command == {fail_command!r} else 0)
''')
        engine.chmod(0o755)
        return engine

    def test_engine_version_accepts_release_with_exact_help_build(self):
        banner = 'diamond v2.1.24.178 (C) Max Planck Society for the Advancement of Science'
        for token in ('2.1.24', '2.1.24.178'):
            for stderr in (False, True):
                with self.subTest(token=token, stderr=stderr):
                    version = f'diamond version {token}'
                    engine = self.version_engine(version, banner + '\nUsage: diamond COMMAND\n', stderr=stderr)
                    self.assertEqual(self.m.engine_version(engine), version + '\n' + banner)

    def test_engine_version_rejects_incompatible_conflicting_or_malformed_identity(self):
        good = 'diamond v2.1.24.178 (C) Max Planck Society'
        for version, banner in (
            ('diamond version 2.1.23', good),
            ('diamond version 2.1.24.177', good),
            ('diamond version 2.1.24.178', good.replace('.178', '.179')),
            ('diamond version 2.1.24', good.replace('.178', '.1780')),
            ('diamond version 2.1.24', good.replace('.178', '.178-dev')),
            ('diamond version 2.1.24', good.replace('.178', '')),
            ('diamond version 2.1.24.178', ''),
            ('diamond version 2.1.24.178', 'Usage: diamond\n' + good),
            ('diamond version 2.1.24.178 extra', good),
            ('diamond version 2.1.24\ndiamond version 2.1.23', good),
        ):
            with self.subTest(version=version, banner=banner):
                with self.assertRaisesRegex(ValueError, 'unreviewed DIAMOND'):
                    self.m.engine_version(self.version_engine(version, banner))

    def test_engine_version_command_failures_and_invalid_paths_fail_closed(self):
        for command in ('version', 'help'):
            engine = self.version_engine('diamond version 2.1.24.178',
                                         'diamond v2.1.24.178 (C) Max Planck Society', command)
            with self.subTest(command=command):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.m.engine_version(engine)
        for path in ('diamond', self.root / 'missing', self.root):
            with self.assertRaises(ValueError):
                self.m.engine_version(path)

    def test_source_specific_taxids_and_full_ids(self):
        self.assertEqual(self.m.subject_taxid('sp|Q1.2|NAME title OX=3702 GN=x', 'swissprot'), 3702)
        self.assertEqual(self.m.subject_taxid('39947_0:full.id\t39947_0', 'orthodb'), 39947)
        for header in ('sp|X|X', 'sp|X|X OX=bad', 'sp|X|X OX=3702 OX=4530'):
            with self.assertRaises(ValueError):
                self.m.subject_taxid(header, 'swissprot')

    def test_filter_root_descendant_merged_preserves_headers(self):
        source = self.put('source.fa', '>3702_0:same\t3702_0\nMAAA\n>99_0:child\nMCCC\n>88_0:merged\nMDDD\n>4530_0:full.id description\nMEFG\n')
        lineage = self.put('lineage.tsv', 'taxid\tlineage\n3702\t3702;1\n99\t99;3702;1\n88\t88;99;3702;1\n4530\t4530;1\n')
        out = self.root / 'filtered.fa'
        qc = self.m.filter_source(source, 'orthodb', lineage, 3702, out)
        self.assertEqual(out.read_text(), '>4530_0:full.id description\nMEFG\n')
        self.assertEqual(qc['excluded_taxids'], [88, 99, 3702])
        self.assertEqual(qc['sequences_out'], 1)
        self.assertEqual(qc['sequences_in'], 4)
        with self.assertRaises(FileExistsError):
            self.m.filter_source(source, 'orthodb', lineage, 3702, out)
        self.assertEqual(out.read_text(), '>4530_0:full.id description\nMEFG\n')

    def test_missing_or_incomplete_lineage_fails_closed(self):
        source = self.put('source.fa', '>99_0:x\nMAAA\n')
        for text in ('taxid\tlineage\n', '99\t3702;1\n', '99\t99\n'):
            lin = self.put('lin.tsv', text)
            with self.assertRaises((ValueError, SystemExit)):
                self.m.filter_source(source, 'orthodb', lin, 3702, self.root / 'out.fa')
            self.assertFalse((self.root / 'out.fa').exists())

    def test_ncbi_cellular_root_lineage_is_accepted_without_taxid_one(self):
        source = self.put('source.fa', '>4530_0:keep\nMEFG\n')
        lin = self.put('lineage.tsv', '4530\t4530;4527;33090;2759;131567\n')
        qc = self.m.filter_source(source, 'orthodb', lin, 3702, self.root / 'out.fa')
        self.assertEqual(qc['sequences_out'], 1)

    def test_separate_output_and_explicit_search_tokens(self):
        for source in ('orthodb', 'swissprot'):
            cmd = self.m.search_command('/diamond', self.root / 'q.fa', self.root / f'{source}.dmnd', self.root / f'{source}.raw13.tsv', 4, self.root / 'tmp')
            self.assertEqual(cmd[:2], ['/diamond', 'blastx'])
            for flag, value in (('--frameshift', '15'), ('--evalue', '1e-5'), ('--min-orf', '30'), ('--max-target-seqs', '25'), ('--max-hsps', '1')):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertIn('--no-parse-seqids', cmd)
            self.assertNotIn('--seqidlist', cmd)
            self.assertEqual(cmd[cmd.index('--outfmt') + 1:], ['6', 'qseqid', 'qlen', 'sseqid', 'slen', 'qstart', 'qend', 'sstart', 'send', 'length', 'evalue', 'bitscore', 'qframe', 'btop'])
            self.assertIn(str(self.root / f'{source}.dmnd'), cmd)

    def test_prepare_freezes_new_cache_and_rejects_overwrite(self):
        source = self.put('source.fa', '>sp|Q|FULL OX=4530\nMEFG\n')
        lin = self.put('cache.tsv', '4530\t4530;1\n')
        out = self.root / 'prepared'
        self.m.prepare_lineage(source, 'swissprot', self.m.sha256(source), lin, out, False)
        self.assertEqual(self.m.taxonomy.read_lineage(str(out / 'lineage.tsv')), {4530: [4530, 1]})
        self.assertTrue((out / 'DONE').exists())
        with self.assertRaises(FileExistsError):
            self.m.prepare_lineage(source, 'swissprot', self.m.sha256(source), lin, out, False)

    def test_production_guards_precede_engine(self):
        source = self.put('evidence/protein/odb12_Viridiplantae.filtered.fa.gz', 'not used before guard')
        query = self.put('evidence/est/Athaliana/univec/est.univec.fa.gz', 'not used before guard')
        lin = self.put('lineage.tsv', '4530\t4530;1\n')
        roles = self.put('roles.tsv', 'dataset\tspecies\trole\nest/Athaliana\tAthaliana\tc2_training_eligible\n')
        base = dict(species='Athaliana', source_kind='orthodb', source=source, source_sha256=self.m.sha256(source), query=query, query_sha256=self.m.sha256(query), lineage=lin, lineage_sha256=self.m.sha256(lin), roles=roles, roles_sha256=self.m.sha256(roles))
        for change in ({'species': 'Zmays'}, {'source_sha256': '0' * 64}, {'query': source}, {'source': query}):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.m.production_inputs(**{**base, **change})
        roles.write_text('dataset\tspecies\trole\nest/Athaliana\tAthaliana\tb1_validation_only\n')
        with self.assertRaises(ValueError):
            self.m.production_inputs(**{**base, 'roles_sha256': self.m.sha256(roles)})
        self.assertEqual(self.m.SPECIES['Osativa'], 4530)

    def test_screened_input_success_and_short_query_abort(self):
        source = self.put('odb12_Viridiplantae.filtered.fa.gz', '')
        query = self.put('est/Athaliana/univec/est.univec.fa.gz', '')
        lineage = self.put('lineage.tsv', '4530\t4530;1\n')
        roles = self.put('roles.tsv', 'dataset\tspecies\trole\nest/Athaliana\tAthaliana\tc2_training_eligible\n')
        self.put('est/Athaliana/univec/DONE', 'screened\n')
        self.put('est/Athaliana/univec/summary.json', '{}')
        with gzip.open(source, 'wt') as f:
            f.write('>4530_0:cross\n' + 'M' * 50 + '\n')
        with gzip.open(query, 'wt') as f:
            f.write('>est.1/full\n' + 'ATG' * 40 + '\n')
        kwargs = dict(species='Athaliana', source_kind='orthodb', source=source,
                      source_sha256=self.m.sha256(source), query=query, query_sha256=self.m.sha256(query),
                      lineage=lineage, lineage_sha256=self.m.sha256(lineage), roles=roles, roles_sha256=self.m.sha256(roles))
        self.assertEqual(self.m.production_inputs(**kwargs)['queries'], 1)
        with gzip.open(query, 'wt') as f:
            f.write('>short\nATG\n')
        kwargs['query_sha256'] = self.m.sha256(query)
        with self.assertRaises(ValueError):
            self.m.production_inputs(**kwargs)

    def test_subprocess_failure_propagates_without_done_or_overwrite(self):
        out = self.root / 'failed'
        with self.assertRaises(subprocess.CalledProcessError):
            with self.m.work_directory(out):
                self.m.execute([sys.executable, '-c', 'raise SystemExit(7)'], out / 'engine.log')
        self.assertFalse((out / 'DONE').exists())
        self.assertTrue((out / 'FAILED.json').exists())
        with self.assertRaises(FileExistsError):
            with self.m.work_directory(out):
                self.fail('existing workdir reused')

    def test_empty_filtered_input_is_rejected(self):
        source = self.put('same.fa', '>sp|X.1|SELF OX=3702\nMAAA\n')
        lin = self.put('lineage.tsv', '3702\t3702;1\n')
        with self.assertRaises(ValueError):
            self.m.filter_source(source, 'swissprot', lin, 3702, self.root / 'filtered.fa')
        self.assertFalse((self.root / 'filtered.fa').exists())

    def test_existing_resolver_handles_merged_id(self):
        cache = self.put('cache.tsv', 'taxid\tlineage\n')
        xml = '<TaxaSet><Taxon><TaxId>99</TaxId><LineageEx><Taxon><TaxId>1</TaxId></Taxon><Taxon><TaxId>3702</TaxId></Taxon></LineageEx></Taxon></TaxaSet>'
        resolved = self.m.taxonomy.fetch_lineage([88], str(cache), pause=0, fetch=lambda ids: xml)
        self.assertEqual(resolved[88], [88, 99, 3702, 1])
        self.assertEqual(self.m.taxonomy.excluded_taxids({3702}, resolved, {88}), {88})

    def test_sigterm_stops_engine_process_group_and_records_failure(self):
        out = self.root / 'interrupted'
        sleeper = "import subprocess,sys,time,os; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(os.getpid(),p.pid,flush=True); time.sleep(60)"
        worker = f'''import importlib.util, signal
from pathlib import Path
s=importlib.util.spec_from_file_location('launcher', {str(SCRIPT)!r})
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def stop(*args): raise SystemExit(143)
signal.signal(signal.SIGTERM, stop)
with m.work_directory({str(out)!r}) as out:
    m.execute([{sys.executable!r}, '-c', {sleeper!r}], out / 'engine.log')
'''
        process = subprocess.Popen([sys.executable, '-c', worker], start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pids = []
        try:
            deadline = time.monotonic() + 5
            log = out / 'engine.log'
            while time.monotonic() < deadline:
                if log.exists() and log.read_text().strip():
                    pids = list(map(int, log.read_text().split()))
                    break
                time.sleep(0.02)
            self.assertEqual(len(pids), 2, 'test child did not start')
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.fail('launcher did not terminate active engine after SIGTERM')
            self.assertNotEqual(process.returncode, 0)
            self.assertTrue((out / 'FAILED.json').exists())
            self.assertFalse((out / 'DONE').exists())
            for pid in pids:
                stat = Path(f'/proc/{pid}/stat')
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z', 'engine descendant still running')
        finally:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def test_slurm_refuses_missing_arguments(self):
        wrapper = SCRIPT.with_name('est_blastx_fixture.sbatch')
        self.assertTrue(wrapper.exists(), 'fixture Slurm wrapper not implemented')
        result = subprocess.run(['bash', str(wrapper)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Usage:', result.stderr)

    def test_raw_qc_preserves_short_hsps_and_rejects_bad_endpoints(self):
        raw = self.put('raw.tsv', 'q\t120\ts\t50\t90\t31\t1\t20\t20\t1e-9\t40\t-1\t20\n')
        self.assertEqual(self.m.validate_raw(raw)['hsps_lt_30aa'], 1)
        raw.write_text('q\t120\ts\t50\t0\t31\t1\t20\t20\t1e-9\t40\t-1\t20\n')
        with self.assertRaises(ValueError):
            self.m.validate_raw(raw)


if __name__ == '__main__':
    unittest.main()
