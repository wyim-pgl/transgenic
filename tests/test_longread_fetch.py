"""Offline driver tests: saved ENA metadata plus tiny controlled FASTQ payloads."""
import csv
import gzip
import hashlib
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'revision/scripts/evidence'
DRIVER = SCRIPTS / 'longread_fetch_v2.sh'
FIXTURES = Path(__file__).parent / 'fixtures/longread_fetch'
SEQKIT = Path('/data/gpfs/assoc/pgl/bin/conda/conda_envs/GPCR/bin/seqkit')


def rows(name):
    with (FIXTURES / name).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


@pytest.fixture
def harness(tmp_path):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    curl = bindir / 'curl'
    curl.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
args=sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f: f.write(repr(args)+'\\n')
out=pathlib.Path(args[args.index('-o')+1])
url=args[args.index('-o')-1]
src=pathlib.Path(os.environ['PAYLOADS']) / url.rsplit('/',1)[-1]
if not src.exists(): sys.exit(22)
data=src.read_bytes()
if os.environ.get('INTERRUPT') and not out.exists():
    out.write_bytes(data[:len(data)//2]); sys.exit(18)
start=out.stat().st_size if out.exists() else 0
with out.open('ab') as f: f.write(data[start:])
''')
    curl.chmod(0o755)
    payloads = tmp_path / 'payloads'
    payloads.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in (
        'RUN_RE', 'PLATFORM_RE', 'SUBMITTED_RE', 'MODEL_RE', 'STRAT_ALLOW',
        'MAX_READS', 'FILT', 'LONGREAD_FILEREPORT', 'LONGREAD_ROOT')}
    env.update(PATH=f'{bindir}:'+env['PATH'], LONGREAD_ROOT=str(tmp_path / 'output'),
               LONGREAD_RETRY_DELAY='0', PAYLOADS=str(payloads), CALLS=str(tmp_path / 'calls'),
               SEQKIT=str(SEQKIT))

    def run(data=None, filters=None, arg=None, driver=DRIVER):
        report = tmp_path / 'report.tsv'
        if isinstance(data, str):
            report.write_bytes((FIXTURES / data).read_bytes())
        else:
            with report.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]), delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows(data)
        callenv = dict(env, LONGREAD_FILEREPORT=str(report), **(filters or {}))
        result = subprocess.run(['bash', str(driver), 'ont', 'test', 'PRJNA594286'] +
                                ([] if arg is None else [arg]), env=callenv,
                                text=True, capture_output=True, timeout=30)
        return result, Path(env['LONGREAD_ROOT']) / 'ont/test'

    def controlled(submitted=False):
        row = rows('PRJNA594286.tsv')[1].copy()
        row['read_count'] = '2'
        urls, sums = [], []
        for i in range(2):
            name = f'file{i}.fastq.gz'
            data = gzip.compress(f'@read{i} description\nACGT\n+\nIIII\n'.encode(), mtime=0)
            (payloads / name).write_bytes(data)
            urls.append('ftp.example/' + name)
            sums.append(hashlib.md5(data).hexdigest())
        row['fastq_ftp'] = ';'.join(urls)
        row['fastq_md5'] = ';'.join(sums)
        row['submitted_ftp'] = ''
        row['submitted_format'] = ''
        if submitted:
            row['submitted_ftp'] = row['fastq_ftp']
            row['submitted_format'] = 'FASTQ;FASTQ'
            row['submitted_md5'] = row['fastq_md5']
            row['fastq_md5'] = ';'.join(['0'*32]*2)  # deliberately WRONG for submitted bytes
        return row

    return run, controlled, env, payloads


@pytest.mark.parametrize('arg', ['OXFORD_NANOPORE', 'PACBIO', 'SRR10611195[[:space:]]'])
def test_legacy_guard_before_any_write(harness, arg):
    run, _, env, _ = harness
    result, out = run('PRJNA594286.tsv', arg=arg)
    assert result.returncode == 6, result.stderr
    assert 'legacy whole-row' in result.stderr
    assert not out.exists()
    assert not Path(env['CALLS']).exists()


@pytest.mark.parametrize('arg', ['maize', '^NONEXISTENT$', 'SRR10611195[ \\t]'])
def test_unrecognized_stale_expression_is_loud(harness, arg):
    run, _, env, _ = harness
    result, out = run('PRJEB22122.tsv', arg=arg)
    assert result.returncode == 6, result.stderr
    assert 'empty selection' in result.stderr
    assert not list(out.glob('*.DONE'))
    assert not Path(env['CALLS']).exists()


@pytest.mark.parametrize('name,filters,count', [
    ('PRJNA594286.tsv', {'PLATFORM_RE': 'OXFORD_NANOPORE'}, 3),
    ('PRJEB22122.tsv', {'SUBMITTED_RE': 'maize'}, 5),
    ('PRJNA953663.tsv', {'PLATFORM_RE': 'OXFORD_NANOPORE'}, 1),
    ('PRJDB38182.tsv', {'RUN_RE': '^DRR807190$'}, 1),
])
def test_saved_report_selection(harness, name, filters, count):
    run, _, env, _ = harness
    # Deliberately don't fetch real ENA data: sentinel DONEs isolate selection.
    out = Path(env['LONGREAD_ROOT']) / 'ont/test'
    out.mkdir(parents=True)
    for row in rows(name):
        (out / (row['run_accession'] + '.DONE')).touch()
    result, _ = run(name, filters)
    assert result.returncode == 0, result.stderr
    assert f'{count} selected, {count} runs DONE' in result.stderr
    assert not Path(env['CALLS']).exists()


@pytest.mark.parametrize('filter_name,column', [
    ('PLATFORM_RE', 'instrument_platform'), ('SUBMITTED_RE', 'submitted_ftp'),
    ('MODEL_RE', 'instrument_model'), ('STRAT_ALLOW', 'library_strategy'),
])
@pytest.mark.parametrize('regex', ['.', ''])
def test_missing_filter_target_unresolved_even_empty_regex(harness, filter_name, column, regex):
    run, controlled, env, _ = harness
    row = controlled()
    row[column] = ''
    result, out = run([row], {filter_name: regex})
    assert result.returncode == 4, result.stderr
    assert list(out.glob('*.UNRESOLVED'))
    assert not list(out.glob('*.DONE'))
    assert not Path(env['CALLS']).exists()


def test_empty_tabs_keep_read_count_and_model_aligned(harness):
    run, controlled, _, _ = harness
    row = controlled()
    result, out = run([row], {'MODEL_RE': '^PromethION$'})
    assert result.returncode == 0, result.stderr
    assert 'reads=2' in next(out.glob('*.DONE')).read_text()


@pytest.mark.parametrize('submitted', [False, True])
def test_every_file_checksum_count_ids_and_atomic_completion(harness, submitted):
    run, controlled, env, _ = harness
    row = controlled(submitted)
    result, out = run([row])
    assert result.returncode == 0, result.stderr
    assert gzip.decompress(next(out.glob('*.fa.gz')).read_bytes()) == (
        b'>read0 description\nACGT\n>read1 description\nACGT\n')
    assert len(next(out.glob('*.source.md5')).read_text().splitlines()) == 2
    assert 'script_sha256=' in next(out.glob('*.DONE')).read_text()
    assert not list(out.glob('*.raw.gz'))
    assert not list(out.glob('*.part'))
    assert len(Path(env['CALLS']).read_text().splitlines()) == 2


@pytest.mark.parametrize('problem', ['count', 'second_missing', 'second_md5', 'conversion'])
def test_failures_keep_sources_and_never_publish(harness, problem):
    run, controlled, env, payloads = harness
    row = controlled()
    filters = {}
    if problem == 'count': row['read_count'] = '3'
    elif problem == 'second_missing': (payloads / 'file1.fastq.gz').unlink()
    elif problem == 'second_md5': row['fastq_md5'] = row['fastq_md5'].split(';')[0] + ';' + '0'*32
    else: filters['SEQKIT'] = '/bin/false'
    result, out = run([row], filters)
    assert result.returncode == 5, result.stderr
    assert list(out.glob('*.FAILED'))
    assert list(out.glob('*.raw.gz'))
    assert not list(out.glob('*.DONE'))
    assert not list(out.glob('*.fa.gz'))
    assert not list(out.glob('*.part'))
    if problem == 'count': assert 'converted=2 ENA=3' in result.stderr


def test_resume_and_recovery_clear_markers(harness):
    run, controlled, env, _ = harness
    row = controlled()
    row['read_count'] = '3'
    result, out = run([row], {'INTERRUPT': '1'})
    assert result.returncode == 5, result.stderr
    calls = Path(env['CALLS']).read_text()
    assert len(calls.splitlines()) == 4  # each partial resumed, not restarted
    assert all("'-C', '-'" in line and "'--http1.1'" in line for line in calls.splitlines())
    (out / (row['run_accession'] + '.UNRESOLVED')).touch()
    row['read_count'] = '2'
    result, _ = run([row])
    assert result.returncode == 0, result.stderr
    assert Path(env['CALLS']).read_text() == calls  # validated raws reused
    assert not list(out.glob('*.FAILED')) and not list(out.glob('*.UNRESOLVED'))


@pytest.mark.parametrize('problem', ['missing_submitted_md5', 'unpaired', 'missing_count', 'mixed_formats'])
def test_unverifiable_metadata_is_unresolved(harness, problem):
    run, controlled, env, _ = harness
    row = controlled(submitted=True)
    if problem == 'missing_submitted_md5': del row['submitted_md5']
    elif problem == 'unpaired': row['submitted_md5'] = row['submitted_md5'].split(';')[0]
    elif problem == 'missing_count': row['read_count'] = ''
    else: row['submitted_format'] = 'FASTQ;BAM'
    result, out = run([row])
    assert result.returncode == 4, result.stderr
    assert list(out.glob('*.UNRESOLVED')) and not list(out.glob('*.DONE'))
    assert not Path(env['CALLS']).exists()


def test_saved_submitted_md5_absent_is_not_fastq_md5(harness):
    run, _, env, _ = harness
    result, out = run('PRJEB22122.tsv', {'SUBMITTED_RE': 'maize'})
    assert result.returncode == 4, result.stderr
    assert len(list(out.glob('*.UNRESOLVED'))) == 5
    assert not Path(env['CALLS']).exists()


def test_invalid_regex_and_missing_run_are_loud(harness):
    run, controlled, _, _ = harness
    result, _ = run([controlled()], {'RUN_RE': '['})
    assert result.returncode == 6
    row = controlled(); row['run_accession'] = ''
    result, _ = run([row])
    assert result.returncode == 4


def test_subreads_still_excluded(harness):
    run, _, env, _ = harness
    result, _ = run('PRJNA594286.tsv', {'PLATFORM_RE': 'PACBIO'})
    assert result.returncode == 0, result.stderr
    assert 'skipped: subreads' in result.stderr
    assert not Path(env['CALLS']).exists()


@pytest.mark.parametrize('name', ['longread_run_all', 'training_run_all', 'training_run_ont_missing',
                                  'queue_fetch', 'ont_parallel'])
def test_all_callers_propagate_failure_without_finished(harness, name):
    _, _, env, _ = harness
    out = Path(env['LONGREAD_ROOT'])
    out.mkdir()
    stub = out / 'longread_fetch.sh'
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> args\nexit 6\n')
    stub.chmod(0o755)
    queue = out / 'ont_parallel.tsv'
    queue.write_text('ont\ttest\tPRJNA594286\tSRR10611195\n')
    args = [str(queue)] if name == 'queue_fetch' else []
    result = subprocess.run(['bash', str(SCRIPTS / (name + '.sh')), *args], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 6, result.stderr
    assert not (out / 'longread.log').exists()
    if name in ('queue_fetch', 'ont_parallel'):
        assert (out / 'args').read_text().splitlines()[-1] == '^SRR10611195$'


def test_all_literal_caller_filters_have_migrated():
    import shlex
    calls = 0
    for name in ('longread_run_all', 'training_run_all', 'training_run_ont_missing'):
        for line in (SCRIPTS / (name + '.sh')).read_text().splitlines():
            if "$S " not in line:
                continue
            words = shlex.split(line, comments=True)
            if '$S' not in words:
                continue
            idx = words.index('$S')
            args = words[idx + 1:]
            assert len(args) in (3, 4)
            if len(args) == 4:
                assert args[3].startswith(('SRR', 'ERR', 'DRR'))
            else:
                assert any(w.startswith(('PLATFORM_RE=', 'SUBMITTED_RE=')) for w in words[:idx])
            calls += 1
    assert calls == 35


def test_failed_validation_keeps_previous_final(harness):
    run, controlled, env, _ = harness
    row = controlled(); row['read_count'] = '3'
    out = Path(env['LONGREAD_ROOT']) / 'ont/test'
    out.mkdir(parents=True)
    final = out / (row['run_accession'] + '.fa.gz')
    final.write_bytes(b'previous output')
    result, _ = run([row])
    assert result.returncode == 5
    assert final.read_bytes() == b'previous output'


def test_saved_b73_run_regex(harness):
    run, _, env, _ = harness
    out = Path(env['LONGREAD_ROOT']) / 'ont/test'; out.mkdir(parents=True)
    for row in rows('PRJNA1470126.tsv'):
        (out / (row['run_accession'] + '.DONE')).touch()
    result, _ = run('PRJNA1470126.tsv', {'RUN_RE': '^SRR388187(69|70|71)$'})
    assert result.returncode == 0, result.stderr
    assert '3 selected, 3 runs DONE' in result.stderr


def test_header_column_order_is_parsed(harness):
    run, controlled, _, _ = harness
    row = dict(reversed(list(controlled().items())))
    result, _ = run([row])
    assert result.returncode == 0, result.stderr


def test_stale_done_does_not_hide_empty_selection(harness):
    run, controlled, env, _ = harness
    out = Path(env['LONGREAD_ROOT']) / 'ont/test'; out.mkdir(parents=True)
    (out / 'SRR1.DONE').touch()
    result, _ = run([controlled()], arg='maize')
    assert result.returncode == 6
    assert 'empty selection' in result.stderr


def test_swapped_entrypoint_and_symlink_keep_guard(harness):
    run, _, env, _ = harness
    alias = Path(env['LONGREAD_ROOT']).parent / 'fetch-link.sh'
    alias.symlink_to(SCRIPTS / 'longread_fetch.sh')
    result, out = run('PRJEB22122.tsv', arg='maize', driver=alias)
    assert result.returncode == 6, result.stderr
    assert 'empty selection' in result.stderr
    assert not list(out.glob('*.DONE'))
