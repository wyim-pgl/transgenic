"""Small files exercise the actual streaming binder; helper is test-only SAM reader."""
import hashlib
import importlib
import json
import base64
import os
import signal
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision/scripts'))


def api():
    return importlib.import_module('evidence.est_real_sample')


def fixture(tmp_path, *, dna='ATG' * 30, protein='M' * 30, kept='1-90', flag=0, samseq=None):
    files = {}
    def put(key, text):
        p = tmp_path / key
        p.write_text(text)
        files[key] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    put('screened', '>q univec=keep:1-90/90\n' + dna + '\n')
    put('original', '>q\n' + dna + '\n')
    put('report', 'accession\tqlen\taction\tkept\tcategories\nq\t90\tkeep\t' + kept + '\t\n')
    put('reference', '>chr\n' + 'A' * 200 + '\n')
    put('bam', '@HD\tVN:1.6\n@SQ\tSN:chr\tLN:200\nq\t' + str(flag) + '\tchr\t1\t60\t90M\t*\t0\t0\t' + (samseq or dna) + '\t*\tts:A:+\n')
    row = 'q\t90\tp\t30\t1\t90\t1\t30\t30\t1e-9\t60\t1\t30\n'
    for source in ('orthodb', 'swissprot'):
        put(source + '_raw', row + row.replace('\t90\t1\t30\t30\t', '\t87\t1\t29\t29\t').replace('\t1\t30\n', '\t1\t29\n'))
        put(source + '_protein', '>p\n' + protein + '\n')
        put(source + '_provenance', json.dumps({'engine': 'fixture-v1'}))
        put(source + '_commands', json.dumps(['fixture-reader']))
    return {'species': 'Sitalica', 'arm': 'primary', 'files': files,
            'selection': {'source': 'orthodb', 'count': 16, 'prefix_lines': 1000},
            'searches': [{'source': s, 'raw': s + '_raw', 'protein': s + '_protein',
                          'c1_provenance': s + '_provenance', 'c1_commands': s + '_commands'}
                         for s in ('orthodb', 'swissprot')]}


def run(m, tmp_path, **kw):
    command = [sys.executable, '-c', 'import shutil,sys; shutil.copyfileobj(open(sys.argv[1],"rb"),sys.stdout.buffer)', m['files']['bam']['path']]
    return api().run(m, tmp_path / 'out', sam_command_for_test=command, **kw)


def records(tmp_path):
    return [json.loads(x) for x in (tmp_path / 'out/diagnostics.jsonl').read_text().splitlines()]


def test_actual_join_geometry_and_unknown_a11(tmp_path):
    r = run(fixture(tmp_path), tmp_path)
    assert r['status'] == 'COMPLETE'
    assert r['selected_qnames'] == ['q']
    assert r['selected_search_rows_complete'] and r['selected_bam_records_complete']
    assert not r['training_eligible'] and not r['full_c0'] and not r['molecule_counts_complete']
    assert r['a11_status'] == 'unassessable'
    hsps = [x for x in records(tmp_path) if x['type'] == 'hsp']
    assert len(hsps) == 4
    assert {x['length_accepted'] for x in hsps} == {True, False}
    assert all(x['geometry'][0]['result']['segments'] for x in hsps)
    assert all(x['raw']['ordinal'] in (1, 2) for x in hsps)
    assert all(v['fresh_whole_file_verified'] for v in r['sources'].values())


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'interval', 'mutation', 'linecap', 'noncanonical', 'orientation'])
def test_failure_or_source_bound_disposition(tmp_path, problem):
    m = fixture(tmp_path, protein='X' * 30 if problem == 'noncanonical' else 'M' * 30,
                kept='2-90' if problem == 'interval' else '1-90',
                flag=16 if problem == 'orientation' else 0)
    if problem in ('duplicate', 'missing', 'mutation'):
        p = Path(m['files']['orthodb_protein']['path'])
        p.write_text(p.read_text() * 2 if problem == 'duplicate' else '>other\nM\n')
        if problem != 'mutation':
            m['files']['orthodb_protein']['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    if problem == 'linecap':
        m['limits'] = {'line_bytes': 40}
    r = run(m, tmp_path)
    if problem in ('noncanonical', 'orientation'):
        assert r['status'] == 'COMPLETE'
        hsps = [x for x in records(tmp_path) if x['type'] == 'hsp']
        assert all(('noncanonical protein' in x.get('disposition', '') if problem == 'noncanonical'
                    else 'SEQ identity mismatch' in x['geometry'][0]['error']) for x in hsps)
    else:
        assert r['status'] in ('FAILED', 'INCOMPLETE')
        assert not (tmp_path / 'out/COMPLETE').exists()
        assert not r['selected_search_rows_complete']


@pytest.mark.parametrize('parts', [False, True])
def test_report_proven_single_survivor_and_parts(tmp_path, parts):
    m = fixture(tmp_path)
    def replace(key, text):
        p = Path(m['files'][key]['path'])
        p.write_text(text)
        m['files'][key]['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    name = 'q_part1' if parts else 'q'
    replace('original', '>q\n' + 'C' * 10 + 'ATG' * 30 + 'G' * 100 + '\n')
    replace('screened', '>' + name + ' univec=split:11-100/200\n' + 'ATG' * 30 + '\n')
    replace('report', 'accession\tqlen\taction\tkept\tcategories\nq\t200\tsplit\t11-100' + (';101-200' if parts else '') + '\t\n')
    for key in ('orthodb_raw', 'swissprot_raw', 'bam'):
        replace(key, Path(m['files'][key]['path']).read_text().replace('q\t', name + '\t'))
    r = run(m, tmp_path)
    assert r['status'] == 'COMPLETE'
    member = next(x for x in records(tmp_path) if x['type'] == 'member')
    assert member['relationship']['original_id'] == 'q'
    assert member['identity']['source_start0'] == 10


def test_mutation_during_sam_and_partial_nonzero_eof(tmp_path):
    m = fixture(tmp_path)
    command = [sys.executable, '-c', 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_text(p.read_text()+"\\n")', m['files']['original']['path']]
    r = api().run(m, tmp_path / 'out', sam_command_for_test=command)
    assert r['status'] == 'FAILED'
    assert not r['selected_bam_records_complete']


def test_receipt_binding_mismatch(tmp_path):
    m = fixture(tmp_path)
    m['files']['orthodb_raw']['receipt'] = {'file': 'report', 'field': ['raw_sha256']}
    r = run(m, tmp_path)
    assert r['status'] == 'FAILED'
    assert not r['selected_search_rows_complete']


def test_cli_compute_enforcement(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(SystemExit):
        api().main(['--manifest', 'none', '--sha', 'none', '--out', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def repin(m, key, text):
    p = Path(m['files'][key]['path'])
    p.write_text(text)
    m['files'][key]['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize('key', ['screened', 'original', 'report'])
def test_duplicate_selected_identity_fails_closed(tmp_path, key):
    m = fixture(tmp_path)
    text = Path(m['files'][key]['path']).read_text()
    repin(m, key, text + (text.split('\n', 1)[1] if key == 'report' else text))
    r = run(m, tmp_path)
    assert r['status'] == 'FAILED'
    assert 'duplicate' in r['error']


@pytest.mark.parametrize('kind', ['noncanonical_dna', 'bad_trace', 'self_overlap', 'valid_reverse', 'secondary', 'no_placement'])
def test_precise_diagnostics_not_biological_zero(tmp_path, kind):
    m = fixture(tmp_path, dna='A' * 90 if kind == 'self_overlap' else 'ATG' * 30,
                protein='K' * 30 if kind == 'self_overlap' else 'M' * 30,
                flag=16 if kind == 'valid_reverse' else 256 if kind == 'secondary' else 0,
                samseq='CAT' * 30 if kind == 'valid_reverse' else None)
    if kind == 'noncanonical_dna':
        for key in ('original', 'screened', 'bam'):
            repin(m, key, Path(m['files'][key]['path']).read_text().replace('ATG', 'NTG'))
    if kind in ('bad_trace', 'self_overlap'):
        for key in ('orthodb_raw', 'swissprot_raw'):
            row = Path(m['files'][key]['path']).read_text().splitlines()[0]
            f = row.split('\t')
            f[12] = '29' if kind == 'bad_trace' else '15/-15'
            if kind == 'self_overlap':
                f[5] = '89'
            repin(m, key, '\t'.join(f) + '\n')
    if kind == 'no_placement':
        repin(m, 'bam', Path(m['files']['bam']['path']).read_text().replace('q\t', 'other\t'))
    r = run(m, tmp_path)
    assert r['status'] == 'COMPLETE'
    hsps = [x for x in records(tmp_path) if x['type'] == 'hsp']
    for h in hsps:
        assert not h['biological_negative']
        if kind == 'noncanonical_dna':
            assert 'noncanonical DNA' in h['disposition']
        elif kind == 'bad_trace':
            assert 'source_bound_replay_failure' in h['disposition']
        elif kind == 'self_overlap':
            assert h['length_accepted']
            assert h['replay']['frameshifts'][0]['delta'] == -1
            assert not h['geometry'][0]['geometry_success']
            assert 'self_overlap_policy_unresolved' in h['geometry'][0]['result']['reasons']
        elif kind == 'valid_reverse':
            assert h['geometry'][0]['geometry_success']
            assert h['geometry'][0]['result']['segments'][0]['strand'] == '-'
        elif kind == 'secondary':
            assert h['geometry'][0]['disposition'] == 'nonprimary_or_unmapped_diagnostic'
        else:
            assert h['placement_disposition'] == 'no_selected_qname_records_seen_not_biological_zero'


@pytest.mark.parametrize('cap', ['expanded_bases', 'retained_bytes', 'records', 'read_bytes'])
def test_caps_preserve_incomplete(tmp_path, cap):
    m = fixture(tmp_path)
    m['limits'] = {cap: 1}
    r = run(m, tmp_path)
    assert r['status'] == 'INCOMPLETE'
    assert not r['selected_search_rows_complete']
    assert (tmp_path / 'out/INCOMPLETE').exists()
    assert not (tmp_path / 'out/COMPLETE').exists()


@pytest.mark.parametrize('kind', ['mutation', 'nonzero', 'timeout'])
def test_no_completion_before_successful_eof_and_final_stats(tmp_path, kind):
    m = fixture(tmp_path)
    suffix = {'mutation': 'p=pathlib.Path(sys.argv[2]); p.write_text(p.read_text()+"\\n")',
              'nonzero': 'sys.exit(2)', 'timeout': 'time.sleep(5)'}[kind]
    code = 'import pathlib,sys,shutil,time; shutil.copyfileobj(open(sys.argv[1],"rb"),sys.stdout.buffer); sys.stdout.flush(); ' + suffix
    if kind == 'timeout':
        m['limits'] = {'seconds': 1}
    r = api().run(m, tmp_path / 'out', sam_command_for_test=[sys.executable, '-c', code,
                   m['files']['bam']['path'], m['files']['original']['path']])
    assert r['status'] == ('INCOMPLETE' if kind == 'timeout' else 'FAILED')
    assert not r['selected_bam_records_complete']
    assert not (tmp_path / 'out/COMPLETE').exists()
    assert {'mutation': 'source mutation', 'nonzero': 'nonzero exit', 'timeout': 'signal/timeout'}[kind] in r['error']


def test_expansion_cap_uses_trace_not_untrusted_reported_length(tmp_path):
    m = fixture(tmp_path)
    f = Path(m['files']['orthodb_raw']['path']).read_text().splitlines()[0].split('\t')
    f[8], f[12] = '1', '400000'
    repin(m, 'orthodb_raw', '\t'.join(f) + '\n')
    r = run(m, tmp_path)
    assert r['status'] == 'INCOMPLETE'
    assert 'expanded_bases' in r['error']


@pytest.mark.parametrize('field', ['c1_provenance', 'c1_commands'])
def test_search_and_hsp_ids_bind_search_contract(tmp_path, field):
    m = fixture(tmp_path)
    assert run(m, tmp_path)['status'] == 'COMPLETE'
    before = records(tmp_path)
    key = m['searches'][0][field]
    repin(m, key, json.dumps({'changed': field}))
    second = tmp_path / 'second'
    second.mkdir()
    assert run(m, second)['status'] == 'COMPLETE'
    after = records(second)
    for kind in ('search', 'hsp'):
        a = [x['id'] for x in before if x['type'] == kind]
        b = [x['id'] for x in after if x['type'] == kind]
        assert a[0] != b[0]
        assert a[-1] == b[-1]  # untouched SwissProt contract
    search = next(x for x in after if x['type'] == 'search')
    assert search['preimage'][field + '_sha256'] == m['files'][key]['sha256']


@pytest.mark.parametrize('bad', ['missing', 'unknown', 'unpinned', 'malformed'])
def test_search_contract_requires_explicit_pinned_specs(tmp_path, bad):
    m = fixture(tmp_path)
    search = m['searches'][0]
    if bad == 'missing':
        del search['c1_commands']
    elif bad == 'unknown':
        search['c1_commands'] = 'absent'
    elif bad == 'unpinned':
        del m['files'][search['c1_commands']]['sha256']
    else:
        m['files'][search['c1_commands']] = []
    r = run(m, tmp_path)
    assert r['status'] == 'FAILED'
    assert 'search contract' in r['error']
    assert not r['selected_search_rows_complete']


def test_abort_kills_descendant_after_leader_exit(tmp_path):
    m = fixture(tmp_path)
    m['limits'] = {'seconds': 1}
    pidfile = tmp_path / 'child.pid'
    code = ('import subprocess,sys,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(10)"]); '
            'pathlib.Path(sys.argv[1]).write_text(str(p.pid))')
    try:
        r = api().run(m, tmp_path / 'out',
                      sam_command_for_test=[sys.executable, '-c', code, str(pidfile)])
        assert r['status'] == 'INCOMPLETE'
        child = int(pidfile.read_text())
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            path = Path('/proc') / str(child) / 'stat'
            if not path.exists() or path.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail('stdout-holding descendant survived abort cleanup')
        assert (tmp_path / 'out/INCOMPLETE').exists()
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_reader_stderr_preserves_bounded_bytes_hash_and_returncode(tmp_path):
    m = fixture(tmp_path)
    payload = b'UNIQUE_BAM_CORRUPTION_DIAGNOSTIC\n' + b'x' * 200000
    # The full marker is NOT in the recorded command; only actual stderr can supply it.
    code = ('import sys;sys.stderr.buffer.write(b"UNIQUE_BAM_" + '
            'b"CORRUPTION_DIAGNOSTIC\\n" + b"x"*200000);sys.exit(7)')
    r = api().run(m, tmp_path / 'out', sam_command_for_test=[sys.executable, '-c', code])
    assert r['status'] == 'FAILED' and 'nonzero exit: 7' in r['error']
    stderr = r['sam_stderr']
    assert base64.b64decode(stderr['bytes_base64']) == payload[:65536]
    assert stderr['total_bytes'] == len(payload)
    assert stderr['sha256'] == hashlib.sha256(payload).hexdigest()
    assert stderr['truncated'] and stderr['eof']
    assert r['sam_returncode'] == 7
    assert r['resources']['used']['retained_bytes'] >= 65536
    assert r['resources']['used']['output_bytes'] >= len(stderr['bytes_base64'])


def test_cleanup_error_does_not_prevent_failure_receipt(tmp_path, monkeypatch):
    m = fixture(tmp_path)
    m['limits'] = {'seconds': 1}
    real_killpg = os.killpg

    def kill_then_error(pid, sig):
        real_killpg(pid, sig)  # own helper is cleaned even when exercising error handling
        raise PermissionError('injected cleanup diagnostic')

    monkeypatch.setattr(api().os, 'killpg', kill_then_error)
    r = api().run(m, tmp_path / 'out',
                  sam_command_for_test=[sys.executable, '-c', 'import time;time.sleep(10)'])
    assert r['status'] == 'INCOMPLETE'
    assert r['sam_returncode'] == -signal.SIGKILL
    assert 'injected cleanup diagnostic' in r['sam_cleanup_errors'][0]
    data = (tmp_path / 'out/report.json').read_bytes()
    assert (tmp_path / 'out/INCOMPLETE').read_text().strip() == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('cap', ['read_bytes', 'retained_bytes'])
def test_stderr_is_charged_and_caps_fail_closed(tmp_path, cap):
    m = fixture(tmp_path)
    m['limits'] = {cap: 60000}
    code = 'import sys;sys.stderr.buffer.write(b"x"*200000);sys.stdout.buffer.write(open(sys.argv[1],"rb").read())'
    r = api().run(m, tmp_path / 'out',
                  sam_command_for_test=[sys.executable, '-c', code, m['files']['bam']['path']])
    assert r['status'] == 'INCOMPLETE'
    assert cap in r['error']
    assert not r['selected_bam_records_complete']
    assert (tmp_path / 'out/INCOMPLETE').exists()


def test_stderr_large_binary_prefix_does_not_block_valid_sam(tmp_path):
    m = fixture(tmp_path)
    code = ('import sys;sys.stderr.buffer.write(bytes(range(256))*1000);'
            'sys.stderr.flush();sys.stdout.buffer.write(open(sys.argv[1],"rb").read())')
    r = api().run(m, tmp_path / 'out',
                  sam_command_for_test=[sys.executable, '-c', code, m['files']['bam']['path']])
    assert r['status'] == 'COMPLETE'
    stderr = r['sam_stderr']
    assert base64.b64decode(stderr['bytes_base64']) == bytes(range(256)) * 256
    assert stderr['sha256'] == hashlib.sha256(bytes(range(256)) * 1000).hexdigest()
    assert stderr['total_bytes'] == 256000 and stderr['eof']
    assert r['sam_returncode'] == 0
    assert r['resources']['used']['read_bytes'] >= 256000
    assert sum(p.stat().st_size for p in (tmp_path / 'out').iterdir()) <= r['resources']['limits']['output_bytes']


def test_exclusive_output_and_domain_scope(tmp_path):
    m = fixture(tmp_path)
    run(m, tmp_path)
    with pytest.raises(FileExistsError):
        run(m, tmp_path)
    assert api().bound_id('member', {'a': 'x|y', 'b': 'z'}) != api().bound_id('member', {'a': 'x', 'b': 'y|z'})
    assert api().bound_id('member', {'a': 1}) != api().bound_id('hsp', {'a': 1})
