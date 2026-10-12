"""Independent tiny local fixtures; observed-defect tests assert reproduction, not approval."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import pytest

ROOT = Path('/tmp/est-real-source-binding-20260917')
spec = importlib.util.spec_from_file_location('binder_fixtures', ROOT / 'tests/test_est_real_sample.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def execute(m, out, code=None):
    command = [sys.executable, '-c', code or 'import sys;sys.stdout.buffer.write(open(sys.argv[1],"rb").read())', m['files']['bam']['path']]
    r = f.api().run(m, out, sam_command_for_test=command)
    data = (out / 'report.json').read_bytes()
    assert (out / r['status']).read_text().strip() == hashlib.sha256(data).hexdigest()
    assert r['diagnostics_sha256'] == hashlib.sha256((out / 'diagnostics.jsonl').read_bytes()).hexdigest()
    assert not r['training_eligible'] and not r['route_a_complete'] and r['a11_status'] == 'unassessable'
    return r, [json.loads(x) for x in (out / 'diagnostics.jsonl').read_text().splitlines()]


def add_provenance(m, directory, version):
    for s in m['searches']:
        key = s['source'] + '_provenance'
        p = directory / key
        p.write_text(json.dumps({'diamond_version': version, 'raw_sha256': m['files'][s['raw']]['sha256']}))
        m['files'][key] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        m['files'][s['raw']]['receipt'] = {'file': key, 'field': ['raw_sha256']}
        s['c1_provenance'] = key
        ck = s['source'] + '_commands'
        cp = directory / ck
        cp.write_text(json.dumps(['diamond', '--sensitivity', version]))
        m['files'][ck] = {'path': str(cp), 'sha256': hashlib.sha256(cp.read_bytes()).hexdigest()}
        s['c1_commands'] = ck


def test_reproduce_search_and_hsp_identity_collision(tmp_path):
    m = f.fixture(tmp_path)
    add_provenance(m, tmp_path, 'engine-A')
    a, ar = execute(m, tmp_path / 'a')
    add_provenance(m, tmp_path, 'engine-B')
    b, br = execute(m, tmp_path / 'b')
    assert a['status'] == b['status'] == 'COMPLETE'
    assert a['canonical_manifest_sha256'] != b['canonical_manifest_sha256']
    for kind in ('search', 'hsp'):
        assert [x['id'] for x in ar if x['type'] == kind] == [x['id'] for x in br if x['type'] == kind]
    assert a['sources']['orthodb_raw']['receipt_bound'] and b['sources']['orthodb_raw']['receipt_bound']
    print('REPRO: different pinned commands/engine provenance, identical search/HSP IDs; report manifests differ')


def test_reproduce_stderr_loss(tmp_path):
    m = f.fixture(tmp_path)
    r, _ = execute(m, tmp_path / 'out', 'import sys;sys.stderr.write("UNIQUE_BAM_" + "CORRUPTION_DIAGNOSTIC\\n");sys.exit(7)')
    assert r['status'] == 'FAILED' and 'nonzero exit: 7' in r['error']
    assert not any(b'UNIQUE_BAM_CORRUPTION_DIAGNOSTIC' in p.read_bytes() for p in (tmp_path / 'out').iterdir())
    print('REPRO: real child diagnostic discarded; only exit 7 remains')


def test_reproduce_exited_parent_descendant_cleanup_gap(tmp_path):
    m = f.fixture(tmp_path)
    m['limits'] = {'seconds': 1}
    pidfile = tmp_path / 'child.pid'
    code = ('import subprocess,sys,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]); '
            f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))')
    try:
        r, _ = execute(m, tmp_path / 'out', code)
        assert r['status'] == 'INCOMPLETE'
        child = int(pidfile.read_text())
        os.kill(child, 0)
        assert Path(f'/proc/{child}/stat').read_text().split()[2] != 'Z'
        print('REPRO: leader exited, stdout-holding child survives timeout cleanup (reviewer kills it)')
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('kind', ['duplicate_ts', 'missing_seq', 'wrong_contig_length'])
def test_sam_boundary_diagnostics(tmp_path, kind):
    m = f.fixture(tmp_path)
    text = Path(m['files']['bam']['path']).read_text()
    if kind == 'duplicate_ts':
        text = text.replace('ts:A:+', 'ts:A:+\tts:A:-')
    elif kind == 'missing_seq':
        text = text.replace('ATG' * 30, '*')
    else:
        text = text.replace('LN:200', 'LN:199')
    f.repin(m, 'bam', text)
    r, rows = execute(m, tmp_path / 'out')
    if kind == 'wrong_contig_length':
        assert r['status'] == 'FAILED' and not r['selected_bam_records_complete']
    else:
        assert r['status'] == 'COMPLETE'
        hsps = [x for x in rows if x['type'] == 'hsp']
        assert all(not h['geometry'][0]['geometry_success'] for h in hsps)
        assert all(('duplicate ts' if kind == 'duplicate_ts' else 'SEQ unavailable') in h['geometry'][0]['error'] for h in hsps)


def test_output_cap_and_marker_integrity(tmp_path):
    m = f.fixture(tmp_path)
    m['limits'] = {'output_bytes': 2 * 1024**2}
    text = Path(m['files']['bam']['path']).read_text()
    f.repin(m, 'bam', text.split('q\t')[0] + ('@CO\t' + 'x' * 20000 + '\n') * 50 + 'q\t' + text.split('q\t')[1])
    r, _ = execute(m, tmp_path / 'out')
    assert r['status'] == 'INCOMPLETE' and 'output_bytes' in r['error']
    assert sum(p.stat().st_size for p in (tmp_path / 'out').iterdir()) <= 2 * 1024**2


@pytest.mark.parametrize('bad', [False, True])
def test_text_md5_receipt(tmp_path, bad):
    m = f.fixture(tmp_path)
    digest = hashlib.md5(Path(m['files']['bam']['path']).read_bytes()).hexdigest()
    p = tmp_path / 'receipt'
    p.write_text('bam_md5 ' + ('0' * 32 if bad else digest) + '\n')
    m['files']['receipt'] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    m['files']['bam'].update(md5=digest, receipt={'file': 'receipt', 'text_key': 'bam_md5', 'algorithm': 'md5'})
    r, _ = execute(m, tmp_path / 'out')
    assert r['status'] == ('FAILED' if bad else 'COMPLETE')
    assert r['sources']['bam']['receipt_bound'] == (not bad)


def test_cli_manifest_mismatch_does_not_read_sources(tmp_path, monkeypatch):
    p = tmp_path / 'manifest'
    p.write_text('{"files":{"huge":{"path":"/must/not/open"}}}')
    monkeypatch.setenv('SLURM_JOB_ID', 'local-validation-only')
    with pytest.raises(SystemExit):
        f.api().main(['--manifest', str(p), '--sha', '0' * 64, '--out', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()
