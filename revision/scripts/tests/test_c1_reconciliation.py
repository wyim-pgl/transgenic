"""Read-only synthetic trees; anchors patched only inside tests, never production CLI."""
import datetime
import hashlib
import json
from pathlib import Path

import pytest
from test_track_c_recovery import audit, make_audit_tree, synthetic_identity

RECEIPT = Path(__file__).with_name('pilot_submission_recovered.json')


def pilot(tmp_path, monkeypatch):
    tree, sacct = make_audit_tree(tmp_path)
    monkeypatch.setattr(audit, 'HISTORICAL_ROOT', tree, raising=False)
    root = tree / audit.PILOT_DIR
    (root / 'submission.lock').mkdir()
    (root / 'submission.lock/jobid').write_text('6163851\n')
    (root / 'slurm-6163851.out').write_text('historical stdout')
    (root / 'slurm-6163851.err').write_text('')
    return tree, sacct, root


@pytest.mark.parametrize('mutation', ['none', 'absent_receipt', 'altered_receipt', 'wrong_job',
    'missing_err', 'renamed_out', 'wrong_wrapper', 'wrong_root', 'symlink_jobid'])
def test_pilot_logs_need_verified_receipt_and_exact_wrapper(tmp_path, monkeypatch, mutation):
    tree, _, root = pilot(tmp_path, monkeypatch)
    receipt = RECEIPT
    if mutation == 'absent_receipt':
        receipt = None
    elif mutation == 'altered_receipt':
        receipt = tmp_path / 'receipt.json'
        receipt.write_text(RECEIPT.read_text().replace('6163851', '6163852'))
    elif mutation == 'wrong_job':
        (root / 'submission.lock/jobid').write_text('6163852\n')
    elif mutation == 'missing_err':
        (root / 'slurm-6163851.err').unlink()
    elif mutation == 'renamed_out':
        (root / 'slurm-6163851.out').rename(root / 'other.out')
    elif mutation == 'wrong_wrapper':
        (root / 'run.sbatch').write_text('wrong')
    elif mutation == 'wrong_root':
        monkeypatch.setattr(audit, 'HISTORICAL_ROOT', tmp_path / 'wrong')
    elif mutation == 'symlink_jobid':
        (root / 'submission.lock/jobid').rename(root / 'jobid')
        (root / 'submission.lock/jobid').symlink_to(root / 'jobid')
    check = audit.Check()
    audit.hash_batch_artifacts(root, 'batch_artifacts:pilot', check, receipt)
    assert (check.results['batch_artifacts:pilot:logs']['status'] == 'PASS') == (mutation == 'none')


def snapshot(path):
    s = path.stat()
    value = {'kind': 'directory' if path.is_dir() else 'file', 'uid': s.st_uid, 'gid': s.st_gid}
    if path.is_dir():
        value['children'] = {p.name: snapshot(p) for p in sorted(path.iterdir())}
    else:
        value['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def manifest_fixture(tmp_path, monkeypatch):
    tree, sacct, root = pilot(tmp_path, monkeypatch)
    (root / 'tmp').mkdir()
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command': 'squeue --noheader --format=%i|%T|%j', 'stdout': '', 'returncode': 0}))
    entries = []
    for name in ('tmp', 'submission.lock'):
        entries.append({'path': str(root / name), 'jobids': ['6163851'],
            'wrapper': str(root / 'run.sbatch'), 'wrapper_sha256': audit.launcher.sha256(root / 'run.sbatch'),
            'snapshot': snapshot(root / name)})
    data = {'schema': 'est-c1-reconciliation-v1', 'blastx_root': str(tree),
            'accounting': {'path': str(sacct), 'sha256': audit.launcher.sha256(sacct)},
            'queue': {'path': str(queue), 'sha256': audit.launcher.sha256(queue)}, 'entries': entries}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    return tree, sacct, root, path, data


@pytest.mark.parametrize('mutation', ['none', 'new_file', 'symlink', 'bad_done', 'failed',
    'changed_owner', 'wrong_kind', 'wrong_job', 'wrong_wrapper', 'wildcard', 'duplicate',
    'active_job', 'active_other_est', 'stale_queue', 'failed_accounting', 'approval_only',
    'bad_version', 'bad_digest', 'inside_input', 'changed_child', 'missing_manifest',
    'active_extra_step', 'parent_traversal', 'capture_symlink', 'malformed_queue'])
def test_manifest_is_exact_evidence_not_approval(tmp_path, monkeypatch, mutation):
    tree, sacct, root, path, data = manifest_fixture(tmp_path, monkeypatch)
    if mutation == 'new_file':
        (root / 'tmp/new').write_text('new')
    elif mutation == 'symlink':
        (root / 'tmp/alias').symlink_to(sacct)
    elif mutation in ('bad_done', 'failed'):
        (root / 'tmp' / ('DONE' if mutation == 'bad_done' else 'FAILED.json')).write_text('bad')
    elif mutation == 'changed_owner':
        data['entries'][0]['snapshot']['uid'] += 1
    elif mutation == 'wrong_kind':
        data['entries'][0]['snapshot']['kind'] = 'file'
    elif mutation == 'wrong_job':
        data['entries'][0]['jobids'] = ['999']
    elif mutation == 'wrong_wrapper':
        data['entries'][0]['wrapper_sha256'] = '0' * 64
    elif mutation == 'wildcard':
        data['entries'][0]['path'] = str(root / '*')
    elif mutation == 'duplicate':
        data['entries'].append(data['entries'][0])
    elif mutation in ('active_job', 'active_other_est', 'stale_queue'):
        queue = Path(data['queue']['path'])
        q = json.loads(queue.read_text())
        if mutation == 'stale_queue':
            q['captured_at'] = '2020-01-01T00:00:00+00:00'
        else:
            q['stdout'] = '6163851|RUNNING|payload\n' if mutation == 'active_job' else '999|PENDING|b6_blastx_remaining8\n'
        queue.write_text(json.dumps(q))
        data['queue']['sha256'] = audit.launcher.sha256(queue)
    elif mutation == 'failed_accounting':
        sacct.write_text(sacct.read_text().replace('6163851.batch|COMPLETED', '6163851.batch|RUNNING'))
        data['accounting']['sha256'] = audit.launcher.sha256(sacct)
    elif mutation == 'active_extra_step':
        sacct.write_text(sacct.read_text() + '\n6163851.0|RUNNING|0:0\n')
        data['accounting']['sha256'] = audit.launcher.sha256(sacct)
    elif mutation == 'parent_traversal':
        (tmp_path / 'alias').mkdir()
        path = tmp_path / 'alias/../manifest.json'
    elif mutation == 'capture_symlink':
        link = tmp_path / 'queue-link'
        link.symlink_to(data['queue']['path'])
        data['queue']['path'] = str(link)
    elif mutation == 'malformed_queue':
        queue = Path(data['queue']['path'])
        queue.write_text('[]')
        data['queue']['sha256'] = audit.launcher.sha256(queue)
    elif mutation == 'approval_only':
        data = {'approved': True}
    elif mutation == 'bad_version':
        data['schema'] = 'v999'
    elif mutation == 'inside_input':
        path = root / 'manifest.json'
    elif mutation == 'changed_child':
        (root / 'submission.lock/jobid').write_text('6163852\n')
    path.write_text(json.dumps(data))
    digest = '0' * 64 if mutation == 'bad_digest' else audit.launcher.sha256(path)
    if mutation == 'missing_manifest':
        path = None
    check = audit.Check()
    reconciled = audit.load_reconciliation(path, digest, tree, sacct.read_text(), check)
    assert bool(reconciled) == (mutation == 'none')
    assert bool(check.failed()) == (mutation != 'none')
    if mutation == 'none':
        scan = audit.Check()
        states = audit.scan_dispositions(root, set(audit.SOURCES), scan, reconciled)
        assert not scan.failed()
        assert states['tmp']['disposition'] == 'NEITHER'
        assert states['tmp']['reconciled'] is True
        assert not (root / 'tmp/DONE').exists()


@pytest.mark.parametrize('species,job', [('Gmax', '6164049_2'), ('Sbicolor', '6164049_6'), ('Vvinifera', '6164049_7')])
def test_array_scratch_binds_actual_log_recorded_task(tmp_path, monkeypatch, species, job):
    tree, sacct, _, path, data = manifest_fixture(tmp_path, monkeypatch)
    batch = tree / audit.REMAINING8_DIR
    scratch = batch / species / 'tmp'
    scratch.mkdir()
    data['entries'].append({'path': str(scratch), 'jobids': [job], 'wrapper': str(batch / 'run.sbatch'),
                           'wrapper_sha256': audit.launcher.sha256(batch / 'run.sbatch'), 'snapshot': snapshot(scratch)})
    path.write_text(json.dumps(data))
    check = audit.Check()
    assert audit.load_reconciliation(path, audit.launcher.sha256(path), tree, sacct.read_text(), check)
    assert not check.failed()


def test_late_sentinel_mutation_fails_rollup(tmp_path, monkeypatch):
    tree, sacct, root, path, _ = manifest_fixture(tmp_path, monkeypatch)
    original = audit.audit_unit
    def mutate_after_scan(*args):
        (root / 'submission.lock/jobid').write_text('late modification\n')
        return original(*args)
    monkeypatch.setattr(audit, 'audit_unit', mutate_after_scan)
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, tmp_path / 'out', sacct, RECEIPT, path, audit.launcher.sha256(path))
    result = json.loads((tmp_path / 'out/rollup.json').read_text())
    assert result['units_audited'] == 18


def preflight_fixture(tmp_path, monkeypatch):
    tree, sacct, _, path, data = manifest_fixture(tmp_path, monkeypatch)
    pre = tree / 'preflight_20260914'
    for name in ('logs', 'receipts', 'submission.lock', 'tmp', 'xdg-cache', 'orthodb', 'swissprot'):
        (pre / name).mkdir(parents=True)
    for name in ('accounting-bounded-check.txt', 'duplicate-check-correct-user.txt', 'duplicate-check.txt',
                 'expected-sources.sha256', 'independent_lineage_audit.json', 'lineage_ncbi.baseline.tsv',
                 'reviewed-small-files.sha256', 'source-hash-attestation.txt', 'execution.lock'):
        (pre / name).write_text('')
    (pre / 'jobid').write_text('6163397\n')
    (pre / 'batch.exitcode').write_text('0\n')
    (pre / 'sbatch-receipt.json').write_text(json.dumps({'returncode': 0, 'stdout': '6163397\n', 'stderr': ''}))
    (pre / 'prepare.sbatch').write_text('synthetic preflight wrapper')
    pin = audit.launcher.sha256(pre / 'prepare.sbatch')
    monkeypatch.setattr(audit, 'PREFLIGHT_WRAPPER_SHA256', pin)
    lineage_pins = {}
    for kind in ('orthodb', 'swissprot'):
        (pre / kind / 'DONE').write_text(audit.DONE_TEXT)
        (pre / kind / 'lineage.tsv').write_text('taxid\tlineage\n1\t1\n')
        lineage_pins[kind] = audit.launcher.sha256(pre / kind / 'lineage.tsv')
        (pre / kind / 'provenance.json').write_text(json.dumps({'lineage_sha256': lineage_pins[kind]}))
    monkeypatch.setattr(audit, 'LINEAGE_SHA256', lineage_pins)
    for name in ('DB.sha256', 'PROVENANCE.txt', 'SOURCE_COMMIT.txt', 'build_dbs.sh',
                 'swissprot_viridiplantae_reviewed.fa.gz.sha256'):
        (pre / 'receipts' / name).write_text('synthetic historical receipt')
    for ext in ('out', 'err'):
        (pre / 'logs' / f'slurm-6163397.{ext}').write_text('')
    sacct.write_text(sacct.read_text() + '\n6163397|COMPLETED|0:0\n6163397.batch|COMPLETED|0:0\n6163397.extern|OUT_OF_MEMORY|0:125\n')
    data['accounting']['sha256'] = audit.launcher.sha256(sacct)
    for name in ('', 'logs', 'receipts', 'submission.lock', 'tmp', 'xdg-cache', 'execution.lock'):
        data['entries'].append({'path': str(pre / name), 'jobids': ['6163397'],
            'wrapper': str(pre / 'prepare.sbatch'), 'wrapper_sha256': pin, 'snapshot': snapshot(pre / name)})
    return tree, sacct, pre, path, data


@pytest.mark.parametrize('mutation', ['none', 'extra_receipt', 'changed_receipt', 'missing_receipt',
    'lineage_changed', 'bad_marker', 'missing_done', 'missing_root_entry', 'missing_lock_entry',
    'directory_lock', 'nonempty_lock', 'wrong_jobid', 'wrong_receipt', 'missing_accounting',
    'snapshot_blesses_extra_receipt', 'snapshot_blesses_missing_receipt'])
def test_preflight_requires_exact_small_tree_and_lineage_bindings(tmp_path, monkeypatch, mutation):
    tree, sacct, pre, path, data = preflight_fixture(tmp_path, monkeypatch)
    if mutation in ('extra_receipt', 'snapshot_blesses_extra_receipt'):
        (pre / 'receipts/new').write_text('unexpected')
    elif mutation == 'changed_receipt':
        (pre / 'receipts/DB.sha256').write_text('changed')
    elif mutation in ('missing_receipt', 'snapshot_blesses_missing_receipt'):
        (pre / 'receipts/DB.sha256').unlink()
    elif mutation == 'lineage_changed':
        (pre / 'orthodb/lineage.tsv').write_text('changed')
    elif mutation == 'bad_marker':
        (pre / 'orthodb/FAILED.json').write_text('{}')
    elif mutation == 'missing_done':
        (pre / 'orthodb/DONE').unlink()
    elif mutation in ('missing_root_entry', 'missing_lock_entry'):
        name = pre if mutation == 'missing_root_entry' else pre / 'execution.lock'
        data['entries'] = [e for e in data['entries'] if e['path'] != str(name)]
    elif mutation == 'directory_lock':
        (pre / 'execution.lock').unlink()
        (pre / 'execution.lock').mkdir()
    elif mutation == 'nonempty_lock':
        (pre / 'execution.lock').write_text('locked')
    elif mutation == 'wrong_jobid':
        (pre / 'jobid').write_text('999\n')
    elif mutation == 'wrong_receipt':
        (pre / 'sbatch-receipt.json').write_text('{"returncode": 1}')
    elif mutation == 'missing_accounting':
        sacct.write_text(sacct.read_text().replace('6163397.batch|COMPLETED|0:0\n', ''))
        data['accounting']['sha256'] = audit.launcher.sha256(sacct)
    # Adversarial self-consistent manifest cannot widen known historical layout.
    if mutation.startswith('snapshot_blesses') or mutation in ('directory_lock', 'nonempty_lock'):
        for entry in data['entries']:
            entry['snapshot'] = snapshot(Path(entry['path']))
    path.write_text(json.dumps(data))
    check = audit.Check()
    reconciled = audit.load_reconciliation(path, audit.launcher.sha256(path), tree, sacct.read_text(), check)
    assert bool(reconciled) == (mutation == 'none')
    assert bool(check.failed()) == (mutation != 'none')
    if mutation == 'none':
        scan = audit.Check()
        audit.scan_dispositions(tree, {audit.PILOT_DIR, audit.REMAINING8_DIR}, scan, reconciled)
        assert not scan.failed()
        assert not (pre / 'DONE').exists()


@pytest.mark.parametrize('valid', [True, False])
def test_integration_retains_18_units_and_semantic_gates(tmp_path, monkeypatch, valid):
    tree, sacct, root, path, _ = manifest_fixture(tmp_path, monkeypatch)
    digest = audit.launcher.sha256(path)
    if not valid:
        (root / 'tmp/new').write_text('late arrival')
    out = tmp_path / 'out'
    if valid:
        result = audit.audit(tree, out, sacct, RECEIPT, path, digest)
    else:
        with pytest.raises(ValueError, match='audit FAILED'):
            audit.audit(tree, out, sacct, RECEIPT, path, digest)
        result = json.loads((out / 'rollup.json').read_text())
    assert result['units_audited'] == 18 and result['failed_units'] == []
    assert len(list((out / 'units').glob('*.json'))) == 18
    assert result['status'] == ('PASS' if valid else 'FAIL')
    assert not result['clean_accounting'] and not result['accepted_phase_evidence'] and not result['production_authorized']
    assert 'UNRESOLVED' in result['sacct_note']


@pytest.mark.parametrize('malformed', ['csv_field', 'deep_json'])
def test_parser_limits_preserve_18_unit_inventory(tmp_path, monkeypatch, malformed):
    tree, sacct, _, path, data = manifest_fixture(tmp_path, monkeypatch)
    if malformed == 'csv_field':
        capture = tmp_path / 'oversized-accounting.psv'
        capture.write_text('JobID|State|ExitCode\n' + 'x' * (audit.csv.field_size_limit() + 1)
                           + '|COMPLETED|0:0\n')
        data['accounting'] = {'path': str(capture), 'sha256': audit.launcher.sha256(capture)}
        path.write_text(json.dumps(data))
    else:
        path.write_text('[' * 1100 + '0' + ']' * 1100)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, out, sacct, RECEIPT, path, audit.launcher.sha256(path))
    result = json.loads((out / 'rollup.json').read_text())
    assert result['status'] == 'FAIL' and result['units_audited'] == 18
    assert result['failed_units'] == []
    assert result['tree_checks']['historical_reconciliation']['status'] == 'FAIL'
    assert len(list((out / 'units').glob('*.json'))) == 18
    assert (out / 'FAILED.json').exists() and not (out / 'DONE').exists()


@pytest.mark.parametrize('target', ['pilot_wrapper', 'array_wrapper', 'pilot_log', 'array_log'])
@pytest.mark.parametrize('mutation', ['bytes', 'delete', 'symlink'])
def test_late_producer_artifact_changes_preserve_failed_inventory(tmp_path, monkeypatch, target, mutation):
    tree, sacct, root, path, _ = manifest_fixture(tmp_path, monkeypatch)
    batch = tree / audit.REMAINING8_DIR
    artifact = {'pilot_wrapper': root / 'run.sbatch', 'array_wrapper': batch / 'run.sbatch',
                'pilot_log': root / 'slurm-6163851.out',
                'array_log': batch / 'logs/6164049_0.err'}[target]
    original = audit.audit_unit
    mutated = False
    def mutate_after_binding(*args):
        nonlocal mutated
        if not mutated:
            before = artifact.read_bytes()
            if mutation == 'bytes':
                # Equal length where possible: size checks alone cannot detect this.
                artifact.write_bytes(b'x' * len(before) if before else b'changed')
            else:
                artifact.unlink()
                if mutation == 'symlink':
                    replacement = tmp_path / 'replacement'
                    replacement.write_bytes(before)
                    artifact.symlink_to(replacement)
            mutated = True
        return original(*args)
    monkeypatch.setattr(audit, 'audit_unit', mutate_after_binding)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, out, sacct, RECEIPT, path, audit.launcher.sha256(path))
    result = json.loads((out / 'rollup.json').read_text())
    assert result['status'] == 'FAIL' and result['units_audited'] == 18
    assert result['failed_units'] == []
    assert result['tree_checks']['batch_artifact_final:' + str(artifact)]['status'] == 'FAIL'
    assert len(list((out / 'units').glob('*.json'))) == 18
    assert (out / 'FAILED.json').exists() and not (out / 'DONE').exists()


def test_final_checks_do_not_rehash_payload_or_age_out_queue(tmp_path, monkeypatch):
    tree, sacct, _, path, _ = manifest_fixture(tmp_path, monkeypatch)
    original_unit, original_hash = audit.audit_unit, audit.launcher.sha256
    real_datetime = audit.datetime.datetime
    class Later(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz) + audit.datetime.timedelta(hours=2)
    count = 0
    def no_payload_hash(path):
        assert not str(path).endswith(('.raw13.tsv', '.dmnd', '.filtered.fa'))
        return original_hash(path)
    def finish_units(*args):
        nonlocal count
        result = original_unit(*args)
        count += 1
        if count == 18:
            monkeypatch.setattr(audit.datetime, 'datetime', Later)
            monkeypatch.setattr(audit.launcher, 'sha256', no_payload_hash)
        return result
    monkeypatch.setattr(audit, 'audit_unit', finish_units)
    result = audit.audit(tree, tmp_path / 'out', sacct, RECEIPT, path, original_hash(path))
    assert result['status'] == 'PASS' and result['units_audited'] == 18
    final = [c for name, c in result['tree_checks'].items() if name.startswith('batch_artifact_final:')]
    assert len(final) == 20 and all(c['status'] == 'PASS' for c in final)


def test_reconciliation_programmer_error_still_propagates(tmp_path, monkeypatch):
    tree, sacct, _, path, _ = manifest_fixture(tmp_path, monkeypatch)
    def broken(*args):
        raise TypeError('programmer defect')
    monkeypatch.setattr(audit, 'completed_jobs', broken)
    with pytest.raises(TypeError, match='programmer defect'):
        audit.load_reconciliation(path, audit.launcher.sha256(path), tree, sacct.read_text(), audit.Check())
