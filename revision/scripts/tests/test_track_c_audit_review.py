"""Round-one independent repros; synthetic anchors never authorize real data."""
import json
from pathlib import Path

import pytest

from test_track_c_recovery import audit, make_unit, make_audit_tree, synthetic_identity


@pytest.mark.parametrize('container', ['pilot', 'batch', 'species'])
@pytest.mark.parametrize('marker', ['FAILED.json', 'contradictory', 'invalid_DONE'])
def test_known_container_markers_cannot_hide_behind_children(tmp_path, container, marker):
    tree, sacct = make_audit_tree(tmp_path)
    roots = {'pilot': tree / audit.PILOT_DIR, 'batch': tree / audit.REMAINING8_DIR,
             'species': tree / audit.REMAINING8_DIR / 'Athaliana'}
    root = roots[container]
    if marker in ('FAILED.json', 'contradictory'):
        (root / 'FAILED.json').write_text('{}')
    if marker == 'contradictory':
        (root / 'DONE').write_text(audit.DONE_TEXT)
    if marker == 'invalid_DONE':
        (root / 'DONE').write_text('not a completion receipt')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, out, sacct)
    rollup = json.loads((out / 'rollup.json').read_text())
    assert rollup['units_audited'] == 18 and not rollup['failed_units']
    assert rollup['tree_checks'][f'container_markers:{root}']['status'] == 'FAIL'
    assert not (out / 'DONE').exists()


@pytest.mark.parametrize('field', ['diamond_version', 'launcher_sha256',
                                  'taxonomy_helper_sha256', 'diamond_sha256'])
@pytest.mark.parametrize('mutation', ['absent', 'wrong'])
def test_producer_identity_is_required_and_historically_bound(tmp_path, field, mutation):
    unit = make_unit(tmp_path)
    path = unit / 'provenance.json'
    data = json.loads(path.read_text())
    if mutation == 'absent':
        del data[field]
    else:
        data[field] = 'NOT-DIAMOND' if field == 'diamond_version' else '0' * 64
    path.write_text(json.dumps(data))
    result = audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
    assert result['status'] == 'FAIL'
    assert 'provenance.json' in result['files']


def test_self_hashed_arbitrary_binary_does_not_establish_approved_identity(tmp_path, monkeypatch):
    # Undo ONLY the fixture engine trust anchor; the real approved pin must reject it.
    monkeypatch.setattr(audit, 'APPROVED_DIAMOND_SHA256',
                        '0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d', raising=False)
    make_unit(tmp_path)
    result = audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
    assert result['status'] == 'FAIL'


@pytest.mark.parametrize('missing', ['swissprot', 'orthodb'])
def test_both_named_source_timing_files_required(tmp_path, missing):
    (tmp_path / 'batch.exitcode').write_text('0')
    for kind in audit.SOURCES:
        if kind != missing:
            (tmp_path / f'{kind}.time.txt').write_text('elapsed')
    (tmp_path / 'arbitrary.time.txt').write_text('not a substitute')
    check = audit.Check()
    audit.audit_species_root('Sitalica', tmp_path, 'Sitalica', check)
    assert check.failed()


@pytest.mark.parametrize('mutation', ['missing_log', 'renamed_log', 'wrong_script'])
def test_remaining8_requires_script_identity_and_every_named_log(tmp_path, mutation):
    tree, _ = make_audit_tree(tmp_path)
    batch = tree / audit.REMAINING8_DIR
    if mutation == 'wrong_script':
        (batch / 'run.sbatch').write_text('unapproved')
    elif mutation == 'missing_log':
        (batch / 'logs' / '6164049_3.err').unlink()
    else:
        (batch / 'logs' / '6164049_3.out').rename(batch / 'logs' / 'other.out')
    check = audit.Check()
    audit.hash_batch_artifacts(batch, 'batch_artifacts:remaining8', check)
    assert check.failed()


def test_timing_directory_is_failed_evidence_not_an_inventory_abort(tmp_path):
    (tmp_path / 'batch.exitcode').write_text('0')
    (tmp_path / 'swissprot.time.txt').mkdir()
    (tmp_path / 'orthodb.time.txt').write_text('elapsed')
    check = audit.Check()
    audit.audit_species_root('Sitalica', tmp_path, 'Sitalica', check)
    assert check.failed()


def test_remaining8_empty_stderr_is_inventory_not_missing(tmp_path):
    tree, _ = make_audit_tree(tmp_path)
    check = audit.Check()
    hashes = audit.hash_batch_artifacts(tree / audit.REMAINING8_DIR, 'batch_artifacts:remaining8', check)
    assert not check.failed()
    assert 'logs/6164049_0.err' in hashes


def test_arbitrary_pilot_log_cannot_resolve_missing_layout_receipt(tmp_path):
    (tmp_path / 'run.sbatch').write_text('# fixture; never submitted\n')
    (tmp_path / 'slurm-6163851.out').write_text('guessed name')
    check = audit.Check()
    audit.hash_batch_artifacts(tmp_path, 'batch_artifacts:pilot', check)
    assert check.failed()
    assert 'UNRESOLVED' in str(check.results)


def test_nested_unit_payload_is_not_blessed_by_nested_done(tmp_path):
    unit = make_unit(tmp_path)
    nested = unit / 'unexpected'
    nested.mkdir()
    (nested / 'DONE').write_text(audit.DONE_TEXT)
    (nested / 'payload').write_text('not inventoried')
    result = audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
    assert result['status'] == 'FAIL'
    assert 'provenance.json' in result['files']


@pytest.mark.parametrize('malformed', [[], None, 'text', {'inputs': []}, {'inputs': None},
                                     {'inputs': {'query': []}}])
def test_json_shape_errors_preserve_hashes_and_continue_17_units(tmp_path, malformed):
    tree, sacct = make_audit_tree(tmp_path)
    prov = tree / audit.PILOT_DIR / 'orthodb' / 'provenance.json'
    prov.write_text(json.dumps(malformed))
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, out, sacct)
    result = json.loads((out / 'rollup.json').read_text())
    assert result['units_audited'] == 18
    assert result['failed_units'] == ['Sitalica/orthodb']
    failed = json.loads((out / 'units' / 'Sitalica_orthodb.json').read_text())
    assert failed['files']['provenance.json'] == audit.launcher.sha256(prov)
    assert failed['checks']['done_gate']['status'] == 'PASS'
    assert len(list((out / 'units').glob('*.json'))) == 18


@pytest.mark.parametrize('payload', [{}, None, ['bad'], [{'species': []}],
                                   [{'species': 'Athaliana', 'query': []}]])
def test_batch_inputs_schema_errors_are_failed_units_not_abort(tmp_path, payload):
    tree, sacct = make_audit_tree(tmp_path)
    batch = tree / audit.REMAINING8_DIR
    (batch / 'inputs.json').write_text(json.dumps(payload))
    (batch / 'inputs.sha256').write_text(audit.launcher.sha256(batch / 'inputs.json') + '\n')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, out, sacct)
    result = json.loads((out / 'rollup.json').read_text())
    assert result['units_audited'] == 18 and len(result['failed_units']) == 16
    assert len(list((out / 'units').glob('*.json'))) == 18


@pytest.mark.parametrize('marker', ['absent', 'valid_DONE', 'directory_DONE', 'dangling_DONE'])
def test_container_marker_presence_contract(tmp_path, marker):
    if marker == 'valid_DONE':
        (tmp_path / 'DONE').write_text(audit.DONE_TEXT)
    elif marker == 'directory_DONE':
        (tmp_path / 'DONE').mkdir()
    elif marker == 'dangling_DONE':
        (tmp_path / 'DONE').symlink_to(tmp_path / 'absent')
    check = audit.Check()
    audit.audit_container_markers(tmp_path, check)
    assert bool(check.failed()) == (marker in ('directory_DONE', 'dangling_DONE'))


@pytest.mark.parametrize('key,value', [('counts_by_taxid', []), ('counts_by_taxid', {'1': []}),
    ('retained_taxids', {}), ('fields', None), ('queries', True), ('inputs', {'extra': 'x'})])
def test_malformed_nested_producer_schema_preserves_prior_checks(tmp_path, key, value):
    unit = make_unit(tmp_path)
    prov = unit / 'provenance.json'
    data = json.loads(prov.read_text())
    data[key] = value
    prov.write_text(json.dumps(data))
    result = audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
    assert result['status'] == 'FAIL'
    assert result['files']['provenance.json'] == audit.launcher.sha256(prov)
    assert result['checks']['required_files']['status'] == 'PASS'


def test_unknown_approved_engine_identity_is_explicitly_unresolved(tmp_path, monkeypatch):
    make_unit(tmp_path)
    monkeypatch.setattr(audit, 'APPROVED_DIAMOND_SHA256', None)
    result = audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
    assert result['status'] == 'FAIL'
    assert 'UNRESOLVED' in str(result['checks']['historical_producer_identity'])


def test_missing_code_fields_fail_in_integrated_inventory(tmp_path):
    tree, sacct = make_audit_tree(tmp_path)
    prov = tree / audit.REMAINING8_DIR / 'Gmax' / 'swissprot' / 'provenance.json'
    data = json.loads(prov.read_text())
    del data['launcher_sha256']
    prov.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, tmp_path / 'out', sacct)
    rollup = json.loads((tmp_path / 'out' / 'rollup.json').read_text())
    assert rollup['units_audited'] == 18
    assert rollup['failed_units'] == ['Gmax/swissprot']


@pytest.mark.parametrize('contents', ['not FASTA\n', '', '>valid\nM\n>empty\n'],
                         ids=['malformed', 'empty', 'valid-prefix-then-malformed'])
def test_filtered_fasta_iteration_failure_preserves_all_unit_inventory(tmp_path, contents):
    tree, sacct = make_audit_tree(tmp_path)
    damaged = tree / audit.PILOT_DIR / 'orthodb' / 'orthodb.filtered.fa'
    damaged.write_text(contents)
    if contents.startswith('>valid'):
        records = audit.launcher.records(damaged)
        assert next(records) == ('valid', 'M')
        with pytest.raises(ValueError, match='malformed FASTA record'):
            next(records)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, out, sacct)
    rollup = json.loads((out / 'rollup.json').read_text())
    assert rollup['units_audited'] == 18
    assert rollup['failed_units'] == ['Sitalica/orthodb']
    manifests = [json.loads(p.read_text()) for p in (out / 'units').glob('*.json')]
    assert len(manifests) == 18
    assert sum(m['status'] == 'PASS' for m in manifests) == 17
    failed = next(m for m in manifests if m['unit'] == 'Sitalica/orthodb')
    assert failed['files']['orthodb.filtered.fa'] == audit.launcher.sha256(damaged)
    assert failed['checks']['done_gate']['status'] == 'PASS'
    assert failed['checks']['provenance_hashes']['status'] == 'FAIL'
    assert 'FASTA' in failed['checks']['malformed_or_unreadable']['detail']
    assert (out / 'FAILED.json').is_file() and not (out / 'DONE').exists()


def test_programmer_error_not_silently_treated_as_bad_input(tmp_path, monkeypatch):
    make_unit(tmp_path)
    def broken(*args, **kwargs):
        raise TypeError('programmer defect')
    monkeypatch.setattr(audit.launcher, 'production_inputs', broken)
    with pytest.raises(TypeError, match='programmer defect'):
        audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
