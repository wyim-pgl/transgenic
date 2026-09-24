"""CPU-only recovery regressions. No DIAMOND or remote jobs are invoked."""
import importlib.util
import json
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1] / 'evidence'


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('est_blastx_audit')
accept = load('est_phase_accept')
pin = load('est_phase_accept_fixture')


@pytest.fixture(autouse=True)
def synthetic_identity(monkeypatch):
    # Fixture-only trust anchors. No production override or executable invocation.
    monkeypatch.setattr(audit, 'APPROVED_DIAMOND_SHA256', hashlib.sha256(
        b'fixture-engine-not-executed').hexdigest(), raising=False)
    monkeypatch.setattr(audit, 'APPROVED_WRAPPER_SHA256', dict.fromkeys(
        ('pilot', 'remaining8'), hashlib.sha256(b'# fixture; never submitted\n').hexdigest()),
        raising=False)


@pytest.mark.parametrize('entrypoint', ['stream', 'run'])
def test_acceptance_blocked_before_io(tmp_path, entrypoint):
    raw = tmp_path / 'raw.tsv'
    raw.write_text('q\t90\tp\t30\t1\t90\t1\t30\t30\t1e-20\t100\t1\t30\n')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='UNRESOLVED.*A46'):
        if entrypoint == 'stream':
            accept.accept_stream(raw, 'orthodb', out)
        else:
            accept.run(SimpleNamespace(raw13=raw, raw13_sha256=audit.launcher.sha256(raw),
                                      source_kind='orthodb', species='Athaliana', out_dir=out))
    assert not out.exists()


@pytest.mark.parametrize('done,failed,ok', [(False, False, False), (True, False, True),
                                         (False, True, False), (True, True, False)])
def test_marker_truth_table(tmp_path, done, failed, ok):
    if done:
        (tmp_path / 'DONE').write_text(audit.DONE_TEXT)
    if failed:
        (tmp_path / 'FAILED.json').write_text('{}')
    assert audit.disposition(tmp_path)['ok'] == ok


def test_recursive_preflight_tmp_not_hidden_by_done(tmp_path):
    pre = tmp_path / 'preflight_20260914'
    (pre / 'tmp').mkdir(parents=True)
    (pre / 'DONE').write_text(audit.DONE_TEXT)
    (pre / 'tmp' / 'orphan').write_text('unfinished')
    check = audit.Check()
    entries = audit.scan_dispositions(tmp_path, set(), check)
    assert check.failed()
    assert 'preflight_20260914/tmp' in entries


def test_symlink_is_not_a_completed_directory(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'DONE').write_text(audit.DONE_TEXT)
    (tmp_path / 'alias').symlink_to(real, target_is_directory=True)
    check = audit.Check()
    audit.scan_dispositions(tmp_path, {'real'}, check)
    assert check.failed()


def test_accounting_requires_all_jobs_and_steps():
    lines = ['JobID|State|ExitCode']
    for job in ['6163851'] + [f'6164049_{i}' for i in range(8)]:
        lines.extend([f'{job}|COMPLETED|0:0', f'{job}.batch|COMPLETED|0:0',
                      f'{job}.extern|OUT_OF_MEMORY|0:125'])
    capture = '\n'.join(lines)
    good = audit.Check()
    audit.audit_accounting(capture, good)
    assert not good.failed()
    for bad_capture in [capture.replace('6163851.batch|COMPLETED', '6163851.batch|FAILED'),
                        '\n'.join(lines[:-1]), '', capture + '\n' + lines[1]]:
        bad = audit.Check()
        audit.audit_accounting(bad_capture, bad)
        assert bad.failed()


def test_missing_timing_is_failure(tmp_path):
    (tmp_path / 'batch.exitcode').write_text('0')
    check = audit.Check()
    audit.audit_species_root('Sitalica', tmp_path, 'Sitalica', check)
    assert check.failed()


@pytest.mark.parametrize('layout', ['pilot', 'remaining8'])
def test_query_layouts_and_unresolvable_query(tmp_path, layout):
    query = tmp_path / 'query.fa'
    query.write_text('>q\n' + 'A' * 100 + '\n')
    digest = audit.launcher.sha256(query)
    provenance = {'inputs': {'query': str(query), 'query_sha256': digest}}
    (tmp_path / 'query.sha256').write_text(digest + '  ' + str(query) + '\n')
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps([{'species': 'Sitalica', 'query': str(query),
                                  'query_md5': audit.md5(query)}]))
    (tmp_path / 'inputs.sha256').write_text(audit.launcher.sha256(inputs) + '  inputs.json\n')
    check = audit.Check()
    resolved = audit.resolve_query(layout, 'Sitalica', tmp_path, tmp_path, provenance, check)
    assert resolved == query and not check.failed()
    raw = tmp_path / 'raw'
    raw.write_text('q\tignored\n')
    audit.audit_query_file(resolved, provenance, None, raw, check)
    assert not check.failed()
    query.unlink()
    missing = audit.Check()
    audit.audit_query_file(resolved, provenance, None, raw, missing)
    assert missing.failed()


def test_audit_rejects_output_inside_input_without_mutation(tmp_path):
    sacct = tmp_path / 'sacct'
    sacct.write_text('capture')
    out = tmp_path / 'audit'
    with pytest.raises(ValueError, match='outside'):
        audit.audit(tmp_path, out, sacct)
    assert not out.exists()


def test_audit_rejects_scratch_inside_input(tmp_path, monkeypatch):
    tree = tmp_path / 'blastx'
    tree.mkdir()
    monkeypatch.setattr(audit.tempfile, 'tempdir', str(tree))
    with pytest.raises(ValueError, match='scratch'):
        audit.audit(tree, tmp_path / 'out', tmp_path / 'unused-sacct')
    assert not (tmp_path / 'out').exists()
    assert list(tree.iterdir()) == []


def test_fixture_failure_does_not_leave_outer_done(tmp_path):
    out = tmp_path / 'fixture'
    with pytest.raises((ValueError, OSError)):
        pin.run(tmp_path / 'missing-diamond', out, tmp_path, 1)
    assert (out / 'FAILED.json').exists()
    assert not (out / 'DONE').exists()


def make_unit(root, species='Sitalica', kind='orthodb', inputs_root=None):
    inputs_root = inputs_root or root
    inputs_root.mkdir(parents=True, exist_ok=True)
    taxid = 3702 if species == 'Osativa' else 4530
    subject = f'{taxid}_0:cross' if kind == 'orthodb' else 'sp|CROSS.1|CROSS'
    unit = root / kind
    unit.mkdir(parents=True)
    query = inputs_root / 'est' / species / 'univec' / 'est.univec.fa.gz'
    query.parent.mkdir(parents=True, exist_ok=True)
    if not query.exists():
        with gzip.open(query, 'wt') as f:
            f.write('>q\n' + 'A' * 120 + '\n')
    (query.parent / 'DONE').write_text('screened')
    (query.parent / 'summary.json').write_text('{}')
    source = inputs_root / audit.launcher.SOURCE_NAMES[kind]
    with gzip.open(source, 'wt') if source.suffix == '.gz' else open(source, 'w') as f:
        f.write(f'>{subject} OX={taxid}\n' + 'M' * 40 + '\n')
    lineage = inputs_root / 'lineage.tsv'
    lineage.write_text(f'taxid\tlineage\n{taxid}\t{taxid};1\n')
    roles = inputs_root / 'roles.tsv'
    roles.write_text('dataset\tspecies\trole\nest/' + species + '\t' + species + '\tc2_training_eligible\n')
    inputs = dict(species=species, source_kind=kind)
    for name, path in [('source', source), ('query', query), ('lineage', lineage), ('roles', roles)]:
        inputs[name] = str(path)
        inputs[name + '_sha256'] = audit.launcher.sha256(path)
    qc = audit.launcher.production_inputs(**inputs)
    filtered = unit / (kind + '.filtered.fa')
    qc.update(audit.launcher.filter_source(source, kind, lineage, audit.launcher.SPECIES[species], filtered))
    raw = unit / (kind + '.raw13.tsv')
    raw.write_text(f'q\t120\t{subject}\t40\t1\t90\t1\t30\t30\t1e-20\t100\t1\t30\n')
    db = unit / (kind + '.dmnd')
    db.write_bytes(b'fixture-db-not-a-real-engine')
    diamond = inputs_root / 'diamond'
    diamond.write_bytes(b'fixture-engine-not-executed')
    commands = [[str(diamond), 'makedb', '--in', str(filtered), '--db', str(db), '--threads', '1', '--no-parse-seqids'],
                audit.launcher.search_command(diamond, query, db, raw, 1, root / 'scratch')]
    (unit / 'commands.json').write_text(json.dumps(commands))
    for name in ['blastx.log', 'makedb.log']:
        (unit / name).write_text('fixture log')
    (unit / 'DONE').write_text(audit.DONE_TEXT)
    qc.update(audit.launcher.validate_raw(raw), inputs=inputs, elapsed_seconds=1.0,
              fields=audit.launcher.RAW13, raw_sha256=audit.launcher.sha256(raw),
              db_sha256=audit.launcher.sha256(db), diamond_sha256=audit.launcher.sha256(diamond),
              diamond_version='diamond version 2.1.24\ndiamond v2.1.24.178 (C) fixture',
              launcher_sha256='5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291',
              taxonomy_helper_sha256='32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302')
    (unit / 'provenance.json').write_text(json.dumps(qc))
    (root / 'query.sha256').write_text(inputs['query_sha256'] + '  ' + str(query) + '\n')
    return unit


@pytest.mark.parametrize('mutation', ['none', 'command_query', 'raw_hash', 'raw_count',
    'elapsed', 'input_hash', 'species', 'filter_count', 'malformed_provenance', 'unit_tmp',
    'subject_not_in_filtered'])
def test_unit_reconciliation(tmp_path, mutation):
    unit = make_unit(tmp_path)
    prov = unit / 'provenance.json'
    data = json.loads(prov.read_text())
    if mutation == 'command_query':
        path = unit / 'commands.json'
        commands = json.loads(path.read_text())
        i = commands[1].index('--query')
        commands[1][i + 1] = '/another/query.fa'
        path.write_text(json.dumps(commands))
    elif mutation == 'raw_hash':
        data['raw_sha256'] = '0' * 64
    elif mutation == 'raw_count':
        data['raw_hsps'] = 999
    elif mutation == 'elapsed':
        data['elapsed_seconds'] = -1
    elif mutation == 'input_hash':
        data['inputs']['lineage_sha256'] = '0' * 64
    elif mutation == 'species':
        data['inputs']['species'] = 'Athaliana'
    elif mutation == 'filter_count':
        data['sequences_excluded'] = 999
    elif mutation == 'unit_tmp':
        (unit / 'tmp').mkdir()
    elif mutation == 'subject_not_in_filtered':
        raw = unit / 'orthodb.raw13.tsv'
        raw.write_text(raw.read_text().replace('4530_0:cross', '3702_0:self'))
        data['raw_sha256'] = audit.launcher.sha256(raw)
    prov.write_text('{' if mutation == 'malformed_provenance' else json.dumps(data))
    result = audit.audit_unit('pilot', 'Sitalica', 'orthodb', tmp_path, tmp_path)
    assert result['status'] == ('PASS' if mutation == 'none' else 'FAIL')


@pytest.mark.parametrize('flag,ts,want', [(0, '+', '+'), (0, '-', '-'),
                                        (16, '+', '-'), (16, '-', '+')])
def test_flag_ts_composition(flag, ts, want):
    assert load('est_phase_project').genomic_strand(flag, ts) == want


@pytest.mark.parametrize('phase_a,phase_b,want', [(0, 0, (0, 0, False)),
    (1, 2, (None, None, True)), (None, 0, (None, 0, False))])
def test_route_disagreement_is_neutral_not_phase_zero(phase_a, phase_b, want):
    assert load('est_phase_project').reconcile_phases(phase_a, phase_b) == want


def test_projector_refuses_unresolved_semantics():
    with pytest.raises(ValueError, match='UNRESOLVED.*A46'):
        load('est_phase_project').project()


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('dna,btop,cigar,qg', [
    ('ATGAAACGCT', '2\\-1', '2M1\\1M', 'MK\\A'),
    ('ATGAAGCT', '2/-1', '2M1/1M', 'MK/A')])
def test_rich17_candidate_diagnostics_are_not_approval(tmp_path, reverse, dna, btop, cigar, qg):
    # Hand-built candidate fixtures exercise plumbing, NOT real DIAMOND semantics.
    row = ['q', str(len(dna)), 'p', '3', '1', str(len(dna)), '1', '3',
           '4', '1e-20', '100', '1', btop, cigar, qg, 'MK-A', 'MKA']
    if reverse:
        dna = pin.fixture.reverse_complement(dna)
        row[4], row[5], row[11] = str(len(dna)), '1', '-1'
    raw = tmp_path / 'raw17'
    raw.write_text('\t'.join(row) + '\n')
    report = pin.pin_rows(raw, {'q': dna}, 'MKA')
    assert report['rows'][0]['readings']['B_ungapped_query_aa'] == 3
    assert report['production_authorized'] is False
    assert report['verdict'] != 'PINNED'


@pytest.mark.parametrize('length', [29, 30, 31])
def test_boundary_lengths_remain_raw_not_accepted(tmp_path, length):
    raw = tmp_path / 'raw'
    line = f'q\t93\tp\t31\t1\t{length * 3}\t1\t{length}\t{length}\t1e-20\t100\t1\t{length}\n'
    raw.write_text(line)
    assert accept.parse_row(1, line)['length'] == length
    with pytest.raises(ValueError, match='UNRESOLVED'):
        accept.accept_stream(raw, 'orthodb', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def make_audit_tree(tmp_path):
    tree = tmp_path / 'blastx'
    entries = []
    for species in ('Sitalica',) + audit.REMAINING8_SPECIES:
        root = tree / audit.PILOT_DIR if species == 'Sitalica' else tree / audit.REMAINING8_DIR / species
        for kind in audit.SOURCES:
            unit = make_unit(root, species, kind, tmp_path / 'inputs' / species)
        query = json.loads((unit / 'provenance.json').read_text())['inputs']['query']
        entries.append({'species': species, 'query': query, 'query_md5': audit.md5(query)})
        (root / 'batch.exitcode').write_text('0')
        for kind in audit.SOURCES:
            (root / f'{kind}.time.txt').write_text('fixture timing')
    batch = tree / audit.REMAINING8_DIR
    (batch / 'inputs.json').write_text(json.dumps(entries[1:]))
    (batch / 'inputs.sha256').write_text(audit.launcher.sha256(batch / 'inputs.json') + '  inputs.json\n')
    for root in (tree / audit.PILOT_DIR, batch):
        (root / 'run.sbatch').write_text('# fixture; never submitted\n')
    sacct = tmp_path / 'sacct'
    lines = ['JobID|State|ExitCode']
    for job in ['6163851'] + [f'6164049_{i}' for i in range(8)]:
        lines.extend([f'{job}|COMPLETED|0:0', f'{job}.batch|COMPLETED|0:0',
                      f'{job}.extern|OUT_OF_MEMORY|0:125'])
    sacct.write_text('\n'.join(lines))
    (batch / 'logs').mkdir()
    for i in range(8):
        for ext in ('out', 'err'):
            (batch / 'logs' / f'6164049_{i}.{ext}').write_text('' if ext == 'err' else 'fixture log')
    return tree, sacct


def test_full_18_unit_audit_and_preflight_failure(tmp_path):
    tree, sacct = make_audit_tree(tmp_path)
    lines = sacct.read_text().splitlines()
    # The locally unavailable pilot log receipt cannot be fabricated by this fixture.
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, tmp_path / 'out', sacct)
    result = json.loads((tmp_path / 'out' / 'rollup.json').read_text())
    assert result['status'] == 'FAIL' and result['units_audited'] == 18
    assert result['failed_units'] == []
    assert result['clean_accounting'] is False
    assert result['accepted_phase_evidence'] is False
    sacct.write_text('\n'.join(lines[:-1]))
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, tmp_path / 'bad_accounting', sacct)
    bad = json.loads((tmp_path / 'bad_accounting' / 'rollup.json').read_text())
    assert bad.get('units_audited') == 18
    sacct.write_text('\n'.join(lines))
    (tree / 'preflight_20260914').mkdir()
    with pytest.raises(ValueError, match='audit FAILED'):
        audit.audit(tree, tmp_path / 'failed', sacct)
    failed = json.loads((tmp_path / 'failed' / 'rollup.json').read_text())
    assert failed['units_audited'] == 18
    assert (tmp_path / 'failed' / 'FAILED.json').exists()
    assert not (tmp_path / 'failed' / 'DONE').exists()


def test_inconclusive_readings_cannot_pin():
    result = pin.summarize_readings([
        {'col9_matches': ['A', 'B'], 'frameshift_ops': 1},
        {'col9_matches': ['A', 'B'], 'frameshift_ops': 1}])
    assert result['verdict'] == 'INCONCLUSIVE'
    assert result['production_authorized'] is False
