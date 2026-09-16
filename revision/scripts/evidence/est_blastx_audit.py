#!/usr/bin/env python3
"""Fail-closed audit of completed EST blastx output trees (stage C1). stdlib only.

Audits the 9-species x 2-source raw13 units produced by est_blastx.py under
  <blastx-root>/sitalica_pilot_20260914/          (pilot layout)
  <blastx-root>/remaining8_20260915/<species>/    (batch layout)
per unit checks (a)-(f):
  (a) sha256 every unit file; recomputed raw13/dmnd/filtered hashes must equal
      the values recorded inside provenance.json;
  (b) re-run the exact validate_raw() row contract over raw13 and reconcile
      raw_hsps / hsps_lt_30aa with provenance;
  (c) commands.json argv must equal the frozen makedb/search command tokens;
  (d) resolve the query FASTA for BOTH provenance layouts (pilot: query.sha256
      at the species root; remaining8: inputs.json + inputs.sha256 at the batch
      root), re-hash it, and require qseqid set <= query record IDs.  An
      unresolvable query file is a HARD FAIL, never a skip;
  (e) species batch.exitcode must be 0; timing files are hashed and recorded;
  (f) DONE-presence gate: absence of DONE always means incomplete.
Additionally EVERY directory entry under the scanned roots -- including
preflight_20260914/ siblings and tmp/ dirs -- receives an explicit fail-closed
disposition (DONE / FAILED.json / NEITHER); contradictions, NEITHER and any
tmp/ directory require reconciliation and fail, never silently skipped.  run.sbatch and slurm .out logs are hashed into
the manifest.  sacct evidence is embedded VERBATIM from a login-node capture;
the persisting `.extern OUT_OF_MEMORY 0:125` anomaly is recorded as UNRESOLVED
-- this audit never claims clean accounting.

This script performs no Slurm submission and never writes inside the audited
tree.  Production execution requires the separate reviewed sbatch template and
author authorization.
"""
from __future__ import annotations

import argparse
import csv
import io
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path
import re
import signal
import sys

spec = importlib.util.spec_from_file_location('est_blastx', Path(__file__).with_name('est_blastx.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

PILOT_DIR = 'sitalica_pilot_20260914'
REMAINING8_DIR = 'remaining8_20260915'
PILOT_SPECIES = ('Sitalica',)
REMAINING8_SPECIES = ('Athaliana', 'Bdistachyon', 'Gmax', 'Osativa',
                      'Ppatens', 'Ptrichocarpa', 'Sbicolor', 'Vvinifera')
SOURCES = ('swissprot', 'orthodb')
UNIT_FILES = ('blastx.log', 'commands.json', 'DONE', 'makedb.log', 'provenance.json')
DONE_TEXT = 'Outputs and QC complete; see provenance.json\n'
SACCT_NOTE = ('.extern OUT_OF_MEMORY 0:125 across all 9 jobs is a recorded, '
              'UNRESOLVED anomaly; allocation/.batch COMPLETED 0:0 is the only '
              'accounting claim made.  No clean-accounting claim.')


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Check:
    """Accumulates named PASS/FAIL results; fail-closed rollup."""

    def __init__(self):
        self.results = {}

    def add(self, name, ok, detail):
        if name in self.results:
            raise ValueError(f'duplicate check name: {name}')
        self.results[name] = {'status': 'PASS' if ok else 'FAIL', 'detail': detail}

    def failed(self):
        return [k for k, v in self.results.items() if v['status'] != 'PASS']


def parse_sha256_file(path):
    """First token of the first non-empty line must be 64 hex chars."""
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts:
            if not re.fullmatch('[0-9a-f]{64}', parts[0]):
                raise ValueError(f'malformed sha256 record: {path}')
            return parts[0], (parts[1] if len(parts) > 1 else None)
    raise ValueError(f'empty sha256 record: {path}')


def resolve_query(layout, species, species_root, batch_root, provenance, check):
    """Must-fix 4: handle BOTH provenance layouts; unresolvable = HARD FAIL."""
    recorded = provenance.get('inputs', {})
    query = recorded.get('query')
    query_sha = recorded.get('query_sha256')
    if not query or not query_sha:
        check.add('query_resolution', False, 'provenance.json lacks inputs.query/query_sha256')
        return None
    if layout == 'pilot':
        record = species_root / 'query.sha256'
        if not record.is_file():
            check.add('query_resolution', False, f'missing {record}')
            return None
        digest, recorded_path = parse_sha256_file(record)
        ok = digest == query_sha and (recorded_path is None or recorded_path == query)
        check.add('query_resolution', ok,
                  {'layout': 'pilot', 'query.sha256': digest, 'provenance_query_sha256': query_sha,
                   'recorded_path': recorded_path, 'provenance_query': query})
        return Path(query) if ok else None
    # remaining8 layout: inputs.json + inputs.sha256 at the batch root.
    inputs_json = batch_root / 'inputs.json'
    inputs_sha = batch_root / 'inputs.sha256'
    if not inputs_json.is_file() or not inputs_sha.is_file():
        check.add('query_resolution', False, f'missing {inputs_json} or {inputs_sha}')
        return None
    digest, _ = parse_sha256_file(inputs_sha)
    if launcher.sha256(inputs_json) != digest:
        check.add('query_resolution', False, 'inputs.json does not match inputs.sha256')
        return None
    entries = [e for e in json.loads(inputs_json.read_text())
               if isinstance(e, dict) and e.get('species') == species]
    if len(entries) != 1:
        check.add('query_resolution', False, f'{len(entries)} inputs.json entries for {species}')
        return None
    entry = entries[0]
    ok = entry.get('query') == query
    detail = {'layout': 'remaining8', 'inputs_query': entry.get('query'),
              'provenance_query': query, 'inputs_query_md5': entry.get('query_md5')}
    check.add('query_resolution', ok, detail)
    return Path(query) if ok else None


def audit_query_file(query_path, provenance, expected_md5, raw13, check):
    """Re-hash the query and require qseqid set <= query record IDs."""
    if query_path is None or not query_path.is_file():
        check.add('query_file', False, f'query FASTA unresolvable or absent: {query_path}')
        return
    fresh = launcher.sha256(query_path)
    ok = fresh == provenance['inputs']['query_sha256']
    detail = {'fresh_sha256': fresh, 'recorded_sha256': provenance['inputs']['query_sha256']}
    if expected_md5 is not None:
        fresh_md5 = md5(query_path)
        ok = ok and fresh_md5 == expected_md5
        detail.update(fresh_md5=fresh_md5, recorded_md5=expected_md5)
    check.add('query_file', ok, detail)
    if not ok:
        return
    qseqids = set()
    with open(raw13) as f:
        for line in f:
            qseqids.add(line.split('\t', 1)[0])
    query_ids = {header.split()[0] for header, _ in launcher.records(query_path)}
    orphans = sorted(qseqids - query_ids)
    check.add('qseqid_subset', not orphans,
              {'distinct_qseqids': len(qseqids), 'query_records': len(query_ids),
               'orphan_qseqids': orphans[:10]})


def expected_commands(recorded):
    """Rebuild the frozen argv from the variable slots of the recorded argv."""
    if not (isinstance(recorded, list) and len(recorded) == 2
            and all(isinstance(c, list) and all(isinstance(t, str) for t in c) for c in recorded)):
        raise ValueError('commands.json must be [makedb_argv, search_argv]')
    makedb, search = recorded

    def slot(argv, flag):
        return argv[argv.index(flag) + 1]

    exp_makedb = [makedb[0], 'makedb', '--in', slot(makedb, '--in'), '--db', slot(makedb, '--db'),
                  '--threads', slot(makedb, '--threads'), '--no-parse-seqids']
    exp_search = launcher.search_command(
        search[0], slot(search, '--query'), slot(search, '--db'), slot(search, '--out'),
        int(slot(search, '--threads')), slot(search, '--tmpdir'))
    return exp_makedb, exp_search


def audit_unit(layout, species, kind, species_root, batch_root):
    try:
        return _audit_unit(layout, species, kind, species_root, batch_root)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {'unit': f'{species}/{kind}', 'layout': layout, 'status': 'FAIL',
                'checks': {'malformed_or_unreadable': {'status': 'FAIL', 'detail': str(exc)}},
                'files': {}}


def _audit_unit(layout, species, kind, species_root, batch_root):
    unit = species_root / kind if layout == 'remaining8' else species_root / kind
    check = Check()
    files = {}
    # (f) DONE gate first: absence of DONE always means incomplete.
    done = unit / 'DONE'
    if not unit.is_dir():
        check.add('unit_dir', False, f'missing unit directory: {unit}')
        return {'unit': f'{species}/{kind}', 'checks': check.results, 'files': files,
                'status': 'FAIL'}
    scan_dispositions(unit, set(), check)
    check.add('done_gate', disposition(unit)['ok'],
              {'present': done.is_file()})
    if (unit / 'FAILED.json').exists():
        check.add('failed_marker', False, 'FAILED.json present in unit')
    provenance = None
    prov_path = unit / 'provenance.json'
    expected = set(UNIT_FILES) | {f'{kind}.dmnd', f'{kind}.filtered.fa', f'{kind}.raw13.tsv'}
    present = {p.name for p in unit.iterdir() if p.is_file()}
    check.add('required_files', expected <= present, {'missing': sorted(expected - present)})
    # (a) hash every file present; compare recorded hashes afterwards.
    for name in sorted(present):
        files[name] = launcher.sha256(unit / name)
    if prov_path.is_file():
        provenance = json.loads(prov_path.read_text())
    if provenance is None or expected - present:
        check.add('provenance_hashes', False, 'provenance.json or required files absent')
        return {'unit': f'{species}/{kind}', 'checks': check.results, 'files': files,
                'status': 'FAIL'}
    hash_pairs = {f'{kind}.raw13.tsv': 'raw_sha256', f'{kind}.dmnd': 'db_sha256',
                  f'{kind}.filtered.fa': 'filtered_fasta_sha256'}
    mismatched = {n: {'fresh': files[n], 'recorded': provenance.get(key)}
                  for n, key in hash_pairs.items() if files[n] != provenance.get(key)}
    check.add('provenance_hashes', not mismatched, mismatched or 'all recorded hashes reproduced')
    # Reproduce screening and PRE-search taxonomic exclusion, never post-top25 filtering.
    inputs = provenance['inputs']
    check.add('input_identity', inputs['species'] == species and inputs['source_kind'] == kind,
              {'species': inputs['species'], 'source_kind': inputs['source_kind']})
    input_qc = launcher.production_inputs(**inputs)
    check.add('screening', all(provenance.get(k) == v for k, v in input_qc.items()), input_qc)
    with tempfile.TemporaryDirectory(prefix='blastx-audit-filter-') as scratch:
        filter_qc = launcher.filter_source(inputs['source'], kind, inputs['lineage'],
                                          launcher.SPECIES[species], Path(scratch) / 'filtered.fa')
    check.add('presearch_exclusion', all(provenance.get(k) == v for k, v in filter_qc.items()),
              filter_qc)
    elapsed = provenance.get('elapsed_seconds')
    check.add('elapsed_seconds', type(elapsed) in (int, float) and math.isfinite(elapsed)
              and elapsed >= 0, elapsed)
    check.add('raw_fields', provenance.get('fields') == launcher.RAW13, provenance.get('fields'))
    # (b) validate_raw contract + count reconciliation.
    try:
        qc = launcher.validate_raw(unit / f'{kind}.raw13.tsv')
        ok = (qc['raw_hsps'] == provenance.get('raw_hsps')
              and qc['hsps_lt_30aa'] == provenance.get('hsps_lt_30aa'))
        check.add('raw13_contract', ok, {'recomputed': qc,
                  'recorded': {k: provenance.get(k) for k in ('raw_hsps', 'hsps_lt_30aa')}})
    except ValueError as exc:
        check.add('raw13_contract', False, str(exc))
    subjects = {h.split()[0] for h, _ in launcher.records(unit / f'{kind}.filtered.fa')}
    unknown_subjects = set()
    with open(unit / f'{kind}.raw13.tsv') as raw:
        for line in raw:
            subject = line.rstrip('\n').split('\t')[2]
            if subject not in subjects:
                unknown_subjects.add(subject)
    check.add('subjects_presearch_filtered', not unknown_subjects, sorted(unknown_subjects)[:10])
    # (c) frozen command tokens.
    try:
        recorded = json.loads((unit / 'commands.json').read_text())
        expected = list(expected_commands(recorded))
        makedb, search = recorded
        bindings = [(makedb, '--in', unit / f'{kind}.filtered.fa'),
                    (makedb, '--db', unit / f'{kind}.dmnd'),
                    (search, '--db', unit / f'{kind}.dmnd'),
                    (search, '--out', unit / f'{kind}.raw13.tsv'),
                    (search, '--query', Path(inputs['query']))]
        bound = all(Path(argv[argv.index(flag) + 1]).is_absolute() and
                    Path(argv[argv.index(flag) + 1]).resolve() == path.resolve()
                    for argv, flag, path in bindings)
        engine_ok = (makedb[0] == search[0] and Path(search[0]).is_absolute()
                     and launcher.sha256(search[0]) == provenance.get('diamond_sha256'))
        check.add('frozen_commands', recorded == expected and bound and engine_ok,
                  {'recorded': recorded, 'paths_bound': bound, 'engine_bound': engine_ok})
    except (ValueError, IndexError) as exc:
        check.add('frozen_commands', False, str(exc))
    # (d) query resolution across both layouts + qseqid subset.
    expected_md5 = None
    if layout == 'remaining8':
        try:
            for e in json.loads((batch_root / 'inputs.json').read_text()):
                if isinstance(e, dict) and e.get('species') == species:
                    expected_md5 = e.get('query_md5')
        except (OSError, ValueError):
            pass
    query_path = resolve_query(layout, species, species_root, batch_root, provenance, check)
    audit_query_file(query_path, provenance, expected_md5, unit / f'{kind}.raw13.tsv', check)
    status = 'FAIL' if check.failed() else 'PASS'
    return {'unit': f'{species}/{kind}', 'layout': layout, 'checks': check.results,
            'files': files, 'status': status}


def audit_species_root(species, species_root, check_prefix, check):
    # (e) batch.exitcode + timing files.
    exitcode = species_root / 'batch.exitcode'
    ok = exitcode.is_file() and exitcode.read_text().strip() == '0'
    check.add(f'{check_prefix}:batch_exitcode', ok,
              exitcode.read_text().strip() if exitcode.is_file() else 'missing batch.exitcode')
    timing = {p.name: launcher.sha256(p) for p in sorted(species_root.glob('*.time.txt'))}
    check.add(f'{check_prefix}:timing_files', bool(timing) and all(
        (species_root / name).stat().st_size > 0 for name in timing),
              timing or 'missing timing evidence')
    return timing


def disposition(path):
    """Fail-closed DONE / FAILED.json / NEITHER disposition for a directory."""
    if path.is_symlink():
        return {'disposition': 'SYMLINK', 'ok': False}
    if (path / 'DONE').exists() and (path / 'FAILED.json').exists():
        return {'disposition': 'CONTRADICTORY', 'ok': False}
    if (path / 'DONE').is_file():
        return {'disposition': 'DONE', 'ok': (path / 'DONE').read_text() == DONE_TEXT}
    if (path / 'FAILED.json').is_file():
        return {'disposition': 'FAILED', 'ok': False}
    return {'disposition': 'NEITHER', 'ok': False}


def scan_dispositions(root, known, check):
    """Must-fix 3: EVERY sibling dir (preflight_20260914/, tmp/, ...) gets an
    explicit disposition; NEITHER and unreconciled tmp/ are failures."""
    entries = {}

    def visit(parent):
        for entry in sorted(parent.iterdir()):
            key = str(entry.relative_to(root))
            if entry.is_symlink():
                entries[key] = {'disposition': 'SYMLINK', 'ok': False}
                continue
            if not entry.is_dir():
                continue
            if parent == root and entry.name in known:
                continue  # explicitly audited separately
            state = disposition(entry)
            if entry.name == 'tmp':
                contents = sorted(p.name for p in entry.iterdir())
                state.update(ok=False, entries=contents,
                             note='tmp requires explicit reconciliation; never silently skipped')
            entries[key] = state
            visit(entry)
    visit(root)
    bad = {k: v for k, v in entries.items() if not v['ok']}
    check.add(f'dispositions:{root}', not bad, entries or 'no unexpected sibling dirs')
    return entries


def hash_batch_artifacts(root, check_name, check):
    """Reviewer suggestion: run.sbatch + slurm .out logs join the manifest."""
    hashed = {}
    for pattern in ('run.sbatch', 'slurm-*.out', 'slurm-*.err', 'logs/*.out', 'logs/*.err'):
        for p in sorted(root.glob(pattern)):
            hashed[str(p.relative_to(root))] = launcher.sha256(p)
    check.add(check_name, bool(hashed), hashed or 'no batch artifacts found')
    return hashed


def audit_accounting(text, check):
    """Require sacct -P --format=JobID,State,ExitCode (header, no truncation).

    Payload completion and extern OOM are deliberately separate claims.
    Unexpected extern state still fails; this is not an OOM waiver.
    """
    reader = csv.DictReader(io.StringIO(text), delimiter='|')
    if not reader.fieldnames or not {'JobID', 'State', 'ExitCode'} <= set(reader.fieldnames):
        check.add('accounting_format', False, 'need JobID|State|ExitCode header')
        return
    rows = {}
    for row in reader:
        rows.setdefault(row['JobID'], []).append(row)
    for job in ['6163851'] + [f'6164049_{i}' for i in range(8)]:
        for suffix in ('', '.batch', '.extern'):
            key = job + suffix
            matches = rows.get(key, [])
            expected = ('OUT_OF_MEMORY', '0:125') if suffix == '.extern' else ('COMPLETED', '0:0')
            ok = len(matches) == 1 and (matches[0]['State'], matches[0]['ExitCode']) == expected
            check.add('accounting:' + key, ok, matches)


def audit(blastx_root, out_dir, sacct_file):
    blastx_root = Path(blastx_root).resolve()
    output = Path(out_dir).resolve()
    if output == blastx_root or blastx_root in output.parents:
        raise ValueError('audit output must be outside the input tree')
    scratch = Path(tempfile.gettempdir()).resolve()
    if scratch == blastx_root or blastx_root in scratch.parents:
        raise ValueError('audit scratch must be outside the input tree')
    sacct_text = Path(sacct_file).read_text()
    with launcher.work_directory(out_dir) as out:
        (out / 'units').mkdir()
        check = Check()
        audit_accounting(sacct_text, check)
        units = []
        pilot_root = blastx_root / PILOT_DIR
        batch_root = blastx_root / REMAINING8_DIR
        for root, name in ((pilot_root, PILOT_DIR), (batch_root, REMAINING8_DIR)):
            check.add(f'root_present:{name}', root.is_dir(), str(root))
        if not pilot_root.is_dir() or not batch_root.is_dir():
            launcher.write_json(out / 'rollup.json',
                                {'status': 'FAIL', 'checks': check.results})
            raise ValueError(f'audit FAILED: {check.failed()}')
        # Pilot layout: unit dirs + query.sha256 at species root.
        audit_species_root('Sitalica', pilot_root, 'Sitalica', check)
        hash_batch_artifacts(pilot_root, 'batch_artifacts:pilot', check)
        scan_dispositions(pilot_root, set(SOURCES), check)
        for kind in SOURCES:
            units.append(audit_unit('pilot', 'Sitalica', kind, pilot_root, pilot_root))
        # remaining8 layout: <batch>/<species>/<kind>; inputs at the batch root.
        hash_batch_artifacts(batch_root, 'batch_artifacts:remaining8', check)
        scan_dispositions(batch_root, set(REMAINING8_SPECIES) | {'logs'}, check)
        if (batch_root / 'logs').is_dir():
            scan_dispositions(batch_root / 'logs', set(), check)
        for species in REMAINING8_SPECIES:
            species_root = batch_root / species
            if not species_root.is_dir():
                check.add(f'{species}:species_root', False, f'missing {species_root}')
                continue
            audit_species_root(species, species_root, species, check)
            scan_dispositions(species_root, set(SOURCES), check)
            for kind in SOURCES:
                units.append(audit_unit('remaining8', species, kind, species_root, batch_root))
        # Sibling dirs of the two batch roots (e.g. preflight_20260914/) --
        # enumerated with explicit dispositions, never silently skipped.
        scan_dispositions(blastx_root, {PILOT_DIR, REMAINING8_DIR}, check)
        for unit in units:
            launcher.write_json(out / 'units' / (unit['unit'].replace('/', '_') + '.json'), unit)
        failed_units = [u['unit'] for u in units if u['status'] != 'PASS']
        status = 'FAIL' if (failed_units or check.failed() or len(units) != 18) else 'PASS'
        rollup = {'status': status, 'units_audited': len(units), 'units_expected': 18,
                  'failed_units': failed_units, 'tree_checks': check.results,
                  'sacct_evidence_verbatim': sacct_text,
                  'sacct_sha256': launcher.sha256(sacct_file),
                  'sacct_note': SACCT_NOTE, 'clean_accounting': False,
                  'accepted_phase_evidence': False, 'production_authorized': False,
                  'audit_script_sha256': launcher.sha256(__file__),
                  'launcher_sha256': launcher.sha256(launcher.__file__)}
        launcher.write_json(out / 'rollup.json', rollup)
        if status != 'PASS':
            raise ValueError(f'audit FAILED: units={failed_units} '
                             f'tree_checks={check.failed()} n_units={len(units)}')
        return rollup


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blastx-root', type=Path, required=True,
                   help='directory containing sitalica_pilot_20260914/ and remaining8_20260915/')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--sacct-file', type=Path, required=True,
                   help='sacct -P --format=JobID,State,ExitCode, with header; all allocation/batch/extern rows')
    args = p.parse_args(argv)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    audit(args.blastx_root, args.out_dir, args.sacct_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
