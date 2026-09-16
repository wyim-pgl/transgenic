#!/usr/bin/env python3
"""Fail-closed audit of completed EST blastx output trees (stage C1). stdlib only.

Audits the 9-species x 2-source raw13 units produced by est_blastx.py under
  <blastx-root>/sitalica_pilot_20260914/          (pilot layout)
  <blastx-root>/remaining8_20260915/<species>/    (batch layout)
per unit checks (a)-(f):
  (a) sha256 every immediate regular unit file; nested directories are unresolved
      FAIL (closed, flat inventory). Recomputed raw13/dmnd/filtered hashes must equal
      the values recorded inside provenance.json;
  (b) re-run the exact validate_raw() row contract over raw13 and reconcile
      raw_hsps / hsps_lt_30aa with provenance;
  (c) commands.json argv must equal the frozen makedb/search command tokens;
      producer code and engine hashes bind to historical approved identities;
  (d) resolve the query FASTA for BOTH provenance layouts (pilot: query.sha256
      at the species root; remaining8: inputs.json + inputs.sha256 at the batch
      root), re-hash it, and require qseqid set <= query record IDs.  An
      unresolvable query file is a HARD FAIL, never a skip;
  (e) species batch.exitcode must be 0; both SOURCE.time.txt files are required;
  (f) DONE-presence gate: absence of DONE always means incomplete.
Additionally EVERY directory entry under the scanned roots -- including
preflight_20260914/ siblings and tmp/ dirs -- receives an explicit fail-closed
disposition (DONE / FAILED.json / NEITHER); contradictions, NEITHER and any
tmp/ directory require reconciliation and fail, never silently skipped.  run.sbatch and slurm .out logs are hashed into
the manifest. Wrapper bytes must match the archived historical scripts; all
remaining8 logs are required. Pilot log names require the pinned recovered
submission receipt. Historical directories require an external reviewed manifest;
missing or invalid evidence remains UNRESOLVED FAIL, never a guessed default.
Container-owned markers are checked even when their child layouts are known;
normal containers need no DONE. sacct evidence is embedded VERBATIM from a capture;
the persisting `.extern OUT_OF_MEMORY 0:125` anomaly is recorded as UNRESOLVED
-- this audit never claims clean accounting.

This script performs no Slurm submission and never writes inside the audited
tree.  Production execution requires the separate reviewed sbatch template and
author authorization.
"""
from __future__ import annotations

import argparse
import csv
import datetime
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


# Historical trust anchors, NOT hashes of the current auditor or caller-supplied code.
# Launcher/helper bytes verified from git 950cf4cd9ef55db276ffa1b713d8925ed6dd6dd2.
# Engine pin and wrapper byte identities: notes/b6_{sitalica_blastx_20260914,
# eight_species_blastx_20260915}.sbatch. No binary is executed by this auditor.
APPROVED_LAUNCHER_SHA256 = '5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291'
APPROVED_HELPER_SHA256 = '32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302'
APPROVED_DIAMOND_SHA256 = '0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d'
APPROVED_WRAPPER_SHA256 = {
    'pilot': '8e8f501ebaa0cbd15ee8d61aef127f5cb7ee3fdc3a070d3c7d18923440de2f4a',
    'remaining8': '4cdad18b443c13ad564be640859419e86def7fb255fe82cabe797888913fdd7c'}


HISTORICAL_ROOT = Path('/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx')
PILOT_RECEIPT_SHA256 = 'f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23'
PREFLIGHT_DIR = 'preflight_20260914'
PREFLIGHT_WRAPPER_SHA256 = 'dfe295d76d9f4f15a5b153e6dcd4fb1e9fbd658aa1a6d59539c92bb8c33bf318'
LINEAGE_SHA256 = {'orthodb': '812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5',
                  'swissprot': '402bca970569cc2c0192ba185c12c8fcaed99ecafd71f46a5e19f4f2789615ce'}


class EvidenceError(ValueError):
    """Invalid external evidence, as distinct from a programming error."""


def evidence_call(function, *args, **kwargs):
    """The historical launcher's validation APIs signal bad input with ValueError."""
    try:
        return function(*args, **kwargs)
    except ValueError as exc:
        raise EvidenceError(str(exc)) from exc


def validate_provenance(value):
    if type(value) is not dict:
        raise EvidenceError('provenance must be a JSON object')
    inputs = value.get('inputs')
    keys = set('species source_kind source source_sha256 query query_sha256 lineage lineage_sha256 roles roles_sha256'.split())
    if type(inputs) is not dict or set(inputs) != keys or any(
            type(v) is not str or not v for v in inputs.values()):
        raise EvidenceError('inputs must contain exactly the producer string fields')
    hash_keys = ('raw_sha256 db_sha256 filtered_fasta_sha256 screening_summary_sha256 '
                 'screening_done_sha256 source_sha256 lineage_sha256 diamond_sha256 '
                 'launcher_sha256 taxonomy_helper_sha256').split()
    for key in hash_keys:
        if type(value.get(key)) is not str or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise EvidenceError(f'invalid or missing producer hash: {key}')
    for key in ('source', 'source_kind', 'lineage', 'diamond_version'):
        if type(value.get(key)) is not str or not value[key]:
            raise EvidenceError(f'invalid or missing producer string: {key}')
    for key in ('queries', 'species_root', 'sequences_in', 'sequences_out',
                'sequences_excluded', 'raw_hsps', 'hsps_lt_30aa'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise EvidenceError(f'invalid or missing producer count: {key}')
    for key in ('retained_taxids', 'excluded_taxids'):
        if type(value.get(key)) is not list or any(type(x) is not int or x < 1 for x in value[key]):
            raise EvidenceError(f'invalid producer taxid list: {key}')
    counts = value.get('counts_by_taxid')
    if type(counts) is not dict or any(
            not re.fullmatch('[0-9]+', taxid) or type(row) is not dict or set(row) != {'in', 'out'}
            or any(type(n) is not int or n < 0 for n in row.values())
            for taxid, row in counts.items()):
        raise EvidenceError('invalid counts_by_taxid object')
    if type(value.get('fields')) is not list or any(type(x) is not str for x in value['fields']):
        raise EvidenceError('invalid raw fields list')
    if type(value.get('elapsed_seconds')) not in (int, float):
        raise EvidenceError('invalid elapsed_seconds type')
    return value


def batch_inputs(path):
    value = json.loads(path.read_text())
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise EvidenceError('batch inputs.json must be a list of objects')
    for row in value:
        if any(type(row.get(k)) is not str or not row[k] for k in ('species', 'query', 'query_md5')):
            raise EvidenceError('batch species/query/query_md5 must be nonempty strings')
        if not re.fullmatch('[0-9a-f]{32}', row['query_md5']):
            raise EvidenceError('invalid batch query_md5')
    return value


def producer_identity(provenance, check):
    version = provenance['diamond_version']
    lines = version.split('\n')
    # Exact accepted release/build contract of 950cf4c: no current binary probes.
    version_ok = (len(lines) == 2 and re.fullmatch(r'diamond version 2\.1\.24(?:\.178)?', lines[0])
                  and re.fullmatch(r'diamond v2\.1\.24\.178 \(C\) .+', lines[1]))
    expected = {'launcher_sha256': APPROVED_LAUNCHER_SHA256,
                'taxonomy_helper_sha256': APPROVED_HELPER_SHA256,
                'diamond_sha256': APPROVED_DIAMOND_SHA256}
    ok = bool(version_ok) and all(expected.values()) and all(provenance[k] == v for k, v in expected.items())
    check.add('historical_producer_identity', ok, {
        'expected': expected, 'recorded': {k: provenance[k] for k in expected},
        'diamond_version': version,
        'authority': 'git 950cf4c launcher/helper; archived launch wrappers engine pin',
        'note': 'UNRESOLVED if mismatched/missing; no independent version-banner receipt or binary execution'})


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
    entries = [e for e in batch_inputs(inputs_json) if e['species'] == species]
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
    check, files = Check(), {}
    try:
        return _audit_unit(layout, species, kind, species_root, batch_root, check, files)
    except (OSError, EvidenceError, json.JSONDecodeError, UnicodeError) as exc:
        check.add('malformed_or_unreadable', False, str(exc))
        return {'unit': f'{species}/{kind}', 'layout': layout, 'status': 'FAIL',
                'checks': check.results, 'files': files}


def _audit_unit(layout, species, kind, species_root, batch_root, check, files):
    unit = species_root / kind
    # (f) DONE gate first: absence of DONE always means incomplete.
    done = unit / 'DONE'
    if not unit.is_dir():
        check.add('unit_dir', False, f'missing unit directory: {unit}')
        return {'unit': f'{species}/{kind}', 'checks': check.results, 'files': files,
                'status': 'FAIL'}
    scan_dispositions(unit, set(), check)
    nested = sorted(p.name for p in unit.iterdir() if p.is_dir())
    check.add('closed_unit_inventory', not nested,
              {'UNRESOLVED_nested_directories': nested})
    check.add('done_gate', disposition(unit)['ok'],
              {'present': done.is_file()})
    if (unit / 'FAILED.json').exists():
        check.add('failed_marker', False, 'FAILED.json present in unit')
    provenance = None
    prov_path = unit / 'provenance.json'
    expected = set(UNIT_FILES) | {f'{kind}.dmnd', f'{kind}.filtered.fa', f'{kind}.raw13.tsv'}
    present = {p.name for p in unit.iterdir() if p.is_file() and not p.is_symlink()}
    check.add('required_files', expected <= present, {'missing': sorted(expected - present)})
    # (a) hash every file present; compare recorded hashes afterwards.
    for name in sorted(present):
        files[name] = launcher.sha256(unit / name)
    if prov_path.is_file():
        provenance = validate_provenance(json.loads(prov_path.read_text()))
    if provenance is None or expected - present:
        check.add('provenance_hashes', False, 'provenance.json or required files absent')
        return {'unit': f'{species}/{kind}', 'checks': check.results, 'files': files,
                'status': 'FAIL'}
    producer_identity(provenance, check)
    hash_pairs = {f'{kind}.raw13.tsv': 'raw_sha256', f'{kind}.dmnd': 'db_sha256',
                  f'{kind}.filtered.fa': 'filtered_fasta_sha256'}
    mismatched = {n: {'fresh': files[n], 'recorded': provenance.get(key)}
                  for n, key in hash_pairs.items() if files[n] != provenance.get(key)}
    check.add('provenance_hashes', not mismatched, mismatched or 'all recorded hashes reproduced')
    # Reproduce screening and PRE-search taxonomic exclusion, never post-top25 filtering.
    inputs = provenance['inputs']
    check.add('input_identity', inputs['species'] == species and inputs['source_kind'] == kind,
              {'species': inputs['species'], 'source_kind': inputs['source_kind']})
    input_qc = evidence_call(launcher.production_inputs, **inputs)
    check.add('screening', all(provenance.get(k) == v for k, v in input_qc.items()), input_qc)
    with tempfile.TemporaryDirectory(prefix='blastx-audit-filter-') as scratch:
        filter_qc = evidence_call(launcher.filter_source, inputs['source'], kind, inputs['lineage'],
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
    try:
        # records() validates lazily: catch evidence errors during consumption.
        subjects = {h.split()[0] for h, _ in launcher.records(unit / f'{kind}.filtered.fa')}
    except ValueError as exc:
        raise EvidenceError(str(exc)) from exc
    unknown_subjects = set()
    with open(unit / f'{kind}.raw13.tsv') as raw:
        for line in raw:
            columns = line.rstrip('\n').split('\t')
            if len(columns) != 13:
                raise EvidenceError('malformed raw13 subject row')
            subject = columns[2]
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
        for e in batch_inputs(batch_root / 'inputs.json'):
            if e['species'] == species:
                expected_md5 = e['query_md5']
    query_path = evidence_call(resolve_query, layout, species, species_root, batch_root, provenance, check)
    evidence_call(audit_query_file, query_path, provenance, expected_md5, unit / f'{kind}.raw13.tsv', check)
    status = 'FAIL' if check.failed() else 'PASS'
    return {'unit': f'{species}/{kind}', 'layout': layout, 'checks': check.results,
            'files': files, 'status': status}


def audit_species_root(species, species_root, check_prefix, check):
    # (e) batch.exitcode + timing files.
    exitcode = species_root / 'batch.exitcode'
    ok = exitcode.is_file() and exitcode.read_text().strip() == '0'
    check.add(f'{check_prefix}:batch_exitcode', ok,
              exitcode.read_text().strip() if exitcode.is_file() else 'missing batch.exitcode')
    timing = {p.name: launcher.sha256(p) for p in sorted(species_root.glob('*.time.txt'))
              if p.is_file() and not p.is_symlink()}
    required = {f'{kind}.time.txt' for kind in SOURCES}
    missing = sorted(required - set(timing))
    invalid = sorted(name for name in required & set(timing)
                     if (species_root / name).is_symlink() or (species_root / name).stat().st_size == 0)
    check.add(f'{check_prefix}:timing_files', not missing and not invalid,
              {'hashes': timing, 'missing': missing, 'invalid': invalid})
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


def audit_container_markers(root, check):
    """Containers need no DONE, but present invalid/failed markers cannot be ignored."""
    markers = [root / name for name in ('DONE', 'FAILED.json')]
    present = [p for p in markers if p.exists() or p.is_symlink()]
    ok = not present
    detail = 'NO_MARKERS_REQUIRED'
    if present:
        try:
            ok = (len(present) == 1 and present[0].name == 'DONE'
                  and not present[0].is_symlink() and present[0].is_file()
                  and present[0].read_text() == DONE_TEXT)
            detail = {'present': [p.name for p in present], 'valid_done_only': ok}
        except (OSError, UnicodeError) as exc:
            ok, detail = False, str(exc)
    check.add(f'container_markers:{root}', ok, detail)


def require_evidence(ok, message):
    if not ok:
        raise EvidenceError(message)


def regular_path(path):
    return (path.is_absolute() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)))


def historical_snapshot(path):
    """Exact small historical tree; never follows links or accepts special files."""
    require_evidence(not any(p.is_symlink() for p in (path, *path.parents)), f'symlink: {path}')
    st = path.stat()
    require_evidence(path.is_dir() or path.is_file(), f'unsupported kind: {path}')
    result = {'kind': 'directory' if path.is_dir() else 'file', 'uid': st.st_uid, 'gid': st.st_gid}
    if path.is_dir():
        names = {p.name for p in path.iterdir()}
        require_evidence(not any(n.startswith('FAILED') for n in names), f'failed marker: {path}')
        require_evidence(not any(n.startswith('DONE') and n != 'DONE' for n in names),
                         f'unknown DONE marker: {path}')
        if 'DONE' in names:
            require_evidence(regular_path(path / 'DONE') and (path / 'DONE').read_text() == DONE_TEXT,
                             f'invalid DONE: {path}')
        result['children'] = {n: historical_snapshot(path / n) for n in sorted(names)}
    else:
        require_evidence(st.st_size <= 1024 * 1024, f'not a small historical receipt: {path}')
        result['sha256'] = launcher.sha256(path)
    return result


def reconciliation_paths(root):
    """Closed path-to-producer map, not a user-extensible waiver pattern."""
    pilot, batch, pre = root / PILOT_DIR, root / REMAINING8_DIR, root / PREFLIGHT_DIR
    allowed = {}
    for name in ('tmp', 'execution.lock', 'submission.lock'):
        allowed[str(pilot / name)] = (['6163851'], pilot / 'run.sbatch', APPROVED_WRAPPER_SHA256['pilot'])
    jobs = [f'6164049_{i}' for i in range(8)]
    allowed[str(batch / 'submission.lock')] = (jobs, batch / 'run.sbatch', APPROVED_WRAPPER_SHA256['remaining8'])
    # Task-to-species identity is recorded in all eight historical stdout logs.
    for i, species in enumerate(REMAINING8_SPECIES):
        allowed[str(batch / species / 'tmp')] = ([f'6164049_{i}'], batch / 'run.sbatch', APPROVED_WRAPPER_SHA256['remaining8'])
    for name in ('', 'logs', 'receipts', 'submission.lock', 'tmp', 'xdg-cache', 'execution.lock'):
        allowed[str(pre / name)] = (['6163397'], pre / 'prepare.sbatch', PREFLIGHT_WRAPPER_SHA256)
    return allowed


def bound_capture(record, root):
    require_evidence(type(record) is dict and set(record) == {'path', 'sha256'}, 'invalid capture binding')
    require_evidence(type(record['path']) is str and type(record['sha256']) is str, 'invalid capture fields')
    path = Path(record['path'])
    source = Path(__file__).resolve().parents[3]
    require_evidence(regular_path(path) and path == path.resolve() and
                     root not in path.parents and source not in path.parents,
                     'manifest/captures must be regular files outside input and source trees')
    require_evidence(launcher.sha256(path) == record['sha256'], f'capture hash mismatch: {path}')
    return path.read_text()


def completed_jobs(text, jobs):
    reader = csv.DictReader(io.StringIO(text), delimiter='|')
    require_evidence(reader.fieldnames and {'JobID', 'State', 'ExitCode'} <= set(reader.fieldnames),
                     'invalid reconciliation accounting header')
    rows = list(reader)
    require_evidence(all(r['State'] in ('COMPLETED', 'OUT_OF_MEMORY', 'FAILED', 'CANCELLED',
                                        'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
                         for r in rows), 'active/unknown state in accounting capture')
    for job in jobs:
        for suffix in ('', '.batch', '.extern'):
            matched = [r for r in rows if r['JobID'] == job + suffix]
            expected = ('OUT_OF_MEMORY', '0:125') if suffix == '.extern' else ('COMPLETED', '0:0')
            require_evidence(len(matched) == 1 and (matched[0]['State'], matched[0]['ExitCode']) == expected,
                             f'uncompleted/ambiguous historical accounting: {job + suffix}')


def preflight_evidence(pre):
    listings = {
        '': set(('accounting-bounded-check.txt batch.exitcode duplicate-check-correct-user.txt '
                 'duplicate-check.txt execution.lock expected-sources.sha256 independent_lineage_audit.json '
                 'jobid lineage_ncbi.baseline.tsv logs orthodb prepare.sbatch receipts reviewed-small-files.sha256 '
                 'sbatch-receipt.json source-hash-attestation.txt submission.lock swissprot tmp xdg-cache').split()),
        'receipts': {'DB.sha256', 'PROVENANCE.txt', 'SOURCE_COMMIT.txt', 'build_dbs.sh',
                     'swissprot_viridiplantae_reviewed.fa.gz.sha256'},
        'logs': {'slurm-6163397.out', 'slurm-6163397.err'},
        'orthodb': {'DONE', 'lineage.tsv', 'provenance.json'},
        'swissprot': {'DONE', 'lineage.tsv', 'provenance.json'}}
    for name, expected in listings.items():
        require_evidence({p.name for p in (pre / name).iterdir()} == expected,
                         f'unknown/missing preflight entry: {pre / name}')
    require_evidence(regular_path(pre / 'jobid') and (pre / 'jobid').read_text() == '6163397\n', 'preflight jobid')
    require_evidence(json.loads((pre / 'sbatch-receipt.json').read_text()) ==
                     {'returncode': 0, 'stdout': '6163397\n', 'stderr': ''}, 'preflight submission receipt')
    require_evidence((pre / 'batch.exitcode').read_text() == '0\n', 'preflight batch exit')
    for kind, digest in LINEAGE_SHA256.items():
        unit = pre / kind
        require_evidence(disposition(unit)['ok'], f'preflight source incomplete: {kind}')
        require_evidence(launcher.sha256(unit / 'lineage.tsv') == digest, f'preflight lineage changed: {kind}')
        prov = json.loads((unit / 'provenance.json').read_text())
        require_evidence(type(prov) is dict and prov.get('lineage_sha256') == digest,
                         f'preflight provenance lineage mismatch: {kind}')


def load_reconciliation(path, digest, root, sacct_text, check):
    """All-or-nothing external evidence validation; malformed input preserves unit audit."""
    try:
        require_evidence(path is not None and digest is not None, 'UNRESOLVED reconciliation manifest absent')
        require_evidence(root == HISTORICAL_ROOT, 'reconciliation historical root mismatch')
        text = bound_capture({'path': str(path), 'sha256': digest}, root)
        data = json.loads(text)
        require_evidence(type(data) is dict and set(data) ==
                         {'schema', 'blastx_root', 'accounting', 'queue', 'entries'}, 'invalid manifest fields')
        require_evidence(data['schema'] == 'est-c1-reconciliation-v1' and data['blastx_root'] == str(root),
                         'manifest version/root mismatch')
        accounting = bound_capture(data['accounting'], root)
        queue = json.loads(bound_capture(data['queue'], root))
        require_evidence(type(queue) is dict and set(queue) == {'captured_at', 'command', 'stdout', 'returncode'},
                         'invalid queue capture')
        require_evidence(type(queue['returncode']) is int and queue['returncode'] == 0 and
                         queue['command'] == 'squeue --noheader --format=%i|%T|%j' and
                         type(queue['stdout']) is str and type(queue['captured_at']) is str, 'invalid queue command/result')
        captured = datetime.datetime.fromisoformat(queue['captured_at'])
        require_evidence(captured.tzinfo is not None, 'queue timestamp lacks timezone')
        age = (datetime.datetime.now(datetime.timezone.utc) - captured).total_seconds()
        require_evidence(0 <= age <= 3600, 'stale/future queue capture')
        for line in queue['stdout'].splitlines():
            fields = line.strip().split('|')
            require_evidence(len(fields) == 3 and all(fields), 'malformed queue row')
            job, _, name = fields
            require_evidence(not job.startswith(('6163851', '6164049', '6163397')) and
                             not any(s in name.lower() for s in ('blastx', 'lineage', 'est_')),
                             'active/queued historical or EST job evidence')
        require_evidence(type(data['entries']) is list and data['entries'], 'empty/invalid entries')
        allowed, reconciled = reconciliation_paths(root), {}
        for entry in data['entries']:
            require_evidence(type(entry) is dict and set(entry) ==
                             {'path', 'jobids', 'wrapper', 'wrapper_sha256', 'snapshot'}, 'invalid entry fields')
            name = entry['path']
            require_evidence(type(name) is str and name in allowed and name not in reconciled,
                             'unknown/duplicate historical path')
            jobs, wrapper, pin = allowed[name]
            require_evidence(entry['jobids'] == jobs and entry['wrapper'] == str(wrapper) and
                             entry['wrapper_sha256'] == pin and regular_path(wrapper) and launcher.sha256(wrapper) == pin,
                             'historical job/wrapper binding mismatch')
            completed_jobs(accounting, jobs)
            if jobs != ['6163397']:
                completed_jobs(sacct_text, jobs)
            target = Path(name)
            fresh = historical_snapshot(target)
            require_evidence(fresh == entry['snapshot'], f'changed kind/owner/listing/child hash: {name}')
            pre = root / PREFLIGHT_DIR
            if target == pre or pre in target.parents:
                # All preflight dispositions depend on the full recursively bound root.
                pre_paths = {p for p in allowed if p == str(pre) or str(pre) + '/' in p}
                supplied = {e['path'] for e in data['entries'] if type(e) is dict and type(e.get('path')) is str}
                require_evidence(pre_paths <= supplied, 'all preflight container/sentinel snapshots required')
                preflight_evidence(pre)
            if target.name in ('tmp', 'xdg-cache') or target == root / PILOT_DIR / 'execution.lock' or target == pre / 'submission.lock':
                require_evidence(fresh['kind'] == 'directory' and fresh['children'] == {}, 'historical scratch/sentinel not empty')
            elif target == pre / 'execution.lock':
                require_evidence(fresh['kind'] == 'file' and target.read_bytes() == b'', 'preflight flock sentinel kind/content')
            elif target.name == 'submission.lock':
                expected_job = '6163851' if target.parent.name == PILOT_DIR else '6164049'
                require_evidence(fresh['kind'] == 'directory' and set(fresh['children']) == {'jobid'} and
                                 (target / 'jobid').read_text() == expected_job + '\n', 'submission sentinel listing/jobid')
            else:
                require_evidence(fresh['kind'] == 'directory', 'historical container kind')
            reconciled[name] = fresh
        check.add('historical_reconciliation', True, {'manifest': str(path), 'sha256': digest,
                  'evidence': data, 'note': 'historical dispositions only; extern and scientific gates unresolved'})
        return reconciled
    except (OSError, ValueError, UnicodeError) as exc:
        check.add('historical_reconciliation', False, {'manifest': str(path), 'error': str(exc)})
        return {}


def scan_dispositions(root, known, check, reconciled=None):
    """Must-fix 3: EVERY sibling dir (preflight_20260914/, tmp/, ...) gets an
    explicit disposition; NEITHER and unreconciled tmp/ are failures."""
    audit_container_markers(root, check)
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
            if reconciled and str(entry) in reconciled and state['disposition'] == 'NEITHER':
                # Recheck bytes at use, not merely when loading the manifest.
                try:
                    same = historical_snapshot(entry) == reconciled[str(entry)]
                except (OSError, EvidenceError, UnicodeError):
                    same = False
                state.update(ok=same, reconciled=same,
                             note='exact historical evidence; no DONE inferred or written')
            entries[key] = state
            visit(entry)
    visit(root)
    bad = {k: v for k, v in entries.items() if not v['ok']}
    check.add(f'dispositions:{root}', not bad, entries or 'no unexpected sibling dirs')
    return entries


def hash_batch_artifacts(root, check_name, check, pilot_receipt=None):
    """Bind wrapper bytes; require receipt-backed log names, never Slurm defaults."""
    layout = check_name.rsplit(':', 1)[-1]
    if layout not in APPROVED_WRAPPER_SHA256:
        raise ValueError(f'unknown batch layout: {layout}')
    hashed = {}
    for pattern in ('*.sbatch', '*.out', '*.err', 'logs/*.out', 'logs/*.err'):
        for p in sorted(root.glob(pattern)):
            if p.is_file() and not p.is_symlink():
                hashed[str(p.relative_to(root))] = launcher.sha256(p)
    scripts = [name for name, digest in hashed.items()
               if name.endswith('.sbatch') and digest == APPROVED_WRAPPER_SHA256[layout]]
    check.add(check_name + ':script', bool(scripts),
              {'approved_sha256': APPROVED_WRAPPER_SHA256[layout], 'matched': scripts})
    if layout == 'remaining8':
        # notes/RESTART_HANDOFF_20260915.md:38 records these submission paths.
        required = {f'logs/6164049_{i}.{ext}' for i in range(8) for ext in ('out', 'err')}
        missing = sorted(required - set(hashed))
        check.add(check_name + ':logs', not missing, {'missing': missing})
    else:
        detail = {'required': ['slurm-6163851.out', 'slurm-6163851.err']}
        try:
            require_evidence(pilot_receipt is not None, 'UNRESOLVED pilot log-layout receipt absent')
            receipt = Path(pilot_receipt)
            require_evidence(regular_path(receipt) and launcher.sha256(receipt) == PILOT_RECEIPT_SHA256,
                             'UNRESOLVED recovered pilot receipt hash/source mismatch')
            # Whole-extract pin binds source session, raw-record hashes, call/result,
            # command paths and jobid. No shell execution or substring-based approval.
            records = json.loads(receipt.read_text())
            detail.update(receipt_sha256=PILOT_RECEIPT_SHA256, source=records[0]['source'],
                          record_lines=[r['line'] for r in records], jobid='6163851')
            require_evidence(root == HISTORICAL_ROOT / PILOT_DIR, 'pilot historical path mismatch')
            jobid = root / 'submission.lock/jobid'
            require_evidence(regular_path(jobid) and jobid.read_text() == '6163851\n', 'pilot jobid mismatch')
            require_evidence(hashed.get('run.sbatch') == APPROVED_WRAPPER_SHA256['pilot'],
                             'pilot producing wrapper mismatch')
            require_evidence(all(name in hashed for name in detail['required']), 'pilot log missing')
            ok = True
        except (OSError, ValueError, UnicodeError) as exc:
            ok = False
            detail['error'] = str(exc)
        check.add(check_name + ':logs', ok, detail)
    check.add(check_name + ':inventory', True, hashed)
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


def audit(blastx_root, out_dir, sacct_file, pilot_receipt=None,
          reconciliation_manifest=None, reconciliation_sha256=None):
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
        reconciled = load_reconciliation(reconciliation_manifest, reconciliation_sha256,
                                         blastx_root, sacct_text, check)
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
        hash_batch_artifacts(pilot_root, 'batch_artifacts:pilot', check, pilot_receipt)
        scan_dispositions(pilot_root, set(SOURCES), check, reconciled)
        for kind in SOURCES:
            units.append(audit_unit('pilot', 'Sitalica', kind, pilot_root, pilot_root))
        # remaining8 layout: <batch>/<species>/<kind>; inputs at the batch root.
        hash_batch_artifacts(batch_root, 'batch_artifacts:remaining8', check)
        scan_dispositions(batch_root, set(REMAINING8_SPECIES) | {'logs'}, check, reconciled)
        if (batch_root / 'logs').is_dir():
            scan_dispositions(batch_root / 'logs', set(), check)
        for species in REMAINING8_SPECIES:
            species_root = batch_root / species
            if not species_root.is_dir():
                check.add(f'{species}:species_root', False, f'missing {species_root}')
                continue
            audit_species_root(species, species_root, species, check)
            scan_dispositions(species_root, set(SOURCES), check, reconciled)
            for kind in SOURCES:
                units.append(audit_unit('remaining8', species, kind, species_root, batch_root))
        # Sibling dirs of the two batch roots (e.g. preflight_20260914/) --
        # enumerated with explicit dispositions, never silently skipped.
        scan_dispositions(blastx_root, {PILOT_DIR, REMAINING8_DIR}, check, reconciled)
        # Unit hashing can take hours: detect changes since the initial small-tree binding.
        for name, expected in reconciled.items():
            try:
                unchanged = historical_snapshot(Path(name)) == expected
                detail = 'unchanged historical snapshot'
            except (OSError, EvidenceError, UnicodeError) as exc:
                unchanged, detail = False, str(exc)
            check.add('reconciliation_final:' + name, unchanged, detail)
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
                  'audit_runtime_launcher_sha256': launcher.sha256(launcher.__file__),
                  'audit_runtime_taxonomy_helper_sha256': launcher.sha256(launcher.HELPER)}
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
    p.add_argument('--pilot-submission-receipt', type=Path)
    p.add_argument('--reconciliation-manifest', type=Path)
    p.add_argument('--reconciliation-sha256', help='independently reviewed manifest digest; no automatic waiver')
    args = p.parse_args(argv)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    audit(args.blastx_root, args.out_dir, args.sacct_file, args.pilot_submission_receipt,
          args.reconciliation_manifest, args.reconciliation_sha256)
    return 0


if __name__ == '__main__':
    sys.exit(main())
