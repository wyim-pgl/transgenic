#!/usr/bin/env python3
"""Tiny synthetic semantic diagnostic; no biological data, installs, or policy choices."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types

BINARY = '/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare'
PRODUCER_SHA = 'bf7c8284804ee0a12c359a1fbe8f140bfc0f222abb522d6d7b7b4d3a691b417b'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def producer(path):
    """Execute only five EXACT AST definitions from hash-pinned reviewed source.

    Avoid importing unrelated torch/duckdb/finalizer stack. The sole dependency
    shim is w.sha256_file, implemented by stdlib SHA256, not a tool substitute.
    """
    assert sha(path) == PRODUCER_SHA, 'reviewed producer bytes changed'
    names = {'require', 'file_identity', 'structure', 'write_gtf', 'reference_counts'}
    tree = ast.parse(Path(path).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in tree.body} == names
    ns = dict(Path=Path, csv=csv, subprocess=subprocess, w=types.SimpleNamespace(sha256_file=sha))
    exec(compile(tree, str(path), 'exec'), ns)
    return types.SimpleNamespace(**ns)


def tx(exons, strand='+', cds=()):
    return dict(chrom='smoke1', strand=strand, exons=exons, cdss=cds)


def cases():
    # All coordinates are zero-based half-open. Each case is a separate locus/run.
    r = tx([(100, 200), (300, 400), (500, 600)])
    terminal = tx([(120, 200), (300, 400), (500, 580)])
    alt = tx([(100, 200), (310, 400), (500, 600)])
    mono = tx([(1000, 1200)], cds=[(1020, 1180)])
    # Expected tuples are independently hand-counted (TP, FP, FN), not tool output.
    return {
        'ordinary': ({'q': r}, {'r': r}, (1, 0, 0), '='),
        'terminal_short': ({'q': terminal}, {'r': r}, (1, 0, 0), '='),
        'terminal_extend': ({'q': tx([(80, 200), (300, 400), (500, 620)])}, {'r': r}, (1, 0, 0), '='),
        'junction_change': ({'q': alt}, {'r': r}, (0, 1, 1), 'not_equal'),
        'opposite_strand': ({'q': tx(r['exons'], '-')}, {'r': r}, (0, 1, 1), 'not_equal'),
        'unmatched': ({'q': tx([(2000, 2100), (2200, 2300)])}, {'r': r}, (0, 1, 1), 'not_equal'),
        'mono_exact': ({'q': mono}, {'r': mono}, (1, 0, 0), None),
        'mono_utr_variant': ({'q': tx([(990, 1210)], cds=[(1020, 1180)])}, {'r': mono}, (1, 0, 0), None),
        'mono_cds_change': ({'q': tx([(1000, 1200)], cds=[(1021, 1180)])}, {'r': mono}, (0, 1, 1), None),
        'mono_opposite': ({'q': tx([(1000, 1200)], '-', [(1020, 1180)])}, {'r': mono}, (0, 1, 1), None),
        'mono_no_cds': ({'q': tx([(1000, 1200)])}, {'r': tx([(1000, 1200)])}, (0, 1, 1), None),
        'empty_query': ({}, {'r': r}, (0, 0, 1), 'observe'),
        'multiisoform_unique': ({'q': r, 'q_alt': alt}, {'r': r, 'r_alt': alt}, (2, 0, 0), '='),
        'duplicate_query_terminal': ({'q': r, 'q_terminal': terminal}, {'r': r}, 'predicted', 'observe'),
        'duplicate_reference_terminal': ({'q': r}, {'r': r, 'r_terminal': terminal}, 'reference', 'observe'),
        'duplicate_mono_cds': ({'q': mono, 'q_utr': tx([(990, 1210)], cds=[(1020, 1180)])}, {'r': mono}, 'predicted', None),
    }


def generate(root, p):
    root.mkdir(parents=True, exist_ok=False)
    dump(root / 'fixture.json', cases())
    for name, (pred, ref, _, _) in cases().items():
        d = root / name
        d.mkdir()
        for label, ts in [('predicted', pred), ('reference', ref)]:
            p.write_gtf(d / (label + '.gtf'), {k: v for k, v in ts.items() if len(v['exons']) > 1})


def call(command, directory):
    directory.mkdir()
    dump(directory / 'command.json', command)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (directory / 'stdout.txt').write_text(result.stdout)
    (directory / 'stderr.txt').write_text(result.stderr)
    dump(directory / 'exit.json', result.returncode)
    return result


def run(root, source):
    p = producer(source)
    assert Path(BINARY).is_absolute() and Path(BINARY).is_file()
    generate(root, p)
    version = call([BINARY, '--version'], root / 'version')
    assert version.returncode == 0 and re.search(r'\bv?0\.12\.6\b', version.stdout + version.stderr), 'wrong version'
    dump(root / 'runtime.json', {'binary': BINARY, 'binary_resolved': str(Path(BINARY).resolve()),
                               'binary_sha256': sha(BINARY), 'producer_sha256': sha(source),
                               'harness_sha256': sha(__file__), 'python': sys.version})
    for name, (pred, ref, expected, code) in cases().items():
        d = root / name
        # Independent raw invocation retains actual duplicate/empty behavior even
        # when the producer must reject/skip. No tmap is ever manufactured.
        if code is not None:
            cmd = [BINARY, '-r', str(d / 'reference.gtf'), '-o', str(d / 'raw' / 'compare'), str(d / 'predicted.gtf')]
            call(cmd, d / 'raw')
        try:
            counts, receipt = p.reference_counts(pred, ref, d / 'producer', BINARY)
            dump(d / 'result.json', {'counts': counts, 'receipt': receipt})
        except Exception as error:
            dump(d / 'result.json', {'error_type': type(error).__name__, 'error': str(error)})
    runtime = json.loads((root / 'runtime.json').read_text())
    assert runtime['binary_sha256'] == sha(BINARY), 'binary changed during run'
    assert runtime['producer_sha256'] == sha(source), 'producer changed during run'
    assert runtime['harness_sha256'] == sha(__file__), 'harness changed during run'
    dump(root / 'hashes.json', {str(f.relative_to(root)): sha(f) for f in sorted(root.rglob('*')) if f.is_file()})


def analyze(root):
    # Read-only: emit report on stdout; never modify the evidence directory.
    hashes = json.loads((root / 'hashes.json').read_text())
    actual = {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()} - {'hashes.json'}
    assert actual == set(hashes), 'missing or unrecorded raw file'
    assert all(sha(root / f) == h for f, h in hashes.items()), 'raw hash mismatch'
    assert json.loads((root / 'fixture.json').read_text()) == json.loads(json.dumps(cases()))
    runtime = json.loads((root / 'runtime.json').read_text())
    assert runtime['producer_sha256'] == PRODUCER_SHA and runtime['harness_sha256'] == sha(__file__)
    assert runtime['binary'] == BINARY
    assert json.loads((root / 'version/command.json').read_text()) == [BINARY, '--version']
    assert json.loads((root / 'version/exit.json').read_text()) == 0
    assert re.search(r'\bv?0\.12\.6\b', (root / 'version/stdout.txt').read_text() + (root / 'version/stderr.txt').read_text())
    observations = {}
    for name, (pred, ref, expected, code) in cases().items():
        d = root / name
        result = json.loads((d / 'result.json').read_text())
        if isinstance(expected, str):
            assert result == {'error_type': 'ValueError', 'error': expected + ' duplicate isoform structures: author multiplicity policy needed'}, (name, result)
            assert not list((d / 'producer').iterdir()), 'duplicate guard ran tool/wrote GTF'
        else:
            tp, fp, fn = expected
            assert result['counts'] == dict(f1_tp=tp, f1_predicted=tp + fp, f1_reference=tp + fn), (name, result)
            assert [result['receipt'][k] for k in ('TP', 'FP', 'FN')] == list(expected)
            command = result['receipt']['command']
            if code in ('=', 'not_equal'):
                assert command == [BINARY, '-r', str(d / 'producer/reference.gtf'), '-o', str(d / 'producer/compare'), str(d / 'producer/predicted.gtf')]
                # Producer contract intentionally expects one *.tmap (normally
                # compare.predicted.gtf.tmap), not an invented fixed filename.
                maps = list((d / 'producer').glob('*.tmap'))
                assert len(maps) == 1, name
            else:
                assert command is None, 'mono/empty producer must not invoke tool'
        if code is not None:
            assert json.loads((d / 'raw/command.json').read_text()) == [BINARY, '-r', str(d / 'reference.gtf'), '-o', str(d / 'raw/compare'), str(d / 'predicted.gtf')]
            rc = json.loads((d / 'raw/exit.json').read_text())
            maps = list((d / 'raw').glob('*.tmap'))
            rows = []
            for path in maps:
                with path.open() as f:
                    rows.extend(csv.DictReader(f, delimiter='\t'))
            observations[name] = {'exit': rc, 'tmap_names': [f.name for f in maps], 'rows': rows}
            if code != 'observe':
                assert rc == 0 and len(maps) == 1, name
                assert len(rows) == len(pred) and {r['qry_id'] for r in rows} == set(pred), name
                assert all((r['class_code'] == '=') == (code == '=') for r in rows), (name, rows)
            elif pred:
                assert rc == 0 and len(maps) == 1 and rows, 'duplicate raw diagnostic failed'
                # Retain rows/IDs, including possible collapse; choose no isoform.
    return {'status': 'SEMANTIC_CHECKS_PASSED', 'observations': observations,
            'duplicate_policy': 'UNRESOLVED_FAIL_CLOSED', 'biological_compatibility': 'NOT_TESTED',
            'decision': None, 'runtime': runtime}


def selfcheck(source):
    import tempfile
    p = producer(source)
    # Hand calculations independent of tmap and independent of producer matcher.
    for name in ('ordinary', 'terminal_short', 'terminal_extend'):
        exons = cases()[name][0]['q']['exons']
        assert [(exons[0][1], exons[1][0]), (exons[1][1], exons[2][0])] == [(200, 300), (400, 500)]
        assert cases()[name][2] == (1, 0, 0)
    assert cases()['junction_change'][0]['q']['exons'] == [(100, 200), (310, 400), (500, 600)]
    assert cases()['junction_change'][2] == (0, 1, 1)
    assert cases()['multiisoform_unique'][2] == (2, 0, 0)
    assert cases()['empty_query'][2] == (0, 0, 1)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'fixture'
        generate(root, p)
        ordinary = (root / 'ordinary/predicted.gtf').read_text().splitlines()
        assert [line.split('\t')[3:5] for line in ordinary] == [['101', '200'], ['301', '400'], ['501', '600']]
        terminal = (root / 'terminal_short/predicted.gtf').read_text().splitlines()
        assert [line.split('\t')[3:5] for line in terminal] == [['121', '200'], ['301', '400'], ['501', '580']]
        assert all(line.endswith('gene_id "g"; transcript_id "q";') for line in ordinary)
        assert (root / 'mono_exact/predicted.gtf').read_bytes() == b''
        try:
            analyze(root)
        except FileNotFoundError:
            pass  # generated fixtures alone must never yield a tool PASS
        else:
            raise AssertionError('unexecuted fixture accepted')
        # Real-tool cases are deliberately NOT executed or marked passed locally.
        for name, (pred, ref, expected, code) in cases().items():
            if code in ('=', 'not_equal', 'observe') and not isinstance(expected, str) and pred:
                continue
            try:
                counts, receipt = p.reference_counts(pred, ref, Path(td) / name, '/NO_TOOL_MUST_BE_CALLED')
            except ValueError as e:
                assert isinstance(expected, str) and str(e).startswith(expected + ' duplicate isoform structures:')
            else:
                assert not isinstance(expected, str), 'duplicate accepted'
                assert (receipt['TP'], receipt['FP'], receipt['FN']) == expected
                assert receipt['command'] is None
    print('LOCAL_ONLY: generator/mono/empty/duplicate checks passed; real-tool semantics NOT RUN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['generate', 'run', 'analyze', 'selfcheck'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--producer', type=Path)
    args = parser.parse_args()
    if args.mode == 'selfcheck':
        selfcheck(args.producer)
    elif args.mode == 'analyze':
        print(json.dumps(analyze(args.root.resolve()), indent=2, sort_keys=True))
    elif args.mode == 'run':
        run(args.root.resolve(), args.producer)
    else:
        generate(args.root.resolve(), producer(args.producer))


if __name__ == '__main__':
    main()
