#!/usr/bin/env python3
"""Read-only recovery of captured job 6165521; no engine/source execution.

Only tx/cases definitions are loaded from the SHA-pinned historical harness.
The explicit remote origin is authenticated by the pinned status/manifest record,
then checked literally in every command and producer output receipt. It is never
replaced inside evidence. This is a recovery of ONE captured run, not a new runner.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASE = Path(__file__).resolve().parents[1]
ORIGIN = '/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_gffcompare_smoke_20260916/run'
MANIFEST_SHA = '335bee5e27b65cfdd7eea1a32b3ef349ed9f31ee9853074245b2b64eb8f724af'
BINARY = '/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare'
BINARY_SHA = 'a440e1533611a40a51cf2b5b35a2e15ec6b885660dd35f6b531e0d65601a963a'
PINS = {
    'smoke.py': '3c022e41fa2759a33d8fee76452a55eca5ff2b226527a37fddecbba79c6098d1',
    'smoke.sbatch': 'e6d18fcc732e183b7b3f40d3176e76889608f316453070ef4e12c5086ae0258b',
    'mvp_produce_counts.py': 'bf7c8284804ee0a12c359a1fbe8f140bfc0f222abb522d6d7b7b4d3a691b417b',
    'stage.sha256': 'd269951e04a0d4845aef6665c80af38528c539b152f5fe35ba5740ccce6708ff',
    'result_status.txt': '96d8114b5c1c3b3d15a45a036a82d475b22048242c061d47738f30dc5b72a6da',
    'submission_receipt.txt': '7509d598c54cbdd2f1ddec015dc58742b88adf6728d2121e79a62895b5e99ccd',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}


def load(path):
    return json.loads(path.read_text())


def fixture():
    tree = ast.parse((BASE / 'smoke.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'tx', 'cases'}]
    require(len(tree.body) == 2, 'historical fixture definitions absent')
    namespace = {}
    exec(compile(tree, str(BASE / 'smoke.py'), 'exec'), namespace)
    return json.loads(json.dumps(namespace['cases']()))


def chain(tx):
    return tx['chrom'], tx['strand'], tuple((a[1], b[0]) for a, b in zip(tx['exons'], tx['exons'][1:]))


def expected_gtf(transcripts):
    # Independent rendering check of the original exon-only, 1-based fixture.
    return ''.join(f'{tx["chrom"]}\tMVP\texon\t{start + 1}\t{end}\t.\t{tx["strand"]}\t.\tgene_id "g"; transcript_id "{tid}";\n'
                   for tid, tx in sorted(transcripts.items()) if len(tx['exons']) > 1
                   for start, end in tx['exons'])


def command(origin, case, producer=False):
    case_dir = PurePosixPath(origin) / case
    query_dir = case_dir / 'producer' if producer else case_dir
    prefix_dir = query_dir if producer else case_dir / 'raw'
    return [BINARY, '-r', str(query_dir / 'reference.gtf'), '-o', str(prefix_dir / 'compare'), str(query_dir / 'predicted.gtf')]


def maps(root, directory):
    # GFFCompare 0.12.6 writes maps beside the query, NOT beside -o.
    # Exact captured paths only; inventory validation rejects all extra files.
    result = {}
    for suffix in ('tmap', 'refmap'):
        relative = f'{directory}/compare.predicted.gtf.{suffix}'
        with (root / relative).open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        result[suffix + '_path'] = relative
        result[suffix + '_rows'] = rows
    return result


def check_tmap(rows, predicted, reference, code, case):
    require(len(rows) == len(predicted) and {r['qry_id'] for r in rows} == set(predicted), f'{case}: query inventory mismatch')
    if code != 'observe':
        require(all((r['class_code'] == '=') == (code == '=') for r in rows), f'{case}: unexpected raw class')
    for row in rows:
        if row['class_code'] == '=':
            require(row['ref_id'] in reference and chain(predicted[row['qry_id']]) == chain(reference[row['ref_id']]), f'{case}: exact-chain mismatch')


def analyze(root, origin):
    root = Path(root)
    require(origin == ORIGIN, 'logged absolute origin mismatch')
    evidence = {name: identity(BASE / name) for name in PINS}
    require(all(evidence[n]['sha256'] == h for n, h in PINS.items()), 'historical evidence pin mismatch')
    stage = {line.split()[1]: line.split()[0] for line in (BASE / 'stage.sha256').read_text().splitlines()}
    require(stage == {n: PINS[n] for n in ('smoke.py', 'smoke.sbatch', 'mvp_produce_counts.py')}, 'stage inventory mismatch')
    status = (BASE / 'result_status.txt').read_text()
    require(f'{MANIFEST_SHA}  {origin}/hashes.json' in status.splitlines(), 'origin not bound to logged manifest')
    require('6165521|FAILED|1:0|00:00:02|' in status and '6165521.extern|OUT_OF_MEMORY|0:125|' in status, 'scheduler evidence mismatch')
    require('submitted=6165521' in (BASE / 'submission_receipt.txt').read_text(), 'submission mismatch')
    require(identity(root / 'hashes.json')['sha256'] == MANIFEST_SHA, 'remote manifest pin mismatch')
    hashes = load(root / 'hashes.json')
    paths = list(root.rglob('*'))
    require(not root.is_symlink() and not any(p.is_symlink() for p in paths), 'symlink evidence forbidden')
    actual = {str(p.relative_to(root)) for p in paths if p.is_file()}
    require(actual == set(hashes) | {'hashes.json'}, 'inventory mismatch: missing or extra file')
    inventory = {name: identity(root / name) for name in sorted(actual)}
    require(all(inventory[n]['sha256'] == h for n, h in hashes.items()), 'raw hash mismatch')
    cases = fixture()
    require(load(root / 'fixture.json') == cases and len(cases) == 16, 'fixture identity mismatch')
    require({p.name for p in root.iterdir() if p.is_dir()} == set(cases) | {'version'}, 'case directory inventory mismatch')
    runtime = load(root / 'runtime.json')
    require(runtime['producer_sha256'] == PINS['mvp_produce_counts.py'] and runtime['harness_sha256'] == PINS['smoke.py'], 'runtime source mismatch')
    require(runtime['binary'] == runtime['binary_resolved'] == BINARY and runtime['binary_sha256'] == BINARY_SHA, 'runtime binary mismatch')
    require(load(root / 'version/command.json') == [BINARY, '--version'] and load(root / 'version/exit.json') == 0, 'version command failed')
    version = (root / 'version/stdout.txt').read_text() + (root / 'version/stderr.txt').read_text()
    require(re.search(r'\bv?0\.12\.6\b', version), 'wrong version')
    observations = {}
    for name, (pred, ref, expected, code) in cases.items():
        directory = root / name
        for label, transcripts in (('predicted', pred), ('reference', ref)):
            require((directory / f'{label}.gtf').read_text() == expected_gtf(transcripts), f'{name}: fixture GTF mismatch')
        result = load(directory / 'result.json')
        observation = {'expected': expected, 'producer_counts': None, 'producer_error': None, 'raw': None, 'producer_maps': None}
        if isinstance(expected, str):
            error = {'error_type': 'ValueError', 'error': expected + ' duplicate isoform structures: author multiplicity policy needed'}
            require(result == error, f'{name}: duplicate error mismatch')
            require(not any((directory / 'producer').iterdir()), f'{name}: duplicate guard wrote files')
            observation['producer_error'] = result
        else:
            tp, fp, fn = expected
            counts = dict(f1_tp=tp, f1_predicted=tp + fp, f1_reference=tp + fn)
            receipt = result['receipt']
            require(result['counts'] == receipt['counts'] == counts, f'{name}: count mismatch')
            observed = [receipt[k] for k in ('TP', 'FP', 'FN')]
            require(observed == expected, f'{name}: TP/FP/FN mismatch')
            observation['producer_counts'] = observed
            expected_outputs = {f'{name}/producer/{p.name}' for p in (directory / 'producer').iterdir()}
            recorded_outputs = []
            for output in receipt['outputs']:
                # Resolve only under the pinned origin; never discard an arbitrary prefix.
                remote = PurePosixPath(output['path'])
                require(remote.is_absolute() and '..' not in remote.parts, f'{name}: invalid receipt path')
                relative = str(remote.relative_to(PurePosixPath(origin)))
                require(relative in expected_outputs and output['path'] == f'{origin}/{relative}', f'{name}: receipt origin/path mismatch')
                require({k: output[k] for k in ('sha256', 'size')} == inventory[relative], f'{name}: receipt hash/size mismatch')
                recorded_outputs.append(relative)
            require(len(recorded_outputs) == len(set(recorded_outputs)) and set(recorded_outputs) == expected_outputs, f'{name}: receipt inventory mismatch')
            for label, transcripts in (('predicted', pred), ('reference', ref)):
                require((directory / f'producer/{label}.gtf').read_text() == expected_gtf(transcripts), f'{name}: producer GTF mismatch')
            if code in ('=', 'not_equal'):
                require(receipt['command'] == command(origin, name, True), f'{name}: producer command mismatch')
                pmaps = maps(root, f'{name}/producer')
                check_tmap(pmaps['tmap_rows'], pred, ref, code, name)
                unique_matches = {r['ref_id'] for r in pmaps['tmap_rows'] if r['class_code'] == '='}
                require(len(unique_matches) == tp, f'{name}: producer maps/count disagreement')
                observation['producer_maps'] = pmaps
            else:
                require(receipt['command'] is None and expected_outputs == {f'{name}/producer/predicted.gtf', f'{name}/producer/reference.gtf'}, f'{name}: bypass invoked tool')
        if code is not None:
            require(load(directory / 'raw/command.json') == command(origin, name), f'{name}: raw command mismatch')
            rc = load(directory / 'raw/exit.json')
            raw_maps = maps(root, name)
            # All ten actual captured calls returned zero, including empty query.
            # Empty/duplicate observations are not turned into scoring policy.
            require(rc == 0, f'{name}: captured raw call failed')
            check_tmap(raw_maps['tmap_rows'], pred, ref, code, name)
            observation['raw'] = dict(raw_maps, exit=rc, command=load(directory / 'raw/command.json'))
            if observation['producer_maps'] is not None:
                require(raw_maps['tmap_rows'] == observation['producer_maps']['tmap_rows'] and raw_maps['refmap_rows'] == observation['producer_maps']['refmap_rows'], f'{name}: raw/producer map disagreement')
        observations[name] = observation
    return {'status': 'SEMANTIC_CHECKS_PASSED', 'scope': 'CAPTURED_16_CASE_SYNTHETIC_FUNCTION_DIAGNOSTIC_ONLY',
            'controller_independent_review': 'PENDING', 'origin': origin, 'manifest_sha256': MANIFEST_SHA,
            'original_job': {'job_id': 6165521, 'allocation': 'FAILED', 'batch': 'FAILED', 'exit': '1:0', 'elapsed': '00:00:02', 'extern': 'OUT_OF_MEMORY', 'extern_exit': '0:125', 'extern_cause': 'UNRESOLVED'},
            'runtime': runtime, 'version': version.strip(), 'binary_proof': 'HASH_RECORDED_BY_PINNED_HARNESS_BEFORE_AND_AFTER_RUN; BINARY_BYTES_NOT_COPIED_OR_REHASHED_LOCALLY',
            'evidence_inventory': evidence, 'inventory': inventory, 'observations': observations,
            'duplicate_policy': 'UNRESOLVED_FAIL_CLOSED', 'biological_compatibility': 'NOT_TESTED', 'decision': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='read-only local copy of original run/')
    parser.add_argument('--origin', required=True, help='exact logged absolute remote run origin')
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.origin), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
