#!/usr/bin/env python3
"""Reviewable launch recipe, deliberately not an execution authority.
No shell, SSH, subprocess, transport or scheduler operations are performed.
"""
import argparse
import json
import sys
from transport import G, HOST, build, require_execution_ready


def recipe():
    m = build()
    request = m['request']
    return {
        'status': 'BLOCKED_PREPARATION_ONLY',
        'host_namespace': HOST,
        'replica_tree': {'host': None, 'container': G, 'mode': 'ro',
                         'requires': 'reviewed exact 165 nonDB-file tree plus request; DB placeholders, no extras'},
        'existing_database_mounts': [
            {'host': None, 'container': x['path'], 'mode': 'ro', 'sha256': x['sha256'], 'size': x['size'],
             'requires': 'actual reviewed stable P1 stage receipt; no copy and no search fallback'}
            for x in m['historical_files'] if x['action'] == 'reuse_existing_no_copy'],
        'source_mount': {'host': None, 'container': '/work/repo', 'mode': 'ro',
                         'requires': 'exact 103-file current snapshot plus independently inventoried sparse Git authority; reject all extras'},
        'request_mount': {'host': None, 'container': request['path'], 'mode': 'ro',
                          'sha256': request['sha256'], 'size': request['size']},
        'output_mount': {'host': None, 'container': '/panel-output', 'mode': 'rw',
                         'requires': 'separate existing canonical parent, fresh exclusive run child, no receipt/pending reuse'},
        'runtime': m['runtime'],
        'container_options': ['exec', '--cleanenv', '--containall', '--no-home', '--no-mount', 'hostfs'],
        'container_argv': None,
        'environment': {'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
                        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                        'PYTHONPATH': '/work/repo/src:/work/repo/revision/scripts/mvp',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
                        'GIT_NO_REPLACE_OBJECTS': '1'},
        'wrapper_interpreter': None,
        'wrapper_argv': ['-B', '/work/repo/' + m['wrapper']['path'], '--input', request['path'],
                         '--expected-sha256', request['sha256'], '--expected-size', str(request['size']),
                         '--out', '/panel-output/compatibility.json'],
        'required_preimport_checks': [
            'real scheduler allocation verified via controller accounting, aarch64, one node, <=60min; no spoofed Slurm variables',
            'independent amended no-copy review identifies replicas explicitly; transfer inventory equals fixed manifest',
            'P1 terminal reviewed stage receipt pins stable DB backing paths, lifecycle lock and device/inode/stat identity',
            'SIF identity verification only in approved compute context; no DB/SIF login hashing',
            'fixed exact Python/NumPy/DuckDB/Torch/Git versions and module locations in intended SIF',
            'all 103 current source file sizes/hashes, full tree inventory including five separately inventoried Git objects, no extras/symlinks/pycache',
            'existing pinned authority.py verify, then compare returned authority to fixed five-object receipt and original manifest digest',
            'exact original 167 historical identities through canonical G, fixed request identity, no DB WAL',
            'actual /proc/self/mountinfo and stat provenance: canonical G/source/request RO, only separate output RW; no unexpected child overlays',
            'clear inherited container/Python/Git/loader injection; explicit intended interpreter and safe cwd; reject foreign imports',
            'atomic no-clobber namespace claim, receipt/pending collision rejection, retain stdout/stderr and failure sentinel',
            'repeat mount/input/request/source integrity after proof; no automatic retry or deletion'],
        'requires_future_reviewed_adapter': True,
        'budget': m['budget'],
        'warning': 'CPU workload on GPU-backed allocation can bill GPUs. No training; no acceptance receipt is created here.'
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.describe:
        print(json.dumps(recipe(), indent=2, sort_keys=True))
        return 0
    try:
        require_execution_ready(build())
    except ValueError as exc:
        print('BLOCKED: ' + str(exc), file=sys.stderr)
        return 2
    raise RuntimeError('preparation-only guard unexpectedly returned')


if __name__ == '__main__':
    sys.exit(main())
