#!/usr/bin/env python3
"""Preparation only: deterministic metadata, never transfer, mount, submit or run proof.
All reads are local bounded metadata/current-source archive, not DB/SIF payloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent
NOTES = PANEL.parents[1]
GPU = PANEL.parent / 'gpu/candidate'
G = '/data/gpfs/assoc/pgl/data/Transgenic'
HOST = '/work/nvme/bilv/wcyim/transgenic/seed123_panel_transport_20260917'
MANIFEST_SHA = '7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e'
SOURCE_SHA = 'da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7'
REQUEST_SHA = 'c5cc29ecea746375ebb4d5388d38fa2bb8db0d5c42088ac67bc47b4caa7da9a9'
RUNTIME = {'architecture': 'aarch64', 'python': '3.12.3', 'numpy': '2.1.0',
           'duckdb': '1.5.5', 'torch': '2.14.0a0+4fdf77b940.nv26.08', 'git': '2.43.0',
           'sif': '/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif',
           'sif_sha256': '1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6',
           'sif_size': 11120398336}


def fail(message):
    raise ValueError(message)


def relative(name):
    p = PurePosixPath(name)
    if (not name or not p.parts or p.is_absolute() or str(p) != name or '..' in p.parts
            or any(c in name for c in ':,\\\n\r\t\0')
            or name.endswith(('.wal', '-wal'))):
        fail('unsafe relative path: ' + repr(name))
    return name


def absolute(name):
    if not isinstance(name, str) or not name.startswith('/') or name == '/':
        fail('absolute non-root path required')
    relative(name[1:])
    return name


def unique_paths(names):
    seen = set()
    for name in names:
        if name in seen:
            fail('duplicate path: ' + name)
        seen.add(name)
    for name in seen:
        if any(str(p) in seen for p in PurePosixPath(name).parents):
            fail('file/parent collision: ' + name)


def no_symlinks(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        if p.is_symlink():
            fail('symlink rejected: ' + str(p))
    return path


def small_bytes(path, expected=None, size=None):
    path = no_symlinks(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 2_000_000:
        fail('only bounded local regular metadata allowed: ' + str(path))
    if path.suffix in ('.db', '.sif'):
        fail('DB/SIF reads prohibited in preparation')
    data = path.read_bytes()
    if size is not None and len(data) != size:
        fail('size differs: ' + str(path))
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        fail('fixed pin differs: ' + str(path))
    return data


def pinned_json(path, sha):
    return json.loads(small_bytes(path, sha))


def build():
    h = pinned_json(PANEL / 'historical_inputs.json', '5b93339087fe1303920ff255dafbb534ecc634578c2eb2b2f842c13169fa0082')
    accepted = NOTES / 'power_recovery_20260917/panel_arrays/copied/panel'
    raw = small_bytes(accepted / 'manifest.json', MANIFEST_SHA, 38854)
    old = json.loads(raw)
    complete = small_bytes(accepted / 'COMPLETE', '600be19cd4deef9c84be302f0716ac77e8a3429ee70bda4be7ff20dbefe3c7eb', 65)
    if complete != (MANIFEST_SHA + '\n').encode() or old['inputs'] != h['inputs'] or old['authority'] != h['authority']:
        fail('historical metadata disagreement')
    if len(old['inputs']) != 15 or len(old['loci']) != 50:
        fail('historical inventory count differs')
    panel_root = str(PurePosixPath(h['canonical_manifest']).parent)
    files = [dict(x) for x in h['inputs']]
    files.extend([{'path': panel_root + '/manifest.json', 'size': len(raw), 'sha256': MANIFEST_SHA},
                  {'path': panel_root + '/COMPLETE', 'size': len(complete), 'sha256': hashlib.sha256(complete).hexdigest()}])
    for i, locus in enumerate(old['loci']):
        for key, suffix in [('sequence', 'fa'), ('track', 'npy'), ('provenance', 'jsonl')]:
            x = locus[key]
            if x['path'] != f'{i:02d}.{suffix}':
                fail('unexpected historical artifact numbering')
            files.append({'path': panel_root + '/' + relative(x['path']), 'size': x['bytes'], 'sha256': x['sha256']})
    for row in files:
        if not row['path'].startswith(G + '/'):
            fail('historical path outside canonical G')
        row['relative_path'] = relative(row['path'][len(G) + 1:])
        row['original_path'] = row['path']
        row['action'] = 'reuse_existing_no_copy' if row['path'].endswith('.db') else 'replica_after_review'
        row['provenance'] = 'historical recorded bytes; canonical mount is not physical original continuity'
        row['final_backing_path'] = None
    unique_paths([x['relative_path'] for x in files])
    if len(files) != 167 or sum(x['size'] for x in files if x['action'] == 'replica_after_review') != 212152146:
        fail('historical payload inventory differs')
    source = pinned_json(GPU / 'source-manifest.json', '4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a')
    small_bytes(GPU / 'code.candidate.tar', SOURCE_SHA, 1873920)
    rows = []
    # Read current 1.9 MB source only. No extraction or changes to producer archive.
    with tarfile.open(GPU / 'code.candidate.tar', 'r:') as archive:
        for member in archive.getmembers():
            relative(member.name)
            if not member.isfile() or member.size > 2_000_000:
                fail('nonregular/oversized source archive member')
            content = archive.extractfile(member).read()
            rows.append({'path': member.name, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    unique_paths([x['path'] for x in rows])
    if len(rows) != 103 or {x['path']: x['sha256'] for x in rows} != source['source_files']:
        fail('exact 103-file current source inventory differs')
    closure = pinned_json(PANEL / 'consumer_closure_candidate.json', 'f3536323350eb42de37569bf2bac7b0c6810abedfc3e010e1b5766d6f18afa10')
    by_name = {x['path']: x for x in rows}
    for row in closure['consumer_closure']:
        if by_name[row['relative_path']] != {'path': row['relative_path'], 'size': row['size'], 'sha256': row['sha256']}:
            fail('consumer report closure differs')
    wrapper = 'revision/scripts/mvp/mvp_verify_pinned_panel.py'
    if by_name[wrapper]['sha256'] != closure['launcher_separate']['sha256']:
        fail('proof wrapper differs')
    request = pinned_json(PANEL / 'request_pin.json', 'b777128f1275265407281d69e1a0dfc3e3fee5d3b96dfdb8dfa1d4b2e0af7bef')['expected_verification_request']
    small_bytes(PANEL / 'candidate_request.json', REQUEST_SHA, 1264)
    if request['sha256'] != REQUEST_SHA or request['size'] != 1264:
        fail('request fixed pin differs')
    authority_dir = NOTES / 'wait_work_20260916/panel_inputs'
    authority = pinned_json(authority_dir / 'authority_export_receipt.json', '0d913d71994c3146db6922e1a5e80125329fa3e4c624945a089921cc60e38e28')
    small_bytes(authority_dir / 'authority.py', '0b3917b7fa33df0d1818bdbad992f664e8dc6b959639a09aa998f75a97e8b2dc', 3433)
    return {'schema': 1, 'status': 'PREPARATION_ONLY_BLOCKED', 'canonical_G': G, 'host_namespace': HOST,
            'historical_manifest_sha256': MANIFEST_SHA, 'historical_files': files,
            'historical_authority': h['authority'], 'historical_packages': h['historical_packages'],
            'consumer_root': '/work/repo', 'consumer_archive_sha256': SOURCE_SHA,
            'consumer_archive_size': 1873920, 'consumer_files': sorted(rows, key=lambda x: x['path']),
            'consumer_closure': [{'path': '/work/repo/' + x['relative_path'], 'size': x['size'], 'sha256': x['sha256']} for x in closure['consumer_closure']],
            'wrapper': by_name[wrapper], 'authority': authority,
            'authority_verifier_sha256': '0b3917b7fa33df0d1818bdbad992f664e8dc6b959639a09aa998f75a97e8b2dc',
            'request': request, 'runtime': dict(RUNTIME), 'p1_stage_receipt': None,
            'reviewed_resource_binding': None, 'independent_no_copy_amendment': None,
            'output_parent': None, 'mount_binding': None,
            'budget': {'max_nodes': 1, 'max_node_hours': 1, 'campaign_cap_node_hours': 5,
                       'cpu_workload_on_gpu_backed_node': True, 'zero_gpu_billing_claim': False,
                       'wall_minutes_ceiling': 60, 'approved': False}}


def validate_manifest(value):
    # Compare to independent constants plus pinned originals, never trust self-reported hashes.
    if value != build():
        fail('candidate manifest differs from independently fixed metadata')


def validate_tree(root, rows):
    """Local synthetic/current small-tree audit. Not a deployment or DB verifier."""
    root = no_symlinks(root)
    expected = {relative(x['path']): x for x in rows}
    unique_paths([x['path'] for x in rows])
    found = set()
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            p = no_symlinks(Path(parent) / name)
            if p.is_dir():
                if name == '__pycache__':
                    fail('bytecode directory rejected')
                continue
            rel = p.relative_to(root).as_posix()
            if rel not in expected:
                fail('unlisted tree file: ' + rel)
            x = expected[rel]
            small_bytes(p, x['sha256'], x['size'])
            found.add(rel)
    if found != set(expected):
        fail('missing tree file')


def validate_mounts(mounts):
    """Validate proposed top-level roots, not a claim of actual mount installation.
    DB nested RO overlays require a separately reviewed explicit mount binding.
    """
    if len(mounts) != 3 or [x['target'] for x in mounts] != [G, '/work/repo', '/panel-output']:
        fail('exact canonical input/source/output roots required')
    if [x['mode'] for x in mounts] != ['ro', 'ro', 'rw']:
        fail('input/source readonly and output writable required')
    for key in ('source', 'target'):
        names = [absolute(x[key]) for x in mounts]
        unique_paths(names)
    sources = [x['source'] for x in mounts]
    targets = [x['target'] for x in mounts]
    for source in sources:
        for target in targets:
            if source == target or source.startswith(target + '/') or target.startswith(source + '/'):
                fail('host/container namespace alias or collision')


def require_fresh(path):
    p = no_symlinks(path)
    if os.path.lexists(p):
        fail('namespace reuse prohibited: ' + str(p))
    if not p.parent.is_dir():
        fail('existing canonical output parent required')


def require_execution_ready(manifest):
    if manifest.get('p1_stage_receipt') is None:
        fail('actual reviewed P1 stage receipt absent; final backing paths unknown')
    # No operator boolean or edited JSON can unlock this preparation-only revision.
    fail('execution adapter and independent review of actual P1 receipt/resource/mount binding required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'validate'])
    args = parser.parse_args()
    target = HERE / 'transport_manifest.json'
    if args.mode == 'build':
        value = build()
        with target.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print('Created metadata only:', target)
    else:
        validate_manifest(json.loads(small_bytes(target)))
        print('VALID candidate metadata; execution remains BLOCKED')


if __name__ == '__main__':
    main()
