#!/usr/bin/env python3
"""Freeze only an explicitly controller-pinned five-module closure. No deployment."""
import argparse
import importlib.util
import json
from pathlib import Path

# Load the colocated verifier without depending on CWD or PYTHONPATH (supports -I).
spec = importlib.util.spec_from_file_location('deployment_verifier', Path(__file__).with_name('run_compute.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def build(root, out, source_hashes, manifest, manifest_sha):
    root, out = v.no_links(root), v.no_links(out)
    if out.exists():
        raise FileExistsError(str(out))
    v.require(root != out and root not in out.parents, 'release must be outside source root')
    expected = {'revision/scripts/evidence/' + name + '.py' for name in v.MODULES}
    v.require(set(source_hashes) == expected, 'exact five explicit source hashes required')
    payload = {}
    for relative, sha in source_hashes.items():
        data = v.small_bytes(root / relative)
        v.require(v.digest(data) == sha, 'required source SHA mismatch: ' + relative)
        payload['code/evidence/' + Path(relative).name] = data
    payload['code/evidence/__init__.py'] = b''
    payload['manifest.json'] = v.small_bytes(manifest, 1024**2)
    v.require(v.digest(payload['manifest.json']) == manifest_sha, 'manifest SHA mismatch')
    json.loads(payload['manifest.json'])
    for name in ('run_compute.py', 'sample.sbatch'):
        payload[name] = v.small_bytes(Path(__file__).with_name(name))
    # Read and validate everything before creating output. Failures after mkdir leave
    # an incomplete release for inspection, never silently overwrite/rebuild it.
    out.mkdir(exist_ok=False)
    (out / 'code/evidence').mkdir(parents=True)
    for name, data in payload.items():
        with (out / name).open('xb') as stream:
            stream.write(data)
    ledger = dict(schema=1, files={name: v.digest(data) for name, data in sorted(payload.items())},
                  runtime=v.RUNTIME, source_hashes=source_hashes,
                  scope='one bounded Sitalica diagnostic; not biological acceptance')
    v.write_json(out / 'ledger.json', ledger)
    sha = v.digest(v.small_bytes(out / 'ledger.json'))
    v.verify_release(out, sha)
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-hashes', type=Path, required=True,
                        help='controller-reviewed JSON object: exact five relative paths -> SHA256')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    args = parser.parse_args()
    print(build(args.root, args.out, json.loads(v.small_bytes(args.source_hashes)),
                args.manifest, args.manifest_sha))


if __name__ == '__main__':
    main()
