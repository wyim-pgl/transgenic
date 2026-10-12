#!/usr/bin/env python3
"""LOCAL ONLY: fresh immutable bundle from the exact reviewed dirty candidate."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import deploy as d

HERE = Path(__file__).resolve().parent
SMALL_PINS = {
    d.ACCEPT: 'faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e',
    d.INVENTORY: '9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98',
    d.RECEIPTS: '73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac',
    d.PANEL: '79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243',
    d.CONFIG: '9b8bef9b642526835a0817e9b13ae50b7c7f274fe882256426cb78f90c080d79',
}


def candidate_payload(candidate):
    candidate = Path(candidate)
    d.pinned(candidate / 'candidate.sha256', d.CANDIDATE_LEDGER_SHA)
    for line in (candidate / 'candidate.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        d.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe ledger path')
        d.pinned(candidate / name, digest)
    d.pinned(candidate / 'source-manifest.json', d.CANDIDATE_MANIFEST_SHA)
    d.pinned(candidate / 'code.candidate.tar', d.CANDIDATE_ARCHIVE_SHA)
    manifest = json.loads((candidate / 'source-manifest.json').read_text())
    d.require(manifest['base_commit'] == d.BASE_COMMIT and
              manifest['archive_sha256'] == d.CANDIDATE_ARCHIVE_SHA, 'candidate identity mismatch')
    payload = {}
    seen = set()
    with tarfile.open(candidate / 'code.candidate.tar') as tar:
        for m in tar.getmembers():
            p = Path(m.name)
            d.require(m.name not in seen and not p.is_absolute() and '..' not in p.parts and
                      (m.isdir() or m.isfile()), 'unsafe/duplicate candidate member')
            seen.add(m.name)
            if m.isfile(): payload[m.name] = tar.extractfile(m).read()
    measured = {n: hashlib.sha256(raw).hexdigest() for n, raw in payload.items()}
    d.require(len(payload) == 103 and measured == manifest['source_files'], 'candidate inventory/hash mismatch')
    for name in (d.CONFIG, d.PANEL):
        d.require(measured[name] == SMALL_PINS[name], 'candidate small input changed')
    return manifest, payload


def operator_pins(files, payload):
    # Same component selection as final_p1/test_deploy.py and reference-map builder.
    component = {n: h for n, h in files.items() if n.startswith(
        ('src/transgenic/', 'configs/', 'revision/scripts/mvp/')) or n == 'train/train_HyenaTransgenic_MVP.py'}
    tree = ast.parse(payload['revision/scripts/mvp/mvp_decode_runtime.py'])
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'DECODER_ONLY' for t in n.targets))
    decoder = {n: files[n] for n in names}
    return {'source_comparison_files': component, 'source_comparison_sha256': d.binding_digest(component),
            'source_comparison_count': len(component), 'decoder_files': decoder,
            'decoder_code_sha256': d.binding_digest(decoder), 'decoder_file_count': len(decoder),
            'canonicalization': 'JSON sorted keys, compact separators, trailing LF; SHA256',
            'not_runtime_consumed_manifest': True,
            'separately_pinned_helper': {'revision/scripts/mvp/mvp_verify_pinned_panel.py':
                files['revision/scripts/mvp/mvp_verify_pinned_panel.py']}}


def build(candidate, out):
    candidate, out = Path(candidate).resolve(strict=True), Path(out).absolute()
    if out.exists() or out.is_symlink(): raise FileExistsError(str(out))
    d.require(out.resolve() == out, 'output parent must not alias another path')
    manifest, payload = candidate_payload(candidate)
    companions = {n: SMALL_PINS[n] for n in (d.ACCEPT, d.INVENTORY, d.RECEIPTS)}
    for n, h in companions.items(): d.pinned(candidate / n, h)
    out.mkdir()  # exclusive and never recover a partial package
    names = ['code.candidate.tar', 'source-manifest.json', *companions]
    for name in names:
        dest = out / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate / name, dest)
    for name in ('deploy.py', 'admission.py', 'authorization.example.json', 'p1.sbatch', 'build_release.py', 'controller_execution_scope.json'):
        shutil.copyfile(HERE / name, out / name); names.append(name)
    d.save(out / 'operator-pins.json', operator_pins(manifest['source_files'], payload))
    names.append('operator-pins.json')
    release_files = {n: d.sha(out / n) for n in names}
    d.save(out / 'release.json', {'source_provenance': manifest, 'source_files': manifest['source_files'],
        'release_files': release_files, 'accepted_companions_not_in_source_commit': companions,
        'small_input_pins': SMALL_PINS, 'candidate_ledger_sha256': d.CANDIDATE_LEDGER_SHA,
        'profile': 'development-single', 'seed': 123, 'run_role': 'mvp_finetune',
        'scope': 'S0_P1_DIAGNOSTIC_ONLY', 'node_hours_max': 0.5, 'gpu_hours_max': 2,
        'measured': False, 'runtime_ready': False, 'training_authorized': False,
        'review_status': 'PENDING_INDEPENDENT_NEW_PACKAGE_REVIEW'})
    d.verify_source_manifest(out, json.loads((out / 'release.json').read_text()))
    with (out / 'release.sha256').open('x') as f:
        for n in (*names, 'release.json'): f.write(f'{d.sha(out / n)}  {n}\n')
    d.freeze(out)
    return out


if __name__ == '__main__':
    out = build(HERE.parent / 'candidate', HERE / 'release_v3')
    print('Release:', out)
    for name in ('release.json', 'release.sha256', 'operator-pins.json'):
        print(name, d.sha(out / name))
