"""Bound the reviewed archive CLI to three approved old B5 epochs.

Only the immutable operation's allocation may run this guard. Bulk copy,
hashing, durable publication, and source cleanup belong to reviewed3b4eb0d.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    # These pins are only small code/config/checkpoint-metadata files.
    require(path.stat().st_size <= 1024 * 1024, 'unexpected large metadata')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(str(HERE), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main(mode, epoch=None):
    require(mode in ('pre', 'transfer', 'post'), 'unsupported archive phase')
    require(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'allocation required')
    pins = json.loads((HERE / 'pins.json').read_text())
    helper = Path(pins['helper_dir'])
    run = Path(pins['source_root'])
    require((helper / 'SOURCE_COMMIT').read_text().strip() == pins['source_commit'], 'source revision changed')
    for name, expected in pins['helper_sha256'].items():
        require(sha(helper / name) == expected, 'archive helper changed: ' + name)
    require(sha(HERE / 'archive.json') == pins['config_sha256'], 'archive configuration changed')
    require(sha(run / 'latest_state/meta.json') == pins['latest_meta_sha256'], 'live resume position changed')
    require(sha(run / 'trainer_state.json') == pins['trainer_state_sha256'], 'trainer state changed')
    require(os.readlink(str(run / 'best')) == pins['best'], 'best checkpoint changed')
    for protected_epoch in pins['protected_epochs']:
        require((run / ('epoch_%02d' % protected_epoch)).is_dir(), 'protected epoch absent')
    core_path = helper / 'revision/scripts/b6/checkpoint_archive.py'
    spec = importlib.util.spec_from_file_location('archive_core', str(core_path))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    if mode == 'pre':
        plan = core.plan(str(run), min_age_seconds=300, max_epoch=11, max_bytes=48 * 1024 ** 3)
        require(plan['candidates'] == pins['expected_candidates'], 'archive candidates changed')
        require(plan['protected'] == [13, 14] and plan['refused'] == [], 'checkpoint protection changed')
        require(not core.pending_cleanup(str(run)), 'unresolved prior cleanup transaction')
        for item in pins['expected_candidates']:
            require(not os.path.lexists(str(Path(pins['destination_root']) / ('run-' + hashlib.sha256(str(run).encode()).hexdigest()[:16]) / item['name'])),
                    'destination already exists; inspect instead of starting a new operation')
        publish('preflight.json', dict(job_id=os.environ['SLURM_JOB_ID'], plan=plan, pins=pins))
        return
    pre = json.loads((HERE / 'preflight.json').read_text())
    require(pre['job_id'] == os.environ['SLURM_JOB_ID'] and pre['pins'] == pins, 'preflight belongs to another allocation')
    if mode == 'transfer':
        expected = [row for row in pins['expected_candidates'] if row['epoch'] == epoch]
        require(len(expected) == 1, 'epoch outside the pinned transfer scope')
        require(not core.pending_cleanup(str(run)), 'unresolved cleanup requires inspection')
        # Use explicit epochs, never CLI replanning or generic recovery of other
        # workers' transactions. The reviewed APIs still recheck protection and
        # immutable source identity under their own source/destination locks.
        manifest = core.prepare(str(run), epoch)
        require({key: manifest[key] for key in ('epoch', 'name', 'bytes')} == expected[0],
                'prepared checkpoint differs from pinned candidate')
        publish('epoch_%02d_manifest.json' % epoch, manifest)
        sys.path.insert(0, str(core_path.parent))
        import checkpoint_archive_cli as cli
        targets = cli.config(str(HERE / 'archive.json'))
        require(len(targets) == 1, 'exactly one archive target required')
        target = targets[0]
        require(target['source_root'] == str(run) and target['destination_root'] == pins['destination_root'],
                'configuration points outside pinned roots')
        require(manifest['bytes'] <= target['max_bytes'] and epoch <= target['max_epoch'],
                'checkpoint exceeds configured pass limits')
        receipt = core.transfer(manifest, pins['destination_root'],
                                rate_limit_bytes=target['rate_limit_bytes'], **cli.storage_options(target))
        publish('epoch_%02d_verified.json' % epoch, receipt)
        result = core.cleanup(manifest, receipt)
        require(result['manifest_id'] == manifest['manifest_id'], 'cleanup result binding changed')
        publish('epoch_%02d_removed.json' % epoch, result)
        print('B6_HEADROOM_ARCHIVE_REMOVED', epoch, manifest['manifest_id'], flush=True)
        return
    verified, removed = {}, set()
    for epoch in (9, 10, 11):
        receipt = json.loads((HERE / ('epoch_%02d_verified.json' % epoch)).read_text())
        manifest = receipt['manifest']
        require(manifest == json.loads((HERE / ('epoch_%02d_manifest.json' % epoch)).read_text()),
                'prepared and published manifests differ')
        require(manifest['source_root'] == str(run) and manifest['epoch'] == epoch, 'wrong archived source/epoch')
        expected_path = core.archive_path(manifest, pins['destination_root'])
        require(receipt['archive_path'] == expected_path, 'wrong archive destination')
        stored = json.loads((Path(expected_path) / core.RECEIPT).read_text())
        require(stored == receipt, 'published receipt changed')
        require(manifest['name'] not in verified, 'duplicate archive receipt')
        verified[manifest['name']] = receipt
        result = json.loads((HERE / ('epoch_%02d_removed.json' % epoch)).read_text())
        require(result['manifest_id'] == receipt['manifest_id'], 'cleanup receipt differs')
        removed.add(result['manifest_id'])
    require(set(verified) == {'epoch_09', 'epoch_10', 'epoch_11'}, 'missing expected verified epoch')
    require(removed == {receipt['manifest_id'] for receipt in verified.values()}, 'cleanup does not match verified archives')
    for name in verified:
        require(not os.path.lexists(str(run / name)), 'archived source still present')
    require(not core.pending_cleanup(str(run)), 'cleanup still pending')
    publish('complete.json', dict(job_id=os.environ['SLURM_JOB_ID'], status='complete', receipts=verified,
                                 archived_bytes=sum(r['manifest']['bytes'] for r in verified.values()),
                                 protected_epochs=pins['protected_epochs'], best=pins['best'],
                                 latest_meta_sha256=pins['latest_meta_sha256']))
    print('B6_HEADROOM_ARCHIVE_COMPLETE', os.environ['SLURM_JOB_ID'])


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else None)
