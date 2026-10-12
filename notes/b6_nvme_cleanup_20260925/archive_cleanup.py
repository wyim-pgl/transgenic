"""Two-phase archive and verified deletion of four explicitly approved, terminal B6 run trees."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

SOURCE = Path('/work/nvme/bilv/wcyim/transgenic/runs')
ARCHIVE = Path('/work/hdd/bilv/wcyim/transgenic/archives/b6_nvme_20260925')
ROOTS = ('b6_int4/run_3207149', 'b6_int4/run_3207331', 'b6_int4/run_3208442',
         'b6_resume_diagnostic/job_3207863')
JOBS = ('3207149', '3207331', '3208442', '3207863')


def command(*args):
    return subprocess.check_output(list(args), universal_newlines=True, timeout=30)


def idle():
    queue = command('squeue', '-u', 'wcyim', '-h', '-o', '%i %j %T %Z')
    if any('b6-' in line or 'b6_' in line or str(SOURCE) in line for line in queue.splitlines()):
        raise RuntimeError('active/queued B6 work or a scoped source workdir: ' + queue)
    rows = command('sacct', '-j', ','.join(JOBS), '-X', '-n', '-P', '--format=JobID,State')
    states = dict(line.strip().split('|')[:2] for line in rows.splitlines() if line.strip())
    if set(states) != set(JOBS) or any(states[j] not in ('FAILED', 'COMPLETED') for j in JOBS):
        raise RuntimeError('source jobs not confirmed terminal: ' + rows)
    return dict(queue=queue, source_jobs=states)


def stamp(s):
    return [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink]


def inventory(base, hashes):
    contents, signatures = {}, {}
    for rel in ROOTS:
        root = base / rel
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise RuntimeError('non-canonical source/archive root: ' + str(root))
        paths = [root]
        for directory, dirs, files in os.walk(str(root), followlinks=False):
            paths.extend(Path(directory) / name for name in dirs + files)
        for p in sorted(paths):
            key = str(p.relative_to(base))
            before = p.lstat()
            record = dict(mode=stat.S_IMODE(before.st_mode))
            if stat.S_ISLNK(before.st_mode):
                record.update(kind='symlink', target=os.readlink(str(p)))
            elif stat.S_ISDIR(before.st_mode):
                record.update(kind='directory')
            elif stat.S_ISREG(before.st_mode):
                record.update(kind='file', size=before.st_size)
                if hashes:
                    h = hashlib.sha256()
                    with p.open('rb') as f:
                        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                            h.update(chunk)
                    record['sha256'] = h.hexdigest()
            else:
                raise RuntimeError('special file refused: ' + str(p))
            if stamp(p.lstat()) != stamp(before):
                raise RuntimeError('entry changed during inventory: ' + str(p))
            contents[key], signatures[key] = record, stamp(before)
    return contents, signatures


def store(name, value):
    with (ARCHIVE / name).open('x') as f:
        json.dump(value, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def copy_verify():
    observation = idle()
    # Refuse scope drift, including unexpected additional run directories.
    expected = {str(SOURCE / rel) for rel in ROOTS}
    actual = {str(p) for parent in ('b6_int4', 'b6_resume_diagnostic') for p in (SOURCE / parent).iterdir()}
    if actual != expected:
        raise RuntimeError('source scope differs from the four approved trees')
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(mode=0o700)  # Never reuse/overwrite an earlier archive attempt.
    payload = ARCHIVE / 'payload'
    payload.mkdir()
    import importlib.util
    spec = importlib.util.spec_from_file_location('preflight', '/work/hdd/bilv/wcyim/transgenic/repo_b6_filestore_20260924/revision/scripts/b6/storage_preflight.py')
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    space = preflight.check_storage(ARCHIVE)
    metadata, _ = inventory(SOURCE, False)
    logical_bytes = sum(r.get('size', 0) for r in metadata.values())
    if min(space['quota_remaining_bytes'], space['filesystem_available_bytes']) < logical_bytes + 50 * 1024 ** 3:
        raise RuntimeError('insufficient HDD headroom for archive plus 50GiB reserve')
    store('preflight.json', dict(observation=observation, space=space, logical_bytes=logical_bytes, roots=ROOTS))
    print('HASH_SOURCE', logical_bytes, flush=True)
    source, source_stamps = inventory(SOURCE, True)
    store('source_manifest.json', source)
    store('source_stamps.json', source_stamps)
    print('COPY', flush=True)
    subprocess.check_call(['rsync', '-aH', '--relative', '--'] +
                          [str(SOURCE) + '/./' + rel for rel in ROOTS] + [str(payload) + '/'])
    print('HASH_ARCHIVE', flush=True)
    copied, archive_stamps = inventory(payload, True)
    if source != copied:
        raise RuntimeError('archive content/type/mode/link manifest mismatch; originals retained')
    if inventory(SOURCE, False)[1] != source_stamps:
        raise RuntimeError('source changed during copy; originals retained')
    if inventory(payload, False)[1] != archive_stamps:
        raise RuntimeError('archive changed during verification; originals retained')
    store('archive_manifest.json', copied)
    store('archive_stamps.json', archive_stamps)
    store('VERIFIED.json', dict(files=sum(r['kind'] == 'file' for r in source.values()),
                               entries=len(source), logical_bytes=logical_bytes, roots=ROOTS))
    print('VERIFIED', logical_bytes, len(source), flush=True)


def hardlink_groups(manifest, signatures):
    groups = {}
    for key, record in manifest.items():
        if record['kind'] == 'file':
            groups.setdefault(tuple(signatures[key][:2]), []).append(key)
    return sorted(sorted(paths) for paths in groups.values() if len(paths) > 1)


def durable_archive():
    # Flush the verified data and directory entries before removing the only NVMe copy.
    directories = []
    for directory, dirs, files in os.walk(str(ARCHIVE), followlinks=False):
        directories.append(Path(directory))
        for name in files:
            p = Path(directory) / name
            if p.is_symlink():
                continue
            with p.open('rb') as f:
                os.fsync(f.fileno())
    for p in sorted(directories, key=lambda x: len(x.parts), reverse=True) + [ARCHIVE.parent]:
        fd = os.open(str(p), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def delete_verified():
    if ARCHIVE.is_symlink() or ARCHIVE.resolve() != ARCHIVE:
        raise RuntimeError('archive root is not canonical')
    receipt = json.loads((ARCHIVE / 'VERIFIED.json').read_text())
    if receipt['roots'] != list(ROOTS):
        raise RuntimeError('archive receipt has different roots')
    source = json.loads((ARCHIVE / 'source_manifest.json').read_text())
    copied = json.loads((ARCHIVE / 'archive_manifest.json').read_text())
    if source != copied:
        raise RuntimeError('persisted manifests differ')
    source_stamps = json.loads((ARCHIVE / 'source_stamps.json').read_text())
    archive_stamps = json.loads((ARCHIVE / 'archive_stamps.json').read_text())
    if hardlink_groups(source, source_stamps) != hardlink_groups(copied, archive_stamps):
        raise RuntimeError('hardlink topology within approved roots differs')
    durable_archive()
    if inventory(SOURCE, False)[1] != source_stamps:
        raise RuntimeError('source changed after verification')
    if inventory(ARCHIVE / 'payload', False)[1] != archive_stamps:
        raise RuntimeError('archive changed after verification')
    observation = idle()
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError('fd-based symlink-safe deletion unavailable')
    private = SOURCE / '.b6_verified_delete_20260925'
    if private.exists() or private.is_symlink():
        raise RuntimeError('prior deletion staging exists; reconcile instead of repeating')
    store('DELETE_STARTED.json', observation)  # Exclusive journal prevents blind repeat deletion.
    durable_archive()
    private.mkdir(mode=0o700)
    for rel in ROOTS:
        p = SOURCE / rel
        if p.is_symlink() or p.resolve() != p or stamp(p.lstat()) != source_stamps[rel]:
            raise RuntimeError('source root changed before deletion')
        parked = private / rel.replace('/', '__')
        os.rename(str(p), str(parked))
        # Rename may update ctime only. Check identity after the atomic move into our private directory.
        before, after = source_stamps[rel], stamp(parked.lstat())
        if before[:5] + before[6:] != after[:5] + after[6:]:
            raise RuntimeError('renamed root identity differs; retained without deleting: ' + str(parked))
        shutil.rmtree(str(parked))
        print('REMOVED_VERIFIED_SOURCE', str(p), flush=True)
    private.rmdir()
    store('DELETED.json', dict(roots=ROOTS, archive=str(ARCHIVE / 'payload')))
    print('DELETION_COMPLETE', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['copy-verify']:
        copy_verify()
    elif sys.argv[1:] == ['delete-verified']:
        delete_verified()
    else:
        raise SystemExit('usage: archive_cleanup.py copy-verify|delete-verified')
