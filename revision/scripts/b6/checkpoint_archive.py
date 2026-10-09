"""Durable, standard-library checkpoint archive transactions (Python 3.6+).

Public API: plan, prepare, prepare_archive, transfer, verify_archive, cleanup,
locked, archive_path.
Manifest/receipt dictionaries can be persisted or sent as JSON. A remote caller
must supply cleanup's verifier(manifest, receipt), independently rereading the
remote archive over an authenticated channel. Receipt assertions alone NEVER
permit removal. The caller is responsible for keeping published archives
immutable and for supplying authoritative pool/project/quota probes on Lustre.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid

VERSION = 1
RECEIPT = '.archive_receipt.json'
CHUNK = 1024 * 1024
EPOCH = re.compile(r'^epoch_([0-9]+)$')
RNG = re.compile(r'^accelerate_state/random_states_([0-9]+)\.pkl$')


class ArchiveError(RuntimeError):
    """Refusal that keeps the source or its recoverable tombstone intact."""


def _canonical(path, create=False):
    path = os.path.abspath(os.fspath(path))
    current = os.path.sep
    for part in path.split(os.path.sep)[1:]:
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            if create:
                os.mkdir(current)
                _sync(os.path.dirname(current))
            else:
                raise ArchiveError('missing path: ' + current)
        mode = os.lstat(current).st_mode
        if not stat.S_ISDIR(mode):
            raise ArchiveError('directory or symlink refused: ' + current)
    return path


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'r') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ArchiveError('not a regular JSON file: ' + path)
        return json.load(stream)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _save(path, value):
    temp = path + '.tmp'
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(_encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _sync(os.path.dirname(path))


@contextlib.contextmanager
def locked(*roots):
    """Nonblocking persistent flock on canonical roots, in canonical order.

    Never unlink a lock file: its inode is the cross-process exclusion domain.
    """
    descriptors = []
    try:
        for root in sorted(set(_canonical(p) for p in roots)):
            fd = os.open(os.path.join(root, '.checkpoint_archive.lock'),
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ArchiveError('invalid lockfile')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ArchiveError('locked or locking unsupported: %s: %s' % (root, exc))
        yield
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _identity(path):
    s = os.lstat(path)
    if not stat.S_ISDIR(s.st_mode):
        raise ArchiveError('unsafe checkpoint directory: ' + path)
    return [s.st_dev, s.st_ino]


def _signature(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _digest(path, signature=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveError('special file refused: ' + path)
        if signature is not None and _signature(before) != signature:
            raise ArchiveError('source changed: ' + path)
        h = hashlib.sha256()
        for data in iter(lambda: stream.read(CHUNK), b''):
            h.update(data)
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ArchiveError('file changed while hashing: ' + path)
        return h.hexdigest()


def _tree(path, hashing=False, ignore_receipt=False):
    _identity(path)
    files, directories = [], []
    for root, dirs, names in os.walk(path, followlinks=False):
        dirs.sort()
        for name in dirs:
            p = os.path.join(root, name)
            _identity(p)
            directories.append(os.path.relpath(p, path))
        for name in sorted(names):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, path)
            s = os.lstat(p)
            if not stat.S_ISREG(s.st_mode):
                raise ArchiveError('symlink/special file refused: ' + p)
            if rel == RECEIPT and ignore_receipt:
                continue
            if rel == RECEIPT:
                raise ArchiveError('reserved archive receipt in raw source')
            row = dict(path=rel, size=s.st_size, signature=_signature(s))
            if hashing:
                row['sha256'] = _digest(p, row['signature'])
            files.append(row)
    return sorted(files, key=lambda f: f['path']), sorted(directories)


def _complete(path, number):
    files, dirs = _tree(path)
    names = {f['path'] for f in files}
    required = {'meta.json', 'eval.json', 'model.safetensors',
                'accelerate_state/model.safetensors',
                'accelerate_state/optimizer.bin', 'accelerate_state/scheduler.bin'}
    if not required.issubset(names):
        raise ArchiveError('incomplete checkpoint core files')
    meta, evaluation = _json(os.path.join(path, 'meta.json')), _json(os.path.join(path, 'eval.json'))
    if not isinstance(meta, dict) or not isinstance(evaluation, dict):
        raise ArchiveError('malformed metadata')
    for field in ('epoch', 'global_step', 'seed', 'pending_micro'):
        if type(meta.get(field)) is not int or meta[field] < 0:
            raise ArchiveError('malformed metadata field: ' + field)
    if meta['epoch'] != number or type(evaluation.get('epoch')) is not int or evaluation['epoch'] != number:
        raise ArchiveError('epoch metadata mismatch')
    if 'global_step' in evaluation and evaluation['global_step'] != meta['global_step']:
        raise ArchiveError('global step mismatch')
    ranks = sorted(int(RNG.match(n).group(1)) for n in names if RNG.match(n))
    if not ranks or ranks != list(range(len(ranks))):
        raise ArchiveError('missing/noncontiguous rank RNG state')
    if meta['pending_micro']:
        for rank in ranks:
            alternatives = {'pending_grads/rank%d.safetensors' % rank,
                            'pending_grads_rank%d.pt' % rank}
            if not alternatives.intersection(names):
                raise ArchiveError('missing pending gradients for rank %d' % rank)
    if any(f['size'] == 0 for f in files if f['path'] in required or RNG.match(f['path'])):
        raise ArchiveError('empty checkpoint core file')
    return files, dirs


def _best(run):
    path = os.path.join(run, 'best')
    if not os.path.lexists(path):
        return None
    if not os.path.islink(path):
        raise ArchiveError('best must be an epoch symlink')
    target = os.readlink(path)
    match = EPOCH.match(target)
    if not match or os.path.basename(target) != target:
        raise ArchiveError('unsafe best target')
    return int(match.group(1))


def _classified_epochs(run):
    """Shared metadata-only definition of completed, distinct epochs."""
    entries = sorted((int(EPOCH.match(n).group(1)), n) for n in os.listdir(run) if EPOCH.match(n))
    numbers = [n for n, name in entries]
    if len(set(numbers)) != len(numbers):
        raise ArchiveError('duplicate epoch numbers')
    completed, refused = [], []
    for epoch, name in entries:
        try:
            files, dirs = _complete(os.path.join(run, name), epoch)
        except (ArchiveError, OSError, ValueError, TypeError) as exc:
            refused.append(dict(epoch=epoch, name=name, reason=str(exc)))
            continue
        completed.append((epoch, name, files, dirs))
    return completed, refused


def _protected_epochs(run, completed, extra=None):
    numbers = {entry[0] for entry in completed}
    if extra is not None:
        numbers.add(extra)  # A validated, receipt-bound renamed source tombstone.
    protected = set(sorted(numbers)[-2:])
    best = _best(run)
    if best is not None:
        protected.add(best)
    return protected


def plan(run_dir, min_age_seconds=0, max_epoch=None, max_bytes=None, now=None):
    """Metadata-only inventory; max_bytes bounds total candidate bytes.

    Retain the latest two completed epochs. Refuse incomplete epochs separately;
    they cannot displace completed epochs from protection. No weights read/hash.
    """
    run = _canonical(run_dir)
    if min_age_seconds < 0 or (max_bytes is not None and max_bytes < 0):
        raise ArchiveError('negative policy limit')
    completed, refused = _classified_epochs(run)
    protected = _protected_epochs(run, completed)
    candidates, total = [], 0
    now = time.time() if now is None else now
    for epoch, name, files, dirs in completed:
        size = sum(f['size'] for f in files)
        newest = max([os.lstat(os.path.join(run, name)).st_mtime] +
                     [f['signature'][3] / 1e9 for f in files])
        if epoch in protected or now - newest < min_age_seconds or (max_epoch is not None and epoch > max_epoch):
            continue
        if max_bytes is not None and total + size > max_bytes:
            continue
        candidates.append(dict(epoch=epoch, name=name, bytes=size))
        total += size
    return dict(version=VERSION, source_root=run, candidates=candidates,
                protected=sorted(protected), refused=refused, bytes=total)


def _eligible(run, epoch, extra=False):
    completed, refused = _classified_epochs(run)
    if epoch in _protected_epochs(run, completed, extra=epoch if extra else None):
        raise ArchiveError('epoch is currently protected')


def _manifest_id(manifest):
    value = dict(manifest)
    value.pop('manifest_id', None)
    return hashlib.sha256(_encode(value)).hexdigest()


def _validate(manifest):
    if manifest.get('version') != VERSION or manifest.get('manifest_id') != _manifest_id(manifest):
        raise ArchiveError('invalid manifest version or digest')
    match = EPOCH.match(manifest.get('name', ''))
    if not match or int(match.group(1)) != manifest.get('epoch'):
        raise ArchiveError('invalid epoch name')
    seen = set()
    for name in manifest['directories'] + [f['path'] for f in manifest['files']]:
        if (not isinstance(name, str) or name.startswith('/') or
                any(p in ('', '.', '..') for p in name.split('/')) or name == RECEIPT or name in seen):
            raise ArchiveError('unsafe or duplicate manifest path')
        seen.add(name)
    return manifest


def prepare(run_dir, epoch):
    """Hash a currently eligible raw epoch and persist its bound manifest."""
    run = _canonical(run_dir)
    with locked(run):
        choices = [c for c in plan(run)['candidates'] if c['epoch'] == epoch]
        if len(choices) != 1:
            raise ArchiveError('epoch is incomplete or protected')
        name = choices[0]['name']
        path = os.path.join(run, name)
        files, dirs = _tree(path, hashing=True)
        manifest = dict(version=VERSION, source_root=run, run_identity=_identity(run),
                        name=name, epoch=epoch, source_identity=_identity(path),
                        files=files, directories=dirs, bytes=sum(f['size'] for f in files))
        manifest['manifest_id'] = _manifest_id(manifest)
        _source(manifest)
        state = _canonical(os.path.join(run, '.checkpoint_archive', 'manifests'), create=True)
        _save(os.path.join(state, manifest['manifest_id'] + '.json'), manifest)
        return manifest


def prepare_archive(archive_path):
    """Prepare a second-tier transfer only from a verified published archive.

    Preserve the full previous receipt as provenance, excluding its reserved
    file from checkpoint payload. Cleanup removes that receipt last.
    """
    path = _canonical(archive_path)
    run = os.path.dirname(path)
    with locked(run, os.path.dirname(run)):
        prior = _json(os.path.join(path, RECEIPT))
        verify_archive(prior['manifest'], path)
        files, dirs = _tree(path, hashing=True, ignore_receipt=True)
        manifest = dict(version=VERSION, source_root=run, run_identity=_identity(run),
                        name=os.path.basename(path), epoch=prior['manifest']['epoch'],
                        source_identity=_identity(path), files=files, directories=dirs,
                        bytes=sum(f['size'] for f in files), upstream_receipt=prior,
                        upstream_signature=_signature(os.lstat(os.path.join(path, RECEIPT))),
                        upstream_lock_root=os.path.dirname(run))
        manifest['manifest_id'] = _manifest_id(manifest)
        _validate(manifest)
        _source(manifest)
        state = _canonical(os.path.join(run, '.checkpoint_archive', 'manifests'), create=True)
        _save(os.path.join(state, manifest['manifest_id'] + '.json'), manifest)
        return manifest


def _protect(manifest, extra=False):
    if 'upstream_receipt' not in manifest:
        _eligible(manifest['source_root'], manifest['epoch'], extra=extra)


def _source(manifest, path=None, partial=False):
    run = _canonical(manifest['source_root'])
    if _identity(run) != manifest['run_identity']:
        raise ArchiveError('source run identity changed')
    path = path or os.path.join(run, manifest['name'])
    if _identity(path) != manifest['source_identity']:
        raise ArchiveError('checkpoint identity changed')
    files, dirs = _tree(path, hashing=True, ignore_receipt='upstream_receipt' in manifest)
    if 'upstream_receipt' in manifest:
        receipt_path = os.path.join(path, RECEIPT)
        if not partial or os.path.lexists(receipt_path):
            if (_signature(os.lstat(receipt_path)) != manifest['upstream_signature'] or
                    _json(receipt_path) != manifest['upstream_receipt']):
                raise ArchiveError('upstream receipt changed')
    expected = {f['path']: f for f in manifest['files']}
    if not partial and (files != manifest['files'] or dirs != manifest['directories']):
        raise ArchiveError('source changed since prepare')
    if partial:
        for row in files:
            if expected.get(row['path']) != row:
                raise ArchiveError('tombstone changed')
        if not set(dirs).issubset(manifest['directories']):
            raise ArchiveError('tombstone directory changed')


def _separate(source, dest):
    common = os.path.commonpath([source, dest])
    if common == source or common == dest:
        raise ArchiveError('overlapping source and destination roots')


def _storage(path, amount, storage_check, required_pool, required_project, min_free_bytes):
    stats = os.statvfs(path)
    free = stats.f_bavail * stats.f_frsize
    evidence = dict(storage_check(path) or {}) if storage_check is not None else {}
    if required_pool is not None and evidence.get('pool') != required_pool:
        raise ArchiveError('destination pool mismatch/unverified')
    if required_project is not None and evidence.get('project_id') != required_project:
        raise ArchiveError('destination project mismatch/unverified')
    if (required_pool is not None or required_project is not None) and 'quota_free_bytes' not in evidence:
        raise ArchiveError('destination quota unverified')
    for key in ('free_bytes', 'quota_free_bytes'):
        if key in evidence:
            free = min(free, evidence[key])
    if free < amount + min_free_bytes:
        raise ArchiveError('insufficient destination free space/quota')
    return evidence


def archive_path(manifest, destination_root):
    """Deterministic target, unique across canonical source runs."""
    _validate(manifest)
    key = hashlib.sha256(manifest['source_root'].encode('utf-8')).hexdigest()[:16]
    return os.path.join(os.path.abspath(destination_root), 'run-' + key, manifest['name'])


def _verify_bytes(manifest, path):
    files, dirs = _tree(path, hashing=True, ignore_receipt=True)
    actual = [(f['path'], f['size'], f['sha256']) for f in files]
    expected = [(f['path'], f['size'], f['sha256']) for f in manifest['files']]
    if actual != expected or dirs != manifest['directories']:
        raise ArchiveError('archive content mismatch')


def _recover_receipt_temp(manifest, stage, target):
    """Preserve interrupted receipt control bytes outside checkpoint payload.

    Ownership comes from the exact private manifest-named stage/control path,
    not from trusting possibly empty or truncated receipt JSON. A new receipt
    is constructed only after the ordinary independent payload verification.
    """
    path = os.path.join(stage, RECEIPT + '.tmp')
    if not os.path.lexists(path):
        return
    if (stage != os.path.join(os.path.dirname(target), '.stage-' + manifest['manifest_id']) or
            any(row['path'] == RECEIPT + '.tmp' for row in manifest['files'])):
        raise ArchiveError('temporary receipt is not private transaction control state')
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ArchiveError('unsafe temporary archive receipt')
    digest = _digest(path, _signature(before))
    control = _canonical(os.path.join(os.path.dirname(stage),
                         '.receipt-recovery-' + manifest['manifest_id']), create=True)
    preserved = os.path.join(control, 'receipt-' + uuid.uuid4().hex)
    # link() exclusively creates a new name: never overwrite unknown control
    # data. Both names stay on the destination filesystem; no data is discarded
    # if the process dies between this link and unlinking the staging name.
    os.link(path, preserved, follow_symlinks=False)
    after = os.lstat(preserved)
    if (not stat.S_ISREG(after.st_mode) or _signature(after)[:4] != _signature(before)[:4] or
            _signature(os.lstat(path)) != _signature(after) or _digest(preserved) != digest):
        raise ArchiveError('temporary archive receipt changed during quarantine')
    fd = os.open(preserved, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync(control)
    os.unlink(path)
    _sync(stage)


def verify_archive(manifest, archive_path):
    """Independently read all destination bytes and validate published receipt."""
    _validate(manifest)
    path = _canonical(archive_path)
    _verify_bytes(manifest, path)
    receipt = _json(os.path.join(path, RECEIPT))
    if (receipt.get('version') != VERSION or receipt.get('manifest') != manifest or
            receipt.get('manifest_id') != manifest['manifest_id'] or receipt.get('archive_path') != path):
        raise ArchiveError('archive receipt mismatch')
    # A prior process may have died immediately after rename but before syncing
    # its parent. Re-establish barriers before issuing a durable acknowledgment.
    for row in manifest['files'] + [dict(path=RECEIPT)]:
        filename = os.path.join(path, row['path'])
        _canonical(os.path.dirname(filename))
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for directory in sorted(manifest['directories'], key=lambda p: p.count('/'), reverse=True):
        _sync(os.path.join(path, directory))
    _sync(path)
    _sync(os.path.dirname(path))
    return receipt


def _copy(source, destination, signature, rate_limit_bytes):
    _canonical(os.path.dirname(source))
    _canonical(os.path.dirname(destination))
    start, copied = time.monotonic(), 0
    src = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(src, 'rb') as read:
        if _signature(os.fstat(read.fileno())) != signature:
            raise ArchiveError('source changed during copy')
        dst = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(dst, 'wb') as write:
            for data in iter(lambda: read.read(CHUNK), b''):
                write.write(data)
                copied += len(data)
                if rate_limit_bytes:
                    delay = copied / float(rate_limit_bytes) - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
            write.flush()
            os.fsync(write.fileno())
        if _signature(os.fstat(read.fileno())) != signature:
            raise ArchiveError('source changed during copy')


def transfer(manifest, destination_root, storage_check=None, required_pool=None,
             required_project=None, min_free_bytes=0, rate_limit_bytes=None):
    """Copy new files, rehash source+destination, fsync receipt and publish.

    storage_check(actual_path) must return pool, project_id, quota_free_bytes
    when these restrictions are requested. It is called on new files too, so
    directory defaults alone cannot masquerade as correct HDD placement.
    Interrupted stages are reused only after checking each completed file.
    """
    _validate(manifest)
    if min_free_bytes < 0 or (rate_limit_bytes is not None and rate_limit_bytes <= 0):
        raise ArchiveError('invalid space/rate limit')
    run = _canonical(manifest['source_root'])
    dest = os.path.abspath(destination_root)
    _separate(run, dest)
    dest = _canonical(dest, create=True)
    roots = [run, dest]
    if 'upstream_lock_root' in manifest:
        roots.append(manifest['upstream_lock_root'])
    with locked(*roots):
        _protect(manifest)
        _source(manifest)
        target = archive_path(manifest, dest)
        parent = _canonical(os.path.dirname(target), create=True)
        if os.path.lexists(target):
            receipt = verify_archive(manifest, target)
            for row in manifest['files']:
                _storage(os.path.join(target, row['path']), 0, storage_check,
                         required_pool, required_project, min_free_bytes)
            return receipt
        evidence = _storage(parent, manifest['bytes'], storage_check, required_pool,
                            required_project, min_free_bytes)
        stage = _canonical(os.path.join(parent, '.stage-' + manifest['manifest_id']), create=True)
        _storage(stage, manifest['bytes'], storage_check, required_pool, required_project, min_free_bytes)
        for directory in manifest['directories']:
            _canonical(os.path.join(stage, directory), create=True)
        for row in manifest['files']:
            path = os.path.join(stage, row['path'])
            if os.path.lexists(path):
                if os.lstat(path).st_size == row['size'] and _digest(path) == row['sha256']:
                    _storage(path, 0, storage_check, required_pool, required_project, min_free_bytes)
                    continue
                # Only this transaction's exact expected regular file is replaceable.
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    raise ArchiveError('unsafe interrupted stage file')
                os.unlink(path)
            _copy(os.path.join(run, manifest['name'], row['path']), path, row['signature'], rate_limit_bytes)
            _storage(path, 0, storage_check, required_pool, required_project, min_free_bytes)
        _recover_receipt_temp(manifest, stage, target)
        _verify_bytes(manifest, stage)
        _source(manifest)
        _protect(manifest)
        return _publish_stage(manifest, stage, target, dest, evidence, bytes_verified=True)


def _publish_stage(manifest, stage, target, dest, evidence, bytes_verified=False):
    """Common independently verified, durable destination-native publication."""
    _recover_receipt_temp(manifest, stage, target)
    if not bytes_verified:
        _verify_bytes(manifest, stage)
    for row in manifest['files']:
        fd = os.open(os.path.join(stage, row['path']), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    receipt = dict(version=VERSION, manifest_id=manifest['manifest_id'], manifest=manifest,
                   archive_path=target, storage=evidence)
    _save(os.path.join(stage, RECEIPT), receipt)
    for directory in sorted(manifest['directories'], key=lambda p: p.count('/'), reverse=True):
        _sync(os.path.join(stage, directory))
    _sync(stage)
    os.rename(stage, target)
    _sync(os.path.dirname(target))
    _sync(dest)
    return verify_archive(manifest, target)


def publish_received(manifest, destination_root, receiver, storage_check=None,
                     required_pool=None, required_project=None, min_free_bytes=0):
    """Receive into a controlled stage under destination lock, verify and publish.

    receiver(stage) is the authenticated transport boundary. It must not remove
    source data. Publication independently verifies every byte, exact topology,
    current storage placement and durability; transport exit status is no proof.
    This API cannot inspect a remote source: its helper must revalidate source
    before publication and again before any opt-in cleanup acknowledgment.
    """
    _validate(manifest)
    if min_free_bytes < 0:
        raise ArchiveError('invalid space limit')
    dest = _canonical(destination_root, create=True)
    with locked(dest):
        target = archive_path(manifest, dest)
        parent = _canonical(os.path.dirname(target), create=True)
        if os.path.lexists(target):
            receipt = verify_archive(manifest, target)
            for row in manifest['files']:
                _storage(os.path.join(target, row['path']), 0, storage_check,
                         required_pool, required_project, min_free_bytes)
            return receipt
        evidence = _storage(parent, manifest['bytes'], storage_check, required_pool,
                            required_project, min_free_bytes)
        stage = _canonical(os.path.join(parent, '.stage-' + manifest['manifest_id']), create=True)
        _storage(stage, manifest['bytes'], storage_check, required_pool, required_project, min_free_bytes)
        _recover_receipt_temp(manifest, stage, target)
        # Refuse unsafe existing stage entries before a transport can follow them.
        files, dirs = _tree(stage, ignore_receipt=True)
        if (not set(f['path'] for f in files).issubset(f['path'] for f in manifest['files']) or
                not set(dirs).issubset(manifest['directories'])):
            raise ArchiveError('unexpected received stage contents')
        # In-place transport may resume only stage-owned payload inodes. A
        # hardlink would otherwise let it overwrite data outside this stage.
        if any(os.lstat(os.path.join(stage, row['path'])).st_nlink != 1 for row in files):
            raise ArchiveError('hardlinked received stage payload refused')
        receiver(stage)
        _verify_bytes(manifest, stage)
        for row in manifest['files']:
            _storage(os.path.join(stage, row['path']), 0, storage_check,
                     required_pool, required_project, min_free_bytes)
        return _publish_stage(manifest, stage, target, dest, evidence, bytes_verified=True)


def pending_cleanup(run_dir):
    """Metadata-only discovery of bound unfinished source cleanup journals."""
    run = _canonical(run_dir)
    state = os.path.join(run, '.checkpoint_archive', 'cleanup')
    if not os.path.lexists(state):
        return []
    _canonical(state)
    result = []
    for name in sorted(os.listdir(state)):
        if not name.endswith('.json'):
            continue  # A private interrupted .tmp is not yet an authorized journal.
        transaction = _json(os.path.join(state, name))
        manifest = _validate(transaction['manifest'])
        receipt = transaction['receipt']
        if (name != manifest['manifest_id'] + '.json' or manifest['source_root'] != run or
                transaction.get('version') != VERSION or
                transaction.get('status') not in ('prepared', 'removed') or
                receipt.get('manifest') != manifest or
                receipt.get('manifest_id') != manifest['manifest_id'] or
                receipt.get('version') != VERSION):
            raise ArchiveError('invalid pending cleanup journal')
        if transaction['status'] != 'removed':
            result.append(transaction)
    return result


def cleanup(manifest, receipt, verifier=None):
    """Opt-in source removal with a durable, resumable source tombstone.

    verifier(manifest, receipt) is required for nonlocal destinations and MUST
    return the exact receipt after independently verifying destination content.
    It is invoked again on every restart before any remaining files are removed.
    """
    _validate(manifest)
    if receipt.get('manifest') != manifest or receipt.get('manifest_id') != manifest['manifest_id']:
        raise ArchiveError('cleanup receipt is not bound to prepared source')
    run = _canonical(manifest['source_root'])
    roots = [run]
    if 'upstream_lock_root' in manifest:
        roots.append(manifest['upstream_lock_root'])
    if verifier is None:
        archive = _canonical(receipt['archive_path'])
        _separate(run, archive)
        roots.append(os.path.dirname(os.path.dirname(archive)))
        verifier = lambda m, r: verify_archive(m, r['archive_path'])
    with locked(*roots):
        if verifier(manifest, receipt) != receipt:
            raise ArchiveError('destination verifier acknowledgment mismatch')
        if _identity(run) != manifest['run_identity']:
            raise ArchiveError('source run identity changed')
        state = _canonical(os.path.join(run, '.checkpoint_archive', 'cleanup'), create=True)
        journal = os.path.join(state, manifest['manifest_id'] + '.json')
        source = os.path.join(run, manifest['name'])
        tomb = os.path.join(run, '.archive-removing-' + manifest['manifest_id'])
        transaction = dict(version=VERSION, manifest=manifest, receipt=receipt, status='prepared')
        if os.path.lexists(journal):
            transaction = _json(journal)
            if transaction.get('manifest') != manifest or transaction.get('receipt') != receipt:
                raise ArchiveError('cleanup transaction mismatch')
        else:
            if not os.path.lexists(source) or os.path.lexists(tomb):
                raise ArchiveError('missing source or unbound tombstone')
            _source(manifest)
            _protect(manifest)
            _save(journal, transaction)
        if transaction['status'] == 'removed':
            if os.path.lexists(source) or os.path.lexists(tomb):
                raise ArchiveError('source replaced after cleanup')
            return dict(status='removed', manifest_id=manifest['manifest_id'])
        _protect(manifest, extra=True)
        if os.path.lexists(source):
            if os.path.lexists(tomb):
                raise ArchiveError('source replaced while tombstone exists')
            _source(manifest)
            _protect(manifest, extra=True)
            os.rename(source, tomb)
            _sync(run)
        if os.path.lexists(tomb):
            _source(manifest, tomb, partial=True)
            for row in manifest['files']:
                path = os.path.join(tomb, row['path'])
                if os.path.lexists(path):
                    _canonical(os.path.dirname(path))
                    if _digest(path, row['signature']) != row['sha256']:
                        raise ArchiveError('tombstone content changed')
                    _protect(manifest, extra=True)
                    os.unlink(path)
                    _sync(os.path.dirname(path))
            if 'upstream_receipt' in manifest:
                path = os.path.join(tomb, RECEIPT)
                if os.path.lexists(path):
                    if (_signature(os.lstat(path)) != manifest['upstream_signature'] or
                            _json(path) != manifest['upstream_receipt']):
                        raise ArchiveError('upstream receipt changed')
                    os.unlink(path)
                    _sync(tomb)
            for directory in sorted(manifest['directories'], key=lambda p: p.count('/'), reverse=True):
                path = os.path.join(tomb, directory)
                if os.path.lexists(path):
                    _identity(path)
                    os.rmdir(path)
                    _sync(os.path.dirname(path))
            os.rmdir(tomb)
            _sync(run)
        transaction['status'] = 'removed'
        _save(journal, transaction)
        return dict(status='removed', manifest_id=manifest['manifest_id'])
