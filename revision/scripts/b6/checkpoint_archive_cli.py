#!/usr/bin/env python3
"""Bounded checkpoint archive plans, local copies and authenticated SSH pulls."""
import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import select
import shlex
import signal
import stat
import subprocess
import sys
import time
import uuid

import checkpoint_archive as core
from storage_preflight import quota_headroom

LIMIT = 64 * 1024 * 1024  # Bound protocol metadata, not checkpoint data.


class CleanupTransportError(core.ArchiveError):
    def __init__(self, message, authorized):
        super().__init__(message)
        self.phase = 'after_ack' if authorized else 'before_ack'
        self.source_state = 'cleanup_outcome_unknown' if authorized else 'retained'



def emit(status, **fields):
    fields.update(status=status)
    print(json.dumps(fields, sort_keys=True), flush=True)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise core.ArchiveError('invalid ' + name)
    return value


def number(value, name, minimum=0, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum or (positive and value == 0):
        raise core.ArchiveError('invalid ' + name)
    return value


def keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise core.ArchiveError('unknown or missing configuration/protocol fields')


def absolute(path):
    if not isinstance(path, str) or not os.path.isabs(path) or '\x00' in path or '\n' in path or '\r' in path:
        raise core.ArchiveError('absolute single-line path required')
    return os.path.normpath(path)


def config(path):
    value = core._json(path)
    keys(value, ('version', 'targets'), ('version', 'targets'))
    if value['version'] != 1 or not isinstance(value['targets'], list) or not value['targets']:
        raise core.ArchiveError('config version1 and targets required')
    names = set()
    for target in value['targets']:
        keys(target, ('name', 'transport', 'source_kind', 'source_root', 'destination_root', 'storage',
                      'min_age_seconds', 'min_free_bytes', 'max_bytes', 'max_checkpoints', 'max_epoch',
                      'rate_limit_bytes', 'remote', 'rsync_timeout'),
                     ('name', 'transport', 'source_kind', 'source_root', 'destination_root', 'storage',
                      'min_age_seconds', 'min_free_bytes', 'max_bytes', 'max_checkpoints'))
        if not isinstance(target['name'], str) or not target['name'] or target['name'] in names:
            raise core.ArchiveError('unique target name required')
        names.add(target['name'])
        if target['transport'] not in ('local', 'ssh-pull') or target['source_kind'] not in ('run', 'archive'):
            raise core.ArchiveError('explicit transport/source_kind required')
        target['source_root'] = absolute(target['source_root'])
        target['destination_root'] = absolute(target['destination_root'])
        for name in ('max_bytes', 'max_checkpoints'):
            integer(target[name], name, 1)
        integer(target['min_free_bytes'], 'min_free_bytes')
        number(target['min_age_seconds'], 'min_age_seconds')
        for name in ('max_epoch', 'rate_limit_bytes'):
            if name in target: integer(target[name], name, 1 if name == 'rate_limit_bytes' else 0)
        storage = target['storage']
        keys(storage, ('kind', 'required_pool', 'required_project', 'mount'), ('kind',))
        if storage['kind'] == 'lustre':
            keys(storage, ('kind', 'required_pool', 'required_project', 'mount'),
                 ('kind', 'required_pool', 'required_project', 'mount'))
            if storage['required_pool'] != 'ddn_hdd':
                raise core.ArchiveError('Lustre archive requires ddn_hdd')
            integer(storage['required_project'], 'required_project', 1)
            storage['mount'] = absolute(storage['mount'])
        elif storage != dict(kind='local'):
            raise core.ArchiveError('unsupported storage policy')
        if target['transport'] == 'ssh-pull':
            remote = target['remote']
            keys(remote, ('host', 'python', 'helper', 'timeout', 'connect_timeout'),
                 ('host', 'python', 'helper', 'timeout', 'connect_timeout'))
            if not isinstance(remote['host'], str) or not re.match(r'^[A-Za-z0-9][A-Za-z0-9_.@-]*$', remote['host']):
                raise core.ArchiveError('invalid SSH host')
            remote['python'] = absolute(remote['python'])
            remote['helper'] = absolute(remote['helper'])
            number(remote['timeout'], 'remote timeout', positive=True)
            integer(remote['connect_timeout'], 'connect_timeout', 1)
            integer(target.get('rsync_timeout', 120), 'rsync_timeout', 1)
        elif 'remote' in target or 'rsync_timeout' in target:
            raise core.ArchiveError('remote fields require ssh-pull')
    return value['targets']


def encoded(value):
    data = core._encode(value) + b'\n'
    if len(data) > LIMIT:
        raise core.ArchiveError('protocol metadata exceeds 64MiB')
    return data


def stop(process):
    # The leader may already have exited while a descendant holds the pipes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None: stream.close()
        if process.poll() is None: process.kill()
        process.wait(timeout=1)


def command(argv, data=None, timeout=120):
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = process.communicate(data, timeout=timeout)
    except subprocess.TimeoutExpired:
        stop(process)
        raise core.ArchiveError('transport/probe timed out')
    if process.returncode:
        # Do not print transport stderr: it can contain credentials/host setup secrets.
        raise core.ArchiveError('transport/probe failed (exit %d)' % process.returncode)
    if len(out) > LIMIT:
        raise core.ArchiveError('protocol response exceeds 64MiB')
    return out


def ssh_argv(target):
    remote = target['remote']
    invocation = ' '.join(shlex.quote(x) for x in (remote['python'], remote['helper'], '--helper'))
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=%d' % remote['connect_timeout'],
            remote['host'], invocation]


def remote_call(target, request):
    out = command(ssh_argv(target), encoded(request), target['remote']['timeout'])
    result = json.loads(out.decode('utf-8'))
    if result.get('status') != 'ok':
        raise core.ArchiveError('source helper refused request: ' + str(result.get('error', 'unknown')))
    return result['result']


def run_root(source_kind, source_root):
    source_root = absolute(source_root)
    return source_root if source_kind == 'run' else os.path.dirname(source_root)


def inventory(source_kind, source_root, age, max_epoch, max_bytes):
    source_root = absolute(source_root)
    root = run_root(source_kind, source_root)
    if not os.path.exists(root):
        return dict(waiting=True, pending=[], plan=None)
    pending = core.pending_cleanup(root)
    pending = [t for t in pending if
               ('upstream_receipt' in t['manifest']) == (source_kind == 'archive') and
               (source_kind == 'run' or
                os.path.join(t['manifest']['source_root'], t['manifest']['name']) == source_root)]
    if not os.path.exists(source_root):
        return dict(waiting=not bool(pending), pending=pending, plan=None)
    if source_kind == 'run':
        planned = core.plan(source_root, min_age_seconds=age, max_epoch=max_epoch, max_bytes=max_bytes)
    else:
        path = core._canonical(source_root)
        receipt = core._json(os.path.join(path, core.RECEIPT))
        manifest = core._validate(receipt['manifest'])
        if (receipt.get('version') != core.VERSION or receipt.get('manifest_id') != manifest['manifest_id'] or
                receipt.get('archive_path') != path or core.archive_path(manifest, os.path.dirname(os.path.dirname(path))) != path):
            raise core.ArchiveError('archive input receipt/path mismatch')
        files, _ = core._tree(path, ignore_receipt=True)
        size = sum(f['size'] for f in files)
        eligible = (time.time() - os.stat(path).st_mtime >= age and size <= max_bytes and
                    (max_epoch is None or manifest['epoch'] <= max_epoch))
        planned = dict(version=1, source_root=root, candidates=[dict(epoch=manifest['epoch'],
                       name=os.path.basename(path), bytes=size)] if eligible else [], protected=[], refused=[], bytes=size if eligible else 0)
    return dict(waiting=False, pending=pending, plan=planned)


def request_for(target, action, **fields):
    fields.update(action=action, source_kind=target['source_kind'], source_root=target['source_root'])
    if target['transport'] == 'ssh-pull': fields['protocol_timeout'] = target['remote']['timeout']
    return fields


def stored_manifest(request):
    root = core._canonical(run_root(request['source_kind'], request['source_root']))
    manifest_id = request['manifest_id']
    if not isinstance(manifest_id, str) or not re.match(r'^[0-9a-f]{64}$', manifest_id):
        raise core.ArchiveError('invalid manifest ID')
    path = os.path.join(root, '.checkpoint_archive', 'manifests', manifest_id + '.json')
    manifest = core._validate(core._json(path))
    if manifest['source_root'] != root or manifest['manifest_id'] != manifest_id:
        raise core.ArchiveError('prepared manifest root mismatch')
    if ('upstream_receipt' in manifest) != (request['source_kind'] == 'archive'):
        raise core.ArchiveError('prepared source kind mismatch')
    if request['source_kind'] == 'archive' and os.path.join(root, manifest['name']) != request['source_root']:
        raise core.ArchiveError('prepared archive path mismatch')
    return manifest


class FrameChannel:
    """Deadline-bounded POSIX frame I/O; retain prefetched bytes on one channel."""
    def __init__(self, reader, writer, timeout):
        number(timeout, 'protocol timeout', positive=True)
        self.reader = reader.fileno() if reader is not None else None
        self.writer = writer.fileno() if writer is not None else None
        self.deadline = time.monotonic() + timeout
        self.buffer = bytearray()
        for fd in (self.reader, self.writer):
            if fd is not None:
                fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    def _wait(self, writing=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise core.ArchiveError('protocol deadline exceeded')
        reads, writes, _ = select.select([] if writing else [self.reader],
                                         [self.writer] if writing else [], [], remaining)
        if not (writes if writing else reads):
            raise core.ArchiveError('protocol deadline exceeded')

    def read_json(self):
        while b'\n' not in self.buffer:
            if len(self.buffer) > LIMIT:
                raise core.ArchiveError('oversized protocol message')
            self._wait()
            try:
                data = os.read(self.reader, min(65536, LIMIT + 1 - len(self.buffer)))
            except BlockingIOError:
                continue
            if not data:
                raise core.ArchiveError('missing protocol message')
            self.buffer.extend(data)
        boundary = self.buffer.index(b'\n') + 1
        if boundary > LIMIT:
            raise core.ArchiveError('oversized protocol message')
        line = bytes(self.buffer[:boundary])
        del self.buffer[:boundary]
        return json.loads(line.decode('utf-8'))

    def write_json(self, value):
        data = encoded(value)
        offset = 0
        while offset < len(data):
            self._wait(writing=True)
            try:
                sent = os.write(self.writer, data[offset:offset + 65536])
            except BlockingIOError:
                continue
            if not sent:
                raise core.ArchiveError('protocol write failed')
            offset += sent


def read_json(stream, timeout=120):
    return FrameChannel(stream, None, timeout).read_json()


def helper():
    channel = FrameChannel(sys.stdin.buffer, sys.stdout.buffer, 120)
    request = channel.read_json()
    protocol_timeout = request.get('protocol_timeout', 120)
    number(protocol_timeout, 'protocol_timeout', positive=True)
    channel.deadline = time.monotonic() + protocol_timeout
    action = request.get('action')
    allowed = {'plan': ('min_age_seconds', 'max_epoch', 'max_bytes'), 'prepare': ('epoch',),
               'validate': ('manifest_id',), 'cleanup': ('manifest_id', 'receipt', 'ack_timeout')}
    if action not in allowed:
        raise core.ArchiveError('unknown helper action')
    keys(request, ('action', 'source_kind', 'source_root', 'protocol_timeout') + allowed[action],
         ('action', 'source_kind', 'source_root') + allowed[action])
    if request['source_kind'] not in ('run', 'archive'):
        raise core.ArchiveError('invalid source_kind')
    request['source_root'] = absolute(request['source_root'])
    if action == 'plan':
        number(request['min_age_seconds'], 'age')
        integer(request['max_bytes'], 'max_bytes', 1)
        if request['max_epoch'] is not None: integer(request['max_epoch'], 'max_epoch')
        result = inventory(request['source_kind'], request['source_root'], request['min_age_seconds'],
                           request['max_epoch'], request['max_bytes'])
    elif action == 'prepare':
        integer(request['epoch'], 'epoch')
        result = (core.prepare(request['source_root'], request['epoch']) if request['source_kind'] == 'run'
                  else core.prepare_archive(request['source_root']))
        if result['epoch'] != request['epoch']:
            raise core.ArchiveError('prepared epoch mismatch')
    else:
        manifest = stored_manifest(request)
        if action == 'validate':
            roots = [manifest['source_root']]
            if 'upstream_lock_root' in manifest: roots.append(manifest['upstream_lock_root'])
            with core.locked(*roots):
                core._source(manifest); core._protect(manifest)
            result = dict(manifest_id=manifest['manifest_id'])
        else:
            receipt = request['receipt']
            number(request['ack_timeout'], 'ack_timeout', positive=True)
            if (receipt.get('version') != core.VERSION or receipt.get('manifest') != manifest or
                    receipt.get('manifest_id') != manifest['manifest_id']):
                raise core.ArchiveError('stale cleanup receipt')
            absolute(receipt['archive_path'])
            def verified(m, r):
                # This live authenticated boundary trusts the coordinator's independent
                # hashing/fsync under destination lock, not a dictionary's assertion.
                nonce = uuid.uuid4().hex
                channel.write_json(dict(status='ack_required', nonce=nonce, manifest_id=m['manifest_id']))
                channel.deadline = min(channel.deadline, time.monotonic() + request['ack_timeout'])
                ack = channel.read_json()
                if ack != dict(nonce=nonce, manifest_id=m['manifest_id'], receipt=r):
                    raise core.ArchiveError('stale or mismatched live acknowledgment')
                return r
            result = core.cleanup(manifest, receipt, verifier=verified)
    channel.write_json(dict(status='ok', result=result))


def storage_options(target):
    storage = target['storage']
    result = dict(min_free_bytes=target['min_free_bytes'], storage_check=None, required_pool=None, required_project=None)
    if storage['kind'] == 'lustre':
        def check(path):
            project = command(['lfs', 'project', '-d', path], timeout=20).decode().strip().split(None, 2)
            if (len(project) != 3 or not project[0].isdigit() or project[2] != path or
                    (os.path.isdir(path) and project[1] != 'P')):
                raise core.ArchiveError('unverified Lustre project/inheritance')
            stripe = ['lfs', 'getstripe'] + (['-d'] if os.path.isdir(path) else []) + [path]
            layout = command(stripe, timeout=20).decode()
            pools = set(re.findall(r'pool:\s*(\S+)', layout))
            if pools != {storage['required_pool']}:
                raise core.ArchiveError('destination Lustre pool mismatch')
            quota = command(['lfs', 'quota', '-p', project[0], '--pool', storage['required_pool'], storage['mount']], timeout=20).decode()
            return dict(pool=next(iter(pools)), project_id=int(project[0]),
                        quota_free_bytes=quota_headroom(quota, storage['mount']))
        result.update(storage_check=check, required_pool=storage['required_pool'],
                      required_project=storage['required_project'])
    return result


@contextlib.contextmanager
def worker(destination):
    root = core._canonical(destination, create=True)
    fd = os.open(os.path.join(root, '.checkpoint_archive.worker.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise core.ArchiveError('unsafe worker lock')
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: raise core.ArchiveError('archive worker busy or locking unsupported')
        yield
    finally:
        os.close(fd)


def pull(target, manifest, options):
    def receive(stage):
        remote = target['remote']
        shell = ' '.join(shlex.quote(x) for x in ['ssh', '-o', 'BatchMode=yes', '-o',
                     'ConnectTimeout=%d' % remote['connect_timeout']])
        # Only the locked, unpublished private stage is writable here. Keep
        # interrupted data at its expected pathname rather than rsync tempnames.
        argv = ['rsync', '-r', '--inplace', '--checksum', '--protect-args', '--no-links', '--no-devices', '--no-specials',
                '--no-perms', '--no-owner', '--no-group', '--omit-dir-times',
                '--exclude=/.archive_receipt.json', '--timeout=%d' % target.get('rsync_timeout', 120), '-e', shell]
        if 'rate_limit_bytes' in target:
            # rsync's integer KiB/s must never round up above the configured rate.
            if target['rate_limit_bytes'] < 1024: raise core.ArchiveError('rsync rate must be at least 1024 bytes/s')
            argv.append('--bwlimit=%d' % (target['rate_limit_bytes'] // 1024))
        argv += ['--', remote['host'] + ':' + os.path.join(manifest['source_root'], manifest['name']) + '/', stage + '/']
        command(argv, timeout=remote['timeout'])
        remote_call(target, request_for(target, 'validate', manifest_id=manifest['manifest_id']))
    return core.publish_received(manifest, target['destination_root'], receive, **options)


def remove_remote(target, manifest, receipt):
    # Hold destination root through independent verification, live ACK and outcome.
    # cleanup on the source only takes source/upstream locks; no recursive flock.
    with core.locked(target['destination_root']):
        if core.verify_archive(manifest, receipt['archive_path']) != receipt:
            raise core.ArchiveError('destination receipt changed')
        request = request_for(target, 'cleanup', manifest_id=manifest['manifest_id'], receipt=receipt,
                              ack_timeout=target['remote']['timeout'])
        process = subprocess.Popen(ssh_argv(target), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        authorized = False
        channel = FrameChannel(process.stdout, process.stdin, target['remote']['timeout'])
        try:
            channel.write_json(request)
            challenge = channel.read_json()
            if (challenge.get('status') != 'ack_required' or
                    challenge.get('manifest_id') != manifest['manifest_id'] or
                    not re.match(r'^[0-9a-f]{32}$', str(challenge.get('nonce', '')))):
                raise core.ArchiveError('source cleanup challenge refused')
            # Any attempted ACK write is conservatively treated as possibly delivered.
            authorized = True
            channel.write_json(dict(nonce=challenge['nonce'], manifest_id=manifest['manifest_id'], receipt=receipt))
            answer = channel.read_json()
            remaining = channel.deadline - time.monotonic()
            if remaining <= 0: raise core.ArchiveError('protocol deadline exceeded')
            out, err = process.communicate(timeout=remaining)
            if process.returncode or out.strip() or channel.buffer.strip():
                raise core.ArchiveError('source cleanup transport failed; inspect/resume journal')
            if answer.get('status') != 'ok' or answer.get('result') != dict(status='removed', manifest_id=manifest['manifest_id']):
                raise core.ArchiveError('source cleanup did not acknowledge removal')
            return answer['result']
        except (core.ArchiveError, OSError, ValueError, subprocess.SubprocessError) as exc:
            stop(process)
            raise CleanupTransportError(str(exc), authorized)
        except BaseException:
            stop(process)
            raise


def pass_target(target, apply, remove):
    policy = dict(min_age_seconds=target['min_age_seconds'], max_epoch=target.get('max_epoch'), max_bytes=target['max_bytes'])
    info = (remote_call(target, request_for(target, 'plan', **policy)) if target['transport'] == 'ssh-pull'
            else inventory(target['source_kind'], target['source_root'], policy['min_age_seconds'], policy['max_epoch'], policy['max_bytes']))
    if info['plan']:
        info['plan']['candidates'] = info['plan']['candidates'][:target['max_checkpoints']]
        info['plan']['bytes'] = sum(c['bytes'] for c in info['plan']['candidates'])
    emit('waiting' if info['waiting'] else 'planned', target=target['name'], plan=info['plan'], pending=info['pending'])
    if not apply or info['waiting']: return
    bytes_left, count_left = target['max_bytes'], target['max_checkpoints']
    candidates = info['plan']['candidates'] if info['plan'] else []
    options = storage_options(target)
    with worker(target['destination_root']):
        if remove:
            for transaction in info['pending']:
                manifest, receipt = transaction['manifest'], transaction['receipt']
                expected = core.archive_path(manifest, target['destination_root'])
                if receipt.get('archive_path') != expected:
                    raise core.ArchiveError('pending transaction destination does not match configured root')
                if manifest['bytes'] > bytes_left or not count_left:
                    emit('deferred', target=target['name'], manifest_id=manifest['manifest_id'], reason='pass budget')
                    continue
                # Recovery rechecks current destination storage before authorizing source removal.
                for row in manifest['files']:
                    core._storage(os.path.join(expected, row['path']), 0, **options)
                result = (remove_remote(target, manifest, receipt) if target['transport'] == 'ssh-pull'
                          else core.cleanup(manifest, receipt))
                emit('removed', target=target['name'], **{k: v for k, v in result.items() if k != 'status'})
                bytes_left -= manifest['bytes']; count_left -= 1
                candidates = [c for c in candidates if c['name'] != manifest['name']]
        for candidate in candidates:
            if not count_left or candidate['bytes'] > bytes_left: continue
            manifest = (remote_call(target, request_for(target, 'prepare', epoch=candidate['epoch'])) if target['transport'] == 'ssh-pull'
                        else core.prepare(target['source_root'], candidate['epoch']) if target['source_kind'] == 'run'
                        else core.prepare_archive(target['source_root']))
            if manifest['bytes'] > bytes_left:
                raise core.ArchiveError('prepared checkpoint exceeds remaining pass budget')
            receipt = (pull(target, manifest, options) if target['transport'] == 'ssh-pull' else
                       core.transfer(manifest, target['destination_root'], rate_limit_bytes=target.get('rate_limit_bytes'), **options))
            emit('verified', target=target['name'], receipt=receipt)
            if remove:
                result = (remove_remote(target, manifest, receipt) if target['transport'] == 'ssh-pull' else core.cleanup(manifest, receipt))
                emit('removed', target=target['name'], manifest_id=result['manifest_id'])
            bytes_left -= manifest['bytes']; count_left -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--target', action='append', help='select a named target; may be repeated')
    parser.add_argument('--helper', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--remove-source', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--cycles', type=int)
    parser.add_argument('--interval', type=float, default=300)
    args = parser.parse_args()
    try:
        if args.helper:
            if args.config or args.target or args.apply or args.remove_source or args.once or args.watch or args.cycles is not None:
                raise core.ArchiveError('helper accepts only stdin protocol')
            helper(); return 0
        if not args.config or (args.remove_source and not args.apply):
            raise core.ArchiveError('--config required; --remove-source requires --apply')
        if args.watch:
            integer(args.cycles, 'positive bounded --cycles', 1)
            number(args.interval, '--interval', positive=True)
        elif args.cycles is not None:
            raise core.ArchiveError('--cycles requires --watch')
        targets = config(args.config)
        if args.target:
            if set(args.target) - set(t['name'] for t in targets):
                raise core.ArchiveError('unknown target selector')
            targets = [t for t in targets if t['name'] in args.target]
        failed = False
        for cycle in range(args.cycles if args.watch else 1):
            for target in targets:
                try: pass_target(target, args.apply, args.remove_source)
                except (core.ArchiveError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    fields = dict(target=target['name'], error=str(exc))
                    if isinstance(exc, CleanupTransportError):
                        fields.update(phase=exc.phase, source_state=exc.source_state)
                    emit('error', **fields); failed = True
            if args.watch and cycle + 1 < args.cycles:
                time.sleep(args.interval)
        return 1 if failed else 0
    except (core.ArchiveError, OSError, ValueError, KeyError, TypeError) as exc:
        emit('error', error=str(exc)); return 1


if __name__ == '__main__':
    sys.exit(main())
