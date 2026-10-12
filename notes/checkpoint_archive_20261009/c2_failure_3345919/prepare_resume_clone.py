"""Run on DeltaAI once: preserve the failed run and create a verified update200 recovery copy."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

work = Path('/work/nvme/bilv/wcyim/transgenic')
source = work / 'runs/b6_wp5_pilot_C2'
target = work / 'runs/b6_wp5_pilot_C2_recovery_20261011'
temporary = target.with_name(target.name + '.tmp')
assert not target.exists() and not temporary.exists(), 'inspect existing recovery output before retry'
assert (source / 'TRAINING_FAILED').is_file()
assert not (source / 'TRAINING_DONE').exists()
job = subprocess.check_output(['scontrol', 'show', 'job', '-o', '3345920'], universal_newlines=True)
assert 'JobState=PENDING' in job and 'Reason=JobHeldUser' in job
assert 'UserId=wcyim(' + str(os.getuid()) + ')' in job
active_jobs = subprocess.check_output(['squeue', '-u', 'wcyim', '-h', '-o', '%i'], universal_newlines=True).split()
assert '3345919' not in active_jobs
meta_bytes = (source / 'latest_state/meta.json').read_bytes()
meta = json.loads(meta_bytes.decode())
assert (meta['epoch'], meta['step'], meta['global_step'], meta['pending_micro'], meta['arm']) == (0, 4800, 200, 0, 'C2')
subprocess.check_call(['python3', str(work / 'repo_b6_233574a/revision/scripts/b6/storage_preflight.py'), str(source)])


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


records = {}


def copy_verified(src, dst):
    assert src.is_file() and not src.is_symlink(), str(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    before = src.stat()
    h = hashlib.sha256()
    with src.open('rb') as reader, dst.open('xb') as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b''):
            h.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    after = src.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(src)
    assert sha(dst) == h.hexdigest(), str(dst)
    shutil.copystat(str(src), str(dst))
    records[str(dst.relative_to(temporary))] = dict(source=str(src), bytes=after.st_size, sha256=h.hexdigest())


temporary.mkdir()
for path in sorted((source / 'latest_state').rglob('*')):
    if path.is_file():
        copy_verified(path, temporary / path.relative_to(source))
for name in ('run_config_b6.json', 'runtime_identity.txt', 'data_order.jsonl'):
    copy_verified(source / name, temporary / name)
audit = temporary / 'failed_attempt_3345919'
for name in ('train.err', 'TRAINING_FAILED'):
    copy_verified(source / name, audit / name)
draws = {}
for rank in range(4):
    name = 'rank%02d.tsv' % rank
    src = source / 'draw_log' / name
    copy_verified(src, audit / 'draw_log' / name)
    lines = src.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 4800 and all(line.endswith(b'\n') for line in lines)
    prefix = b''.join(lines[:4800])
    path = temporary / 'draw_log' / name
    path.parent.mkdir(exist_ok=True)
    with path.open('xb') as stream:
        stream.write(prefix)
        stream.flush()
        os.fsync(stream.fileno())
    assert sha(path) == hashlib.sha256(prefix).hexdigest()
    draws[name] = dict(original_rows=len(lines), committed_rows=4800,
                       committed_prefix_sha256=sha(path), failed_tail_rows=len(lines) - 4800)
assert (source / 'latest_state/meta.json').read_bytes() == meta_bytes
report = dict(source=str(source), target=str(target), failed_job=3345919, checkpoint=meta,
              copied_files=records, draw_logs=draws, created_unix=time.time(),
              note='Original unchanged. Full failed logs preserved; recovery logs contain only committed prefixes.')
with (temporary / 'RECOVERY_PROVENANCE.json').open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
for path in [p for p in temporary.rglob('*') if p.is_dir()] + [temporary]:
    descriptor = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
os.rename(str(temporary), str(target))
descriptor = os.open(str(target.parent), os.O_DIRECTORY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print('CLONE_VERIFIED ' + json.dumps(dict(target=str(target), global_step=200,
      copied_files=len(records), copied_bytes=sum(r['bytes'] for r in records.values()), draw_logs=draws)))
