"""One authorized bounded production transfer; all payload operations use reviewed CLI."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

run = Path(sys.argv[1]).resolve(strict=True)
python = '/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python'
tool = Path('/data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive/3b4eb0d')
source = '/work/nvme/bilv/wcyim/transgenic/runs/seed123'
archive = Path('/data/gpfs/assoc/pgl/data/Transgenic/archives/checkpoint_archive')

def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(name, value):
    temp = run / (name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temp), str(run / name))

lock = (run.parent / '.b5-first8.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
command = [python, '-u', str(tool / 'revision/scripts/b6/checkpoint_archive_cli.py'),
           '--config', str(run / 'config.json'), '--target', 'pronghorn-b5',
           '--apply', '--remove-source', '--watch', '--cycles', '8', '--interval', '300']
state = dict(state='running', started_at=timestamp(), pid=os.getpid(), host=socket.gethostname(),
             command=command, source_root=source, epochs=list(range(1, 9)),
             destination_root=str(archive), code_commit=(tool / 'SOURCE_COMMIT').read_text().strip())
save('status.json', state)
with (run / 'transfer.jsonl').open('a') as out, (run / 'transfer.stderr').open('a') as err:
    result = subprocess.run(command, stdout=out, stderr=err)
state.update(finished_at=timestamp(), returncode=result.returncode)
verified = {}; removed = set(); errors = []
for line in (run / 'transfer.jsonl').read_text().splitlines():
    entry = json.loads(line)
    if entry.get('status') == 'verified':
        receipt = entry['receipt']; manifest = receipt['manifest']
        if manifest.get('source_root') != source or manifest.get('epoch') not in range(1, 9):
            errors.append('unexpected manifest source or epoch')
        verified[manifest['epoch']] = receipt
    elif entry.get('status') == 'removed':
        removed.add(entry['manifest_id'])
    elif entry.get('status') == 'error':
        errors.append(entry)
complete = result.returncode == 0 and sorted(verified) == list(range(1, 9)) and not errors
for epoch, receipt in verified.items():
    complete = complete and receipt['manifest_id'] in removed
    complete = complete and (Path(receipt['archive_path']) / '.archive_receipt.json').is_file()
state.update(state='complete' if complete else 'needs_attention', errors=errors,
             verified_epochs=sorted(verified), removed_count=len(removed),
             archived_payload_bytes=sum(r['manifest']['bytes'] for r in verified.values()),
             archive_paths={str(n):r['archive_path'] for n,r in verified.items()})
save('status.json', state)
print(json.dumps(state, sort_keys=True), flush=True)
raise SystemExit(0 if complete else 1)
