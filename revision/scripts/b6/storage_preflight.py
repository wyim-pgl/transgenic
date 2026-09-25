"""DeltaAI CP4 result-pool headroom checks, compatible with host Python 3.6."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MIN_FREE_BYTES = 100 * 1024 ** 3  # CP4's two checkpoint histories plus temporary publication files.


def quota_headroom(text, mount):
    rows = [line.split() for line in text.splitlines() if line.split() and line.split()[0] == mount]
    if len(rows) != 1 or len(rows[0]) < 4:
        raise ValueError('missing or ambiguous project pool quota')
    values = [field.rstrip('*') for field in rows[0][1:4]]
    if not all(value.isdigit() for value in values):
        raise ValueError('non-numeric project pool quota')
    used, soft, hard = map(int, values)
    limits = [value for value in (soft, hard) if value > 0]
    if not limits:
        raise ValueError('no bounded project pool quota reported')
    return max(0, min(limits) - used) * 1024


def check_storage(root):
    root = Path(root).resolve(strict=True)
    mounts = {'/work/nvme/bilv': 'ddn_ssd', '/work/hdd/bilv': 'ddn_hdd'}
    matches = [mount for mount in mounts if str(root).startswith(mount + '/')]
    if len(matches) != 1:
        raise ValueError('CP4 results must be below a known bilv work pool')
    mount = matches[0]

    def command(*args):
        return subprocess.check_output(list(args), universal_newlines=True, timeout=20)

    project = command('lfs', 'project', '-d', str(root)).strip().split(None, 2)
    if len(project) != 3 or not project[0].isdigit() or project[1] != 'P' or project[2] != str(root):
        raise ValueError('results directory must inherit its Lustre project ID')
    layout = command('lfs', 'getstripe', '-d', str(root))
    if set(re.findall(r'pool:\s*(\S+)', layout)) != {mounts[mount]}:
        raise ValueError('results directory stripe pool does not match its quota pool')
    quota = command('lfs', 'quota', '-p', project[0], '--pool', mounts[mount], mount)
    remaining = quota_headroom(quota, mount)
    fs = os.statvfs(str(root))
    available = fs.f_bavail * fs.f_frsize
    if min(remaining, available) < MIN_FREE_BYTES:
        raise RuntimeError('insufficient CP4 storage: quota_remaining={}, filesystem_available={}, required={}'
                           .format(remaining, available, MIN_FREE_BYTES))
    return dict(results_dir=str(root), pool=mounts[mount], project_id=int(project[0]),
                quota_remaining_bytes=remaining, filesystem_available_bytes=available,
                required_bytes=MIN_FREE_BYTES)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: storage_preflight.py RESULTS_DIR')
    print(json.dumps(check_storage(sys.argv[1]), sort_keys=True))
