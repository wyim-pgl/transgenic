"""Tiny actual-Lustre tests; only generated fixture directories are removed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location('archive', Path(sys.argv[1]) / 'revision/scripts/b6/checkpoint_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
ssd = '/work/nvme/bilv/wcyim/transgenic/runs'
hdd = '/work/hdd/bilv/wcyim/transgenic'

def storage(path):
    project = subprocess.check_output(['lfs', 'project', '-d', str(path)], universal_newlines=True).split(None, 2)
    layout = subprocess.check_output(['lfs', 'getstripe', str(path)], universal_newlines=True)
    pools = set(re.findall(r'pool:\s*(\S+)', layout))
    assert pools == {'ddn_hdd'}, pools
    quota = subprocess.check_output(['lfs', 'quota', '-p', project[0], '--pool', 'ddn_hdd', '/work/hdd/bilv'], universal_newlines=True)
    row = [line.split() for line in quota.splitlines() if line.split() and line.split()[0] == '/work/hdd/bilv'][0]
    used, soft, hard = [int(x.rstrip('*')) for x in row[1:4]]
    available = (min(x for x in (soft, hard) if x > 0) - used) * 1024
    return {'pool': 'ddn_hdd', 'project_id': int(project[0]), 'quota_free_bytes': available}

for iteration in range(3):
    with tempfile.TemporaryDirectory(prefix='.archive_fixture_', dir=ssd) as source, tempfile.TemporaryDirectory(prefix='.archive_fixture_', dir=hdd) as target:
        run = Path(source) / 'run with spaces'
        run.mkdir()
        for number in range(1, 5):
            epoch = run / ('epoch_%02d' % number)
            state = epoch / 'accelerate_state'
            state.mkdir(parents=True)
            (epoch / 'meta.json').write_text(json.dumps({'epoch': number, 'global_step': 10 * number, 'seed': 123, 'pending_micro': 0}))
            (epoch / 'eval.json').write_text(json.dumps({'epoch': number, 'global_step': 10 * number}))
            for relative in ['model.safetensors', 'accelerate_state/model.safetensors', 'accelerate_state/optimizer.bin', 'accelerate_state/scheduler.bin', 'accelerate_state/random_states_0.pkl']:
                (epoch / relative).write_bytes((('fixture-%d-%s' % (number, relative)).encode() * 2048)[:32768])
        (run / 'best').symlink_to('epoch_01')
        (run / 'latest_state').mkdir()
        (run / 'latest_state' / 'KEEP').write_text('live state')
        selected = archive.plan(str(run))
        assert [item['epoch'] for item in selected['candidates']] == [2], selected
        manifest = archive.prepare(str(run), 2)
        receipt = archive.transfer(manifest, target, storage_check=storage, required_pool='ddn_hdd', required_project=22169, min_free_bytes=1024**3)
        assert archive.verify_archive(manifest, receipt['archive_path']) == receipt
        assert archive.transfer(manifest, target, storage_check=storage, required_pool='ddn_hdd', required_project=22169, min_free_bytes=1024**3) == receipt
        outcome = archive.cleanup(manifest, receipt)
        assert outcome['status'] == 'removed', outcome
        assert not (run / 'epoch_02').exists()
        assert (run / 'epoch_01' / 'model.safetensors').is_file()
        assert (run / 'epoch_03' / 'model.safetensors').is_file()
        assert (run / 'epoch_04' / 'model.safetensors').is_file()
        assert (run / 'latest_state' / 'KEEP').read_text() == 'live state'
        assert archive.cleanup(manifest, receipt)['status'] == 'removed'
        print(json.dumps({'iteration': iteration + 1, 'result': 'PASS', 'bytes': manifest['bytes'], 'pool': 'ddn_hdd', 'protected_epochs': [1, 3, 4], 'manifest_id': manifest['manifest_id']}), flush=True)
print('ALL_GENERATED_FIXTURES_REMOVED', flush=True)
