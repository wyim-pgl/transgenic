#!/usr/bin/env python3
"""Controller-only S0/P1, fixed paths, single attempt. Never submits a job."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import admission

BASE_COMMIT = '9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558'
CANDIDATE_ARCHIVE_SHA = 'da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7'
CANDIDATE_MANIFEST_SHA = '4eb7476aa36e77ed8967bc5d6ea73f005deb078b5a5e9b9491976f3d4c79eb0a'
CANDIDATE_LEDGER_SHA = 'd1ae4155fdd31fcb30e8d96f046bf40734b4bad3d65535df2766ba55fc9ae48a'
ROOT = Path('/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545')
WORK = ROOT.parent
SIF = Path('/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif')
CORPUS = WORK / 'db/b5_full_a40_v1.db'
SOURCE = WORK / 'runs/seed123'
REV = '0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446'
MODEL = 'models--LongSafari--hyenadna-large-1m-seqlen-hf'
RESERVE = 40 * 2**30
EVIDENCE_SOURCE = ROOT / 'final-p1-job-3163986/inputs/evidence/fullc0lite.db'
EVIDENCE_BYTES = 1068773376
HF_COPY_MAX = 2000000000
CHECKPOINT_COPY_MAX = 2200000000
STAGING_MAX = 5 * 2**30
PINS = {
    'weights': '6e5e408681dd5356548a8f4b93d28e561c22d885668a4e0015d3e314e28f8d22',
    'run_config': 'c3ca0b4dfc1c9fbedb0cfba4b42887b22d4a62bc5b0c84be6b711ec630a1ef18',
    'runtime_identity': '44b6f578ec50d373d01e764d391b2a559ce4d3012c52388fc160f94fef6f887e',
    'meta': '844ecbc41f19a9287eb0101f9047b3b0100f88d2bfed1a5057261296d86943d7',
    'eval': '0df4e4a7350f3e1cec0884f835aa94f148cbf5042998c25d296e424c5554df4f',
    'corpus': '6840c93fd0b999e09b8dc2897adca1f4dbae36f2ef0f8fcc10ee66371295e222',
    'evidence': 'ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d',
    'sif': '1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6',
    'acceptance': 'faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e',
}
ACCEPT = 'notes/b6_mvp_technical_candidate_20260915.json'
INVENTORY = 'notes/b6_mvp_candidate_inventories_20260915.json'
RECEIPTS = 'notes/b6_mvp_candidate_receipts_20260915.json'
CONFIG = 'configs/b5_400m_win_v3.json'
PANEL = 'revision/protocols/b6_panels_v1.tsv'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) for k in fields), f'changed during hash: {path}')
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')


def pinned(path, digest, size=None):
    p = Path(path)
    require(p.absolute() == p.resolve(strict=True) and p.is_file(), f'not canonical regular: {p}')
    require(size is None or p.stat().st_size == size, f'size mismatch: {p}')
    require(sha(p) == digest, f'SHA256 mismatch: {p}')


def source_identity(path):
    require(path.absolute() == path.resolve(strict=True), 'source path is not canonical')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_uid == admission.UID and
            not s.st_mode & 0o222, 'source must be owner-matched read-only regular file')
    require(not any(os.path.lexists(str(path)+suffix) for suffix in ('-wal', '-shm', '-journal')),
            'source database has adjacent WAL/SHM/journal')
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def recover_evidence(destination):
    """Only the explicitly authorized old read-only file; no alternate origin."""
    source = EVIDENCE_SOURCE
    before = source_identity(source)
    pinned(source, PINS['evidence'], EVIDENCE_BYTES)
    require(source_identity(source) == before, 'recovery source changed before copy')
    fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as out, source.open('rb') as stream:
        require(tuple(getattr(os.fstat(stream.fileno()), k) for k in
                ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')) == before,
                'recovery source changed on open')
        shutil.copyfileobj(stream, out, 8 * 1024 * 1024)
        out.flush(); os.fsync(out.fileno())
    pinned(destination, PINS['evidence'], EVIDENCE_BYTES)
    pinned(source, PINS['evidence'], EVIDENCE_BYTES)
    require(source_identity(source) == before, 'recovery source changed during copy')
    destination.chmod(0o444)
    return dict(source=str(source), destination=str(destination), bytes=EVIDENCE_BYTES,
                sha256=PINS['evidence'], source_before=list(before), source_after=list(source_identity(source)),
                source_preserved=True)


def checkpoint_inventory(source):
    required = {'model.safetensors', 'meta.json', 'eval.json'}
    optional = {'accelerate_state', 'pending_grads'}
    require(source.absolute() == source.resolve(strict=True), 'epoch05 path must be canonical')
    entries = {p.name: p for p in source.iterdir()}
    require(required <= set(entries) <= required | optional, 'epoch05 exact inventory changed')
    for name, path in entries.items():
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) if name in required else stat.S_ISDIR(mode),
                'epoch05 entry type/symlink changed: '+name)
    return sorted(set(entries) - required)


def run(argv):
    print(json.dumps([str(a) for a in argv]), flush=True)
    subprocess.run([str(a) for a in argv], check=True)


def quota_bytes(token, upper):
    # quota -s rounds and does not document SI vs IEC here. Conservative bounds:
    # used upper = IEC + one displayed quantum; limit lower = SI - one quantum.
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([kKMGT])', token)
    require(m is not None, f'unrecognized/over-limit quota token: {token}')
    number, unit = m.groups()
    power = 'KMGT'.index(unit.upper()) + 1
    quantum = 10 ** (-len(number.split('.')[1])) if '.' in number else 1
    return int((float(number) + quantum if upper else float(number) - quantum) * ((1024 if upper else 1000) ** power))


def parse_quota(text):
    rows = [line.split('|')[1:-1] for line in text.splitlines() if line.startswith('|')]
    rows = [[v.strip() for v in row] for row in rows]
    rows = [r for r in rows if r and r[0] == '/work/nvme/bilv']
    require(len(rows) == 1 and len(rows[0]) == 7, 'missing/ambiguous allocation quota row')
    r = rows[0]
    used = quota_bytes(r[1], True)
    limit = min(quota_bytes(r[2], False), quota_bytes(r[3], False))
    require(all(re.fullmatch(r'\d+', x) for x in r[4:]), 'unrecognized file quota units')
    require(min(int(r[5]), int(r[6])) - int(r[4]) > 10000, 'insufficient inode quota reserve')
    require(limit > used, 'quota exhausted')
    return limit - used


def quota_record(destination, parent):
    now = time.time()
    proc = subprocess.run(['/sw/user/scripts/quota', '-s'], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    save(destination.with_suffix('.raw.json'), dict(command=proc.args, measured_at_unix=now,
         exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
    require(proc.returncode == 0 and not proc.stderr.strip(), 'quota command failed/warned; inspect raw record')
    available = min(parse_quota(proc.stdout), shutil.disk_usage(parent).free)
    require(available >= RESERVE + STAGING_MAX, '40GiB reserve plus 5GiB staging envelope unavailable')
    save(destination, dict(path='/mvp/inputs', available_bytes=available, measured_at_unix=now,
         command=proc.args, method='conservative rounded SI-limit/IEC-usage bound; min filesystem free',
         raw_sha256=sha(destination.with_suffix('.raw.json')), reserve_bytes=RESERVE))


def within(path, root):
    return os.path.commonpath([str(path), str(root)]) == str(root)


def copy_cache(source, destination):
    source = source.resolve(strict=True)
    copied = 0
    for layout in ('', 'hub'):
        origin = source / layout / MODEL
        ref = origin / 'refs/main'
        require(within(ref.resolve(), source), 'HF ref escapes source cache')
        require(ref.read_text().strip() == REV, 'HF main is not pinned revision')
        snapshot = origin / 'snapshots' / REV
        require(within(snapshot.resolve(), source), 'HF snapshot escapes source cache')
        target = destination / layout / MODEL
        (target / 'refs').mkdir(parents=True)
        (target / 'refs/main').write_text(REV + '\n')
        files = sorted(snapshot.iterdir())
        expected = ({'model.safetensors', 'config.json', 'configuration_hyena.py', 'modeling_hyena.py'}
                    if layout else {'special_tokens_map.json', 'tokenization_hyena.py', 'tokenizer_config.json'})
        require({p.name for p in files} == expected, f'unreviewed HF snapshot inventory: {snapshot}')
        out = target / 'snapshots' / REV
        out.mkdir(parents=True)
        for p in files:
            require(within(p.resolve(strict=True), origin.resolve()), f'HF link escapes model cache: {p}')
            require(p.is_file(), f'HF entry is not regular: {p}')
            copied += p.stat().st_size
            require(copied <= HF_COPY_MAX, 'HF cache exceeds reviewed staging allowance')
            before = sha(p)
            shutil.copyfile(p, out / p.name)  # dereference only the checked, pinned snapshot
            require(before == sha(p) == sha(out / p.name), f'HF changed during copy: {p}')
        require(ref.read_text().strip() == REV, 'HF ref changed during copy')


def extract_archive(archive, target, files):
    target.mkdir()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        require(len({m.name for m in members}) == len(members), 'duplicate tar member')
        regular = set()
        for m in members:
            p = Path(m.name)
            require(not p.is_absolute() and '..' not in p.parts and (m.isdir() or m.isfile()), 'unsafe tar entry')
            if m.isfile():
                regular.add(m.name)
        require(regular == set(files), 'archive manifest inventory mismatch')
        tar.extractall(target)  # validated: no links, special files, escapes, or duplicate entries
    for name, digest in files.items():
        pinned(target / name, digest)


def freeze(root):
    for p in Path(root).rglob('*'):
        require(not p.is_symlink(), f'unexpected staged symlink: {p}')
        if p.is_file():
            p.chmod(0o444)
    for p in sorted(Path(root).rglob('*'), reverse=True):
        if p.is_dir():
            p.chmod(0o555)
    Path(root).chmod(0o555)


def binding_digest(bindings):
    return hashlib.sha256((json.dumps(bindings, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False) + '\n').encode()).hexdigest()


def validate_prepared(prepared):
    bindings = prepared['bindings']
    require(bindings.get('profile') == 'development-single' and
            type(bindings.get('seed')) is int and bindings['seed'] == 123 and
            bindings.get('run_role') == 'mvp_finetune', 'old/cross-profile prepared receipt rejected')
    require(prepared.get('bindings_sha256') == binding_digest(bindings), 'prepared digest mismatch')


def verify_source_manifest(release, manifest):
    pinned(release / 'source-manifest.json', CANDIDATE_MANIFEST_SHA)
    source = json.loads((release / 'source-manifest.json').read_text())
    require(source['base_commit'] == BASE_COMMIT and manifest.get('source_provenance') == source and
            manifest['source_files'] == source['source_files'] and len(source['source_files']) == 103,
            'wrong candidate source provenance/inventory')
    require('commit' not in manifest, 'dirty snapshot is not a Git commit')
    pinned(release / 'code.candidate.tar', CANDIDATE_ARCHIVE_SHA)


def validate_controller_gate(gate, release_sha, now=None):
    admission.validate_authorization(gate, release_sha,
        os.environ.get('DEPLOY_RELEASE_TRANSPORT_SHA256', ''),
        os.environ.get('DEPLOY_RELEASE_LEDGER_SHA256', ''), now=now)


def controller_gate(release):
    path = ROOT / 'staging/dev123_p1_controller_authorization_v7.json'
    digest = os.environ.get('DEPLOY_CONTROLLER_GATE_SHA256', '')
    require(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'independently pinned controller gate required')
    pinned(path, digest)
    require(path.stat().st_uid == os.getuid(), 'controller gate ownership mismatch')
    text = path.read_text()
    gate = json.loads(text)
    require(text == json.dumps(gate, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n',
            'authorization must be canonical JSON')
    pinned(release / 'release.sha256', os.environ.get('DEPLOY_RELEASE_LEDGER_SHA256', ''))
    validate_controller_gate(gate, sha(release / 'release.json'))
    return gate


def host():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'one-node compute allocation required')
    require(ROOT.resolve(strict=True) == ROOT and ROOT.stat().st_uid == os.getuid(), 'namespace ownership/path mismatch')
    require(SOURCE.resolve(strict=True) == SOURCE, 'source run root must not be an alias')
    release = Path(admission.NAMESPACE)
    manifest = json.loads((release / 'release.json').read_text())
    gate = controller_gate(release)
    # Only a fresh control directory and durable permission are touched before admission.
    # The persistent shell claim is deliberately never removed on failure.
    # Keep the existing namespace intact; each allocation owns one fresh subtree.
    job_id = os.environ['SLURM_JOB_ID']
    require(re.fullmatch(r'[0-9]+', job_id) is not None, 'invalid Slurm job ID')
    job = ROOT / ('dev123-p1-v7-job-' + job_id)
    job.mkdir()  # collision is a blocker, never resume or overwrite
    control = job / 'probe/p1_control'
    control.mkdir(parents=True)
    save(control / 'controller-authorization.snapshot.json', gate)
    admission.live(control, gate, job_id)
    verify_source_manifest(release, manifest)
    for name, digest in manifest['release_files'].items():
        pinned(release / name, digest)
    for name in (*manifest['release_files'], 'release.json', 'release.sha256'):
        destination = job / 'release' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release / name, destination)
    release = job / 'release'
    for name, digest in manifest['release_files'].items():
        pinned(release / name, digest)
    freeze(release)
    for name in ('code', 'inputs', 'cache'):
        (job / name).mkdir()
    scratch = job / 'cache/p1'
    scratch.mkdir()
    for name in ('modules', 'inductor', 'triton', 'tmp', 'home'):
        (scratch / name).mkdir()
    quota_record(control / 'quota-before.json', job / 'inputs')
    extract_archive(release / 'code.candidate.tar', job / 'code/source', manifest['source_files'])
    (job / 'code/source/notes').mkdir()
    require(set(manifest['accepted_companions_not_in_source_commit']) == {ACCEPT, INVENTORY, RECEIPTS},
            'wrong candidate companion inventory')
    for name, digest in manifest['accepted_companions_not_in_source_commit'].items():
        shutil.copyfile(release / name, job / 'code/source' / name)
        pinned(job / 'code/source' / name, digest)
    (job / 'code/source/HFmodels').mkdir()
    freeze(job / 'code/source')
    measured = {}
    for path, key, size in ((SIF, 'sif', 11120398336), (CORPUS, 'corpus', 23534252032),
                           (EVIDENCE_SOURCE, 'evidence', EVIDENCE_BYTES)):
        pinned(path, PINS[key], size)
        measured[key] = dict(path=str(path), sha256=PINS[key], bytes=size, measured_at_unix=time.time())
    save(control / 'large-identities.json', measured)
    # Fixed failure-recovery source is preserved; destination is private and exclusive.
    require(EVIDENCE_BYTES + CHECKPOINT_COPY_MAX + HF_COPY_MAX <= STAGING_MAX, 'staging budget exceeded')
    (job / 'inputs/evidence').mkdir(mode=0o700)
    save(control / 'evidence-recovery.json', recover_evidence(job / 'inputs/evidence/fullc0lite.db'))
    copy_cache(WORK / 'repo_seeds/HFmodels', job / 'inputs/hf')
    freeze(job / 'inputs/hf')
    freeze(job / 'inputs/evidence')
    # Re-measure just before reviewed checkpoint copy; do not reuse reconnaissance quota.
    quota_record(control / 'quota-checkpoint.json', job / 'inputs')
    common = ['/usr/bin/apptainer', 'exec', '--nv', '--writable-tmpfs', '--cleanenv', '--containall',
              '--no-mount', 'hostfs', '--pwd', '/work/repo',
              '--bind', f'{job}/code/source:/work/repo:ro',
              '--bind', f'{job}/inputs/hf:/work/repo/HFmodels:ro',
              '--bind', f'{job}/probe:/mvp/probe:rw',
              '--bind', f'{scratch}:/scratch:rw',
              '--bind', f'{release}:/deployment:ro',
              '--bind', f'{CORPUS}:/corpus.db:ro', '--bind', f'{SIF}:/image.sif:ro']
    envs = dict(PYTHONPATH='/work/repo/src:/work/repo/revision/scripts/mvp', PYTHONDONTWRITEBYTECODE='1',
                PYTHONNOUSERSITE='1', HF_HOME='/work/repo/HFmodels', HF_HUB_CACHE='/work/repo/HFmodels/hub',
                HF_MODULES_CACHE='/scratch/modules', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_DATASETS_OFFLINE='1', TORCHINDUCTOR_CACHE_DIR='/scratch/inductor', TRITON_CACHE_DIR='/scratch/triton',
                TMPDIR='/scratch/tmp', HOME='/scratch/home', XDG_CACHE_HOME='/scratch', OMP_NUM_THREADS='8',
                TRITON_LIBCUDA_PATH='/.singularity.d/libs', NCCL_DEBUG='WARN',
                SLURM_JOB_ID=os.environ['SLURM_JOB_ID'], CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key, value in envs.items():
        common += ['--env', key + '=' + value]
    run(common + ['--bind', f'{job}/inputs:/mvp/inputs:rw', '--bind', f'{SOURCE}:/source/seed123:ro',
                  SIF, 'python', '/deployment/deploy.py', 'container-stage'])
    freeze(job / 'inputs')
    run(common + ['--bind', f'{job}/inputs:/mvp/inputs:ro', SIF, 'python', '/deployment/deploy.py', 'container-p1'])


def container_stage():
    sys.path.insert(0, '/work/repo/revision/scripts/mvp')
    import make_launch_readiness as ready
    base = Path('/work/repo')
    control = Path('/mvp/probe/p1_control')
    source = Path('/source/seed123/epoch_05')
    excluded = checkpoint_inventory(source)
    before = {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
              for p in source.iterdir() if p.name not in excluded}
    for p in (source / 'config.json', source / 'runtime.json', source / 'runtime_identity.txt', source.parent / 'config.json'):
        require(not os.path.lexists(p), f'expected absent sidecar: {p}')
    pinned(source / 'meta.json', PINS['meta'])
    pinned(source / 'eval.json', PINS['eval'])
    checkpoint = Path('/mvp/inputs/b5_seed123_epoch05/checkpoint')
    pins = []
    for prefix in ('weights', 'run_config', 'runtime_identity'):
        pins += ['--expected-' + prefix.replace('_', '-') + '-sha256', PINS[prefix]]
    run([sys.executable, base / 'revision/scripts/mvp/stage_checkpoint.py', '--source-epoch', source,
         '--dest', checkpoint.parent, *pins, '--quota-record', control / 'quota-checkpoint.json',
         '--quota-sha256', sha(control / 'quota-checkpoint.json'), '--max-copy-bytes', CHECKPOINT_COPY_MAX,
         '--reserve-bytes', RESERVE])
    require(checkpoint_inventory(source) == excluded, 'epoch05 inventory changed during staging')
    after = {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
             for p in source.iterdir() if p.name not in excluded}
    require(before == after, 'epoch05 files changed during staging')
    pinned(source / 'meta.json', PINS['meta'])
    pinned(source / 'eval.json', PINS['eval'])
    require(checkpoint_inventory(checkpoint) == [], 'resume state must never be staged')
    save(control / 'checkpoint-weights-only.json', dict(excluded_resume_directories=excluded,
         policy='weights-only initialization; no resume state read, copied, or deleted'))
    pinned(checkpoint / 'meta.json', PINS['meta'])
    pinned(checkpoint / 'eval.json', PINS['eval'])
    require(ready.optional_sidecars(checkpoint) == {'config.json': None, 'runtime.json': None,
            'meta.json': {'sha256': PINS['meta'], 'bytes': 132}}, 'staged optional sidecar inventory mismatch')
    # PREPARE invokes the actual consumer closure and installed package metadata in this SIF.
    run([sys.executable, base / 'revision/scripts/mvp/make_launch_readiness.py', 'prepare',
         '--db', '/corpus.db', '--evidence-db', '/mvp/inputs/evidence/fullc0lite.db',
         '--panel', base / PANEL, '--sif', '/image.sif', '--config', base / CONFIG,
         '--init-from', checkpoint, '--acceptance-record', base / ACCEPT,
         '--acceptance-sha256', PINS['acceptance'], '--inventory-record', base / INVENTORY,
         '--receipts-record', base / RECEIPTS, '--profile', 'development-single', '--seed', 123, *pins,
         '--output', control / 'prepared.json', '--manifest-output', control / 'consumed-code.json'])
    validate_prepared(json.loads((control / 'prepared.json').read_text()))
    from real_probes import tree_identity
    save(control / 'assets.json', {'roots': {p: tree_identity(p) for p in
         ('/work/repo/HFmodels', '/work/repo/HFmodels/hub')}})
    select_tiles(control)
    names = ('prepared.json', 'consumed-code.json', 'assets.json', 'tiles.json', 'tile-availability.json')
    save(control / 's0-inputs.json', {name: sha(control / name) for name in names})
    for name in (*names, 's0-inputs.json'):
        (control / name).chmod(0o444)


def pair_candidates(records):
    groups = {}
    for r in records:
        require(type(r['is_rc']) is bool, 'nonboolean strand')
        key = (r['chromosome'], r['start0'], r['end0'])
        group = groups.setdefault(key, {})
        require(r['is_rc'] not in group, 'duplicate forward/RC interval')
        group[r['is_rc']] = r['rn']
    return {str(t): [list((g[False], g[True])) for key, g in sorted(groups.items())
                    if key[2] - key[1] == t and set(g) == {False, True}]
            for t in (30720, 61440, 129024)}


def select_tiles(control):
    import torch
    from types import SimpleNamespace
    from transgenic.datasets.datasets import isoformDataHyena
    from transgenic.training.mvp_cli_runtime import _close_dataset
    from real_probes import check_tracks, equal_tensors, measured_batches
    inventory = json.loads((Path('/work/repo') / INVENTORY).read_text())
    records = [r for r in inventory['records'] if r['split'] == 'train']
    pairs = pair_candidates(records)
    report = {'candidate_pairs': {k: len(v) for k, v in pairs.items()}, 'attempted': {}, 'selected': {},
              'inventory_sha256': sha(Path('/work/repo') / INVENTORY), 'status': 'searching'}
    ds = None
    try:
        ds = isoformDataHyena('/corpus.db', mode='train', split='train', strict=True, species='Athaliana',
             evidence_db='/mvp/inputs/evidence/fullc0lite.db', evidence_k=8, aug_seed=20260915,
             eval_fraction=1, return_tile_ids=True, gff_vocab_version='v3')
        lookup = {rn: i for i, rn in enumerate(ds._index_map)}
        require(set(lookup) == {r['rn'] for r in records}, 'accepted/train dataset inventory mismatch')
        con = ds._get_connection()
        expected_rows = {r['rn']: r for r in records}
        # One bounded species/split scan, not thousands of repeated sequence scans.
        stats = {}
        for rn, chrom, start, end, split, rc, length, has_n in con.execute(
                "SELECT rn,chromosome,start,fin,split,is_rc,length(sequence),contains(upper(sequence),'N') "
                "FROM geneList WHERE species_id='Athaliana' AND split='train' AND gff IS NOT NULL "
                "AND COALESCE(train_weight,1.0)>0").fetchall():
            exp = expected_rows[rn]
            require([chrom, start, end, split, rc] == [exp['chromosome'], exp['start0'], exp['end0'], exp['split'], exp['is_rc']],
                    'accepted corpus interval inventory mismatch')
            stats[rn] = (length, has_n)
        have_n = False
        complement = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
        for tier, candidates in pairs.items():
            # Search every full twin if necessary, N-containing candidates first; no crops or made-up RNs.
            annotated = []
            for pair in candidates:
                if all(stats[rn][0] == int(tier) for rn in pair):
                    annotated.append((any(stats[rn][1] for rn in pair), pair))
            annotated.sort(key=lambda r: (not r[0], r[1]))
            report['attempted'][tier] = 0
            for n_present, pair in annotated:
                report['attempted'][tier] += 1
                rows = [con.execute('SELECT sequence,is_rc,chromosome,start,fin FROM geneList WHERE rn=?', [rn]).fetchone() for rn in pair]
                for rn, row in zip(pair, rows):
                    exp = expected_rows[rn]
                    require(list(row[1:]) == [exp['is_rc'], exp['chromosome'], exp['start0'], exp['end0']], 'accepted interval identity mismatch')
                samples = [ds[lookup[rn]] for rn in pair]
                if not all(s[9]['has_evidence'] and torch.count_nonzero(s[8][:, :7]).item() for s in samples):
                    continue
                for row, sample in zip(rows, samples):
                    require(sample[0].shape == (1, int(tier)), 'tokenizer cropped full tier')
                    check_tracks(row[0], sample[8])
                require(rows[0][0].translate(complement)[::-1] == rows[1][0], 'RC sequence mismatch')
                equal_tensors(samples[0][8].flip(0)[:, [1, 0, 2, 5, 6, 3, 4, 7]], samples[1][8])
                report['selected'][tier] = pair
                have_n |= n_present
                break
            require(tier in report['selected'], f'no full nonzero-EST forward/RC twin for tier {tier}')
        require(have_n, 'no selected actual twin exercises N masking')
        save(control / 'tiles.json', report['selected'])
        measured_batches(SimpleNamespace(tile_rns=str(control / 'tiles.json'), tile_rns_sha256=sha(control / 'tiles.json')), ds)
        report['status'] = 'available_and_actual_measured_batches_checked'
        report['tiles_sha256'] = sha(control / 'tiles.json')
    except BaseException as error:
        report['status'] = 'BLOCKED'
        report['blocker'] = repr(error)
        raise
    finally:
        save(control / 'tile-availability.json', report)
        if ds is not None:
            _close_dataset(ds)


def container_p1():
    control = Path('/mvp/probe/p1_control')
    stage_pins = json.loads((control / 's0-inputs.json').read_text())
    require(set(stage_pins) == {'prepared.json', 'consumed-code.json', 'assets.json', 'tiles.json', 'tile-availability.json'},
            'incomplete S0 input pins')
    for name, digest in stage_pins.items():
        pinned(control / name, digest)
    require(json.loads((control / 'tile-availability.json').read_text())['status'] ==
            'available_and_actual_measured_batches_checked', 'tile selection blocked')
    validate_prepared(json.loads((control / 'prepared.json').read_text()))
    # Fresh compile scratch remains empty: S0 did CPU data checks, not a GPU warmup.
    require(not any(Path('/scratch/inductor').iterdir()) and not any(Path('/scratch/triton').iterdir()),
            'compiler scratch unexpectedly populated before P1')
    argv = [sys.executable, '/work/repo/revision/scripts/mvp/real_probes.py', '--mode', 'real_tokenizer',
            '--bindings', str(control / 'prepared.json'), '--bindings-file-sha256', sha(control / 'prepared.json'),
            '--assets-manifest', str(control / 'assets.json'), '--assets-sha256', sha(control / 'assets.json'),
            '--db', '/corpus.db', '--evidence-db', '/mvp/inputs/evidence/fullc0lite.db',
            '--config', '/work/repo/' + CONFIG, '--tile-rns', str(control / 'tiles.json'),
            '--tile-rns-sha256', sha(control / 'tiles.json'), '--scratch-root', '/scratch']
    save(control / 'p1.argv.json', argv)
    run([sys.executable, '/work/repo/revision/scripts/mvp/run_bounded.py', '--name', 'real_tokenizer',
         '--bindings', control / 'prepared.json', '--bindings-file-sha256', sha(control / 'prepared.json'),
         '--command-file', control / 'p1.argv.json', '--command-sha256', sha(control / 'p1.argv.json'),
         '--output-dir', '/mvp/probe/real_tokenizer', '--timeout-seconds', 900])


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('host', 'container-stage', 'container-p1'), 'invalid phase')
    {'host': host, 'container-stage': container_stage, 'container-p1': container_p1}[sys.argv[1]]()
