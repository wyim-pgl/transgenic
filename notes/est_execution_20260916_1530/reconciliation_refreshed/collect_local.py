"""Operator capture utility. Remote actions are read-only; do not rerun blindly."""
import base64, datetime, hashlib, importlib.util, json, pathlib, subprocess, sys
sys.dont_write_bytecode = True
OUT = pathlib.Path(__file__).resolve().parent
CODE = pathlib.Path('/tmp/b6-p2-phase-1535433/revision/scripts/evidence/est_blastx_audit.py')
EXPECTED = 'f6bef565f0bb432256b7e596a81b896afd2266c5'
assert subprocess.check_output(['git', '-C', str(CODE.parents[3]), 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED
spec = importlib.util.spec_from_file_location('reviewed_auditor', CODE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
root = a.HISTORICAL_ROOT
allowed = a.reconciliation_paths(root)
assert len(allowed) == 19

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def ssh(label, command, stdin=None):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'pronghorn', command], input=stdin, capture_output=True, text=True, timeout=120)
    save(label + '.transport.json', dict(started_at=start, captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), argv=['ssh', '-o', 'BatchMode=yes', 'pronghorn', command], returncode=p.returncode, stderr=p.stderr))
    (OUT / (label + '.stdout.txt')).write_text(p.stdout)
    assert p.returncode == 0, p.stderr
    return p.stdout

baseline = ssh('baseline_final', 'sacct -j 6165461 -P --format=JobID,State,ExitCode,Elapsed,MaxRSS,Start,End')
assert '6165461|FAILED|1:0|03:00:24|' in baseline
accounting = ssh('accounting', 'sacct -j 6163851,6164049,6163397 -P --format=JobID,State,ExitCode')
(OUT / 'accounting.psv').write_text(accounting)
for jobs, _, _ in allowed.values():
    a.completed_jobs(accounting, jobs)
# Exact directory allowlist: reject unknown children before descending, never glob source trees.
layouts = {}
for name in allowed:
    if name.endswith('/tmp') or name.endswith('/xdg-cache') or name == str(root / a.PILOT_DIR / 'execution.lock') or name == str(root / a.PREFLIGHT_DIR / 'submission.lock'):
        layouts[name] = []
    elif name.endswith('/submission.lock'):
        layouts[name] = ['jobid']
pre = root / a.PREFLIGHT_DIR
for name, children in {
    '': 'accounting-bounded-check.txt batch.exitcode duplicate-check-correct-user.txt duplicate-check.txt execution.lock expected-sources.sha256 independent_lineage_audit.json jobid lineage_ncbi.baseline.tsv logs orthodb prepare.sbatch receipts reviewed-small-files.sha256 sbatch-receipt.json source-hash-attestation.txt submission.lock swissprot tmp xdg-cache',
    'receipts': 'DB.sha256 PROVENANCE.txt SOURCE_COMMIT.txt build_dbs.sh swissprot_viridiplantae_reviewed.fa.gz.sha256',
    'logs': 'slurm-6163397.out slurm-6163397.err',
    'orthodb': 'DONE lineage.tsv provenance.json',
    'swissprot': 'DONE lineage.tsv provenance.json',
}.items():
    layouts[str(pre / name)] = children.split()
extras = sorted({str(wrapper) for _, wrapper, _ in allowed.values()} | {
    str(root / a.PILOT_DIR / 'slurm-6163851.out'), str(root / a.PILOT_DIR / 'slurm-6163851.err')})
config = dict(paths=list(allowed), layouts=layouts, extras=extras, done=a.DONE_TEXT)
program = '''import os, stat, pathlib, hashlib, base64, json, datetime
C = CONFIG
files = {}
errors = {}
cache = {}
def snapshot(name):
    if name in cache: return cache[name]
    p = pathlib.Path(name)
    for parent in (p, *p.parents):
        if parent.is_symlink(): raise ValueError('symlink: ' + str(parent))
    st = p.stat()
    r = dict(kind='directory' if stat.S_ISDIR(st.st_mode) else 'file', uid=st.st_uid, gid=st.st_gid)
    if stat.S_ISDIR(st.st_mode):
        names = sorted(os.listdir(p))
        if name not in C['layouts'] or names != sorted(C['layouts'][name]):
            raise ValueError('listing mismatch: ' + name + ' observed=' + repr(names))
        r['children'] = {n: snapshot(str(p/n)) for n in names}
    else:
        if not stat.S_ISREG(st.st_mode) or st.st_size > 1048576: raise ValueError('unsupported/oversize: ' + name)
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1048576: raise ValueError('changed/oversize: ' + name)
            data = f.read(1048577)
            after = os.fstat(f.fileno())
        if len(data) > 1048576 or (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns) or len(data) != st.st_size: raise ValueError('changed: ' + name)
        r['sha256'] = hashlib.sha256(data).hexdigest()
        files[name] = dict(bytes=len(data), sha256=r['sha256'], base64=base64.b64encode(data).decode())
        if p.name == 'DONE' and data.decode() != C['done']: raise ValueError('malformed DONE: ' + name)
    cache[name] = r
    return r
snapshots = {}
for name in C['paths'] + C['extras']:
    try: snapshots[name] = snapshot(name)
    except (OSError, ValueError, UnicodeError) as e: errors[name] = str(e)
print(json.dumps(dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), snapshots=snapshots, files=files, errors=errors), sort_keys=True))
'''.replace('CONFIG', repr(config))
(OUT / 'remote_readonly.py.txt').write_text(program)
facts = json.loads(ssh('static', '/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3 -B -', program))
save('static_facts.json', facts)
old_facts = json.loads((OUT.parent/'reconciliation_proposal/static_facts.json').read_text())
save('static_delta.json', {k: facts[k] == old_facts[k] for k in ('snapshots', 'files', 'errors')})
assert not facts['errors'] and all(facts[k] == old_facts[k] for k in ('snapshots', 'files', 'errors')), 'FAIL: static facts changed'
# Read-only path availability and interpreter identity; no mkdir, lock or write probe.
probe = """import os, pathlib, json, hashlib, sys, datetime
p=pathlib.Path('/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled')
paths=[p, p/'inputs', p/'code', p/'audit', p/'scratch', p/'logs', p/'run.sbatch']
parents={str(q):dict(is_symlink=q.is_symlink(), exists=q.exists()) for q in p.parents}
e=pathlib.Path(sys.executable).resolve()
print(json.dumps(dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), paths={str(q):os.path.lexists(q) for q in paths}, parents=parents, python=dict(path=sys.executable, resolved=str(e), version=sys.version, sha256=hashlib.sha256(e.read_bytes()).hexdigest())),sort_keys=True))
"""
(OUT/'remote_pathcheck.py.txt').write_text(probe)
pathfacts=json.loads(ssh('pathcheck', '/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3 -B -', probe))
save('pathcheck.json', pathfacts)
assert not any(pathfacts['paths'].values()) and not any(v['is_symlink'] for v in pathfacts['parents'].values()), 'FAIL: destination exists or symlink parent'
# One post-static full queue observation, after baseline final accounting returned.
queue_text = ssh('queue_final', 'squeue --noheader --format="%i|%T|%j"')
queue = dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), command='squeue --noheader --format=%i|%T|%j', stdout=queue_text, returncode=0)
save('queue.json', queue)
remote = root.parent / 'est_followup_20260916_1530/c1_reconciled/inputs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
entries = []
for name, (jobs, wrapper, pin) in allowed.items():
    if name not in facts['snapshots']: continue
    entries.append(dict(path=name, jobids=jobs, wrapper=str(wrapper), wrapper_sha256=pin, snapshot=facts['snapshots'][name]))
manifest = dict(schema='est-c1-reconciliation-v1', blastx_root=str(root), accounting=dict(path=str(remote/'accounting.psv'), sha256=sha(OUT/'accounting.psv')), queue=dict(path=str(remote/'queue.json'), sha256=sha(OUT/'queue.json')), entries=entries)
save('manifest.proposed.json', manifest)
old = OUT.parents[1] / 'est_ops_20260916_1530/pilot_submission_recovered.json'
assert sha(old) == a.PILOT_RECEIPT_SHA256
(OUT/'pilot_submission_recovered.json').write_bytes(old.read_bytes())
pins = {str(p): sha(p) for p in [CODE, CODE.with_name('est_blastx.py'), CODE.with_name('est_blastx_audit.sbatch'), CODE.parent.parent/'taxonomy_lineage.py'] if p.exists()}
save('reviewed_code_pins.json', dict(commit=EXPECTED, files=pins, lineage_pins=a.LINEAGE_SHA256, historical_wrapper_pins={str(w): pin for _,w,pin in allowed.values()}, pilot_receipt_sha256=a.PILOT_RECEIPT_SHA256))
print(json.dumps(dict(entries=len(entries), files=len(facts['files']), errors=facts['errors'], active_est=[l for l in queue_text.splitlines() if 'est_' in l.lower()]), indent=2))
