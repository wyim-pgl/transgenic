"""Local verification only; writes status/pins/checksums, never SSH or auditor audit()."""
import base64, csv, datetime, hashlib, importlib.util, io, json, pathlib, subprocess, sys, textwrap
sys.dont_write_bytecode = True
P = pathlib.Path(__file__).resolve().parent
OLD = P.parent/'reconciliation_proposal'
CODE = pathlib.Path('/tmp/b6-p2-phase-1535433/revision/scripts/evidence/est_blastx_audit.py')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())
f, old = load(P/'static_facts.json'), load(OLD/'static_facts.json')
assert not f['errors']
assert all(f[k] == old[k] for k in ('snapshots', 'files', 'errors')), 'FAIL: changed static evidence'
assert load(P/'static.stdout.txt') == f
assert (P/'accounting.stdout.txt').read_bytes() == (P/'accounting.psv').read_bytes()
assert (P/'accounting.psv').read_bytes() == (OLD/'accounting.psv').read_bytes()
pins = load(OLD/'reviewed_code_pins.json')
assert subprocess.check_output(['git','-C',str(CODE.parents[3]),'rev-parse','HEAD'],text=True).strip() == pins['commit']
for name, digest in pins['files'].items():
    path = pathlib.Path(name)
    rel = path.relative_to(CODE.parents[3])
    assert sha(path) == digest
    assert sha(P/'code'/rel) == digest
    assert hashlib.sha256(subprocess.check_output(['git','-C',str(CODE.parents[3]),'show',pins['commit']+':'+str(rel)])).hexdigest() == digest
(P/'reviewed_code_pins.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
spec = importlib.util.spec_from_file_location('reviewed_auditor',CODE)
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
m, om = load(P/'manifest.proposed.json'), load(OLD/'manifest.proposed.json')
assert set(m) == set(om) and m['schema'] == om['schema'] and m['blastx_root'] == om['blastx_root']
assert m['entries'] == om['entries'] and m['accounting'] == om['accounting']
assert m['queue']['path'] == om['queue']['path']
allowed = a.reconciliation_paths(a.HISTORICAL_ROOT)
assert len(m['entries']) == len(allowed) == 19
for e in m['entries']:
    jobs, w, pin = allowed[e['path']]
    assert (e['jobids'],e['wrapper'],e['wrapper_sha256']) == (jobs,str(w),pin)
    assert e['snapshot'] == f['snapshots'][e['path']]
    assert f['files'][str(w)]['sha256'] == pin
    a.completed_jobs((P/'accounting.psv').read_text(),jobs)
leaves = {}
def check_snapshot(name, s):
    assert type(s['uid']) is int and type(s['gid']) is int
    assert (s['uid'],s['gid']) == (3092957,3092957)
    if s['kind'] == 'directory':
        assert set(s) == {'kind','uid','gid','children'}
        for n,c in s['children'].items(): check_snapshot(name+'/'+n,c)
    else:
        assert set(s) == {'kind','uid','gid','sha256'}
        assert name not in leaves or leaves[name] == s
        leaves[name] = s
for name,s in f['snapshots'].items(): check_snapshot(name,s)
assert set(leaves) == set(f['files']) and len(leaves) == 32
for name,v in f['files'].items():
    b = base64.b64decode(v['base64'],validate=True)
    assert len(b) == v['bytes'] <= 1048576
    assert hashlib.sha256(b).hexdigest() == v['sha256'] == leaves[name]['sha256']
assert sha(P/'pilot_submission_recovered.json') == a.PILOT_RECEIPT_SHA256
assert (P/'pilot_submission_recovered.json').read_bytes() == (OLD/'pilot_submission_recovered.json').read_bytes()
for k in ('queue','accounting'): assert sha(P/pathlib.Path(m[k]['path']).name) == m[k]['sha256']
rows = list(csv.DictReader(io.StringIO((P/'accounting.psv').read_text()),delimiter='|'))
assert len(rows) == len({r['JobID'] for r in rows}) == 30
check = a.Check(); a.audit_accounting((P/'accounting.psv').read_text(),check); assert not check.failed()
baseline = list(csv.DictReader(io.StringIO((P/'baseline_final.stdout.txt').read_text()),delimiter='|'))
assert len(baseline) == 3
assert [(r['JobID'],r['State'],r['ExitCode'],r['Elapsed']) for r in baseline] == [('6165461','FAILED','1:0','03:00:24'),('6165461.batch','FAILED','1:0','03:00:24'),('6165461.extern','OUT_OF_MEMORY','0:125','03:00:24')]
q = load(P/'queue.json')
assert q['stdout'] == (P/'queue_final.stdout.txt').read_text()
assert not any(l.split('|')[0] == '6165461' for l in q['stdout'].splitlines())
# Execute ONLY the unchanged queue validation statements extracted from reviewed source.
source = CODE.read_text()
start = source.index('        require_evidence(type(queue) is dict')
end = source.index("        require_evidence(type(data['entries'])",start)
try:
    exec(compile(textwrap.dedent(source[start:end]),str(CODE)+':queue-only','exec'),dict(queue=q,datetime=datetime,require_evidence=a.require_evidence))
    queue_result = 'PASS at local verification time; remote startup must revalidate age'
except ValueError as e:
    queue_result = 'FAIL: '+str(e)
pf = load(P/'pathcheck.json')
assert not any(pf['paths'].values()) and not any(v['is_symlink'] for v in pf['parents'].values())
for label in ('baseline_final','accounting','static','pathcheck','queue_final'):
    assert load(P/(label+'.transport.json'))['returncode'] == 0
transports = [load(P/(s+'.transport.json')) for s in ('baseline_final','accounting','static','pathcheck','queue_final')]
assert all(a['captured_at'] <= b['started_at'] for a,b in zip(transports,transports[1:]))
subprocess.run(['bash','-n',str(P/'run.sbatch')],check=True)
for line in (P/'run.sbatch').read_text().split("<<'PINS'\n",1)[1].split('\nPINS',1)[0].splitlines():
    digest,name = line.split('  ',1)
    if name.startswith('/'):
        assert name == pf['python']['resolved'] and digest == pf['python']['sha256']
    else:
        local = P/name if name.startswith('code/') else P/pathlib.Path(name).name
        assert sha(local) == digest
baseline_dir = P.parent/'c1_baseline_result'
rollups = list(baseline_dir.rglob('rollup.json'))
assert len(rollups) == 1
rollup = load(rollups[0])
units = list(rollups[0].parent.glob('units/*.json'))
assert len(units) == 18 and all(load(u)['status'] == 'PASS' for u in units)
assert rollup['status'] == 'FAIL' and rollup['failed_units'] == [] and rollup['units_audited'] == 18
assert all(k == 'batch_artifacts:pilot:logs' or k.startswith('dispositions:') for k,v in rollup['tree_checks'].items() if v['status'] == 'FAIL')
status = dict(status='LOCAL PROPOSAL; independent review and staging authorization pending', static_equal=True, entries=19, receipts=32, receipt_bytes=sum(v['bytes'] for v in f['files'].values()), historical_rows=30, baseline_final=baseline, queue_validation=queue_result, queue_captured_at=q['captured_at'], queue_rows=len(q['stdout'].splitlines()), mvp_6165521_present=any(l.startswith('6165521|') for l in q['stdout'].splitlines()), manifest_sha256=sha(P/'manifest.proposed.json'), wrapper_sha256=sha(P/'run.sbatch'), destination_absent=True, checks='exact old static equality, recursive UID/GID/listings/hash/bytes, bindings, committed pins, accounting, verbatim queue-only validator, bash -n; NOT operational full auditor PASS')
(P/'status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
(P/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(P))+'\n' for p in sorted(P.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
print(json.dumps(status,indent=2))
