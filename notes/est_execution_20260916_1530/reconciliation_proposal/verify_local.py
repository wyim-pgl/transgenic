"""Offline verification/report only: no remote calls, no approval decision."""
import base64, hashlib, importlib.util, json, pathlib, subprocess, sys, tempfile
sys.dont_write_bytecode = True
P = pathlib.Path(__file__).resolve().parent
code = pathlib.Path('/tmp/b6-p2-phase-1535433/revision/scripts/evidence/est_blastx_audit.py')
spec = importlib.util.spec_from_file_location('audit_verify', code)
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
sha = lambda b: hashlib.sha256(b).hexdigest()
f = json.loads((P/'static_facts.json').read_text())
m = json.loads((P/'manifest.proposed.json').read_text())
pins = json.loads((P/'reviewed_code_pins.json').read_text())
helper = pathlib.Path(a.launcher.HELPER)
pins['files'][str(helper)] = sha(helper.read_bytes())
assert pins['files'][str(helper)] == a.APPROVED_HELPER_SHA256
(P/'reviewed_code_pins.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
assert set(m) == {'schema','blastx_root','accounting','queue','entries'}
assert not f['errors']
allowed = a.reconciliation_paths(a.HISTORICAL_ROOT)
assert len(m['entries']) == len(allowed) == 19
for name, receipt in f['files'].items():
    b = base64.b64decode(receipt['base64'],validate=True)
    assert len(b) == receipt['bytes'] <= 1048576 and sha(b) == receipt['sha256']
for record in (m['accounting'],m['queue']):
    assert sha((P/pathlib.Path(record['path']).name).read_bytes()) == record['sha256']

def verify_snapshot(name, snap):
    assert type(snap['uid']) is int and type(snap['gid']) is int
    if snap['kind']=='directory':
        assert set(snap)=={'uid','gid','kind','children'}
        for child,sub in snap['children'].items(): verify_snapshot(name+'/'+child,sub)
    else:
        assert set(snap)=={'uid','gid','kind','sha256'}
        assert snap['sha256']==f['files'][name]['sha256']
for e in m['entries']:
    assert set(e)=={'path','jobids','wrapper','wrapper_sha256','snapshot'}
    jobs,w,pin=allowed[e['path']]
    assert (e['jobids'],e['wrapper'],e['wrapper_sha256'])==(jobs,str(w),pin)
    assert f['files'][str(w)]['sha256']==pin
    a.completed_jobs((P/'accounting.psv').read_text(),jobs)
    verify_snapshot(e['path'],e['snapshot'])
    if pathlib.Path(e['path']).name in ('tmp','xdg-cache') or e['path']==str(a.HISTORICAL_ROOT/a.PILOT_DIR/'execution.lock') or e['path']==str(a.HISTORICAL_ROOT/a.PREFLIGHT_DIR/'submission.lock'):
        assert e['snapshot']['children']=={}
pre=a.HISTORICAL_ROOT/a.PREFLIGHT_DIR
with tempfile.TemporaryDirectory() as td:
    dest=pathlib.Path(td)
    def materialize(path,snap,original):
        if snap['kind']=='directory':
            path.mkdir(exist_ok=True)
            for n,s in snap['children'].items(): materialize(path/n,s,original+'/'+n)
        else: path.write_bytes(base64.b64decode(f['files'][original]['base64']))
    materialize(dest,f['snapshots'][str(pre)],str(pre))
    a.preflight_evidence(dest)
    assert (dest/'execution.lock').read_bytes()==b''
for batch, job in [(a.PILOT_DIR,'6163851'),(a.REMAINING8_DIR,'6164049')]:
    name=str(a.HISTORICAL_ROOT/batch/'submission.lock/jobid')
    assert base64.b64decode(f['files'][name]['base64'])==(job+'\n').encode()
queue=json.loads((P/'queue.json').read_text())
blocked=[line for line in queue['stdout'].splitlines() if line.split('|')[0].startswith(('6163851','6164049','6163397')) or any(s in line.split('|')[2].lower() for s in ('blastx','lineage','est_'))]
assert '6165461|RUNNING|est_c1_20260916_1530' in blocked
# Confirm exact approved source bytes, not merely the worktree HEAD label.
for name,digest in pins['files'].items():
    rel=str(pathlib.Path(name).relative_to(code.parents[3]))
    committed=subprocess.check_output(['git','-C',str(code.parents[3]),'show',pins['commit']+':'+rel])
    assert sha(committed)==digest
status=dict(status='DRAFT/BLOCKED', staging='NOT STAGED; planned remote paths only; existence not probed', blocking_queue_rows=blocked, queue_captured_at=queue['captured_at'], missing_or_mismatched_historical_paths=[], static_entries=19, unique_small_files=len(f['files']), unique_small_file_bytes=sum(v['bytes'] for v in f['files'].values()), manifest_sha256=sha((P/'manifest.proposed.json').read_bytes()), verification='offline schema, 19 bindings, 30 accounting rows, receipt bytes/hashes, preflight_evidence on reconstructed small-file copy, committed source pins; not an operational load_reconciliation PASS')
(P/'status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
rows=['# Captured remote small receipts','', 'All bytes are losslessly preserved as base64 in `static_facts.json`; paths relative to historical ROOT. No large payloads read.','', '| Path | Bytes | SHA256 |','|---|---:|---|']
for name,v in sorted(f['files'].items()): rows.append(f"| `{pathlib.Path(name).relative_to(a.HISTORICAL_ROOT)}` | {v['bytes']} | `{v['sha256']}` |")
(P/'RECEIPTS.md').write_text('\n'.join(rows)+'\n')
report=f'''# Real C1 reconciliation manifest proposal — 2026-09-16

## DRAFT / BLOCKED — not approved, not staged

Approved code commit `{pins['commit']}` was read from `/tmp/b6-p2-phase-1535433`; all four deployment source hashes match committed bytes (see `reviewed_code_pins.json`). No source changes, remote writes, job actions, baseline changes, merge, push or subagents. Advisor unavailable due to client-version mismatch; no advisor review claimed.

Bundle: [reconciliation_proposal/](est_execution_20260916_1530/reconciliation_proposal/). Proposed manifest SHA256: `{status['manifest_sha256']}`.

First remote action was unfiltered live queue at 23:08:34 UTC. Static capture completed {f['captured_at']}. Exactly one post-static queue capture at {queue['captured_at']} retains **6165461 RUNNING est_c1_20260916_1530**. The reviewed queue validator correctly rejects this row. No polling or attempt to omit the baseline. This is a failed quiescence observation, not quiescence evidence establishing safety.

## Captured facts

All19 allowlisted paths have exact numeric UID/GID/kind/recursive children and bounded small-file SHA256 snapshots. No unresolved/mismatched historical path found in this capture. All owners in top-level snapshots: {sorted({(e['snapshot']['uid'], e['snapshot']['gid']) for e in m['entries']})}. Preflight traversal was restricted to its closed expected layout; other sentinels were empty or jobid-only. Unknown listing, link, special or >1MiB file would stop that path. No raw13, large FASTA, DB, source-tree recursion or lock probe.

**{len(f['files'])} unique small receipts, {status['unique_small_file_bytes']:,} total bytes**; all bytes preserved losslessly in `static_facts.json`, full per-file byte counts/hashes in [RECEIPTS.md](est_execution_20260916_1530/reconciliation_proposal/RECEIPTS.md). This includes five preflight receipt files, lineage/provenance/DONE, preflight logs/submission receipt/exitcode/sentinels, all three producing wrappers, and pilot stdout/empty stderr. Existing pinned recovered pilot extract is copied unchanged. SHA256SUMS binds every bundle file except itself.

Accounting contains exactly30 rows: nine search jobs plus preflight6163397, allocation/batch COMPLETED0:0 and each extern OUT_OF_MEMORY0:125. This anomaly remains unresolved. Offline verification passed canonical schema, all19 fixed job/wrapper bindings, decoded bytes/hashes, lineage/provenance/preflight validation, sentinel contents and exact committed code hashes. It is **not** a real-tree auditor run or an operational manifest PASS.

## Planned paths, not staging

Manifest references planned external captures under `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/c1_reconciled/inputs/accounting.psv` and `queue.json`. Proposed deployment manifest name: `inputs/manifest.proposed.json`; pilot receipt: `inputs/pilot_submission_recovered.json`. These remote paths were neither created nor existence-probed. Controller must check exclusive fresh availability; do not overwrite. Local proposal is intentionally in main notes, not a production external capture location.

Schema references: reviewed auditor `load_reconciliation:554–625`, `reconciliation_paths:483–496`, `historical_snapshot:462–480`, `preflight_evidence:527–551`; C worktree plan/report read in full. Main operations report and its capture bundle were retained. No approval boolean was generated.

## Controller independent-review checklist

- [ ] Review source commit/pins, command transports, all receipt bytes/hashes,19 exact paths and job/species bindings; verify bundle SHA256SUMS.
- [ ] Wait for baseline6165461 completion without cancellation/resubmission; preserve baseline outputs. Independently inspect its final accounting/results.
- [ ] Obtain a **new full unfiltered queue capture after completion**, no relevant active/held/pending EST jobs; <=1h old at auditor startup. Current capture is blocked and must not be staged as acceptable evidence.
- [ ] Revalidate static paths/owners/listings/hashes and accounting before deployment. A capture is not continuous locking or proof of live lock ownership.
- [ ] Rebind changed queue hash in a newly reviewed immutable manifest and record its separate digest; current proposed digest cannot approve a future edited manifest.
- [ ] Confirm new external input/run/snapshot paths absent, exclusively stage a **new immutable auditor** and capture copies, verify all code/input hashes, and independently review invocation. Never patch baseline.
- [ ] Keep extern anomaly, ancestry/merged-taxid limitation, C2/C3, phase acceptance and production authorization unresolved; reconciliation does not clear scientific gates.

Graft estimated1,044,588 tokens saved across orientation/schema calls (tool estimate). No deployment or scientific acceptance claim.
'''
(P.parents[1]/'EST_C1_REAL_MANIFEST_REPORT.md').write_text(report)
files=sorted(p for p in P.iterdir() if p.is_file() and p.name!='SHA256SUMS')
(P/'SHA256SUMS').write_text(''.join(sha(p.read_bytes())+'  '+p.name+'\n' for p in files))
print(json.dumps(status,indent=2))
