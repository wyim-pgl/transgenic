#!/usr/bin/env python3
"""One-shot local evidence capture; exclusive outputs, no runtime imports/remote I/O."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
OUT = HERE / 'captured'
OUT.mkdir(exist_ok=False)
records = []

def capture(src, name, kind):
    src = Path(src)
    before = src.stat()
    if before.st_size > 10 * 1024**2:
        raise ValueError(f'capture exceeds 10MiB: {src}')
    data = src.read_bytes()
    after = src.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    dest = OUT / name
    with dest.open('xb') as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    records.append(dict(path=str(dest.relative_to(HERE)), origin=str(src), kind=kind,
                        captured_at=NOW, original_mtime_ns=before.st_mtime_ns,
                        size_bytes=len(data), sha256=digest))

for name in ['mvp-final-source-full-cpu.log','mvp-final-source-focused.log',
             'mvp-final-source-mvp-cpu.log','mvp-final-source-mvp-cpu-retry.log','mvp-final-env-check.log']:
    capture(Path('/tmp') / name, name, 'finished_test_log')
refs = {
 'resume.md': '.atl/resume.md', 'todo.md': '.atl/todo.md',
 'priority.md': 'notes/MVP_TRAINING_PRIORITY_20260916_1545.md',
 'decoder.md': 'notes/MVP_DECODER_CONNECTION_20260916.md',
 'est.md': 'notes/EST_EXECUTION_20260916_1530.md',
 'initial_live_status.txt': 'notes/wait_work_20260916/initial_live_status.txt',
 'final_source_report.md': 'notes/mvp_execution_20260916/final_source_report.md',
 'source_review.md': 'notes/mvp_execution_20260916/final_source.independent_review.md',
 'deployment_review.md': 'notes/mvp_execution_20260916/final_p1/deployment.independent_review.md',
 'controller_scope.json': 'notes/mvp_execution_20260916/final_p1/controller_execution_scope.json',
 'initialization.json': 'notes/MVP_INITIALIZATION_SELECTION_20260916.json',
 'submission.txt': 'notes/mvp_execution_20260916/final_p1/submission_receipt.txt',
 'cancellation.txt': 'notes/mvp_execution_20260916/final_p1/old_pending_cancellation.txt',
 'release.json': 'notes/mvp_execution_20260916/final_p1/release/release.json',
 'release.sha256': 'notes/mvp_execution_20260916/final_p1/release/release.sha256',
 'deploy.py': 'notes/mvp_execution_20260916/final_p1/release/deploy.py',
 'code.tar': 'notes/mvp_execution_20260916/final_p1/release/code.tar',
}
for name, src in refs.items():
    capture(ROOT / src, name, 'timestamped_document_snapshot' if name.endswith('.md') else 'timestamped_record_snapshot')
commit = 'b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'
for src,name in [('revision/scripts/mvp/mvp_decode_panel.py','exact_decode_panel.py'),
                 ('tests/test_mvp_decode.py','exact_test_mvp_decode.py'),
                 ('notes/MVP_FINAL_SOURCE_INTEGRATION_REPORT.md','git_final_report.md')]:
    data = subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{src}'])
    dest = OUT / name
    with dest.open('xb') as f: f.write(data)
    records.append(dict(path=str(dest.relative_to(HERE)),origin=f'git:{commit}:{src}',kind='immutable_git_blob',captured_at=NOW,size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
pins = next(ast.literal_eval(n.value) for n in ast.parse((OUT/'deploy.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
release = json.loads((OUT/'release.json').read_text())
ledger = dict(schema_version=1, captured_at=NOW, status_as_of='2026-09-16T21:14:15-07:00',
 live_queries_this_task=0, owner='controller only: remote writes/submissions; this worker local repro records only',
 evidence=records, reference_documents=refs,
 source=dict(commit=commit,executable_commit='362eb899d8d43164ec1e0fb0f3b174ead9b1ba40',relative_pin='555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44',decoder_pin='b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb',component_count=86,decoder_count=13,release_archive_sha256=release['release_files']['code.tar'],release_ledger_sha256='5b36bdf09a8b8eb548147654e23d6ad59fe18542642a05eb720f22be102fdfb8'),
 expected_remote_pins=pins,
 artifacts=dict(checkpoint='/work/nvme/bilv/wcyim/transgenic/runs/seed123/epoch_05',corpus='/work/nvme/bilv/wcyim/transgenic/db/b5_full_a40_v1.db',evidence_staged='/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/evidence/fullc0lite.db',evidence_bytes=1068773376,sif='/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif',panel='revision/protocols/b6_panels_v1.tsv',panel_sha256=release['small_input_pins']['revision/protocols/b6_panels_v1.tsv'],reference_authority='92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv',hyena_revision='0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446',remote_bytes_rehashed_by_this_task=False),
 budget=dict(scope='cumulative diagnostics; not training',cap_node_hours=5,cap_gpu_hours=20,actual_node_hours_at_snapshot=0,reserved_node_hours=0.5,reserved_gpu_hours=2,remaining_unreserved_node_hours=4.5,includes='all allocation startup/copy/hash/verification/teardown and prior attempts',training_budget='unestablished pending resource measurement',automatic_retry=False),
 jobs=[dict(id=3163545,role='obsolete d99 P1',state='CANCELLED',elapsed_seconds=0,allocated=False,evidence='captured/initial_live_status.txt'),dict(id=3163986,role='b660 S0/P1 only',state='PENDING',reason='Priority',elapsed_seconds=0,allocated=False,evidence='captured/initial_live_status.txt'),dict(id=6165525,role='C1 reconciled audit',state='RUNNING',elapsed='02:06:47',evidence='captured/initial_live_status.txt'),dict(id=6165461,role='C1 baseline',state='FAILED',exit_code='1:0',unit_pass=18,unit_total=18,extern='OOM 0:125',evidence='captured/est.md'),dict(id=6165460,role='HSP initial diagnostic',state='COMPLETED',exit_code='0:0',extern='OOM 0:125',scientific_acceptance=False,evidence='captured/est.md'),dict(id=6165462,role='HSP expanded diagnostic',state='COMPLETED',exit_code='0:0',extern='OOM 0:125',semantic_result='INCONCLUSIVE:132 ELICITED/12 NOT_ELICITED/0 INVALID',evidence='captured/est.md'),dict(id=6165521,role='GFFCompare synthetic diagnostic',state='FAILED',exit_code='1:0',extern='OOM 0:125',recovered_semantic_result='PASS 16 cases,236 files,34 maps; no rerun',evidence='captured/decoder.md')],
 protected_b5=dict(next_job=3131538,next_state='PENDING Priority',dependent_jobs=list(range(3149356,3149364)),prior_job=3131537,prior_job_live_state_not_in_capture=True,modify=False,seeds_456_789='deferred'),
 gates=dict(controller_authorization='bounded S0/P1 only in captured/controller_scope.json',source_review='ACCEPT bounded export/fresh binding only',deployment_review='ACCEPT bounded S0/P1 only',cpu_verification='reported scoped599PASS/1SKIP/64subtests; controller46PASS/1SKIP; whole repo crashed',real_gpu='no PASS observed',six_gate_readiness=False,training_done=False,real_350_predictions=False,phase='pure geometry reviewed; production projection/policy unfinished; not MVP prerequisite',scientific='HSP length policy, duplicate isoforms, HiFi vs ONT-only and withholding unresolved; no GO/STOP emitted'),
 native_crash=dict(status='UNRESOLVED',reruns_this_task=0,reason_no_rerun='quiet log gives failure positions but no predecessor node IDs or failure tracebacks; minimal predecessor cannot be identified',reported_exit=139,current_frame='b660 mvp_decode_panel.py:141 ids,mask = ids.to(device),mask.to(device)',adjacent_frame='line142 tracks[locus.gene_id].to(device); earlier review paraphrased this as141',fixture='tests/test_mvp_decode.py:190 test_runner_exact_350_no_overwrite_and_partial_failure',cause='unknown; separate missing Accelerate/psutil Python failures do not establish native cause',passing_reruns_prove_fixed=False),
 historical_environment=dict(source='captured/git_final_report.md',python='3.12.2',torch='2.6.0',transformers='4.46.3',accelerate='0.34.2',duckdb='1.5.5',exact_original_full_command_not_preserved=True,current_environment_not_substituted=True),
 advisor='consultation failed client/model version incompatibility; no advice or approval')
(HERE/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
(HERE/'captured.sha256').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
print(json.dumps(dict(captured_at=NOW,files=len(records),bytes=sum(r['size_bytes'] for r in records)),indent=2))
