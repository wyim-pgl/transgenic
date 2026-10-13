# Validation400 independent review — 2026-10-12

**Spec compliance: one Important issue. Quality: Needs fixes before validation receipt publication/staging.** Scope: main `bf29761..0f3ae90`, complete package `/tmp/c2-validation400-review.diff`, authorized plan and implementation report. No source/controller review outside this task, source edits, commits, remote jobs or broad test rerun.

## Strengths and requirements checked

- `history_proof.py:353–402`: shared profile coordinates preserve resume210 and add continuous200→410 /resumed400→410, exact micro/boundary ranges, committed400 snapshot validation, full model/six-state/history/input/RNG/gradient comparisons, and mandatory original210 scalar3.2567708492279053.
- `history_proof.py:328–338`: both final histories must contain exactly one observed rank3 event at epoch0/slot6718/optimizer_step279, positive rank3 graph delta, DNA[1,129024] and labels[1,3924]. A later update count alone cannot satisfy the specialization gate.
- `history_proof.py:405–448`: predecessor validation checks exact boolean successes, six state results, four complete eight-field empty mismatch maps and exact historical fields; job ID, pinned bootstrap and220/210 checkpoint/history metadata are checked without model/optimizer loading. Immutable v1 reports lacking profile remain compatible.
- `history_validation.sbatch:26–30`, `:36–43`, `:55–85`: source/helper checksums, exact COMPLETED0:0 predecessor state, unchanged100GiB storage_preflight, read-only predecessor/original/input binds, fresh allocation verification and separate400 resume precede success. End source/helper rehashes gate HISTORY_VALIDATION_PASS; no production action is included.
- `history_proof.py:94–99` and `probe.py:146`: delaying dataclasses to their run functions preserves training closures while allowing stdlib-only pure prior validator import. Reported compatibility testing correctly distinguishes Python3.6 grammar/import compatibility from actual Python3.6 execution; it does not claim host support for check_prior/training.
- Recorded evidence:17 full focused tests passed in42.542s, then four relevant prior/compatibility checks passed after the isolated import relocation; no full suite rerun was warranted by that relocation. Reported bash/compile/diff checks passed. Unit fixtures do not establish real GPU equality or feasibility within2h.

## Important — receipt can rebind an invalid current candidate

`history_proof.py:451–489`, especially`:474–480`: write_validation_receipt revalidates individual current journals and their declared report hashes, but trusts the report's history_equal=True instead of requiring the **actual current continuous and resumed hashes to equal**. It also does not call `_pinned_manifest(root)` for the **current validation root**; check_prior performs that check only for the predecessor.

Two targeted offline reproductions both published kind=`c2_history_validation_ready`:

1. After a valid compare_pair(validate400), change a valid resumed journal's rank0 metadata, rebind its meta.compile_history_sha256, update report.checkpoint_history_sha256.resumed and write that report back to comparison.json. Keep historical/model/state/trace success fields intact. write_validation_receipt succeeds although current continuous/resumed history SHAs differ.
2. After a valid comparison, substitute root/bootstrap.json's original model SHA, rebind bootstrap manifest+SHA in all three current journals, update their metadata and both declared report history hashes, and rewrite comparison.json. write_validation_receipt succeeds with a bootstrap that no longer matches the pinned original receipt.

This is a concrete false READY path after mutually consistent declarations are recomputed. Comparing report to comparison.json or validating each journal separately does not establish the required shared current history or original provenance. Add cheap current-root `_pinned_manifest(root)` and actual current-history equality checks before publication, with regressions that rebind both artifact metadata **and report/comparison declarations**. These fixes require no additional heavy checkpoint read and preserve the intended small receipt boundary.

### Exact fixture reproducer

Use the existing HistoryProofTests fixture from test_history_proof.py. Common setup (temporary directory only):

```python
fixture.fixture(root, validation=True)
fixture.fixture(prior)
(prior / 'comparison.json').write_text(json.dumps(fixture.prior_report()))
(prior / 'verified_record.json').write_text(json.dumps({'job_id': '3369338'}))
(root / 'verified_record.json').write_text(json.dumps({'job_id': '3369999'}))
source_file.write_text('a' * 40 + '\n')
os.environ['SLURM_JOB_ID'] = '3369999'
report = proof.compare_pair(root, profile='validate400')
```

History variant:

```python
hp = root / 'resumed/latest_state/compile_history.json'
history = json.loads(hp.read_text())
history['entries'][-1]['ranks'][0]['meta']['row'] = ['changed-after-comparison']
hp.write_text(json.dumps(history))
mp = hp.with_name('meta.json')
meta = json.loads(mp.read_text())
meta['compile_history_sha256'] = proof.probe.sha(hp)
mp.write_text(json.dumps(meta))
report['checkpoint_history_sha256']['resumed'] = proof.probe.sha(hp)
(root / 'comparison.json').write_text(json.dumps(report))
receipt = proof.write_validation_receipt(root, report, source_commit='a' * 40,
    predecessor_root=prior, predecessor_job_id='3369338', source_commit_file=source_file)
```

Observed: `receipt_created_with_current_histories_different c2_history_validation_ready True`.

Manifest variant, on a fresh valid fixture: set `bootstrap['checkpoint_files']['accelerate_state/model.safetensors']='b'*64`, write root/bootstrap.json, then for continuous/resumed/at400 set history.bootstrap to `{'manifest': altered_manifest, 'sha256': proof.probe.sha(root/'bootstrap.json')}`, write each history and update its metadata digest. Rebind both report final-history hashes, write comparison.json, invoke the same publisher. Observed: `receipt_created_with_substituted_original_manifest c2_history_validation_ready`.

## Review boundaries

- Complete diff read once; its first tool display truncated the check_prior/receipt/batch portion, so only that omitted section was retrieved from the same package. Existing reviewed helpers/legacy facts were reused; no broader source crawl or suite execution.
- Both probes used isolated temporary fixtures and the existing real journal validators; no remote write, GPU allocation or permanent artifact mutation.
- The source transport failure of initial job3369338 is a separate follow-on review and does not justify relaxing predecessor or scientific gates. The validation script correctly refuses a FAILED predecessor. Initial immutable v1 artifacts remain protected.

**Assessment:** fix the receipt's two missing current-artifact assertions and verify the recomputed-declaration cases before approval. The rest of this scoped extension matches its plan.
