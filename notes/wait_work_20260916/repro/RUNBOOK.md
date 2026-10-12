# Task 4: reproduction and non-destructive recovery

## Executed plan / evidence boundary

Read recovery and the three execution authorities; preserve finished CPU logs and timestamped document copies; bind source/job/artifact inventory; run local integrity checks. No SSH, queue check, GPU, jobs, dependency installation, source/test edits, merge or push. Only this directory written. Advisor request failed (client version), not an approval.

`ledger.json` is the machine-readable inventory. `captured.sha256` hashes **captured copies**, not a global freeze of changing notes, scheduler state or active logs. Each item carries origin, capture time, size and digest. Snapshot captured 2026-09-17 04:20:19 UTC; scheduler evidence is older: 2026-09-16 21:14 PDT. Do not infer later state. Frozen source report contains historical HOLDs and old3163545 references; later controller scope/reviews/priority snapshot supersede those operational statuses, not its source identities.

## Locally executable preflight (no runtime imports)

From repository root:

```sh
python3 -B notes/wait_work_20260916/repro/preflight.py
(cd notes/wait_work_20260916/repro && sha256sum -c captured.sha256)
bash -n notes/mvp_execution_20260916/final_p1/p1.sbatch
```

Preflight validates 25 captured files, cross-references, 100 archive files against immutable Git b660, the 86-file source pin and 13-file decoder pin, release ledger digest, and snapshot budget arithmetic. It does **not** read/hash remote DB/checkpoint/SIF or certify a completed probe. `capture.py` is the historical one-shot capture procedure: it refuses an existing capture directory. Do not rerun it over this bundle.

## Controller gates and job ownership

- 3163545: old d99 P1 CANCELLED, elapsed0/unallocated; never monitoring/retry target.
- 3163986: final b660 S0/P1 PENDING Priority at snapshot, elapsed0/unallocated. Do not submit a duplicate.
- 6165525: reconciled C1 audit RUNNING 2:06:47 at snapshot; no terminal result claimed.
- 6165461: baseline FAILED1:0 despite 18/18 units PASS; historical receipt/layout/disposition failure remains distinct.
- 6165460/6165462: actual HSP diagnostics completed, `.extern OOM0:125`; expanded semantics INCONCLUSIVE132/12, not policy approval.
- 6165521: original GFFCompare job FAILED; recovered existing evidence has reviewed 16-case semantic PASS,236 files/34 maps. No rerun and no biological approval.
- Protect B5 next3131538 and dependencies3149356–3149363. Prior3131537 not in latest capture: do not infer terminal state. Seeds456/789 deferred.

Only controller submits/writes remotely. Fresh owner/state/full own queue/accounting check is required before any later action; coordinate once rather than duplicating this snapshot. Source ACCEPT, deployment ACCEPT, CPU tests, real six-gate readiness, real training completion and scientific decisions are separate records. Source/decoder pins are in ledger and are not absolute manifests or authority to run. Current actual diagnostic use0 node-hours vs reserved0.5; cap5 node-hours/20GPU-hours cumulative, remaining unreserved4.5. Pending reservation is not consumed time. Reconcile allocation-wide startup/copy/hash/verification/teardown and all attempts before reserving more. Training budget is not yet established.

## P1 recovery decision table — all failures STOP, preserve, controller review

| Failure / stage | Evidence and non-destructive checks | Forbidden shortcut |
|---|---|---|
| Quota / inode / filesystem | Preserve raw quota receipt, filesystem free and stage identity. Existing implementation requires conservative quota calculation, >10000 inode reserve,40GiB reserve+5GiB staging envelope. Recheck B5 growth before separately reviewed recovery. | Deleting B5/cache/failed artifacts, assuming old350G/500G remains live |
| Source/input hash mismatch | Compare exact b660 archive/release ledger plus config/panel/candidate and expected DB/SIF/epoch05 pins. Large byte checks on compute only. Identify which source/destination changed. | Recomputing an expected pin from failed payload or copying receipt identity as authority |
| Cache/import/offline closure | Preserve both root and hub layout refs/main, revision0a629ab, model snapshot, modules/import paths, versions and original identities. Real SIF must supply Torch2.14/CUDA13.4/Transformers4.57.6/Accelerate1.15.0; local4.46.3 tests are not substitutes. | Global installs, network fallback, broad cache exemption or empty-scaffold waiver |
| Missing tiers/RC/N | Preserve tiles.json and tile-availability/report, exact train RN twins30720/61440/129024 and nonzero EST0–6; availability7 alone insufficient. | Cropping, synthetic RN, changing train membership, changing vocabulary/cap |
| Timeout before/after S0 | Retain outer status/signal, scheduler allocation+batch+extern, stdout/stderr, S0 control inventory, partial probe measurement. Outer1700s+20s grace/30min allocation; P1 inner900s unchanged. | Extending deadlines, receipt completion fabrication or eager fallback |
| OOM/native crash | Separate payload failure from extern0:125; preserve memory/time/runtime evidence. Native crash stops bounded runtime; require diagnosis before retry. | Declaring extern success from batch success or native cause from missing Python dependency |
| Evidence moved / START collision | Namespace `/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545`; stage evidence may have moved into final-p1-job-3163986/inputs. Inspect receipts/path existence read-only, preserve original START marker and job subtree. Missing stage evidence is a blocker. | Searching partial jobs to auto-reuse, moving/copying DB back, deleting marker or automatic transfer/retry |
| Receipt/DONE mismatch | Require real completed official wrapper, expected independently trusted receipt SHA and matching DONE, no FAILED sibling, correct status/mode/source/paths/binding. Retain measurement and log. Raw measurement alone is insufficient. | Repairing DONE, copying old d99/772 identities or promoting partial evidence |

After successful P1: remaining five real gates (Hyena step0 all tiers, generation/cache parity, native bf16/Inductor,4rank, resource) and seed-specific bindings remain mandatory. Resource measurements must precede separately budgeted3seed×250updates/12warmup. Any source change requires fresh review/pins/manifests/bindings; paths/runtime changes require corresponding fresh bindings. Existing source/import/cache paths must remain available through decode. Frozen-reference Git object `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv` is a later panel/evaluation requirement, not an extra P1 dependency. No automatic resubmission or cleanup anywhere.

## Native crash: preserved, not rerun

Full original log preserved as `captured/mvp-final-source-full-cpu.log`, with matching exact Git final report, historical version report and finished focused/scoped retry logs. Reported full-run exit139; log itself records fatal SIGSEGV, quiet progress6/12/18/22%, and16 `F` markers before crash, but no failed node IDs, failure tracebacks or explicit full argv/environment record. Tqdm monitor thread is waiting; current Python thread is in decode_panel called by the350-output synthetic CPU fixture. Exact b660 line141 is **ids,mask = ids.to(device),mask.to(device)**; tracks.to(device) is adjacent line142. Earlier independent review paraphrased line141 as tracks conversion; preserve that report unchanged and retain the exact source excerpt to avoid perpetuating a false pinpoint.

Minimal predecessor sequence cannot be identified from this evidence. **Zero CPU repro runs** used, rather than another blind whole-suite run or an already passing standalone fixture. Failure ordering may matter, but is not proven causal. Separate scoped Python child dependency failures (Accelerate visibility/psutil) do not explain this native crash. Scoped599PASS/1SKIP and focused passes do not prove fixed.

Needed before a justified minimal reproduction: original ordered pytest node IDs (including failure setup/teardown), exact full argv/cwd/PYTHONPATH/plugins and original package/native library inventory; an original crash core plus matching Python/Torch/NumPy shared objects/build IDs, process mappings and all-thread native backtrace (`gdb ...; thread apply all bt full`, Python `py-bt` if available). No core/backtrace is in supplied evidence; no systemwide core search or collection setting was changed. A newly measured environment must not be mislabeled original. If controller later identifies a minimal predecessor, allow at most one separately recorded CPU run <=900s with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, exact read-only b660 export, no cache writes into source (`-B`, disable pytest cache), unchanged test deadlines, output only to fresh owned path. No GPUs/jobs/global installs. Record exit/signal/time and original-order limitation; success/nonreproduction never means fixed.

## Local preflight corrections

Initial verifier failed on local Python lacking `Path.is_relative_to`; replaced with parent containment. Second check incorrectly counted archive directory entries as payload files; inspection found100 regular files plus14 directories, now validated separately (all symlinks/other member kinds rejected). Original failed results are retained in `testresults.txt`; final rerun exited0. These were verifier-only changes, not production or test-deadline changes.

## Remaining blockers

Actual3163986 output and6165525 terminal audit unobserved here; six real gates/training/350 predictions incomplete. Frozen reference/panel compatibility and independent long-read evidence still need actual validation. HSP30aa definition, duplicate isoform policy, HiFi-vs-ONT-only and withholding are not selected here; pure phase geometry is not production projection and not an EST8 training prerequisite. `.extern` OOM and native SIGSEGV causes remain unresolved.
