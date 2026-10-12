# Bounded EST diagnostic submission plan

Scope: one Sitalica primary100 diagnostic, at most16 selected query names with original accepted manifest. No blastx rerun, DB rebuild, A11 certification, routeA/fullC0/raster or scientific/training acceptance. Existing B5/P1 and other projects untouched.

Authority: SEED123_THREE_TRACK_EXECUTION_20260917.md and its INTERFACE_AND_OWNERSHIP.md authorize controller submission only after reviewed package, explicit budget and live queue check. Source review EST_SAMPLE_FIX1_REVIEW accepted; deployment D1 lifecycle accepted by EST_DEPLOY_FIX_REVIEW; D2.1 resolved by permission_fix_review. Controller verified26tests and package inventory. ExternOOM remains unresolved and must be reported separately; ONT full-run gate is not waived. This bounded EST sample is not an ONT full run.

Resources: CPU account/partition cpu-s1-pgl-0, one node/task,2CPU,8GiB,30minutes, no requeue. Worker limits40GiB charged reads,256MiB output,1M expanded bases,1740seconds. Grace TERM60seconds; postchecks35seconds/USR1watchdog40seconds. Total memory/kernel/filesystem behavior is scheduler-enforced, not proven by synthetic tests.

14:38 PDT live pronghorn queue/accounting: no EST sample job; unrelated sylvan jobs preserved. Python3.11.14 present, GPFS27TiB free. Evidence est_predeploy_live.txt. Immediately before submitting, query queue again and exclusively reserve a single submission-attempt file. No retries after uncertain submission; reconcile queue/accounting instead.

Fresh remote namespace: `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_real_sample_recovery_20260917_1438`. Exclusive0700 creation required, existing path rejects. Stage release and controller helper without original-data writes; rehash transferred archive and every package file using trusted controller code before running helper. Final helper requires effective private owner/modes. Freeze released source files read-only. Preserve all logs, incomplete namespaces and submission errors.

Reviewed pins:
- ledger46563a19f5699c55df12df7241af7f807e6d98bed0d7c0666138ce78804c40d6
- launcher0c4321cb199374907dc99680a9ad8fb2c01202e7e17a4c62f6573543dc0697ea
- batch7fe7c107e5796043393ee6eea52d8db96cf327877577cf1679c658425ef31f1e
- preparation helper9f30e4c828abe84ce2b2df5e1ad8ea648dc32414207a9f13a37204ff3acb9b8f
- manifest7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c

No payload/read-source hash on login node. Compute runtime checks exact Python/samtools identity and pinned module inventory. Result acceptance separately requires terminal job state, controller result/verification, worker report/diagnostics/marker identity and independent interpretation; COMPLETE alone is insufficient. Preserve extern step failure separately.
