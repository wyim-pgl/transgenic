# Old P1 terminal reconciliation, 2026-09-17 16:59 PDT

## Actual event, not a cancelled pending job

Job3163986 started18:47:46CDT and failed18:48:37CDT ongh077:51seconds, allocation/batchFAILED1:0,externCOMPLETED0:0. One node/four GPUs, AllocCPUS288 while requestedCPU64. B5chain stillpending onfreshquery. Do not replay old pending-cancellation instructions or claim zero resource use.

Charge full51seconds:51/3600node-hours and204/3600GPU-hours. Include this permanently in unchanged5node-hour/20GPU-hour diagnostic cap. New attempt0.5node-hour/2GPU-hour and proposedpanelreservation1node-hour/4GPU-hour total1.5141667node-hours/6.0566667GPU-hours before any other diagnosed usage. Independent live accounting remains required.

## Failure root cause

Actual stderr: container_stage rejected `epoch05 exact inventory changed` before checkpoint copy/readiness/P1. Current epoch05 contains model.safetensors,meta.json,eval.json and directoriesaccelerate_state,pending_grads. The B5 producer explicitly writes those directories (`train/train_HyenaTransgenic.py:432-449`), including pending gradient state. Meta reports pending_micro15. These are legitimate resumable-checkpoint state, not extra initialization weights. Model initialization is weights-only with new optimizer state; never load or delete these B5resume directories.

Current small meta/eval hashes still exactly match selected pins844ecbc4... and0df4e4a7.... Large weights are not freshly hashed on login; newcompute staging must reverify original6e5e... bytes before/after copying. Permit only the known two real nonsymlink directories in addition to the three required pinned regular files, record they are excluded from weights-only staging; unknown files/directories/symlinks still reject.

## Evidence lifecycle and explicit recovery proposal

Old host successfully hashed SIF/corpus/evidence then renamed the staged evidence into its job before the checkpoint guard failed. The old original staging pathname is gone. Preserve the failed tree and claim/logs.

Exact recoverable candidate, not a search/fallback:
`/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/final-p1-job-3163986/inputs/evidence/fullc0lite.db`

Fresh metadata: canonical regular file1068773376B,owner97434,mode0444,inode522420016516182512,mtime_ns1789516052000000000,no adjacent WAL. Historical compute large-identities receiptSHA0956ff829771470815dc212e224de6471dc8cd9b929c6c4c89b6269e5b87d33a records expected evidenceSHAed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d before that known rename. This is not a fresh hash of the relocated artifact.

Controller proposes separately reviewed recovery of only this named read-only file: newcompute must rehash to original expectedbytes, copy into a fresh private newjob destination, verify source before/after metadata/hash and destinationhash, retain oldsourceunchanged. No moving it back, no deleting failed trees, no directory search, no alternative-path fallback, no new DB build. Extra1.069GBcopy remains inside existing5GiBstaging allowance with2.091GBcheckpoint and boundedHFcache; quota must be refreshed. No copy occurs until new code/package/recovery plan passes independent review.

## Two further actual compatibility facts

- Exclusive GH200node allocation grants288CPUs although task requests64. Validate exact requestedCPUsPerTask/ReqTRES64 and allocatedCPU count against actualexclusive node capacity, not a false assumption of64allocatedCPUs. Node/GPU/time caps remain unchanged.
- Actual host `/usr/bin/python3` is3.6.15. The newadmission `Popen(text=True)` fails immediately on that interpreter, reproduced with harmless /bin/true. Use compatible universal_newlines or explicitly verified availablehostinterpreter; no installation. Syntax/purevalidator/subprocess tests on actualhost must precede submission. SIFPython3.12.3 is a separate runtime.

## Next authority

This is a reviewed-failure recovery proposal, not unused-pending replacement. New durable authorization must bind this terminal failure record, selected evidence identity and true costs. Unknown/changed history still fails closed. Existing FAILED does not generally authorize arbitrary reuse of partialoutputs. Independent review of narrowed recovery and code is required; no GPU cancellation/submission has occurred in this continuation.

Evidence:old_p1_terminal_inspection.txt,old_p1_inventory.txt,old_p1_small_receipts.txt and deployment_v2/review/old3163986-history.json. Original artifacts preserved.
