# Proof harness fix round1 — 2026-10-12 15:36 PDT

All3 Important findings addressed; scoped re-review pending. No remote write orGPU submission.

1. Acceptance now calls the real local-only CompileHistory.load with expectedworld4, checkpointposition and savedidentity; validates effectiveoperatingfingerprint; binds fullbootstrap to root/bootstrap.json and immutableoriginal_checkpoint_receipt.json copied byte-for-byte from3361666/B/DIAGNOSTIC_SOURCE.json. Requires exact16positions1..15+3915/reconstructedorigins and onlyobservedlaterentries. Literal original4800drawprefixhashes remain pinned. Both220checkpoints and committedat210journal are validated.
2. Proofrun wraps runtime_fingerprint to reject disabledFXlocal/AOTlocal, force_disable, wrongworld/backend/bf16/checkpointing/determinism/sharedcachepaths beforetraining. Comparator independently repeatschecks. Remote-cache options remainruntime defaults. No cachepolicy changed to make proofpass.
3. Trace captures actual module.training/torch.is_grad_enabled and declaredtrainarm view; rankownership isrequired for everyrow. Exactlyoneprepared/finished event perrank, expectedstart4800/5040 andstop5280, actual4rankNCCLmetadata and preempted220result required. Exactlyonerank0 continuoussnapshot at5040 binds210position and snapshotcopyreceipt. Snapshotregularfilehashes are checked againstreceipt. Its pathbasename plusreceipt digest allows collectedartifacts to relocate without weakening contentbinding. Missing/duplicate lifecycle/snapshot records and changedcontext fail. BoundaryRNG nowcomparedalongsidegradients.

Tests now construct complete valid journal fixtures using the independentlytested journalAPI; they no longer bless origin-only records. Mutations with recomputedouterhash cover missingrank, disabledcache, wrongbootstrapslot and substitutedoriginalcheckpoint. Wrongrank/prepared/finished/context and missing/duplicatesnapshot regressions cover collectionfailures. ActualtinyTorchforward/backward provesprimers excluded and context/gradient recorded.

RED: expanded6-test command failed7assertions plus1missingcontext error beforefix, exactly exposing reviewgaps. Extraoriginal-receipt substitutiontest initially hit oldsnapshotreceipt binding; fixture was corrected to rebind its altered snapshotreceipt too, then intendedRED was ValueError notraised. Productionpinnedreceiptcheck added after thatRED.

GREEN command:
`PYTHONPATH=/tmp/b6-spec-review-deps:/tmp/b6-c2-recompile/src:/tmp/b6-c2-recompile /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism -p test_history_proof.py -v`

Finalresult:7tests,18.350s,OK. bash -n history_proof.sbatch andpy_compile history_proof.py exit0. The included immutableoriginalreceipt is an additional requiredstagingartifact; helpermanifest must includeit. Productionintegration whole-branchreview separatelyfound resolveddefaultcachepaths missingfromfingerprint; originalimplementer isfixingthat beforestaging.

## Final two scoped corrections

Re-review found actual trainer returns `history: []` alongside preempted/global_step; the exacttwo-key comparator would reject correct output. Updated realfixture to actualreturn, observed expectedRED rejection, and nowvalidate requiredpreempted220 fields, emptyepochhistory and absenceofdone. Also require allseven bootstrappositionfields including originaloptimizer_step=(slot-1)//24. Addedwrong-update regression; its firstRED attempt was maskedby the independently knownfinished-resultbug. Afterfixingthat, temporarily restoring index-onlypositionvalidation produced the intendedRED (wrong_bootstrap_update:ValueErrornotraised); fullvalidation restored in finally. Finalscopeddelta `/tmp/c2-history-proof-final-delta.diff` identifies exactsourceSHA. No scientificcondition relaxed.
