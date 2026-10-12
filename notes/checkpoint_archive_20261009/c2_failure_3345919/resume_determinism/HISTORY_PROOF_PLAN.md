# Implemented compile-history proof

Continue from3364480 without repeating it. Source implementation remains on
b6-c2-recompile-20261011; staged immutable source and helpers must pass local,
NGC and independent review before this allocation is submitted. Query live Slurm
and quota first, maintain a new submission journal, never reuse old controller.

The validation allocation has4GPUs and uses original200 as read-only source.
Once-per-allocation full input verification is shared across diagnostic phases
under the same recorded file stats/bindings/job ID, following probe.py's existing
verified-record protocol. Original checkpoint hashes and4800 draw-prefix hashes
are pinned to prior preserved evidence. All priming uses the production helper.

1. Clone original200 into continuous. Build explicit manifest for slots1..15 and
3915 from verified original receipt/prefixes. No diagnostic256 promotion.
2. Run new trainer200→220 with bootstrap, snapshot committed210 as at210 while
the same process continues through220. Trace actual training inputs/RNG/loss and
boundary gradients; shared execute_replay wrapper excludes primers from traces.
3. Clone at210 into resumed and run210→220 using embedded history, no bootstrap.
4. Compare all model tensors, optimizer/scheduler/all4RNG states, history binding,
actual input/context/loss traces and boundary gradient digests. Require original
update210 scalar3.2567708492279053. Missing records or nonfinite loss fail closed.

Use rank-shared compiler caches with normal FX/AOT options, never the old
diagnostic cache disables. Capture effective fingerprint and cache environment.
One matched pair is the first full-checkpoint gate, not4099 readiness: next verify
future compile event beyond280, committed400 and another resume, space, then
launch authorized4099 through new journal/source/receipt-bound operational plan.
Epoch eval/PILOT_FINAL/normal exit/DONE remain required final checks.

TDD for new harness focuses on refusing foreign/mutated bootstrap inputs and
detecting weights/state/trace/historical mismatch. Reuse probe.clone/verify/tree
and checkpoint value canonicalization, preserving original artifacts. Do not
repeat tests of unchanged helper implementations. No trial is submitted until
its concrete source/helper hashes and scripts are reviewed.

## Local harness evidence (15:22 PDT)

- Manifest/comparison tests RED: missinghistory_proof.py. First implementation exposed a fixture arithmetic typo (checkpoint has9files:2metadata+3state+4RNG), corrected without changing implementation expectations.
- TrainingTrace test RED: missingTrainingTrace; GREEN after implementation of primer exclusion and actualgradient trace.
- Final command: `/home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism -p test_history_proof.py -v`:4passed,1.964s.
- `bash -n history_proof.sbatch` and `python -m py_compile history_proof.py`:exit0.
- Harness source/helper/sbatch independent review and actualNGC execution still pending. No job has been submitted.
- Intended operational caches `/work/compiler_cache/c2_history_v1/{inductor,triton}` are shared byallranks and retained acrossproof/validation/production links, with effectiveFX/AOToptions fingerprinted. Cache namespace differs from unrecorded historical location; historical210match remains required. No cache disable or rankisolation.
