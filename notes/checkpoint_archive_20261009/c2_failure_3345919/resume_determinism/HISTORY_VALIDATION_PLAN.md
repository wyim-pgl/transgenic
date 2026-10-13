# C2 validation across280 and checkpoint400

> Agentic execution: use subagent-driven-development with TDD and independent review. This is the already-authorized next scientific gate, prepared while3369338 waits for allocation.

**Goal:** After the initial200→220 proof passes, use another protected clone of original200 to prove the future specialization and exact400→410 resume before promoting any production path.

**Context:** Main notes hold diagnostic tools and evidence; production source is separately committed188bf3e in `/tmp/b6-c2-recompile`. Initial proof3369338 already uses immutable remote helpers `ops/c2_compile_history_20261012/v1_188bf3e`; NEVER mutate that directory or resubmit it. Initial main snapshotbf29761 contains the reviewed first harness. User explicitly authorized validation then C2 production4099 without another approval, preserving scientific gates.

## Global constraints

- No remote writes, Slurm submissions, production promotion, original200/diagnostic256 edits or source-branch changes in this task.
- Keep original200→210 historical loss3.2567708492279053 mandatory in both proof profiles.
- Actual4ranks, C2, batch1, bf16, deterministic, same operational shared-cache namespace and exact whole-checkpoint/input/RNG/gradient/history comparisons.
- New validation starts from original200, retains every original bootstrap slot1..15+3915, trains through the known rank3 specialization at slot6718/update280, snapshots400, continues410, then a fresh process resumes400→410.
- A previous job's mere Slurm0 or lone passed flag is insufficient. Prior comparison must contain all expected successful model/state/history/historical/rank-trace fields, and its verified_record job ID must match the requested predecessor.
- New helpers are a new immutable deployment later; current3369338 uses its pinned v1 bytes.

## Task 1: Add bounded validation profile and its batch artifact

**Own files:** modify `history_proof.py` and `test_history_proof.py` in this directory; create `history_validation.sbatch` and `HISTORY_VALIDATION_IMPLEMENTATION.md`. Do not modify the initialproof sbatch, receipts/reports, trainer modules, .atl files, or other agents' files.

**Interfaces:**

- Preserve `compare_pair(root)` as initial resume210 behavior. Add keyword `profile='resume210'`, accepting only `resume210` and `validate400`.
- Profile coordinates are explicit:

```python
PROFILES = {'resume210': (200, 210, 220), 'validate400': (200, 400, 410)}
```

- Initial root names stay continuous/at210/resumed; validation names continuous/at400/resumed.
- `run` adds `--snapshot-step` default210; only210 or400 accepted, and snapshot_step must precede stop. A missing optional field on programmatic legacy args retains210. Snapshot events/receipt validation use the selected exact checkpoint; every other trace/context condition remains strict.
- `compare --profile validate400` checks full continuous200→410 and resumed400→410 traces, rank completion at9840, snapshot at9600, allmodel/state/history exact equality at410, original210 scalar, and a recorded observed rank3 graph event at epoch0/batch_index6717 (slot6718), optimizer_step279, with input_ids shape[1,129024], labels shape[1,3924]. This event is retained in both final histories; do not merely check update>=280. If real evidence later contradicts this known event, report it instead of weakening the gate.
- Successful validation report additionally contains `profile: 'validate400'` and `validation400: {'checkpoint_step':400,'stop_step':410,'rank3_specialization_slot':6718,'rank3_specialization_recorded':True}`. Initial profile may add `profile:'resume210'` but must preserve all existing fields/defaultbehavior.
- Add a pure prior-proof validator that requires passed/model_equal/history_equal allTrue, no different_model_keys, exact six true optimizer/scheduler/fourRNG states, exact four rank trace maps with all eight required empty mismatch lists, and historical210 expected/actual exact with equalTrue. Old immutablev1 reports have no profile key; accept that known shape. Reject missing/string-valued/false/truncated fields even when passed=True.
- Expose CLI `check-prior --root PATH --job-id DIGITS` for batch preflight; validate prior comparison and verified_record.json job_id, require initial proof stop220 checkpoint metadata and pinned bootstrap receipt binding without rereading huge model tensors on login. This CLI runs inside the new allocation; full new input verification still runs once for that allocation.

**Batch artifact:** adapt reviewed history_proof.sbatch into new history_validation.sbatch. Require REPO_DIR/OPS_DIR/PROOF_JOB_ID; same4GPU/64CPU/ghx4-interactive, initially2h budget, same read-only binds/cachepolicy/hashchecks. Preflight predecessor actualCOMPLETED0:0 via sacct -X and the pure prior validator before cloning/training. Root `/work/runs/c2_compile_history_validation/$SLURM_JOB_ID`. Do a fresh probe.verify using this allocation's job ID, clone original200, build explicit manifest, continuous --stop410 --snapshot-step400 --snapshot at400, clone at400 into resumed, resumed --stop410 without bootstrap. Compare validate400; end marker HISTORY_VALIDATION_PASS only after source rehash and exit0. No production action or dependency submission in this script/task. Controller decides submission only after independent review.

**Tests:** first add failing tests proving400 profile is unsupported and prior passed=True can hide missing proof fields. Refactor shared coordinates rather than duplicating the entire acceptance implementation. Complete actual-schema fixtures for400 and its6718 observed event; mutate/remove that event and snapshot400, replace boundary RNG, forge prior historical/state/rank fields. All must fail. Keep existing7 regressions passing, run bash-n/py_compile. Local command prefix `PYTHONPATH=/tmp/b6-spec-review-deps:/tmp/b6-c2-recompile/src:/tmp/b6-c2-recompile /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s <this-directory> -p test_history_proof.py -v`.

**Deliverable:** commit only owned code/test/newscript/report files on current main (diagnostic/evidence scope established by user; production branch untouched). Report exactRED/GREEN/commands/passcounts, helper API, sourceSHA and any remaining constraint. No agents spawned by implementer. Controller owns independent review, staging and submission journal.
