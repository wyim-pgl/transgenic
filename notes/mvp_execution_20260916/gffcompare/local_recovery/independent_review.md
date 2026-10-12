# Independent read-only recovery review — job 6165521

## Verdict

**ACCEPT recovered actual-tool semantic checks, restricted to the 16 captured synthetic function-level cases. No blocking finding in that scope.** Original allocation/batch remains **FAILED 1:0**; `.extern` remains **OUT_OF_MEMORY 0:125**, cause unresolved. **Full scientific/pipeline validation: NO.** Biological compatibility is not tested, duplicate multiplicity policy remains fail-closed, and `decision=null` remains appropriate.

Reviewed `notes/MVP_GFFCOMPARE_REAL_RESULTS_20260916.md`, recovery source/JSON, original harness/producer, stored scheduler/submission records, and captured `real_run/` evidence. No engine, producer, original runner, recovery analyzer, SSH, jobs, or subagents were invoked. No original artifacts or source were edited. The controller's reported 10-test/33.36-second run was not duplicated and is not claimed as my test run. Two independent stdlib read-only assertion commands completed successfully instead.

## Findings and evidence

Paths below are relative to `notes/mvp_execution_20260916/gffcompare/` unless stated otherwise.

1. **Confirmed original observation-layer failure, not repaired job success.** `smoke.py:160–167` searches `raw/*.tmap`; all ten raw invocations actually have exact query-sibling `CASE/compare.predicted.gtf.{tmap,refmap}` and zero tmaps in `CASE/raw/`. `result_status.txt:6–8,18–28` records FAILED allocation/batch, unresolved extern OOM, and the `ordinary` assertion. Recovery `recover.py:73–83` uses exact sibling paths, not a wildcard/fallback. The seven exercised producer comparisons colocate query and prefix under `producer/`, so the original producer lookup at `mvp_produce_counts.py:331–334` works for those captured cases. This does not certify all producer inputs.

2. **Inventory and origin binding independently verified.** Rehashed all **236 manifest entries**, checked exact inventory including the manifest (**237 files, 122,149 bytes**), and counted **34 actual maps** (10 raw pairs + 7 producer pairs). Manifest SHA256 is `335bee5e27b65cfdd7eea1a32b3ef349ed9f31ee9853074245b2b64eb8f724af`; its exact remote absolute path is bound in the pinned status record. Verified all recovery JSON inventory hashes/sizes, its six historical artifact identities, three-entry stage manifest, and JSON SHA256 `273865876c1861f474c3b95e28ebbc28c61c2c6be14993ae8b05c67a604fad4a`. All raw/producer commands and producer output paths/hashes/sizes match the original remote origin literally. Relevant implementation: `recover.py:95–120,143–155`.

3. **No expected-value change or runner reuse.** Historical `smoke.py:45–73` defines the original fixtures/expectations. Recovery `recover.py:46–52` selects only `tx` and `cases` AST function definitions after historical hash verification in `analyze`; it does not import the harness, invoke its analyzer/runner, or load producer functions. The selected definitions have no decorators or side-effecting defaults. The new analyzer has no engine subprocess invocation. Its AST execution is acceptable for these exact pinned definitions, not a general sandbox guarantee.

4. **All 16 expectations independently confirmed.** My assertions used a separately written literal expectation table, checked captured result/receipt values and recovery observations, then separately recomputed counts from fixture strand/junction or nonempty CDS structure sets. All raw and producer exon-only GTF coordinates were checked against half-open fixture geometry with start+1/end conversion. All seven producer map pairs agree with raw pairs; all ten raw exit codes are zero. `recover.py:126–175` matches these observations.

   | Cases | TP/FP/FN or required rejection | Captured raw classes |
   |---|---|---|
   | ordinary, terminal_short, terminal_extend | 1/0/0 each | `=` each |
   | junction_change | 0/1/1 | `j` |
   | opposite_strand | 0/1/1 | `s` |
   | unmatched | 0/1/1 | `u` |
   | mono_exact, mono_utr_variant | 1/0/0 each | no raw call |
   | mono_cds_change, mono_opposite, mono_no_cds | 0/1/1 each | no raw call |
   | empty_query | 0/0/1 | header-only maps |
   | multiisoform_unique | 2/0/0 | `=`, `=` |
   | duplicate_query_terminal | predicted-duplicate ValueError | `=`, `=` |
   | duplicate_reference_terminal | reference-duplicate ValueError | `=` |
   | duplicate_mono_cds | predicted-duplicate ValueError | no raw call |

   Reference multi chain is `(200,300),(400,500)`; terminal changes preserve it, middle-start310 does not, and opposite strand does not match. Mono matching uses CDS `[1020,1180)` and strand, ignoring UTR edges; CDS-start1021, opposite strand, or absent CDS does not match. These mono results are producer semantics, not GFFCompare CDS behavior.

5. **Duplicate/empty semantics preserved; no silent tie policy.** All three duplicate errors have the exact required `author multiplicity policy needed` suffix and empty producer directories. The duplicate-query raw map retains both `q` and `q_terminal` (length300/260), both assigned to `r`; this is not two independent reference TPs. Duplicate-reference raw refmap records only `r -> g|q` despite both reference inputs being retained. That particular choice does not establish a tie-breaking rule. Empty raw query returns0 with header-only tmap/refmap; producer command is null, predicted GTF empty, and counts remain0/0/1. Six mono/empty producer cases bypass the engine. Relevant original contract: `mvp_produce_counts.py:297–347`; recovery guards: `recover.py:133–137,164`.

6. **Provenance and claim limits correctly qualified.** `smoke.py:96–121` records and rechecks executable/source hashes around the original calls. Local verification authenticates the copied manifest/runtime/log chain, **not fresh remote executable bytes**: binary bytes are absent. `analysis.json:2–6,1751–1770` retains this limitation, original failed scheduler status, unresolved duplicate policy and null decision. No inference that extern OOM was fixed is warranted. The report's `PENDING` review text and JSON review field are historical pre-review state; this separate document supplies the independent acceptance without rewriting evidence.

## Acceptance boundary

Accept the recovered **actual GFFCompare 0.12.6 synthetic diagnostic observations and exercised function counts**. Do not promote this to successful Slurm execution, frozen biological-reference compatibility, full `produce()`/decoder/parser integration, independent-read validation, GPU/training readiness, biological performance, or authorization for a new run. No expected values, thresholds, inputs, or multiplicity/reference tie policy changed.

Operational note: the mandatory `graft map` lookup reported an automatic graph refresh (2 changed files). I made no explicit graph/source edits; original evidence was independently hash-checked afterward. Graft reported approximately 835,464 tokens saved across the two context queries.
