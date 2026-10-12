# Independent scoped ONT full-wrapper review

## Verdict: BLOCK production execution and acceptance

Review completed locally. No implementation source/test edits, remote commands, submissions, commits, installations, real alignments, canonical data hashing, or large inputs. No actual50/full structural acceptance and no runtime acceptance are granted. The new review harness is evidence of selected wrapper behavior, not scientific validation or a real-tool end-to-end fixture.

Read all 267 lines of `ont_full/wrapper.py`, all `test_wrapper.py` and `REPORT.md`, the pinned old `gate.py` and `source_pins.json`, and the actual frozen driver CLI and relevant import sites. Required graft orientation/query attempts each timed out (60 s and 25 s); no graph result was available, so inspection fell back to the exact supplied release paths. Advisor consultation failed due to the installed provider client's model compatibility. This review does not claim an additional advisor endorsement.

## Findings

### F1. High: the pinned list is not a closed import tree

`ont_full/wrapper.py:183-194` hashes the nine listed files and rejects `.pyc` and `__init__.py`, but permits other executable source files. `longread/gate.py:68-79` imports the driver before checking the imported windows module's pathname. The driver inserts `frozen/src` into `sys.path` at lines 21-23.

Reproduction: a copied, otherwise unchanged pinned tree with an extra `frozen/src/transgenic.py` passes every listed source hash, with no initializer or bytecode present. Calling the real pinned gate's `load_driver` under `-I -B` executes a marker write from that unlisted module, then fails because it is not a package. Subsequent rejection does not undo arbitrary execution. Test: `test_unlisted_source_module_executes_before_gate_rejects`.

A second route exists in subprocesses: wrapper line 221 isolates only the top-level driver. Frozen driver lines 86-88, 183-184 and 202-204 launch helper Python scripts without `-I`; isolation flags are not inherited through `sys.executable`. Under the wrapper's clean environment, the actual frozen sample script launched from an isolated parent successfully imports an added `revision/scripts/json.py` and executes its marker. Test: `test_isolation_not_inherited_by_nested_sample`.

Action: require a closed, independently verified immutable deployment tree, including rejection of unlisted importable files, and validate the import paths of every Python subprocess with the approved interpreter/runtime. Do not claim initializer/bytecode rejection alone closes imports. Any change to frozen invocations/source needs a separately reviewed release and new pins. These are adverse-release tests, not evidence that the existing release contains malicious files.

### F2. High: existing helper tests do not establish executable deployment compatibility

All eight existing tests pass with the local default `python` (3.8.20), but every synthetic `execute` scenario on that interpreter fails immediately at `wrapper.py:166`: `AttributeError: 'PosixPath' object has no attribute 'is_relative_to'`. Failure occurs before the private control receipt is created. `Path.is_relative_to` requires Python 3.9 or later. A matching executable hash does not establish API compatibility, and the documented request contract specifies no interpreter version floor.

The same review harness succeeds on `/usr/bin/python3` 3.10.12. This does not establish the production interpreter's compatibility. Action: specify and independently pin/test a supported interpreter and its runtime closure, and add an implementation-level execute-path test before approving any request. If 3.8 must be supported, fix the compatibility issue rather than accepting helper-only green tests.

### F3. Medium: stderr mutation can still produce COMPLETE

`wrapper.py:103-114` records both streams but `check_boundary` rechecks only stdout. At lines 243-247, diagnostics freshly hash stderr without requiring equality with the original `receipt['stderr']` identity.

The synthetic execute test changes `driver.stderr` during the postflight seam. Execution returns `VERIFIED_DESCRIPTIVE` and publishes a valid receipt-hash COMPLETE while `receipt['stderr'] != receipt['diagnostics']['driver.stderr']`. Test: `test_execute_stderr_mutation_still_publishes`. This is an injected mutation, not an assertion that the real gate edits stderr.

Action: compare both original stream identities before publication; reject inconsistent diagnostic identities. The documented downstream controller check should catch this discrepancy, but it is not implemented by this wrapper and does not justify internally inconsistent COMPLETE evidence.

### F4. High acceptance blocker: real execute integration remains untested

The implementation's eight tests call helpers only. This review adds execute orchestration coverage but explicitly substitutes scheduler approval, input/reference bindings, driver object/panel length, scientific postflight, and child workload. The actual pinned driver was invoked only with `-I -B --help`; helper import behavior was tested with actual sample `--help`. No real minimap2 is on PATH, so there is still no real minimap2/samtools/seqkit fixture traversing the actual driver, eight-command ledger, scientific postflight and publication together.

Action: after F1-F3, run a bounded independently reviewed real-tool fixture using the exact interpreter, source layout, CLI and tool environment, then validate live Slurm parsing/accounting and reviewed resource limits. Do not replace this requirement with mocked success or `runtime_reviewed=true`. No full canonical run is authorized by this report.

## Checked behavior and limits

- **Trust/request:** Hardcoded gate and manifest hashes match disk; all nine frozen hashes match the manifest. Request/wrapper SHA arguments and fixed canonical path derivation are useful consistency boundaries. Independent controller retention of approved hashes is still required. `runtime_reviewed=true` is an assertion, not evidence. Hash-before-read/import plus later identity checking is not an immutable snapshot; concurrently mutable deployments remain outside a strong trusted-execution claim.
- **Exact driver CLI:** The wrapper uses the correct actual script, `--root`, `--run`, `--reads`, `--ref`, `--gff`, `--panels`, `--ref-md5`, `--gff-md5`, `--threads`, `--execute`. The driver's `SCRIPTS.parents[1] / 'src'` correctly resolves to `frozen/src`; no top-level path arithmetic bug was found. Top-level isolated `--help` works on both checked interpreters. Nested helper isolation is F1, not an invented top-level import failure.
- **Scheduler:** Synthetic live-response parsing accepts one owned RUNNING task/node with exact CPUs, affinity and 1G memory. GPU TRES, insufficient remaining time, wrong memory and unsupported `UNLIMITED` reject. No actual scheduler was queried. Strict hostname/time/memory formats can reject legitimate differently formatted allocations; this is fail-closed compatibility, not demonstrated authorization bypass. No duplicate-field/adversarial scheduler output exhaustiveness is claimed.
- **Budget:** Driver timeout reserves cleanup time; total alarm covers preflight/postflight after setup. CPU/memory checks compare the live allocation, not actual peak use. Scratch has no request field or enforcement. Executable/runtime libraries and real resource use still need the already-required runtime review; this review does not invent default memory/scratch or missingness policy. Initial wrapper/request hashing is before the total alarm, and masked group cleanup may extend past an alarm. Treat the wall budget as soft application handling, with scheduler enforcement/accounting still necessary.
- **Process handling:** Real tiny Python children verify ordinary success, nonzero exit, timeout, SIGTERM, retained logs and termination of a same-process-group descendant even after its driver exits. A descendant explicitly creating a new session survives `killpg(driver_pid)`; the harness kills that synthetic escaped process itself. Test: `test_escaped_session_descendant_survives_group_cleanup`. Process groups are not complete descendant containment. No claim that the frozen tools daemonize is made. Require Slurm/cgroup cleanup/accounting for an all-descendants guarantee. SIGKILL/OOM/node failure and late worker-stdout failure remain external acceptance concerns as already documented.
- **Failure/publication:** Synthetic nonzero, timeout, TERM, input mutation, stdout mutation and recursive output mutation all leave INCOMPLETE plus FAILED and no COMPLETE. Synthetic success preserves scope, null decision, false training eligibility and marker-to-receipt hash agreement. Early failures before control creation cannot leave the later failure receipt. stderr mutation is the concrete gap in F3. Mutable on-disk receipts/inventories must still be independently anchored by the controller.
- **Pilot/full boundary:** No pilot receipt is accepted as full evidence. The old gate's nested pending-inventory/policy label is preserved, and the wrapper's outer status is descriptive only. There is no basis here for structural availability, pooling, HiFi policy, scores, training eligibility, or treating missing evidence as zero.

## Reproducible review evidence

All additions are under `ont_full/review_evidence/` except this report. Synthetic release copies live temporarily inside that directory and are cleaned by the harness. Implementation files were not changed.

Commands from `ont_full/`:

```sh
timeout 20s python -B -m unittest -v test_wrapper.py
cd review_evidence
timeout 45s /usr/bin/python3 -I -B review_harness.py
timeout 40s python -I -B review_harness.py
```

Results retained:

- `existing_tests.txt`: 8 tests, OK on Python 3.8.20.
- `review_results.txt`: 9 review tests, OK on Python 3.10.12, including six execute failure subcases. Some passing tests intentionally prove vulnerabilities, not their absence.
- `review_python38_results.txt`: 9 review tests, FAILED with 8 execute-related errors (six subcases and two success/mutation cases), all at `is_relative_to`.
- `review_harness.py`: review-only synthetic orchestration/adversarial checks. Uses byte-identical wrapper/release copies, mocks canonical/scientific/scheduler bindings, and substitutes tiny real Python children for the actual alignment workload. It must not be described as an unmodified production end-to-end run.
- `tested_hashes.json`: absolute-path SHA256 inventory for all reviewed primary files, all nine frozen files, harness and final logs; all source pins checked successfully. `minimap2` absent, samtools and seqkit present. No tools installed.

Primary tested SHA256 identities:

| File | SHA256 |
| --- | --- |
| `ont_full/wrapper.py` | `887608643b1136e2ac64f288eef6b53b25690f258c23902e3d61a3acd9537b91` |
| `ont_full/test_wrapper.py` | `d527860b7d9ef61a0c1f6fde0f727a4b7bd3e4b774b174b0d682936e2f404cba` |
| `ont_full/REPORT.md` | `06abcf115c61e6408608569c624ed0598895b3a2f76a01e904206a28e21297bb` |
| old `gate.py` | `0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f` |
| old `source_pins.json` | `38c56c608336ef3fdcb83eaa5eb35a355ddc77ea1e8aff8387160603bf88ce34` |
| frozen driver | `fcf5be4412a25c399df9b9ea1f9637a1e9a733b336794a2244a6c8e9ce19564b` |
| review harness | `8759dc04b3c20e9c67e5744d1c94d734f66ec3e470b3795952faa678b395c61c` |

## Scoped TODO disposition

Completed: full requested source review, bounded existing tests, review-only synthetic execute/adversarial harness, exact tested hashes and actionable report.

Blocked: production approval pending closed import deployment, supported interpreter execute testing, stderr identity consistency, real-tool integration fixture, live scheduler/runtime/resource review and independent controller approval. No implementation remediation, remote experiment or downstream structural-scoring work was authorized or performed.
