# Track B producer — round-1 scoped independent re-review

## Verdict

**Accept the five requested corrections within this partial code-review scope. R1–R5 are addressed. No new Critical or Important regression identified in the fix.** This is not operational promotion, real-tool certification, training authorization, or full scientific acceptance.

Reviewed `/tmp/b6-p2-scoring-1535144` at `c02444541f22b7ed04b8e87db1815bc04f3359c6`, implementation `64b6d5bf2494bb5b23e061c6c0917b4bbafcdbb9`, against `8803afb44189e7510e67e39976676dcad664c164`. The post-implementation commit changes only the provenance report. Read the original five findings, the complete controller `B.producer.fix1.diff`, the worktree report's round-1 section, and the added/changed tests. Queried graft first and inspected the relevant loader/finalizer/helper definitions and positive-fixture setup.

## Finding disposition

All producer locations below refer to `revision/scripts/mvp/mvp_produce_counts.py` at the reviewed HEAD.

| Finding | Disposition and evidence |
|---|---|
| **R1 — source/package closure** | **Addressed**, lines 73–109. Closure is enumerated in the checkout bound by the audited `b5_runtime.py` path: every `src/transgenic/**/*.py` plus the MVP trainer, matching the existing training rule (`mvp_cli_runtime.py:146–151`). Exact path inventory, canonical paths, and file hashes are required. Exactly five nonblank package versions are required; four shared fields agree with finalizer metadata, while duckdb remains mandatory without an invented finalizer field. Historical versions are not compared to installed scorer versions. Both audited implementation identities bind to their expected inventory locations. Three local semantic-helper implementations must byte-match the bound copies; unsupported historical implementations fail closed rather than being imported. The relocated-checkout positive test includes an extra training-only module; omitted/extra modules and incomplete package records are rejected. |
| **R2 — initialization contradictions** | **Addressed**, lines 112–157 and 177–182. Source paths and bytes are bound; seed 123, the embedded recipe's required fields, and zero source evidence channels are checked. The canonical digest is recomputed from `run_config['recipe']`, using the actual loader canonicalization. Source semantic values and origins are reconstructed and compared with the audit, present optional source-config fields are reconciled, and source/target semantic equality is checked. Source safetensors must open, be nonempty, omit the evidence projection, and have metadata equal to the audit. Implementation identities are checked through R1. The replacement positive fixture calls real `load_weights_only` and the real finalizer publisher on tiny model artifacts; only the remote Hyena factory is replaced with the existing local test double (`tests/test_weights_init.py:45–89`). Eight negative variants cover the original contradictions. |
| **R3 — ID normalization/parser loss** | **Addressed**, lines 222–281. ID/Parent tokens use an explicit canonical alphabet; whitespace and ambiguous tokens are rejected rather than normalized. Attribute keys are checked too. Parsed gene and per-gene transcript identity sets must exactly equal the preflight inventories. Tests cover the original two-transcript collision, leading/trailing/nonbreaking whitespace, and simulated gene/transcript loss. This is a deliberately strict supported subset, not a claim to accept every legal GFF3 identifier. |
| **R4 — incomplete empty output** | **Addressed**, lines 225–261. Empty files require the exact GFF3 version header. Zero-byte, whitespace-only, and comment-only inputs are rejected; genuine header-only output remains accepted. Tests exercise all four cases. This marker distinguishes the specified transport representation; it does not independently witness successful decoder execution. |
| **R5 — final config** | **Addressed**, lines 141–149 and 182. The duplicate-key-rejecting reader parses final config before semantic reconstruction. It must be an object, contain audited fields, reconstruct the target manifest exactly, and preserve raw semantic values rather than relying on silent config normalization. Tests reject invalid JSON, non-object JSON, duplicate keys, and a valid object with inconsistent target dimensions. |

## Regression assessment and boundaries

The fix is limited to the producer, its fixtures/regression tests, and its report: five files, 418 insertions and 24 deletions. No trainer/finalizer, metric math, bootstrap, thresholds, legacy parser, or 350-file transport schema changes are present. The integration fixture now shares one serialized baseline across seeds instead of assuming separately saved equal tensors have identical safetensors bytes; this repairs test identity without weakening production checks.

The provenance checks operate under the existing trusted-operator receipt model. They establish the requested local consistency, not historical execution against wholesale artifact forgery. The verifier does not replay training or repeat the loader's full tensor-loading compatibility checks; the real-loader fixture validates the normal receipt path. No new cryptographic attestation requirement is imposed by this review. The report appropriately retains the external concrete pin for the intended historical seed-123 best selection as an operational prerequisite.

Unchanged blockers remain: real GFFCompare 0.12.6/frozen-reference validation, duplicate-isoform multiplicity policy, independent read-splice provenance and coverage/HiFi/ONT author policy, withholding, and operational decoder execution. Header-only integration predictions and stubbed biological/tool authority do not constitute empirical evaluation. Full scientific acceptance remains blocked.

## Verification evidence

- Controller-supplied result: **142 tests + 21 subtests passed, 42.58 s**. Accepted as supplied evidence; not independently rerun here.
- Worktree report additionally records **16 candidate tests passed** and the separate 27-test review suite. These are reported results, not fresh reviewer runs.
- Independently verified HEAD, the five-file diff scope, and report-only changes after the implementation commit. `git diff --check 8803afb44189e7510e67e39976676dcad664c164..HEAD` passed.
- No full-suite rerun, source edits, SSH, jobs, subagents, merge, or push. Pre-existing worktree `.gitignore` modification and untracked `.ignore` were left untouched. Only this requested review document was written in main.
- Advisor consultation was attempted but unavailable due to client/model-version incompatibility; no advisor conclusion is claimed.
