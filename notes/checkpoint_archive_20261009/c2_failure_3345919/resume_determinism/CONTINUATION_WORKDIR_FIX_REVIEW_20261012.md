# C2 WorkDir normalization independent review — 2026-10-12

**Disposition: approved for the narrow controller correction. No material findings.** This resolves the3345920 command-identity blocker found while reviewing the concrete deployment draft. Final deployment configuration remains conditional until the corrected immutable runtime bundle is staged/pinned, actual host parser smoke is complete, and the full-proof/fingerprint evidence is adjudicated.

Reviewed the complete three-file change from32b14d5 in `continue_c2_history.py`, `test_continue_c2_history.py`, and `CONTINUATION_IMPLEMENTATION.md`. The reviewed controller SHA256 is `d4e59576141d7f08d08832383cf1f246828ad3c4df0f0c7e8517426fc9d1b96b`. This digest identifies the reviewed implementation before the author's scoped commit; this review did not edit those files.

The correction requests `WorkDir%4096` in exact job identity accounting. `submission_command` preserves the original SubmitLine argv and computes a separate command identity: absolute script paths are normalized lexically; relative script paths require recorded absolute, unambiguous WorkDir and are joined to it before lexical normalization. It does not use the controller's current directory, filesystem resolution, basename matching or suffix matching. NUL/newline/carriage-return/pipe ambiguity is refused. The parser accepts the expected seven identity columns with or without the normal trailing delimiter and rejects additional ambiguous columns.

For live jobs, normalized accounting command must equal the normalized absolute scontrol Command. That comparison prevents a same-named script in another repository from satisfying the held-job gate. The resulting absolute command remains the same once accounting becomes terminal. Original SubmitLine/argv, token extraction, UID and submission timestamp are retained; exact submission reconciliation and existing held/retired owner/state checks are unchanged. Absolute submissions remain independent of WorkDir, preserving the controller's own explicit absolute batch-script submissions.

The author's five new regressions were reviewed. They exercise actual `Slurm.observe` parsing from scheduler-shaped text, active PENDING/JobHeldUser through terminal CANCELLED and existing retirement gates, missing/relative/ambiguous WorkDir, disagreement with live Command, and absolute/relative lexical normalization with unchanged argv. The implementation report records RED before the change, then the focused10-test parser/retirement/accounting set passing in7.943s with zero skips; Python3.6 grammar/compile and stdlib-only checks are author-provided evidence.

An additional reviewer probe imported the changed controller and used the **actual3345920 SubmitLine, SubmitTime, WorkDir and live scontrol record** from `/tmp/c2-controller-deployment-evidence.json`. Only scheduler responses were supplied locally; no controller operation was initialized and no scheduler command ran. The accounting response included the trailing pipe emitted by parsable accounting. Results:

```text
actual recorded3345920 held_to_cancelled command/gates PASS
missing_relative_ambiguous_WorkDir_and_live_command_mismatch_rejected PASS
```

The probe asserted that live and terminal `command` both equal:

```text
/work/nvme/bilv/wcyim/transgenic/repo_b6_233574a/revision/scripts/b6/wp5_pilot.sbatch
```

It independently preserved the exact original argv/SubmitLine, blank historical comment token, UID97434 and recorded submission time, then passed `held_old` and `retired_old`. Blank WorkDir, relative WorkDir, an added pipe field and a conflicting live absolute Command each raised the intended block. Scoped diff whitespace checks also passed. The already passing broad controller/producer suites were not rerun for this isolated parser change.

The parent separately established that real cluster accounting supports and supplies absolute WorkDir for the historical relative submissions, and owns actual host3.6 parser smoke. Runtime bundle `runtime_v2_1dd19f0` contains the old parser and must be superseded with a new reviewed immutable bundle and corresponding configuration/receipt pins; it must not be modified in place. This review does not approve the old draft or claim that local parser evidence substitutes for proof/validation gates. The parent has reported proof3369694 terminal completion and is collecting raw comparison/metadata for final adjudication; validation400 and every subsequent gate remain mandatory.

No remote actions, source changes, operation initialization or delegation were performed by this reviewer. The sole artifact written for this fix review is this note.
