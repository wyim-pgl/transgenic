# EST blastx controller verification — 2026-09-14

**Restart update (2026-09-15):** final reviewed launcher is950cf4c, notcc9735f below; getseq and version corrections passed26controller tests and scoped reviews. Actual fixture6163396 passed both sources/all6cases. Real searches6163851/6164049 are submitted; current accounting and outputs are in [restart handoff](RESTART_HANDOFF_20260915.md). Earlier unsubmitted/review-running statements below are historical snapshots, not current gates.

## Implementation snapshot

Preserved tool branch: `pi-agent-462c26cc-1192-466`, HEAD `cc9735f`, base `9a8c8e1`. Branch-local commits `278cca7`, `5e0ab24`, `cc9735f`. Original main not merged or modified by implementation. Branch also contains a copy of already-uncommitted English workplan edits; do not blindly merge over the current main documents.

## Controller checks (fresh)

- Extracted git archive of preserved branch to temporary local directory.
- `python3 -m pytest -q tests/test_orthodb_filter.py revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py`: **22 passed in 0.34s**.
- `bash -n revision/scripts/evidence/est_blastx_fixture.sbatch`: exit0.
- `git diff --check 9a8c8e1..pi-agent-462c26cc-1192-466`: exit0.
- pronghorn interpreter `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python` and DIAMOND executable exist and executable.
- Current queue inspection showed no blastx fixture job; target checkout/output paths unused before staging.

## Remote staging, not execution

Staged only the lineage helper, launcher, fixture, fixture sbatch, and plan/report into new isolated code directory:
`/data/gpfs/assoc/pgl/data/Transgenic/worktrees/b6-est-blastx-reviewed-20260914`.

This is an immutable source snapshot, not a remote git worktree. `SOURCE_COMMIT.txt` records cc9735f; SOURCE_FILES.sha256 records four executable/helper files. SHA256 lists independently computed locally from git archive and remotely compare byte-for-byte (diff exit0). No existing code directory replaced.

Independent code/safety review running as agent `3ee31928-096d-47f`. Review must distinguish synthetic fixture blockers from production blockers. **No sbatch submitted yet. Real DIAMOND fixture not run.** Expected fixture allocation4CPUs/8G/30min; production resources not established.
