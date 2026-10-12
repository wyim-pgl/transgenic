# Checkpoint archive implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for implementation and independent review.

**Goal:** Build tested automatic archive tools for completed checkpoints to DeltaAI HDD and pronghorn, preserving live resume state and validating every removal.

**Architecture:** A standard-library local transaction module owns candidate selection, manifests, publication and source cleanup. A CLI/remote orchestrator invokes that same module on the DeltaAI source and pulls immutable files to pronghorn; a bounded watcher applies the configured policy without touching trainer code.

**Tech Stack:** Python >=3.6, unittest, POSIX file locks, SSH/rsync, Lustre lfs probes.

**Spec:** notes/checkpoint_archive_20261009/SPEC.md

## Global constraints

- Python 3.6+ stdlib; immutable epoch directories; protect latest two and best; never latest_state, temporary or final step directories.
- No training-code or queued-job modifications. Production data remain intact during tests and deployment.
- Host small-file tests and read-only inventories authorized; new GPU jobs and bulk destructive cleanup not part of validation.
- Review every source-removal path. Destination pool/space must be verified on the real host.

## Task 1 — Local archive transaction

**Files:** create revision/scripts/b6/checkpoint_archive.py and tests/test_checkpoint_archive.py.
**Interfaces:** expose documented functions/classes for plan, prepare/snapshot, transfer, verify and opt-in finalize/cleanup. JSON-serializable manifests/receipts; core callable from Task2. Decide exact names in implementation report before Task2 begins.

- [ ] Write executable unittest fixtures using actual epoch directories (meta.json, eval.json, model.safetensors, accelerate_state, optional pending gradients). A run with epochs01..04 and best01 must offer only epoch02; incomplete/symlink checkpoints are refused.
- [ ] Watch the new behavior fail before implementation.
- [ ] Implement locks, protected-candidate selection, bounded streaming copy, independent destination hashing, source-stability checks, durable receipt publication and safe recoverable cleanup.
- [ ] Add red/green tests for exception or SIGKILL at copy/publish/cleanup boundaries; corruption; changed source; existing identical/conflicting archives; symlink/path overlap; concurrent locking; pool and free-space errors. Prefer real subprocesses for process-level guarantees.
- [ ] Run focused suite, self-review, commit source/tests, deliver API contract and exact tests/results. Independent review and fixes before Task2.

## Task 2 — Remote transfer and bounded automation

**Files:** create revision/scripts/b6/checkpoint_archive_cli.py, tests/test_checkpoint_archive_cli.py, deploy/deltaai/checkpoint_archive.example.json, docs/checkpoint_archive.md. May extend Task1 API only with reviewed justification.
**Interfaces:** consume Task1 manifest/verification/cleanup APIs; produce CLI plan/once/watch and remote helper protocol used on DeltaAI and pronghorn.

- [ ] Write failing CLI tests that run the real command against tiny trees and controlled SSH/rsync boundaries; verify default read-only behavior and explicit transfer/removal behavior.
- [ ] Implement both HDD-local and pronghorn-pull targets. Remote paths must be safely quoted and operations bounded by timeouts; never trust only transport success for removal.
- [ ] Implement config limits, age/keep rules, one pass vs watch, progress/error JSON, fail-closed destination checks, dry-run inventory with no hashing/large reads, and bounded retries without hiding failure.
- [ ] Test auth loss, transport interruption/corruption, replayed acknowledgment, changing source, quota/pool mismatch, duplicate workers, process restart, and shell-special-character paths.
- [ ] Run tests repeatedly with varied seeded ordering/fault timing, self-review, commit and obtain independent review.

## Task 3 — Host verification, installation and handoff

**Files:** operational evidence under main checkout notes/checkpoint_archive_20261009/; update .atl/resume.md, .atl/todo.md and linked progress.

- [ ] Inspect live DeltaAI jobs/quotas and verify pronghorn Python/SSH direction/storage; never resubmit training jobs.
- [ ] Deploy versioned tools to a separate operational directory, compare SHA256 to reviewed local code.
- [ ] Run stdlib suite and multiple tiny cross-pool transfers on DeltaAI login node; verify ddn_hdd placement, content hashes, preserved live-state fixtures and successful fault recovery.
- [ ] Run actual tiny DeltaAI→pronghorn pull with cleanup confined to generated fixture; repeat, interrupt and restart; verify archived bytes independently.
- [ ] Run read-only production plans for seed123 and future WP5 runs. Record protected/candidate counts, actual destination headroom and available first batch.
- [ ] Whole-branch review; fix material issues; report commit, test counts, installed paths, activation command and any host/auth limits. Keep production watcher inactive until its bulk I/O location/limits are concretely reviewed; do not imply installed means active.
