# Independent recovered panel CPU deployment review

**APPROVE — recovered local package and deployment/repin delta. No blocking defect found in this scope.** This is permission to proceed through controller staging checks for one bounded CPU observation attempt, not evidence of actual-data compatibility, completed deployment, GPU readiness, or scientific acceptance.

Reviewed 2026-09-17; live pronghorn observation at **00:35:29 PDT**. Read Task 2, release plan, scoped fix review, and actual builder, packaging tests, packaged authority helper, CPU entrypoint, batch script and complete evaluation producer. Graft queried first; packaged producer was not returned by its graph query and was inspected directly. Source implementation `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558`, implementation report `1de033a`, scoped independent review `8508c4b` remain distinct.

## Independent recovered-byte verification

Evidence: [audit script](evidence/panel/audit.py), [actual PASS output](evidence/panel/audit-result.txt), [focused tests](evidence/panel/packaging-tests2.txt), [exit status](evidence/panel/packaging-tests2.status.txt).

- All seven transport payloads rehashed successfully. Transport SHA256 **`11d0550bd46c121c6b53e1cb51f0c528381ff41763e7e95e754d286a4d0448b5`** agrees with release plan. Entry point SHA256 **`2db226c71352c55b5e5b48133a4785e304fe323a936dc5356c1c33094c9cdb18`** is the final repinned executable, not the template hash.
- All 100 source paths match actual `git show 9603e20:<path>` bytes. Relative to b660 only `revision/scripts/mvp/mvp_produce_counts.py` changes. Git invariant diff is empty for src/train/configs/protocols and named probe/runtime files. No b660 GPU release or probe was modified by this review.
- Recomputed inventory `5570f32378928ccd627494652988a0ec8cda3c657cff5d63e0c496ec8c97010b`, 86-file component `ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e`, and decoder closure `b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088` agree. Tests execute packaged source verification and confirm exactly three literal template substitutions: source commit, component guard, inventory digest. Old b660 and initial mapping-template guards reject the new inventory.
- Complete source archive inventory equals inspected source tree: **113 regular files**, including 100 selected sources, three accepted companions, source inventory and nine sparse-authority files. No duplicate member names, traversal, absolute paths or links in any of the three archives. Every source member matches local bytes. CPU archive contains exactly the dependency receipt's **54 files**, each rehashed. Authority archive's **nine files** equal their embedded source counterparts.
- All **five original Git objects** validate in the sparse source repository and match original repository `cat-file` bytes. Manifest commit remains `92d8020ee79eeedd7f1d50167b1c8dc94700fb16`, manifest SHA256 `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`. Authority helper/archive, CPU site/receipt and bindfix1 batch are byte-identical to prior release.

## Execution, isolation and publication

- `build_release.py:32–91` refuses an existing destination. `run_panel_cpu.py:135–183` creates a new job output with `exist_ok=False`; exceptions leave diagnostics. No old panel manifest/arrays/COMPLETE are reused. Actual consumer stale-producer inventory rejection passes.
- Batch is CPU-only, 4 CPUs/16GB/1h/no-requeue, with 3500s timeout and 20s kill grace. Full mode is default; no `--reference-only`. It retains Singularity 3.6.1 bindfix: `/data/gpfs:/data/gpfs:ro` and dedicated output-parent default-RW bind, **no unsupported `:rw`**. Source/site/candidate/reference are under the RO project bind. No `--nv`, install or retry logic. This does not claim every container filesystem is read-only.
- Candidate paths come from fixed accepted companions; reference paths come from original Git authority. Mapping uses `duckdb.connect(..., read_only=True, config={'threads':1})`, exact 50-key bijection, reference hashes and geometry; no heuristic remapping or DB edits.
- Read actual `mvp_produce_counts.py:327–369,562–576` and CPU wrapper completion paths. Closed/fsynced exclusive `.pending` JSON precedes revalidation; WAL absence surrounds full source hash/stat recheck; final publication uses no-replace hard links. Compatibility precedes completion marker. Late integrity failure does not become successful compatibility/completion.
- Added a direct **packaged-code** tiny local boundary exercise: WAL present after staging rejects publication and preserves WAL/pending; positive publication succeeds; occupied final destination is preserved. This supplements, not replaces, attributed review8508c4b's 32-test serialization-hook coverage. Hard links are individually atomic, not a multi-file transaction or protection against later external writers. Compatibility alone is not completion.

## Fresh bounded tests

Executed existing packaging suite, not merely inspected old logs: **12 passed in 46.72s, exit 0**, under 180s timeout (+15s kill grace), no bytecode/cache writes, fixture outputs confined to this review's evidence directory. `bash -n panel_cpu.sbatch` exit 0. Separate 120s-bounded independent audit exited 0.

First pytest launch failed **before tests ran**: pytest FD capture hit `FileNotFoundError` while truncating its temporary file on this mount. Original traceback/status retained in `evidence/panel/packaging-tests.txt` and `.status.txt`. Retried unchanged suite with `-s` (capture disabled) and a fresh evidence fixture directory; no test/source edits or deleted evidence. No broad suite, production DB/hash scan, or container execution was performed.

## Live duplicate-submission check and controller boundary

[Read-only pronghorn evidence](evidence/panel/live-pronghorn.txt): no active `mvp50-input-audit`; account history since September 16 lists only original **6165556 FAILED127** and **6165559 COMPLETED0:0 /12m46s**, with both extern steps still OOM0:125. No newer panel submission observed. Planned `/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916` was **absent**. This is a timestamped observation, not a lock/reservation.

Controller must recheck live ownership/history and namespace absence immediately before exclusive staging; transfer only seven payloads plus transport; enforce canonical non-symlink ancestors, exact archive inventory, no overwrites/extras, RO source/site and fresh outputs/log paths; rehash at destination. Preserve 6165556/6165559 artifacts and all existing GPU/b660 state. One full fresh CPU attempt only; no automatic retry. Actual result acceptance still requires marker/receipt/output identity validation and accounting review; extern OOM cannot be relabeled clean.

**No ready-to-run controller submission command supplied:** remote staging does not yet exist, and destination hashes plus current approved account/partition/log arguments have not been verified in this review. This is a remaining execution precondition, not a defect in the approved local package. Do not execute the unrepinned Git template.

Review wrote only this report and its evidence subdirectory. No source/release changes, remote writes, submissions or cancellation. Advisor consultation failed client/model-version compatibility and provides no endorsement. Graft-reported savings: approximately 1,035,876 tokens.
