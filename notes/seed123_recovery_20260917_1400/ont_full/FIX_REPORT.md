# Scoped F1-F3 remediation

## Disposition

Implemented only F1-F3 in `wrapper.py`, `test_wrapper.py` and `REPORT.md`. Read `../ONT_FULL_REVIEW.md` completely. The original three files were copied before modification to `fix_backup_20260917_143737/`. No original review proof, old gate, source pin or frozen source was edited. A pre-change inventory of 93 files under `review_evidence/` and the old longread release was rehashed after testing: all original hashes match.

No actual data, remote command, installation, scheduler query, alignment, submission or commit was used. Graft orientation/query attempts each timed out after 30 seconds; exact supplied paths were inspected instead. Advisor consultation failed because its installed client does not support the configured model. No advisor approval is claimed.

## Changes

- F1: `check_source_closure` rejects any frozen-tree entry except the nine manifest files and their necessary parent directories. It rejects missing files, extra source/data/bytecode/initializer files, unexpected directories, symlinks and nonregular entries. Validation occurs before any gate/driver import and again after postflight. Existing hash checks remain. The manifest hash and frozen invocations are unchanged. Exact inventory is intentionally stricter than the historical tree, which contains ancillary unlisted material. Do not prune that original tree; separately stage and approve an exact deployment copy.
- F2: `execute` explicitly requires Python >=3.9 before any request/path access, including `Path.is_relative_to`. This floor does not replace independently approving and pinning the exact deployment interpreter and runtime closure. Tests now traverse the real wrapper execute/publication path with explicitly named synthetic substitutions, on installed 3.10.12 and 3.12.2 interpreters.
- F3: both stdout and stderr are compared to their original captured identities at each receipt boundary. Final diagnostic stream identities must also equal those originals before receipt/COMPLETE publication. Postflight mutations reject with INCOMPLETE, FAILED and no COMPLETE.

## Test-first evidence

All logs are in `fix_backup_20260917_143737/`.

1. Added regression tests before changing `wrapper.py`. Command: `/usr/bin/python3 -B -m unittest -v test_wrapper.py`. Result: 14 tests, 8 failures, exit 1 (`red_tests.txt`). Failures reproduced extra directory/symlink/FIFO acceptance, postflight stderr and closure mutation acceptance, both malicious import markers executing before rejection, and missing explicit Python-version rejection. Clean synthetic execution and actual nested helper imports already passed. Existing initializer/bytecode/missing-file/redirect checks continued to reject in their covered cases.
2. Applied the minimal production changes. Final command: `/usr/bin/python3 -I -B test_wrapper.py -v`. Result: 14 tests, OK, exit 0 (`green_python310.txt`).
3. Final command: `/tmp/mvp-final-source-env/bin/python3.12 -I -B test_wrapper.py -v`. Result: 14 tests, OK, exit 0 (`green_python312.txt`).
4. Actual local Python 3.8.20 CLI with nonexistent request and deliberately invalid pins: `python -I -B wrapper.py --request /nonexistent/request.json --request-sha256 0 --wrapper-sha256 0`. Result: exit 1 with `ValueError: Python >=3.9 required; use the independently pinned deployment interpreter`, before any pin or request-path failure (`python38_rejection.txt`). The suite additionally simulates 3.8 at the implementation entry point.

The unlisted `src/transgenic.py` and `revision/scripts/json.py` regression cases now assert that the gate was never imported and their markers were never written. Other tests cover a correct clean tree, missing/extra entries, file/directory symlinks, FIFO, happy publication, nonzero child exit, timeout, input mutation, recursive output mutation, stdout/stderr mutation and an added shadow source during postflight.

Synthetic execute tests preserve actual wrapper pre/post checks, source pins/hashes, original gate inventory routines, real tiny Python process handling and receipt/COMPLETE hashing. They substitute scheduler approval, canonical/reference bindings, driver object/panel loading, scientific postflight and child workload. Dummy tool identities are not real tool execution. These tests are not a production end-to-end fixture.

The actual frozen sample, orientation and junction helpers were each invoked with `--help` from an isolated parent using their unchanged non-isolated nested Python invocation style, correct copied layout and a clean environment. All succeeded on both supported test interpreters. This checks import layout, not scientific behavior. No frozen script was edited. The original vulnerability proofs remain intact and intentionally demonstrate the old weaknesses when bypassing the new wrapper preflight.

## SHA256 identities

| File | Before | After |
| --- | --- | --- |
| `wrapper.py` | `887608643b1136e2ac64f288eef6b53b25690f258c23902e3d61a3acd9537b91` | `314d6027d9d146d63c6f1b2e2b71783ac5e1f63e95e6e31252614ddcf20728dd` |
| `test_wrapper.py` | `d527860b7d9ef61a0c1f6fde0f727a4b7bd3e4b774b174b0d682936e2f404cba` | `4cb563cbb7726456c6c6d053e99180a7acf3aed03545ea90b72441f26482c38a` |
| `REPORT.md` | `06abcf115c61e6408608569c624ed0598895b3a2f76a01e904206a28e21297bb` | `6ff568f8f459ea07cba19c7bd4281d67b7b388758ea9e682642be9bd70ecf462` |

Unchanged old gate: `0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f`.

Unchanged source manifest: `38c56c608336ef3fdcb83eaa5eb35a355ddc77ea1e8aff8387160603bf88ce34`.

Unchanged review harness: `8759dc04b3c20e9c67e5744d1c94d734f66ec3e470b3795952faa678b395c61c`.

Evidence log hashes:

- `red_tests.txt`: `1f6ff112e99a98074222f467882ab5921f5fda0d8e1d77043aa64ed7c19ec20c`
- `green_python310.txt`: `3343c602bd5cddf8bbd0d3a68c69c67ed79c62b9449531c5a59d656b06a34a87`
- `green_python312.txt`: `c317590ffb962c58f0a04b1e7c1be5e0c21d473ce32568b6cb650a827d40e4be`
- `python38_rejection.txt`: `bd4b41d2c0a98f45c96f8a31a49e013138c4d1a8a1e5e59de3bbf0fa544270e1`

## Remaining blockers and exact trust limit

Completed locally: F1 closed-layout validation/revalidation, F2 clear version gate and supported-interpreter synthetic execute coverage, F3 both-stream identity consistency. No scoped implementation TODO remains.

Production execution remains BLOCKED. The controller must independently verify and keep the exact deployed tree immutable for the whole execution, approve the interpreter installation/runtime closure, and retain approved request/wrapper/source/tool identities. Before/after hashing is not an immutable snapshot and cannot stop concurrent swap-and-restore attacks. Non-isolated nested helpers still load approved system site configuration and installed runtime packages. A clean environment and exact script tree exclude local injected shadow candidates, but do not certify the external Python installation. This runtime trust gate cannot be removed by this patch without widening the release scope.

F4 remains BLOCKED: minimap2 is absent, and no real minimap2/samtools/seqkit fixture through the actual driver, ledger and scientific postflight was run. Live Slurm parsing/accounting, cgroup cleanup, explicit CPU/memory/wall/scratch resources, externOOM investigation and controller approval remain unverified. Process-group cleanup is not complete containment of descendants that escape their session. No actual50/full structural acceptance, runtime acceptance, pooling, scores, missingness policy or training eligibility is granted.
