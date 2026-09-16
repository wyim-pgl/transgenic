# C1 receipt/reconciliation follow-up (base 5c7a0aa)

Scope authorized by controller: auditor, CPU tests, documentation only. Immutable deployed baseline, jobs, raw inputs, dependencies and ignore files are untouched. No C2/A46 semantic changes.

1. Add negative and positive CPU regressions, observe RED, implement stdlib-only checks, run the existing 119 recovery/review tests plus new cases.
2. Pilot: optional local recovered JSON receipt must match the independently measured historical extract SHA256 f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23. This binds original session source, records 540/541, call/result IDs, explicit remote D/output/error paths and job 6163851. Require exact historical audited root, submission.lock/jobid bytes, run.sbatch approved pin, and both named logs (empty stderr valid). No receipt means UNRESOLVED.
3. Optional external reconciliation manifest + separately supplied reviewed SHA256. No manifest generation from current filesystem, approval boolean, wildcard entries or marker fabrication. Validate all manifest evidence before enabling any dispositions. Errors become tree FAIL, not lost 18-unit inventory.

## v1 schema

JSON object, exact keys: `schema` = `est-c1-reconciliation-v1`, `blastx_root` (absolute historical root), `accounting` and `queue` (each `{path, sha256}` external regular capture), `entries` (nonempty list).

Accounting capture: pipe-delimited JobID|State|ExitCode rows; exact allocation/.batch COMPLETED 0:0 and .extern OUT_OF_MEMORY 0:125 for each bound job. Also checked against primary C1 sacct for search jobs. Queue capture JSON: exact keys `captured_at` (timezone-aware ISO8601, at most 1h old), `command` = `squeue --noheader --format=%i|%T|%j`, `stdout` (raw output), `returncode` = 0. Any EST-named queued job or historical job ID fails (including pending/held jobs). Captures and manifest are independently hash-bound for review, not scheduler authentication.

Each entry: exact keys `path` (absolute), `jobids` (fixed historical IDs for this path), `wrapper` (absolute producing wrapper path), `wrapper_sha256` (hardcoded historical pin), `snapshot`. Snapshot recursively records exact kind (`directory` / `file`), owner (`uid`, `gid`), and either `children` mapping or `sha256`. Every child is checked; symlinks, special files, unknown marker variants, malformed DONE or FAILED.json fail. Reconciliation only changes NEITHER to historical reconciliation, never claims DONE. Regular preflight execution.lock is explicitly checked too.

Allowlist: pilot tmp/execution.lock/submission.lock; remaining8 submission.lock and eight exact species tmp; preflight root/logs/receipts/submission.lock/tmp/xdg-cache/execution.lock. Scratch and execution/submission sentinels must have the known empty or jobid-only contents. Preflight root snapshot binds all descendants including five receipt files, lineage hashes, source DONE, preflight sbatch receipt/jobid/batch exit and producing prepare.sbatch pin. Missing any actual hash means no usable manifest yet, not an invented value.

Trust boundary: reviewed manifest SHA and independently captured receipts are supplied by controller outside input/source trees. Auditor cross-checks live bytes, fixed paths/jobs/wrapper identities and accounting; it does not prove scheduler authenticity or live flock ownership. Extern remains unresolved, clean_accounting/accepted_phase_evidence/production_authorized remain false. Review must not treat capture freshness as continuous locking.
