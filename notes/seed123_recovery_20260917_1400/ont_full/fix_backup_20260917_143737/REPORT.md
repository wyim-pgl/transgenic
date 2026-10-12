# Full ONT controller wrapper, local implementation only

## Status

Added `wrapper.py` and `test_wrapper.py`. No alignment, submission, remote command, canonical input hash, large data scan, scoring, source/data modification, or commit was performed. The mandated graft lookup reported an automatic graph refresh; no graph files were edited deliberately. Advisor consultation failed because the installed provider client does not support its configured model. No independent approval is claimed.

The wrapper consumes the existing pinned `gate.load_driver` and `gate.verify_alignment` interfaces. Exact CLI was inspected in `notes/wait_work_20260916/longread/frozen/revision/scripts/mvp/mvp_align_panel_longreads.py:133-233`, including its eight recorded commands, actual flags, and fixed output namespace. No pilot receipt is accepted as full-run evidence.

Scope is always `full_alignment_descriptive`, `structural_status=UNASSESSABLE`, `decision=null`, `training_eligible=false`. The unchanged nested gate result retains its more specific pending-inventory/policy structural label. Driver reference/scientific validation labels are not rewritten. One complete canonical ONT run still does not establish full scientific availability, pooling, HiFi policy, training eligibility, or structural scoring.

## Request and launch contract

This is an execution gate, not a scheduler script. No submission command or executable production request is supplied. A controller must independently review and retain both the exact wrapper SHA256 and request SHA256 outside the release. Supplying self-computed hashes beside mutable files is consistency checking, not independent approval.

The CLI requires `--request /absolute/request.json --request-sha256 APPROVED_SHA --wrapper-sha256 APPROVED_SHA`. Launch using the approved, resolved Python interpreter in isolated mode (`-I -B`). The request has these required fields:

- `scope`: exactly `full_alignment_descriptive`.
- `runtime_reviewed`: true only after the outstanding source/runtime/tool-fixture and resource review. This is a controller assertion, not proof supplied by this implementation.
- `release`: absolute nonredirected old longread release directory containing pinned `gate.py`, `source_pins.json`, `authority.tsv`, and `frozen/`.
- `control`: NEW absolute nonredirected private control directory, with existing parent, outside release, wrapper directory, and canonical data root. Existing directories are rejected, never reused.
- `run`: one of SRR28341943, SRR28341944, SRR28341945.
- `budget`: positive integer `cpus`, `memory_mib`, `wall_seconds`, `cleanup_seconds`. There are no default execution budgets, automatic retries, or invented full-run caps. CPU count must be at least four because frozen seqkit uses its four-thread default; this is a compatibility floor, not a budget recommendation. Cleanup must fit inside the reviewed wall budget.
- `tools`: exactly `minimap2`, `samtools`, `seqkit`, `python`, each an object with absolute resolved executable `path` and independently reviewed `sha256`. The three bare-command tools must occupy the same directory under their exact names. Python must match the running interpreter's resolved path.
- `scontrol`: absolute resolved `path` and reviewed `sha256` for the live scheduler query.

ROOT is deliberately fixed to `/data/gpfs/assoc/pgl/data/Transgenic`. Reads, roles, TAIR10 FASTA/GFF, and frozen panel are derived, not freely redirected by request. Reference MD5 flags come from the fixed-authority gate, never request-provided values. Old gate and source-pin manifest SHA256 values are hardcoded from the existing independent review. All nine source closure files are checked before gate/driver imports; package initializers and cached source/gate bytecode are rejected. Do not erase old artifacts to satisfy these checks; stage a separately reviewed fresh release.

## Allocation, capture, failure, and publication

The gate requires a live owned RUNNING Slurm job, one node and one task, exact requested CPU and memory allocation, no GPU TRES, matching compute hostname, exact local CPU affinity, and enough remaining allocation time. Environment variables alone do not suffice. Unsupported scheduler output, compressed node naming, or memory-unit formatting fail closed; runtime compatibility must be tested before deployment. This does not establish controller queue/overlap approval.

A fresh mode-0700 control namespace records an initial INCOMPLETE receipt before allocation/source/data preflight. Scheduler stdout/stderr and driver stdout/stderr are exclusive files. The canonical driver's output directory must be absent both before input checks and immediately before launch. Driver `mkdir(exist_ok=False)` handles the final race. No existing driver output is removed or reused.

The real frozen CLI runs in a new process group with a restricted PATH, clean environment, caller budget, and TERM then KILL cleanup of the entire group. SIGTERM, SIGINT, and the total wall alarm trigger incomplete capture; cleanup blocks repeated termination signals until the group is handled. The total alarm also bounds preflight and postflight. stdout/stderr identity, argv, exit status, scheduler details, source/input/tool identities and the recursive driver inventory feed the outer receipt. Canonical inputs, code and tools are rechecked after gate postflight. COMPLETE is written only after successful exit, inventory/postchecks and identity rechecks. Exception handling withdraws only this controller's own COMPLETE marker if needed, never driver artifacts.

Failure retains a checkpoint and FAILED diagnostic. The recursive failure inventory is not guaranteed: avoiding unbounded post-timeout hashing takes precedence. A SIGKILL, node loss or OOM cannot execute Python cleanup or guarantee a final receipt update. The preexisting INCOMPLETE checkpoint and scheduler accounting remain authoritative in those cases. Keep all incomplete outputs; there is no automatic recovery/reuse.

Controller acceptance must still require independently pinned release/request, final parseable worker stdout equal to receipt, absence of FAILED, COMPLETE hash agreement, unchanged stdout/stderr and output inventory, compute-side `gate.verify_alignment` with recorded args/receipt, and successful batch/extern Slurm accounting. COMPLETE alone is not authorization. Files remain mutable on disk; preserve the returned receipt identity independently. Tool ELF/shared-library and Python runtime closure compatibility is a runtime-review gate, not certified by executable hashes alone.

## Local verification and remaining gates

Initial test-first run failed because `wrapper.py` did not exist. After implementation, eight tiny local tests passed: wrong pin, exclusive no-reuse directory, pilot scope confusion, stdout mutation, recursive output mutation using the actual pinned old gate, nonzero child exit, real child success, real child timeout and retained logs, plus rejection without a real allocation. Multiple assertions share tests. These tests use synthetic temporary files and harmless Python children, not canonical data or actual alignment tools.

The full `execute` path has NOT been exercised end to end. The tests are wrapper/receipt-boundary checks, not a real minimap2/samtools/seqkit fixture, live Slurm validation, scientific result validation, or a deployment-readiness claim. No real tool fixture was available here. Production execution remains blocked pending that fixture and source/runtime review, explicitly reviewed CPU/memory/wall/scratch budgets, environment compatibility, and controller approval. The underlying externOOM is not solved by this wrapper. SIGKILL and late controller stdout loss still require external accounting checks.

TODO completed locally: frozen interface inspection, minimal wrapper and synthetic boundary tests, explicit incomplete/descriptive receipt contract. TODO blocked before actual execution: real tool fixture, live allocation and runtime/source review, externOOM/resource investigation, independent controller pinning and approval. No downstream structural scoring TODO is authorized by this task.

Graft reported approximately 1,072,880 tokens saved across the orientation and source queries.
