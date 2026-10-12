# Controller check — Task 4

Accepted the local reproducibility inventory and non-destructive recovery runbook as snapshot documentation, not runtime/scientific approval.

Fresh controller checks:
- `python3 -B notes/wait_work_20260916/repro/preflight.py`: PASS_LOCAL_INTEGRITY_ONLY,25 captures/100 Git-equal archive files/86 source-pin files/13 decoder-pin files.
- `sha256sum -c captured.sha256`: all25 PASS.
- Final-P1 wrapper `bash -n`: PASS.
- Read the entire RUNBOOK and exact b660 decoder source lines136–147: line141 converts IDs/mask; line142 converts tracks. The earlier report's shorthand pinpoint is corrected in this additive record; original reports/logs preserved.

No CPU rerun or remote query in this controller check. Zero native-crash reproductions were performed by Task4 because original predecessor IDs/native backtrace are unavailable. The source-level correction does not establish a cause or fix. The original fatal log and separate dependency-failure logs remain distinguishable.

Jobs and budget in this bundle are historical at the explicitly recorded21:14PDT observation. Current job status/charged budget must be remeasured before action. No collection script may be rerun over the existing capture directory. The self-consistency checker does not authenticate fresh remote artifacts or authorize resubmission.

Remaining: update the runtime ledger with future actual job/receipt results; obtain native crash core/backtrace and exact predecessor/environment if a justified reproduction becomes possible. No full-repository green/reliability claim.
