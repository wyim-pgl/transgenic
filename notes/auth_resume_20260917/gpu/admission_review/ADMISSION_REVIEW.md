# Focused admission review

## Verdict

Two remaining P2 robustness findings, both in the query boundary. No additional deterministic genuine-job blocker was established beyond the explicitly excluded reconciliation fixes. This is not execution approval and does not review the separately changing deployment_v3.

Reviewed all 414 lines through `live()` in `deployment_v2/admission.py`, the host authorization/admission call path, launcher, executable schema, both admission test files, IMPLEMENTATION.md, OLD_P1_RECONCILIATION.md, and recorded raw scheduler/account-script evidence. No source edits, remote calls, scheduler submissions, or cancellations. All new artifacts are in this directory.

## Findings

### P2: Non-timeout communication failures do not clean up the detached query process

Location: `deployment_v2/admission.py:115-143`, especially the generic exception handler at 132-133.

`Popen(start_new_session=True)` detaches the query process group. Only `TimeoutExpired` executes `killpg`; another `communicate()` exception merely records an error and proceeds to rejection. A pipe I/O failure therefore has no kill or reap path. Admission correctly fails, but the scheduler client or account-script descendants can remain running after the host exits. The fifteen-second timeout is implemented by the now-abandoned caller, not by the child itself.

Pure mocked reproduction: `communicate()` raises `OSError`; the query rejects with `failed/incomplete observation`, persists the error, and makes zero `killpg` and zero `wait` calls. This demonstrates the missing cleanup branch, not an observed remote leak. Evidence: `review_evidence.json:communication_error` and `verify_review.py`.

Minimal correction: on any exception after a process was created, terminate its process group and boundedly reap/drain it, preserving the original error and evidence. Add a non-timeout exception case alongside the existing timeout test. No retry is needed.

### P2: The 16 MiB query ceiling is enforced after unbounded capture and evidence writing

Location: `deployment_v2/admission.py:125,135-143`.

`communicate()` first buffers all stdout and stderr; the finally block then hashes and serializes both full strings. Only afterward is stdout length checked. Stderr has no size ceiling. Thus this is an acceptance limit, not a resource bound: a large query response can consume memory and control-directory storage before rejection, potentially failing the actual diagnostic without a decision record. The runtime uses a global visible-queue census, so response size is not exclusively controlled by this campaign. The retained implementation probe already illustrates that scheduler output can reach megabytes, although it does not prove a current runtime response will exceed the limit.

Pure mocked reproduction: a 16 MiB plus one character stdout is fully persisted into a 16,777,616-byte JSON file, then rejected as exceeding the reviewed bound. Evidence: `review_evidence.json:output_limit`. The temporary oversized file was removed by the fixture.

Minimal correction: enforce an aggregate stdout/stderr byte bound during capture, terminate and reap on overflow, and save a bounded failure record. Do not remove the global overlap check or silently truncate a response and admit it.

## Checked constraints and evidence

- Durable permission versus observation freshness: authorization has no queue-age expiration, but rejects future issuance and binds external hashes and exact scope. `host()` validates it and creates only control evidence before calling `live()`. Payload verification/copies/probes follow admission. The persistent launch claim survives rejection. The known old-evidence-path issue is intentionally not repeated here.
- Actual query boundary: no input observation-file API; `live()` constructs its own commands and checks local UID. Local tests mock subprocess output rather than invoking real scheduler commands. Two scheduler snapshots detect relevant-job changes; no retry is performed. This remains a read-only snapshot with explicitly required controller serialization, not a future-activity guarantee.
- Accounting and caps: allocation rows are keyed by integer job ID; duplicate allocations reject, recognized steps are ignored, prior diagnostic IDs must be present, unknown diagnostic history and active diagnostics reject. Current elapsed is not added twice. The full 0.5 node-hour/2 GPU-hour self reservation plus declared reservations and spent history is checked independently against 5 node-hours and 20 GPU-hours.
- B5 reserve: live pending jobs use requested GPUs, live running jobs allocated GPUs, both multiplied by the complete maximum wall limit, not remaining time. All live B5 candidates require the expected owner/account/partition/name/command. Full diagnostic envelope and margin add another 40 GPU-hours. Node overlap is checked for relevant B5 and the final global active queue.
- Parsing/time: bounded known wall formats, nonnegative numeric accounting, malformed/duplicate allocation rejection, future accounting timestamp rejection, ambiguous/nonexistent Chicago civil-time rejection, command success and nonempty warning-free output checks, and final 300-second monotonic observation age are present. These checks do not establish freshness of Slurm's internal database itself; the current allocation must nevertheless appear as running. No new database-freshness framework is requested.

## Account unit grounding

This conversion is grounded in recorded script bytes, not merely the label `Balance(Hours)`:

- Extracted the complete recorded `/sw/user/scripts/accounts` source from `deployment_v2/review/machine-probe.json`; its 8,705 bytes independently hash to pinned `44302f2e6bd0773e1acd87c521947e3dac4ca9c581e9c79b7433489f22be35f5`.
- The Perl script uses actual `scontrol show assoc qos=<account> flags=qos -o` GrpTRESMins billing counters, expansion factor 1000 and 60 minutes/hour. It obtains the caller via effective UID, so the sanitized query environment does not require USER for that lookup.
- Recorded actual counters are `billing=89940000(20667433)` and `gres/gpu=N(20667)`. Billing use divided by 1000 differs from GPU minutes by 0.433, consistent with the one-minute tolerance. Allocated SU-hours are 1499; the recorded displayed balance is 1154.
- `unit-factor-probe.json` records the actual account QOS factor `1.000000` and flags `NoDecay,UsageFactorSafe`; live admission rechecks these, the script path/hash/metadata, counter consistency, report balance, and billing=1000 per GPU for allocations being costed/reserved.
- These are expanded billing service units, not universally synonymous with GPU-hours. The recorded partition weights include CPU and memory as well as GPUs. The per-allocation billing equality guards the conversion for this workload, rejecting incompatible charging geometry rather than assuming every account job is GPU-priced. The historical balance is not used as a runtime constant.

No additional finance conversion bug was demonstrated against the recorded fields. Exact extracted conversion lines and QOS record are in `review_evidence.json:finance_grounding`.

## Verification and scope limits

- `PYTHONDONTWRITEBYTECODE=1 python admission_review/verify_review.py`: passed, local mocked adversarial checks and recorded-source hash validation only.
- `PYTHONDONTWRITEBYTECODE=1 TMPDIR=<admission_review/tmp> python -m unittest test_queued_gate test_admission_cases`: 15 tests passed in 0.440 seconds; output in `local_tests.txt`.
- Source/test SHA256 values are recorded in `review_evidence.json`. Test fixtures were not changed. No runtime pass is inferred from synthetic fixtures or historical probes.
- The four explicitly reconciled old failures were excluded from findings. No new approval framework or whole-account security proof is requested.
- An advisor consultation failed due to client/model-version incompatibility; this report does not claim advisor review.

Completed: focused source review, local bounded reproductions, finance provenance validation, evidence and findings. Outstanding: implementer consideration of the two query-boundary fixes and independent review of the separate new package. No execution action authorized.
