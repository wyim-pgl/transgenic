# Independent deployment review: release_v2

## Verdict

**BLOCKED. Do not transfer as an accepted executable release, cancel old3163986 for this candidate, or submit the replacement on this review.** The packaging/profile adaptation passes the bounded local checks, but the controller gate is incompatible with an ordinary Slurm queue. A pending-job hand-refresh of its external pin is not an acceptable deployment procedure.

No additional concrete deployment blocker was found in the requested builder, profile/seed/role, archive, mounts, quota and original-safeguard checks. This is a scoped review, not a full review of all archived source or a claim that no other defect exists. The blocking gate correction below must be reviewed in a new immutable release, not patched into release_v2.

Scope remains one conditional S0/P1 allocation: one node, four GPUs, at most1800s,0.5node-hour/2GPU-hours, within the existing cumulative5node-hours/20GPU-hours. Startup, checks, copying, hashing, failures and teardown count. Training and followups are not authorized. No new financial or training cap is inferred.

## Verified identity and local evidence

Selected candidate only: `release_v2/` and `release_v2.tar`. The superseded sibling `release/` was not reviewed. Historical `notes/mvp_execution_20260916/final_p1/` was read only as the original safeguard/test reference.

| Item | Independently measured result |
| --- | --- |
| Transport SHA256 | `5232741b2e7a0b722d9d63d3a62d23ff1cee17fd42486979b081bd17cd731ab9` |
| Source archive SHA256 | `da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7` |
| release.json SHA256 | `03099aa9d6e40a33e1b66365cc10d5c8a4cb7a264a2015e467baec93ebb3e281` |
| release.sha256 SHA256 | `466d23365d45718a4d3b023123ce8098161d57f5a33954bda5b6575dac88917e` |
| Source closure |103 regular files, exact per-file hashes and membership |
| Source comparison |89 entries, `4a68eb3d299cd7e7cf7af9ce4e774f2ff18359f094d425cc9cde7becf4d67e97` |
| Literal DECODER_ONLY closure |14 entries, `a651b9fbec9e477198d15c401276d8b45240d5d16212357e1c6eef73e7a67800` |
| Adapted/candidate tests |19 passed, exit0,16.218s |
| Original unchanged tests |11 passed, exit0,37.589s |
| Selected shell syntax and ledger |Both passed |
| Original and selected payload preservation |Before/after file inventories and hashes identical; transport unchanged |

`collect_evidence.py` ran both suites in separate Python processes, with bytecode disabled, temporary writes under `review/tmp/`, and a180s timeout per command. The top-level adapted scripts tested are byte-identical to the selected release scripts. Tests are local fixtures, not Slurm/SIF/GPU measurements. The complete original directory also matches `originals-preservation.json`.

The transport's exact regular-file inventory and bytes equal release_v2, without duplicate members, links, special files or escaping names. Nested code archive membership and hashes equal `release.json`'s103-file manifest. Canonical aggregate hashes were independently recomputed from the payload, not merely copied from operator-pins.json. The controller's separate comparison of those103files to its scoped source approvals is accepted as supplied evidence, not mislabeled as this review's source-worktree comparison.

## Blocking finding: an immutable observation expires while queued

Locations: `release_v2/deploy.py:202-233`, host call at `:245`; `release_v2/p1.sbatch:32-42`; `controller_execution_scope.json:execution_gate`; IMPLEMENTATION's queue-refresh paragraph.

1. `validate_controller_gate` requires `0 <= now - measured_at_unix <= 300` at host entry. `controller_gate` requires the entire file to match `DEPLOY_CONTROLLER_GATE_SHA256`.
2. Host entry occurs after Slurm allocates the job, not when the controller submits it. The supplied preflight observed old3163986 PENDING at2026-09-17 18:10:48CDT and predicted start2026-09-18 08:44CDT. That prediction is not a guarantee, but already demonstrates the relevant hours-long queue scenario.
3. `gate-boundary-results.json` reproduces acceptance at age300s and rejection at301s,1hour and24hours against the actual selected validator. It also verifies that changing the observation timestamp changes the external digest. Fixture timestamps are explicitly test-only; no authorization file was created.
4. Refreshing the on-disk file alone breaks the queued digest. Updating both the queued job and the file requires a timed controller intervention and coordination the package does not implement. Waiting for likely start, advancing the timestamp, expanding its freshness interval, or deleting freshness would not establish current accounting/no-overlap.
5. `S0_P1_STARTED` is claimed before host entry. Thus the normal long-queue case consumes the one-shot claim and some charged allocation time, then fails before useful S0/P1 work. The claim must not be removed to retry this release.

The validator is explicitly an attestation parser. It executes no live scheduler/accounting query; the only existing host live resource measurement is storage quota. Consequently, deleting its age predicate would remove the stale-observation defense without replacing it. This is a deployment usability blocker with a necessary safety-preserving fix, not evidence that the current code runs unguarded.

## Exact minimal usable correction

Keep the existing one-shot launcher, staging, quota, mounts and source/probe code. Split the gate's two responsibilities within the deployment wrapper, without a background refresher, another allocation, or a new orchestration service.

### 1. Durable externally pinned authorization

Retain a canonical owner-checked JSON and externally supplied SHA256, but make it an authorization rather than a300second observation. It must bind:

- Exact new release identity and external transport/ledger acceptance; development-single, exact integer123, mvp_finetune; training=false and followup=false.
- One replacement in the fixed namespace; one node/four GPUs;1800s,0.5node-hour/2GPU-hours; unchanged cumulative5node-hours/20GPU-hours; no requeue, automatic retry, restart, resume or extra submission.
- Controller identity/review references and the old3163986 identity. Actual evidence must establish controller-authorized cancellation after fresh owned/PENDING/elapsed0/noAlloc verification, followed by confirmed terminal cancellation. Preserve old receipts and old3163545 in the accounting scope. A boolean invented by the deployer cannot establish this history.
- The controller's diagnostic accounting scope/start boundary, approved baseline-preservation/account-reserve/margin policy, staging ownership mapping, and reviewed exclusion policy. Do not substitute the limited since-Sep16 reconnaissance window for complete accounting.
- The controller's reservation of this attempt's maximum cost while it is pending and until terminal reconciliation, with serialized MVP diagnostic admissions. Preserve B5 jobs and their reserved capacity through their legitimate lifecycle. This addresses the race that a read-only snapshot alone cannot prevent.

Keep the true authorization issuance and historical observation times as provenance. Do not subject durable permission itself to a300second age test or alter old measurement times. Any revocation or changed scope remains controller-controlled; durable does not mean authority to ignore a revoked/cancelled allocation. The independent pin must be established before submission and remain unchanged across the queue wait. No per-job queued-environment refresh is required.

### 2. Fresh trusted read-only checks at compute start

Add one small host preflight, inside the existing1700s outer bound, after validating durable authority and before large hashes, evidence rename, cache/checkpoint copy, S0 or P1. Reuse the existing exclusive claim and fresh output namespace. Write actual raw query outputs, command identities, query start/end times, exit status, digests and derived decisions in the fresh job's control evidence. Do not accept a caller-provided fresh JSON as a substitute for the queries.

Required checks:

1. Use trusted, reviewed scheduler/accounting executables with explicit bounded subprocess timeouts and version-supported machine-readable output. Query the actual current allocation, current relevant pending/running jobs, old jobs, and complete authorized diagnostic accounting. Validate cluster/job identity, UID/account, one node/four allocated GPUs, actual time limit at most1800s, no requeue/restarts, and actual placement/exclusions. Environment values and sbatch directives alone do not prove the submitted resource request was not overridden.
2. Verify old3163986 is terminal cancelled with no allocation charged under the authorized replacement history. Missing/lagging/conflicting scheduler/accounting records, evidence of an old running/racing job or unexpected consumption must fail closed for controller reconciliation. Compute code must never cancel anything.
3. Establish no overlapping MVP diagnostic allocation, excluding only this exact current allocation, not all jobs with the same name. Include other outstanding reservations and relevant jobs outside a short recent window. Preserve B5 jobs, account reserve and the existing margin policy. Query the actual account capacity/usage mechanism already trusted by the controller; invent neither a balance nor a financial ceiling.
4. Reconcile cumulative node-hours AND GPU-hours. For the current allocation reserve its full0.5node-hour/2GPU-hours once, including elapsed startup, rather than counting current elapsed plus the full reservation twice. Include completed/failed allocations and all other unresolved/pending reservations conservatively. Do not double-count `.batch`/`.extern`/step rows as separate allocations. Require totals within5node-hours and20GPU-hours. The existing gate numerically checks only node-hours; its GPU statement depends on external accounting. In the replacement live checker, enforce both dimensions or explicitly validate a complete four-GPU-per-node invariant before deriving one from the other.
5. Keep a strict freshness bound for these actual observations, such as the existing300s, at the admission decision. Require successful complete responses and reject stale, future-dated, malformed, ambiguous or timed-out observations. A failed or overlong query means STOP, not polling/retrying until success. Retain live quota remeasurement immediately before checkpoint staging and its native freshness check.

Read-only checks are not atomic with future scheduler/account activity. Their soundness depends on the durable controller reservation and serialized diagnostic admissions above, plus scheduler exclusivity. If another controller can independently spend the same diagnostic envelope or create overlapping MVP jobs, that must be resolved at admission; changing a timestamp cannot resolve it. Do not add a broad distributed locking service where the existing single controller can simply retain the reservation.

The observed exclusion list is exactly `gh062,gh093`; release_v2 correctly preserves it. Do not restore historical gh086 merely because it appears in the old script. Reconcile current B5 state/placement at submission and at compute start under the approved policy; preserve baseline jobs, do not cancel them or guess new exclusions.

### Minimal changed surfaces and re-review criteria

- Change the deployment wrapper's `validate_controller_gate`/`controller_gate` contract and add the bounded read-only host check. Keep authorization validation and current observations distinct in stored evidence.
- Update `controller_execution_scope.json` and operator documentation. Retain the shell resource limits, environment cleanup and exclusive persistent marker; only rename its gate variable/path if needed for clarity.
- Add focused tests: old durable authority plus fresh compute observations succeeds after24hours queued with an unchanged external digest; changed release/authority fails; stale/future/failed/incomplete observations fail; missing/active old job, another active diagnostic, changed allocation geometry/timeout/restarts, each cap exceeded, and missing B5/account margin fail. Check self-accounting once, step-row deduplication, and no stage/hash/rename/probe before successful admission. Test no retry and marker preservation on failure.
- Build a new immutable release and transport, externally repin it, rerun the bounded adapted and unchanged original suites plus new gate tests, and independently review the new bytes. Leave original final_p1 and this release_v2 unchanged. Source archive need not change for this wrapper-only correction.

A strictly immediate-allocation-only policy could avoid a queue-age failure, but it would abandon the required usable queued workflow and still need current safety checks. It is not a reason to approve this package or repeatedly resubmit it. The two-part authorization/observation split above is the smallest practical queued solution.

## Requested safeguards checked

- **Builder/closure:** `build_release.py:21-46,49-94` validates pinned candidate ledger, manifest/archive, exact103 members, safe archive entry types/names, small-input/companion hashes and refuses an existing destination. Base9603e20 plus dirty-snapshot provenance is retained honestly. Candidate companions remain separate. The two producer helpers, profile module, panel compatibility and separately pinned panel verifier are present. Aggregates are comparison identities, not invented runtime import manifests.
- **Profile and role:** `deploy.py:184-199,317-356,445-468` sends development-single/123 to the actual prepare entry and rechecks canonical bindings plus mvp_finetune after prepare and before P1. Archived `mvp_profiles.py:7-13`, `make_launch_readiness.py:251-253,295-297,382-384` and `real_probes.py:137-145,255-257,684-686` match this interface. B5 seed123 initialization does not become B5-role MVP output. Production augmentation is20261037. S0's existing paired fraction1 selection constant20260915 is unchanged.
- **Original input/staging protections:** original-to-selected `deploy.py.diff.txt` confirms unchanged quota parser,40GiB reserve plus5GiB staging envelope,10000-file headroom,2200000000B copy ceiling and checkpoint hash/sidecar handling. The archived native checkpoint quota age remains3600s. Quota warnings/unknown formats fail closed. Large identities are still computed only in allocated host execution, not by the builder/reviewer.
- **Read-only source/input handling:** retained explicit RO code, SIF, corpus, source checkpoint and private HF mounts; source mount absent from P1; inputs writable only for S0 staging then frozen/RO. Existing staging evidence is renamed into a fresh job only, never recovered from an old partial tree. Exact cache inventories, pinned revision and source-before/source-after/destination comparisons remain. These are checked staging contracts, not claims of newly measured remote HF bytes.
- **Execution bounds:** selected sbatch retains exclusive one-node/four-GPU, no-requeue,1800s,1700s+20s outer guard and900s inner probe limit. Environment override clearing, cleanenv/offline settings, fresh compile directories, exclusive marker and no-overwrite namespace remain. No followup/training launcher was added or invoked.
- **Filesystem/transfer limitation:** DrvFS directory permissions do not establish immutability. The externally pinned transport plus exact ledger/membership is the identity. Remote deployment must preserve payload bytes/read-only access while providing only the already-required writable directory for the exclusive claim, not a copied claim marker. No remote filesystem property was verified here.

## Observations, limits and handoff

The supplied `../controller_remote_preflight.txt` reports old3163986 owned by UID97434, PENDING, elapsed0, AllocTRESnull at16:10:48PDT; old3163545 cancelled with elapsed0; evidence1068773376B, epoch05 weights2091010540B, SIF11120398336B; quota363.3G/500Gsoft/550Ghard and4568files. It is historical evidence, not this review's current measurement. It does not establish terminal cancellation, complete campaign spending, present margin or admission permission. No cancellation occurred in this review.

All intentional file writes are under this `review/` directory. No source edits, remote actions, deployment invocation, jobs, cancellation, large-input copying or readiness/training claims. Graft was consulted for orientation; it reported a rebuild already in flight and returned historical indexed followup material, which was not substituted for the known unindexed selected archive. No graph rebuild was requested. Advisor consultation failed because of client/model incompatibility, so no advisor approval is claimed.

Completed: independent scoped inspection, exact closure/digest checks,19+11tests, shell/ledger checks, original preservation verification and gate-boundary reproduction. Blocked: new queue-safe gate implementation/review and all controller execution gates. Per this review's explicit scope, no fix was implemented. The next owner is the controller/implementer, not an unattended execution worker.

Evidence: `identity-verification.json`, `adapted-tests.txt`, `original-tests.txt`, `shell-syntax.txt`, `release-ledger.txt`, `test-results.json`, `preservation-before.json`, `preservation-after.json`, four original-to-selected diff files, and `gate-boundary-results.json`. `collect_evidence.py` and `gate_boundary_evidence.py` are reproducible local evidence collectors only.
