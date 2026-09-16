# C1 historical receipt/reconciliation implementation report

Implementation commit: **06835580c18905fe28f38eb43f28596b9bde06b4**, based on **5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35**, worktree `/tmp/b6-p2-phase-1535433`. This report is a subsequent documentation-only commit.

## Scope and current disposition

Implemented only `revision/scripts/evidence/est_blastx_audit.py`, new CPU tests plus the unmodified recovered pilot receipt test fixture, and scoped plan/schema documentation. **No operational reconciliation manifest has been issued, no real-tree reconciled PASS claimed, and no historical artifact waived automatically.** Baseline without valid manifest still FAILs while preserving all18 unit reports. Extern remains explicitly UNRESOLVED; `clean_accounting`, `accepted_phase_evidence`, and `production_authorized` remain false even on a synthetic reconciled PASS.

No remote access, submission, cancellation, snapshot modification, raw-data read/write, or deployment was performed in this task. Controller reports immutable baseline job6165461 submitted15:47PDT; its code.tar, receipts, jobs and output namespace were not touched. MVP8-channel GPU work remains independent. Existing `.gitignore`, `.ignore`, `.testdeps/` state was not edited/staged; no dependency installation, graft commit, merge, push, or subagent was used. Advisor attempt failed due to client-version incompatibility; no advisor review is claimed. Independent controller review remains pending.

## Changes

- `--pilot-submission-receipt`: only the local recovered extract with SHA256 `f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23` can establish pilot log layout. This whole-file anchor binds historical session source, raw record hashes, records540/541, call/result IDs and timestamps, explicit submission paths and job6163851. The code records source/lines/job in the check, requires exact historical root, exact `submission.lock/jobid` bytes and approved `run.sbatch` pin, and both `slurm-6163851.out` and `.err`. Empty stderr remains valid. This authenticates against the reviewed recovered extract, not an unavailable immutable scheduler archive or a fresh reread of the original session.
- `--reconciliation-manifest` plus `--reconciliation-sha256`: consumes version `est-c1-reconciliation-v1`. Manifest and hash-bound accounting/queue captures must be canonical absolute regular paths outside audited input and auditor source trees. No approval boolean, wildcard, auto-generation or arbitrary path extension.
- Closed allowlist of19 exact historical paths. Each entry binds job(s), producing wrapper path/pin, recursively exact kind/UID/GID/listing/file hashes. Preflight requires all7 container/sentinel entries together, exact known root/receipt/log/source-unit listings, both lineage pins/provenance matches/source DONEs, preflight submission receipt/jobid/batch exit. The preflight `execution.lock` must be an empty regular file; pilot execution lock remains an empty directory.
- Fresh queue capture at validation (timezone-aware, <=1hour, command/stdout/returncode bound) rejects relevant queued/active jobs; accounting requires exact completed allocation/batch and recorded anomalous extern, rejects active/unknown extra states and duplicate/missing bound rows. Reconciliation captures are reviewed evidence, not cryptographically authenticated scheduler responses or live locks.
- Checks exact snapshots again at disposition use and before final rollup after potentially lengthy unit hashing. Changed/new files, links, markers, ownership, kinds or child bytes fail. Missing or malformed reconciliation yields an ordinary tree-check FAIL, not an early loss of18unit inventory. Original NEITHER remains recorded with `reconciled: true`; no DONE is inferred or created in historical input.

Full schema and trust boundaries: `notes/EST_C1_RECONCILIATION_PLAN.md`. The existing sbatch template was intentionally not changed; deployment/invocation changes require controller review and a new immutable snapshot, never patching baseline.

## Scientific-neutral facts versus policy decisions

**Scientific-neutral receipt reconstruction:** exact submitted pilot output/error names; job/task-to-species mapping; wrapper identities; actual marker presence/absence; file-vs-directory distinction for locks; empty stderr; observed empty scratch; full byte/owner/listing inventories; accounting literals. These changes do not alter raw13 contracts, filtering, HSP acceptance, phase interpretation, ancestry claims or biological conclusions. Array task order is the recorded alphabetical order: Athaliana0, Bdistachyon1, Gmax2, Osativa3, Ppatens4, Ptrichocarpa5, Sbicolor6, Vvinifera7 (checked against all8 historical stdout captures).

**Explicit policy/review required:** deciding that exact markerless historical sentinels/containers may receive reconciled dispositions; approving the manifest and its separately recorded digest; trusting operator captures and choosing the one-hour freshness window; judging operational quiescence without a live flock probe. Hash equality plus completion evidence implements that narrowly reviewed decision; it does not itself prove absence of a current writer. Only the explicitly listed paths are eligible. No real manifest was created or approved here.

**Not resolved or waived:** extern OUT_OF_MEMORY0:125 root cause/acceptability, clean scheduler accounting, authoritative ancestry/merged-taxid limitations, A46/C2/C3 semantics, phase evidence acceptance, production authorization, or permission to mutate/delete historical locks/scratch. These cannot be derived from directory reconciliation.

## Remaining unbound paths/evidence

All19 paths below remain **operationally unbound until a separately reviewed external manifest and captures are supplied**, relative to `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx`:

1. `sitalica_pilot_20260914/tmp`
2. `sitalica_pilot_20260914/execution.lock`
3. `sitalica_pilot_20260914/submission.lock`
4. `remaining8_20260915/submission.lock`
5. `remaining8_20260915/Athaliana/tmp`
6. `remaining8_20260915/Bdistachyon/tmp`
7. `remaining8_20260915/Gmax/tmp`
8. `remaining8_20260915/Osativa/tmp`
9. `remaining8_20260915/Ppatens/tmp`
10. `remaining8_20260915/Ptrichocarpa/tmp`
11. `remaining8_20260915/Sbicolor/tmp`
12. `remaining8_20260915/Vvinifera/tmp`
13. `preflight_20260914`
14. `preflight_20260914/logs`
15. `preflight_20260914/receipts`
16. `preflight_20260914/submission.lock`
17. `preflight_20260914/tmp`
18. `preflight_20260914/xdg-cache`
19. `preflight_20260914/execution.lock`

Available main captures supplied wrapper/lineage/provenance/log and jobid hashes, but not every small-file child hash or numeric owner ID needed for recursive snapshots. In particular the five preflight receipt files (`DB.sha256`, `PROVENANCE.txt`, `SOURCE_COMMIT.txt`, `build_dbs.sh`, `swissprot_viridiplantae_reviewed.fa.gz.sha256`) have stat listings, not complete fresh hashes in the inspected captures. Several other preflight root files similarly need exact small-file bindings. No values were invented. A fresh queue capture and preflight6163397 accounting in the manifest-bound accounting file are also necessary; primary C1 sacct contains the nine search jobs only. Obtaining these small facts and reviewing the real manifest is follow-up work, not authorization for any large scan or job rerun.

## Verification

Existing environment only: `/home/wyim/miniconda3/envs/agent-reach/bin/python`3.11.15 and existing `.testdeps`. Initial default Python3.8 and system Python3.10 attempts failed in pytest import (incompatible type syntax / missing exceptiongroup); no dependencies were installed. Switched to existing Python3.11.

From the worktree, exact final commands:

```sh
PYTHONPATH=.testdeps /home/wyim/miniconda3/envs/agent-reach/bin/python -m pytest revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py revision/scripts/tests/test_track_c_recovery.py revision/scripts/tests/test_track_c_audit_review.py -q
# 119 passed in 5.85s
PYTHONPATH=.testdeps /home/wyim/miniconda3/envs/agent-reach/bin/python -m pytest revision/scripts/tests/test_c1_reconciliation.py -q
# 56 passed in 5.22s
git diff --check
# exit0
```

TDD: initial32 new cases failed on missing APIs; later negative tests reproduced missing preflight-lock coverage, self-consistent broadened receipt listings, late mutation acceptance, active extra accounting steps and noncanonical capture paths before fixes.56new tests cover pilot positive/negative bindings, manifest mutations, preflight full small tree, actual array task bindings and18unit integrated PASS/FAIL with semantic gates false.119original tests remain unchanged. Tests use synthetic anchors only through pytest monkeypatch; production exposes no anchor override.

Graft orientation/query/caller output estimated **885,659 tokens saved** (tool estimate). No claim of independent code review or production C1 success is made.
