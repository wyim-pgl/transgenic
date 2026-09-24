# A11 FIX1 independent scoped re-review — 2026-09-17

## Verdict: APPROVE within the bounded normalized-consumer scope

The single Important finding in `A11_REVIEW.md` is resolved. No new blocking finding in this fix. This verdict covers the accession/library consistency correction and its regressions, not whole-branch approval, production C0/import/migration, complete nine-species source certification, or broader A46 freezing.

Worktree: `/tmp/est-approved-contract-20260917`; HEAD remains `c9a351431b7dd8fc8ad45be35a9df29ca8445f81`.

## Actual change verified

`revision/scripts/evidence/est_molecule_closure.py:153,180–185` adds exactly seven lines: collect resolved canonical library IDs per `(species, assembly, arm, accession stem)` and reject more than one **before union construction**. The dataset is deliberately absent from that consistency key; aliases resolve before comparison. Unknown libraries are not invented or treated as another resolved preparation; their existing whole-input blocker remains active.

Because these implementation files are untracked at this base, ordinary `git diff` does not show their content changes. Instead, the reviewer removed only the seven additions **in memory** and verified that the remaining bytes hash exactly to the original reviewed consumer (`53d2f8fb…`). Removing only the added regression block likewise recovers the original test-file hash (`88ddb85f…`). Thus the rest of the consumer and all original tests are unchanged. No source file was modified by this comparison.

Read the updated tasknote, including its explicit acknowledgement of the original defect and reported four-red-to-green sequence. That historical red run is implementer-reported; this review independently verified present behavior and the exact source delta.

## Reproduction and controls

- The preserved original `a11_review_evidence/adversarial.py` now fails closed at its first conflicting-library call with `ValueError: accession stem has conflicting canonical libraries`; it no longer returns two ready version components.
- Independently exercised the eight new regression cases: clean/conflicting versions and old excluded-secondary placement, each within/across download datasets (four rejection cases); clean/excluded canonical-alias controls (two); distinct-accession/different-library independence; unknown-library version whole-input blocking.
- Valid aliases still yield one component with latest-only support; the old excluded placement still excludes that full component.
- Different accessions in genuinely different libraries stay independent. An unknown library still clears every ready/training-support view, rather than letting the known remainder through.
- The fix does not change library-scoped clone/signature union, arm separation, placement retention, audit thresholds, context exclusions, deterministic components, or retained rejoin. No new policy or fallback was introduced.

Evidence: `a11_review_evidence/fix1_verify.py` and `fix1_verify.json`. The script verifies the byte-level delta, catches the expected failure from the unchanged original reproducer, and executes the regression controls.

## Fresh focused verification

Executed with bytecode disabled, pytest cache provider disabled, and pytest temporary files directed into the review evidence directory:

```text
cd /tmp/est-approved-contract-20260917
PYTHONDONTWRITEBYTECODE=1 /tmp/mvp-final-source-env/bin/python -m pytest \
  -p no:cacheprovider \
  --basetemp=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/approved_execution_20260917/a11_review_evidence/fix1_pytest_tmp \
  revision/scripts/tests/test_est_molecule_closure.py \
  revision/scripts/tests/test_evidence_support.py \
  revision/scripts/tests/test_univec_trim.py \
  tests/test_est_approved_contract_integration.py -q
```

**101 passed in 2.82 s:** 47 closure cases (39 original + 8 regression cases), 50 focused existing scorer/UniVec cases, 4 integration cases. `git diff --check` passed. Output: `a11_review_evidence/fix1_pytest.txt`.

## Narrow integration-test claim

Read `tests/test_est_approved_contract_integration.py:1–63`. Its explicit “Small synthetic API connection, NOT real BAM/C0/route-A certification” claim is appropriate. Four combinations of HSP frame ±1 and FLAG 0/16 connect raw13→rich17 acceptance, codon projection over `5M7N115M`, a caller-constructed normalized placement and A11 component identity, then route-B observations and conflict resolution. Assertions compare the complete expected genomic phase-count map, require `A_missing`, and preserve non-training eligibility.

It is a useful positive API smoke test, not an independent placement adapter, conflict stress suite, real-input proof, or full-branch certification. It does not purport to test actual route-A agreement. No expansion of its claims is needed for this fix approval.

## Final reviewed SHA256

| File (worktree-relative) | SHA256 |
|---|---|
| `revision/scripts/evidence/est_molecule_closure.py` | `fd05c3ca0ef674b10962efac88861131aa3f54fbb1fa5f7e196c335740dc67b5` |
| `revision/scripts/tests/test_est_molecule_closure.py` | `cc750e8fc6b54714235492fce6d34f0adb845fdba5e141f7e67533897f78febd` |
| `tasknotes/a11_molecule_closure_20260917.md` | `13b99d7435430cbc67c54e40f5de2990dc0c81f05edd03e2cf3340c28616961e` |
| `tests/test_est_approved_contract_integration.py` | `3eb1f36ffaa3928ba4d30fd82553696e2b06356f9bae09d9f93cc9cdc8ef8c6f` |

No code, Git index, HEAD, commits, remote resources or subagents were mutated/launched by this review. The required first Graft query automatically refreshed its derived graph for two changed files; no explicit build was requested. Intentional writes are this report and review evidence. Graft reported approximately 71,236 tokens saved this turn.

Caller completeness flags/provenance/hash syntax remain declarations rather than source attestation, as already disclosed. Those unchanged production limitations are not reopened as blockers to this narrow library approval.
