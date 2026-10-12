# Independent scoped producer-length implementation review

## Decision

No blocking defect found in the controller-approved, two-record historical producer binding or retained-only replay. The tested bytes are suitable for the narrowly authorized diagnostic code change, with one nonblocking test regression below. This is not scientific approval, deployment authorization, training eligibility, full C0, A11 certification, or permission to replace historical results.

Reviewed `/tmp/est-real-source-binding-20260917`: IMPLEMENTATION.md, diffs of both affected implementation files and both affected tests against `tasknotes/producer_length_fix/backup/`, the affected source/test bodies, and new `tasknotes/real_sample/replay_retained.py`. Consulted graft first and used its exhaustive identifier search to check adapter/classifier consumers. Read the controller's LENGTH_SEMANTICS_REVIEW.md. This review relies on its static upstream attribution and the controller-approved proof pin, not on a new source/build reproduction. Advisor consultation failed due to client/model incompatibility and contributed no evidence.

## Finding: nonblocking regression in compute-guard test

`tests/test_est_real_sample.py:224-228` still invokes `main` without the newly required `--producer-path-proof`. It now passes because argparse rejects missing arguments, before the SLURM guard executes. Removing the compute-only guard would no longer fail this test. Add the required proof argument and assert the SLURM diagnostic in a subsequent test-only correction.

The runtime guard itself remains intact. An independent invocation with all required arguments and SLURM_JOB_ID unset exited 2 with `real-source CLI requires SLURM_JOB_ID; no login-node execution`. Evidence: `producer_length_fix_review_evidence/cli_guard.txt`. No implementation or test files were edited during this review.

## Attempts to refute the change

- **Formula selection:** `est_hsp_contract.py:32-47,71-80,87-90,130` permits only explicit MID or MIDF, keeps MID as the keyword-only default, and preserves literal reported-column `int(row[8]) >= 30`. The adapter forwards the same contract to its internal classifier. No acceptance of either formula based on row fit, shift sign, or database label was introduced. Existing exact query/protein truth, endpoint, X rejection and geometry remain enforced.
- **Authority before eligibility:** `est_real_sample.py:29-90,379-396` checks exact bounded proof bytes against the compiled SHA256, then resolves the complete current search preimage. Both ordered searches resolve before QNAME selection, raw-row parsing and SAM subprocess creation. The proof is not taken from a row or manifest pin. Missing proof fails even for F=0 and with the test SAM hook. Hashing current files before resolution is integrity checking, not row selection.
- **Closed registry:** The actual 235630-byte proof rehashes to the approved pin. Its two resolved routes are blocked transcript join/MIDF and direct legacy/MID. Unknown routes, duplicate preimages and inconsistent embedded text links are checked. The strongest schema protection is the exact immutable byte pin: this is deliberately not a general validator for arbitrary future producer registries. Package/archive/log semantics are trusted through that reviewed pin, not newly inferred by runtime string searches. No caller pin override exists in the API or CLI; synthetic test monkeypatches are test authority only.
- **Identity propagation:** New search preimages include producer-record hash, registry hash, route and formula; old search identity is separately labeled `legacy_search`. HSP identity binds the new search ID; projected diagnostics retain that HSP ID and classifier source hash. The binder forwards the resolved contract to both calls at `est_real_sample.py:535-537`. Retained replay does likewise at `replay_retained.py:155-157`. Independent checks verified the corresponding search/HSP IDs and formula propagation for every replayed HSP and linked projected result. Members and placements retain their original identities because their underlying evidence did not change.
- **Other consumers:** Graft's exhaustive identifier search found the legacy `tasknotes/hsp/replay_actual.py` mathematical diagnostic and synthetic integration tests still using the MID default. They were not silently upgraded into producer-authorized MIDF consumers. Backup files are not live consumers. Geometry still consumes validated codon events and proof-bound HSP identity rather than independently choosing a length formula.
- **Retained replay:** `replay_retained.py:57-199` requires fixed parent hashes and byte inputs before JSON parsing, verifies parent IDs, raw bytes, sequence/header/source identities, member relationships, dictionary and row linkage, and resolves both searches before HSP processing. Original historical classifications do not choose new outcomes. The API performs no input file reads or subprocess calls; source hashing happens at import. Existing focused tests forbid opens, binder scans, FASTA access and subprocess creation during byte-only replay. CLI inputs are fixed retained paths and output directory creation is exclusive.
- **Limits:** Proof cap is 256 KiB, retained report 1 MiB, retained diagnostics 16 MiB, with sentinel-byte reads in CLI. The streaming binder's previous resource accounting and timeout behavior remain unchanged. Retained replay is intentionally restricted to exact pinned finite artifacts, not offered as a general untrusted replay service. No broad resource-safety claim follows from these pins.

## Fresh verification

Executed with `/home/wyim/miniconda3/bin/python`, Python 3.8.20, PYTHONDONTWRITEBYTECODE=1, pytest cache disabled, external timeout 90 seconds:

```
python -m pytest -p no:cacheprovider -q tests/test_est_hsp_contract.py tests/test_est_real_sample.py tests/test_est_phase_conflicts.py tests/test_est_approved_contract_integration.py
```

Result: **200 passed in 40.15 seconds**, exit 0. This was a focused run, not the whole suite.

Additional independently written bounded check: **50 expected rejections** using the real, unmodified compiled proof authority. Cases covered missing, empty, altered, truncated, non-bytes and oversize proof; changed or omitted values for every full-preimage field; and an extra self-asserted proof field. Both real producer records were exercised. No compiled pin was monkeypatched in these additional checks.

Independently replayed the exact retained parents, recomputed all new search/HSP identities and inspected all propagated contracts. The new replay is byte-identical to the implementation's recorded replay hash:

| Quantity | Verified result |
| --- | ---: |
| Retained HSPs | 462 |
| HSP_RULE_ACCEPTED | 414 |
| Query-codon truth failures | 30 |
| Subject-residue truth failures | 18 |
| Geometry success | 380 |
| Geometry unsuccessful | 34 |
| Newly rule-accepted versus historical diagnostics | 29 |

All 29 newly rule-accepted HSPs have unsuccessful geometry, with `self_overlap_policy_unresolved` recorded for all 29. They are not residual length-forwarding failures. This observation does not resolve self-overlap or motif/strand policy, and is not biological rejection or masking approval. All scientific gates remain false; A11 is unassessable.

## Tested identities

| Artifact | SHA256 |
| --- | --- |
| `revision/scripts/evidence/est_hsp_contract.py` | `83343fc7c687ecc417ce6cbeb4bdbff8f0e8e532f9e2423146dadfc08bf4513f` |
| `revision/scripts/evidence/est_real_sample.py` | `bd51f14592ff1131d4eae4c2948c514efd42a7f82fda0f2e423ecb57a87bc7a8` |
| `tasknotes/real_sample/replay_retained.py` | `b239c6affec26edb1f9ceb4a6edd777ef235011e1ceb4ee7f275a9856374c508` |
| `tests/test_est_hsp_contract.py` | `579672e2423e48b54e91cdee34445a2023dc6402283d049c9d4c91097e895dfd` |
| `tests/test_est_real_sample.py` | `f68869fc79e5852992e94989af35eba957449f4c3aa637c4bb3b8d11a64e5515` |
| Controller producer proof | `a2843aa36a006730c1fd3fe02a34b76afd772f1e01328162f7c0f9af88a2b6a9` |
| Parent report | `2e97691107440fd6d9e09341d3feb2588823cc28f9c9da5bf4f064a6d95728d8` |
| Parent diagnostics | `8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d` |
| Independent new replay | `30840e05d6f800aad7500e2b5d667128575094a33fe265830405d60aa65ba2f5` |

Full test/source hashes, resolved producer-record hashes, interpreter identity, counts and unchanged parent hashes are in `producer_length_fix_review_evidence/verification.json`. Runnable independent checks are in `check_review.py`; focused test output is in `focused_tests.txt`; new replay is in `replay.json` in that same evidence directory.

## Scope and remaining work

Completed: scoped diff/source review, focused tests, real-authority adversarial checks, retained replay and identity verification. Nonblocking follow-up: restore effective compute-guard test coverage. No implementation edit, original-data access, remote operation, aligner execution, deployment, historical artifact rewrite or whole-suite test was performed. Parent report, diagnostics and proof were rehashed before and after the independent replay and are unchanged.

This review did not inspect the remote/deployed package and therefore cannot independently attest its current bytes or prove historical deployment immutability. No deployment was changed by this review, and no deployment change is authorized by this decision. Scientific acceptance, masking interpretation, conflict policies, accounting/OOM uncertainty and future producer support remain separately gated. Do not promote old results from these new per-HSP diagnostic counts.
