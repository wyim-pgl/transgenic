# Producer-bound length recovery implementation

## Scope and status

Implemented in the existing isolated `/tmp/est-real-source-binding-20260917` worktree. No reset, commit, remote operation, job, new dependency, original-input scan, DIAMOND execution, or deployed-package edit was performed. The pre-existing dirty binder/tests and unrelated worktree changes were preserved. Before source/test edits, the affected files and existing `tasknotes/real_sample` directory were copied to the fresh `backup/` directory here.

Completed: mathematical contract, pinned producer resolver, binder integration, retained-only replay, focused tests, local replay and graph refresh. Independent controller review is pending. This is not production or scientific acceptance. No historical result was overwritten or reclassified.

Read the controller's LENGTH_SEMANTICS_REVIEW.md and UPSTREAM_SEMANTICS.md before implementing. Graft context covered the adapter, classifier, interpreter and binder. The strategic advisor request failed because its client version does not support the configured model; no advisor result was used as evidence.

## Changes

- `revision/scripts/evidence/est_hsp_contract.py`: keyword-only `length_contract` in both APIs. Default remains strict MID. Exactly one MIDF alternative uses the existing interpreter's `columns_including_F`; unknown modes fail. The literal reported-column `int(row[8]) >= 30` gate, sequence truth, endpoints, identity and geometry semantics were preserved. Mathematical mode selection is explicitly not producer authority.
- `revision/scripts/evidence/est_real_sample.py`: `resolve_producer_path` accepts bounded proof bytes and a full exact search preimage. Its compiled SHA256 pin is outside all incoming records/manifests, with no caller override. It checks registry/record shape, unique preimages, embedded text hashes/sizes, recipe/archive agreement, provenance/command digest links and allowed reviewed routes. The exact registry pin protects the complete embedded historical evidence; the resolver does not infer the route from source names, row fit or dynamically interpreted logs. This is static attribution, not a reproduced binary build.
- Both searches resolve after current file/receipt checks, before selection or row scanning, including F=0 and synthetic SAM runs. New search identities and report entries bind the producer-record digest, proof digest and resolved contract. Old search identities remain separately labeled `legacy_search`. Adapter and classifier calls both receive the resolved contract. CLI now requires `--producer-path-proof` and reads at most 256 KiB plus one sentinel byte.
- `tasknotes/real_sample/replay_retained.py`: fixed local job6165875 report/diagnostics paths and compiled parent pins. Its replay function accepts bytes only. It validates parent IDs, raw bytes, sequence/header/source identities, member relationships, dictionary and row linkage, resolves both searches with the same resolver, and uses the existing adapter/classifier/projector. It does not call binder `run`, scan original files, or spawn processes. Implementation files alone are hashed at module import for new-source identities. CLI writes a new exclusive output directory. Original classifications are not consumed to determine the new diagnostic result.

## RED/GREEN evidence

Interpreter: `/home/wyim/miniconda3/bin/python`, Python 3.8.20. No interpreter or package installation was needed.

- `red_math.txt`: 21 expected failures for absent length-contract keyword support.
- `red_proof.txt`: 17 expected failures, including old binder completion without proof and absent resolver.
- `red_retained.txt`: 4 expected failures for absent retained replay, plus 2 existing retained-budget tests passing.
- First combined GREEN attempt exposed six test-fixture compatibility failures. Four were invalid-manifest cases whose synthetic proof builder failed before reaching the binder. Two expected an unchanged second-search ID despite a newly changed whole-registry identity. The fixture now lets invalid manifests reach their existing checks, and the identity test checks the unchanged legacy identity while requiring new registry-bound identities. No production gate was relaxed.
- `green_focused.txt`: 159 passed, 25.34 seconds.
- `green_final.txt`: 200 passed, 27.01 seconds. Command:

```sh
python -m pytest -q tests/test_est_hsp_contract.py tests/test_est_real_sample.py tests/test_est_phase_conflicts.py tests/test_est_approved_contract_integration.py
```

Coverage includes both signs/strands, cancelling shifts, gaps, substitutions, 29/30/31 reported boundaries, default/wrong formulas, unknown modes, X rejection, endpoint rejection, MID29+F1 reported30 requiring a proven MIDF route in the binder, missing proof at F=0, altered/truncated/swapped/self-repinned records, ambiguous/unknown routes, changed query/protein/raw/provenance/command/source/scope/reference bindings, and changed parent IDs. Existing conflict and geometry tests remain passing. Tests alone monkeypatch the compiled registry pin for synthetic authorities. The retained replay test forbids file opening, binder scans, `run`, FASTA access and subprocess creation during byte-only replay.

`git diff --check` passed. Modified/new source and tests were checked for the prohibited punctuation character. `graft build` completed locally; output is in `graft_build.txt`.

## Actual retained replay

Command:

```sh
python tasknotes/real_sample/replay_retained.py --out tasknotes/producer_length_fix/replay_job6165875
```

New artifact: `replay_job6165875/replay.json`. Actual observations from this new artifact only:

| Diagnostic quantity | Count |
| --- | ---: |
| Retained HSPs replayed | 462 |
| HSP_RULE_ACCEPTED | 414 |
| Exact query-codon replay failures | 30 |
| Exact subject-residue replay failures | 18 |
| Successful geometry projections | 380 |
| Unsuccessful geometry projections | 34 |

These are not molecule counts, biological negatives, training eligibility or promotion of any historical result. Every scientific gate remains false and A11 remains unassessable. Historical accounting/OOM uncertainty and masking evidence limitations are unchanged.

## Identities

- Parent report SHA256: `2e97691107440fd6d9e09341d3feb2588823cc28f9c9da5bf4f064a6d95728d8`
- Parent diagnostics SHA256: `8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d`
- Proof SHA256: `a2843aa36a006730c1fd3fe02a34b76afd772f1e01328162f7c0f9af88a2b6a9`, 235630 bytes
- New replay SHA256: `30840e05d6f800aad7500e2b5d667128575094a33fe265830405d60aa65ba2f5`
- Classifier source SHA256: `83343fc7c687ecc417ce6cbeb4bdbff8f0e8e532f9e2423146dadfc08bf4513f`
- Binder source SHA256: `bd51f14592ff1131d4eae4c2948c514efd42a7f82fda0f2e423ecb57a87bc7a8`
- Replay script SHA256: `b239c6affec26edb1f9ceb4a6edd777ef235011e1ceb4ee7f275a9856374c508`

`hashes.json` also records test hashes and interpreter identity. All embedded new-source hashes were rechecked against current worktree bytes after replay. Existing `est_release_final`, historical diagnosis files and retained parent artifacts were not written. No remote package verification or redeployment was attempted.

## Limitations and remaining review

The pin intentionally authorizes only the two reviewed historical search preimages. A new query, protein, raw output, provenance, command or reference cannot reuse this proof. Future producer support requires separately reviewed immutable evidence and a code review, not a caller-selected pin. No masking wildcard, original-file reread, whole-repository test, fresh aligner execution, full C0, A11, production, training or biological certification is supplied by this change. Independent review remains the next controller-owned step.
