# Independent evaluation reference-map review — 2026-09-16

Reviewed `66f130d7abee262b2c3c840fc129ad8fd03ec29a` against `b660bf1`, in an isolated detached worktree. Report-only review; no source edits, SSH, jobs, subagents, merge or push.

## Verdict

**Mapping correction is sound; request one Important publication-integrity fix before deployment. No Critical defect found.** Actual accepted-corpus 50-locus compatibility is still unverified. Synthetic tests and the real first-gene prefix do not certify that population.

## Important: mapping validation ends before publication

Locations: `revision/scripts/mvp/mvp_produce_counts.py:519–533`; `tasknotes/reference_mapping/run_panel_cpu.py:119–123,153–163`; helper `mvp_produce_counts.py:326–329`.

The producer calls `recheck_reference_mapping` before writing counts/statistics/receipt, then returns without checking again. The CPU audit similarly checks before writing `reference_compatibility.json`; the wrapper's later source/authority checks do not recheck the mapping corpus/WAL. Reference-only mode returns immediately after its marker. A corpus mutation or new WAL during publication therefore leaves an apparently successful receipt/report bound to a snapshot that no longer satisfies the frozen-source contract. The helper itself checks WAL only before potentially lengthy source hashing, not after it.

**Concrete local reproduction succeeded:** ran the existing full350 synthetic integration function with an in-memory `json.dump` hook that created `corpus.json.wal` immediately after dumping `counts.json`. Production `produce` returned normally; `receipt.json` existed with all 350 comparisons and the integration assertions passed. Explicitly calling `recheck_reference_mapping` on that published receipt then raised `ValueError: corpus WAL appeared`. No production input or source was modified. The pre-existing producer publication loop already lacked a final check; the new mapping contract remains incomplete at that boundary rather than introducing a different scoring algorithm bug.

Minimal corrective scope: recheck mapping sources/WAL after output serialization and before reporting successful completion; bracket source hashing with WAL checks. Apply to both producer and CPU publication paths, including reference-only and final CPU completion. Ensure detected mutation leaves an unambiguously failed publication, not a consumable success receipt/marker. Add late-mutation tests. This is enforcement of the requested source/publication identity contract, not a new scientific policy.

Reproduction core (inside `TemporaryDirectory` and `pytest.MonkeyPatch.context`, importing `test_mvp_score_producer_integration as t` through `PYTHONPATH=src:tests`):

```python
original = t.p.json.dump

def injected(obj, f, *args, **kwargs):
    result = original(obj, f, *args, **kwargs)
    if Path(f.name).name == 'counts.json':
        (root / 'corpus.json.wal').write_bytes(b'synthetic concurrent WAL')
    return result

mp.setattr(t.p.json, 'dump', injected)
t.test_full_producer_actual_finalizer_frozen_panel_synthetic_biology(root, mp)
receipt = json.loads((root / 'result/receipt.json').read_text())
# Returns above; this explicit check then rejects:
t.p.recheck_reference_mapping(receipt['reference_mapping'])
```

## Verified correction and unchanged boundaries

- `resolve_reference_mapping:284–323` requires exactly 50 unique panel keys, parameterizes Athaliana-only queries, and checks complete unique source keys and one-to-one nonempty original IDs. Original Name is diagnostic only. No Name, prefix, counter or coordinate fallback is used.
- Accepted corpus/panel hashes precede the read-only one-thread query. Exact singleton build-manifest FASTA/GFF full SHA256 values must match frozen reference identities. `frozen_reference:54–70` still obtains authority from the immutable Git object, not editable working-tree manifest bytes.
- Map coordinates must equal panel geometry. `prepare_mapped_reference:332–346` restricts by stored original IDs, checks raw gene count and exact parsed inventory, then uses `select_transcripts:349–360` to check raw chromosome/start/end/strand and nonempty transcripts. Empty references fail, while legal empty predictions remain supported.
- Producer and CPU joint/per-locus fallback use the same preparation helper. CPU joint errors remain visible even when individual diagnostics succeed. Neither duplicate structures nor strict parser failures are silently repaired.
- Receipt includes both identifiers, species/geometry/strand, diagnostic Name, source identities and authority. Normal prepublication source mutation is detected; the late-publication gap above is the exception.
- Prediction selection, scoring denominator logic, 350 seed/fraction/key transport, panel ordering/core20/windows, training source, model/config/finalizer and numerical probe bytes are unchanged by Git comparison. Production-source diff is confined to the evaluation producer; the copied CPU template and tests/documentation are separate.

## Independent local verification

Environment: existing `/tmp/mvp-final-source-env/bin/python`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src`.

1. `pytest -q -rs tests/test_mvp_reference_mapping.py tests/test_mvp_score_producer.py`: **39 passed, 263.55s**. Includes positive original-ID/identity-map cases, bad/missing/duplicate/wrong-species/null/many-to-one mapping, hash/geometry/strand/empty-reference negatives, and CPU shared-helper/fallback coverage.
2. `pytest -q -rs 'tests/test_mvp_decoder_integration.py::test_panel_rejects_mutations[stale_producer]'`: **1 passed, 27.49s**. Old panel source identity is not accepted under the changed producer.
3. Full350 integration function independently executed with the late-WAL hook above: its existing assertions passed, exposing the missing rejection. This is a standalone adversarial execution, not an additional normal pytest-suite PASS claim.
4. Compared actual retained 131072-byte prefix with the fixture: fixture is its exact leading bytes. Prefix SHA256 `ee62582867993bb5a31a5f665027fbd93f1331e51729ebe4f0faa5ac967cf19c`; fixture SHA256 `192ae64c5fb3a81067400b4ea8d6600e314e8e4da1747ec7be4b50d4c29b9350`. Real first gene supports `Ath000001 → ATMG00010.TAIR10`; the other mapping rows are synthetic, not actual nuclear panel mappings.
5. Inspected both changed integration fixtures: explicit synthetic map/build-manifest records are consistent with their synthetic raw-reference bytes. Candidate/authority substitutions are scoped fixtures, not evidence of real accepted-data compatibility.
6. `git diff --exit-code b660bf1 -- src train configs revision/protocols revision/scripts/mvp/real_probes.py revision/scripts/mvp/run_bounded.py revision/scripts/mvp/make_launch_readiness.py revision/scripts/mvp/mvp_decode_runtime.py` returned no differences. `git diff --check` passed before report creation.

The implementation report's **98 passed / 1 skipped / 439.67s** remains attributed evidence, not this review's full-suite result. The controller's prior 120-second timeout is not a test failure or completion claim; this review ran the focused checks separately to completion. No full-suite rerun, GPU/SIF validation or actual corpus query was performed.

## Evaluation deployment and operator pins

Independently recomputed values agree with the implementation report:

| Identity | SHA256 |
|---|---|
| Producer | `ace42b9233de3b49c40e2b7b930e65c4567ef1cd49ac193bc9472030f0bf806b` |
| Decoder operator closure | `c0e8c141b1963abe1bd5b1b167bada5e3089057da7c4855dd61aa4fb4610b0d7` |
| Evaluation component closure, 86 files | `44078a800aab35c991d6792899659627f76a1867520a40724887c9003bd9133f` |
| CPU template | `ac727d1caab301eb2c35782512710713ad9ad8c90715653600eb9c4d824bc202` |
| Unchanged panel | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` |

Recomputed b660 component closure reproduces `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`. Direct `project_closure` validation rejects the old decoder operator pin `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb` and accepts the new pin above using a synthetic probe-identity input. This tests the operator-pin boundary only, not real runtime preflight.

**The CPU wrapper intentionally retains old hardcoded source/inventory/component guards. It is a template, not a deployable release; this is an explicit deployment gate, not a hidden defect.** Fresh packaging must repin the evaluation commit, full source inventory and final staged entrypoint; the template digest is not that future entrypoint digest. Any publication-integrity correction will change the producer and closure pins again.

A new immutable panel namespace is required: `mvp_prepare_panel_inputs.preflight:30–45` includes producer identity and `mvp_verify_panel_inputs.load_panel_inputs:27–32` compares exact input/code inventory. Do not rewrite old job6165559 manifest/COMPLETE or waive that gate. Preserve the separately bound b660 training source and 772 probe bytes. Actual 50-map/strict-parser/geometry observations and actual decoder package/SIF compatibility remain distinct gates; duplicate-policy decisions remain outside this fix.

Graft was queried first in this worktree; reported context savings total approximately 1,067,498 tokens. Advisor consultation failed with client/model-version incompatibility and supplied no review advice.
