# MVP panel decoder implementation — 2026-09-16

Branch `mvp-panel-decoder-20260916`, based on `8803afb`. Only new decoder, test, and report files. No training/model/producer edits, jobs, SSH, merges, or pushes.

## Status

**CPU-tested implementation; production execution remains blocked on integration.**

The controller reported incomplete verification in the base Track-B producer. The runner explicitly rejects that exact old producer SHA256 (`a3b731c217e2da92eb044e1c7815a310fbd78b18a707487b12eb20a897576f1a`). Integrate the producer owner's independently reviewed fixes before execution. A different hash is not itself approval. This branch does not reproduce or modify those fixes.

The separate panel-input worker's final contract was unavailable. The implemented loader is explicitly `mvp_panel_batch_internal_v1`, a byte/shape/identity normalization interface, not an approved scientific input receipt. Controller adaptation to the final contract is still required. No ONT/HiFi inputs or training retained-population changes are introduced.

## APIs and consumption interface

- `src/transgenic/evidence/mvp_decode.py`: `load_panel_batch(payload, panel, sequences, pins)`, `encode_window`, `generate_one`, `validate_decode`, `gsf_gff`, `validate_gff`, `load_final_model`, `load_baseline_model`.
- `revision/scripts/mvp/mvp_decode_panel.py`: `decode_panel(payload, new_output_directory, command)` and CLI:

  ```sh
  python revision/scripts/mvp/mvp_decode_panel.py --input decode-input.json --out NEW_DIRECTORY
  ```

`decode-input.json` contains the existing Track-B transport fields `panels`, `fasta`, `reference`, `candidate`, `runs`, plus:

```json
{
  "panel_inputs": "/absolute/internal-batch.json",
  "panel_input_pins": {
    "evidence_db": {"path": "/absolute/c0lite.db", "sha256": "...", "size": 1},
    "panels": {"path": "/absolute/panels.tsv", "sha256": "...", "size": 1},
    "fasta": {"path": "/absolute/reference.fa", "sha256": "...", "size": 1},
    "rasterizer_code": {"path": "/absolute/mvp_tracks.py", "sha256": "...", "size": 1}
  },
  "encoder_tokenizer": {
    "path": "/absolute/pinned-local-tokenizer-snapshot",
    "files": {"tokenizer_config.json": "...", "ALL_OTHER_SNAPSHOT_FILES": "..."}
  },
  "device": "cuda:0"
}
```

Sizes above are schematic. All snapshot files, including remote-code Python files, must be enumerated with actual hashes. Loader is offline/local-only; the controller must supply the actual pinned Hyena nucleotide tokenizer, not a substitute. Tokenizer validation requires exactly 30,720 base tokens plus trailing SEP, strips only that SEP, and constructs the dataset-equivalent pad mask. Sequences come from the anchored reference FASTA through `mvp_windows`; gene strand never triggers RC.

Internal batch shape:

```json
{
  "schema": "mvp_panel_batch_internal_v1",
  "provenance": {
    "evidence_db": "same identity object as independently supplied pin",
    "panels": "same identity object",
    "fasta": "same identity object",
    "rasterizer_code": "same identity object",
    "population": "excluded_heldout",
    "keep_fraction": 1,
    "is_rc": false
  },
  "rows": [{
    "gene_id": "EXACT_PANEL_ID",
    "window": ["Chr1", 10000, 40720],
    "sequence_sha256": "sha256 of exact genomic-forward ASCII sequence",
    "tracks": {"path": "/absolute/locus.npy", "sha256": "...", "size": 1},
    "dtype": "float16",
    "shape": [30720, 8]
  }]
}
```

Exactly 50 distinct panel rows. Finite nonnegative capped tracks, N-base zeros, and availability-one on unmasked bases are checked. These checks cannot prove excluded-heldout provenance from an array alone; that responsibility remains with the independently reviewed worker. Caller hashes are bindings, not approval.

## Model and output behavior

- Final runs use existing five-payload manifest, `TRAINING_DONE`, and metadata verification via Track B. Independently parse final configuration, compare audited target semantic values, require K=8/v3/8,192 positions, and use **alias-aware `safetensors.torch.load_model(strict=True)`**. No final load uses `load_weights_only`.
- Construction temporarily disables external pretrained encoder *weight* initialization, then restores the recorded config flag. Strict final weights supply every learned parameter. The external Hyena config/code must already be available in the pinned offline runtime cache.
- Baseline initialization uses the concrete source checkpoint path from the verified initialization inventory and the approved `load_weights_only` 0→8 loader; projection is asserted zero. No mutable `best` lookup.
- Paired runs reuse the same loaded final checkpoint. Fractions are all-eight-channel zeros versus the separate on artifact. Seed 123 has only fraction 0.
- Explicit `<s>` decoder prefix/start ID 0, grammar v2+v3, beam 2, deterministic sampling disabled, cache enabled, **`max_length=8192` total**, not a new-token budget. A fresh generation config avoids inherited generation overrides. Runtime encoder hook rejects lost evidence or repeated encoding.
- Raw IDs/tokens/planning markers are written before string conversion and grammar validation; missing EOS/truncation/malformed output fails the run. Canonical GSF is checked against `gsf_contract`; canonicalization is compared, never used to silently merge isoforms.
- Split `<gene>` blocks before legacy conversion. Deterministic gene/transcript/feature IDs replace UUID roots without collapsing inventory; minus strands remain genomic forward. Relative half-open coordinates become genomic 1-based inclusive coordinates. Strict ID/Parent syntax, uniqueness, hierarchy and containment checks apply.
- A successfully decoded `<empty>` yields exactly `##gff-version 3\n`. Zero-byte/whitespace GFF is invalid. No malformed/truncated sequence is turned into an empty prediction.
- Fresh directory only, distinct output file per row, exact 350 inventory: 50×(1/2/3×off/on + 123×off). `evaluation.json` is the last publication, using the existing producer transport with checkpoint/file hashes. `decode_audit.json` binds input/source bytes and raw outputs. Input identities are rechecked after decoding. Failure leaves `FAILED.json` and partial diagnostics, not a final transport/approval/count receipt.

## Verification

Command in this worktree:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src .cpu/bin/pytest -q \
  tests/test_mvp_decode.py tests/test_mvp_model.py tests/test_weights_init.py
```

Result: **35 passed, 24 subtests passed** (12 new decoder tests). Python 3.10.12, torch **2.6.0+cpu**, Transformers **4.46.3**, NumPy **2.2.6**. CLI `--help`, Python compilation, and `git diff --check` also passed. Graft refreshed locally, not committed.

Coverage includes real alias-safe strict checkpoint loading with learned nonzero projection; missing projection rejection; config/audit mismatch; actual approved source-weight loader baseline reconstruction; tiny real-model beam-2 cache/no-cache parity and evidence forwarding; malformed evidence; negative dropped-evidence hook; grammar masks under beam reorder; raw transcript-plan mismatch; EOS/truncation; multi-gene/minus coordinates and deterministic hierarchy; duplicate isoform rejection; invalid/whitespace IDs; N-mask/artifact checks; mocked full 350-output orchestration, overwrite rejection and partial-failure publication behavior.

The 350-output orchestration fixture intentionally mocks reference/final/input gates and generation. Tiny model tests substitute only external Hyena factories. Neither establishes production reference approval, real evidence provenance, GPU execution, or full-model sequence quality.

## Remaining integration/runtime gates

1. Integrate reviewed producer fixes and finalized panel-input contract; rerun real finalizer/candidate/source inventory checks. The old verifier is deliberately unusable here.
2. Supply and validate the actual pinned offline encoder tokenizer and external Hyena config/code cache; byte identities must be approved by the controller.
3. Run the separate worker-A **Transformers 4.57.6 / Accelerate 1.15.0** probes and actual production GPU model/checkpoint decode smoke. This branch has not run those versions or any GPU job. Confirm explicit grammar start prefix and 8,192-total-token stopping on the production checkpoint.
4. Run a real panel decode and Track-B GFFCompare 0.12.6 smoke. No real 350 prediction corpus, counts, or scientific receipt was produced here.
5. HiFi/ONT scoring, withholding policy, and scientific decisions remain outside this decoder.
