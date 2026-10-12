# Independent MVP panel decoder review — 2026-09-16

## Scope and verdict

Reviewed detached `668299a` against `8803afb` in an isolated worktree. The complete supplied neutral diff is byte-identical to that git diff: 415 new implementation lines, 261 test lines, and the implementation report. Read the separate **final** `panel_input_contract.md`, committed evaluation transport contract, B6 MVP protocol, and A27 token-budget authority. No source edits, SSH, jobs, subagents, merges, or pushes.

- **Specification: PARTIAL.** The model/generation/output core substantially follows the requested decoder boundary, but the finalized producer artifact is not consumable, external runtime identity is not enforced, and publication is not atomic.
- **Correctness: CHANGES REQUIRED.** A definite channel-cap error rejects valid junction evidence. The atomic-publication defect is independently reproduced below. This is not a production-ready or scientifically approved decoder.
- The base producer's five reported findings and later `64b6d5b` fixes are a separate review. This review neither repeats nor certifies them. The old-hash veto blocks this exact base; changing a hash is not verification or approval.

## Findings

### D1 — High: coverage cap is incorrectly imposed on junction channels

`src/transgenic/evidence/mvp_decode.py:80` checks **all seven biological channels** against float16 `log1p(64)`. B6 §1.4 caps only coverage channels 0–2; junction channels 3–6 are distinct-molecule counts transformed with log1p, without that cap.

The actual rasterizer implements this distinction at `src/transgenic/evidence/mvp_tracks.py:77,102`. Existing `tests/test_mvp_tracks.py:73–84` explicitly expects coverage `log1p(64)` and a junction `log1p(70)` from 70 molecules. That valid output necessarily fails the decoder predicate. Any such boundary in any of 50 windows aborts the entire decode; silently clipping it instead would change the approved input semantics.

**Required:** cap-check only channels 0–2, preserve finite/nonnegative checks for all channels, and add a producer/rasterizer-to-decoder regression with >64 junction-supporting molecules, plus over-cap coverage rejection. The current decoder fixture uses ones and cannot catch this.

### D2 — Medium: last-file publication is not atomic and failure can leave `evaluation.json`

`revision/scripts/mvp/mvp_decode_panel.py:26–29,148–159` opens the final name with `open('x')` and streams JSON into it. Writing it last prevents normal generation failures from publishing, but does not make publication atomic. A write/flush failure or termination during serialization leaves a visible partial final-named transport. The exception handler does not remove it.

**Focused reproduction performed:** extracted the unchanged `write_json` function with Python AST (stdlib only); replaced `json.dump` with a writer that writes `{"predictions": [` then raises `OSError`. Observed:

```
evaluation.json remains after failure: True '{"predictions": ['
```

The temporary reproduction directory was removed. This demonstrates the publication boundary, not a successful scientific-receipt forgery: the ordinary downstream JSON parser should reject those truncated bytes. Nevertheless it contradicts the stated no-final-transport-on-failure behavior and makes concurrent consumption unsafe.

**Required:** write/close a sibling temporary file, complete final validation, then atomically rename to the publication name in the already-fresh directory. Add failure injection at final serialization/close and assert the final name is absent; retain diagnostics separately. Do not treat mere presence of a JSON filename as completion.

### D3 — High runtime identity gap: offline Hyena config/code is not pinned or recorded

`mvp_decode.py:219–229` suppresses pretrained *weight* loading but still constructs `HyenaEncoder`. `src/transgenic/model/modeling_HyenaTransgenic.py:423–430` calls `AutoConfig.from_pretrained(config.encoder_model, trust_remote_code=True)` and `AutoModel.from_config(..., trust_remote_code=True)` without an immutable revision or verified local snapshot. The runner's offline environment flags (`mvp_decode_panel.py:18–19`) prevent intended network access; they do not identify the selected cache bytes. Its source inventory (`:90–100`) does not include the external Hyena config or dynamically loaded implementation.

Strict safetensors loading validates tensor keys/shapes/aliases, not implementation semantics: two cache versions can expose identical parameter names and still calculate different outputs. The independently supplied local encoder-tokenizer snapshot is hashed, but that is not a pin for the model's separate external configuration/code. The implementation report correctly lists this as a remaining runtime gate; there is currently no executable enforcement of that gate before a complete transport can be emitted after producer integration.

**Required before operational use:** resolve approved immutable model config/code and tokenizer snapshots, verify the actual loaded files (including remote-code dependencies), bind their identities into the decode audit, and freshly recheck them. Bind the decoder environment/container and library versions as well. A controller-enforced frozen runtime wrapper can satisfy this; no new general loader framework is needed. Offline cache existence alone cannot.

## Separate, acknowledged integration blocker: final panel-input adapter

`load_panel_batch` (`mvp_decode.py:46–83`) accepts only `mvp_panel_batch_internal_v1`, absolute track paths and four caller-supplied pins. It does **not** consume the final producer directory with `COMPLETE`, its manifest digest, relative output paths, or complete sequence/provenance inventory. The implementation report discloses this accurately. Do not relabel an arbitrary internal JSON as the producer's verified output.

Concrete adapter acceptance requirements:

1. Require `COMPLETE`; verify its manifest SHA256 and the actual schema-version-1 manifest, purpose `est_mvp_panel_inputs_not_training`, and `training_eligible=false`. Reject partial directories, missing completion, and mismatches. Follow the producer's exact field spellings when integrating; the supplied contract does not enumerate all serialized key names.
2. Resolve generated paths against the producer output directory, reject escapes, and verify **all** manifest identities: sequence FASTAs, float16 `[30720,8]` arrays, per-locus provenance JSONL, and input/request/code inventories. Preserve byte-size checks rather than mapping only a track hash.
3. Join exactly the frozen 50 gene IDs and genomic-forward windows. Compare FASTA sequence bytes and declared sequence digests to the anchored reference slices; do not RC minus-strand loci. Bind the producer's candidate/corpus/evidence/reference/panel identities to the decoder's independently verified inputs.
4. Preserve the worker's excluded-heldout/primary-count selection and its seed-1/2/3 fraction-one equality checks. Reuse the same on-array; off is all eight channels zero. Do not replace absent heldout support with retained flank evidence, RN/tile membership, training Dataset output, ONT, or HiFi.
5. Preserve biological support summaries separately from availability. Accept biological-zero windows, N/n zero masking, junction-only overlap, and uncapped valid junction signals. Availability-one is not support.
6. Rehash completion/manifest and all consumed artifacts at the end of decoding, then publish atomically. Bind the consumed producer manifest into the decode audit. Changing files or completion state must prevent transport publication.

This is an integration requirement distinct from the three findings above, not evidence that the existing internal loader already verifies excluded-heldout scientific provenance.

## Positive boundary assessment

- **Final versus baseline loading:** `mvp_decode.py:203–242` uses alias-aware `safetensors.torch.load_model(strict=True)` for final models; it does not re-zero learned EST projection weights. It checks final K=8, vocabulary 290, decoder positions 8192, and available audited target semantic fields. The verified-run wrapper is essential; the helper alone is not a publication verifier. Baseline reconstruction uses the concrete initialization source directory with the existing approved `load_weights_only` path and asserts zero projection. It does not resolve a mutable `best` name.
- **Generation:** `:169–200` explicitly supplies `<s>` as both prefix and start, beam 2, no sampling, cache enabled, a fresh generation config, and `max_length=8192` **total**, not `max_new_tokens`. This agrees with A27's 8192-token/position authority (`PROTOCOL_B1_frozen_v1.md:376–379`). Encoder hook checks one non-null equal evidence tensor. Paired generation uses the same final checkpoint, with baseline only off.
- **Tokenizer/grammar:** `:86–109` requires 30720 tokens plus final SEP, strips only SEP and uses pad-ID masking. Raw IDs/tokens/planning markers are persisted before conversion/grammar validation. GSF v3 grammar, planned transcript counts, final EOS, bounds and canonical equality are checked. Length alone does not authenticate nucleotide token mapping; the approved tokenizer identity gate remains necessary.
- **Normalization:** `:112–166` splits gene blocks before legacy conversion, preserves minus strands in genomic-forward coordinates, checks feature bounds and CDS minimum lengths, and rewrites full hierarchical ID/Parent roots deterministically. Transcript inventory is checked; canonicalization is compared, not silently substituted to deduplicate isoforms. The legacy converter's feature-skipping conditions are screened by the preceding grammar/canonical/bounds checks. Valid `<empty>` becomes header-only GFF, not missing/whitespace output. No additional concrete normalization defect was established.
- **Inventory and failure handling:** runner `:116–159` constructs exactly 350 independent GFF rows with actual checkpoint hashes and calls the producer inventory validator. It refuses existing output directories. It freshly checks tracked input identities and reruns final/candidate gates after decoding. Normal generation failures leave diagnostics without publishing evaluation JSON; D2 covers the untested final-write exception.
- **Reference/provenance:** the runner calls the producer's anchored reference gate, not a caller-selected replacement hash. Its transport fields match `notes/MVP_EVALUATION_INPUT_CONTRACT.md`. Decoder file/command hashes remain provenance, not proof of execution. Producer integration must verify the reviewed code and rerun actual gates; the blacklist is only a veto.

## Evidence and limitations

Read all 415 implementation and 261 test lines, not just report summaries. `git diff 8803afb 668299a --check` passed; supplied neutral diff matched byte-for-byte. The focused publication failure above was executed without modifying source.

The controller supplied **35 tests passed in 6.98 s** in a different CPU environment. I did not independently rerun that suite; this isolated worktree's default Python cannot import torch. The implementer's “24 subtests” count is not independently observed here and is not combined with the controller result. No GPU or production decoder run occurred.

The tests genuinely include strict alias/projection roundtrip and baseline loader checks, tiny actual-model beam/cache behavior (external Hyena factories substituted), fake-model generation-argument/hook tests, raw-plan and normalization rejection, and mocked 350-output orchestration. They do **not** establish a full production-model `generate_one` run with the actual tokenizer/grammar/8192 budget, approved offline cache identity, final producer schema consumption, high junction counts, input mutation failure at publication, or final-write atomicity. The successful orchestration fixture mocks reference/final/input gates and generation.

Advisor consultation was attempted but unavailable because of a model/client-version incompatibility; no advisor conclusion was used.

## Integration next steps

1. Fix D1 and D2 with focused regressions; implement the final producer adapter and D3 runtime identity gate.
2. Integrate separately reviewed producer fixes (reported later commit `64b6d5b`); verify actual boundary behavior and immutable reference/final/candidate/source inventories, not just a new hash.
3. Run the approved production-version compatibility probes and real checkpoint/tokenizer/grammar cached beam-2 smoke, including total-token stopping and evidence forwarding. Then run the real 350-output decode and GFFCompare 0.12.6 transport smoke.
4. Keep all output exploratory/MVP. Independent structural scoring, coverage/withholding policy, and scientific approval remain separate gates.

Only this review report is intended for commit.
