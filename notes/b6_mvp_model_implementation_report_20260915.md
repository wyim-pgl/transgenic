# B6 MVP model-only implementation — 2026-09-15

## Scope and provenance

- Base: `abca40f` (`pi-agent-611e8432-78e4-474`), checked out detached first as requested; implementation branch `pi-agent-b6-model-increment` in isolated worktree `/tmp/pi-agent-1765f0e2-cf6c-4ed-469f7cc7`.
- Implementation commit: the commit containing this report (resolve with `git log -1 --format=%H -- notes/b6_mvp_model_implementation_report_20260915.md`). No merge/push or main checkout writes.
- Approved parent spec read completely: `revision/protocols/B6_MVP_PILOT_v1.md`, SHA256 `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`.
- Copied model-only brief: `notes/b6_mvp_model_increment_plan_20260915.md`.
- Production edits limited to `src/transgenic/model/configuration_transgenic.py` and `src/transgenic/model/modeling_HyenaTransgenic.py`; new tests in `tests/test_mvp_model.py`.

## Implemented behavior

Validated nonnegative integer `evidence_channels` (boolean rejected), default zero; config save/load roundtrip. Enabled encoders have one bias-free K-to-encoder-width projection. A module marker directs the shared HF initializer to zero this projection, including parent initialization traversals. Disabled models construct no evidence parameter and preserve baseline state keys and CPU RNG consumption.

Tracks are fused on per-base Hyena states before positional injection and actual convolutional downsampling. Validation covers exact B/L/K shape, floating tensor type, and matching device; only dtype is converted. Omission gives no residual; supplying tracks to a fresh disabled encoder fails. Both model wrappers forward the kwarg. Precomputed encoder states are treated as already fused and do not run the encoder again. Generation/cache internals were not changed. No synchronizing finite-value scan was added.

The encoder kwarg is keyword-only after the existing `*args`, preserving positional call compatibility; wrapper kwargs are appended to existing signatures.

## Actual red/green evidence

Environment: `/home/wyim/miniconda3/envs/pdb/bin/python`, Python 3.12, torch `2.6.0+cu124`, Transformers `4.46.3`, safetensors `0.5.2`. CPU only; no downloads, installations, remote access, GPU jobs, or nested agents.

Repeated test command from worktree root:

```sh
PYTHONPATH=src /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s tests -p test_mvp_model.py -v
```

1. Before production edits: baseline actual wrapper forward/generation test passed (1 test). Captured ordered-state-key and RNG hashes from fixed-seed tiny construction at `abca40f`.
2. Config/projection tests added: 4 tests, 2 expected missing-feature errors (`evidence_channels` and `evidence_proj` absent). Config validation/projection initialization implemented; rerun 4/4 passed.
3. Fusion/wrapper/generation tests added: 9 tests, 5 expected missing-feature errors (unsupported forward kwarg and GenerationMixin rejecting unused `evidence_tracks`). Implemented encoder fusion and kwarg threading; rerun 9/9 passed without cache changes.
4. Added ordinary checkpoint roundtrip regression (no production changes needed): 10/10 passed. Added zero-projection training-loss gradient regression: final **11/11 passed**, 1.948 seconds, no skips.

Tests use only patched Hyena `AutoConfig.from_pretrained` and `AutoModel.from_config` factories; the tiny deterministic embedding backbone follows seeded torch initialization. Actual TransGenic wrappers, positional injection, convolutional downsampler, LED decoder, weighted training loss, and HF GenerationMixin execute.

Verified:
- Config 0/8 roundtrips and invalid negative/bool/float/string/None rejection.
- Default/explicit-zero state key and RNG digests equal the preincrement baseline in this environment.
- Encoder width 8 versus decoder width 16; projection is 8-by-8, bias-free, zero after nested construction and explicit shared initialization traversal.
- After copying identical shared weights, bitwise equal step-zero logits for omitted, all-zero, and finite nonzero tracks versus disabled B5-style model. The only missing key in this test-only shared-weight load is `transgenic.encoder.evidence_proj.weight`; no unexpected keys.
- Nonzero projection changes pre-positional encoder states and logits, accepts float64 tracks by dtype casting, and gets finite nonzero gradients. Zero-initialized projection also receives finite nonzero gradient from actual label loss.
- Learned projection with zero evidence equals the current model's omitted-evidence logits, not an assertion about original seed123 weights.
- Invalid rank/B/L/K, integer/complex/non-tensor input, and CPU/meta device mismatch rejected; disabled fresh input rejected.
- Generation encoder hook sees the original tracks tensor exactly once per call. Cached and uncached greedy sequences match for four generated tokens with nonzero projection. Every cached next-token argmax equals full-prefix no-cache teacher forcing. Float32 logits comparison uses `rtol=1e-5, atol=1e-6` for incremental/full-prefix arithmetic.
- Precomputed states bypass encoder and are not fused twice.
- Disabled and learned-enabled models roundtrip through both `safetensors.torch.save_model/load_model` and HF `save_pretrained/from_pretrained` with safe serialization. All state tensors and logits match bitwise; decoder embedding aliases and tied LM head identity remain intact, including the saved nonzero projection.

Additional check: loaded unchanged `configs/b5_400m_win_v3.json` with `HyenaTransgenicConfig.from_json_file`, asserting `evidence_channels == 0` and encoder/decoder widths `(768, 1536)`: PASS. `git diff --check`: PASS.

## Unavailable checks and deferred work

Attempted broader command:

```sh
PYTHONPATH=src /home/wyim/miniconda3/envs/pdb/bin/python -m unittest discover -s tests -p 'test_mvp_*.py' -v
```

It ran the then-current 10 model tests successfully but failed importing `test_mvp_tracks` because this environment lacks `duckdb` (11 reported entries, one import error). This is not a passing data regression run. Existing pytest-based encoder-init/model-adjacent tests were not run: `importlib.util.find_spec('pytest')` returned absent; no installs authorized. No full-suite success is claimed.

**Real Hyena + production SIF validation remains pending**, including real seed123 shared-checkpoint initialization, three real tiles, production Transformers 4.57.6, bf16, compile, DDP, and the fail-closed generation hook in that runtime. CPU factory substitution does not certify these gates or Hyena400M.

Audited `--init-from`, checkpoint/config compatibility enforcement, dataset/collate/trainer changes, and new runtime loading adapters are deliberately deferred under the model-only brief. The test's explicit missing-projection assertion is not a permissive production initializer. Ordinary evidence-model checkpoint roundtrip is not certification of original seed123 checkpoint loading. Controller independent review is still required before integration.
