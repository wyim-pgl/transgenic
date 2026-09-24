# WP4 Writer C progress (branch b6-wp4-model-trainer, base daf8436)

Contract: notes/a46_approval/B6_WP2_WP4_PLAN_v4_20260923.md (v4.1), sections 9 and 10.

**Gate status: UNGATED: memory bench not yet run.** Trainer code (CP3) is written ahead of the CP2 memory gate
as instructed. It is not ready: `main` refuses unless configs/b6_400m_win_v3.json says
`gate_status = "GATED: memory bench passed"` (or `--allow-ungated`, integration only). Nothing was submitted.

## Environment

- torch tests: private venv on the `pdb` conda env (torch 2.6.0+cu124, transformers 4.46.3, accelerate 1.10.1,
  pytest 9.1.1, duckdb 1.5.5), created with `python -m venv --system-site-packages` from
  ~/miniconda3/envs/pdb at
  /tmp/claude-1000/-mnt-f-scratch-develop-Transgenic-transgenic-local/1a00e258-5d2a-4bd8-a5a4-61423fb2e076/scratchpad/b6venv.
  Nothing was installed into a shared env. It lacks pyarrow, so WP1 tests run in the MVP venv instead.
- torch-free tests and the MVP byte-pin test: /tmp/b6-mvp-verify.iT7cl7/.venv (no torch).
- The host has a Quadro K620; trainer tests pin `Accelerator(cpu=True)` so they are CPU tests.

## Track spec (Writer B)

`/tmp/b6-wp3-tracks/tasknotes/TRACK_SPEC_READY.txt` appeared mid-task. Copied byte-identically:
`src/transgenic/evidence/b6_track_spec.py` sha256
`6f5f1d6d1cfb983a2422070b184f4be10fa6f9a978edc3c9fd6c96c726b59712` = marker. `b6_runtime` now imports its
constants and calls `validate_batch`; the model was aligned to it (scalar `w_count`, spec meta keys).
Re-verified after the lead's confirmation. `tests/b6_fixtures.py` copied byte-identically, sha256
`b17c3f92aefdec227c21fb56314a83f3afe7d4e9ceef3fbeda66ca127f1d0f10`; `WP3StubBatchTests` runs the model, every arm,
the de novo view and beam generation on B's `stub_batch_torch` (including its chain-less row).

## Components

- [x] CP1 `src/transgenic/model/modeling_b6.py`: ChainMemory, gated ChainCrossAttention, B6DecoderLayer,
      `b6_decoder_forward` (chain tensors explicit under non-reentrant checkpointing; LayerDrop as a select),
      B6TransgenicModel, B6 LM model (B5 loss term for term), generation (encoder + chain memory once, HF beam
      repeat, no chain K/V cache). Contract integration `tests/test_b6_integration_contract.py`.
- [x] CP2 `revision/scripts/b6/bench_memory_129k.py` + `.sbatch` written, NOT run.
- [x] CP3 (UNGATED) `src/transgenic/training/b6_runtime.py`, `train/train_HyenaTransgenic_B6.py`,
      `configs/b6_400m_win_v3.json`.
- [x] CP4 driver `revision/scripts/b6/integration_4rank.py` + `.sbatch` written, NOT run on GPUs; CPU smoke
      (2 gloo ranks, tiny model, aot_eager) PASS.
- Shared synthetic plan-7.7 rows: `revision/scripts/b6/wp4_synthetic.py`.

## Tests

- torch venv: 50 WP4 tests pass (49 + WP3 stub-batch test); MVP byte-pin test `tests/test_mvp_final_source_integration.py` 3/3 pass.
- MVP venv: all torch-free `tests/test_b6_*.py` (WP1) pass, 186 passed; the only 2 failures are the two pin-file tests
  that import torch (`No module named 'torch'`), which pass in the torch venv.
- Only new files; `git status` shows no tracked file modified.

Command (torch venv):

```
PYTHONPATH=src:revision/scripts:revision/scripts/evidence:revision/scripts/mvp $B6VENV/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_b6_model.py tests/test_b6_integration_contract.py tests/test_b6_trainer.py \
  tests/test_b6_bench.py tests/test_b6_ddp.py
```

## Defects found and fixed while building (all covered by tests)

1. DataLoader iterator creation draws its worker seed from the global torch RNG; on a resumed link that draw
   lands after the restored RNG state, so dropout masks diverged and the step checkpoint was not byte-stable.
   Fix: each loader owns a `torch.Generator`.
2. `safetensors.save_model` writes the tied-alias map in a varying key order (three names share the decoder
   embedding), so identical weights gave different bytes. Fix: `save_weights_deterministic`.
3. accelerate 1.10.1 `unwrap_model` raises KeyError('_orig_mod') for DDP(OptimizedModule). No matching issue
   found (gh search huggingface/accelerate, two queries). Fix: own unwrap.
4. accelerate `skip_first_batches` loader resets the sampler to epoch 0 on iteration. Fix: set_epoch on it.
5. Eval batches with varying label length recompiled after warm-up (guard tripped). Fix: mark labels dynamic.
6. LayerDrop as a Python branch on a tensor is a Dynamo graph break whose frames compile late. Fix: select.

## Spec update (D7, WP3 Codex item M5)

Re-copied byte-identically on 2026-09-24: `b6_track_spec.py` sha256
`516835ca42189680020d7db324a8ff1a5e9d89cfa7c8ef4037bcb93bc22c8e43` (= new marker); AUG_SPEC_SHA now
`42cc22a056a70c062fcfb6c60fbf41537c4906eb86be43d2d46d3137263bfdf7`; TRACK `eed27580…` and CHAIN `e7d35fdd…`
unchanged. WP4 pins no spec sha literal: identity and tests read `SPEC_SHAS` from the spec. `tests/b6_fixtures.py`
re-copied, sha256 `6d2d4698df1a8fda326d255e985a21ec080a16d57fa1769c1f709e4ed60128d6` (was `b17c3f92…`).
Torch venv after the update: 62 passed (59 WP4 + 3 pin); MVP venv torch-free tests/test_b6_*.py: 185 passed. One earlier full run had the 2-process DDP test abort
(SIGABRT in a gloo worker); it passed 3/3 alone and in the next full run, so it is recorded as intermittent.

## Fix round 3 (Codex review 3: APPROVE_WITH_FIXES; M8 finished)

- CP4 re-verifies on every simulated link: in `--data wp3-fixtures` each of the three links (straight, preempted,
  resumed) calls the real verify_inputs again through `verify_training_inputs` with a fresh cache handle, as
  production `main()` does; the report lists the three record_sha256 values and fails unless there are three and
  they are identical.
- `--data` is a required argument (no implicit synthetic default); the sbatch requires `DATA=synthetic|wp3-fixtures`
  and documents reachable counts on 4 ranks over 2 epochs: synthetic 48 rows, accumulation 8 -> 12 updates,
  n-pilot 5, stop-at 3; wp3-fixtures 5 train rows -> ceil(5/4) = 2 micro/rank/epoch, accumulation 4 -> 4 updates,
  n-pilot 3, stop-at 1. The driver refuses `stop_at < n_pilot <= reachable` violations (exit 4; the distributed
  launcher reports a child failure as 1, and the REFUSED line is in the log).
- TODO post-merge: label RC on WP3 fixtures. The fixture gff is a placeholder (`gsf<rn>`), so fixture-mode CP4 does
  not assert label RC (synthetic mode does, via reverse_complement_v3). The lead adds real paired labels to the
  fixture at integration; then add the label assertion to `wp3_fixture_data`.

Results: torch venv 5 WP4 files + MVP pin: 64 passed, 3 skipped (needs merged b6_bindings), 91 subtests. Synthetic
CP4 (2 CPU ranks, aot_eager), stops 2 and 4: PASS, resume byte-identical. Unreachable n-pilot 7 refused. Overlay
`--data wp3-fixtures` (2 ranks, accumulation 2, n-pilot 3, stop-at 1): PASS, 2 twin pairs, 3 links re-verified with
one identical record_sha256, resume byte-identical.

## Fix round 2 (Codex review 2: B6_WP4_CODE_REVIEW2_20260924.md) + WP3 verify_inputs record change

- H3 architecture gate: `b6_runtime.FROZEN_ARCHITECTURE` (the win_v3 400M block plus the built facts: 12 decoder
  layers, 12 chain blocks, evidence_proj [768, 31], chain memory). `model_architecture(model)` reads the BUILT
  model; the bench's gate condition `architecture_sha256` must equal `FROZEN_ARCHITECTURE_SHA256`
  (`7f941f5c32ba77dc79dbcf8c179758d378f0aa012b0cf1a155fc4a6b9d450b7a`), else NOT_A_GATE_RUN. `load_b6_config`
  also refuses any drift from the frozen architecture fields. Tests: the frozen config built on CPU (stubbed Hyena)
  equals FROZEN_ARCHITECTURE; decoder_layers = 1 gives NOT_A_GATE_RUN; the loader refuses decoder_layers 1,
  d_model_decoder 768, decoder heads 2.
- Freeze path (WP2/WP3 HIGH): `verify_training_inputs` passes `paths["freeze"]` (the FILE path) to WP3's
  verify_inputs, plus optional references, and checks the record's record_sha256 with WP3's
  `canonical_record_sha256` before the identity is built; `main` sets `record_sha256_fn` for every link.
- WP3 record shape: identity maps family, evidence/roles shas, coverage, spec (flattened), aug_seed, draws, evlabel,
  [reference_sha256], exhaustiveness, corpus/base shas, interval_manifest_sha256 {species: sha}, revalidated,
  zero_lists, provenance {name: sha256}, cache_manifest_sha256 and record_sha256 (binding.record_sha256 = WP3's).
  A record edited after sealing is refused.
- Bug found by the new WP3-fixture CP4 mode and fixed: the live record holds tuples (coverage) and the stored
  identity JSON lists, so a legitimate resume was refused. The identity is now built in canonical JSON form;
  regression test `test_resume_accepts_a_record_with_tuples`.
- M8: `integration_4rank.py --data wp3-fixtures`: rank 0 builds WP3's fixture world once (it is not byte-
  deterministic), every rank verifies it through the trainer's `verify_training_inputs` (real verify_inputs,
  freeze path), asserts each forward/RC twin (rn 1/2, 7/8) under one fixed draw: DNA reverse complement, tracks
  = rc_tracks(forward), chains = rc_chain(forward) apart from rule 6a slots (keyed by row). The fixture gff is a
  placeholder, so the label RC is asserted only in synthetic mode. It then trains through WP3's B6Dataset and
  b6_collate_fn_torch with preempt + resume and runs generation parity on the twin batch. Post-merge mandatory.
- DDP diagnostics: each worker logs native stderr and faulthandler tracebacks per rank, with
  TORCH_DISTRIBUTED_DEBUG=DETAIL and TORCH_SHOW_CPP_STACKTRACES=1; a failure prints both logs. 10/10 passes.
- The three WP3-dependent tests skip in this worktree with "needs merged b6_bindings (mandatory after the
  b6-integration merge)": WP3StubBatchTests, test_corpus_rows_on_the_wp3_fixture_corpus,
  test_verify_training_inputs_runs_the_real_wp3_verifier_with_the_freeze_path.

Overlay (read-only scratch copy of this worktree + WP3 files, nothing written back to either worktree):
<scratchpad>/overlay. WP3 sources (uncommitted, /tmp/b6-wp3-tracks):
  b6_aug 15328a26…, b6_bindings afc39336d38d3e1ba75a83c6a4f9e27b81f5a3baf9355dd47f28824802285d37, b6_chains c6efeb1b…,
  b6_context 4ecec15f…, b6_obs_cache 846292d4…, b6_tracks 99509d52…, b6_dataset 1c2ddb9e…, b6_track_spec 516835ca…,
  b6_fixtures 1a7b6b07…. In the overlay the 3 skipped tests pass (7 selected passed), and
  `--data wp3-fixtures` on 2 CPU ranks (aot_eager): PASS, 2 twin pairs, resume byte-identical, generation parity.
  pyarrow was installed into the private venv for this (WP3's cache writes parquet).

Worktree tests after round 2: torch venv, 5 WP4 files + MVP pin: 64 passed, 3 skipped, 91 subtests; MVP venv
torch-free tests/test_b6_*.py: 185 passed. Synthetic
CP4 smoke (2 ranks, stops 2 and 4): PASS, resume byte-identical, 10 RC pairs.

## Spec/fixture check after the lead's stale-copy note

On re-check the spec copy was already current (516835ca…, cmp-identical to WP3). WP3's fixtures had moved on
again: re-copied `tests/b6_fixtures.py`, sha256 `1a7b6b07fcee9164d3855908a1a30b20d173e076995181da288ef907d640291a`
(was 6d2d4698…). This version imports WP3's `b6_bindings` at module level, which is not in this worktree, so the
two tests that use it (`WP3StubBatchTests`, `test_corpus_rows_on_the_wp3_fixture_corpus`) skip with that reason
until the b6-integration merge. The pinned-schema corpus_rows test is split out and always runs. Torch venv:
61 passed, 2 skipped (59 WP4 + 3 pin, minus 2 skipped, plus the split). CP4 smoke (2 CPU ranks, aot_eager),
stops at update 2 and 4: PASS, resume byte-identical, 10 RC pairs checked.

## Fix round 1 (Codex review notes/a46_approval/B6_WP4_CODE_REVIEW1_20260924.md: REJECT, 3 HIGH + 5 MEDIUM)

All eight fixed, each with a test in `tests/test_b6_trainer.py::ReviewRound1Tests`, `tests/test_b6_bench.py`
or `tests/test_b6_model.py`:
- H1 `corpus_rows` selects `gene_id AS wid` (twin identity, as WP3's `B6Corpus.twin_rns`); tested on
  `build_b5.ensure_schema` and on WP3's `b6_fixtures.write_corpus`; a schema without gene_id is refused.
- H2 every resumable checkpoint (latest_state, epoch_NN) and every step checkpoint carries `identity.json`; it is
  compared with the current verified identity before `acc.load_state`. Foreign-arm, foreign-seed and missing-
  identity checkpoints are refused, and the test shows load_state is never called.
- H3 bench status starts FAIL. `steps > warmup >= 1` is required (exit 4 otherwise). PASS needs every rank to
  complete all measured updates with verified compiled execution. Settings other than 4 ranks, 129,024 nt,
  511 elements / N_tok 512, 8,192 labels and 100 GB give NOT_A_GATE_RUN (exit 3). The verdict is a pure,
  tested function.
- M4 chain block: dropout 0.1 on the softmax weights before the value product, plus the output dropout. The
  eval-mode bit-identity tests still pass; C2/C7 still execute the block.
- M5 compile verification runs on every link after that link's own warm-up (per-process update count), or at its
  last update if shorter. It requires new Dynamo graphs, and for Inductor generated kernels or FX-graph cache
  hits (a resumed link on a warm cache loads, rather than generates, kernels).
- M6 identity now requires `init_weights_sha256` (digest of the actual built weights), `binding.provenance`
  ({PROVENANCE file: sha256}), `binding.draws`, `draw_spec` and `binding.record_sha256` (the whole verifier
  record). WP3's draft verifier returns no `provenance` yet, so training refuses until it does (fail closed).
- M7 `data_order.jsonl`: per epoch, sha256 of the first 1,000 rows of the sampler's canonical order
  (`epoch_rows`), independent of sharding and resume. `draw_log/rankNN.tsv`: row, epoch, uniforms_sha256 for
  every trained sample. Tested equal across B6/C2 and across a resume.
- M8 synthetic rows are forward/RC twins: DNA reverse complement, `reverse_complement_v3` label, `rc_tracks`,
  `rc_chain`. `assert_rc_pair` re-derives all four; the 4-rank driver asserts every pair before training and
  runs generation parity on an evidence twin pair plus a dropped tile. The post-merge rerun on WP3 fixture
  batches remains required acceptance work.

Tests after round 1: torch venv, 5 WP4 files + MVP pin file: 62 passed (59 WP4 + 3 pin), 81 subtests.
MVP venv, torch-free tests/test_b6_*.py: 185 passed.

Local 4-rank driver smoke (2 gloo CPU ranks, tiny model, aot_eager), resume at update 2 and at 4: PASS, resume
byte-identical, 10 RC pairs checked, generation parity greedy and beam.

## Open issues for the lead

- GPU byte stability of step_04142 across a resume needs deterministic kernels (atomics in gather/embedding
  backward). `--deterministic` exists, is recorded, and is on in the 4-rank driver. Whether production arms use
  it (speed cost) is a decision.
- Inductor on CPU (torch 2.6) fails inside its C++ codegen (`AssertionError: bufNN`) for the unchanged B5
  model too, so it is not B6-specific; local compile tests use aot_eager. Real Inductor is exercised only by
  the DeltaAI bench and 4-rank run. No matching PyTorch issue found (closest: #136419, closed).
- WP3 interfaces used by `main()` are Writer B's uncommitted drafts of 2026-09-24 (verify_inputs record keys,
  B6Dataset, B6EpochSampler, B6Corpus, ObsCache) and Writer A's `load_roles`; re-check at the merge.
- The WP3 identity record has no PROVENANCE sha list (rule 9 tool versions). Training now refuses without it
  (`binding.provenance`, {name: sha256}); Writer B must add it to `verify_inputs`' return.
- Plan 9 says B6 data order = B5's. `b5_data_order` reproduces accelerate's SeedableRandomSampler over the
  B5 train list (tested against accelerate). B5's LR schedule counts steps from the unsharded loader length
  over the per-rank accumulation (4x more steps than real under 4 ranks); B6 copies that for parity. Flag.
- Zero tracks + masked chains give the sequence-only bits exactly in eval mode (tested). In train mode the
  chain block's dropout draws RNG even when every chain is masked, so a train step differs from the chain-less
  (B5-structured) forward. Every arm, C2 and C7 included, always runs the chain block, so the arms' RNG
  consumption is identical.
- The 4-rank driver uses synthetic plan-7.7 rows; rerun it on WP3 fixture batches after the merge.
