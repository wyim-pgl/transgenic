# Frozen WP5 readiness audit — 2026-10-21

**WP5 is not ready for a gate decision.** C2 has a recorded fixed-step 4099 final; B6 final is absent in the latest receipt. Native CPU GFFCompare compatibility passed, but that establishes neither a WP5 evaluator nor a C5 result. This is a local, read-only audit except for this note: no SSH, network, job submission, large-data hashing, implementation, or scientific-policy change.

Source pin: `/tmp/b6-c2-recompile`, clean HEAD `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4`. Source citations below are relative to that worktree; `notes/` and protocol citations use the main operations checkout unless specified. The separate `50a7fac` proof/controller helper is not the model/evaluator source.

## Observed contract and identities

- A46 is approved; its numeric rules/commands require amendment to change. The later rule 11 amendment fixes **4099**, seed 123, matched fixed-step checkpoints—not best-validation checkpoints. Sources: `notes/a46_approval/A46_APPROVAL_RECORD_20260922.md:3-7`; `notes/a46_approval/A46_RULE11_NPILOT_AMENDMENT_20260927.md:8-21`; `revision/protocols/PROTOCOL_B1_frozen_v1.md:999`.
- Panel: `revision/protocols/b6_panels_v1.tsv`, SHA256 `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`; **pilot 300**, MVP 50, final 1729, disjoint 2079. WP5 uses only `panel=pilot`; final-panel gates remain separate. Sources: B1:1000; `B6_EVIDENCE_MODE_WORKPLAN_v1.md:64` (its 4142 is superseded).
- G1: lower bound of B6-de-novo minus C2 F1 **≥−1 percentage point**. G2: both B6-evidence minus B6-de-novo and B6-evidence minus C5 lower bounds **>0**. Pool transcript counts; recall denominator includes every reference transcript at all 300 loci, precision includes every predicted transcript under the frozen panel restriction. Empty C5 loci retain their reference denominator. GFFCompare 0.12.6 exact intron-chain; monoexonic exact CDS. Paired locus bootstrap: 2000, seed 20260914, percentile 95% interval. Preserve leakage/determinism/fallback/memory aborts; record applicable G3 and descriptive power. Source: B1:1000-1004.
- C2 final: `/work/nvme/bilv/wcyim/transgenic/runs/c2_compile_history_validation/3370048/resumed/step_04099`; metadata SHA `abcca47754137cbc04d68ab8b93ddd256b034c2f2ecdfe45ca81888eef942b9b`, declared model SHA `411358d18d51c36f16b191180d6c7b05d821025388b1bb50ca153a719e66d2cd`, fixed_step/pilot_final/seed 123. B6 expected final is `runs/b6_compile_history_validation/3425703/resumed/step_04099`; metadata absent. Production 3429921 remains pending in the recorded observation. Sources: `SCORER_RESULT.json:34-100`; `PROGRESS.md:7-9`. Model bytes were not rehashed here.
- Live binding-file pin: `b6_prep_20260925/bindings/train_v1.deltaai.json`, SHA `64e7fa926ecbcc74062f9cff9c3978f3bdd9de461029fd4b32dd78ee9e08eab3` (`notes/b6_pilot_resume_20261020/CONTROLLER_CONFIG.json:2-5`). Historical run declarations identify repaired corpus SHA `6583b1b2…f15111c`, cache manifest `11b4c757…9f42718`, merged evidence `22af410f…e01f98e`, and Athaliana FASTA `63d56962…9c63973`: `notes/checkpoint_archive_20261009/c2_failure_3345919/resume_determinism/3364480/cold/run_config_b6.json:10-13,129,427-439`. These historical declarations must be reconciled with current final-run bindings before use.
- Reference authority is TAIR10 `Athaliana_167_TAIR10.fa` / `Athaliana_167_TAIR10.gene.clean.gff3`, manifest MD5s `513e5ef30845ed754b00816a99abbf8e` / `0264c349bc61ccceb546648edec46ebb` (`data/manifests/b5_species_v1.tsv:2`). Current mounted GFF path/SHA and all 300 locus-ID/geometry mappings remain unverified.

## Reusable code; missing integration

**No dedicated WP5 evaluator or C5 reconstruction entrypoint was found in this source checkout.** Coverage: `graft map`; exhaustive `graft grep` for `WP5`, `C5`, `evidence-only`, `splice_graph` over 422 indexed files; tracked evaluator/scorer/decoder filename inventory; raw search only over 49 unindexed files among 432 tracked code/shell/Slurm files. Hits identify training controls, panel construction and MVP code. This does not establish absence on other branches or remote storage.

Reusable APIs:

- `b6_context.Panel(..., panel_id="pilot")`, `WindowManifest`, `eval_context`: verify panel/reference/roles/window identities and training-family Athaliana `eval_input`; source `src/transgenic/evidence/b6_context.py:163-220,258-325`. Tests exercise heldout-only inputs and reject scoring molecules (`tests/test_b6_eval_path.py:51-110`).
- `b6_build_tracks.dose_records` uses the same resolved molecule draw for `rasterise_with_qc` and `build_chains`; it archives retained IDs and track/chain hashes. The existing **input-production**, not evaluation, CLI is:

```sh
python revision/scripts/evidence/b6_build_tracks.py \
  --cache "$CACHE" --evidence "$EVIDENCE" --freeze "$FREEZE" --corpus "$CORPUS" \
  --reference "$FASTA" --panel revision/protocols/b6_panels_v1.tsv \
  --panel-sha 79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243 \
  --panel-id pilot --windows "$WINDOWS" --generator "$GENERATOR" \
  --roles data/manifests/DATASET_ROLES.b6_v1.tsv --out "$NEW_ARCHIVE" --arrays "$NEW_ARRAYS"
```

Variables are unresolved verified-input slots, not an executable readiness manifest. The CLI verifies upstream bindings and emits all frozen dose draws; source `revision/scripts/evidence/b6_build_tracks.py:50-122`.
- B6 generation exists with `evidence_tracks, chain_idx, chain_feat, chain_mask`, including cached/beam propagation (`src/transgenic/model/modeling_b6.py:546-581`). Integrate a strict B6 checkpoint loader and `b6_collate_fn` boundary conversion (`src/transgenic/datasets/b6_dataset.py:121-168`); raw `chain_pos` is not `chain_idx`.
- `mvp_produce_counts.reference_counts`: exact-chain/mono-CDS counting and receipt generation are useful (`revision/scripts/mvp/mvp_produce_counts.py:482-545`). It rejects duplicate structures; preserve that refusal pending a frozen multiplicity policy.

**Do not relabel MVP.** `mvp_decode_panel.py:119-221` requires 50/MVP profiles; `mvp_decode.py:128-188` requires 8 channels and loads the MVP model, whereas B6 uses 31 plus chains. `mvp_produce_counts.py:616-738` fixes MVP inventories/outputs. `mvp_bootstrap.py:161-189` returns percentiles **5/95**, appropriate to its one-sided MVP bounds—not automatically B1's stated 95% interval. Freeze an explicit WP5 statistical interpretation and tests; never silently reuse these bounds or alter thresholds.

## C5, independent scoring, runtime, and next work

Input roles are already frozen: `excluded_heldout` A-EST plus A-ONT2 SRR10611193–95; roles SHA `36aa934d29759e1594f098bd74b4a43225a64a77f9673b373e99a8db5f2f09f7` (`data/manifests/DATASET_ROLES.b6_v1.tsv:25-29` and sidecar). Independent scoring is A-ONT1 SRR28341943–45 plus A-CCS23 SRR22719002–07, panel-window±10kb prefilter followed by alignment against the full reference. Pool within source only, junction thresholds ONT 3/PacBio 2; retain per-run results. A-HiFi is missing, never zero; A-CCS23 cannot score TES. Source: B1:998. Actual 300-locus alignment products and measured MVP-based cost extrapolation are unverified.

C5 needs qualified termini, ≤32 support-product-ranked splice paths, canonical motifs, longest complete ORF ≥90 aa (`PROTOCOL_M_exploratory_v1.md:20`), and WP2c before any C5/G2 number (`notes/a46_approval/B6_WP2_WP4_PLAN_v4_20260923.md:97-113`). **WP2c code exists**: `b6_build_read_ends.py:137-166`, `b6_build_completeness.py:30-61`; frozen `CAGE_RECIPE.b6_v1.md:3-19` and CAGE inputs exist. Real read-end DB, approved sheets, mapped CAGE, four population-specific admitted-chain manifests, completeness receipt and C5 outputs remain unverified. C5's exact A-ONT2 integration, path ties and candidate-to-transcript policy need specification review if not already frozen elsewhere.

CPU 23105376 completed 0:0 / 7 s / 1 CPU / 1 GiB / 0 GPU: pinned `/u/wcyim/transgenic_ops/wp5_cpu_scoring_20261021/bin/gffcompare`, SHA `a440e1533611a40a51cf2b5b35a2e15ec6b885660dd35f6b531e0d65601a963a`, exact=`=`, altered=`j` (`SCORER_RESULT.json:3-30`). CPU evidence preparation/scoring/bootstrap and GH200/ARM model decoding require separate verified environments. This tiny fixture does not establish NumPy/DuckDB/PyTorch, real-data correctness, memory, or allocation bounds.

**Smallest next work:** inventory current bindings/reference/window/WP2c artifacts into a fail-closed manifest; then implement and independently test a separate WP5 count/bootstrap adapter on synthetic 300-locus inventories. Resolve window-generator/prediction-scope/C5 ambiguities against frozen sources before changing scientific behavior. Missing generated hashes and routine packaging are implementation work; changed panel, denominator, thresholds, source roles or frozen commands require amendment. B6 final completion and reviewed GPU inference remain later dependencies. No ready-manifest, WP5-pass, or model-completion claim is supported yet.
