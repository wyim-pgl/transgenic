**Verdict: REJECT.** The [plan](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/B6_WP2_WP4_PLAN_v1_20260923.md) is not ready for writer handoff. Its header acknowledges requirements that remain absent from the implementation contracts and assignments.

References below use **P** = [frozen protocol](/mnt/f/scratch/develop/Transgenic/transgenic_local/revision/protocols/PROTOCOL_B1_frozen_v1.md), **SPEC** = [command specification](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/A46_C0_AND_TRACKS_COMMAND_SPEC_v2_20260922.md), and repository-relative code paths.

1. **CRITICAL — Add the missing C2 label-builder package and resolve its consumer.**  
   **Plan:** header, §1/WP2, §2. **Spec:** A18.4, A20, A50/A50.1; P:263–267, 318–325, 1162–1170.

   Add an explicitly owned **WP2b**: library, CLI, validator, fixtures, and hashed label artifacts. Specify the 14-class positive-cell/weight representation, coordinates, strand handling, phase representation, and unsupported cells with weight zero.

   Map `aligned_block` to exon/UTR evidence; accepted protein placements to CDS-family evidence; and accepted canonical `junction_observation` joined with molecule-level `junction_support` to **the entire intron interval plus donor/acceptor boundary cells**, respecting `excluded_4c`. Include A11/A16 units, library caps, genotype multipliers, `min(4,1+source_weight×genotype×log1p(n))`, retained-intron exon weight 0.25, protein junction support from ≥2 organisms, and the unit-weight ablation.

   Crucially, **C2 segmentation labels and A46’s C2 architectural control are different things**. A46 rule 8 still mandates GSF-only loss. State whether these artifacts are a separate deliverable or require an amended B6 training contract. Their absence blocks WP2 completion; the unresolved consumer blocks WP4/WP5 readiness approval. Independent model and benchmark development can proceed.

2. **CRITICAL — A50 cannot be implemented from the promised protein-C0 interface; completeness is needed before WP7.**  
   **Plan:** §2, §4. **Spec:** A49/A49.1, A50 safeguards; P:1137, 1149–1152, 1163.

   [Protein ingestion](/tmp/b6-wp1-20260922/revision/scripts/evidence/b6_build_c0.py:1042) discards CDS phase; lines 1124–1125 leave query length/aligned-query fields NULL. C0 therefore cannot establish query-start/query-end completeness, start/stop codons, or full CDS phase from the listed fields alone.

   Add a hash-bound protein-placement supplement or schema extension containing those facts and organism identity, derived from the frozen miniprot output and sequence inputs.

   Implement same-strand, splice-consistent CDS containment; incomplete-side neutrality; conflicting-CDS neutrality; long-read and EST UTR counters; EST interiors only; and 5C/3C-only long-read terminal boundaries. **§4’s “blocks WP7” is incomplete:** completeness gates WP2 terminal labels and any pilot C5 reconstruction using those boundaries. R8’s side DB is reasonable if its content, policy, CAGE inputs, population, and merged-DB hash are verified.

3. **HIGH — R1/R2 propose choices that conflict with frozen masking rules.**  
   **Plan:** §1/WP2, R1–R2. **Spec:** A33.3; A46 rules 5, 12; P:472–475, 991, 1000.

   A fixed flank of 150 is **not** the frozen uniform seeded 50–150 flank. WP1’s conservative exclusion fallback does not authorize replacing the DNA-mask rule.

   R2 cannot offer “partial mask” as an interchangeable solution: overlap closure requires removing affected labels, and masked fraction >0.60 requires dropping the tile. A46’s byte identity is **B6 versus its comparator**, not B6 versus A40.

   Edit the plan to reproduce the applicable seeded component masks, measure closure and threshold effects, and require a further amendment if preserving the existing row count conflicts with those rules. Keep `N_pilot=4,142` fixed unless amended; do not silently redefine it as two epochs over a changed corpus.

4. **HIGH — The proposed raster/cache interface cannot enforce its leakage promises.**  
   **Plan:** §1/WP3 lines 27, 32; §2. **Spec:** A46 rules 4, 6a, 10; A48/A49.

   `rasterise` lacks explicit tile identity/split, population, exclusions, and DNA/N-mask input. Track rows omit dataset/run, molecule state, and `train_tiles_excluded`; chain rows also omit enough role information to distinguish A-ONT2 from scoring ONT.

   Add a validated tile/population context and retain eligibility provenance or equivalent checked indexes. Enforce training/validation distinctions, molecule-wide exclusion, coordinate exclusions, and dataset-level `b6_input` selection. Reject `never`, `b6_score`, and the 121-nt sensitivity arm as training inputs. Evaluation chains must use A-EST/A-ONT2 or maize M-EST as applicable; M-HQ18 is tracks-only.

   Apply the same closure to labels and completeness clustering. For C3, restrict donor selection to the appropriate split: a same-tier derangement across the whole database would import test evidence into training.

5. **HIGH — Hash recording is underspecified and schema verification is insufficient.**  
   **Plan:** §2, R3, WP3 cache manifest. **Spec:** A46 rules 9, 13, 15; A49.2.

   [verify_schema](/tmp/b6-wp1-20260922/src/transgenic/evidence/c0_schema.py:394) checks metadata and column names, not publication validity, leakage counts, or exhaustiveness. Require a published merged snapshot, actual digest verification, zero leakage, and A49.2 checks: `C≤floor(0.01M)` per source and pooled species loss ≤0.001, without averaging percentages or small-denominator exemptions.

   R3’s A40→C0→repaired-corpus relationship is sound, but the actual binding is per-species/source `c0_provenance.inputs.corpus.sha256`, not a scalar `corpus_sha256` column.

   Bind and verify cache shards, roles, zero lists, completeness/label artifacts, both specification hashes, and repaired-corpus bytes at consumption and resume. Missing identity keys must fail: wrapping B5’s `if key in first` comparison preserves a bypass.

6. **HIGH — One cached rank cannot serve all training seeds and five evaluation draws; R7’s zero list is population-incomplete.**  
   **Plan:** WP3 lines 31–32, R4/R7. **Spec:** A46 rules 8, 13; A48; SPEC:238–242.

   Cache ranks by `aug_seed`, or regenerate them once per requested seed. A single stored rank cannot implement evaluation seeds `20260914+d`.

   R4’s counter-based draws are sensible; freeze encoding, purpose domains, epoch propagation to persistent workers, and resume behavior. Tracks and chains must share draws.

   R7 needs more than “static zero list”: the current [agreement producer](/tmp/b6-wp1-20260922/revision/scripts/evidence/b6_routea_agreement.py:28) excludes both leakage-excluded and train-tiles-excluded observations. Its list cannot cover all validation/held-out phase populations. Define population-specific disagreement handling, independent of scoring molecules, and settle static-versus-post-thinning semantics explicitly. Availability staying 1 while an admitted library remains is reasonable.

7. **HIGH — Complete the tensor, RC, and decoder contracts.**  
   **Plan:** WP3 lines 33–34; WP4. **Spec:** A46 rules 6a–7.

   The specified track RC permutation and **`chain_idx=(pad+p)//6` are correct**. Preserve that arithmetic: `pad=1,p=5` gives 1, whereas separately dividing gives 0.

   Specify `labels:int64[B,T]` separately from nucleotide length, attention-mask dtype, chain-mask polarity, bounds, and safe dummy indices. Cast `chain_feat:int16` categorical fields before embedding lookup. State decoder query projection **1536→512** and residual output projection **512→1536**, layerdrop/checkpoint propagation, and finite all-masked attention; multiplying NaN by zero is insufficient.

   For negative-strand labels, do not blindly sort donor/acceptor into [add_junction_evidence](/tmp/b6-wp1-20260922/src/transgenic/utils/gsf_contract.py:526): it labels the lower coordinate as donor. Preserve biological roles while rasterizing the interval. Also qualify the α=0 identity test: α alone does not cancel nonzero track fusion.

8. **HIGH — Arm names do not specify the required trainer behavior.**  
   **Plan:** §1/WP4. **Spec:** A46 rules 8, 9, 11, 13.

   Add an arm matrix and acceptance checks: C2 zeros tracks/masks chains; C3 swaps both from one donor; C4 binarizes tracks with chains on; C6 removes phase with chains on; C7 executes masked chain blocks to preserve dropout consumption.

   Require identical initialization/data order, effective batch 96, de novo validation checkpoint selection, patience 3, ceiling 22, evidence-mode diagnostic loss, and exact step-4,142 pilot checkpoint publication. R5’s deterministic derangement is appropriate once its split, RC, and epoch rules are fixed.

   Correct the fallback citation: the operative problem includes `suppress_errors=True` at trainer line 320. Merely replacing the warning will not catch lazy compilation fallback.

9. **MEDIUM — “Only NEW files” overstates the byte pin.**  
   **Plan:** §0.2, §6. **Spec:** A28; chain-design §6.

   [The test](/tmp/b6-wp1-20260922/tests/test_mvp_final_source_integration.py:19) enumerates files existing at `681c5a0`; it does not freeze all future contents of those directories. The inspected pin contains 92 files, with no tracked pinned-file differences.

   The four named legacy modules are pinned, so new adapters/subclasses are appropriate. Existing WP1 `c0_schema.py` and `leakage.py` are **not pinned** and may be extended. Replace the blanket rule with an explicit protected-file list. New files also need explicit import/entrypoint wiring because pinned package initializers cannot casually change.

10. **MEDIUM — The writer split is nominally disjoint but operationally incomplete.**  
    **Plan:** §3, §6. **Spec:** A46 rule 15; SPEC:321, 338–340.

    Assign owners for labels, protein supplementation, completeness, role manifests, and shared fixtures. Resolve benchmark ownership explicitly.

    Model-plus-benchmark before trainer code is the correct ordering. However, B’s closing gate depends on C’s model, so final merge order A→B→C is insufficient. Add an early shared-contract/model integration checkpoint, then throughput and full training-memory benchmarks, then trainer implementation. Clarify that §3 does not authorize early trainer code before those gates.

11. **MEDIUM — Close the remaining ambiguities without reopening settled rules.**  
    **Plan:** R6 and omitted decisions. **Spec:** A46 rules 3, 6a, 8, 12.

    “Binary” C4 means 0/1, not `log1p(1)`. EST withholding must remove EST-derived phase. Protein query-level exclusion versus organism-level counting must remain distinct; protein thinning/withholding still needs an explicit contract.

    Add decisions for mixed-source identical-chain aggregation/source embeddings, 64 allocated slots versus 63 usable non-dummy slots, and genotype-name translation (`known_nonreference` in C0 versus `non_reference` in the weight helper). These are interface decisions with observable consequences, not details safely deferred to separate writers.

No files were modified; no training or jobs were run.