**Verdict: REJECT — HIGH contract gaps remain.** This reviews the [v2 plan](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/B6_WP2_WP4_PLAN_v2_20260923.md), not an implementation. A50.2’s separate `evlabel` scope, frozen RC permutations, and accepted right-padding of decoder labels are respected.

**A. Disposition of v1 findings**

| # | Status | V2 sections and remaining issue |
|---|---|---|
| 1 | **PARTIALLY** | §§1, 5 assign `evlabel` and separate its consumer correctly. Protein class semantics and junction-count validation remain incorrect. |
| 2 | **PARTIALLY** | §§4–6 add the protein supplement and completeness gates. Completeness still lacks necessary inputs and population-qualified output keys. |
| 3 | **PARTIALLY** | §2 fixes fixed-150 flanks, label removal, comparator identity and amendment handling. Component-flank semantics and the edge-gene API remain problematic. |
| 4 | **PARTIALLY** | §§3, 7.4–7.5 substantially repair eligibility and provenance. Evaluation-window construction and completeness population isolation remain incomplete. |
| 5 | **PARTIALLY** | §§8–9 add snapshot, exhaustiveness and required-key checks. Zero-list consumption and snapshot immutability need explicit checks. |
| 6 | **RESOLVED** | §§7.3–7.4 provide seed-specific ranks, counter draws, worker epoch propagation and population-specific zero lists. |
| 7 | **RESOLVED** | §§5, 7.7, 9 specify padding, gather arithmetic, categorical casts, projections, finite masking and qualified identity tests. |
| 8 | **PARTIALLY** | §9 supplies the arm matrix, checkpoints and fallback checks. C3’s withholding contract remains inconsistent. |
| 9 | **RESOLVED** | §0 correctly scopes protection. I independently verified all **92 pinned files**, with zero drift. |
| 10 | **PARTIALLY** | §§1, 10 resolve ownership and model/benchmark ordering. CP4’s pilot gate contradicts the completeness dependency. |
| 11 | **PARTIALLY** | §§7, 11 settle binary values and EST-phase withholding. Remaining decisions need the distinctions below; cross-source chain ties remain unspecified. |

**B. New or newly exposed problems**

1. **HIGH — Evaluation cannot use the sole context constructor (§§7.4, 7.9).**  
   `tile_context(corpus, rn, mode)` derives coordinates and split from a corpus row. Held-out inference windows need not coincide with corpus tiles, and **maize has no B5/B6 corpus rows**, as the [excluded-species manifest](/tmp/b6-wp1-20260922/data/manifests/b5_excluded_species_v1.tsv:2) explicitly states. Add an evaluation constructor accepting verified reference/window/panel identity, with a deterministic window key for augmentation and slots. It must enforce evaluation roles without requiring training-corpus membership.

2. **HIGH — Completeness is neither computable nor population-safe as specified (§6).**  
   The listed inputs lack read sequence or measured primer/adapter/poly(A) evidence and reference sequence needed for internal-priming checks. [C0’s alignment schema](/tmp/b6-wp1-20260922/src/transgenic/evidence/c0_schema.py:157) does not store these measurements. Bind the original sequence/alignment/reference inputs or a validated supplement, plus CAGE and admitted-chain hashes.

   Moreover, the same alignment can occur in both `train_track` and the larger validation population and receive different cluster-derived codes. Keying only by `(species, source, alignment_id)` cannot preserve both. Include **population** in the key and every consumer lookup.

3. **HIGH — C3 can silently skip withholding (§§7.3–7.4, 9).**  
   A recipient’s resolved library choice generally does not exist in a cross-species donor. Applying that `DrawSpec` therefore withholds nothing from the donor; source selection has the analogous problem when availability differs. Reuse recipient-keyed **uniform variates**, but resolve choices against the donor’s admitted sources/libraries. Compare underlying draws across arms, not selected library identities. Specify rejection/handling of singleton donor pools.

4. **HIGH — `evlabel` contains two concrete contract errors (§5).**  
   “Protein CDS … `protein_coding_gene` only” omits `exon`, which [A19.3](/mnt/f/scratch/develop/Transgenic/transgenic_local/revision/protocols/PROTOCOL_B1_frozen_v1.md:299) explicitly requires over aligned CDS blocks.

   Also, `junction.n_molecules_library_capped` is **not the filtered population count**: the [producer](/tmp/b6-wp1-20260922/revision/scripts/evidence/b6_build_c0.py:1567) aggregates all states and groups by dataset/run/library. Direct equality with post-eligibility, post-exclusion `n` can reject valid labels or encourage using excluded/duplicate support. Recompute distinct eligible units, deduplicate across runs, then apply library caps; define a scope-matched independent check.

5. **HIGH — Hash binding still permits unverified consumed inputs (§8).**  
   Zero-list and producer-QC hashes are explicitly *recorded*, while only cache manifests/shards are explicitly reverified. Rehash the actual zero lists/QC used and validate their population and upstream bindings at consumption/resume. A modified zero list must fail even when cache bytes remain unchanged.

   Explicitly reject WAL-backed corpus/evidence snapshots, matching [the existing published-C0 opener](/tmp/b6-wp1-20260922/revision/scripts/evidence/b6_build_c0.py:1139). Read-only access plus a main-file digest is insufficient as a snapshot contract.

6. **MEDIUM — Repair specification and fixture disagree (§2).**  
   A33.3 describes a seeded flank per **component**; v2 assigns independent flanks per newly masked **gene** using a new stream. Existing B5 code itself uses per-gene draws, so this discrepancy needs an explicit reconciliation, not an implicit claim of component-level compliance.

   Separately, [`genes_in_window`](/tmp/b6-wp1-20260922/src/transgenic/utils/gsf_contract.py:537) returns `(inside, partial_count)`, not partial genes. Enumerate edge-crossers explicitly. The existing synthetic `gL` is train/non-hard, so D1’s default does **not** select it; adjust the fixture or expected result.

7. **MEDIUM — Remaining ordering and deterministic-encoding gaps (§§6, 7.2, 10).**  
   §6 gates pilot C5 on completeness, while §10 says WP2c never blocks CP4, which includes “the pilot.” Separate pilot training readiness from completion of pilot comparisons.

   Specify slot hashing with UTF-8, `digest_size=8`, digest extraction and ordering exactly as rule 6a. The displayed shorthand otherwise differs from Python’s default digest size. With D8’s source-separated chains, identical ONT/PacBio chains can also share every stated sorting key; freeze a final source-order tie-break.

**C. Nine author decisions**

| Decision | Genuinely open? Is the default sound? |
|---|---|
| **D1** | **Partly.** Extending existing leak/hard masking to edges is defensible. Strict-held-out-only cannot waive applicable orthogroup/hard-mask obligations. Record the precise reading and reconcile flanks. |
| **D4** | **Yes.** Static lists per population are a sound conservative default; freeze this choice and verify their actual bytes. |
| **D5** | **Mostly settled.** Protein-query exclusion and dataset-level evaluation eligibility already bind. The default is sound. Low coverage warrants reporting; an amendment is needed to change eligibility, not merely because coverage is low. |
| **D6** | **Yes.** Fixed, within-split/tier, RC-matched donors are sound, subject to the withholding correction above. |
| **D7** | **Partly.** Query-level thinning versus organism-level support is not fully determined by rule 3. The default is defensible if explicitly frozen; distinguish thinning from whole-source withholding/dropout. |
| **D8** | **Yes.** Within-source merging preserves a meaningful source embedding. Sound once equal-key ordering is specified. |
| **D9** | **No.** Rule 6a already specifies slots 1–63 and dummy slot 0. Default is correct. |
| **D10** | **Partly.** `known_nonreference → 0.5` is fixed; unknown/hybrid remain 0.5. Protein `not_applicable → 1.0` is a reasonable explicit adapter convention. |
| **D11** | **Mostly settled by existing class semantics.** ORF-supported UTR bases belonging to a protein-coding gene is consistent with `_TYPE_CLASSES`; excluding stop-codon bases from UTR is correct. This does not excuse the missing protein-CDS `exon` label. |

No files were modified; no training or jobs were run.