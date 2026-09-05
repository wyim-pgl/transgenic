# Author decisions — 2026-09-05

Pending decisions, not author approvals. Reply with one line per issue using the choices below. No author-approved recipe, tokenizer, input annotation, frozen database, or protocol text has been changed for these decisions. The #55 ordering correction is implemented for future builds only; applying it to replace the frozen corpus remains pending below.

Measurements: GitHub #58 and #56, read 2026-09-05; outer `resume.md`, measured 2026-09-04. Frozen corpus: 272,224 rows. Full nine-species rebuild budget: approximately 1.5 hours, plus merge, validation and a new freeze; a species-only rebuild is cheaper but still requires a new merged artifact and freeze. These are not timings measured in this change.

Shared amendment constraint, protocol opening freeze declaration (applies to both issues): “Amendments are allowed only (a) before the first evidence alignment is run, or (b) for defects that make a rule inapplicable, and must be logged in §12 with date, reason and the state of the analysis at that moment.” Also: “Nothing in §§3–9 may be changed after the first result table is produced.” Evidence alignment is already underway; the proposed records are defect amendments appended to the protocol, not edits to frozen §§3–9.

## #58 — empty-label decoder sequence

The frozen database contains **25,159 empty tiles (9.24%)**, split 22,311 train / 1,809 valid / 1,039 test: the contract stores three tokens (`<s> <empty> </s>`), but the trainer emits two (`<empty> </s>`). Non-empty labels agree; the maximum stored label is 5,920 against 8,192, so this is a decoder-sequence mismatch, not observed cap truncation. This issue is independent of #56's **6,205 no-CDS genes, including 1,605 valid/test genes**; fixing tokenization does not repair their masking.

Frozen constraints (verbatim):

- GSF spec §1b: “a tile with no complete gene is labelled `<empty>` and only 10 % of such tiles are kept.”
- GSF spec §1b, token accounting: “`<s>` + blocks + separators + `</s>`”.
- Protocol §A27: “The v3 caps are therefore **96 genes and 8,192 tokens per window**, decoder positions 8,192.”
- GSF spec status: “Changes require a new version of this file and are logged in `revision/protocols/PROTOCOL_B1_frozen_v1.md` §12.”
- Protocol §A28: “Code changes inside a chain are forbidden; a changed recipe is a new labelled run.”

Open choices:

| Reply | Action and cost | Downstream effect | Required protocol record |
|---|---|---|---|
| `58 A: use three tokens` | Add `<s>` to the tokenizer's empty path and test the actual tokenizer/dataset sequence. **No database rebuild or refreeze**: stored counts already equal three; no nine-species ~1.5 h build. | All empty decoder targets become consistent with non-empty targets and stored counts; validation loss also changes. Apply before all B5 seeds; any already-started run needs a new labelled run, not an in-chain patch. | Append a dated defect correction in §12 / next amendment confirming the empty sequence under §A26/§A27; existing spec wording need not change. Record code version with the run. |
| `58 B: retain two tokens` | Keep tokenizer behavior; version the spec, counter, and tests to explicitly exempt empty labels from `<s>`. Stored counts must become two in a **new derived database and freeze**, or a full rebuild (~1.5 h for nine species); a metadata-only rewrite is sufficient technically. Never patch the frozen file in place. | Training targets stay as today; audits/counts agree after correction. Decoder-start convention remains asymmetric and must be documented; dataset/tokenizer integration tests must assert that intended behavior. | Append explicit §A26/§A27 empty-token exception and §12 entry; new GSF spec version and derived-artifact provenance. |

**Recommendation: 58 A.** It matches the frozen accounting and the start-token convention of every other label, costs no data rebuild, and makes the intended empty signal explicit before training. This is a recommendation only.

**If the author does nothing:** the corpus can be read and trained mechanically, with no measured token-cap overflow, but the corpus-plus-tokenizer is not consistent with the frozen token contract. It is not ready for an unqualified protocol-conformant B5 run; preserve it and defer that run. This does not make the stored labels irrecoverable.

## #56 — no-CDS genes bypass leakage masking

The builder drops **6,205 V. vinifera genes** before tiling because they have no CDS; they are correctly linked lncRNA-only loci, not broken protein-coding annotations. Of these, **1,605 carry a valid or test split (634 valid, 971 test; the other 4,600 train)**, so their sequence bypasses A29 masking in train tiles, and all 6,205 are absent from the species decoy-rate population. This affects a different layer from #58's **25,159 empty tiles**; their missing `<s>` does not explain or correct this bypass.

Frozen constraints (verbatim):

- Protocol §A29: “Inside a tile, any gene whose orthogroup split (#14) is more restrictive than the tile split (a test or valid gene in a train tile, a test gene in a valid tile) is **removed from the label and its sequence, with 100 nt of flank, is replaced by N**”.
- Protocol §A33.3 (superseding the flank above): “The masked interval of each component is padded by a seeded flank drawn uniformly from **[50, 150] nt** (replacing the fixed 100 nt)”.
- Protocol §A33.3: “Masking is applied to whole overlap-connected components: if any gene of a component is leak-, hard- or decoy-masked, every gene of that component is N-masked and left out of the label.”
- Protocol §A33.3: “In **train tiles only**, each otherwise labelled gene is masked with probability `min(0.05, m/3)`, where `m` is the species' realised leak+hard mask rate”.
- Protocol §A30.2: “Reference annotations are not edited; evaluation uses the full GFF (A22 rationale).”
- GSF spec §8: “Builders read only these manifests; a species not listed cannot enter the DB.” The manifest freezes exact annotation paths and md5s.

Open choices:

| Reply | Action and cost | Downstream effect | Required amendment |
|---|---|---|---|
| `56 A: accept quantified exception` | Keep the current database. **No rebuild**, no nine-species ~1.5 h cost. Surface no-CDS counts and split breakdown in reporting. | Existing sequences, tile membership and decoy rate stay fixed. Claims of A29 protection must explicitly exclude these loci; evaluate/report this limitation. | Append a dated A29 exception for these 6,205 lncRNA-only loci, including 1,605 valid/test, and the A33.3 population exclusion. The author must explicitly accept sequence exposure; omission of noncoding labels alone is not this approval. |
| `56 B: protect no-CDS loci` | Keep no-CDS loci in the masking/overlap population while excluding them from coding labels. Rebuild V. vinifera, remerge/revalidate/refreeze; **the frozen merged database must be replaced**. Full nine-species rebuild is a conservative ~1.5 h option, not intrinsically necessary if the other eight input/output hashes remain identical. | N-runs, component closure, decoy masks, mask-fraction drops, empty sampling/tile membership can change. Counts and training/validation losses must be regenerated; use the new artifact for every B5 seed. | Append an A29/A33.3 clarification separating masking eligibility from coding-label eligibility. Explicitly include these loci in the leak+hard rate population (numerator when leak/hard eligible, denominator all eligible loci); decoy candidates remain otherwise labelable coding genes. Freeze the revised recipe and artifact, keeping evaluation GFF unchanged. |
| `56 C: CDS-bearing training annotation only` | Derive a V. vinifera training-only GFF containing CDS-bearing transcripts (including removal of lncRNA transcripts from mixed loci); record hash and stable gene-key mapping. Rebuild that species and merged freeze; full nine-species alternative ~1.5 h. | Noncoding-only loci become outside the declared training annotation/masking universe, so their sequence remains visible. Mixed loci (1,647) and generated identifiers/split mapping require explicit verification; cannot assume this yields identical rows. Evaluation retains the full original GFF. | Amend training input manifest/spec §8 and A29/A33.3 scope to CDS-bearing training loci; expressly permit this training-only derivation under A30.2 while preserving evaluation reference. Record the excluded 6,205 and 1,605 split assignments, input hashes and mapping validation. This is a scope exception, not restored masking. |

**Recommendation: 56 B with the population definition in the table.** It preserves the full annotation and the broad A29 promise while keeping noncoding structures out of coding targets; the rebuild cost is modest relative to training. This population definition is part of the proposed author decision, not an implementation choice made here. Merely moving a rejection below masking without specifying component closure and the decoy-rate population is insufficient.

**If the author does nothing:** the database is mechanically usable but **not usable as a corpus satisfying the current unqualified A29 leakage rule**. Keep it frozen for audits; do not silently treat its valid/test sequence exposure as an accepted exception or launch a protocol-conformant B5 run on that assumption.

## #57 — implementation boundary

The missing diagnostics are reporting/validation defects. Emitting the A33.4 predicate on original gene coordinates (including cap/no-CDS rejects), aggregating reason classes, and checking `gene_split` do not require changing labels, masks, random draws, thresholds, offsets, splits, or recipe. No author decision is needed for those changes. The existing frozen database remains untouched; its report-time recomputation lacks contig lengths and is an upper bound, whereas a new builder can credit true contig ends from FASTA. These two measurements must remain distinguished rather than forced to agree.


### #57 implementation and verification

Implemented in `src/transgenic/datasets/build_b5.py`: nullable `build_manifest.tier_margin_unguaranteed` JSON per species, with all-original-gene denominator, per-tier bound exceedances and margin failures, FASTA contig-edge credit and missing-contig count. Non-tile builds store NULL. Stable rejection classes include `mask_fraction_dropped`, `no_cds`, gene/token/transcript/feature caps; raw rejection reasons remain intact. The validator checks the full `gene_split` orthogroup invariant, strict holdouts, invalid/excluded assignments, missing assignments and duplicate gene keys. This is an assignment-table check, not a claim that it detects #56's sequence masking bypass.

The report exposes recorded counters separately from its legacy recomputation. Existing databases without the new column remain readable, and merging preserves available counters while leaving older sources NULL. No frozen artifact was rewritten or rebuilt; no #57 recipe ambiguity required stopping.

Tested from isolated GPU-host stage `/tmp/transgenic-issue57-Fts0Rc`:

```sh
ssh gpu 'cd /tmp/transgenic-issue57-Fts0Rc && ~/micromamba/envs/transgenic/bin/python -m pytest -q tests/test_build_b5.py tests/test_merge_b5_databases.py tests/test_gsf_v3_windows.py tests/test_orthogroup_split_integrity.py tests/test_make_orthogroup_splits.py'
```

```text
62 passed in 9.84s
```

An independent synthetic before/after build comparison against `1413efd`, with RC enabled and seeds 123, 456 and 789, found all six non-manifest tables identical (`geneList`, `gene_key_map`, `gene_split`, `window_genes`, `tile_blocks`, `rejected_records`). This is fixture-level recipe parity, not a nine-species rebuild claim.

Read-only verification against the frozen merged database reproduced:

```text
genes considered: 334642
exceeds length guarantee: [3418, 653, 86]
margin failures (upper bound): [2034, 434, 44]
longest gene: 196414
gene_split: 334642 rows; 0 violations
```

`git diff --check` passed. No commits, protocol edits, `evidence/` changes or `*.sbatch` changes. #58 and #56 remain pending author decisions.

## #55 — disposition of frozen ordering rejections (pending)

**Correction to the issue's diagnosis:** the 2026-09-05 read-only audit of the frozen merged database finds 97 canonical-order rejections (Athaliana 8, Ppatens 10, Ptrichocarpa 1, Vvinifera 78). Only **13** compare equal spans; **84** compare decreasing feature spans, with zero unparsed messages. The emitter used annotation coordinates; the validator uses emitted feature coordinates and canonical block text. A tie-break-only patch is insufficient. The emitter now canonicalizes the emitted blocks with the same ordering as the validator, and diagnostics name equal-span tie breaks explicitly. This is a future-build code correction, not authorization to replace the frozen object.

| Pending author choice | Cost and effect |
|---|---|
| Retain current frozen corpus and defer recovery | No rebuild. Report all 97 discarded windows and the corrected 13/84 diagnosis as a corpus limitation; explicitly accept retaining these omissions for the current run. |
| Recover ordering failures in a replacement corpus | Rebuild affected species, then merge, validate and freeze a **new artifact**. Conservative full-nine-species budget is approximately 1.5 hours plus merge/validation/refreeze (inherited estimate, not a timing from this audit). Four species have recorded ordering rejects; prove other species unchanged before reusing them. Recompute all membership, split and masking counts and training provenance. |

**Recommendation:** coordinate recovery with whichever #56 remedy the author selects, avoiding repeated rebuilds. Fixing order removes this rejection reason but does **not** prove all 97 windows survive subsequent mask-fraction filters or establish the final RC row count. The rejection log does not retain discarded labels. Do not patch the frozen database in place or advertise “97 recovered” without a replacement build. Record the approved disposition and recipe correction in a dated §12 defect amendment before applying it to the corpus; frozen §§3–9 remain untouched.

#58 does not block the ordering fix or read-only diagnostics: neither changes empty-label tokenization. #56 does not block these implementations, but its masking/population choice affects a replacement build and must be coordinated before producing that artifact. No rebuild was performed here.

## #61 — metric universe (pending definition, completed accounting)

The read-only frozen audit establishes **15,005 / 91,787** nominal test genes ever labelled, **7,316 / 29,153** validation genes, **196,592 / 213,702** training genes, and **431 / 3,430** strict holdouts. The test-tile target population is **54,188** unique genes, comprising 34,606 train-assigned + 4,577 valid-assigned + 15,005 test-assigned genes. Frozen row counts are 198,171 train / 25,910 valid / 48,143 test (RC included). All three label-membership integrity checks pass. These checks do not absolve #56's unlabelled sequence exposure.

There is no single denominator shared by coverage, validation loss, gene/reference accuracy and evidence-support outcomes. The actual trainer averages validation-batch losses; the nominal validation-gene count is not its divisor. Full-GFF evaluation remains as specified in A22/A30; callability-based outcomes retain their frozen definitions. The implementation reports the assignment-by-tile-split matrix, strict counts, per-species length and reference transcript-count distributions, and the Methods insert distinguishes these populations. No metric was rerun or test set redefined.

**Author decision still required:** identify the intended scoring universe for any paper claim called “test performance” (existing tile targets, nominal test-gene intersection, or another explicitly approved reference evaluation). Choosing a new subset is a protocol/scorer decision, not accomplished by dividing by 15,005. Existing full-GFF outcomes must not silently be narrowed to labelled loci. Reporting costs no rebuild; changing only scoring eligibility needs a versioned scorer and rerun, while changing tile/block allocation needs a new corpus/build budget and freeze. Those costs depend on the author's choice.

#58 prevents asserting an exact decoder-token denominator until the empty-target convention is resolved. #56 affects corpus eligibility and the interpretation of unlabelled test genes; all present measurements describe the existing frozen corpus, not a repaired one. Neither blocks factual population/distribution reporting. Descriptive differences are reported without claiming random representativeness.

## #54 — existing decisions implemented, no new decision requested

The author already declined the rice RAP–MSU map (2026-09-02) and retained all Arabidopsis caution masks (2026-09-03). Per-species mapping/flag accounting, the unchanged 571-row / 567-gene Arabidopsis detail table, and Methods wording now appear in `revision/Methods_B5_corpus_accounting.md` and the linked supplementary tables. All nine flag hashes match the freeze. The frozen inputs also contain **one soybean hard-flagged gene**, explicitly reported; “effectively Arabidopsis-only” does not mean exclusively Arabidopsis. No flags, evidence inputs or frozen objects were changed.
