**Verdict: REJECT — two HIGH contract gaps remain.**

Reviewed the [v3 plan](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/B6_WP2_WP4_PLAN_v3_20260923.md), last-read SHA `edcb798b09b9…`, against the frozen specifications and code at `e35a5bd`. The file changed during review: its actual §2.4 now specifies **per-component repair flanks**, preserving historical A40 per-gene masks.

**A. Prior findings**

| Finding / prior PARTIALLY item | Status | V3 assessment |
|---|---|---|
| **B1 / A4** | **PARTIALLY** | §§7.4, 7.9 provide evaluation contexts and maize ingestion; §6 isolates completeness populations. However, §8 still requires corpus provenance incompatible with that evaluation path. |
| **B2 / A2** | **RESOLVED** | §6 supplies BAM/reference/read-end measurements, bound CAGE/chain inputs, population-qualified keys and mandatory population lookups. |
| **B3 / A8** | **RESOLVED** | §§7.3–7.4 resolve recipient uniforms against donor availability, compare uniforms rather than selected identities, and reject singleton donor pools. |
| **B4 / A1** | **RESOLVED** | §5 restores both protein CDS classes and specifies eligible, cross-run-deduplicated junction counts with independent scope-matched validation. |
| **B5 / A5** | **RESOLVED** | §8 explicitly rehashes consumed zero lists/QC, checks upstream bindings, and rejects WAL-backed snapshots. The separate evaluation-contract conflict is below. |
| **B6 / A3** | **PARTIALLY** | §2 fixes edge enumeration and fixtures, and adopts component draws. Historical-mask reconciliation and propagation into molecule exclusion remain incomplete. |
| **B7 / A10** | **RESOLVED** | §§6, 7.2, 10 separate pilot training from C5 readiness and specify slot encodings and the final source tie-break. |
| **A11** | **PARTIALLY** | §§7.2, 11 settle ordering and most decision distinctions; D1’s mask/exclusion reconciliation remains unresolved. |

Kimi’s downstream boundary-deferral and per-species stop-convention requests are addressed in §§5 and 4.

**B. New or newly exposed HIGH problems**

**1. HIGH — The new repair geometry is not explicitly bound to C0’s molecule-wide exclusions (§§2, 8, 10).**

The textual reading is unambiguous: [A33.3](/mnt/f/scratch/develop/Transgenic/transgenic_local/revision/protocols/PROTOCOL_B1_frozen_v1.md:471) specifies one sampled flank per component; A46 rule 4(a) reinforces that reading. The historical implementation differs: [A40’s builder](/tmp/b6-wp1-20260922/src/transgenic/datasets/build_b5.py:598) samples per gene, and [WP1](/tmp/b6-wp1-20260922/src/transgenic/evidence/leakage.py:353) replays those draws to verify corpus bytes. Its genome-level exclusion uses the [maximum observed component-member flank, or 150 when none exists](/tmp/b6-wp1-20260922/src/transgenic/evidence/leakage.py:422).

Thus **the reviewers were correct about the frozen text; v3 is correct about the historical code**. Existing code establishes a discrepancy, not permission to reinterpret “per component.”

V3’s new `:b6edge:` draw can exceed the historical exclusion flank. Exporting `component_flank()` and comparing two returned values does not establish that published C0 actually used the resulting intervals. §8 binds C0 to A40, but supplies no required repair-policy/interval binding or rebuild gate. A molecule touching a newly excluded flank could therefore remain admitted elsewhere; zeroing its local counts on N would not satisfy rule 4(b).

Required plan changes:

- Record an author-approved reconciliation: correct B6’s component policy or explicitly grandfather the historical per-gene geometry. Do not describe the hybrid as unqualified literal A33.3 compliance.
- Define one versioned interval specification/manifest consumed by both repair and exclusion, with component identity, coordinates, flank, cause and scope.
- Bind that identity into C0 provenance and require affected molecule/query exclusions, derived HSPs, merged evidence, zero lists and caches to be rebuilt or revalidated before consumption.
- Test **actual interval equality and molecule-wide consequences**, including a new flank larger than the historical one.

“Identical” must apply to the **shared masking causes**, not the entire DNA-mask and exclusion sets: A30-only masks intentionally do not create rule-4(a) exclusions, while A48 excludes every edge-partial span, including the unmasked negative-control `gL`. Preserve these explicit differences.

**2. HIGH — The common verifier rejects the new corpus-free evaluation path (§§7.4, 8).**

§7.4 creates maize C0 under `eval_only_no_leakage_model`, without a corpus. But §8 says every missing key fails and requires, for **every `(species, source)`**, `inputs.corpus.sha256 == freeze.base_sha256`, plus the B6 corpus hash.

These contracts cannot both hold. Following §8 literally rejects the maize evaluation path, including its intended end-to-end integration.

Define explicit required-key schemas selected by verified artifact family/policy: training requires the A40/B6 bindings; evaluation-only requires reference, roles, panel/window and appropriate evidence bindings. Training must continue rejecting evaluation-only artifacts. Add an evaluation-C0 → cache → batch acceptance test requiring no corpus. This must be a typed contract, not an “if key exists” bypass.

**C. Remaining author decisions**

| Decision | Genuine? Default assessment |
|---|---|
| **D1** | **Partly.** Leak/hard edge seeds plus closure are defensible. Flank reconciliation is a specification discrepancy requiring explicit resolution, not an ordinary implementation preference. |
| **D4** | **Yes.** Static, population-specific zero lists are sound and now properly verified. |
| **D6** | **Yes.** Fixed within-tier/split donors, donor-resolved withholding and singleton rejection are sound. |
| **D7** | **Partly.** Query-level thinning with organism-level support is defensible if explicitly frozen; withholding is now distinguished correctly. |
| **D8** | **Yes.** Source-separated merging is defensible; the tie-break closes the ordering gap. |
| **D10** | **Only the protein adapter.** `not_applicable → 1.0` is reasonable; other multipliers are already fixed. |
| **D12** | **Yes.** The clustering default is reasonable provisionally. Freeze the complete mapping/tag-extraction and peak-calling recipe for the designated assays before WP2c. |

D5, D9 and D11 are correctly moved to settled.

No files were modified; no training or jobs were run.