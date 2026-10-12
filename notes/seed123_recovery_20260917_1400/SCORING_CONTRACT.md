# Scoring contract reconciliation, 2026-09-17

## Verdict

**The approved F1 is a pooled count ratio, not macro-F1. The inherited `mvp_score.metric` aggregation does not contradict approval. Do not replace it with an arithmetic mean of per-locus F1.** Loci are the bootstrap sampling units, not equal-weight F1 averaging units. Equal-tile averaging applies to the primary loss, a different estimand.

Source roots used below:
- `P`: `/mnt/f/scratch/develop/Transgenic/transgenic_local` (approval/protocol/notes).
- `W`: `/tmp/mvp-seed123-integration-20260917` (current implementation).

## Authority and scope

1. `P/notes/b6_execution_approval_20260914.md:3-8` records the author's “mvp 승인할께 진행해줘” and approved v1.2 SHA256 `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`. Fresh `sha256sum` of both P and W `revision/protocols/B6_MVP_PILOT_v1.md` exactly matched this hash. Its approval-pending header is therefore stale, not a current veto.
2. `P/notes/b6_mvp_metrics_increment_20260915.md:3,11-17,25` approves composite training, pure weighted-CE paired primary, actual composite diagnostic, target-weight CE denominator, equal-weight evidence-positive tile differences, unchanged 0.005-nat threshold. This changes/clarifies loss, not F1. Current resume repeats this at `P/.atl/resume.md:58`; the archived approval is also explicit at `P/.atl/archive/resume_before_20260916.md:3-6`.
3. `P/notes/b6_parallel_execution_authorization_20260916.md:3-8` authorizes independent long-read scoring/bootstrap and keeps the finalized specification authoritative. It does not authorize changing thresholds.
4. Latest `P/notes/SEED123_ONLY_DECISION_20260917.md:3-16` fixes current development to seed123, defers other seeds, forbids relabeling seed1 receipts, and explicitly forbids treating single-seed development as completed multi-seed replication/final aggregation. No F1 redefinition appears there.

## Exact statistical contract and implementation comparison

| Item | Approved contract | Current source and finding |
|---|---|---|
| Reference F1 | Protocol `:105-108`: GFFCompare v0.12.6 exact intron-chain match, monoexonic exact CDS; manifest reference restricted to panel gene, predictions overlapping gene; “pooled numerator/denominator over the 50 loci”. `:248-250` repeats pooled transcript F1, on-minus-off, delta 2.0 percentage points, core20 alongside all50. | `W/revision/scripts/mvp/mvp_score.py:65-73` sums counts over selected loci/seeds and returns `200 * sum(f1_tp) / (sum(f1_predicted) + sum(f1_reference))`. This matches pooled percent F1. It is not a per-locus macro mean. |
| Locus bootstrap | Protocol `:245-250`, referring to `:242-244`: paired locus resampling for F1 and junction P/R, 2,000 replicates, `default_rng(20260914)`, one-sided 95% bounds from the 5th/95th percentiles. Pairing and seed columns travel with the sampled locus. | `W/src/transgenic/evidence/mvp_bootstrap.py:31-35,161-189` resamples whole locus records and recomputes the statistic. `W/revision/scripts/mvp/mvp_score.py:160-171` uses paired on-minus-off pooled ratios for all50/core20; historical pooled and per-seed groups, or only seed123 in development. No aggregation conflict. |
| Primary loss | Protocol `:238-244`, supplemented by metrics approval above: arithmetic mean of per-tile pure weighted-CE off-minus-on, evidence-positive valid tiles, delta 0.005 nats. Tile draw carries all seed pairs; per-seed and pooled bounds. Not global pooled-token CE. | `W/src/transgenic/evidence/mvp_bootstrap.py:128-158` averages tile/seed delta arrays, with paired tile indices and specified percentiles. `W/revision/scripts/mvp/mvp_score.py:104-107` keeps composite separate. |
| Current seed123 scope | Latest decision `:12-16`: development only; no manufactured three-seed completion/final inference. | `W/revision/scripts/mvp/mvp_score.py:76-104,149-158,164-165` selects `(123,)`, validates run identity/role, emits `DESCRIPTIVE_SINGLE_SEED_NO_FINAL_DECISION`, `decision=None`, and conditional single-training-seed bootstrap scope. This is the appropriate current interpretation. |

A zero pooled denominator is not a score of zero: `metric` returns NaN (`:69-73`); bootstrap rejects nonfinite statistics (`mvp_bootstrap.py:171-180`), and scorer reports `UNASSESSABLE` (`mvp_score.py:168-171`). This is existing fail-closed behavior, not a newly proposed scientific rule.

The recovery review's macro-F1 concern is conditional, not evidence of approved macro-F1: `P/notes/seed123_execution_20260917/EVAL_REVIEW.md:48` explicitly calls it inherited and says “If” the intended requirement means macro-F1. The exact approved protocol resolves that conditional in favor of the existing pooled statistic. Also, that review's phrase “structural F1” should not conflate reference-dependent transcript F1 with independent structural junction P/R.

## Structural source OR, denominators, and missingness

- **Positive support is OR, not AND and not mixed-source read addition.** Protocol `:137-139`: exact `(chrom, intron_start0, intron_end0, tx_strand)` in at least 2 A-HiFi **or** at least 3 A-ONT1 reads within a panel window. One HiFi plus two ONT does not meet either threshold. Per-run provenance/scores and pooled results are required at `:135-136`.
- **Current OR implementation is correct.** `W/revision/scripts/mvp/mvp_junction_table.py:53-56,116-140` pools read identities within each source before applying HiFi2/ONT3 and unions supported keys. Table reader validates the same OR at `:155-181`. Pooling thresholded per-run booleans instead would lose junctions that reach threshold only across runs.
- **Keep denominator scopes distinct.** Protocol `:139-141`: predicted junctions come from decoded transcripts overlapping the gene; precision is supported-predicted/predicted; recall denominator is read-supported junctions inside panel genes; pool counts over loci. `W/revision/scripts/mvp/mvp_score.py:34-62` retains all introns of overlapping predictions, intersects precision with supported junctions in the window, and restricts recall's support set to the gene. Wrong-strand predictions are not silently filtered out.
- **Missing source is not observed zero.** B1 `P/revision/protocols/PROTOCOL_B1_frozen_v1.md:159-167` defines per-source and union callability and prohibits treating uncovered/non-callable additions as negative. These B1 addition-level denominators must not silently replace MVP's explicit junction denominators. A junction table's `n_HiFi=0` alone cannot distinguish an absent HiFi input from an inspected HiFi source with no qualifying reads: the current table initializes both source maps (`mvp_junction_table.py:117`) and fills absent-key counts with zero (`:132-133`). Source/run inventory and availability provenance must accompany it.
- **OR support does not itself certify complete STOP availability.** Approved MVP `:255-273` evaluates STOP first, then GO, then availability safeguards; missing structural scoring or unassessable withholding downgrades prospective STOP to INCONCLUSIVE. `:278-285` forbids treating absent library metadata as proven withholding equivalence. Missing evidence is not a blanket prohibition on descriptive readouts. No reviewed author instruction permits substituting a different accession or treating missing named A-HiFi as fully available merely because ONT yields positives. `P/notes/HIFI_ONT_POLICY_RECHECK_20260917.md:5-22` documents this distinction, but is a reconciliation note rather than new author approval.

## Real implementation gap and minimal next work

**No F1/bootstrap estimator fix is needed.** Preserve the existing formulas, replicate count/seed, and paired units. If adding one small regression later, unequal-sized loci should explicitly distinguish pooled F1 from macro-F1 and cite protocol `:108,248-250`; no full or expensive rerun is justified by this clarification.

The actual unfinished path is independently evidenced structural-count production:
- `W/revision/scripts/mvp/mvp_produce_counts.py:668-686` currently produces reference counts then calls the scorer without structural counts; `:713-718` truthfully reports `ABSENT_NOT_NEGATIVE` / `MISSING_UNASSESSABLE_NOT_ZERO` in development.
- `W/revision/scripts/mvp/mvp_score.py:134-158` preserves structural missingness, omits whole-panel P/R unless all paired structural count records exist, and never gives the development profile a final decision. Preserve this boundary.
- `W/revision/scripts/mvp/mvp_score.py:28-31` still unconditionally raises a broad “coverage policy pending author approval” message for the historical coverage-policy path. This is an implementation placeholder, not proof that source OR or missing-evidence rules lack approval. Do not delete the guard and thereby permit final decisions; the current seed123 path does not need a final-decision policy.

Smallest subsequent implementation: connect validated per-run and pooled independent junction tables plus their exact source/run/reference/panel receipts to the existing `predicted_junctions`/`junction_counts` helpers and counts producer. Retain explicit per-source missingness, preserve full-panel unassessable status when evidence is incomplete, and keep ONT-only descriptive observations separate from certification of complete structural availability. Do not add macro-F1, invent ONT-only STOP permission, acquire/substitute sources, or run extra seeds as part of this fix. Final three-seed decision publication remains outside current seed123 authorization.

## Verification and boundaries

Read-only document/source inspection and the two small protocol SHA256 checks only; no tests, training, decoding, remote actions, publication, or code edits. Only this report was intentionally written. Required graft queries automatically reported local graph refreshes; these were tool side effects, not source edits. Graft reported approximately 1,160,315 tokens saved across map/ask/skeleton calls.
