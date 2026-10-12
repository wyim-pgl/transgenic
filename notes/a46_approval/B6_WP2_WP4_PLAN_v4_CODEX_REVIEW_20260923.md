**Verdict: APPROVE_WITH_FIXES.** Both v3 HIGH findings are resolved at the design level. No new CRITICAL/HIGH found; one manifest-contract clarification remains.

Reviewed v4 SHA `702c10696fc2…` against the frozen protocol and code at `e35a5bd`.

**A. v3 HIGH 1 — RESOLVED.**

The leakage citations support the proposed geometry:

- [leakage.py:353](/tmp/b6-wp1-20260922/src/transgenic/evidence/leakage.py:353) replays historical per-gene draws; lines 370–381 accumulate maxima from kept **train/valid forward tiles**.
- [leakage.py:418](/tmp/b6-wp1-20260922/src/transgenic/evidence/leakage.py:418) pads the genome-level component by its maximum recorded member flank, falling back to 150, and clips to chromosome bounds.
- Historical per-gene sampling is specifically at [build_b5.py:600](/tmp/b6-wp1-20260922/src/transgenic/datasets/build_b5.py:600); v4’s 597–599 citation stops just before it.

For each **shared cause and applicable scope**, repair interval = existing exclusion interval ∩ tile. Therefore a placement touching newly masked DNA necessarily touches an existing exclusion interval. [Molecule closure](/tmp/b6-wp1-20260922/src/transgenic/evidence/leakage.py:495) propagates that hit molecule-wide. The fresh-draw counterexample is removed.

This guarantee correctly excludes the declared differences: A30-only repair can mask DNA whose evidence remains admitted; A48 can exclude an unmasked edge-partial span. Thus “never masks admitted DNA” is valid only for the shared causes.

[§2](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/B6_WP2_WP4_PLAN_v4_20260923.md:32) now specifies the single producer, C0/freeze binding, equality and molecule-consequence tests, historical grandfathering, and rebuilding downstream artefacts if the author chooses fresh draws. D1 remains an explicit author decision.

**B. v3 HIGH 2 — RESOLVED.**

[§8](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/B6_WP2_WP4_PLAN_v4_20260923.md:198) selects required keys from verified provenance policy, separates `training` from `eval_only`, and requires consumers to declare accepted families. Evaluation no longer requires corpus bindings. Training rejects evaluation-only artefacts, including those carrying an extra corpus key. The specified eval-C0 → cache → context → batch test with **no corpus file present** directly covers the former contradiction.

**C. New CRITICAL/HIGH — NONE. One MEDIUM clarification:**

[§2.5 and §8.3](/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/a46_approval/B6_WP2_WP4_PLAN_v4_20260923.md:210) need an explicit definition of “projects exactly.” The legacy [interval hash](/tmp/b6-wp1-20260922/revision/scripts/evidence/b6_build_c0.py:1829) includes `gene_id` and `tile_rn`, and covers test blocks, tile-specific masks and edge-partial rows as well as genome-level components. The proposed component-only manifest cannot directly reproduce that entire row set.

Specify that revalidation recomputes and verifies the **complete legacy interval hash**, then compares the manifest’s shared-cause component projection against the corresponding recomputed intervals. Represent A30-only repair rows explicitly without adding them to C0 exclusions. Hash all versioned manifest fields canonically. This makes the fail-closed migration implementable without changing the approved geometry.

No files were modified; no tests, training or jobs were run.