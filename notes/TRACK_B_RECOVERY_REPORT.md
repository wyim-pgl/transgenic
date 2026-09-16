# Track B recovery report

## Identity and scope

- Worktree: `/tmp/b6-p2-scoring-1535144`; branch `b6-p2-scoring-1535144`.
- Verified starting HEAD: `f847cb80c7750326f0781f81481891ab0c9ea309`.
- Implementation commit: **`eefe5b505b8c3315dbe432ecf882b6e85296576d`**.
- Preserved and completed the three supplied partial files; no worktree creation,
  reset, clean, main edits, merge, push, SSH, remote writes, jobs, or subagents.
- Graft queried in this worktree before source exploration and rebuilt afterward.
  Existing `.gitignore` modification and `.ignore` remain uncommitted; graph and
  temporary dependencies are not committed.

## Implemented boundaries

1. Shared centered 30,720-nt window geometry; pass1 window ± **10,000** nt,
   explicitly distinguished from 10,240 tile geometry. Separate window support,
   gene-overlap prediction selection, and gene-contained recall denominator.
   Public panel paths are checked against
   `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`.
2. Local scoring-only alignment driver, canonical actual-read MD5 and roles gate
   (`ROOT/evidence/DATASET_ROLES.longread_v1.tsv`), no training-role bypass.
   Fresh disjoint output namespace, explicit `--execute`, frozen pass2 commands,
   full reference, orientation sample/scorer reuse without modifying legacy code.
   New-namespace audit retains legacy four-MD5 DONE key plus audit MD5.
   Real unchanged scorer exercised at a non-default root on a tiny SAM/GFF;
   UNRESOLVED correctly means no `-uf`, not an invented orientation PASS.
3. Junction-table wrapper reuses frozen SAM/strand/read-rejection logic and
   thresholds; exact coordinates; ONT3/HiFi2 read thresholds, per-input QC,
   rejects duplicate canonical alignment inputs and inconsistent table rows.
4. Pure CE paired-record validation, equal evidence-positive tile off−on primary,
   composite separately reported, exact three-seed inventory. Deterministic
   2000-replicate rng20260914 paired tile/locus bootstraps; pooled count ratios,
   all50/core20 and per-seed readouts. Missing structural scoring is absent,
   never negative; partially missing structural inputs are rejected.
5. Pure STOP-first §5.3 decision helper, withholding reversal/unassessable and
   missing-structural safeguards. Public final-decision publication **fails
   closed pending author coverage policy**; count statistics output has null
   decision and pending-policy status. No ONT-only/HiFi approval is inferred.

Usage/schema/limitations: `revision/scripts/mvp/SCORING.md`.

## Actual verification

Baseline attempts before implementation:
- PDB Python could import Torch 2.6.0+cu124 but had no pytest.
- Default Python had pytest/NumPy but no Torch; metrics collection failed.
- Installed pytest only into `/tmp/b6-trackb-testdeps` (not repository/system).

Final targeted CPU command:

```sh
cd /tmp/b6-p2-scoring-1535144
PYTHONPATH=/tmp/b6-trackb-testdeps:src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  tests/test_mvp_metrics.py tests/test_mvp_scoring.py \
  tests/test_mvp_scoring_driver.py \
  revision/scripts/tests/test_evidence_support.py -q
```

**86 passed, 21 subtests passed** (3.20 s at final behavioral verification).
`compileall` on the five task modules and `git diff --check` also succeeded.
This is a targeted regression run, **not the full repository suite**.

Tests exercised real CPU functions, SAM parsing, the unchanged orientation
scorer subprocess, and the new score CLI on the actual frozen 50-locus panel
with explicitly synthetic counts. No external aligner behavior was mocked into
an empirical success claim. Existing partial behavior was characterized;
new missing-function and validation tests were observed failing before fixes
(duplicate input, seed override, boolean statistic, forged support, seed alias,
namespace redirection). The existing metrics and frozen evidence tests stayed
passing in the final run.

## Remaining work / blockers (not claimed complete)

- **Not an end-to-end biological scoring pipeline yet.** The score CLI consumes
  externally supplied per-locus GFFCompare/exact-CDS counts. Raw GFF restriction,
  GFFCompare execution/count production, and independent verification of those
  counts against raw GFFs remain upstream work. Input hashes are recorded, but
  hashing caller-supplied counts does not establish their scientific provenance.
- No GPU/model decoding or 300 predictions were produced. Normalized-transcript
  junction helpers are not a raw-GFF decoder. Integrate decoder window use via
  the shared module when the decoder is implemented.
- The full minimap2/seqkit/samtools `--execute` path has **not** been run with
  real binaries/data. It needs reviewed compute-allocation smoke testing,
  including command failure behavior and full provenance inspection, before
  operational readiness. No production alignment or coverage facts were measured.
- A-HiFi availability and whether ONT-only satisfies §5.3 require author policy;
  no approval knob was invented. Missing required canonical rows/read MD5s fail.
- Precision contradiction uses delta_P **point estimate** >=0, explicitly
  recorded as an interpretation pending author review. Final decisions remain
  blocked rather than silently treating that interpretation as approved.
- Withholding evidence/run or substantiated equivalence argument still needs
  real provenance. The pure decision helper accepts already-assessed inputs;
  it is not an evidentiary approval mechanism. Complete-chain support is left
  to the unchanged frozen evidence scorer, not fabricated in this wrapper.
- Only fresh per-run alignment directories are supported; interrupted output
  is retained and refused on reuse, not resumed or overwritten.

No empirical efficacy, biological support rate, STOP/GO outcome, training
readiness, author approval, or external-tool certification is claimed.
Controller independent review remains required.
