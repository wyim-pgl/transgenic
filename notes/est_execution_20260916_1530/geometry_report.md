# EST phase geometry diagnostic — 2026-09-16

## Scope and revision

- Isolated worktree: `/tmp/pi-agent-7cd2efcf-bbfd-404-ba02ce92`.
- Branch: `est-phase-geometry-20260916-1541`, branched directly from `5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35`.
- Source/tests commit: `46ed7150035d955ccea7269c0148e798770204aa`. This report is a subsequent documentation-only commit.
- Read the full main working-copy `notes/EST_PHASE_DESIGN_20260916_1530.md` and authoritative working-copy workplan §2.6. Neither main nor the C worktree was modified. No SSH, jobs, subagents, merge or push.
- Graph built and queried before implementation, then refreshed. The original graph pointed to `revision/scripts/evidence/est_phase_project.py`; refreshed spans: mapper 26–96, projector 106–185, multi-observation refusal 188–196, unchanged public production refusal 224–225.

## Implemented contract

`query_reference_runs(placement)` validates a full original screened sequence and its uppercase SHA256, member/source-file SHA256/source interval identity, SAM FLAG, reference-left POS, named contig and supplied contig length, every CIGAR operation, terminal H/S structure, full query length including H, and optional BAM-oriented SEQ length and sequence identity. Reverse SEQ is checked against the reverse complement before terminal H removal. Missing SEQ is allowed only with the full screened sequence. The caller must obtain contig names/lengths from a validated reference dictionary; this library does not read BAM headers or verify reference files.

The mapper emits affine M/=/X runs and compact unmapped spans retaining I/D/N/S/H/P reasons. H advances only the full-query cursor; P advances neither cursor. D/N are never phase-labelled.

`project_codons(hsp, codon_events, placement)` takes labels supplied by a separately validated producer. Each event is a compact labelled run `{q0, length, direction, c0}` in original screened-query coordinates. Ordinals advance by one in coding direction. Split runs at producer discontinuities; single-base runs can represent noncontiguous codon tuples. Event indices survive projection. The input must declare `c0_first_codon_base` and name a producer-contract SHA256. This SHA is a provenance reference, **not an approval boolean or verification of producer acceptance**. No candidate HSP decomposition, traceback interpretation, frame extrapolation or acceptance threshold is called.

The diagnostic manifest always declares `training_eligible: false` and six **local** channels `[+0,+1,+2,-0,-1,-2]`. Segments carry start ordinal and +/-1 step, not a constant phase over an interval. Original HSP endpoints/frame and placement geometry are retained. Source-accession offset is diagnostic only and never shifts BAM coordinates. Strand is FLAG16 XOR HSP direction. Diagnostic anchor is `(g-c) mod3` on plus and `(g+c) mod3` on minus. N changes this anchor but never advances the supplied ordinal.

Self overlap (including repeated equal labels), ts/HSP disagreement, and nonprimary emission return explicit unresolved-policy reasons and no segments. `project_observations` refuses **all** multi-HSP inputs, including apparently compatible ones; it does not pretend to implement conflict detection, rejection scope or deduplication. These are inability-to-produce diagnostics, not scientific whole-HSP rejection decisions. Missing ts is reported without erasing protein-derived direction.

No production raster or CLI was added. Existing `project()` refusal, FLAG×ts primitive and scalar A/B comparison remain unchanged. No MVP channels or training files changed. Standard library only in implementation.

## CPU verification

Initial TDD run: 38 failing tests due to the absent geometry API, before implementation. The first implementation passed all 38. Additional boundary/reflection/identity tests brought geometry coverage to 51 cases. A second red/green cycle verified raw HSP/placement coordinate retention (12 failures before adding metadata).

Exact final command:

```sh
python -m pytest revision/scripts/tests/test_est_phase_project.py revision/scripts/tests/test_track_c_recovery.py revision/scripts/tests/test_track_c_audit_review.py revision/scripts/tests/test_est_blastx.py revision/scripts/tests/test_est_blastx_fixture.py -q
```

Result: **170 passed in 6.14s**. `git diff --check` passed. Graph refreshed after changes. Advisor consultation was attempted but unavailable (provider requires newer Claude Code); no advisor review is claimed.

Coverage:

- F1–F3: hand-computed exon labels, split codon tuples and anchor changes for introns 3/4/5 nt; all four FLAG/HSP-sign combinations. No double intron correction.
- F4: I/D/P, M/=/X, hard clipping, reverse hard-clipped sequence identity, actual minus-strand reflection of I/D/P examples.
- F7: one-base support, all-clipped events, half-open endpoint and tile intersection expectations, original-source offset kept separate.
- F8: hand-computed reflected label locations, actual reverse placement/CIGAR projection, involution and genomic translation invariance. This checks ordinal/strand geometry, **not float16 raster byte replay**, since rasterization is deliberately absent.
- Producer-supplied discontinuity leaves omitted query bases unpainted; no signed-frame extrapolation across the gap.
- Invalid clipping placement/order, malformed/zero/unknown CIGAR operations, mismatched full/SEQ lengths or identity, invalid positions, reference overrun, unmapped FLAG, invalid ts, mismatched HSP identity/direction/convention/event bounds; ambiguity refusals; unchanged production refusal.
- 300,000-base aligned event emits one segment. Memory scales with CIGAR/event runs and intersections, not reference length or per-base Python objects. Sequence hashing/checking remains linear in full query length. Projection uses binary search into mapped runs; events are sorted for overlap checks.

## Exact open choices for the next reviewed policy task

1. Producer contract and authentic receipt/schema verification: real event/codon IDs, subject residue correspondence, frameshift reuse/omission semantics, endpoint/frame-magnitude acceptance and minimum HSP-length semantics remain upstream. Current API checks signed direction and bounds, not DIAMOND acceptance. Query-insertion support must be chosen explicitly.
2. Self-reused nucleotide: reject entire HSP, mask participating events/bases, or another reviewed rule? Equal-label reuse is also unresolved here.
3. Multi-HSP inconsistency: participating whole HSPs versus conflict-only masking; compatible-overlap deduplication; member versus clone/molecule scope; cross-member disagreement; input-order independence and unrelated survivors.
4. ts/HSP contradiction: quarantine the HSP-placement observation versus another disposition. Primary/supplementary/secondary emission eligibility and MAPQ policy also remain unselected.
5. Route-A adapter: retain miniprot phase, strand, protein sequence identity, Target/traceback and split-codon correspondence; verify real plus/minus partial/split-CDS GFF conventions before interpreting phase. Existing coverage TSV is insufficient.
6. A/B neutralization: pair-local contribution removal versus final-cell veto in the presence of independent support (F6 discriminator). Required base/strand/protein phase comparison versus stronger same-residue/codon validation must remain distinct; missing A is not agreement.
7. Molecule aggregation: A11 clone/version/mate/split closure, cross-source union, phase count scaling and any explicit cap. No molecule-count/log1p/float16 implementation or training-raster claim yet.
8. Real input authenticity: full provenance receipts, reference dictionary/aliases, duplicate identity inventory, exclusion placement audit/leakage closure, source/library/genotype/role metadata, nine-species input inventory and paired-arm accounting are not verified by this pure transform.

No scientific conflict policy, route-A convention, molecule aggregation policy or training eligibility has been selected implicitly by this implementation.
