# Track B CPU scoring boundary

Exploratory / MVP only. No training authorization, GPU decoder, job submission,
or empirical results are supplied by these programs.

## Raw-GFF producer (new, controller review required)

`mvp_produce_counts.py --input evaluation.json --out NEW_DIRECTORY` derives
reference-dependent counts in the same invocation that verifies finalizer
receipts and computes the existing statistics. Exact decoder transport contract:
`notes/MVP_EVALUATION_INPUT_CONTRACT.md`. Do not feed `counts.json` back into the
legacy adapter and describe its output as independently verified: that adapter
still deliberately labels caller-supplied counts unverified.

The producer binds the existing finalizer five-file manifest, marker, production
metadata, seed/augmentation recipe, checkpoint and source initialization bytes,
consumed code manifest/source bytes, paired summary, and complete valid RN set
from the existing candidate/inventory/actual-DB verifier. All three runs must
share consumed code and initialization source identities. Hashes establish local
consistency, not proof of historical execution, authority, or scientific approval.

FASTA/GFF MD5 authority is the Athaliana row of commit
`92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`,
read from git, not supplied by the caller. Missing anchor or wrong bytes fails.
The reference is restricted by the existing script 24, then checked against
panel gene coordinates/strand; raw predictions are restricted by gene overlap.
A strict wrapper reuses script 07's half-open parser. Explicit exon rows and
ordered gene/transcript/features are required; unsupported features and malformed
geometry fail instead of silently changing denominators. Monoexonic matches
require identical nonempty CDS intervals and strand. Multiexonic matching runs
GFFCompare 0.12.6 and accepts only `=` with independently equal intron chains and
strand. Exact commands, executable bytes/version, comparison output bytes,
TP/FP/FN and per-locus input identities are retained. Empty prediction files are
valid completed zero predictions; missing files are errors.

**Unresolved multiplicity:** duplicate identical intron-chain structures (or
identical monoexonic CDS structures) in either selected prediction/reference
set currently fail closed. No silent deduplication, many-to-one TP inflation,
or new author-approved multiplicity policy is asserted. This may block real
panel scoring and needs controller/author resolution before operational use.
Reference-dependent F1 and the seed-123 baseline are not independent splice P/R.
Read-evidence coverage and withholding remain unavailable/unassessable, with
unchanged author HiFi-policy block and `decision=null`. No training code or
bootstrap math/thresholds changed. The legacy alignment driver below retains
its explicit external-reference-gate limitation; this producer's reference gate
does not retroactively certify any alignment outputs.

## Alignment driver

`mvp_align_panel_longreads.py --help` lists required local inputs. Without
`--execute`, it only performs preparatory schema/geometry/byte checks, not
scientific input validation. The explicit `--execute` mode invokes minimap2,
seqkit, samtools, the existing orientation sampler/scorer, and the junction
wrapper locally. It never submits jobs. Run only in an authorized compute
allocation after independent review and real-tool smoke tests.

Canonical roles are **ROOT/evidence/DATASET_ROLES.longread_v1.tsv**, not the
repository copy. The one matching run must be reference Athaliana,
`b1_validation_only`, with `local_fa_md5` matching fresh MD5 of the canonical
actual read file. Only SRR28341943–45 and ERR13994458 are admitted. Missing HiFi
manifest/read inputs fail; nothing is fetched or inferred. Reference/GFF MD5
values are caller assertions: equality verifies bytes, not their frozen identity.
An **external frozen-reference identity gate is required**, but not implemented
here. No caller approval flag can establish that gate. Both return modes record
`reference_identity_status=UNVERIFIED`, `scientific_input_validation=false`, and
`external_frozen_reference_gate_required=true`. Operational promotion requires
anchored reference and producer receipts, not self-supplied hashes. Existing
training drivers are not modified or bypassed.

Outputs require a fresh `ROOT/evidence/mvp_scoring_20260916/RUN` directory.
No resume/reuse/overwrite is supported. ONT orientation lives under that run's
`orientation/`, never `orientation_audit_v1/`. Its DONE key is unchanged:
`FA_MD5:REF_MD5:GFF_MD5:SCORER_MD5`, plus `audit_md5` and `audit.json`.
The actual unchanged scorer is run at defaults: MAPQ20, 1000 eligible, 0.95.
PASS enables `-uf`; FAIL and UNRESOLVED disable it, as explicitly required by
MVP §2. Missing/stale/malformed artifacts are errors, not UNRESOLVED verdicts.

Every consumer uses `mvp_windows`: 30,720 nt centered on the half-open panel
gene, odd padding base to genomic right. Pass1 adds **10,000 nt per side**,
not 10,240: decimal 10 kb from §2, not a tile geometry constant. Boundary
windows fail rather than shift. Pass1 selects MAPQ >=5 overlapping candidate
reads; pass2 uses the full reference and frozen source-specific flags.
`alignment.json` records commands, input/output MD5, run/source, and pending
coverage status. Without execution the status is
`PREPARATORY_BYTE_CHECKS_ONLY`; after commands it is
`PREPARATORY_COMMANDS_COMPLETED`. Neither is full input validation or a
scientific DONE. Failures retain FAILED.json;
partial output is not reused. Orientation completion alone is not alignment
completion.

## Junctions and denominators

`mvp_junction_table.py` accepts repeated `--alignment SOURCE:uf=BOOL:PATH`
(SOURCE ONT or HiFi), frozen `--panels`, `--out`, `--qc-out`.
It reuses the frozen scorer's SAM parser, read-group rejection, strand
composition, and thresholds (3 ONT or 2 HiFi reads, exact raw coordinates).
The standalone table command is a low-level adapter, not a role/provenance gate;
use driver-produced inputs for operational scoring. Per-run tables come from
the driver; repeat validated run BAMs to produce pooled tables. Counts are
reads, not molecules. Duplicate canonical alignment paths are rejected.

Read support is restricted to centered **windows**. The separately exposed
`mvp_score.predicted_junctions` helper selects normalized transcripts overlapping
the **gene** and retains all their introns; `junction_counts` restricts the
recall denominator to supported introns inside the gene. Flanking supported
introns may contribute to precision but not recall. These normalized-transcript
helpers do not parse raw GFF or perform decoding. Full-chain scoring remains the
existing `59_evidence_support.py` interface; this wrapper does not invent chain
support columns.

## Statistics adapter

`mvp_score.py --input counts.json --panels ... --out statistics.json` consumes:

- `pairs`: exactly keys `"1"`, `"2"`, `"3"`, each a final deduplicated eval-pair
  JSONL path. Repeated resolved paths, symlinks to the same file, and samefile
  aliases (including hardlinks) are rejected. Identical numbers in distinct
  files remain allowed: different filenames/bytes do **not** prove three runs.
  All score API and CLI outputs carry `pair_run_provenance_status=UNVERIFIED`,
  `three_seed_identity_verified=false`, and `three_seed_policy_verified=false`.
  Producer receipts binding seed/run, file hash, and final inventory remain
  unimplemented. Rank-shard duplicates must be resolved upstream by the audited
  finalizer, not silently dropped here.
- `loci`: exactly all 50 `gene_id` values, each with `seeds` keys `"1"/"2"/"3"`,
  each seed with `off`/`on` count objects.
- Each count object has integer `f1_tp`, `f1_predicted`, `f1_reference` from the
  separately run restricted GFFCompare v0.12.6/exact-monoexonic-CDS producer.
  Optionally every object also has `precision_tp`, `predicted`, `recall_tp`,
  `supported`. Partial structural inputs are rejected; unavailable is absence,
  never zero counts. Zero bootstrap denominators are UNASSESSABLE, not zero.
  Recall true positives must be a subset of precision true positives (and thus
  of predicted junctions). For each locus, `f1_reference` and supplied `supported`
  must be frozen across all fractions and seeds. These consistency checks do
  not authenticate the caller-supplied counts.

This is a **count-input adapter**. It does not verify those counts against raw
prediction/reference GFFs, run GFFCompare, or generate predictions. Input hashes
and this provenance limitation are recorded. An upstream GFF/count producer
and end-to-end scientific provenance verification remain required.

Primary is equal-weight evidence-positive tile pure weighted CE off−on,
not pooled token CE or composite loss. Composite is a separate diagnostic.
Bootstrap: 2000 draws, NumPy default_rng(20260914), tile carries all three seeds;
locus bootstrap carries paired fractions/seeds and recomputes pooled ratios.
5th/95th percentiles; F1 in percentage points, P/R in fractions. Outputs include
per-seed/pooled and all50/core20 strata.

`mvp_bootstrap.decide` implements the tested STOP-first §5.3 mathematical rule
with explicit three seed points, withholding assessed/unassessable state,
reversal upper bound >=0.005, and missing-structural downgrade (never automatic
GO). It is a pure rule helper, **not an approval gate**. The precision-clause
interpretation uses delta_P point >=0 and is explicitly pending author review.
The public statistics CLI always emits `decision=null`. `--final-decision` now
requires `--coverage-policy PATH`, a JSON policy validated by
`mvp_score.require_coverage_policy` against the contract frozen in
`PROTOCOL_B1_frozen_v1.md` **A47** (v1.37, 2026-09-21): `policy_id`
`B1-A47-v1.37/MVP-v1`, structural sources `['ONT']` only, thresholds equal to
the frozen `59_evidence_support` values, `unavailable_sources` `['HiFi']`,
`pooling` `within_source`, `per_run_retained` true. A-HiFi carries no structural
authority because ERR13994458 publishes only subreads and FLNC is never produced
here; it is recorded as missing per source, never as zero support. Every field
must be stated explicitly -- nothing is defaulted, no approval switch exists, and
any deviation (a missing field, HiFi as a structural source, a threshold that
contradicts the frozen scorer, cross-source pooling) fails closed. The validated
policy is copied into the result as `coverage_policy`.

`mvp_junction_binding.bind(runs)` stays single-run without a policy. Given the
A47 contract it pools the runs of one source by summing read counts per junction
and re-deriving `supported` through the frozen thresholds; per-run rows are
untouched and cross-source pooling is refused.
