# Track B CPU scoring boundary

Exploratory / MVP only. No training authorization, GPU decoder, job submission,
or empirical results are supplied by these programs.

## Alignment driver

`mvp_align_panel_longreads.py --help` lists required local inputs. Without
`--execute`, it only validates. The explicit `--execute` mode invokes minimap2,
seqkit, samtools, the existing orientation sampler/scorer, and the junction
wrapper locally. It never submits jobs. Run only in an authorized compute
allocation after independent review and real-tool smoke tests.

Canonical roles are **ROOT/evidence/DATASET_ROLES.longread_v1.tsv**, not the
repository copy. The one matching run must be reference Athaliana,
`b1_validation_only`, with `local_fa_md5` matching fresh MD5 of the canonical
actual read file. Only SRR28341943–45 and ERR13994458 are admitted. Missing HiFi
manifest/read inputs fail; nothing is fetched or inferred. Supply reference/GFF
MD5 values from the frozen species manifest. Existing training drivers are not
modified or bypassed.

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
coverage status. It is not a scientific DONE. Failures retain FAILED.json;
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
  JSONL path. Rank-shard duplicates must be resolved upstream by the audited
  finalizer, not silently dropped here.
- `loci`: exactly all 50 `gene_id` values, each with `seeds` keys `"1"/"2"/"3"`,
  each seed with `off`/`on` count objects.
- Each count object has integer `f1_tp`, `f1_predicted`, `f1_reference` from the
  separately run restricted GFFCompare v0.12.6/exact-monoexonic-CDS producer.
  Optionally every object also has `precision_tp`, `predicted`, `recall_tp`,
  `supported`. Partial structural inputs are rejected; unavailable is absence,
  never zero counts. Zero bootstrap denominators are UNASSESSABLE, not zero.

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
The public statistics CLI always emits `decision=null`; `--final-decision`
errors pending the author's HiFi versus ONT-only coverage policy. No approval
switch or default silently chooses that policy.
