# Job 6165628 independent result review — 2026-09-17 01:05 PDT

## Decision and scope

**ACCEPT: actual 50-locus reference-mapping/geometry and small-file publication evidence. PARTIAL: complete array-content acceptance remains unverified.** This is not total pipeline completion, scientific acceptance, decoder readiness, training authorization, or a clean scheduler exit across all steps.

Observed only job6165628, first at00:42:51, then00:45:34 and at120–121s intervals until terminal at01:01:39. Last exact-job accounting at01:05:20. Observation was below4200s; no unrelated jobs inspected. No remote writes, submissions, cancellations, builder runs, full database scans, large-array reads/hashes or old6165559 output reuse.

## Scheduler versus payload

[terminal_accounting.txt](terminal_accounting.txt): allocation and batch **COMPLETED0:0**,00:41:03–00:59:56,18m53s,4CPU/16Gn,cpu-50; batch MaxRSS787524K. **extern OUT_OF_MEMORY0:125**,MaxRSS1480K, remains a separate unresolved failure. It must not be relabeled clean or inferred to invalidate/validate payload bytes by itself. Both copied payload logs are empty. Output inventory contains no FAILED, REFERENCE_ONLY or pending publication files.

## Independently executed checks

[validate.py](validate.py) executed successfully, with [validation.json](validation.json) and [stdout](validation.stdout.txt). These checks use copied small result files and local reviewed package bytes, not marker presence alone.

- Frozen panel TSV SHA79367154… rehashed. Exactly20mvp_core+30mvp; ordered50-ID equality across TSV, mapping rows, reference report, manifest and counts. Fifty distinct internal IDs map bijectively to50 distinct raw TAIR10 gene IDs, all Athaliana, with matching chromosome,0-based half-open start/end and strand. Every30,720nt window independently recomputed and bounded by reported chromosome length.
- Copied actual restricted GFF88,875B independently hashed to33e98f9e… and parsed using separate stdlib validation:50original-ID genes/52transcripts; raw gene coordinates/strand/name agree with the mapping and TSV, transcript containment and exon/CDS containment validated. Producer strict-parser receipt reports50/50compatible; independent geometry checks agree. No heuristic remapping or GFF edits.
- Duplicate structure observation independently reconstructed from restricted GFF under the existing intron-chain/mono-CDS rule: **0ambiguous loci**, matching report; transcript counts and noncoding-mono lists agree. This observes this panel only, does not change deduplication/F1 policy or grant scientific approval.
- Remote copied transport bytes equal reviewed transport11d0550b…; source inventory5570f323… equals local frozen package. Source identity equals reviewed SOURCE_VERIFICATION receipt, with100files, commit9603e20…,86-file componentad3e5a7a…, producer6069fb5f…. Manifest/mapping producer and source input identities agree with local package files. Git authority receipt equals reviewed transport authority: original92d8020… commit,3ca7a3b9… manifest and five objects. No authority reconstruction.
- Runtime receipt reports Python3.10.20, DuckDB1.5.5, NumPy1.26.4, Torch1.13.1+cpu, empty CUDA visibility; SIF9e2be34c…. This is recorded CPU utility evidence, not independent current SIF rehash or GPU decoder certification.
- Final marker job ID is6165628. Marker→counts→manifest/reference report and panel COMPLETE→manifest SHA links all independently recomputed. Request identity matches manifest. Reference identity is identical across its receipt, report, mapping authority and panel authority. All50count rows equal manifest projections.
- Fresh namespace inventory shows50regular NPY files, each491648B, plus50FASTA/50JSONL; all sizes match manifest and mtimes fall after this job start. Metadata reportsfloat16[30720,8]. No input path points to6165559. Fresh publication provenance is supported by reviewed exclusive-output execution and new job/source/request pins; file stats are not proof of array content.

## Raw observations, not recomputed array measurements

Counts report3443selected molecules **summed over windows**, range3–398per window; not a unique genome-wide molecule count. Zero-biological-support loci0/50. Biological channel nonzero-position sums:[34789,52639,71107,138,140,193,204]; availability sum1536000. These are validated receipt/manifest agreement, not independent NPY/provenance recounts. All remain excluded_heldout,training_eligible=false,RN=null,scientific_verdict=null.

Publication hashes:
- OBSERVATIONS_COMPLETE: `acfe378a02c3a499c301a535f73e80db8072bcea69dade025272b7ca4002c93f`
- counts: `e245caf8741ef607db9f89271412340e5af72030d36acf5cc803e21a68e1bf1d`
- manifest: `7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e`
- compatibility: `9ac63bf3d88c7161e82a73eff89048fd4faa4fbf765dff98f2f5cd881e46498d`

## Exact remaining checks / controller blockers

1. **Array-content acceptance:** under an approved compute allocation, read-only validate all50NPY/FASTA/JSONL hashes against this manifest; actual NPY dtype/shape/finiteness, availability and seven-channel counts, FASTA sequence hashes/orientation, and provenance row counts. No active approved allocation remained at observation completion; this observer cannot submit one. None of these hashes/content checks was run on login. Preserve this publication; do not rerun builder or reuse6165559.
2. **Source authority freshness distinction:** corpus/whole-reference hashes, mutation/WAL guards and source rechecks are execution-produced receipts from reviewed compute code. This review verifies their consistency and the small restricted GFF, not a second23.5GBcorpus query or independent full reference/database rehash. If a downstream gate demands such a recheck, run it read-only in approved compute, not login.
3. **externOOM:** unresolved scheduler-step anomaly; payload success does not close it. Any clean-pilot/final gate requiring all steps clean remains blocked pending scoped diagnosis/decision.
4. **Downstream evaluation:** same new producer pin must be accepted by the decoder consumer; an old b660 producer expectation is stale. Decoder package/SIF/GPU preflight, valid training checkpoints, inference/scoring, independent long-read evaluation and scientific/reference-policy approvals are not established here. No automatic GPU/CPU submission is authorized by this review.

Evidence writes were confined intentionally to this panel_result directory; mandatory graft context commands reported automatic graph refreshes as a tool side effect (no source edits or manual graph writes). Advisor failed client/model-version compatibility and supplies no endorsement. Graft-reported context savings approximately1,031,858tokens.
