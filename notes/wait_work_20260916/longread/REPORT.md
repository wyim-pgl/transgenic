# Task 3 — independent long-read preparation and bounded orientation pilot

## Status / controller handoff

Prepared locally; **no remote write, job submission, full-read hash, raw-read scan, GPU import, fetch, merge, push, source edit, or child agent** was performed. `b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6` source remains unchanged. `frozen/` is a Git-object export for inspection/use, verified by `source_pins.json`, not a new implementation branch.

**Next executable scope:** controller-reviewed `orientation_pilot.sbatch`, one SRR28341943 orientation sample, 4 CPU / 8 GiB / 30 minutes. This is not the full ONT aligner and cannot establish structural availability. Review `gate.py` before deployment. No array or retry. Full alignment/postflight and scoring integration require a subsequent review; the alignment validator is preparatory and has not been exercised on actual aligner outputs.

## Actual readonly observations

`readonly_actual_inputs.txt` records metadata observed 2026-09-16 21:21 PDT on pronghorn (use its exact timestamp). Canonical authority is **ROOT/evidence/DATASET_ROLES.longread_v1.tsv**, SHA256 `288390b0e75a07845159f93f6a07c0033e4980b9dfe862dc2d982cac1704e348` in the probe, not a repo copy. ROOT is `/data/gpfs/assoc/pgl/data/Transgenic`.

All three FLIC runs are `Athaliana/reference/b1_validation_only`, dataset `ont/Athaliana_FLIC_PRJNA1087576`. Consumed path is exactly `$ROOT/evidence/$dataset/$run.fa.gz`:

| Run | Compressed bytes (stat) | Expected records (roles; not recounted) | Local FASTA MD5 (roles + small .md5 receipt; not freshly hashed) |
|---|---:|---:|---|
| SRR28341943 | 1,915,883,586 | 6,952,504 | d802a86ae1673909ee21a0cd663c99ac |
| SRR28341944 | 2,980,861,262 | 10,389,060 | 7a9bca96833a99e0cebba8a1995f575e |
| SRR28341945 | 2,769,421,478 | 9,459,558 | 414412803d96e3e58c9fc22172e0bbd4 |

Download `.DONE` files exist but are **empty historical markers**; `.source.md5` is absent. Do not describe these as modern download provenance attestations. Roles' `source_checksum` is ENA FASTQ MD5, **not** the converted local `.fa.gz` MD5. The small MD5 receipts agree with roles; only fresh compute hashing verifies consumed FASTA bytes. The training-oriented `longread_runs_v1.tsv` has no matching FLIC entries; absence there does not exclude these validation runs.

ERR13994458 has no canonical role row in the queried authority. Shallow `pacbio/` inventory has no PRJEB77203 dataset; explicit pacbio/hifi candidate directories are absent. This corroborates prior discovery, not an exhaustive filesystem absence proof. **No HiFi acquisition or ONT-only STOP policy selected.** Other Athaliana PacBio datasets are not substitutes.

Tools actually executed read-only with version flags from `RNASEQ_bch709/bin`: minimap2 **2.28-r1209**, samtools **1.3.1**, seqkit **2.13.0**, all exit 0. Remote login has no `python3` on PATH; use the explicit sylvan Python in the sbatch. Versions are observations, not independent certification against every frozen protocol requirement. First attempted probe failed (missing python3, then quoting typo); final probes use a checked local stdin script. No tools were installed.

## Reference authority coordination with Task 2

Same authority as b660 `mvp_produce_counts.frozen_reference`:

- Git commit `92d8020ee79eeedd7f1d50167b1c8dc94700fb16`.
- Object `data/manifests/b5_species_v1.tsv`.
- Exact blob SHA256 `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`.
- `authority.tsv` was exported with `git show <commit>:<path>` locally; gate hardcodes the blob digest, selects the unique Athaliana row, and hashes actual consumed FASTA/GFF on compute against that row's MD5s. Caller-supplied arbitrary reference MD5s cannot pass.
- Uses Task 2's observed paths: `genomes/Athaliana_167_TAIR10.fa`, `genomes/Athaliana_167_TAIR10.gene.clean.gff3` (121,183,059 / 60,503,119 bytes respectively).
- Task 2 exporter may replace delivery of this identical blob and supply its Git-object proof. Require exact blob agreement, not merely matching filenames or a `verified:true` field. Common returned contract here: `reference_gate={authority,manifest_sha256,files:{fasta:{path,size,sha256,md5},gff:{...}}}`. Controller can adapt Task 2 field names without modifying either authority.

This is a **new external gate**, requiring independent review; it does not rewrite the frozen scoring driver's truthful `reference_identity_status=UNVERIFIED` / `PREPARATORY_COMMANDS_COMPLETED` fields. The separate outer gate records its own checked reference identity. No existing training role filter is changed.

## Exact scientific connection contract

Use b660 definitions, not new statistics:

1. `mvp_windows.py:139–190`: gene `[start1−1,fin1)`; `pad_left=(30720−gene_length)//2`, centered window `[gene_start0−pad_left, +30720)`. Odd pad goes genomic right. No silent window edge clamp. Pass-1 prefilter expands this window by 10,000 on each side (left floor 0), MAPQ >=5. This is a candidate selector only.
2. `mvp_align_panel_longreads.py:37–97,133–213`: canonical reads/role → deterministic uniform 10,000-record sample of the **whole** run using `68_sample_fasta.py` → full-reference no-uf splice SAM → frozen `67_orientation_audit.py` → `DONE key=FA_MD5:REF_MD5:GFF_MD5:SCORER_MD5`, run, audit JSON MD5. Scorer MD5 means **67_orientation_audit.py**. Actual `read_sense/valid_for_orientation`, minimum 1000 and threshold .95 determine PASS/FAIL/UNRESOLVED; only PASS enables uf. FAIL/UNRESOLVED are not alignment absence.
3. Whole canonical run pass1 → retained IDs → `seqkit grep` from that same run → pass2 against **full reference**, ONT `-ax splice [-uf only if PASS] -k14 -G 200000 --secondary=no --cs=long` → sorted/indexed BAM. Sample-only alignment never substitutes for full pass1.
4. `mvp_junction_table.py:64–140,183–221` reuses `59_evidence_support.py`: whole-read-group rejection, primary/MAPQ20, supplementary/chimeric rejection and FLAG × ts strand. Read counts, not EST molecules. Threshold ONT3 / HiFi2. Pool from all separately bound BAMs via repeated `--alignment`, not by summing thresholded per-run tables. Preserve per-run scores/QC alongside pooled.
5. `mvp_score.py:34–60`: predictions selected by **transcript overlaps panel gene**, retaining all their introns (wrong-strand predictions remain penalized). Precision support is window-contained; recall denominator support is **gene-contained**. Introns 0-based inclusive. Do not confuse 30,720 window or prefilter extent with gene denominator.
6. Full decoder inventory: 50 loci × 3 seeds × off/on =300 predictions, plus 50 baseline =350; bind each GFF to finalizer/decoder receipt and actual checkpoint source, shared panel/reference. Use existing normalized transcript reader and `predicted_junctions` / `junction_counts`; preserve established GFFCompare reference counts separately.
7. Build counts payload with actual table/GFF identities, then invoke unchanged `mvp_score.score` / `mvp_bootstrap` for 2000 paired-locus/tile replicates, seed20260914. Retain scorer input JSON, three finalizer-backed pair files, code pins, per-run+pooled junction tables, GFF/count producer receipts, score output, stdout/stderr/exit. No custom estimator or relabeling of provenance fields. Integration to this final producer is **not yet implemented here**.
8. Any sampled/one-run/unresolved-coverage readout remains **UNASSESSABLE for full independent structural availability**, not negative. Missing HiFi is not n_HiFi=0 evidence of biological absence. No final STOP/GO publication from this wrapper; `require_final_coverage` always rejects. Existing scorer final-decision block stays intact. Withholding and duplicate-isoform policy remain external unresolved requirements.

## Capture and validator boundaries

`gate.py`:
- `read_binding`: exact canonical ROOT/dataset/run filename, symlink refusal for source file, unique validation role, **fresh MD5**, roles/source SHA256.
- `reference_binding`: fixed Git blob digest plus fresh actual FASTA/GFF MD5 and SHA256.
- `load_driver`: every consumed frozen source/panel pin checked before import; imported windows path checked; no torch import.
- `pilot`: complete expanded argv, every external command's stdout/stderr/exit, source+tool binaries+input inventory, scheduler job/host, elapsed seconds; retained sample FASTA and SAM; audit/DONE; recursive output size/SHA inventory; exclusive output directory, COMPLETE last. Scorer rc4 is accepted only for orientation UNRESOLVED, never treated as a negative structural result. Failure keeps outputs and FAILED JSON (scheduler timeout before publication is diagnosed from Slurm accounting; no COMPLETE).
- `verify_publication`: exact pilot stage inventory, exit codes, stdout/stderr linkage, recursive output inventory, original input byte identities, fresh role/reference checks, audit/DONE agreement. This is consistency verification, not proof against adversarially fabricated manifests. Controller must bind release hash, actual scheduler accounting and stdout; do not trust arbitrary caller-created receipts as authorization.
- `verify_alignment`: planned full-driver postflight checks outer exit/stdout/full recursive inventory, unchanged raw driver labels, canonical consumed inputs, exact full-run pass1 and pass2, extraction→BAM→junction command links, orientation key/audit, QC/scorer/panel pins. **No actual full alignment fixture/run was tested for this helper yet; not a full operational gate or mathematical table recomputation.** Review/extend with actual tool fixture before full use.

## Controller-only commands / first resource bound

Deploy only after independent review, with a fresh release/output directory; do not invoke training-oriented `orientation_audit.sbatch` (it gates c2 roles and declares a huge array). Deliver `gate.py`, `source_pins.json`, `authority.tsv`, `orientation_pilot.sbatch`, and only the pinned files under `frozen/`; hash the entire release including wrapper and sbatch before sending. Preserve that release manifest independently in the controller ledger. `load_driver` checks each frozen Git-derived file; Task 2 can also deliver the fixed authority object.

After checking current Slurm ownership/queue, no overlap and budget, controller may run:

```sh
# RELEASE and OUT are controller-selected NEW absolute remote directories.
# Release/log directory must exist before sbatch; worker has not created it remotely.
sbatch --parsable --output="$RELEASE/slurm-%j.out" --error="$RELEASE/slurm-%j.err" \
  --export=ALL,RELEASE="$RELEASE",OUT="$OUT" "$RELEASE/orientation_pilot.sbatch"
```

The template invokes exact paths for SRR28341943 and reference/GFF, and `gate.py` pins the authority and source. Gate only samples/alignment-audits 10,000 reads, never starts full pass1. The sampler makes **two full sequential decompression passes** over the 1.916 GB compressed input, plus compute hashes; 30 minutes is a hard cap, not a guarantee. If timed out, retain failed output/accounting, mark pilot incomplete and re-budget explicitly; no automatic retry. Fresh MD5s and reference validation happen only on compute. Four CPU/8GiB/30min bounds 2 CPU-hours; no GPU.

Retrieve receipt, every source/tool identity, command logs, retained sample+SAM, inventory and Slurm `State/ExitCode/Elapsed/TotalCPU/MaxRSS/AllocCPUS` (including batch/extern). Controller validates both workload exit and scheduler accounting, then runs `verify_publication` **on compute** because it rehashes inputs. A completed orientation pilot does not authorize full alignment or score publication.

### Separate full ONT alignment planning (NOT submitted)

Observed compressed total is 7,666,166,326 bytes across ~26.8M expected records. Largest run is 1.56× first run's compressed bytes. Full uncompressed SAM/PAF sizes and memory are unknown; do not schedule a three-run large array from these compressed sizes. First inspect pilot MaxRSS and actual full-read scan timings. Next request would be **one** complete SRR28341943 run, provisionally 16 CPU / 32 GiB / 4h only after controller-approved throughput/disk estimate and full-wrapper integration review. This is a planning envelope, not measured adequacy. Full driver retains pass1 PAF, selected FASTA, SAM, BAM and orientation intermediates; reserve scratch based on a bounded tool fixture/selected-read pilot before approval. Do not extrapolate pass2 sample speed as a guaranteed full-run pass1 rate.

Use unchanged `mvp_align_panel_longreads.py --root "$ROOT" --run SRR28341943 --reads <canonical-file> --ref <fixed-fasta> --gff <fixed-gff> --panels <pinned-panel> --ref-md5 <Git92-row-value> --gff-md5 <Git92-row-value> --threads 16 --execute` only through the separately reviewed external gate and exit/stdout/inventory capture. The driver output path is fixed `$ROOT/evidence/mvp_scoring_20260916/SRR28341943`; **if occupied, stop** rather than deleting/reusing it or aliasing ROOT to another authority. Driver redoes orientation; current code does not import pilot audit caches. Existing outputs must be preserved. This pilot writes elsewhere and will not occupy that path.

## Verification and limitations

Local `python -m unittest test_gate -v`: **11 tests passed**, including actual frozen orientation scorer CLI on a 1000-read synthetic SAM/GFF, wrong file with identical bytes, wrong role, fresh-MD5 mismatch, wrong authority blob, missing DONE, stale/wrong-run/duplicate-key DONE, changed audit, incomplete output inventory, stdout-only false completion, and sample/one-run/full-ONT final-coverage rejection. `bash -n orientation_pilot.sbatch` and Python compilation passed. `test_red.txt` preserves initial missing-module red run; `test_results.txt` records final run. These do not certify real minimap2/SAM processing or pilot end-to-end execution.

Advisor consultation was attempted but failed due provider client/model version incompatibility; no advice or review approval was received. No independent review has occurred for this new wrapper. Full postflight and bootstrap producer integration remain explicit work, not a silently claimed completed chain.
