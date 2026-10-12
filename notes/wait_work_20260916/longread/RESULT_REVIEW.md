# Actual SRR28341943 orientation result — read-only review

## Disposition

**Bounded payload consistency verified; scheduler acceptance BLOCKED. Do not label PILOT_COMPLETE or clean-scheduler success.** Job 6165555 produced the final worker receipt after returning from compute-side `verify_publication`. The frozen orientation result is **FAIL / use_uf=false**, a finding against the sense-orientation assumption, not a failed alignment payload and not negative full structural evidence. Full independent structural availability remains **UNASSESSABLE**, training eligibility false, decision null. No full alignment is approved by this review.

All remote actions here were read-only SSH metadata/accounting queries and explicit bounded file transfers. No jobs, remote writes, canonical FASTA reads/hashes, reference scans, DB scans, source changes, policy changes, or subagents. Verification ran locally on downloaded bytes, not the login node. Advisor consultation failed with provider/client incompatibility; no advisor endorsement was received. Graft auto-refreshed its local context index during lookup; no implementation was edited.

## Retained evidence and verification

Owned directory: `result_review/`. `remote_stat_accounting.txt` records live observations at **2026-09-16T21:54:40-0700**. `fetch.py`, `fetch_stat.txt`, and `fetch_manifest.json` document the explicit transfer list. Remote stat established regular files and bounded transfer size before fetching. Sample FASTA is **8,538,965 bytes**, SAM (`sample_alignment.stdout`) **20,228,425 bytes**; their combined 28,767,390 bytes justified downloading both for complete local output-hash verification. Total transfer: **33 files / 29,109,352 file bytes**, including small release sources and complete Slurm logs. No 1.916-GB canonical input was copied.

`python result_review/verify_local.py` completed successfully; machine-readable results are `result_review/verification.json`. This checks bounded consistency, not the scheduler acceptance conjunction. Specifically:

- Full `release/slurm-6165555.out` is 14,790 bytes, entirely parseable as JSON, exactly equal to `receipt.json` as JSON and byte-for-byte equal to its 14,789 bytes plus newline. Full Slurm stderr is empty; it was not inferred empty from a tail.
- COMPLETE exactly equals receipt SHA256 plus newline: **b8c5c78776a83ca1c5b25731cbfdf1bfd9c554dd3863ab2431bd38435dd3dc33**. Remote FAILED.json was absent at observation time.
- **All 16 receipt-listed output files / 28,775,072 bytes** were fetched and independently size/SHA256-verified, including sample FASTA, full SAM, commands.json, all six command stdout/stderr pairs, audit.json and DONE. Thus these output hashes are recomputed here, not merely receipt claims. This is the listed inventory, not a new exhaustive remote directory census.
- commands.json equals receipt commands; orientation command stdout equals audit.json and receipt audit. All six returned command exits are **0**, including the orientation FAIL (frozen scorer returns 0 for resolved FAIL).
- Nine fetched frozen source/panel files match source_pins.json, local frozen exports, and actual `git show b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6:<path>` bytes. Fetched gate.py, sbatch, authority.tsv and source_pins.json match reviewed local bytes. Gate SHA256 **0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f**; sbatch **e54378e5f4d86659d6ba44d4fe8ef88c53ca6b57b6ff5f0e939519268bb39832**. Every release-path input entry matches its downloaded size/hash. Existing `submission_receipt.txt` supplies controller deployment checks; this review also freshly compares fetched release bytes.

### Scheduler and measured resources — preserve the contradiction

| Step | State | ExitCode | Elapsed | TotalCPU | MaxRSS | AllocCPUS |
|---|---|---|---|---|---|---:|
| 6165555 | COMPLETED | 0:0 | 00:02:28 | 02:45.569 | not reported | 4 |
| 6165555.batch | COMPLETED | 0:0 | 00:02:28 | 02:45.568 | 51000K | 4 |
| 6165555.extern | **OUT_OF_MEMORY** | **0:125** | 00:02:28 | 00:00.001 | 1480K | 4 |

Receipt host is `cpu-50`, job `6165555`; wrapper wall_seconds **108.40214534010738**. That timer begins after initial preflight/input hashing and excludes final verification after receipt construction; it is not whole-job time. Actual minimap2 stderr reports **11.906 s real / 32.846 s CPU / 1.732 GB peak RSS**, mapped 10,000 sequences, and reference index total length 119,667,750 over seven target sequences. Keep both minimap2 RSS and Slurm's 51000K measurement; do not select the smaller figure as trustworthy full-run provisioning evidence.

Independent review explicitly requires rejection of OOM/nonzero accounting including extern. The observed extern failure violates that condition even though allocation/batch completed and full final stdout exists. Cause and impact of extern OOM are **not diagnosed** here; no assertion that it is harmless or that it invalidates the observed orientation arithmetic. Controller must reconcile operational accounting before advancing. Empty Slurm stderr does not erase sacct's OOM.

### Exact runtime commands and identity bindings

`receipt.json` and `commands.json` retain every expanded argv and corresponding stdout/stderr; the local verifier asserts exact arrays, not just stage names. Executed versions from retained outputs: **minimap2 2.28-r1209; samtools 1.3.1 / htslib 1.3.1; seqkit v2.13.0**. All three use `/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/`; sampler/audit use explicit sylvan Python. Tool binary sizes/SHA256s remain compute-receipt attestations, not freshly downloaded binary verification. No Python version was captured by this worker; an executable path is not a version observation.

Sampler invoked frozen `68_sample_fasta.py --fasta <canonical> --out <OUT>/sample.fa --n 10000 --md5 d802a86ae1673909ee21a0cd663c99ac`. Alignment is exactly minimap2 `-t 4 -ax splice -k14 -G 200000 --secondary=no --cs=long <fixed-full-reference> <OUT>/sample.fa`, **without -uf**. Audit consumes that exact retained SAM, fixed GFF, run SRR28341943, species Athaliana, dataset ont/Athaliana_FLIC_PRJNA1087576. No full pass1 command was executed.

Canonical input is `/data/gpfs/assoc/pgl/data/Transgenic/evidence/ont/Athaliana_FLIC_PRJNA1087576/SRR28341943.fa.gz`, 1,915,883,586 compressed bytes. Worker reports local FASTA MD5 **d802a86ae1673909ee21a0cd663c99ac**, SHA256 **11a79eb12ac43ae09c6fe1e64febfe1629c2111ee799a979aabcd526f78cf3ef**. Role is Athaliana/reference/**b1_validation_only**, expected_reads 6,952,504. Canonical roles SHA256 **288390b0e75a07845159f93f6a07c0033e4980b9dfe862dc2d982cac1704e348** matches earlier observation. ENA `source_checksum=f7a2b67708e0f0ef454fe6ddc2156301` is explicitly **FASTQ MD5**, not this converted FASTA MD5. Roles/read bytes were not freshly downloaded or hashed here.

Fetched authority is byte-identical to Git **92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv**, SHA256 **3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783**. Receipt reference paths/MD5s equal its Athaliana row:

- `genomes/Athaliana_167_TAIR10.fa`: MD5 **513e5ef30845ed754b00816a99abbf8e**, size 121,183,059; receipt SHA256 **63d56962ccbbdcc974950e5ccdc3e1d3d37a665f7161f658edcd3d19e9c63973**.
- `genomes/Athaliana_167_TAIR10.gene.clean.gff3`: MD5 **0264c349bc61ccceb546648edec46ebb**, size 60,503,119; receipt SHA256 **529510b60756c9a1ae5a590ec4097fd4b9773649c29e6a5543553118bb042c38**.

Those actual reference byte identities were verified by the worker, not recomputed now. Under operator trust in the bound release/scheduler/stdout, `gate.py:143–163,207–223` establishes that `verify_publication` returned before the final JSON print: it rechecked inputs, reference/roles, source and outputs on compute. COMPLETE alone would not prove this because it precedes verification. Full final stdout supplies the missing control-flow evidence; it does **not** resolve extern accounting or prove current large-input freshness.

## Sample and orientation interpretation

Sample metadata records **sampled=10,000**, **run_records=6,952,504**, seed **15565188439147493641**, short_run=false. Local FASTA header recount confirms 10,000; locally recomputed sample-ID MD5 **5c1db94ffec26f2b92c4dceab627c59a** matches metadata, and seed matches canonical-MD5 derivation. Whole-run record count is the worker's two-pass sampler result, consistent with roles; it was **not independently recounted** here, nor was uniform sample selection regenerated from the full run.

Full SAM locally contains **10,129 records / 10,000 distinct QNAMEs**. Recomputed annotation-independent ordered filters match: unmapped 171, secondary/supplementary 129, low MAPQ 313, unspliced 3,180, missing ts 15. Remaining 6,321 require annotation. Audit reports no qualifying gene 1,199, ambiguous gene 92, no intron inside gene 5, eligible 5,025; these partition all 10,129 records. Audit qualifying genes 18,838 and dropped opposite-strand genes 2,652 remain worker observations, not a local GFF recomputation.

- Read sense **2,679 / 5,008 = 0.534944089456869**; valid orientation exceeds minimum 1,000, MAPQ threshold 20.
- Wilson95 recomputed using frozen scorer's z=1.96: **[0.521108, 0.548726]**.
- Threshold **0.95**, hence frozen **FAIL**, **use_uf=false**. This supports withholding the sense-only assumption for this sampled run, not absence of transcript evidence.
- Motif/annotation agreement **5,008 / 5,025 = 0.9966169154228856** is a separate diagnostic and cannot authorize -uf.
- Counts and fractions agree across receipt/audit/command stdout. Annotation-dependent counts themselves were not independently recalculated without the GFF; output hashing is not biological re-analysis.

DONE was checked as exact bytes: run SRR28341943; audit MD5 **a329049df63a3fa3d640dfdc89b4c854**; four-MD5 key:

`d802a86ae1673909ee21a0cd663c99ac:513e5ef30845ed754b00816a99abbf8e:0264c349bc61ccceb546648edec46ebb:65949011a59729f1bcba3a34ae8353b6`

The fourth MD5 was recomputed from fetched frozen **67_orientation_audit.py**, not a different scorer. First three bind worker-observed canonical reads/reference/GFF; no fresh large-file MD5 is claimed here.

## Whole-chain next gates — controller review before full alignment

1. **Operational blocker first:** reconcile extern OOM/nonzero and discrepant RSS with scheduler/site evidence. Retain these observations; no fabricated clean accounting, automatic retry, or retrospective PILOT_COMPLETE. Recheck queue/ownership, overlap, budget and output isolation before any newly authorized job.
2. Independently review the full-run wrapper, environment/PATH executable identities, command/stdout/exit capture and `verify_alignment` postflight using an actual bounded SAM/BAM/junction fixture. Current 11 gate tests and this orientation payload do not certify that helper. Fix/review publication ordering before relying on standalone marker automation. No full-alignment approval yet.
3. Plan **one SRR28341943 full run first**, not a three-run array. Retain the no-uf direction observation; frozen driver redoes orientation and does not import pilot caches, so bind its own new audit/DONE and enable -uf only if that audit independently passes. No extrapolation of this result to other runs. Provisional 16 CPU / 32 GiB / 4h is only the earlier planning envelope: obtain throughput/scratch estimates and approved resource limits, using all observed memory measurements. Pilot sampling/full-input scans are not full-pass1 throughput evidence.
4. Preserve canonical ROOT/roles/reference/panel/source identities; fresh large-input checks belong on compute only. Check fixed full-driver output `$ROOT/evidence/mvp_scoring_20260916/SRR28341943` is unoccupied; stop if occupied, never overwrite or alias ROOT. Whole-run pass1 must precede retained-ID extraction from the same canonical run and no-uf full-reference splice pass2, sorting/indexing, per-run junction/QC capture. Sample alignment cannot stand in for full pass1 or coverage.
5. Verify each separately bound run and all required locus/evidence coverage, then pool through repeated `--alignment`, retaining per-run outputs. Apply unchanged read-group rejection, primary/MAPQ20, FLAG × ts, ONT3/HiFi2 support thresholds. Centered 30,720 windows and ±10,000 prefilter are not the recall denominator: precision support is window-contained, recall support gene-contained. A one-run/sample outcome remains UNASSESSABLE for full structural availability.
6. HiFi acquisition/identity/roles or any ONT-only policy remains an external authorization gate. **No HiFi action, substitution, zero-support inference, or policy selection here.**
7. Bind 300 paired predictions (50 loci × 3 seeds × off/on) plus 50 baseline to actual finalizer/decoder/checkpoint receipts and shared panel/reference; retain original GFFCompare counts. Complete and review the actual junction/GFF counts producer, per-run+pooled inputs, unchanged `mvp_score` and 2,000 paired bootstrap replicates (seed20260914). Resolve withholding/duplicate-isoform and final coverage/decision gates. No training eligibility or STOP/GO authority follows from this orientation observation.
