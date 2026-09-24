# B6 EST DIAMOND execution plan — scoped authorization

User requested EST blastx execution and separate worktree. Production execution is authorized subject to safe launcher review/preflight; this does not freeze all A46. Worktree-local implementation first; no shared main edits or production jobs by implementer.

## Verified assets

pronghorn root `/data/gpfs/assoc/pgl/data/Transgenic` (= ~/scratch/data/Transgenic).
DIAMOND `/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond`, version2.1.24.178.
DBs at `evidence/protein/diamond/`: `odb12_viridiplantae_filtered.dmnd`, `swissprot_viridiplantae_reviewed.dmnd`; `PROVENANCE.txt`, `DB.sha256`, `build_dbs.sh`, `DONE`.
Swiss source `evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz` (OX=taxid). Ortho source path must be read from build_dbs.sh; retain already-filtered source, never original unfiltered partition. Existing A19 filtering must be preserved.
Nine primary screened inputs `evidence/est/{SP}/univec/est.univec.fa.gz`, DONE/summary; role manifest `evidence/DATASET_ROLES.est_v1.tsv` c2_training_eligible. Total recorded 5,068,470 sequences.

Species roots: Athaliana3702 Bdistachyon15368 Gmax3847 Osativa4530 Ppatens3218 Ptrichocarpa3694 Sbicolor4558 Sitalica4555 Vvinifera29760. Do not substitute Osativa39947 for species root4530.

## Binding execution decisions

- `.dmnd` accession selection is unsupported in v2.1.24 (source dmnd.cpp604–606); --seqidlist is not a valid solution. Existing DBs lack taxonomy maps. Build species-filtered DB before search from already-approved FASTA.
- Filter by frozen lineages of every observed subject taxid, not only exact root matches or live descendant snippets. Reuse `revision/scripts/64_orthodb_filter.py` read_lineage/excluded_taxids/fetch_lineage (merged IDs). Unknown lineage fails closed; no --allow-missing-lineage. Filter Swiss via OX taxid and Ortho via header taxid, preserve full IDs. Verify each lineage includes its queried taxid when accepting cached maps.
- Keep each source separate, preserve prior exact-match exclusions in Ortho input. Count/record source hashes, frozen lineage hash, retained/excluded taxids and sequences, filtered FASTA and DB hashes.
- Minimal approach: explicit prepare-lineage stage scans source identifiers on compute node and optionally fetches missing lineages; freeze cache before any production filtering. Search stage only reads lineage, never silently refreshes it. Use per-species per-source sequential work directories; no full9 array until pilot resources measured.
- Core flags: --frameshift15 --evalue1e-5 --min-orf30 --max-target-seqs25, explicit --max-hsps1 preserving DIAMOND default, --no-parse-seqids. Use actual separated CLI tokens. HSP length>=30aa is a downstream acceptance filter, not min-orf semantics; preserve raw hits and do not claim every output HSP passes.
- Minimum raw columns: qseqid qlen sseqid slen qstart qend sstart send length evalue bitscore qframe btop. Retain input sequences and DB/provenance so BTOP can be reconstructed. Fixture adds cigar qseq_gapped sseq_gapped qseq_translated to validate internal frameshifts.
- Preserve raw 1-based endpoints. No phase rasterization or HSP-to-genome projection in this launcher increment.
- Never overwrite existing output, input, DB or lineage. Publish DONE only after outputs/QC successful; unique workdir, trap/error status, nonzero exit propagation. No eval-only species accepted. Do not modify seed123 jobs.

## Implementation scope and test-first plan

Create minimal files under `revision/scripts/evidence/` for prepare/filter/run plus fixture and Slurm entry point; prefer Python stdlib subprocess and existing taxonomy resolver, no workflow framework. New tests under `revision/scripts/tests/`.

1. Write failing tests for Swiss/Ortho taxid parsing, root+descendant+merged exclusion, missing lineage abort, input guards, no destructive overwrite, correct command construction and distinct source outputs. Reuse existing script via importlib if suitable; do not copy its entire CLI/hash/filtering pipeline.
2. Implement streaming FASTA filtering and command builder/runner with explicit inputs/out dir/threads/tmpdir and immutable source/lineage identity. Validate expected9 species only.
3. Create deterministic tiny same-species/cross-species fixture FASTA + nucleotide EST queries: ordinary, reverse complement, +1/-1-nt frameshift, insertion/deletion; sequences long and diverse enough to satisfy configured thresholds. Verify excluded subject absent in filtered DB/search and retained positive controls hit. Validate BTOP frame-shift markers by recoverable coordinate/translation checks, not merely nonempty output.
4. Fixture is runnable with real DIAMOND; local unit tests must not pretend stub execution is real DIAMOND. Record real-engine tests as not run if executable unavailable locally. CPU Slurm wrapper for fixture account/partition cpu-s1-pgl-0,4 CPUs,8G,30min, explicit GPFS tmp directory (TMPDIR not assumed).
5. Do not submit by yourself. Return tested code, exact fixture sbatch command and required stage paths for controller review and authorized submission. Production time/memory remain unmeasured; do not launch9 by guessing48h capacity.

## Risks recorded

Original DB build job6163304 allocation and batch COMPLETED0:0, extern OOM0:125. Treat existing DONE as insufficient integrity proof; checksums/source verification on compute before use. No heavy scans/hashes on login node.
Existing Ortho lineages referenced on gpu `/home/pgl/scratch1/wyim/transgenic_data/orthodb_filtered/lineage_ncbi.tsv`, absent on pronghorn in inspected paths. Retrieve/copy only with controller oversight; alternatively prepare frozen cache with existing NCBI resolver. Live UniProt descendants are informational, not production lineage authority.
