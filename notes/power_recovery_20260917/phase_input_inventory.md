# Phase input inventory — 2026-09-17

## Scope and disposition

Research/document only, while CPU job 6165628 belongs to another worker. No GPU/panel/job-status checks, remote writes, submissions, large hashes, full downloads, alignment/BLAST reruns, evidence generation, geometry rebuild, or scientific policy selection. Reuse accepted diagnostic geometry **5c46c75**; its scope excludes authentic traceback, route-A conversion, conflict/count policy and rasterization [L3].

**Found:** all nine retained OrthoDB route-A raw GFF.gz files, embedded `##PAF` with `cg`/`cs`, CDS strand/phase/Target, acceptance alignment tables and protein ambiguity inventories. Small real plus/minus split-codon **candidates** are immediately accessible at file starts. **Not established:** replay-validated miniprot phase/Target semantics, complete input hash/producer closure, Swiss route-A alignments, or nine-species full A11 member closure. Existing MVP data is not that closure.

Evidence notation:
- **O** = directly observed during bounded SSH reads on 2026-09-17, first timestamp `00:45:08.874930` from remote clock (timezone not independently queried; session records use PDT).
- **R** = retained local receipt/source/design, not a fresh remote byte attestation.
- **I** = inference/candidate only, not validation.
- **U** = unverified/unavailable in inspected records; not a claim of global absence.

Remote root **R0** = `/data/gpfs/assoc/pgl/data/Transgenic`; **E** = `R0/evidence`; **A(SP)** = `E/protein_alignments/SP`.

### Local evidence citations

- **L1:** `notes/EST_PHASE_DESIGN_20260916_1530.md:89–132,217–267` (route A/conflicts, minimum interface, nine-species inputs); coordinate definitions `:36–85`; synthetic fixtures `:136–211`.
- **L2:** `notes/power_recovery_20260917/remaining_tracks.md:9–22` (preparation permitted; unresolved science/execution gates).
- **L3:** `notes/est_execution_20260916_1530/geometry.fix1_review.md:22–32` (accepted geometry and explicit exclusions).
- **L4:** `notes/b6_mvp_technical_candidate_20260915.json:1`, keys `artifact`, `inputs`, `qc`, `limitations` (historical digest-bound Athaliana MVP candidate).
- **L5:** `notes/b6_blastx_lineage_preflight_20260914.md:13–36,93–107`; `notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md:81` (protein sources/lineage identities and ancestry-authority limitation).
- **L6:** `revision/scripts/evidence/protein_accept.py:106–178` (alignment verdict before ambiguity filtering; CDS phase discarded; protein IDs discarded from final support rows).
- **L7:** retained source `notes/wait_work_20260916/gpu_followup/source/revision/scripts/mvp/mvp_build_c0lite.py:54–55,114–168,171–274` (MVP schema/limited closure).
- **L8:** `.atl/archive/resume_before_20260916.md:465–491` (historical chunk equivalence, ID renumbering, completion; not new execution instructions).

## 1. Route-A retained objects

**O:** for every SP below, stat found `A(SP)/SP.protein.gff.gz`, `PROVENANCE.txt`, `DONE`; listing `A(SP)/accept/` found `ACCEPT_DONE`, `SP.alignments.tsv`, `SP.introns.tsv`, `SP.mapping_ambiguous.txt`, `SP.summary.json`, `SP.tsv`. Read only the first 1,600 decompressed characters per GFF, first two bounded lines of alignment/ambiguity files, and ≤4,096 characters per provenance. `DONE` existence is not a phase certification.

| SP | Raw GFF.gz bytes (stat) | GFF MD5 recorded in PROVENANCE, not recomputed | Reference basename recorded under R0/genomes/ |
|---|---:|---|---|
| Athaliana | 9138413087 | 96f87a553b28613316fb9cdf0bafaa08 | Athaliana_167_TAIR10.fa |
| Bdistachyon | 9419821042 | 669c6ee87c547a12f6509d989c3dba3d | Bdistachyon_314_v3.0.fa |
| Gmax | 13094276506 | cdb5c3bb1a69e4c7b017a0c851586d29 | Gmax_880_v6.0.fa |
| Osativa | 9651015990 | 08b51c7cfc2997c2fc6cf1463cd5f8be | Osativa_323_v7.0.fa |
| Ppatens | 9448979798 | b6ebc1d6149fb8c90ed83f7a8f93baad | Ppatens_318_v3.fa |
| Ptrichocarpa | 11054070858 | 508aa6dcc948c9785ae2347a5be1ca56 | Ptrichocarpa_533_v4.0.fa |
| Sbicolor | 9678736768 | dbde172801d4781521ac2b990661987f | Sbicolor_730_v5.0.fa |
| Sitalica | 9647215761 | 0e17de830f3458c81b25b3366320ec77 | Sitalica_312_v2.fa |
| Vvinifera | 9417952697 | 574bca35166e7ce3e69c75359bcfc4ff | Vvinifera_T2T_ref.fa |

Direct citation for each row: `A(SP)/PROVENANCE.txt`, named `gff_md5`/`reference` fields; bytes are this inventory's stat observations. Corresponding recorded reference MD5s, same order: `513e5ef30845ed754b00816a99abbf8e`, `228abca58ba0585e8a569570501ba82b`, `64ed1e492e5255027306220b514b5daf`, `003aa04040ec2fa51cd8fde02a2515d4`, `6c3477dcd28f561246a00cb90adc1e48`, `20c6f4bb6f8b52887a2a08addedac6ea`, `cdfc7c61116cdf1a796a6b303e248e28`, `d8f77d532d8c87f314f6175146fa6728`, `d286fd52de1e4e55d1550c923976056b`.

**O provenance:** all nine identify miniprot `0.18-r281`, source `E/protein/odb12_Viridiplantae.filtered.fa.gz`, source MD5 `453cb32b02e0799950d7d5f4de5f62ac`. Athaliana's complete 2,003-byte provenance records `--gff --outs=0.97 -N 20`, six chunks, 12,115,085 proteins/408 taxa, merge `2026-09-08T22:54:49-0700`, merged IDs `1..12734296`, and explicitly says acceptance is a separate pass. Other species' parameter fields were not individually extracted; identical full commands are not freshly attested.

**O:** `E/protein/chunks_v1/MANIFEST.tsv:1–7` is 746 bytes, header `chunk\tseqs\tfile\tmd5`; exact files `chunk0.faa.gz` through `chunk5.faa.gz` in that directory. Counts 2,019,181 for chunks0–4, 2,019,180 for chunk5. Recorded MD5s respectively `e8c6980e0ca2611a7e6c721a49253299`, `3883fe99c55a4f522a6d123dd84e8107`, `1328db6d512d42d1656e612e12ecce1b`, `66b1f340418af0f30a95008e94e5d1e1`, `ee2c634eaf3af6275374a34492104ff0`, `e8cc3e707dc301ce6f5edf0e1faccef6`. Chunk bytes were not read. Historical equivalence and ID-renumbering rule are L8; bare `MP000001` is not globally unique across species/chunks/runs.

**R source sequence identities:** OrthoDB compressed FASTA SHA256 `fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f`; Swiss source `E/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz`, SHA256 `8d66a18e2ad350fd640f1fb97a180a8091c544efd33351e5f451bb03e11379c9` [L5]. These bind whole source files, not individual normalized protein sequences. A/B same-accession equality is insufficient. Existing A19 filtering is not proof of per-target same-species/descendant exclusion for all nine targets.

**U:** no Swiss route-A raw alignment path established. No fresh raw-GFF SHA256, miniprot binary hash, producing-wrapper hash, or per-protein sequence-hash map established. Do not promote named versions/old MD5 receipts to a new immutable phase input manifest.

### Exact observed schema and acceptance boundary

All nine `SP.alignments.tsv:1` have the same header:

```text
alignment_id\tprotein\torganism\tchrom\tstrand\tscore\tidentity\tcoverage\tframeshifts\tcds_blocks\tverdict\treasons
```

`SP.mapping_ambiguous.txt` starts with protein IDs, no header (e.g. Athaliana first two `1000413_0:000019`, `1000413_0:0000b5`). Membership was not scanned. `A(Athaliana)/accept/Athaliana.summary.json` directly records `support_version=2`, `support_excludes_mapping_ambiguous=true`, `ambiguous_alignments_excluded=309205`, `ambiguous_blocks_excluded=872638`, `ambiguous_introns_excluded=563433`, 12,734,296 alignments, 10,496,478 accepted, 136,022 ambiguous proteins. Other summaries were not opened. **An `accepted` row alone does not demonstrate A43-retained support:** the stream writes verdicts before downstream ambiguity filtering [L6]. Keep the ambiguity inventory and source identity in the join. The obsolete Athaliana sibling `accept_v1_ambiguous_leak/` was observed; do not substitute it for `accept/`.

Observed GFF starts `##gff-version 3`; `##PAF` carries protein name/length/query endpoints, strand, reference name/length/endpoints, numeric alignment fields and tags `AS`, `ms`, `np`, `fs`, `st`, `da`, `do`, `cg:Z`, `cs:Z`. GFF rows retain nine columns: seqid, source=`miniprot`, type=`mRNA`/`CDS`/occasionally `stop_codon`, start, end, score, strand, phase, attributes. Attributes observed include mRNA `ID`, CDS `Parent`, `Rank`, `Identity`, `Target`; optional `Frameshift`, `StopCodon`, `Acceptor`. This confirms retained detail, **not** validation of miniprot-specific operation units or phase/Target conversion. Coverage `SP.tsv` is unsuitable as route-A phase input [L1,L6].

## 2. Existing real fixture candidates, not certified fixtures

Coordinates below are **raw GFF inclusive coordinates**, not internal zero-based values. Direct evidence is exact raw-file path above plus line/parent ID; only a prefix was read. No reference DNA/protein replay, acceptance reclassification, ambiguity membership lookup, or phase interpretation performed.

| Candidate | Direct raw evidence | Observed detail / limit |
|---|---|---|
| Plus split/partial-protein candidate | Athaliana GFF lines2–5, `MP000001`, protein `1000413_0:000002` | PAF length172, q `[0,157)`, `fs:i:0`, `cg:Z:41M82V8M4I8M2D58M1I36M`; Chr5 CDS6431865–6431989 phase0 Target1–41; CDS6432069–6432405 phase1 Target42–157. Alignment TSV line2 says accepted. Split interpretation requires replay; target truncation157/172 is not proof of a terminal partial nucleotide codon. |
| Minus split/partial-protein candidate | Sitalica GFF lines5–8, `MP000002`, same protein | Minus scaffold_7; coding-order CDS35312723–35312847 phase0 Target1–41, CDS35312121–35312451 phase1 Target42–160. PAF prefix shows length172, q `[0,160)`. Accepted/ambiguity status not inspected for this ID. |
| Matched plus raw candidate | Sitalica GFF lines9–12, `MP000003`, same protein | Plus scaffold_8 CDS1835448–1835572 phase0 Target1–41, CDS1835842–1836172 phase1 Target42–160, Rank2. Not accepted merely because it resembles the minus record. |
| Minus multiexon baseline | Ptrichocarpa GFF first record, `MP000001`, protein `1000413_0:000000` | Minus Chr10 CDS1715712–1715810 phase0 and1712619–1713407 phase0; Target1–21 /22–283. TSV line2 accepted; not a nonzero-phase split fixture. |
| Stop-boundary warning | Vvinifera GFF lines2–5, `MP000001` | PAF `[25142061,25142910)` but CDS25142062–25142913 plus explicit stop_codon25142911–25142913. Do not equate CDS length/3 to Target span or assume identical PAF/CDS endpoints. |
| Rejected negative control | Sitalica GFF lines2–4 / alignment TSV line2, `MP000001` | Minus, Frameshift1/StopCodon1; verdict rejected, reason frameshift. Preserve rejection; not eligible fixture evidence. |

The Sitalica supplementary prefix read was capped at 7,000 decompressed characters, reporting only non-PAF rows among the first18 lines. **No certified terminal partial-CDS/partial-codon example was found in these bounded reads.** Nonzero internal phase and incomplete protein coverage must not be relabelled terminal partial codons. Existing synthetic plus/minus split fixtures F1/F2 and clipping/RC fixtures are already covered by geometry5c46c75 [L1,L3]; they do not certify real miniprot conventions.

## 3. A11 member closure and screened-query identity

> Controller recovery annotation (01:10 PDT): stale UniVec121 PROVENANCE is already documented in `.atl/quarantine.md §1m` and must NOT reopen completed C1 or become a new screening-rebuild blocker. `notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md:53,77` accepts actual query SHA/MD5/path/record-count bindings plus ≥100nt checker scope for all18units. A future phase input manifest should reuse that accepted provenance chain, rather than reread a nearby legacy PROVENANCE as primary authority. Full A11 membership and phase semantics remain separately unresolved; this inventory did not invalidate C1.

**O:** the nine `E/est/SP/` immediate listings contain raw `est.fa.gz`, `.md5`, `.orig`, repair/fetch records and `univec/`; no named member/clone/mate closure table was present at that level. This is a bounded directory observation, not an exhaustive search of E or the cluster.

**O Athaliana:** `E/est/Athaliana/univec/` contains `est.univec.fa.gz`, `.md5`, `est.univec.min121.fa.gz`, `.md5`, `report.tsv`, `report.min121.tsv`, `summary.json`, `summary.min121.json`, `PROVENANCE.txt`, `DONE`, `hits.tsv`, `chunks/`, `trim.100.out`.

- `report.tsv:1` exact header: `accession\tqlen\taction\tkept\tcategories`; first row `BX836347.1\t360\tkeep\t1-360\t`. It is 61,665,242 bytes; only first3,500 characters read. Report contains `ES156584.1\t110\tkeep\t1-110\t` in that prefix.
- Raw FASTA first header: `>BX836347.1 BX836347 Arabidopsis thaliana Silique Col-0 Arabidopsis thaliana cDNA clone GSLTSIL3ZC05 3PRIM, mRNA sequence`.
- Screened FASTA first header is the same plus ` univec=keep:1-360/360`.
- Recorded screened compressed MD5 sidecar: `d1aa1abf703b35285c8a3d0317067759`. No sequence/file hash recomputed.

**Important provenance mismatch (O):** Athaliana `univec/PROVENANCE.txt` still says `min_len=121`, records_out755367, date2026-09-03, while current `summary.json` says records_out1046825 and `report.tsv` retains110nt. This appears to be legacy sensitivity provenance alongside later primary outputs (**I**, not reconciled here). Do not bind primary100 inputs to that provenance by directory proximity or a zero-byte DONE marker. Obtain existing A37 primary rerun provenance/hash closure; do not regenerate screening.

The nine current `univec/summary.json` files were read (≤1,200 chars each):

| SP | records_in | records_out | original records with split action |
|---|---:|---:|---:|
| Athaliana | 1529700 | 1046825 | 5128 |
| Bdistachyon | 206255 | 206123 | 191 |
| Gmax | 1461724 | 1374461 | 1065 |
| Osativa | 1255251 | 1244141 | 8550 |
| Ppatens | 382587 | 381809 | 173 |
| Ptrichocarpa | 89943 | 89450 | 6 |
| Sbicolor | 209835 | 209441 | 67 |
| Sitalica | 66027 | 66040 | 44 |
| Vvinifera | 446853 | 450180 | 11415 |

These are receipt values, not recounted records, and do not establish split-part→molecule closure. Raw description clone/mate text demonstrably survives at least the first Athaliana record, but no normalized clone/library/mate/accession-version table or completeness assertion follows from that one example. Screened `_partN` identity must include arm, original accession interval and sequence identity [L1]; no inference of adjacency across trimmed bases.

**O dataset metadata:** `E/DATASET_ROLES.est_v1.tsv` (3,252 bytes, fully read) header:

```text
dataset\trun\tspecies\tgenotype_stratum\tinstrument\tdata_type\texpected_files\texpected_reads\tsource_checksum\tsource_checksum_authority\tlocal_fa_md5\trole\tbasis\tnote
```

Nine training species have recorded role `c2_training_eligible`; Zmays/Slycopersicum `b1_validation_only`. Athaliana stratum `reference` explicitly says predominantly Col-0, **not resolvable per accession**; other eight training species are `unknown`. These are dataset-level designations, not new phase training authorization or member-level genotype/library evidence. `E/PROVENANCE_est.tsv:1` header `date\tsource\tspecies\ttaxid\tquery\tcount_at_download\trecords_on_disk\tmd5\tstatus\tnote`; prefix records NCBI nuccore/gbdiv_est download/repair and no per-accession library closure.

**R available limited molecule map:** `E/mvp_full_20260915/fullc0lite.db`, 1,068,773,376 bytes, historical SHA256 `ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d` [L4]. No database opened/scanned this task. The retained builder [L7] establishes schema, not a fresh SQL inspection:

- `est_member(member VARCHAR PRIMARY KEY, clone_key VARCHAR, molecule_id VARCHAR)` after temporary component removal.
- `est_placement(placement_id BIGINT PRIMARY KEY, member VARCHAR, flag INTEGER, kind VARCHAR, chrom VARCHAR, alignment_strand VARCHAR, start0 BIGINT, end0 BIGINT, cigar VARCHAR, ts VARCHAR, tx_strand VARCHAR, mapq INTEGER)`.
- `member_block` has placement/chrom/start0/end0, **no query positions**; do not phase-project block spans across D.
- `clone_key` strips QNAME suffixes matching `(?:\.\d+|_[FR])+$`; not a parser of biological clone description, library, mate, genotype or split-part metadata.
- Receipt: 1,046,825 members, 986,381 molecules, rule-a collapsed60,444, rule-b collapsed0. These are MVP observations, not proof no biological clone duplicates exist.

**Blocker:** full nine-species A11 library-aware clone/accession-version/mate/split-part membership and exclusion closure is **U**. L4 explicitly defers full C0/A11 semantics. Do not reuse MVP `clone_key` as full A11 by default or collapse different screening arms by a suffix alone.

## 4. Policy-neutral boundary checklist

This is a required-information checklist from L1, **not** a selected schema release or implementation authorization.

### Input identities

- [ ] Bind species/assembly/reference FASTA+FAI/alias dictionary, raw BAM+BAI/header/source SHA and placement identity. Primary BAM path known by L1: `E/est_align/SP/SP.est.bam`; PAF is a separate pass, not interchangeable geometry.
- [ ] Bind screened FASTA arm/QNAME/length/sequence SHA to source accession-version and retained interval, summary/report and reconciled primary provenance. Resolve Athaliana stale-min121 provenance above without overwriting originals.
- [ ] Supply full member→molecule closure with library/genotype/roles, clone/mate/split/version reasons and exclusion membership; explicitly distinguish unknown fields.
- [ ] Bind C1 raw-file/row identity to a separately reviewed accepted-HSP producer, contract and output hashes/count reconciliation. HSP acceptance remains unresolved [L2]; raw DONE/diagnostic rows are not accepted evidence.
- [ ] Bind A GFF+embedded PAF, acceptance IDs, ambiguity inventory, producer/tool/command hashes, per-target protein exclusions and normalized protein-sequence identities shared with B. Treat missing A as `not_crosschecked`, not agreement/disagreement.
- [ ] Preserve both source FASTA and lineage identities; prove actual source correspondence/per-target exclusions rather than equating organism with protein. L5 ancestry-authority limitation remains.

### Codon-event interface

- [ ] Parent HSP identity includes species/source/raw SHA+row, query member+molecule/sequence SHA+L, protein ID+sequence SHA+species, raw qstart/qend/frame/sstart/send, scores, acceptance-contract SHA and provenance reference [L1:230–234].
- [ ] Event identity carries parent ID, event ID/ordinal, **ordered original-screened-query nucleotide positions**, optional subject-aa index and explicit event class. Encoding may be affine runs; retain exact replay correspondence.
- [ ] Expose omitted/reused positions, gaps, insertion codons, frame-change boundaries and endpoint validation. Do not reconstruct them from signed frame/endpoints or choose whether they contribute support.
- [ ] A adapter must retain parent alignment ID, raw CDS phase/Target/strand and cg/cs; verify actual plus/minus split and terminal partial conventions, including stop-codon boundary behavior. Candidates in §2 do not close this box.

### Output identity / dispositions

- [ ] Bind output to exact input manifest, geometry5c46c75, producer/adapter versions and separately approved policy-contract identity; fresh exclusive namespace, no overwrite, failure artifacts preserved.
- [ ] Preserve HSP/member/molecule/placement/protein/source keys; contig/start0/end0/strand; diagnostic ordinal-at-start plus step and event/query-run references, rather than ambiguous constant phase over an interval [L1:236–242].
- [ ] Retain status/reason, route-A match ID, phase comparison and separate subject-residue/codon correspondence; retain ordered genomic tuples for split codons and missing/unassessable distinctions.
- [ ] QC reconciles received/accepted/projectable/rejected/conflicting/retained and unmapped/clipped/inserted/reused bases, route-A comparability, molecules and source overlaps. Diagnostic output remains `not_training_eligible` until downstream exclusion/role/membership gates are satisfied.

**No choices made:** final phase label meaning/order, nonprimary emission, ts contradiction treatment, insertion support, whole-HSP/self/cross-member conflict rules, same-residue comparison scope, pair-local versus final-cell masking, and phase aggregation conventions stay with the explicit local-contract owner [L1,L2]. No approved accepted-HSP/codon-event producer is supplied by this inventory.

## 5. Work performed and remaining blockers

Completed bounded preparation: located retained raw detail for nine species; recorded exact observable schemas, receipts and small candidate locations; separated the A43 alignment verdict from filtered support; identified retained but limited A11/MVP inputs and a primary/sensitivity provenance mismatch. Further execution is blocked on scientific contract/accepted-HSP boundary, full A11/provenance closure and real miniprot replay validation, **not** on geometry rebuilding or another blastx search.

SSH used fixed paths, `stat`, immediate directory listings, bounded `read(n)`/`readline(n)` and gzip prefixes only, with `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python -B`. An initial `python3` command failed because it is absent from PATH, before file access; then the established explicit interpreter was used. No recursive scan or content search of huge files, no remote hash or temporary file. Gzip may buffer beyond requested characters; each read was a bounded prefix, never decompression to EOF. Small provenance/summary/role files alone were read in full when below caps. This document retains the extracted observations, not a downloadable fixture archive. Advisor consultation failed with client/model-version incompatibility; no advisor review claimed.

Graft was consulted before source exploration; estimated reported savings total **1,104,267 tokens**. No project source/policy edits or main merge. This file is the sole intentional deliverable.
