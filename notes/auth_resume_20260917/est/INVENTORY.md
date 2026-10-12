# Remaining EST input inventory, 2026-09-17

## Result and limits

Read-only investigation for the controller. Application source, original outputs, scientific approvals and remote files were not intentionally changed; no job, aligner, original-data scan, raw/BAM/FASTA read or large-file hash was run. Only this directory contains intentional new deliverables. A required `graft ask` in `/tmp/est-real-source-binding-20260917` automatically reported a graph refresh of one changed file; this was graph housekeeping, not a source edit. Advisor failed with a client/model compatibility error and supplied no review.

**Concrete progress:** one existing, real Sitalica member receipt supplies accession, clone text, genotype text, screened identity and UniVec interval. Two additional historical Ptrichocarpa searches now have captured executable/query/database/raw-output receipts, complete command vectors and complete execution logs. Exact retained Sitalica route-A paths, chunk0 producer log, package identity and acceptance receipts are captured.

**Still blocked:** no authoritative biological-library identity/alias table, documented mate map, all-member audit or full exclusion inventory was located. No molecule ID was invented. No real miniprot phase adapter or new producer registry was implemented. New Ptrichocarpa records are evidence candidates, not accepted producer bindings. Full A11/full C0/training readiness remain false.

## Evidence navigation and budget

All remote reads used `ssh pronghorn` and the fixed Python interpreter under `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python`, with `-B`. `capture.py` used explicit paths only, stat-before/read/stat-after, complete small text files only, a 1 MiB per-file ceiling and cumulative 16 MiB reservation. No recursive filesystem search. Requests, remote timestamps, exact text, size, mtime and SHA256 of captured small files are in `01` through `04.request.json` and corresponding `.capture.json`; stderr is retained. Console copies are redundant, not extra remote reads.

Remote content read: **330,122 bytes** total. Largest remote file read: **142,550 bytes**, Ptrichocarpa SwissProt provenance. Large objects were listed/stat-observed only. Local retained diagnostics: only a 1 MiB prefix was inspected to extract one member, not the 12,557,738-byte full receipt. Original data behind that receipt were not opened. `retained_member_sample.json` records that extraction and the small parent-report hash; whole diagnostic identity is inherited from the accepted review, not freshly recomputed.

- `01.capture.json`: evidence/EST directory listings, dataset roles, download provenance, Sitalica route-A provenance.
- `02.capture.json`: old input inventory, Sitalica input/accept/chunk listings, miniprot index command, protein chunk manifest, primary EST alignment receipt.
- `03.capture.json`: remaining-eight wrapper/input receipt; Sitalica miniprot chunk0 log/DONE; current A43 acceptance summary/DONE; package directory listing.
- `04.capture.json`: complete Ptrichocarpa two-source command/provenance/blastx/makedb receipts, miniprot package record, chunk-equivalence receipt, Ptrichocarpa alignment receipt and output listings.
- `producer_candidates.json`: compact exact extra-producer identities, command vectors, evidence hashes and dispatch log lines. Not an accepted registry.
- `retained_member_sample.json`: actual Sitalica member, original/screened header and report row from the existing accepted sample chain.

## Accepted boundaries consulted

Latest accepted review: `notes/seed123_recovery_20260917_1400/PRODUCER_LENGTH_FIX_REVIEW.md`; mathematical/source attribution: `est_failure_diagnosis/LENGTH_SEMANTICS_REVIEW.md` in that directory. Older `PRODUCER_SEMANTICS.md` is historical and superseded concerning unresolved dispatch. The accepted outcome remains 200 focused tests, 414 rule-accepted HSPs, 48 exact-truth failures, 380 successful geometries and 34 unsuccessful geometries. These 34 are not all interchangeable conflict classifications. No rerun here.

Current source `/tmp/est-real-source-binding-20260917/revision/scripts/evidence/`:

- `est_real_sample.py:29-90`: exact compiled proof SHA256 `a2843aa36a006730c1fd3fe02a34b76afd772f1e01328162f7c0f9af88a2b6a9`, exactly two reviewed searches, full search-preimage equality. New producers cannot be enabled by a manifest edit alone.
- `est_molecule_closure.py:27-87,126-340`: normalized A11 inputs and fail-closed completeness checks. Components use member tuples, not a frozen scientific molecule-ID hash scheme.
- `est_phase_conflicts.py:33-48,65-103,114-120`: route-A observations are prevalidated phase crosschecks, with no EST/HSP/member/placement/molecule IDs; complete input scope is upstream responsibility. No miniprot parser is supplied here.
- `notes/APPROVED_HSP_PHASE_A11_B5_20260917.md`, `notes/approved_execution_20260917/A11_IMPLEMENTATION_BOUNDARY.md` and `notes/power_recovery_20260917/phase_input_inventory.md` retain authoritative A11/phase decisions and original input scouting. Current lineage/library definitions are unchanged. Stale min121 provenance does not reopen accepted C1.

## A11: actual field-to-source mapping

Example from the retained Sitalica sample, **not a normalized complete molecule**:

```text
>JK596275.1 CFGZ9178.g1 CFGZ Setaria italica Yugu1 floral spike Setaria italica cDNA clone CFGZ9178 3', mRNA sequence
report row ordinal 3: JK596275.1  648  keep  1-648  [empty categories]
```

The exact tab-delimited bytes and sequence are retained in `retained_member_sample.json`. Original FASTA header ordinal is 7; screened header ordinal is 3. Neither ordinal proves accession-history or mate completeness.

| Accepted normalized field | Real source/column or record | Concrete mapping / unresolved limit |
| --- | --- | --- |
| `uid` | Existing binder member `id`, with full `preimage` | `93e7f878d4428de707c194eafd51bb92e3aa82f5cfc97c968d3ed9b94b95ffc4`; member identity only, never molecule identity. |
| `species` | Member `preimage.species`; roles `species` | `Sitalica`. |
| `assembly` | Alignment `PROVENANCE.txt:reference` plus reference binding | `/data/gpfs/assoc/pgl/data/Transgenic/genomes/Sitalica_312_v2.fa`; sample reference SHA256 `e95c37173f26fbf8aa51723a2bf18f1bf6e1ce44129f42a57ebb904595f31ac1`. Establish the normalized assembly ID from this receipt, not species alone. |
| `dataset` | `DATASET_ROLES.est_v1.tsv:dataset` | `est/Sitalica`; a download dataset, NOT a biological library. |
| `arm` | Member `preimage.arm`; primary alignment receipt `arm` | `primary`; min121 must remain a separate support view. |
| `accession`, `original_accession` | UniVec `accession`; relationship `original_id`; original header token | Both `JK596275.1`, stem `JK596275`, parsed version 1. No claim all historical versions were collected. |
| `original_length` | Report `qlen`; relationship `original_length` | 648. |
| `original_interval` | Report `kept`, relationship `start0/end0` | `1-648` inclusive becomes `(0,648)` half-open. No suffix-only part inference. |
| `screened_sha256` | Member `screened.sequence_sha256` / `identity.sequence_sha256` | `f9292dc5afe031653a67d1ff7d1afe99a3e7434d43fa9c83b0dc9963e14fc9ae`; normalized sequence hash, NOT compressed FASTA hash `1a2941...`. |
| `provenance` | Hash-bound parent report + member preimage + header/report byte hashes | Existing parent report SHA256 `2e97691107440fd6d9e09341d3feb2588823cc28f9c9da5bf4f064a6d95728d8`. Preserve exact source references, not a free-text claim of completeness. |
| `report_id` | Report file identity + row ordinal/hash | File SHA256 `10091f5ad99fd95001743359206d92f18b57b1286416c7229de84344e4e6345c`, ordinal 3, row SHA256 `239639f354c499682b08fd5430527a21dca61085d0a8b101fd1855cb411cb5fe`. Caller may namespace this authoritative row; no invented sibling links. |
| `clone` | Original header `cDNA clone CFGZ9178 3'` | Raw clone text `CFGZ9178`, normalized `cfgz9178` using existing helper. This supports this record's clone text, not a complete clone inventory or mate edge. |
| `clone_status` | Documented identity inspection | `known` is supportable for the observed clone text once provenance is preserved. Missing library still blocks it. Do not declare `clone_unavailable` for uninspected other records. |
| `library`, `library_aliases` | No authoritative populated source found | `CFGZ` is observed header text only. Its biological-preparation identity and aliases are not proven; never replace with `est/Sitalica`, a batch, plate, clone prefix or run ID. **Blocking missing input.** |
| `bioproject` | Neither this header nor roles/provenance supplies it | Unknown; API requires a nonempty value but an explicit unknown marker is not biological completeness. No BioProject fabricated. |
| `genotype` | This original header says `Yugu1` | Per-record source annotation only. Dataset `genotype_stratum=unknown` is not contradictory; do not apply Yugu1 to all Sitalica records. Athaliana dataset `reference` explicitly is not per-accession Col-0 evidence. |
| `role` | Dataset role `c2_training_eligible`, then existing exclusion/held-out rules | Dataset role is not the API enum. A caller must derive `train_track`/`excluded_heldout`/`never` after actual membership/placement exclusions. No unconditional conversion here. |
| `support_accepted` | Complete upstream support verdict/placement inventory | Not furnished by a FASTA/report row or HSP length acceptance alone. No fabricated boolean. |

### Other required tables and unavoidable gaps

- `Placement(uid,member,contig,strand,start,end,blocks,kind,mapq)`: real source path is `evidence/est_align/Sitalica/Sitalica.est.bam`, bound by the old sample receipts; SAM RNAME/POS/CIGAR/FLAG/MAPQ and transcript-strand evidence must feed the existing geometry API. Blocks are ordered aligned blocks, not literal CIGAR strings or outer spans; strand is not automatically FLAG. No BAM opened here.
- `Relation(left,right,kind,provenance)`: versions derive from original versioned accession with all aliases retained; split relations derive from shared authoritative report row and retained intervals. `3'` or `.g1` alone does not supply another member's ID or a mate edge. Cross-arm links require explicit `parent` evidence. No populated mate/parent table located.
- `Audit(member,total_placements,secondary_placements,complete,provenance)`: current primary alignment receipt explicitly used `--secondary=no`. It cannot supply A46's all-member `--secondary=yes -N 50 -p 0.5` audit. No complete audit receipt located; ordinary BAM counts must not be substituted.
- `Exclusion(uid,kind,scope,context,contig,start,end)`: requires the existing complete global and train-context interval classes. Neither dataset roles nor limited MVP panel/test intervals establish that table. No full exclusion manifest located.
- `inventory_complete`, `identity_complete`, `exclusions_complete`: not established. A sample header or download `status=repaired,complete` attests neither molecular metadata nor all-member leakage closure. Keep diagnostic readiness blocked.

Fresh evidence/EST and two species' immediate listings contained no named library/mate/GenBank-feature table. The retained 83,394-byte `INPUT_INVENTORY_20260903_all.tsv` contains no lines matching clone/library/GenBank/.gbff/.gbk/metadata/BioSample names. This is bounded negative evidence, not a claim of global absence. It is enough to stop inventing a local mapping. Recover an authoritative per-accession feature/library record or a previously curated accession-to-preparation table before full A11. No external retrieval was performed.

## Route A: exact retained inputs and API mapping

Let `R=/data/gpfs/assoc/pgl/data/Transgenic`, `E=R/evidence`, `A=E/protein_alignments/Sitalica`.

| Required input | Exact existing object and evidence |
| --- | --- |
| Raw alignment + traceback | `A/Sitalica.protein.gff.gz`, 9,647,215,761 bytes, recorded MD5 `0e17de830f3458c81b25b3366320ec77`. No bytes read here. Old input inventory already observed embedded `##PAF`, `cg:Z`, `cs:Z`, mRNA IDs and CDS phase/Target. |
| Acceptance before ambiguity filter | `A/accept/Sitalica.alignments.tsv`, 1,096,545,609 bytes. Existing observed schema: `alignment_id,protein,organism,chrom,strand,score,identity,coverage,frameshifts,cds_blocks,verdict,reasons`. No scan here. |
| Mandatory ambiguity veto | `A/accept/Sitalica.mapping_ambiguous.txt`, 2,145,279 bytes, one protein ID per line. Too large for this task; existence/stat only. `accepted` alignment verdict alone is insufficient. |
| Current acceptance authority | `A/accept/ACCEPT_DONE` + `Sitalica.summary.json`, captured: 13,302,148 alignments; 10,384,117 accepted before downstream ambiguity removal; 141,087 ambiguous proteins; 427,602 ambiguous alignments excluded; `support_version=2`, `support_excludes_mapping_ambiguous=true`. Do not use `accept_v1_ambiguous_leak/`. |
| Reference | `R/genomes/Sitalica_312_v2.fa`; recorded MD5 `d8f77d532d8c87f314f6175146fa6728`; exact sample SHA above. Join the same reference dictionary/contig aliases as route B. |
| Protein source | `E/protein/odb12_Viridiplantae.filtered.fa.gz`; recorded MD5 `453cb32b02e0799950d7d5f4de5f62ac`, accepted historical SHA256 `fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f`. A source-file hash is NOT each protein's normalized sequence hash. |
| Actual chunk input | `E/protein/chunks_v1/MANIFEST.tsv`; captured six exact chunk paths/counts/MD5s. Chunk0 is `chunk0.faa.gz`, MD5 `e8c6980e0ca2611a7e6c721a49253299`, 2,019,181 proteins. |
| Producer and chunk output | `A/chunks/chunk0.err` + `chunk0.DONE`, captured. Actual executable `/data/gpfs/assoc/pgl/bin/conda/conda_envs/busco5/bin/miniprot`, `-t 32 --gff --outs=0.97 -N 20 A/Sitalica.mpi E/protein/chunks_v1/chunk0.faa.gz`; version `0.18-r281`, job `6153195_42`. Output `A/chunks/chunk0.gff.gz` recorded MD5 `82e8d874980be4b80bb5aa0c9e159ba5`. |
| Index provenance | `A/miniprot.index.err` records the same executable with `-t 32 -d A/Sitalica.mpi.tmp R/genomes/Sitalica_312_v2.fa`; index itself was not read. |
| Package identity | Captured `busco5/conda-meta/miniprot-0.18-h577a1d6_0.json`, recorded binary SHA256 `f8822f41eceb53a6ca611bd6606faf035aeb913030d1dede7fd7889ab37d4f52`, 118,856 bytes. This is package-record attribution, not a fresh binary hash/build proof. |
| Merge identity | `A/PROVENANCE.txt` says merged IDs `1..13302148`, six chunks; original IDs restart within each chunk. `E/protein_alignments/pilot/split_equiv/REPORT.txt` captured: zero normalized GFF differences for historical test and matching acceptance. It is chunk-equivalence evidence, not a miniprot phase/codon replay. |

For `Observation(route='A')` the adapter must supply `species`, reference SHA, canonical `contig`, zero-based `pos0`, coding `strand`, `c0` in `{0,1,2}`, **normalized protein-sequence SHA256**, and a source/alignment identity retaining raw/producer bindings. HSP/member/placement/molecule fields must remain `None`. Raw CDS column 8 is GFF phase, not directly `c0`; Target gives protein coordinates, not a full split-codon map. Retain CDS order, phase, Target and embedded cg/cs and replay against the exact reference/protein before emitting labels. Do not use coverage `Sitalica.tsv` as phase input.

The old inventory already supplies minimal candidates without new raw reads: Sitalica `MP000002`, minus scaffold_7 CDS 35312723-35312847 phase0 / 35312121-35312451 phase1, Target1-41/42-160, protein `1000413_0:000002`; plus `MP000003` on scaffold_8 is another candidate. Neither was independently proven A43-retained/ambiguity-free there. No certified terminal partial-codon example exists in these receipts. A later bounded existing-fixture extraction must include acceptance and ambiguity membership, full cg/cs and matching reference/protein subsequences, not just the visually suggestive CDS spans.

Missing route-A pieces: replay-validated adapter semantics; SHA-bound merged raw identity beyond historical MD5; per-protein sequence joins and existing per-target exclusions; complete qualifying A scope; historical producing/merge/accept wrapper identity as needed for final immutable packaging. SwissProt route-A paths remain unestablished in the inspected tree. Do not fabricate crosschecks or mark missing A as agreement. No lineage decision was reopened.

## Extra producer-proof evidence: Ptrichocarpa, exactly two searches

Chosen as a small next scope rather than repeating all nine species. Exact directory:
`E/est_blastx/remaining8_20260915/Ptrichocarpa/{orthodb,swissprot}/`.

Each directory has captured `provenance.json`, `commands.json`, `blastx.log`, `makedb.log`; output sizes were listed only. `producer_candidates.json` preserves all command arguments and exact evidence hashes. Both provenance records identify executable SHA256 `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`, DIAMOND2.1.24.178, launcher SHA256 `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291`, and query `E/est/Ptrichocarpa/univec/est.univec.fa.gz`, SHA256 `2fe8f9214c64679f2ad567c1005ae9d643fa767dc577fd737f4e252118d259ee` (89,450 queries).

| Historical receipt field | OrthoDB | SwissProt |
| --- | --- | --- |
| Filtered protein SHA256 | `f10013fee36880ce08db685b3df57a02fa57760aa3935f675522f00463dee6e9` | `37e90b9083edcff8fb407d57a5b0bb933faf81bd00d3484a12a354d30d33acea` |
| Database SHA256 | `00ee8e7c78aa802dae4750039004ec81ccc71746404d4be4793a6c7a7b0bb42f` | `be769b5ec1514c1e976e16156a2ed82bedf338e3dc6dd77a1ff71ce74d066fef` |
| Raw13 SHA256 | `6ba22830ec1259659c4d897fd1f44712c1687efdbb18eac5122224a6ca3989ea` | `3a47bb438c25d99ccd58f343c1b6315666ce5deacfa9e325480c2638439efb54` |
| Raw13 stat bytes | 234,047,656 | 116,081,449 |
| Log/provenance HSP count | 1,905,270 | 587,346 |
| Log database sequences | 12,093,803 | 41,963 |
| Actual dispatch observation | Three reference blocks; temporary output and explicit joining | One reference block; no temporary-output/joining messages in complete log |
| Candidate path for independent review | legacy blocked transcript join | legacy direct |

Both exact commands have nonzero `--frameshift 15`, raw tabular `--outfmt 6` with length ninth/BTOP thirteenth, `--max-hsps 1`, `--max-target-seqs 25`, no DAA/view, iterate/multiprocessing/global-ranking options. `makedb` path and sequence counts match each blastx database. The captured batch wrapper pins the same executable/launcher/taxonomy helper and A37 receipt hashes; it is evidence only and was **not executed**.

These candidate paths are justified by commands + complete logs + the previously reviewed static upstream attribution, not by source labels or trying both formulas on rows. No raw rows were read or evaluated. Existing masking/X truth exclusions remain intact. Whole executable/query/db/raw hashes here are **historical recorded hashes**, not new byte attestations.

**Exact blocker for a new accepted binder record:** the current full search preimage additionally requires `reference_sha256`. Ptrichocarpa's captured primary alignment receipt supplies reference path `R/genomes/Ptrichocarpa_533_v4.0.fa` and MD5 `20c6f4bb6f8b52887a2a08addedac6ea`, but not SHA256. This investigation did not recover its SHA256 and must not substitute MD5, a normalized dictionary hash or Sitalica's hash. Reuse an existing exact reference-file SHA receipt if available; do not trigger a reference scan from this inventory. A new compiled, independently reviewed producer-registry pin/schema extension is also required because the accepted resolver deliberately permits exactly two records. The old registry is untouched.

## Controller handoff and TODO status

- [x] Graft-first accepted source/proof and latest A11/HSP notes checked.
- [x] Actual bounded A11 member example and complete field-to-source/missing-field mapping retained.
- [x] Route-A paths, actual chunk command/package attribution and current acceptance boundary captured without raw reads.
- [x] Extended producer evidence by two exact historical searches, with no whole-nine-species repeat.
- [x] Captures checked for size/hash/accounting and internal command/provenance/log agreement.
- [ ] Blocked input recovery: authoritative library/preparation aliases and member-to-library mapping, mate/version/split coverage, all-member audit and complete exclusions. Do not replace with guessed molecule IDs.
- [ ] Blocked binder expansion: existing exact Ptrichocarpa reference SHA receipt plus reviewed new producer pin; controller owns implementation.
- [ ] Blocked route-A completion: real complete eligible trace fixture with reference/protein identity and replay-validated conversion. Controller owns next implementation; no new miniprot run is needed to establish that retained artifacts exist.

Graft reported approximately **1,216,886 tokens saved** across successful calls. No application tests were run because this task is an input investigation, not a code change; the accepted 200-test review is cited rather than re-claimed as this task's result.
