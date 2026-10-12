# Bounded EST failure diagnosis

## Scope and status

Local retained-evidence analysis of job 6165875 only. Original outputs remain unchanged. Allocation and batch ended 0:0; extern OUT_OF_MEMORY remains unresolved. Nothing here establishes clean accounting, biological correctness, phase/raster eligibility, training eligibility, full C0, route A, or A11 completion.

Completed: reproduce all 462 HSP dispositions and all 385 attempted geometry results; inspect retained original/screened DNA, protein and SAM; test both frameshift directions; distinguish explicit X from canonical source disagreement; write executable reproducer and machine-readable results. Blocked: authoritative external DIAMOND producer semantics and masking provenance are not established by these retained records. No production fix or acceptance promotion was performed.

Inputs: `../est_result/raw/worker/sample/diagnostics.jsonl`, SHA256 `8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d`, and the reviewed Python source under `/tmp/est-real-source-binding-20260917`. `../est_result/RESULT_REVIEW.md` supplies the terminal/accounting context. No original raw search file, BAM, or source FASTA was rescanned; no DIAMOND, job submission, commit, or cluster access was performed. An advisor consultation failed due to client/model incompatibility and supplied no evidence or approval.

Run from this directory:

```sh
./diagnose.py > analysis.json
```

The script uses stdlib plus the existing reviewed Python evidence modules, disables bytecode writes, reads only local retained data/source, and prints JSON. It verifies retained byte/sequence hashes, reproduces existing dispositions without altering code, independently instruments BTOP replay, reproduces geometry against retained SAM, and includes minimal synthetic assertions. `analysis.json` records exact raw13 rows, IDs, hashes, every discrepancy, endpoints, shifts, geometry overlaps and source hashes. It is a diagnostic artifact, not a replacement worker report.

## Results in brief

| Original first-failure class | HSPs | Diagnosis |
|---|---:|---|
| HSP_RULE_ACCEPTED | 385 | Reproduced unchanged; five still fail geometry |
| query codon truth | 30 | Every first discrepancy is BTOP query X versus a canonical DNA translation |
| subject truth | 18 | Every first discrepancy is BTOP subject X versus a canonical retained protein residue |
| reported length must equal M+I+D | 29 | Exact residue replay and endpoints succeed; all are OrthoDB, whose retained rows count F in reported length |
| geometry failure | 5 | Subset of the 385 accepted HSPs; real one-base codon self-overlap in JK608593.1, correctly withheld |

The 48 X-related HSPs are not disjoint by underlying error side: 27 have query-only discrepancies, 11 subject-only, 10 both. Their QNAME counts are JK570678.1: 28, GW787538.1: 12, JK608595.1: 4, JK608594.1: 4. Query-first is JK570678.1: 18 and GW787538.1: 12; subject-first is JK570678.1: 10, JK608595.1: 4, JK608594.1: 4. Across exhaustive diagnostic replay there are 437 query-X and 290 subject-X discrepancies, plus five numeric-match-run canonical disagreements in three HSPs. These are column counts, not extra failed HSPs.

All 16 screened sequences equal their retained original-sequence relationship slices. The 462 replayed paths consume both reported endpoints exactly under the existing frameshift cursor convention. No unsupported token, endpoint failure, or demonstrated strand-sign parser bug explains these failure classes.

## 1. Frameshift interpretation is supported, not the root parser bug

Existing code interprets `\\-` as a +1 nucleotide cursor step and `/-` as -1 in translation-oriented query coordinates, with no subject consumption. A shift is not a translated amino-acid column. All 46 HSPs containing shifts consume both endpoints under this convention. Reversing the signs introduces residue errors and query endpoint failure in every one of the 46. This is evidence for the convention on this retained sample, not independent proof of all DIAMOND versions.

Executable minimal examples in `selftests()`:

- DNA `ATGCAAAGCT`, protein `MKA`, BTOP `1\\-2`: translate ATG, skip C, translate AAA and GCT; 3 residue columns, 1 shift column, 10 query bases.
- DNA `ATGAAGCT`, protein `MEA`, BTOP `1/-2`: translate ATG, reuse its final G in GAA, then GCT; 3 residue columns, 1 shift column, 8 query bases.

The second example necessarily reuses a query base. Changing the sign to avoid overlap would corrupt the sequence/endpoints and is not a fix.

Reviewed source: `revision/scripts/evidence/est_hsp_contract.py:76-126`, `est_hsp_expanded.py:49-126`, `est_phase_accept.py:35-39`. The parser consumes two-character BTOP tokens correctly here. The fixture/source documentation is project-local interpretation, not retained upstream DIAMOND source/help establishing universal format semantics.

## 2. Reported length is source-dependent in this evidence

The unconditional validator equality in `est_hsp_contract.py:41-43` is contradicted by exact retained OrthoDB rows:

| Source | Rows | With shifts | Empirical equality on every row |
|---|---:|---:|---|
| OrthoDB | 400 | 41 | reported = M+I+D+F |
| SwissProt | 62 | 5 | reported = M+I+D |

Without F, both formulas coincide. Of the 41 OrthoDB shift rows, 29 reach the length validator after exact residue replay and 12 fail earlier on X. Among the 29, 28 have one shift and one has two. Therefore 29 is a first-failure count, not the total number of incompatible OrthoDB length assertions. Overall, 421/462 match MID and 457/462 match MIDF. A global change to either equality cannot describe both retained sources.

Minimal actual counterexample:

- HSP `c4fc0beb23a18f150f22f696baadea4e9dc6ac433a47f8f02ece408b3d98cc35`
- OrthoDB ordinal 136, `GW787538.1` against `13427_0:00a9d0`
- qstart/qend 1/131, subject 101/144, frame +1, BTOP `8/-36`, reported length 45.
- Exact canonical replay: M=44, I=D=0, F=1; consumed query 3*44-1=131, subject 44; no residue errors. Reported 45 is MIDF, not MID.

Contrast: SwissProt `JK608593.1` against `sp|O23813|SIR_MAIZE`, HSP `34eb4bea38aa69501b69e5f1a5fc8dae89a0f41d3520cbc86841e650cf0eb8d1`, has M=144, F=1 and reported length 144. Its 431 query bases equal 3*144-1.

Diagnosis: a proven overgeneralized validator assumption, not malformed BTOP on these 29 rows. The empirical relation is established; why the two producer outputs differ is unverified. No upstream DIAMOND source/help was located in the reviewed checkout or retained-help filename search, and no external lookup was made. Do not infer a version, option, bug, or canonical DIAMOND specification from this difference alone.

Minimal future code boundary: replace the universal equality with a separately pinned producer-format validation contract only after establishing that contract. Keep `accepted = int(row[8]) >= 30` unchanged. Do not use query-residue count, block length, corrected length, `reported in {MID, MIDF}`, or a source-name-only heuristic to admit evidence. No unconditional one-line production fix is justified across both sources today. Contract names/docstrings claiming universally reported MID also require revision if a producer-bound fix is authorized.

## 3. X is loss of traceback residue information, not proof of bad source DNA

First query example:

- HSP `235ededa3ef63f796e98255ee412aa0321a603a9e4b2eed11a49759c7046411f`, OrthoDB ordinal 52.
- `JK570678.1` against `4556_0:00067e`.
- Oriented query offset 354 (zero-based): DNA `ACG` translates T; literal BTOP token `XT` says query X, subject T.
- Following tokens include `XS`, `XN`, `XY`; they continue to disagree with canonical retained translations. The first mismatch is not a frameshift, unknown DNA base, or swapped sequence orientation.

First subject example:

- HSP `34757e4ce022c6f5d8c43ff84c30eccc74e317beaaa8e848536384f9bfe6e6c2`, OrthoDB ordinal 53.
- `JK570678.1` against `38727_0:000876`.
- Subject offset 135 (zero-based): retained protein E, literal token `EX` says subject X; surrounding retained residues include E/A/E/E/E/D/S/L while BTOP substitutes X on that side.

These X-rich stretches are consistent with producer masking or an internal ambiguous-residue representation. They are not raw DNA ambiguity in these examples, since the relevant retained codons are canonical. The retained paths alone do not establish which masking mechanism or producer option caused the Xs. Calling this a source-file mismatch or silently treating X as a wildcard would both overclaim. Exact traceback-residue validation is honestly unassessable at these columns under the current strict contract; failures must remain withheld. Better diagnostic specificity is reasonable, but not acceptance.

### Numeric matches can conceal the same uncertainty

The adapter expands each numeric BTOP run by copying the retained subject residue and requiring the query codon to equal it (`est_hsp_contract.py:112-120`). That is not independent knowledge of an unmasked traceback residue. Three subject-first HSPs contain five additional disagreements inside numeric runs:

| HSP | OrthoDB ordinal / subject | Query offset | DNA translation / inferred subject |
|---|---|---|---|
| `34757e4ce022c6f5d8c43ff84c30eccc74e317beaaa8e848536384f9bfe6e6c2` | 53 / `38727_0:000876` | 366, 375 | T/K, H/Q |
| `2e6fcb67117cde1497537f9633b08b4b17f8fc67e30609426af601017c993744` | 58 / `158149_0:0040ea` | 366 | T/A |
| `ebe08824303d41bcfca44b8a562e4aa4d60ce53bb4428e96f55741f50081343d` | 64 / `4540_0:00978c` | 354, 360 | T/A, N/K |

All concern JK570678.1. Every one of these query positions is explicitly X in at least 21 other retained HSPs, including ordinal 52. This strongly supports loss of unmasked equality information in the numeric-run region. The canonical source residues genuinely differ at these five aligned positions, but that does not establish that the wrong source sequence was loaded: a numeric match could represent equality of masked symbols. No explicit non-X mismatch token contradicts its retained canonical residue anywhere in this exhaustive replay. Do not assert either proven source corruption or verified reconstruction of hidden X/X matches.

Minimal executable demonstrations separately cover literal `XT` with ACG/T, literal `EX` with GAA/E, and numeric `1` with ACG/K. Only the last creates a canonical disagreement from an inferred equality; none grants eligibility.

## 4. Geometry failures are honest exact-phase conflicts

All five failures reproduce exactly, including `self_overlap_policy_unresolved`. Retained SAM and full-query identity pass `query_reference_runs`. Placement:

- ID `b6c9273b9500baa5eda9e44d2884a530c90b87c0e35c618a736c360a2357cf22`
- JK608593.1, flag 16, scaffold_3, pos1 14200500, ts `-`
- CIGAR `31M1D69M93N117M117N159M1011N339M1S`
- SAM SHA256 `787263fbbac58c2a0e87e719d21ed1b6d4eceaf183258202aea59033d82ace62`

Every failed HSP starts at qstart 714, frame -3, with `/-` after nine translated residues. Cursor 29 steps back to 28 in oriented coordinates. Original-query codon events 8 and 9 occupy half-open intervals [687,690) and [685,688), reusing original base 687. Event 8 assigns ordinal 2 at that base; event 9 assigns ordinal 0. This is an exact phase conflict, not a harmless duplicate segment. The self-overlap check runs before reference projection (`est_phase_project.py:106-167`); the nearby SAM 1D is not the rejection cause.

Exact affected HSPs:

| Subject | HSP ID |
|---|---|
| `sp|O23813|SIR_MAIZE` | `34eb4bea38aa69501b69e5f1a5fc8dae89a0f41d3520cbc86841e650cf0eb8d1` |
| `sp|O82802|SIR1_TOBAC` | `58b314c684d77655fca9bcd6c7c8bfd2da2c037cdbb72334da2c86ea5489335c` |
| `sp|Q75NZ0|SIR_PEA` | `67453932cb8762dafe443d0a07c0dc4003ed7cfa643cadcb1ab9225788e0403d` |
| `sp|Q9LZ66|SIR_ARATH` | `704ce44ee3470f3fd908c74918417bbf05b20887371b491447d84667bd150c01` |
| `sp|Q9AWB2|SIR_SOYBN` | `cb28a36b05198ee871ad0162464b166d719105a9367c70db149f3fd6a6757e92` |

No geometry code fix is indicated. Do not deduplicate, discard the offending codon, flip a shift sign, or choose a preferred phase to increase counts. The 380 successful geometry outputs remain diagnostic properties only.

## Tests needed before any future source change

1. Producer-bound length fixtures for the actual OrthoDB MIDF and SwissProt MID examples, including both shift signs, reverse strand, multiple shifts, I/D and no-F controls. An unrecognized producer remains unassessable. Independently bind producer version/command/output format before changing validation.
2. Preserve reported-length threshold boundary tests at 29/30/31; never substitute MID, query residues, per-frame blocks, or post hoc repaired length.
3. Distinct diagnostics for query-X, subject-X, numeric-run source disagreement, explicit canonical literal mismatch and endpoint errors. Test both sides X and multi-error rows; preserve original first-failure output and do not count inferred masked equality as exact truth.
4. Retain the actual JK608593.1 negative geometry tests, verifying overlap base 687 and conflicting labels 2 versus 0; test reverse orientation and show a nearby SAM deletion is not causal.
5. Keep identity, sequence hash, terminal coordinates, exact phase/conflict policy and all scientific gates unchanged. Re-run the full contract suite only in a separately authorized source-fix task.

No failures were reclassified as accepted. The only proven validator defect is the universal reported-length equality assumption; ambiguity/masking provenance and external format semantics remain unresolved. Extern OOM remains unresolved independently of these local findings.
