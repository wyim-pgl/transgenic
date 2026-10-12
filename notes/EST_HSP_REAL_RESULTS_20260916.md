# Actual expanded DIAMOND HSP result review — 2026-09-16

> **2026-09-17 author decision:** [approval record](APPROVED_HSP_PHASE_A11_B5_20260917.md) now explicitly selects **whole DIAMOND-reported HSP alignment length≥30 (M+I+D, F excluded)**. Historical statements below requiring that choice are superseded for this unit definition only. Original144-row diagnostic status/12non-elicited cases and artifact bytes remain unchanged; this decision does not certify all trace coverage, projection, lineage/A11 closure or A46 production. Bounded contract implementation is tracked in the approval record.

**Overall status: INCONCLUSIVE.** Job **6165462** supplies strong, specific evidence for column-9 serialization and oriented DNA cursor semantics. It does **not** certify the complete predeclared trace matrix, approve the author's biological 30-aa minimum, open C2/C3, or authorize production/A46. Keep `certified=false`, `production_authorized=false`, `acceptance_policy=UNRESOLVED`.

This review analyzes existing local results only. No engine execution, source modification, threshold change, job submission, remote action, or subagent was performed. Only this report was written. The original authority is [EST_HSP_CONTRACT_20260916_1530.md](EST_HSP_CONTRACT_20260916_1530.md); prior code review is [hsp_expanded.independent_review.md](est_execution_20260916_1530/hsp_expanded.independent_review.md). Advisor consultation failed with a client/model-version error; no advisor approval is claimed.

## 1. Evidence identity and execution status

All result paths below are relative to `notes/est_execution_20260916_1530/hsp_expanded_result/`. Independently recomputed SHA256:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| expanded.raw17.tsv | 40338 | `7530085dc8796ca431127a6804b9ca421656fa1ca92abcc9c698779990cb579c` |
| raw-first.tsv | 40338 | `7530085dc8796ca431127a6804b9ca421656fa1ca92abcc9c698779990cb579c` |
| queries.fa | 48084 | `b34bcc4c6a6f60299f7557a74d2d7fd4a22a35d4a2f40d2709c78ea738dd46f6` |
| proteins.fa | 1834 | `003f0978a1cc163500aa85e3021a49698f7ed460f631e829b4cd8bfc4589e7ad` |
| manifest.json | 235624 | `d35f0343d5a1b99c8dfec9ba59279572e133fb321e92fcb827ffa755484b711d` |
| diagnostic.json | 1285078 | `99fb4acaac0a46712d8723f608a39755926151e85edbf3dd99173d76bd1fbfcb` |
| commands.json | 1446 | `72f9d6073b4f9e7ca6df63d926955354101d160f78ec5e430bc28c0fc91e6893` |
| provenance.json | 1111 | `63bccdc600b7e72dee24be7d68e5efca65a989cfe6b26b4fd0367e854ee42d10` |

Both raw hashes match `raw-first.json` and `diagnostic.json`; FASTA hashes match provenance. Every raw row hash (UTF-8 row without newline) matches its diagnostic record. All 144 query sequences and all 24 target sequences match the manifest. Independently applying every sequential edit ledger to baseline DNA reconstructs the actual oriented FASTA sequence; standard-code translation of each baseline target region matches its protein.

The verify-path receipt is actually `notes/est_execution_20260916_1530/hsp_expanded_verify_path.txt`, resolving to `/tmp/est-hsp-expanded-verify.NwyrGr`. Source inspection used that archive and `git show`, not the changing C worktree. Reviewed code commit is `8dcd5806662e251f04ccd95d9a95c93fd0c7de30`; implementation source commit is `a98f5c0cc08e6f6a5ac27905d71a4478e9a0e789`. Their runtime import-closure bytes are identical. The later commit changes only report/wrapper material relative to the implementation commit.

| Runtime file under revision/scripts/ | SHA256 independently matched against verify tree, staged tar, both commits, provenance and wrapper |
|---|---|
| evidence/est_hsp_expanded.py | `aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4` |
| evidence/est_blastx.py | `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291` |
| 64_orthodb_filter.py | `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302` |

Staged `hsp_expanded_code.tar` SHA256 is `8a115e073b4df14d98004a08a835ddca0f6c7afa5d158c94a6a70d1c9c36534a`. Submission receipt reports tar/wrapper checksum checks OK and `submitted=6165462`. Compute receipt reports all three Python pins and DIAMOND executable hash check OK. Provenance records DIAMOND `2.1.24`, build banner `2.1.24.178`, SHA256 `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`. The immutable runner also checks that binary digest before generating inputs (`est_hsp_expanded.py:275–298`). This is consistent deployed-pin evidence, **not a fresh independent local hash of the remote executable**, a hermetic Python-runtime attestation, or proof against post-execution receipt modification.

Recorded command uses one synthetic database, two threads, `--frameshift 15 --evalue 1e-5 --min-orf 30 --max-target-seqs 25 --max-hsps 1 --no-parse-seqids`, and the contract's 17 fields. No relaxed flags appear. This is one search, not independent biological replication across six orientations.

`hsp_expanded_status.txt` observed at 16:01:24 PDT reports allocation and batch `COMPLETED 0:0`, elapsed **1 s**, batch MaxRSS **1548K**, payload `batch.exitcode=0`; extern is **OUT_OF_MEMORY 0:125**, elapsed 1 s, MaxRSS **1480K**. These small sampled RSS values neither explain nor refute the extern OOM. Payload completion and full artifacts do not make scheduler execution clean. The unrelated 6165461 running row is not this diagnostic. No fresh Slurm observation was made here. `DONE` literally contains `INCONCLUSIVE\nNOT CERTIFICATION\n`; it is diagnostic completion, not success of every intended trace. Runtime/time/full engine logs were not included in the local result copy and were not independently reviewed here.

## 2. Independent raw/DNA verification

Local read-only Python calculations did **not import or call** `interpret`, `replay`, `candidate_decompose_hsp`, or `diagnose`. They:

1. Read FASTAs, manifest and original raw17; counted operations directly from the two gapped strings: paired residues (including substitutions) = M, query-only residue = I, subject-only residue = D, slash/backslash = F.
2. Built a standard genetic-code table independently using NCBI's conventional TCAG codon order; translated actual consumed DNA codons, not the manifest's supplied code table or diagnostic codons.
3. Oriented reverse queries by reverse complement. For positive rows, start/end cursor = `qstart−1`, `qend`; for negative rows, `qlen−qstart`, `qlen−qend+1`. M consumes 3 nt/1 subject residue, I 3 nt/0, D 0/1; `\\` moves +1 oriented nt and `/` −1, on either strand.
4. Checked each translated codon against the raw query residue, each subject residue against actual protein, concatenated translated query against field 17, lengths, endpoints, initial signed frame, expected subject, and evalue ≤1e-5. Reconstructed CIGAR by run-length encoding operations and BTOP by exact-match runs plus residue/operation pairs; all match fields 13–14.
5. Compared independently obtained counts, shift coordinates, blocks, endpoints and initial frames to the manifest's declared elicitation criteria, then checked agreement with diagnostic statuses. Replayed all shifted rows with both cursor shift signs inverted as a negative control.

**Observed:** 144 distinct queries, exactly one 17-field row per query, 24 subjects; **132 ELICITED, 12 NOT_ELICITED, 0 INVALID**. No missing query, clipped row, extra hit, or off-target subject explains the twelve exceptions. All 144 emitted traces pass DNA/protein/endpoints verification, including those twelve. There are 24 rows in each initial signed frame `+1,+2,+3,−1,−2,−3`.

Direct aggregate counts: **M=9294, I=30, D=30, F=96**; **9324 actual query codons translated and checked**. Ninety rows contain shifts (84 with one, six canceling rows with two); wrong-sign replay fails for **all 90**. The other **54 have no F**, so no shift-sign discrimination is claimed for them. These are sequence-specific negative controls, not a theorem that wrong-sign replay must fail for arbitrary DNA.

Every row has `column9 = M+I+D`. The same raw counts distinguish it from `M+I+D+F`, `M+I`, M, and maximum block size across this panel. Sum of query-consuming blocks equals `M+I` by construction and remains algebraically redundant. All six offset/strand copies within a family have identical BTOP, CIGAR, gapped sequences, translated sequence and column 9; coordinates/frames transform with the input.

## 3. Actual family observations

Each table row represents **six actual rows**, offsets 0/1/2 on both strands. Blocks count query residues between shifts, not paired-only columns. M includes mismatches.

| Family | M / I / D / F | Actual blocks | Col9 | Intended trace elicited? |
|---|---|---|---:|---|
| exact29 | 29/0/0/0 | 29 | 29 | 6/6 |
| exact30 | 30/0/0/0 | 30 | 30 | 6/6 |
| exact31 | 31/0/0/0 | 31 | 31 | 6/6 |
| D30 | 29/0/1/0 | 29 | 30 | 6/6 |
| D31 | 30/0/1/0 | 30 | 31 | 6/6 |
| I29 | 29/1/0/0 | 30 | 30 | 6/6 |
| I30 | 30/1/0/0 | 31 | 31 | 6/6 |
| short29_1 | 29/0/0/1 | 14,15 | 29 | 6/6 |
| short29_-1 | 29/0/0/1 | 14,15 | 29 | 6/6 |
| short30_1 | 30/0/0/1 | 14,16 | 30 | 6/6 |
| short30_-1 | 30/0/0/1 | **15,15** | 30 | **0/6** |
| short31_1 | 31/0/0/1 | 14,17 | 31 | 6/6 |
| short31_-1 | 31/0/0/1 | 14,17 | 31 | 6/6 |
| blocks29_29_1 | 58/0/0/1 | 29,29 | 58 | 6/6 |
| blocks29_29_-1 | 58/0/0/1 | 29,29 | 58 | 6/6 |
| blocks29_30_1 | 59/0/0/1 | 29,30 | 59 | 6/6 |
| blocks29_30_-1 | 59/0/0/1 | **30,29** | 59 | **0/6** |
| blocks29_31_1 | 60/0/0/1 | 29,31 | 60 | 6/6 |
| blocks29_31_-1 | 60/0/0/1 | 29,31 | 60 | 6/6 |
| long_plus | 180/0/0/1 | 90,90 | 180 | 6/6 |
| long_minus | 180/0/0/1 | 90,90 | 180 | 6/6 |
| long_I | 180/3/0/0 | 183 | 183 | 6/6 |
| long_D | 177/0/3/0 | 177 | 180 | 6/6 |
| cancel | 90/0/0/2 | 29,30,31 | 90 | 6/6 |

Ordinary boundary DNA is genuine: the 29-aa exact HSP is emitted despite `--min-orf 30`, inside independent stop-free flanks (e.g. exact29_o0_fwd qlen207, qstart61, qend147, evalue1.29e-24). It is not a missing-hit surrogate for a threshold negative. Conditional on choosing col9≥30, ordinary boundaries are fail/pass/pass in all six frames.

Critical distinctions are directly observed:

- D30: 29 actual query codons, 30 alignment columns. Length-based pass differs from query-residue fail.
- I29: 29 paired residues plus one query insertion, 30 query codons/columns. Query-residue pass differs from paired-residue fail.
- Both short29 shift signs: 29 actual codons plus one marker, **col9=29**, not 30. F is excluded even where inclusion would change the threshold outcome.
- Both shifted 30/31 totals are emitted and translate correctly on all six frames; the minus-30 **trace placement** remains non-elicited, not its total-length observation.
- 29+29 and 29+31 shift-delimited blocks are elicited for both signs/strands/all offsets. The desired 29+30 order is elicited for +1; −1 emits 30+29 instead. This distinction must not be hidden by reporting only unordered block counts.

For all 144 observed rows, applying hypothetical ≥30 rules would retain: **col9/M+I+D 126**, **M+I+D+F 138**, **M+I 120**, **M 114**, **maximum block 84**, **every block 54**. These are diagnostic counterfactual counts, not an accepted production set. In particular the 12 non-elicited rows remain in these totals. They are not discarded to improve coverage.

## 4. Exact twelve non-elicited traces

The only exceptions are `short30_-1_o{0,1,2}_{fwd,rev}` and `blocks29_30_-1_o{0,1,2}_{fwd,rev}`. For each family, actual counts, subject, full target coverage, oriented start/end and initial frame equal the ledger expectation. **Only the shift location and resulting ordered block sizes differ.** The shift is delayed by one paired residue, i.e. **+3 oriented nt**, on both strands. The following coordinates are zero-based oriented positions for offset 0; add the offset for offsets 1/2. Negative-strand raw coordinates transform, but the oriented DNA/codons are identical.

### short30_-1

Ledger deletes **T at 102**. Expected shift is `/` at 102, blocks **14+16**; actual shift is at **105**, blocks **15+15**. Actual full row spans qstart61..qend149 on the forward query (qlen209), subject1..30, col9=30, evalue1.25e-22. Actual BTOP `14AC/-15`, CIGAR `15M1/15M`.

Local edited DNA, positions96..110: `ATGTACGCAGCCACC`.

- Previous codon 99..101: `TAC` = Y.
- Intended early `/`: cursor102→101; `CGC` at101 = **R**, versus subject residue15 **C**; next codon104 `AGC` = S.
- Actual path: `GCA` at102 = **A**, versus subject15 **C**; then `/` moves105→104; `AGC` = S. Downstream alignment is restored at the same coordinate.

Thus the expected trace itself entails an R/C substitution, whereas the actual trace entails A/C. It is **not** a missing query or absence of a −1 shift. The engine chose a different locally valid substitution/shift placement.

### blocks29_30_-1

Ledger deletes **A at147**. Expected `/` at147, blocks **29+30**; actual `/` at **150**, blocks **30+29**. Forward qstart61..qend236 (qlen296), subject1..59, col9=59, evalue5.51e-44. Actual BTOP `29AS/-29`, CIGAR `30M1/29M`.

Local edited DNA, positions141..155: `AACGTGGCAACGGCC`.

- Previous codon144 `GTG` = V.
- Intended early `/`: cursor147→146; `GGC` at146 = **G**, versus subject residue30 **S**; next codon149 `AAC` = N.
- Actual path: `GCA` at147 = **A**, versus subject30 **S**; then `/` moves150→149; `AAC` = N. Again downstream alignment converges immediately.

These DNA facts explain why edit location is not automatically the traceback shift location. Under standard BLOSUM62, A/C scores 0 versus R/C −3, and A/S scores 1 versus G/S 0, consistent with preferring the observed placement when all other local operations are unchanged. This is a scoring explanation, **not an independently rerun DP optimality proof or fresh attestation of the binary's internal scoring table**. The decisive observation is the emitted path and its direct codon support. Do not change expected fixtures after the fact to label these twelve ELICITED.

## 5. Canceling shifts are genuinely local frame changes

For `cancel_o0_fwd`, independently observed shifts are **+1 at147** and **−1 at238**, blocks **29,30,31**, CIGAR `29M1\\30M1/31M`, BTOP `29\\-30/-HN30`, col9=90. Actual nearby codons:

- at144 `CAG` = Q; first shift147→148; at148 `AAC` = N; at151 `TAC` = Y;
- at235 `TGC` = C; second shift238→237; at237 `CAC` = H (subject residue60 N); at240 `GTG` = V.

There are 90 translated residues and **two** F markers despite net displacement zero. Start60/end330 consume a net270 nt; the local frame changes +1→+2→+1. Across offsets, positive frame paths are (+1,+2,+1), (+2,+3,+2), (+3,+1,+3); reverse paths have the corresponding negative signs. Flipping both cursor shifts yields a first post-shift codon `GCA` = A at146 instead of N and is rejected in all six copies. Net endpoint equality alone would have missed this error. These data validate local query-codon reconstruction for this panel, not genomic phase rasterization across a spliced mapping.

## 6. Supported claim, remaining limits, smallest next step

**Supported empirical claim:** For these 144 actual rows from hash-pinned DIAMOND 2.1.24.178 under the frozen flags, reported HSP length is the number of paired columns (matches and substitutions) plus query insertions plus query deletions, **excluding frameshift markers**: `length=M+I+D`. Ordinary 29/30/31 boundaries, I/D discriminators, both shift directions on both strands/all initial frames, shifted totals at 29/30/31, and canceling shifts have actual raw/DNA support. This is stronger than two parsers agreeing and stronger than the prior long-only fixture; it remains evidence about this engine/panel, not a universal theorem about all outputs or versions.

**Uncertified:** exact −1 trace placement for short30 and ordered blocks29+30; complete predeclared coverage; arbitrary sequences/ambiguous codons/stops/multiple gaps or shifts outside this panel; universal wrong-sign rejection; genomic projection through BAM/CIGAR/splicing and skipped/reused-base rasterization; biological appropriateness of the minimum; production acceptance implementation and author/A46 authorization. The unrelated extern OOM remains unresolved. Offsets/RC copies share biological sequence and are coordinate tests, not six independent biological replications. A family marked ELICITED checks the manifest's declared criteria, not necessarily every ordinary-indel location: for example D31 emits `13M1D17M` although its ledger deletion is after residue14, illustrating possible local alignment placement ambiguity even without a shift.

### Smallest next step

**For deciding column-9 serialization, no further engine run is necessary on account of these twelve rows.** They are redundant for that narrow question: each retains the same M/I/D/F totals and hence every candidate whole-HSP length statistic; their actual emitted paths are independently DNA-valid. Other fully elicited rows already separate F inclusion/exclusion, I versus paired-only count, D versus query-residue count, and whole-HSP versus segment thresholds. Even here, short30's 14+16→15+15 leaves every segment below30, and blocks29+30→30+29 preserves the max/every-block/qualifying-block-count threshold results. What changes is **which local segment and coordinates** qualify. Keep the global INCONCLUSIVE status and retain all twelve failures; do not rerun merely to obtain a green report.

The immediate next step is **author confirmation of the proposed whole-HSP rule below**, with explicit acknowledgement that this result establishes serialization, not the author's intended biological unit. This recommendation does not itself provide that confirmation or release C2/C3.

If exact trace-matrix completion or ordered segment-coordinate certification is required, there is a genuine, narrower gap: the prescribed −1 placement for short30 and 29+30. Add **two newly named, prospectively fixed −1 fixtures × three offsets × both strands (12 queries)**, preserving old failures and all flags. Design the local synonymous-codon context so the base reused from the preceding codon equals the deleted first base of the following codon: the intended early shift then reconstructs the intended amino acid, while delayed unshifted translation yields a different, less favorable residue. Verify that property directly from DNA, plus independent stop-free flanks and nonrepetitive anchors, before any engine run. If synonymous choices cannot satisfy it, use a newly declared synthetic local peptide rather than relabeling the old case. Specify expected shift coordinates, blocks, thresholds and failure handling in advance; retain every result, including non-elicitation. This is a proposed fixture, not an implemented/tested design or permission to submit. No relaxed flags, adaptive engine-output search, increased resources, replacement of failed cases, or cherry-picking is warranted.

### Recommended frozen HSP definition — REQUIRES AUTHOR CONFIRMATION

> The acceptance unit is one whole engine-reported DIAMOND HSP. Subject to the separately frozen evalue and input/trace validation requirements, the minimum is **DIAMOND-reported HSP alignment length ≥30** (raw column9). In the pinned 2.1.24.178 result, this length counts paired match/substitution columns plus query-insertion and query-deletion columns (M+I+D), and excludes slash/backslash frameshift markers. It is not a requirement for 30 ungapped query residues, 30 paired residues, or 30 residues in every constant-frame segment. Constant-frame decomposition is used separately for phase-location analysis and introduces no additional segment-length filter without explicit approval.

If the author instead wants a query-codon or paired-residue minimum, freeze that different policy explicitly; do not silently equate it to column9. Do not assign initial qframe to an entire shifted HSP or promote this whole-HSP definition into approval of projected genomic base/phase handling. Public acceptance/production gates remain closed pending their separate requirements.
