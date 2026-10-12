# EST genomic-phase projection: scientific contract and implementation handoff

2026-09-16; read-only research/design. No code, data, remote filesystem, or job was changed. This report is the sole requested local deliverable. No subagents were launched. HSP-length/DIAMOND traceback acceptance research belongs to the separate task and is not repeated here.

## 1. Executive finding and authority

**The coordinate transform is derivable now. The six-channel label convention and conflict scope are not fully specified in existing code.** There is no implemented route-A six-phase convention to copy: `protein_accept.py` throws away CDS phase, and its final coverage table also throws away protein identity. The existing C projector only implements strand composition and a scalar disagreement rule.

Do not report this as “all EST work awaits global A46 approval.” The current user request and `notes/b6_parallel_execution_authorization_20260916.md:3–8` scope exploratory EST audit/acceptance/projection separately from production B6. This report does not approve any execution, alter training, or freeze A46. Existing blanket `production blocked by A46` errors need a reviewed scope-specific execution contract later, **not** a bypass flag. Scientific input validation and independent review remain required.

### Sources actually examined

Paths below are relative to the named checkout, not presumed identical across branches.

- **M = current main working files**, `/mnt/f/scratch/develop/Transgenic/transgenic_local`, HEAD `9a8c8e1ba68286e845fb99be3cb76112a8bcd186`; live working-copy protocol text is authoritative for this research, not an old committed snapshot.
- **C = `/tmp/b6-p2-phase-1535433`**, verified HEAD `5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35`.
- **A = `/tmp/b6-p2-readiness-1535112`**, HEAD `ac8ac2d34c3ae403017a4a0d72e7531528926626`; consulted only to check the actual existing MVP channel/schema implementation.

| Reference | Finding |
|---|---|
| M `revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md:19` (§2.2), `:27–35` (§2.6) | Required HSP observation/provenance; six phase × strand channels; intron-mod3 recomputation; inconsistent HSPs dropped; A/B disagreement neutral; molecule counts; no reference-derived phase. |
| M `revision/protocols/PROTOCOL_B1_frozen_v1.md:953–977` | A46 remains draft; amendment governs differences from workplan. `:969` explicitly extends A30 to allow cross-species Swiss-Prot route-B input, reported separately. `:973–977` restates schema/counting/projection. |
| M `.atl/quarantine.md`, §1m, `ts:A` row | FLAG16/ts+ means genomic minus; never use ts alone as genomic strand. |
| C `revision/scripts/evidence/est_phase_project.py:1–37`; `revision/scripts/tests/test_track_c_recovery.py:242–256` | No projector/raster; FLAG×ts four-case primitive; `None` is absence, label `0` is real support; scalar A/B discrepancy returns `(None,None,True)`. Graph callers: no indexed production caller of either primitive. |
| C `revision/scripts/evidence/est_phase_accept.py:42–107,110–134` | Candidate subspans are ORIGINAL screened-query 1-based intervals, signed frame; acceptance blocked; no validated codon-event producer. This candidate is not the downstream scientific contract. |
| C `revision/scripts/evidence/protein_accept.py:62–103,106–178,234–275` | Route A accepts identity ≥.30, coverage ≥.70, fs=0, canonical introns ≤200 kb; CDS ingestion stores only `(start,end)` at `:159–160`; no phase/codon mapping. Final coverage has organism counts, not protein identity/strand/phase. `stream → main` is the graph caller. |
| C `revision/scripts/evidence/est_align.sbatch:41–63,143–176,178–210` | Nine species; screened primary BAM paths; BAM and PAF are separate alignment passes; frozen splice:hq/secondary=no/cs=long; FASTA/reference hashes and alias provenance. |
| C `revision/scripts/61_univec_trim.py:120–142` | Screened records retain forward substrings `seq[a-1:b]`; split IDs `_partN`; header/report carries original `[a,b]`. |
| C `revision/scripts/59_evidence_support.py:107–142` | Existing SAM parser emits genomic exon blocks, not a nucleotide query→reference map. Blocks include D, so cannot directly raster phase over them. |
| C `revision/scripts/67_orientation_audit.py:216–268` | FLAG×ts at `:238–240`; its annotation-agreement audit is NOT an allowable source of B6 phase. |
| A `src/transgenic/evidence/mvp_tracks.py:14–108` | Existing MVP has eight EST-only channels, no phases. Indices: 0 coverage+, 1 coverage−, 2 unknown coverage, 3 donor+, 4 acceptor+, 5 donor−, 6 acceptor−, 7 availability; RC permutation `[1,0,2,5,6,3,4,7]`. |
| A `revision/scripts/mvp/mvp_build_c0lite.py:114–168` | `est_placement` retains member/FLAG/chrom/start0/end0/CIGAR/ts/tx_strand/MAPQ; `member_block` loses query positions. MVP data is not full nine-species C0. |
| M `revision/protocols/IMPLEMENTATION_ORDER_B5_C0_C2_v1.md:27–34` | C0 table names and coordinate policy, not complete SQL definitions for the proposed HSP table. |

Graft was queried first in C and separately in A. Ranked hits were followed with targeted source reads; exhaustive indexed searches for phase/channel usage did not identify another B6 phase raster. GSF A/B/C exon-start phase tokens and old segmentation labels are **not** a B6 channel-number authority.

## 2. What can be specified without guessing

### 2.1 Coordinate spaces and exact transform

Use internal **zero-based coordinates and half-open intervals**. Preserve original raw endpoints verbatim alongside normalized fields.

1. `q0` indexes the **screened FASTA record actually given to blastx and minimap2**, of length `L`. It is not the pre-UniVec accession coordinate and not a molecule-concatenation coordinate.
2. HSP raw `qstart/qend` are one-based, directional. For coding-forward HSP, oriented start `v0=qstart-1`; for coding-reverse HSP, `v0=L-qstart`. The oriented exclusive end is respectively `qend` or `L-qend+1`. Mapping oriented nucleotide `v` back to original query is `q0=v` or `q0=L-1-v`. The separate acceptance producer must validate actual traceback and emit codon events; the projector must not reconstruct frameshifts from an endpoint/frame pair.
3. `u0` indexes the full read oriented as SAM stores/aligned it, i.e. reference-left to reference-right. `u0=q0` for FLAG16 clear; `u0=L-1-q0` for FLAG16 set. **Use FLAG, not ts, for this conversion.** If reading `pysam.query_sequence`, it is already BAM-oriented; do not reverse-complement it twice. Full original screened FASTA is authoritative when SEQ is absent or hard-clipped.
4. Walk CIGAR left-to-right, starting reference `g0=POS-1`, full oriented-query cursor `u0=0`. The mapping is piecewise affine, not exon-length interpolation:

| Op | Full-query advance | Reference advance | Phase mapping |
|---|---:|---:|---|
| M, =, X | n | n | map each `(u+i,g+i)`; X is mismatch, not missing coordinate |
| I | n | 0 | query bases have no genomic counterpart; no painted base |
| D | 0 | n | genomic bases have no EST nucleotide; all phase channels zero there |
| N | 0 | n | intron; no phase evidence in intron; query codon may span the junction |
| S | n | 0 | soft-clipped query bases not projected |
| H | n in **this full-query cursor only** | 0 | hard-clipped original bases unavailable in SEQ; full FASTA required |
| P | 0 | 0 | padding, no mapped nucleotide |

For H, do not claim SAM consumes query sequence: it does not. Full length must equal `sum(M,I,S,=,X,H)` and BAM SEQ length, when present, equals that minus H. Terminal clipping structure must validate. Nonterminal clipping, malformed ops, unknown length, out-of-range positions, inconsistent reference dictionary, or conflicting duplicate record identities fail closed. Do not copy `67_orientation_audit.introns_from_cigar` wholesale: it advances P as reference (`:193`), which is not appropriate for this exact mapper.

5. Coding genomic strand = `(FLAG reverse) XOR (HSP frame negative)`; sign is minus when XOR is true. Separately the transcript strand inferred from splice motifs is `genomic_strand(FLAG,ts)`. Therefore agreement requires **HSP sign == ts** when ts is known. A FLAG16 negative HSP is coding genomic plus. Missing ts does not destroy a protein-derived coding direction; report motif orientation unavailable. Proposal: known ts/HSP contradiction quarantines that HSP-placement phase observation rather than overriding either source. This exclusion policy needs an explicit local contract because the frozen text does not prescribe it.
6. Complement sequence letters only when translating/checking coding-oriented sequence: reverse HSP uses RC(original screened EST); reverse genomic coding strand uses complemented genomic bases in decreasing `g`. Coordinates reverse, but the meaning “first/second/third base of this codon” does **not** complement or swap 1↔2.
7. Preserve source provenance: if screened piece is original inclusive `[a,b]`, original accession position is `a-1+q0` (zero-based). **Do not add this offset when joining BAM to HSP:** both already use screened-piece coordinates. `_part1` from different screening arms is not sufficient identity; verify sequence/hash/source interval.

### 2.2 Separate codon position, GFF phase, and genomic frame anchor

These are three different quantities:

- **Proposed channel label `c`:** codon-base ordinal in coding direction, 0 first, 1 second, 2 third. A real codon emits `0,1,2`, including a codon split by an intron.
- **GFF CDS phase `p`:** number of bases from that feature's coding-oriented start to the next complete codon. If genuinely standard GFF phase, the first base's ordinal is `(-p) mod 3`, not `p`; subsequent bases have `c=(d-p) mod 3`. Need a miniprot fixture verifying its emitted feature phase and partial/split-codon conventions. Never reset every CDS block to zero, and never discard the leading p bases: they may finish the upstream codon.
- **Diagnostic genomic anchor `a`:** within a gap-free reference run, `a=(g-c) mod3` on plus, `a=(g+c) mod3` on minus. It is constant on that run, unlike `c`.

**Recommended local declaration:** use `c` for the six channels, in order `[+c0,+c1,+c2,-c0,-c1,-c2]`; these are six-channel-local indices 0–5, **not** an assignment of final B6 K-axis indices. No existing B6 numeric index contract exists. Do not append phases to the fixed eight-channel MVP or modify its checkpoint.

Reason: frozen RC says reverse coordinates/swap strand channels, with no phase permutation. Codon ordinal obeys that rule exactly. A chromosome-mod3 or tile-mod3 anchor generally needs an offset-dependent relabelling on RC and changes if contig/tile origin changes. Thus `c` best reconciles the frozen requirements, but the protocol never explicitly says “phase0 = first base”; record the declaration instead of claiming it was already implemented.

### 2.3 What “intron length mod 3” changes

The EST is spliced: N consumes no query nucleotide. **The biological codon ordinal is carried through the splice, not advanced by the intron length.** The genomic frame anchor changes.

With no indels, crossing an intron of length `n` in coding order gives:

- plus: `a_after=(a_before+n) mod3`;
- minus: `a_after=(a_before-n) mod3`.

Equivalent implementation: map each labelled EST nucleotide through CIGAR, then calculate `a` as a diagnostic; this automatically makes the correct intron correction. Do not both map the query and add `n mod3` to `c` (double correction). At a plus run boundary, a deletion contributes `+D` and insertion `−I` to the anchor; at a minus coding-order boundary their signs reverse. Frameshift-defined new codon anchors come from acceptance events, not from this arithmetic. Whole-codon indels do not rotate `c`; single-nucleotide indels can rotate the local genomic anchor. A target insertion/deletion and a query/protein alignment gap are different events.

## 3. Route A and conflict contract

### 3.1 Route A must be rebuilt from retained raw alignments, not coverage TSV

Reuse accepted-alignment/A43 ambiguity decisions, but read original miniprot GFF/PAF/traceback with phase, strand, protein ID, subject coordinates and parent alignment identity retained. Existing `protein_accept.py` is an organism-support summarizer and is lossy by design. Its final `<species>.tsv` cannot supply six channels. A GFF phase per block can establish local codon position after its tool convention is verified; detailed same-subject-residue comparison also requires alignment traceback/Target mapping, including protein gaps and split codons.

Use the **same protein sequence identity**, not “same organism” or equal-looking accession across releases. Cross-species and excluded-protein filters must apply to A as well as B. Swiss-Prot route-A alignments/acceptance cannot be presumed present; the current `organism_of` only understands OrthoDB-style IDs. Existing OrthoDB alignments may contain a training species' own proteins; per-target-species exclusion is still mandatory.

### 3.2 Two distinct comparisons; do not conflate them

1. **Phase comparison required by the workplan:** same species/contig/genomic base/coding strand and same normalized cross-species protein; compare normalized `c`. This is the existing primitive's declared base/strand/protein-pair domain. Equal `c` is phase agreement, not proof of the same residue or correct structure.
2. **Same-codon/anchor validation:** each route's codon must have an explicit translated-codon anchor, and three bases are numbered in coding order. When both routes map to the same protein residue, compare `(protein sequence hash, subject aa index, genomic codon base tuple)`; a split codon is an ordered tuple, not a contiguous `[start,start+3)`. This provides stronger replay validation. Same subject residue with different codon placements, or different residues at the same base, must be reported as structural correspondence disagreement/unassessable—not silently counted as same-codon agreement.

**Important open choice:** requiring identical subject residue as a prerequisite for *all* phase comparisons is stricter than the frozen “same proteins … disagree on a base” rule; it could hide actual disagreement. Recommend retain the required base/strand/protein phase comparison and record same-codon correspondence separately. Do not adopt same-residue-only masking without a recorded change. No need for new DIAMOND length research to decide this.

### 3.3 Inconsistent HSPs versus legitimate frame changes

- Different signed frame magnitudes before/after a frameshift or splice are not by themselves inconsistency; compare the labels **after exact projection**.
- Proposal following “HSPs … are dropped”: on the same screened EST member and same genomic placement, find overlaps where two accepted HSPs assign different coding strand or `c` to a genomic base; reject **both entire participating HSPs**, not just a winning bitscore or conflicting base. Evaluate conflicts before removing any HSP so input order does not determine the survivors. All directly conflicting HSPs are dropped; unrelated HSPs remain. Log IDs and conflict intervals. No need to infer conflict from nonoverlapping spans with no shared anchor.
- One HSP mapping a reused nucleotide to two codon ordinals is a self-conflict; conservative proposed result is whole-HSP rejection. A verified +1 frameshift with an unconsumed nucleotide is not automatically a conflict. The separate traceback contract must explicitly describe repeated/omitted nucleotides.
- Frozen wording says **one EST**, not automatically every member of its clone-merged molecule. Use member-level HSP rejection; afterwards deduplicate by A11 molecule. Proposal for incompatible members at a shared genomic base: mark that molecule/base unresolved, emit no phase count there and report it. Do not choose an accession version or concatenate split parts as if they were adjacent.
- Primary placements are the minimum supported initial geometry source; supplementary/secondary placements must still be retained for exclusion/QC, never erased from leakage checks. Whether supplementary geometry can emit phase requires an explicit policy; default proposed exploratory phase output uses primary only, recorded in its manifest. Do not borrow the scorer's MAPQ cutoff without authority.

These scope choices are genuinely underspecified; they are not resolved by `reconcile_phases`' scalar unit test.

### 3.4 A/B disagreement is numeric zero, not the phase-0 channel

For a comparable event, `(A=1,B=2)` becomes `(None,None)` and a QC event. `(A=0,B=0)` remains genuine c0 support. `(A=None,B=0)` retains B c0 and is `A_missing`, not agreement and not disagreement. Opposite-strand alignments must be reported separately, not paired merely because coordinates overlap.

**Unresolved masking scope:** “where A and B disagree on a base, both phase channels are set to 0” can mean (i) remove only the disagreeing pair's contributions before molecule counts or (ii) veto the affected final phase cells at that base even if another protein/molecule agrees. These differ for multiple supports. Existing primitive establishes only the scalar case. Recommend (i), observation-level neutralization plus explicit counts and independent support retained, but require controller's recorded choice; do not silently claim global final-cell zeroing. A is a cross-check, not an extra counted EST molecule or a seventh phase source.

Counts are distinct retained EST molecules at `(base,strand,c)`, then `log1p(n)` in float16; repeated HSPs/proteins/members of a molecule do not multiply n. The 64 cap is expressly on transcribed coverage, **not** expressly on phase counts; do not copy MVP coverage clipping into phase. Keep source-specific OrthoDB/Swiss contributions and overlap counts; union molecules for any combined track rather than adding source counts.

### 3.5 Frozen versus missing versus contradictory

| Item | Resolution |
|---|---|
| FLAG×ts, screened-query identity, CIGAR operation geometry | Derivable; implement exact transformations above. |
| Six labels, molecule counts, no reference phase, A/B neutralization | Explicit existing workplan requirements. |
| Swiss-Prot forbidden by old A30 | Not a remaining contradiction: governing A46 draft `:969` explicitly extends it for route-B input. Do not impose a new OrthoDB-only input restriction. Keep sensitivity/QC separated. |
| Intron correction versus unchanged codon ordinal | Compatible once genomic anchor and ordinal are distinguished. Adding intron length to ordinal is wrong. |
| “phase0” meaning and numeric order | No actual B6 source contract; recommended declaration above, motivated by frozen RC. |
| Global anchor labels plus RC strand-swap-only | Actual semantic incompatibility unless extra origin-dependent relabelling is specified. Avoid that representation. |
| Whole-HSP inconsistency, cross-member conflicts, ts contradictions, A/B masking scope | Genuine unresolved policy details; concrete recommendations above need recording. |
| HSP ≥30-aa definition/BTOP semantics | Separate owner; projector accepts only that owner's reviewed codon-event contract. |
| Exploratory EST execution vs A46 production freeze | Distinct scopes. Current scoped authorization is not global training authorization; blanket code error is not the last word on user authority. |

## 4. Hand-computed fixtures (executable expectations for the proposed c convention)

All genomes/queries below are synthetic; coordinates `g,q` are zero-based; unspecified channels are exactly zero. `Z` means the all-zero six-vector; `E_j` means `log(2)` in local channel j and zero elsewhere (one molecule), converted to float16. Geometry fixtures intentionally use short HSPs; they bypass **no** acceptance threshold because they test the projector on already-labelled synthetic events. Full integration fixtures must add complete codons to meet the separately agreed ≥30-aa rule without changing these local anchors.

### F1. Plus multiexon, intron lengths 0/1/2 modulo 3

`L=12`, FLAG0, ts+, SAM POS101, CIGAR `2S4M{n}N5M1S`, HSP original qstart3/qend11/frame+3, codon labels q2..10 = `012012012`. Introns n=3,4,5 are positive lengths (no illegal 0N).

| n | First exon g100..103 channels | Intron (Z) | Second exon positions and channels | Anchor first→second |
|---:|---|---|---|---|
| 3 | `0,1,2,0` | 104..106 | g107..111: `1,2,0,1,2` | 1→1 |
| 4 | `0,1,2,0` | 104..107 | g108..112: `1,2,0,1,2` | 1→2 |
| 5 | `0,1,2,0` | 104..108 | g109..113: `1,2,0,1,2` | 1→0 |

The codon starting at q5 has genomic tuple `(103,104+n,105+n)` with c=`0,1,2`. Its first base is before the intron; never discard it or reset the second exon to 0. GFF phases, if conventional, are 0 then 2. Soft clips q0,1,11 never paint.

### F2. Minus multiexon, same three intron classes

Same L/POS/CIGAR, FLAG16, ts+, HSP original qstart2/qend10/frame+2. Original q1..9 labels `012012012`, coding direction runs from high genome to low. In ascending genomic order:

- first/low block g100..103 channels `5,4,3,5` (c=`2,1,0,2`);
- second/high block g(104+n)..g(108+n) channels `4,3,5,4,3` (c=`1,0,2,1,0`);
- intron Z.

Coding-first high-block anchor a is 0,1,2 for n=3,4,5 respectively; coding-second low-block anchor always 0. Thus coding-order anchor changes by `−n mod3`. Split codon tuple is `(105+n,104+n,103)`, c=`0,1,2`. Conventional GFF phases in coding order are 0 (high block of length5), then 1 (low block); ascending-file order is the reverse. These are not c labels.

### F3. All four FLAG/ts cases (n=4)

Same CIGAR and L. Endpoints below are raw one-based and deliberately directional.

| FLAG | ts | HSP qstart→qend/frame | Coding genome | low-block channels | high-block channels |
|---:|:---:|---|:---:|---|---|
| 0 | + | 3→11 / +3 | + | 0,1,2,0 | 1,2,0,1,2 |
| 0 | − | 11→3 / −2 | − | 5,4,3,5 | 4,3,5,4,3 |
| 16 | + | 2→10 / +2 | − | 5,4,3,5 | 4,3,5,4,3 |
| 16 | − | 10→2 / −3 | + | 0,1,2,0 | 1,2,0,1,2 |

For each, set ts to opposite HSP sign: expected proposed disposition `strand_conflict`, Z for that HSP-placement. Remove ts: geometry/channels unchanged, `motif_strand_unavailable` QC. Neither test consults gene annotation.

### F4. Insertions, deletions, hard clips and padding

Plus, POS101, `L=9`, complete coding q0..8.

- `3M1I5M`: g100..102 channels `0,1,2`; q3(c0) has no genome; g103..107 channels `1,2,0,1,2`. No synthetic base is inserted in the genome track.
- `3M1D6M`: g100..102 `0,1,2`; g103 Z; g104..109 `0,1,2,0,1,2`. Do not fill deletion from an exon-block span.
- `3M1P6M`: g100..108 `012012012`, P advances neither coordinate.
- F1 with `2H4M4N5M1H`, full FASTA L12 and SEQ length9: exactly F1 n4 labels; alignment sequence cursor differs from full original-query cursor. Missing full-length identity is a refusal, not a guessed L9 reverse map.
- Reverse/RC counterpart of each fixture must preserve c while reversing genomic coordinates/swapping channels (see F8).

### F5. Protein/query gaps and frameshifts supplied as codon events

These test the projector interface, **not DIAMOND symbol semantics**.

- Protein deletion relative to query: query codons q0..2→subject aa0, q3..5→aa2; aa1 has no query codon. On `6M`, channels `012012`; never allocate three imaginary query nucleotides to aa1.
- Query insertion relative to protein: codons q0..2→aa0, q3..5→subject NULL, q6..8→aa1. Proposed span policy paints `012012012` on `9M`, since the translated query-insertion codon is inside the accepted HSP and has a reading frame; its same-subject-codon comparison is unavailable. Acceptance owner must expose this event and controller must explicitly adopt this support policy rather than silently dropping it.
- Verified +1 displacement example: codons `(q0,q1,q2)` and `(q4,q5,q6)`. On `7M`, channels `0,1,2,Z,0,1,2`. q3 is unconsumed by any translated codon. A first-span signed frame extended blindly would incorrectly assign q4 c1.
- Verified −1 displacement example: codons `(q0,q1,q2)` and `(q2,q3,q4)`. q2 is both c2 and c0; proposed self-conflict rule drops entire HSP, yielding Z throughout and one self-conflict record. This rule is a conservative policy choice, not a claim that all negative frameshift alignments are invalid.
- Nonoverlapping frameshift-derived codons with different anchors are permitted; an intron between their bases adds no further query shift.

### F6. HSP and route-A conflicts; counts

- H1 plus labels q0..8=`012012012`; H2 plus anchor q3 gives q3..8=`012012`: agreement; same member/molecule counts once, value log2.
- H2 instead anchor q4 gives q4..9=`012012`: q4 H1=c1 vs H2=c0. Proposed whole-HSP conflict rejection drops H1 and H2, including nonoverlapping flanks. A third HSP with no conflicting overlap survives. Permuting input order changes nothing.
- Same molecule, two identical protein HSPs: log2, not log3. Two distinct compatible molecules: log3. A member-level disagreement after clone merging follows the explicitly chosen cross-member policy, not arbitrary row order.
- Same comparable protein/base/strand: A=c1, B=c2 → Z for that pair and one disagreement; A=c0, B=c0 → E0 (plus) or E3 (minus); A absent, B=c0 → same support, no agreement claim.
- Add independent molecule M2 with agreeing c2 support to the first case. Proposed pair-local neutralization yields E2 from M2; global-cell-veto alternative yields Z. **This fixture is the mandatory decision discriminator before implementation.**
- Same base/protein but different subject aa indices: record phase outcome and `different_subject_residue`; do not report same-codon agreement merely because c matches.

### F7. Boundaries and source offsets

- F1 n4 mapped coding span ends at g113 exclusive; g112 included, g113 Z. HSP start q2 is included; q1 is clipped. Intersection with tile `[102,110)` yields at g102..109 channels `2,0,Z,Z,Z,Z,1,2`; clipped-out positions cannot alter anchor.
- One mapped nucleotide carrying c2 gives E2, not an empty interval. HSP wholly in soft clips gives zero projected bases with an explicit disposition.
- UniVec piece original `[11,22]`, L12: F1 q2 maps original zero-based12, but still genomic100, **not110**. Another split piece sharing molecule ID deduplicates only overlapping genomic support; no inferred adjacency across removed vector sequence.
- An HSP/codon exactly at exon end retains its mapped bases; nothing is painted on the intron boundary unless it is actually an aligned exonic base. No donor/acceptor label substitution.

### F8. Reverse-complement and reference invariance

On tile `[100,114)` (L14), F1 n4 has forward local nonzeros at 0..3=`0120`, 8..12=`12012`. RC must have minus-channel indices at local 1..5=`5,4,3,5,4`, 10..13=`3,5,4,3`, all other positions Z. Apply RC twice: byte-identical float16 array. Keep biological c; do not map c→2−c. Shift the entire genome placement by one, or change tile offset while preserving genomic overlap: c labels remain unchanged, diagnostic a shifts. This catches accidental `g % 3`/tile-relative phase labelling.

## 5. Minimum interface and files (proposal, no implementation here)

Avoid a new C0 framework or trainer changes. Build pure transforms and an immutable diagnostic observation producer in the C branch.

1. Extend **`revision/scripts/evidence/est_phase_project.py`** with:
   - `query_reference_runs(placement, query_length)` → validated affine M/=/X runs plus unmapped-query/reference reasons; accepts full CIGAR, FLAG, POS, contig length and source identity.
   - `project_codons(hsp, codon_events, placement)` → phase spans/events plus QC. Consume validated accepted-HSP events; never call the candidate decomposition as if approved.
   - `resolve_hsp_conflicts(events)` and base/protein reconciliation using the selected scopes, retaining raw HSP IDs and dispositions.
   - Route-A adapter reading raw accepted miniprot alignments; reuse acceptance predicates/A43 exclusions where valid, not their lossy output table. Keep parsing separate from geometry in functions; split into a second module only if size warrants it.
2. Add **`revision/scripts/tests/test_est_phase_project.py`** implementing F1–F8 and failures before code; retain established FLAG/neutral-zero tests. Public refusal tests must be deliberately updated only after the scoped contract is recorded, not simply deleted to make a run succeed.
3. Add a small **versioned contract/input manifest** under the existing evidence-script conventions (path chosen by controller) recording c meaning, six local indices, conflict policies, accepted-HSP contract hash and execution scope. CLI wrapper should require this exact manifest and hash-bound inputs; fresh exclusive output directory, no overwrite, failure artifacts preserved. No arbitrary approval boolean.

### Required accepted-HSP boundary

Per HSP: species, source, raw-file SHA and row identity, member/qseqid, molecule ID, query sequence SHA/L, protein ID+sequence SHA/species, raw qstart/qend/frame/sstart/send, scores, acceptance-contract SHA, provenance reference. Each **codon event** supplies HSP ID, ordinal/event ID, ordered three original-query positions, optional subject aa index, and event class (match/mismatch/query insertion/etc.). Omitted/reused query positions must be explicit and endpoint-validated. A compact affine run encoding is fine; full per-base Python objects for millions of ESTs are not required.

Current candidate `subspan(q_start,q_end,signed_frame,subspan_aa)` alone loses same-subject-codon correspondence and reasons for gaps; require the producer to retain it. Do not reimplement acceptance thresholds in projection.

### Output/storage shape

Preserve workplan `est_protein_hsp` fields; add keys rather than overloading `frame` with genomic phase. One HSP parent row plus a projected-segment child relation (or equivalent nested field):

- keys: HSP ID, member ID, molecule ID, placement ID, source/protein identity;
- geometry: contig, start0,end0,strand, `c_at_start0`, phase step `+1`/`−1`, query mapping run/event reference;
- status/reason, route-A match IDs, per-event disagreement/correspondence records;
- inherited library, genotype stratum, role, provenance hashes on rows or enforced linked immutable provenance keys.

Split segments at CIGAR gaps, HSP events/frame changes, exclusions and conflict intervals. A constant `phase` field on an interval must explicitly mean its start ordinal plus step, **not constant per-base c**. Same-codon records retain ordered genomic tuples when needed. Distinct molecule aggregation and observation→tile membership remain downstream; diagnostic projection alone is not training-ready C0.

## 6. Inputs required before nine-species execution

Nine species exactly as C `est_blastx.py:33–40`: Athaliana, Bdistachyon, Gmax, Osativa, Ppatens, Ptrichocarpa, Sbicolor, Sitalica, Vvinifera. Eighteen primary species×protein-source units; primary ESTs ≥100nt, paired ≥121nt reporting separately, never sensitivity-arm input tracks.

For each species/source bind and verify:

1. C1 audit receipt/manifests and original raw13 hashes; separate accepted-HSP output+contract+producer/code hashes, with raw/accepted/rejected counts reconciling. Existing DONE is not accepted-phase evidence. Do not resubmit blastx merely to obtain projection.
2. Exact screened FASTA `evidence/est/SP/univec/est.univec.fa.gz`, hash and QNAME/length/sequence identity; UniVec report/source intervals, summary/DONE and A37 primary provenance. For ≥121 sensitivity, inspect that arm's own screened records and BAM; filtering primary by length is only equivalent if IDs/sequences/source intervals are verified identical. `_partN` may renumber when short pieces disappear.
3. `evidence/est_align/SP/SP.est.bam`, BAI, BAM header/reference dictionary, `PROVENANCE.txt`, DONE and alias table from the frozen driver; SHA256 freshly bound (old recorded MD5 is additional evidence). Retained PAF is an independently computed pass; never splice its geometry onto a BAM record by QNAME alone. Use BAM as the one mapping authority for this implementation.
4. Reference FASTA/FAI, canonical contig alias bijection and lengths, species manifest and hash; sequence is used for replay/QC, **not reference annotation phase**. Assembly alias collisions or mismatches refuse.
5. A11 member→clone/molecule and accession-version/mate/split-part closure with metadata/library/genotype/roles. MVP's Athaliana-specific c0lite alone is insufficient for all nine species/full A11 closure.
6. Raw route-A miniprot GFF with embedded PAF/traceback, exact tool/command/protein-source hashes, accepted alignment IDs and mapping_ambiguous inventory. Verify per-species same-species/descendant exclusion and protein-query leakage closure against the same normalized sequences used by B. **Remote existence/usable phase/Swiss route-A coverage was not verified in this read-only local task.** Missing A is explicit `not_crosschecked`, not automatically disagreement or PASS.
7. Frozen protein FASTAs, taxonomy/lineage snapshot and same-species exclusion provenance for both sources; OrthoDB organism support and protein-sequence leakage unit must not be conflated. Keep Swiss-Prot separately attributable as required by A46's explicit A30 extension.
8. For eventual training eligibility: full exclusion intervals and placement-audit closure, per-source roles, protected panels/test/valid orthogroups, masks and observation membership. These are **not waived by exploratory projection authorization**. Exploratory output should say `not_training_eligible` until this downstream gate is met.
9. Fresh-output location, execution owner, resource estimate from bounded fixture/pilot, exact script/input/output hashes and live Slurm duplicate check immediately before future submission. No remote verification or submission was done here.

QC denominators must include all inputs and dispositions: HSPs received/accepted upstream/projectable/conflicting/retained; original-query codon bases mapped/clipped/inserted/reused; reference D/N bases unpainted; motif strand missing/conflicting; A present/comparable/missing, phase disagreement and same-codon correspondence; distinct molecules, source overlaps, both A37 reporting arms, per-species/source reconciliation. Preserve `.extern` OOM accounting as unresolved independently of payload success.

## 7. Controller-ready next actions

1. Record a short **local projection contract**, not a global A46 freeze: adopt c0-first-base and local order; choose primary/supplementary emission, ts conflict, query-insertion support, whole-HSP/self-conflict and cross-member policy; settle pair-local versus final-cell veto using F6. Declare base-phase comparison versus same-codon validation as separate outputs.
2. Obtain the separate acceptance owner's validated codon-event interface; leave HSP-length semantics there. Verify a small real miniprot +/− split-codon/partial-CDS fixture to bind GFF phase and Target/traceback conversion. No reference GFF-derived phase.
3. Implement pure projector plus F1–F8 in C, independent review, then a bounded end-to-end synthetic/real alignment fixture. Schema/provenance checks and fresh-output failure handling are part of that review. Do not change MVP K=8/training code.
4. Read-only inventory of the nine species' immutable inputs, especially route-A raw detail and member closure; missing inputs get explicit dispositions. Only then specify reviewed scoped EST execution jobs in a new namespace; live Slurm check first. Scope remains exploratory EST projection, not production B6 or global A46 approval.

**Bottom line:** there is enough to implement the exact query→genome transform without guessing. It is not honest to say the existing source already defines phase0, route-A indices, or multi-support conflict scope. The small explicit contract above closes those gaps and supplies hand-computed discriminating tests rather than hiding policy choices in code.
