# EST HSP contract review — 2026-09-16 15:30 namespace

**Disposition: diagnostic-only launch is technically supportable with the bounded wrapper below, after controller staging/review. The existing sbatch is NOT suitable unchanged. No job was submitted and no remote files were written. C2 acceptance and C3 projection remain blocked. This is not production approval.**

## Scope and evidence

Reviewed `/tmp/b6-p2-phase-1535433`, HEAD `5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35`, via graft then exact source. The worktree has unrelated `.gitignore`, `.ignore`, `.testdeps/` changes; stage a commit archive, NOT a recursive copy of that working directory. Relevant source hashes measured locally:

| File (under revision/scripts/) | SHA256 |
|---|---|
| evidence/est_phase_accept_fixture.py | `8da86ae1bc4323da4efb24dffd177d060479f958e431b7e79722a88b57d80f36` |
| evidence/est_phase_accept.py | `0c6490c76d63463d945ee270f46a4ab8b96e929b57f4ac38dc52229f24c11ae3` |
| evidence/est_blastx_fixture.py | `18da9abd2a51a870e327226c8a1f9d225711a7c4551de40fd620c59470dbbad2` |
| evidence/est_blastx.py | `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291` |
| 64_orthodb_filter.py | `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302` |

Authority: `notes/b6_three_track_discovery_synthesis_20260916.md`, Track C; `revision/protocols/B6_EVIDENCE_MODE_WORKPLAN_v1.md` §2 item 2 (`est_protein_hsp`), §2 item 6 (HSP ≥30 aa and phase), §4 item 6 (A46). Historical pilot and remaining-eight wrappers both pin DIAMOND SHA256 `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d` at `/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond`. Do not upgrade it in response to its age warning.

Live READ-ONLY SSH checks this review: that executable reports `diamond version 2.1.24`; help banner reports `diamond v2.1.24.178`; `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python --version` reports Python 3.11.14. Target fixture namespace was absent. These are availability/version observations, NOT a fresh engine hash, dependency execution test, scheduler receipt, or semantic validation. No large login-node hashes. Historical job 6163396 receipts were not reread; no claim here depends on them.

## What the current fixture actually tests

`est_blastx_fixture.py:39-48,51-107,110-173` creates a deterministic 180-aa protein and six DNA queries: exact forward, exact reverse complement, +1 nt, −1 nt, +9 nt insertion, −9 nt deletion. The shifts/indels are forward only, near the middle. Each source database contains one retained copy after synthetic taxonomy exclusion; both source labels exercise filtering, not two biologically independent searches.

`est_blastx.py:199-206,275-290` runs real makedb and blastx with:

```
--frameshift 15 --evalue 1e-5 --min-orf 30 --max-target-seqs 25
--max-hsps 1 --no-parse-seqids --threads 2 --tmpdir <private scratch>
--outfmt 6 qseqid qlen sseqid slen qstart qend sstart send length evalue bitscore qframe btop cigar qseq_gapped sseq_gapped qseq_translated
```

There is no production FASTA/DB input, download, job submission, or call to an acceptance entrypoint in this execution path. Two tiny searches plus getseq are sequential. `engine_version` (230-246) enforces the .178 build banner; `execute` (249-272) propagates failures and kills the child process group on interruption. `work_directory` (63-74) exclusively creates outputs and writes FAILED or DONE. SIGKILL/OOM can prevent markers; missing markers are incomplete, never passing. A nested engine DONE does not mean the outer pin report completed.

The base fixture checks input-codon translation, protein residues, endpoints, initial signed frame, all three sequence fields, CIGAR and BTOP, plus net edit and bridge anchors. These checks supply genuine evidence beyond two parsers agreeing: fixed input DNA and codon translation constrain the cursor hypothesis. However the CIGAR, BTOP and rich sequence fields share the engine's traceback and are not independent biological truth.

`est_phase_accept_fixture.py:45-90` compares column 9 to gapped-string length including/excluding shifts, translated-query residue count, maximum frame subspan, and sum of frame subspans. It first requires the old replay and candidate decomposition to agree. If they fail, raw17/logs remain, but no complete pin report is produced. This is useful falsification, not an unbiased selector among all possible semantics.

### Certification blockers

1. No 29/30/31 boundary HSPs; all intended alignments are roughly 180 aa. No negative threshold cases or accepted/rejected-set check. `--min-orf 30` is not an HSP-length acceptance filter.
2. No reverse-strand shifted queries, no start-offset coverage of all six signed frames, no shift cancellation/control for multiple shifts. An ordinary reverse match cannot validate reverse frameshift projection.
3. B (`ungapped_query_aa`) and C sum are algebraically identical in this implementation: every query-consuming residue enters exactly one subspan. They can never be distinguished by more examples. C max is a different statistic; neither C sum nor C max fully expresses a per-subspan acceptance policy.
4. The candidate B description says “match+mismatch, gaps excluded”, but the implementation also counts query insertions (q != '-' and s == '-'), as does `qseq_translated`. Clarify query-residue count versus paired non-gap residue count before choosing a rule.
5. The current output does not report separate paired, I, D, shift counts or per-subspan threshold decisions. `len(subspans)==shifts+1` additionally assumes every shift-delimited block is nonempty; this is a fixture-specific precondition, not a universal BTOP law.
6. Selecting a unique column-9 reading identifies engine serialization, NOT what the author intended by “HSP ≥30 aa.” Engine behavior and policy approval are separate questions. An INCONCLUSIVE result can reflect duplicate candidate definitions, not a failed engine.
7. The stock sbatch requests 4 CPU/8 GB/1 h, uses ambient `python3`, checks only the extension's hash, does not pin the binary hash or imported modules, and documents the old routeB namespace. Replace the wrapper for this diagnostic; leave reviewed Python immutable.

## Independent upstream grounding and proposed operational rule

Read directly from upstream tag v2.1.24 (GitHub tree/commit response SHA `23d27d1bd2104b2475a229341f42a825587dbfc2`):

- `https://raw.githubusercontent.com/bbuchfink/diamond/v2.1.24/src/output/blast_tab_format.cpp`: line 72 calls `length` “Alignment length”; 312-314 emits `r.length()`; 375 onward emits BTOP from traceback operations; 518-528 emits one gapped character per operation; 575-582 builds translated query excluding deletion and both frameshift operations, thus including query insertions.
- `.../src/basic/match.h`: 126-145 advances query only for I, both sequences for match/substitution, and uses shift_forward/back for shifts; 338-358 prints backslash for forward shift, slash for reverse shift, subject '-' for either; 400-401 returns stored HSP length.
- `.../src/basic/translated_position.h`: 90-106 changes frame offset with carry/borrow; 144-159 defines oriented position = offset + 3×translated and reverse absolute position = dna_len − oriented − 1. Consequently backslash advances the oriented nucleotide cursor +1, slash −1 on BOTH strands; the sign in original genomic/query coordinates reverses on the reverse strand.

These upstream observations independently support the replay's cursor convention. They do not prove the installed binary's identity (compute hash required), and this review did NOT locate/verify the DP accumulator that populates stored HSP length. Whether shift markers contribute to column 9 therefore remains an empirical question for real RICH17, not a fact inferred from gapped-string length.

**Proposed policy to submit for semantic/author review:** preserve the spec's unit, a whole engine-reported HSP, and accept on its reported `length >= 30` (raw13 column 9), together with the frozen evalue and validation requirements. Label it precisely “DIAMOND-reported HSP alignment length ≥30”, NOT “≥30 ungapped translated residues” or “every constant-frame segment ≥30”. This is the most direct interpretation of the named HSP and the engine's named alignment-length field; the spec never adds a second threshold on each frame segment. Before freezing it, explicitly document inclusion/exclusion of I, D and shift columns from the real-engine result and obtain author confirmation that this is the desired biological minimum. If the desired minimum is 30 query codons or 30 paired residues, that is a different policy requiring explicit wording, not a silent correction.

Decomposition is a separate phase-location task. Do not assign one initial qframe to an entire shifted HSP, or claim the proposed HSP rule approves how indel/shift bases are rasterised. Candidate subspan coordinates and handling of reused/skipped nucleotides still require C3 review. Public `accept_stream` and `run` in `est_phase_accept.py` must remain unconditional blockers.

## Minimal discriminating real-engine fixture plan (not implemented here)

Keep raw output FIRST, then interpret it with an independent scorer that does not import `candidate_decompose_hsp` or `replay`. Generate nonrepetitive proteins/DNA with a written edit ledger and codon table; inspect all expected hits before applying policy. Predeclare the desired trace shapes but never fabricate them: a clipped/split/missing HSP is “not elicited / inconclusive”, not a negative threshold result. Use independent long stop-free flanks so a 29-aa local HSP can exist inside an ORF longer than 30; otherwise `--min-orf 30` confounds the negative. Preserve the frozen production flags; any relaxed discovery run must be separately named and cannot certify production behavior.

Minimal coverage (each critical shape also reverse-complemented):

| Family | Intended observable distinction |
|---|---|
| Ordinary exact HSPs at 29, 30, 31 aa | Expected threshold decisions fail/pass/pass, conditional on actual emitted length and evalue; pad with nonhomologous stop-free context, not extra homologous residues. |
| 29 query residues with one internal query deletion relative to protein: 30 paired-or-gap columns | Alignment length can pass while ungapped query residues fail. Use flanking anchors so the single D is actually observed. |
| 29 paired residues plus one query insertion | Query-residue count 30 vs paired-residue count 29; distinguishes the misleading B wording. Include 31-column variant if a shift-count boundary requires it. |
| One +1 and one −1 nt edit, both query strands | Backslash/slash cursor sign and shifted-codon truth; require reads spanning the edit. Include a long positive anchor version if short HSP scores do not elicit a shift. |
| Shift-delimited blocks 29+29, 29+30, 29+31 | Whole-HSP length/total query residues pass, but a “each block ≥30” or “retain qualifying blocks” policy differs. Report per-block counts, not C sum. |
| A shifted alignment with 29 translated residues plus one marker; companion 30 and 31 residue cases | Tests whether marker inclusion can cross the threshold. Actual emitted column-9 value decides the observed boundary, not the generator's name. |
| Two opposite shifts, nonzero blocks between | Net shift zero is NOT no shift; test both strand orientations and retain each local frame. |

The ordinary/shifted queries should include 0/1/2 leading-base offsets (or equivalent cases) to exercise ±1/±2/±3 initial qframes. A reverse-complement pair need not have numerically identical qstart/qend; check coordinate transforms against sequence length. Long +3-aa insertion/−3-aa deletion controls already present can remain as robust engine diagnostics alongside the small boundary cases.

For every real 17-field row, independently record: raw row and FASTA hashes; reported col9; number of paired columns M (match + mismatch), query insertion residues I, query deletions D, shift count F; `M+I+D`, `M+I+D+F`, query count `M+I`, paired count M; all constant-frame block query counts; endpoints, signed initial and each post-shift frame; reconstructed nucleotide codons/residues; and pass/fail under EACH proposed threshold. Replay raw DNA on its oriented strand, translate each consumed codon, and compare to qseq_translated and protein residues, not just to the other parser. For ordinary gaps require consumption I=(3 nt,0 aa), D=(0 nt,1 aa); shifts consume (+1 or −1 nt,0 aa), not a codon. Explicitly test the wrong shift sign/strand interpretation against that DNA truth; it must fail on at least one discriminating case. A37 ≥100-nt screening can be respected through flanking padding, without changing local HSP length.

## Exact bounded diagnostic wrapper for CURRENT immutable fixture

Controller prerequisites (NOT executed here): recheck live queue/accounting to avoid duplicate diagnostics; review this note; create namespace below exclusively, then stage a git archive of the exact commit at `code/` with its `revision/scripts/` layout. Verify the five hashes against the archive and make the staged code read-only. Do not stage `.testdeps`, a venv, or mutable worktree links. Stage the wrapper as `diagnostic.sbatch` and `logs/` beneath the same namespace. All staging/submission belongs to the controller. The wrapper intentionally refuses reuse. No production inputs or historical outputs are read by the job.

2 CPUs, 4 GB, 30 minutes is appropriate for twelve tiny query/source combinations and two one-protein databases. This is a hard diagnostic budget, not measured runtime/RSS or a success guarantee; an OOM is a failed diagnostic requiring review, not permission to increase limits automatically. No GPU, array, network, or production scan. Compute-node hashing of the pinned binary is permitted here.

```bash
#!/bin/bash
#SBATCH --job-name=est_hsp_diag_1530
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/fixture/logs/slurm-%j.out
set -euo pipefail
umask 077
R=/data/gpfs/assoc/pgl/data/Transgenic
N=$R/evidence/est_followup_20260916_1530/fixture
W=$N/code
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
DIAMOND=/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond
export PATH=/usr/bin:/bin
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 2 ]]
[[ -x "$PY" && -x "$DIAMOND" && -d "$W" && ! -L "$W" ]]
[[ "$(readlink -f "$N")" == "$N" && "$(readlink -f "$W")" == "$W" ]]
[[ ! -e "$N/result" && ! -L "$N/result" ]]
mkdir "$N/execution.lock"
trap 'rc=$?; printf "%s\n" "$rc" > "$N/batch.exitcode"' EXIT
mkdir "$N/tmp"
export TMPDIR=$N/tmp
cd "$W"
# Import closure, not just the top-level extension. Hash on compute only.
printf '%s\n' \
 '8da86ae1bc4323da4efb24dffd177d060479f958e431b7e79722a88b57d80f36  revision/scripts/evidence/est_phase_accept_fixture.py' \
 '0c6490c76d63463d945ee270f46a4ab8b96e929b57f4ac38dc52229f24c11ae3  revision/scripts/evidence/est_phase_accept.py' \
 '18da9abd2a51a870e327226c8a1f9d225711a7c4551de40fd620c59470dbbad2  revision/scripts/evidence/est_blastx_fixture.py' \
 '5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291  revision/scripts/evidence/est_blastx.py' \
 '32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302  revision/scripts/64_orthodb_filter.py' \
 "0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d  $DIAMOND" \
 | sha256sum -c - > "$N/hash-check.txt"
{
 date -u; hostname; printf 'job=%s\n' "$SLURM_JOB_ID"
 printf 'commit=5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35\n'
 printf 'python=%s\ndiamond=%s\nthreads=2\n' "$PY" "$DIAMOND"
 "$PY" -I -B --version
 "$DIAMOND" version
 "$DIAMOND" help
} > "$N/runtime.txt" 2>&1
/usr/bin/time -v -o "$N/time.txt" \
 "$PY" -I -B "$W/revision/scripts/evidence/est_phase_accept_fixture.py" \
 --diamond "$DIAMOND" --out-dir "$N/result" --tmpdir "$N/tmp" --threads 2
```

Controller submission, only AFTER staging/review, using the staged wrapper (not the stock sbatch):

```bash
sbatch --export=NONE /data/gpfs/assoc/pgl/data/Transgenic/evidence/est_followup_20260916_1530/fixture/diagnostic.sbatch
```

On completion inspect allocation, batch AND extern accounting including exit codes/MaxRSS, `batch.exitcode`, outer and nested DONE/FAILED states, real raw17 files, commands and the pin report. Preserve failures and raw output. Do not delete/reuse a failed directory or turn a missing hit/report into a passing skip. A new attempt needs another reviewed namespace. Current wrapper's immutable fixture result is only an initial measurement; the expanded discriminating fixture needs separately reviewed implementation before it can be run.

**C2 gate remains CLOSED until real-result semantic review plus the separate author/A46 production authorization. Neither successful Slurm completion, CPU tests, upstream source inspection, nor a unique current-fixture reading opens it. No final production approval is given.**
