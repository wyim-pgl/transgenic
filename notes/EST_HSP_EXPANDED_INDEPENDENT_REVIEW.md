# Independent EST HSP expanded review

Reviewed detached `8dcd580` against base `5c7a0aa`; implementation is
`a98f5c0cc08e6f6a5ac27905d71a4478e9a0e789`. Review only: no source edits,
DIAMOND execution, SSH, submission, subagents, merge, or push.

## Separate verdicts

- **Code correctness: acceptable for the bounded diagnostic; no blocking defect found.** This is not a universal interpreter validation or evidence that all 144 intended traces will be emitted.
- **Safe to run diagnostic: YES, conditionally through the reviewed wrapper**, after controller live-job/accounting checks and exclusive, immutable archive staging. This is not permission to submit from this review, expand resources, select an acceptance policy, or run production.
- **C2/C3 and production remain CLOSED.** `certified=false`, `production_authorized=false`, and unresolved author policy are appropriate even for complete diagnostic coverage.

## Scope and independent evidence

Read the supplied full contract, committed implementation report, all 330 module
lines, all 193 test lines, wrapper, recheck script, and both actual first-job
raw17 files. The supplied neutral review diff byte-matches `git diff 5c7a0aa
8dcd580`: five new files only; historical launcher, fixture, acceptance, and
projection sources are unchanged. Built/queried the graph in this worktree and
read exact launcher/import spans needed to establish execution closure.

Fresh local SHA256 checks match all three wrapper Python pins:

- expanded module: `aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4`
- launcher: `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291`
- taxonomy helper: `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302`

Did not duplicate the controller's reported 35-test suite, 12-row interpreter
recheck, or bash syntax check. Those remain controller evidence, not fresh test
runs claimed by this reviewer. Independently counted rich-string M/I/D/F across
both actual raw files (without calling the interpreter): all 12 reported lengths
equal M+I+D. Raw hashes independently match the committed report. This confirms
these rows only, not a universal DIAMOND law or the author's intended 30-aa rule.
The two labels repeat the same biological fixture and are not independent
biological replicates. No live binary was hashed here.

## Correctness review

`revision/scripts/evidence/est_hsp_expanded.py:49-126` is genuinely separate from
old replay/candidate decomposition: its top-level imports are stdlib only, and
it translates consumed codons from oriented input DNA using a standard-code
table while checking subject residues against input protein. CIGAR, BTOP,
gapped strings, and translated-query output are redundant engine serialization
checks, not advertised as independent biological ground truth.

The cursor rules are consistent: M consumes 3 nt/1 subject residue; I consumes
3 nt/0; D consumes 0/1; backslash and slash consume +1/-1 oriented nt on either
strand. Reverse start/end transforms and signed initial/post-shift frames are
explicit. Counts include substitutions in M; lengths distinguish M, M+I,
M+I+D, and M+I+D+F. Endpoint, finite-score/evalue, translated sequence, BTOP,
and CIGAR checks constrain valid rows. Sum of blocks is correctly treated as
redundant with query-residue count, rather than a newly distinguishable policy.
No column-9 candidate is silently promoted to policy.

`est_hsp_expanded.py:129-218` defines exactly 24 shapes x three offsets x two
strands. The cases cover ordinary 29/30/31, boundary I/D, both shift signs in
short and 29+29/30/31 block traces, robust long indels/shifts, and two canceling
shifts. Leading offsets imply all six signed frames. Independent stop-free
flanks avoid making a 29-aa target itself the entire short ORF; they do not
promise an emitted HSP under the frozen scoring/min-ORF conditions. Sequential
edit ledgers retain inserted/deleted bases, reconstruct oriented DNA, and verify
baseline protein translation. A -1 edit may change the following amino acid;
this is a legitimate substitution, not an interpreter defect.

`est_hsp_expanded.py:211-218,238-269` requires the expected subject, endpoints,
initial frame, counts, block counts, and shift positions/directions for ELICITED.
Valid clipped/off-target/repositioned traces are NOT_ELICITED; invalid rows are
INVALID. Missing or duplicate elicited cases, extra non-elicited/invalid rows,
no consistent column-9 reading, and surviving wrong-sign controls all prevent
complete status. Missing shapes are not passed off as validated threshold
negatives. Engine scoring can legitimately choose a different trace; that is
not itself evidence of a parser bug.

## Findings and targeted checks

**No high/medium correctness or execution-safety findings.** No source fix is
requested before the bounded diagnostic. Two informational limitations matter
when interpreting its report:

1. **INFO — wrong-sign rejection is a conservative per-row requirement, not a
   mathematical property of all shifted sequences.** Location:
   `est_hsp_expanded.py:247-252,262-264`; CPU example at
   `revision/scripts/tests/test_est_hsp_expanded.py:57-61`.
   Targeted reproduction used the existing canceling example DNA `ATGCGCTGG`,
   qg `M\\A/W`, sg `M-A-W`, BTOP `1\\-1/-1`, CIGAR `1M1\\1M1/1M`, coordinates
   1..9, frame +1, and translated `MAW`. Calling `interpret(..., shift_sign=-1)`
   also succeeds: GCG and GCT both encode alanine and the opposite shifts cancel
   at the endpoint. My initial targeted expectation of rejection was therefore
   disproved. This is not a defect: `diagnose` records a surviving control and
   stays INCONCLUSIVE. A follow-up targeted check of all six generated `cancel`
   cases showed the first post-shift codon is N under the intended sign and A
   under the wrong sign, so this ambiguity does not invalidate their design.
   Do not reinterpret a future surviving control as production approval.

2. **INFO — malformed non-text/truncated process output can leave only partial
   evidence rather than a complete row report.** Location:
   `est_hsp_expanded.py:221-234,300-313`. Raw bytes are copied before text parsing;
   row-level ordinary parse/lookup errors are retained. Decode/I/O failure or
   hard kill can interrupt reporting. Outer failure handling never writes DONE
   after such an exception; hard kill can omit FAILED too. Thus absence of a
   complete report/marker is incomplete, never a passing skip. This is an
   operational limitation, not false certification; no targeted fault injection
   was needed or performed.

## Execution safety and remaining controller gates

`notes/est_hsp_expanded_20260916_1541.sbatch:2-50` bounds execution to one task,
two CPUs, 4 GB, 30 minutes, no requeue, GPU, array, or production scan. Fresh
`est_followup_20260916_1541/expanded` paths, canonical-path checks, exclusive
execution lock/tmp/result creation, umask 077, absolute Python/DIAMOND paths,
Python isolated/no-bytecode mode, sanitized Python environment and thread caps
are appropriate. The wrapper hashes the complete project import closure plus
the pinned DIAMOND binary on compute; the runner repeats binary pin checking
and enforces version 2.1.24.178 through the unchanged launcher. These are not
claims of a fully hermetic or hash-pinned Python installation.

`est_hsp_expanded.py:275-313` generates its own tiny FASTAs and database and
runs one makedb and one blastx. It imports only the pinned launcher and its
stdlib-only taxonomy helper; importing that helper performs no network fetch.
`est_blastx.py:199-206` preserves frozen production search flags.
`est_blastx.py:249-272` supplies process-group interruption cleanup. The
separate first-job recheck script is not imported into this runner. There is
no historical raw/DB/FASTA input or acceptance/projection entrypoint on the
runtime path. Raw-first preservation, per-row errors, FAILED on exceptions,
and unconditional false authorization prevent false certification. DONE means
diagnostic completion, including an INCONCLUSIVE result, not scientific success.

Before execution, controller must verify live queue/accounting to avoid repeat
jobs, stage the exact source-commit archive rather than a mutable worktree,
verify hashes and make code read-only, and confirm the namespace is fresh.
The previous job's reported extern OUT_OF_MEMORY/0:125 remains a controller
accounting issue; allocation/batch completion alone is not clean success.
After execution inspect all scheduler steps, exit code, raw data, diagnostic
coverage, wrong-sign results, and DONE/FAILED/missing artifacts. Do not silently
relax flags or increase the hard budget if shapes fail to elicit. Expanded real
results and separate semantic/author authorization are still required.

Advisor consultation was attempted but unavailable (client/model-version
incompatibility); no advisor opinion is represented as completed review.
Only this report is committed; graph-generated local ignore/cache changes are
excluded from the report commit.
