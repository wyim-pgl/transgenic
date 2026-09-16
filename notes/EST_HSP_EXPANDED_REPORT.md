# EST HSP expanded diagnostic — 2026-09-16 15:41

## Disposition

Diagnostic implementation only. **C2 acceptance and C3 projection remain CLOSED.**
No author 30-aa rule is selected. No engine execution, SSH writes, submissions,
subagents, merge, push, or production-data search was performed in this task.
Branch: `est-hsp-expanded-20260916-1541`, based on exact
`5c7a0aa7e83a33ab3e18abd922e8fff6428b5f35` in the native isolated worktree.
Historical `est_blastx_fixture.py`, `est_blastx.py`, acceptance and projection
modules are unchanged. Implementation follows the full discriminator contract
`EST_HSP_CONTRACT_20260916_1530.md` supplied by the controller.

## Actual first-job finding (not simulated execution)

Read the fetched job 6165460 `fixture_result/engine_fixture` inputs and both
real raw17 files. The new independent interpreter successfully rechecked all
12 rows against the supplied DNA and retained protein FASTAs; input/raw hashes
were checked against fetched provenance. The receipt reports binary hash
`0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`
and version 2.1.24.178. This task did not rehash a live remote executable.

Both source labels give the same counts (M includes substitutions):

| Query | col9 | M | I | D | F | Query residues | Shift-delimited query blocks |
|---|---:|---:|---:|---:|---:|---:|---|
| ordinary | 180 | 180 | 0 | 0 | 0 | 180 | 180 |
| reverse | 180 | 180 | 0 | 0 | 0 | 180 | 180 |
| plus1 | 180 | 180 | 0 | 0 | 1 | 180 | 90,90 |
| minus1 | 180 | 180 | 0 | 0 | 1 | 180 | 90,90 |
| insertion | 183 | 180 | 3 | 0 | 0 | 183 | 183 |
| deletion | 180 | 177 | 0 | 3 | 0 | 177 | 177 |

Observed col9 is **M+I+D, excluding F**, across these rows. Backslash +1 and
slash −1 oriented-nucleotide consumption satisfy codon truth. Deliberately
reversing those signs fails on each of the four shifted source/query rows.
The minus-one row includes a real query/subject substitution (`SA` in BTOP),
which is checked rather than assuming every paired column is an exact match.
This is evidence about these emitted engine traces, not author policy approval,
and the two source labels are not independent biological replicates.

Raw SHA256: OrthoDB
`70dbe2c48ef41d5e984b45e91e0fc324f3f7adb15a2b7bf86e5f6f69e90fb901`;
Swiss-Prot `3b43cbb2618adc69e532775c269b7a1e24f5494c49ef8c6dc8a7e3b64036f3dc`.
`fixture_result.txt` records allocation and batch COMPLETED/0:0, payload exit 0,
but extern **OUT_OF_MEMORY/0:125**. This is not clean accounting. Controller
must inspect all steps before interpreting scheduler success.

Reproduce the read-only recheck (no engine invocation):

```sh
python notes/est_hsp_recheck_first.py /path/to/fixture_result/engine_fixture
```

## Bounded family selected

`revision/scripts/evidence/est_hsp_expanded.py` contains one standalone stdlib
truth interpreter, fixed generator, report writer, and runner. No import of
candidate decomposition, old replay, acceptance, or projection. Only the runner
lazily imports the historical launcher (and its taxonomy helper) to preserve
production DIAMOND flags and child-process termination behavior.

**24 shapes × 3 leading offsets × both strands = 144 queries**, one tiny
24-protein synthetic database, one makedb and one production-flag blastx. This
is a small discriminator family, not an unbounded cross-product of edit sites,
sequences, and penalties. Taxonomy filtering is not retested; the historical
fixture already covers it. No claim that a particular short trace will be
elicited under `--min-orf 30`, scoring, evalue, or `--max-hsps 1`.

Shapes:
- exact 29/30/31;
- one internal D in 30/31 subject residues (29/30 query residues);
- one internal I with 29/30 paired residues (30/31 query residues);
- both shift signs in 29/30/31-residue traces;
- both shift signs between 29+29, 29+30, 29+31 blocks;
- long 180-aa ±1-nt shift anchors and +3/−3-aa gap controls;
- two opposite shifts with nonempty 29/30/31 blocks (net zero).

Each case has 20 independent stop-free codons on each flank and offset 0/1/2;
all queries exceed 100 nt. The target does not include those flanks. Signed
initial frames ±1/±2/±3 are explicitly required, not inferred from names.
Sequential oriented-DNA edit ledgers record actual deleted/inserted bases.
Baseline translation and ledger reconstruction are checked independently before
interpreting a case. Reverse complements reuse the oriented edit truth.

## Reporting and failure behavior

Before any interpretation, raw bytes are copied exclusively to `raw-first.tsv`
and hashed in `raw-first.json` with the entire expected case inventory. Every
row (including malformed/off-target rows) is retained in `diagnostic.json`.
Per-row reports include raw/sequence hashes, M/I/D/F, reported length,
M+I+D, M+I+D+F, query residues M+I, paired M, block counts, initial/post-shift
frames, endpoints, nucleotide codons/residues, and hypothetical ≥30 decisions
for each reading and block. Whole query counts and sum of blocks are explicitly
redundant, not purportedly distinguishable policies.

The interpreter translates each consumed codon on oriented input DNA and checks
protein residues and final endpoints, then checks translated-query serialization,
BTOP and CIGAR. Rich engine fields are cross-checks, not independent truth.
I consumes (3 nt,0 aa); D (0 nt,1 aa); shifts (±1 nt,0 aa) on both strands.
Wrong-sign negative controls are also attempted for every emitted shifted row.

ELICITED requires exact intended subject, endpoints, initial frame, counts,
block counts, and every shift position/direction. Missing, clipped, shifted-at-a-
different-site, or off-target traces remain NOT_ELICITED, not validated negatives.
All missing cases are logged; no filtering to successful traces. Interpretation
errors are INVALID. Any missing/invalid/unexpected row, surviving wrong-sign
control, or absent globally consistent col9 reading makes the summary
INCONCLUSIVE. Even complete coverage says DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED:
`certified=false`, `production_authorized=false`, and policy UNRESOLVED always.
DONE means diagnostic process completion only, not certification. Engine failure
preserves available raw output and attempts reporting, then writes FAILED;
SIGKILL/OOM may leave incomplete artifacts, never evidence of passing.

## CPU verification and limitations

CPU fixtures are explicitly labeled synthetic interpreter tests, not engine
execution. They cover both shift signs on all six initial frames, reverse
coordinates, ordinary and shifted 29/30/31 boundaries, boundary I/D, mismatches,
29+29/29+30/29+31 blocks, canceling shifts, standard-code residues, wrong-sign
rejection, corrupted serialization/endpoints, tampered edit ledger, clipped
trace handling, and raw-first preservation for malformed rows.

Run:

```sh
python -m pytest revision/scripts/tests/test_est_blastx.py \
  revision/scripts/tests/test_est_blastx_fixture.py \
  revision/scripts/tests/test_est_hsp_expanded.py -q
```

Verified result: **35 CPU tests passed** (13 new diagnostic tests and 22 historical
launcher/fixture tests). The fetched first-job read-only recheck also completed
successfully for all 12 rows. Historical source/gate files have zero diff from
the pinned base.

The expanded family has **not** run on DIAMOND. No fabricated raw17, relaxed
flags, or CPU-generated engine success is used. CPU tests cannot prove trace
elicitation or installed-engine behavior. Strategic advisor consultation was
attempted but unavailable due to client/model-version incompatibility; no
independent model review is claimed. Controller review and actual bounded
execution remain mandatory.

## Controller execution handoff

Source/test implementation commit:
`a98f5c0cc08e6f6a5ac27905d71a4478e9a0e789`.
Use the companion `est_hsp_expanded_20260916_1541.sbatch` after reviewing source
and pins. The wrapper is a subsequent documentation-only commit and stages the
source commit above. Complete non-stdlib runtime import pins measured after that
source commit:

| File under revision/scripts | SHA256 |
|---|---|
| evidence/est_hsp_expanded.py | `aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4` |
| evidence/est_blastx.py | `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291` |
| 64_orthodb_filter.py | `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302` |

The interpreter itself has no project imports. The separate read-only recheck
script is not imported by the engine runner and is not part of its closure. Hard allocation: **2 CPUs, 4 GB, 30 minutes**, no GPU/array/network or
production inputs. This is a bounded estimate for 144 short synthetic queries,
not measured runtime/RSS. Do not increase automatically on failure.

Controller must check live queue/accounting before staging/submitting, stage a
commit archive (not a worktree copy) in the wrapper's exclusive namespace,
verify its complete import-closure hashes, make staged code read-only, and inspect
batch AND extern accounting plus raw/report/FAILED/DONE evidence afterward.
No acceptance or projection follows automatically, even if all shapes elicit.
