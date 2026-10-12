# Real GFFCompare semantic smoke — prepared, NOT EXECUTED

## Scope and observed availability

Prepared locally for controller independent review and submission. No SSH writes,
Slurm submissions, subagents, installation/upgrades, external downloads, production
code changes, merge, or push. GPU3163545 Priority-queued is the supplied context,
not a fresh queue observation; this task neither inspects nor changes that job.

Read-only pronghorn queries on 2026-09-17 01:45–01:47 UTC found:

- Executable `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare`,
  336328 bytes, mode775, reports `gffcompare v0.12.6`.
- Python `/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python`,
  resolves to `python3.11`, reports Python3.11.14.
- Exact commands/output: `notes/mvp_execution_20260916/gffcompare/availability.txt`.
  No entire-filesystem search. Binary hashing is deferred to the compute job.
  Login availability is NOT a real comparison or compute-node certification.

## Reviewed contract, not a replacement scorer

Read B worktree `/tmp/b6-p2-scoring-1535144`, verified HEAD `c024445`, and its
`notes/MVP_SCORE_PROVENANCE_REPORT.md` (that report is absent from main).
Graft queried first. Exact code spans in `revision/scripts/mvp/mvp_produce_counts.py`:

- `structure`, lines297–302: chromosome/strand + **all** intron boundaries for
  multi-exon; mono uses exact nonempty CDS intervals, not exon boundaries.
- `write_gtf`, lines305–310: exon-only GTF, half-open start +1, end unchanged,
  sorted transcript IDs, synthetic `gene_id "g"`. No CDS rows sent to GFFCompare.
- `reference_counts`, lines313–346: duplicate structure guard on BOTH sets before
  GTF/tool; split mono/multi; real command `gffcompare -r reference.gtf -o compare predicted.gtf`;
  exactly one `*.tmap`, exact query inventory; only class `=` accepted and then
  exact chain/strand cross-check; mono exact CDS intersection added separately.
  Empty query bypasses tool and returns TP0/FP0/FN=reference size.
- `produce`, lines406–411: absolute resolved binary, `--version`, regex requires
  0.12.6; executable bytes recorded. The smoke mirrors the pinned version check.

Existing tests `tests/test_mvp_score_producer.py:45–107` use an explicitly fake
GFFCompare. The fake writes `compare.predicted.gtf.tmap`; production only requires
one `*.tmap`. This smoke manufactures NO tmap and records the actual filename(s).
It does not claim those existing stub tests certify real tool semantics.

The standalone harness extracts and executes the exact AST definitions of
`require`, `file_identity`, `structure`, `write_gtf`, `reference_counts` from the
SHA-pinned producer source. The only dependency shim is stdlib SHA256 for
`w.sha256_file`. No substitute matcher/tool, heavy training import, or repository
package installation is needed. This is a function-level diagnostic, not a test
of full `produce`, GFF parsing, frozen-reference authority, or decoder integration.

## Hand-counted fixture

16 tiny isolated cases; synthetic chromosome `smoke1`, no FASTA/biological inputs.
Coordinates below are zero-based half-open. Multi reference R has exons
`[100,200),[300,400),[500,600)`, chain `(200,300),(400,500)`, strand+.
Thus emitted GTF exons are 101–200,301–400,501–600.
Mono reference M has exon `[1000,1200)`, CDS `[1020,1180)`, strand+.

| Case | Change from R or M | Expected TP/FP/FN | Raw class expectation |
|---|---|---|---|
| ordinary | R unchanged | 1/0/0 | `=` |
| terminal_short | first start120, last end580 | 1/0/0 | `=` |
| terminal_extend | first start80, last end620 | 1/0/0 | `=` |
| junction_change | middle exon start310; other junction unchanged | 0/1/1 | not `=` |
| opposite_strand | R on minus | 0/1/1 | not `=` |
| unmatched | exons[2000,2100),[2200,2300) | 0/1/1 | not `=` |
| mono_exact | M unchanged | 1/0/0 | producer bypasses tool |
| mono_utr_variant | exon[990,1210), same CDS | 1/0/0 | bypass |
| mono_cds_change | CDS start1021 | 0/1/1 | bypass |
| mono_opposite | M on minus | 0/1/1 | bypass |
| mono_no_cds | both mono have no CDS | 0/1/1 | bypass |
| empty_query | zero queries, R present | 0/0/1 | separate raw empty probe: observation only |
| multiisoform_unique | R and middle-start310 in both sets | 2/0/0 | both `=` |
| duplicate_query_terminal | R and terminal_short queries vs R | producer rejects predicted duplicates | real rows recorded, no chosen isoform |
| duplicate_reference_terminal | R query vs R and terminal_short refs | producer rejects reference duplicates | real rows recorded, no chosen isoform |
| duplicate_mono_cds | M and UTR variant queries vs M | producer rejects predicted duplicates | bypass |

Non-`=` subclasses are deliberately not predicted from a fake tool. Raw empty
query exit/status/files are recorded even if GFFCompare rejects an empty file;
that does not invalidate the producer's intentional empty-query bypass.
Duplicate multi-exon raw invocations must complete and produce a nonempty tmap,
but any collapsed inventory/reference choice is reported, NOT normalized into
counts. Producer duplicate failures are required to occur before any GTF/tool
output. No multiplicity policy is selected. Terminal-only variants remain
structurally identical and fail closed when co-present. The ±20bp cases test
these particular terminal variants, not every possible terminal offset.

## Files and pinned identities

Under `notes/mvp_execution_20260916/gffcompare/`:

- `smoke.py`: stdlib fixture generator, actual runtime driver, read-only analyzer,
  local selfchecks. SHA256 `3c022e41fa2759a33d8fee76452a55eca5ff2b226527a37fddecbba79c6098d1`.
- `smoke.sbatch`: 1task, 2CPU, 4GB, 15min maximum, no GPU.
  SHA256 `e6d18fcc732e183b7b3f40d3176e76889608f316453070ef4e12c5086ae0258b`.
- `availability.txt`: observed read-only commands/results.

Reviewed source to stage byte-for-byte as `mvp_produce_counts.py`:
`/tmp/b6-p2-scoring-1535144/revision/scripts/mvp/mvp_produce_counts.py`, SHA256
`bf7c8284804ee0a12c359a1fbe8f140bfc0f222abb522d6d7b7b4d3a691b417b`.
No full worktree/dependency/cache staging required. Hash gate refuses other source.

## Controller-only execution, NOT performed

1. Independently review harness and template; check live queue before submission.
   Select the established pronghorn CPU account/partition (not guessed here).
2. Stage the above three executable/source files and a controller-verified
   `stage.sha256` into the new namespace
   **`$R/evidence/mvp_gffcompare_smoke_20260916`**. R must be the controller's
   existing absolute revision root. Do not repurpose a biological output path.
   Stage manifest must contain the three hashes above under names `smoke.py`,
   `smoke.sbatch`, `mvp_produce_counts.py`. Preserve this plan/availability locally.
3. Submit once with explicit approved account/CPU partition, exporting absolute R,
   and absolute logs under that namespace. Template, not an executed command:

```sh
BASE="$R/evidence/mvp_gffcompare_smoke_20260916"
sbatch --account="$APPROVED_ACCOUNT" --partition="$APPROVED_CPU_PARTITION" \
  --export="ALL,R=$R" \
  --output="$BASE/slurm-%j.out" --error="$BASE/slurm-%j.err" "$BASE/smoke.sbatch"
```

Pinned payload commands (the template runs these under Slurm, no shell activation):

```sh
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
"$PY" -I "$BASE/smoke.py" selfcheck --producer "$BASE/mvp_produce_counts.py"
"$PY" -I "$BASE/smoke.py" run --producer "$BASE/mvp_produce_counts.py" --root "$BASE/run"
"$PY" -I "$BASE/smoke.py" analyze --root "$BASE/run"
```

`run/` must not exist: no resume/overwrite. Every regular multi case runs the
actual binary twice: raw observation and unchanged producer function. Duplicate
and empty cases get separate raw calls despite the producer's guard/bypass.
Each raw/version subprocess is bounded to60seconds; unchanged producer subprocesses
have no per-call timeout. Slurm bounds the entire payload to15minutes.
Runtime hash checks detect producer/harness/binary changes during execution.

Raw evidence: `fixture.json`; per-case exon-only GTFs; raw exact command JSON,
stdout/stderr/exit, actual `.tmap/.refmap/.stats/...`; producer-generated GTF/tool
outputs and count/error receipt; `version/`; `runtime.json` with real binary hash,
resolved path, producer/harness hashes/Python; `hashes.json` covering all run files.
Slurm logs are outside run and must be retained with scheduler allocation/batch
exit status. Final payload prints `hashes.json` SHA256 for controller pinning.
Hash manifest and logs are provenance evidence, not adversarial attestation.

Analyzer only reads and prints JSON; checks complete file inventory/hashes,
fixture identity, exact commands/version, hand counts, query IDs, class acceptance,
and duplicate rejection. Missing/incomplete runtime evidence raises an error;
there is no generated/stub PASS. Actual observations include all duplicate tmap
rows and filenames and the empty query exit. Any mismatch is investigation, not
permission to weaken the scorer. Analyzer report goes to Slurm stdout; it does
not write into or mutate the raw evidence tree. Keep the original absolute run
path when reanalyzing because exact logged commands bind that path.

## Local verification actually performed

```sh
python3 -I notes/mvp_execution_20260916/gffcompare/smoke.py selfcheck \
  --producer /tmp/b6-p2-scoring-1535144/revision/scripts/mvp/mvp_produce_counts.py
bash -n notes/mvp_execution_20260916/gffcompare/smoke.sbatch
```

Both exited0. Selfcheck printed:
`LOCAL_ONLY: generator/mono/empty/duplicate checks passed; real-tool semantics NOT RUN`.
It checks literal GTF coordinates/attributes, hand-listed retained/changed
junctions, mono counts (including CDS vs UTR), empty counts, duplicate rejection,
and rejects analyzing a generated-but-unexecuted fixture. No fake tmap is used;
real-tool case expectations remain pending. Advisor was attempted but unavailable
(client/model version mismatch); no advisor review is claimed and no upgrade made.

**Current status: LOCAL PREPARATION ONLY.** Actual binary comparison, raw results,
terminal-duplicate behavior, and semantic PASS are NOT yet observed. Frozen-panel
GFF compatibility is a later separate test. Independent-read provenance, author
multiplicity policy, full biological scoring, and final decision remain blocked;
`decision=null` is preserved. No author policy selection or fake PASS.
