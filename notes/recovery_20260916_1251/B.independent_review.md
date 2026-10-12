# Track B independent review

## Verdicts

- **Spec compliance: PARTIAL; not end-to-end Track B acceptance.** The five discovery must-fixes are substantially implemented for the supplied CPU components. The raw-GFF/count producer, decoding, actual alignment verification, and evidentiary decision gate remain incomplete. Three validation gaps below prevent treating these components as a scientifically validated scoring pipeline.
- **Code quality/correctness: REQUEST CHANGES (medium-severity input-validation findings).** Core geometry, strand reuse, bootstrap arithmetic, and STOP-first logic look sound. No currently reachable public CLI GO/STOP publication bug was found: the CLI always sets `decision=null`, and `--final-decision` raises. This mitigation does not validate the numerical outputs it does publish.

Reviewed final HEAD `200f1141708b4f913a7e05ae88661fce4abaf98e`, implementation `eefe5b505b8c3315dbe432ecf882b6e85296576d`, relative to `f847cb80c7750326f0781f81481891ab0c9ea309`. Read the complete supplied diff, recovery report, SCORING.md, binding MVP protocol, and discovery must-fixes. Queried graft in the selected worktree before source inspection. Used the unchanged SAM/strand/sampler implementations to check integration details.

The controller's **86 passed / 21 subtests** result was accepted as supplied, not rerun. Only the focused invalid-count repro below was executed. No source edits, SSH, jobs, subagents, merge, or push. Graft automatically refreshed its local graph on its first query; no implementation files were edited.

## Actionable findings

### B1 — MEDIUM: impossible junction counts and seed-dependent frozen denominators are accepted

**Location:** `revision/scripts/mvp/mvp_score.py:88-108` (especially 101-107); consequences at `:118-125`.

The parser checks each numerator against its named denominator, but omits relationships guaranteed by its own `junction_counts` helper:

- `recall_tp <= precision_tp`, because gene-contained supported junctions are a subset of window-supported junctions;
- consequently `recall_tp <= predicted`;
- a locus's `f1_reference` and, when available, `supported` are identical across all three seeds, not just between off/on within each seed. The reference and independently aligned read set are frozen and do not depend on the trained seed.

**Executed focused repro:** create three distinct one-record pair files with valid CE records and deltas .001/.002/.003; load the actual frozen panel; give every locus/seed the following counts:

```python
off = dict(f1_tp=1, f1_predicted=2, f1_reference=2,
           precision_tp=0, predicted=1, recall_tp=0, supported=10)
on = dict(off, recall_tp=10)
# payload.loci: each frozen gene_id, seeds 1/2/3, off/on above
result = score(payload, panel)
```

Observed successful output for all50 pooled: `delta_P point/lower95/upper95 = 0`, `delta_R point/lower95/upper95 = 1`. One predicted junction, no precision true positives, and ten recall true positives cannot coexist under the implemented set definitions. This even creates an apparent structural contradiction to futility (positive recall lower bound with nonnegative precision delta), although the public decision remains blocked.

In the same focused run, changing **both fractions of seed 2 only** to `f1_reference=200, supported=1000` also succeeded. This is not merely a lack of raw-GFF verification: these contradictions can be rejected using the count payload itself.

**Fix:** enforce the subset invariant and cross-seed frozen denominators per locus before computing statistics. Add rejection fixtures for each independently; keep zero denominators distinguishable from missing inputs.

### B2 — MEDIUM: three seed keys can relabel one evaluation as three runs

**Location:** `revision/scripts/mvp/mvp_score.py:72-76`; `src/transgenic/evidence/mvp_bootstrap.py:83-116`; existing acceptance fixture `tests/test_mvp_scoring.py:158-177`.

The CLI requires string keys `1/2/3`, but loads arbitrary paths without binding the key to a producer seed/run. The existing CLI fixture explicitly assigns one pair file to all three keys and expects successful three-seed statistics. This was established by inspection of the supplied test and code, **not by repeating that test**.

**Repro:** supply `pairs={"1":"/tmp/seed1/eval_pairs.jsonl", "2":"/tmp/seed1/eval_pairs.jsonl", "3":"/tmp/seed1/eval_pairs.jsonl"}` with otherwise valid counts. The same records satisfy all inventory checks and populate all three `per_seed_point` slots. A positive seed-1 effect thus appears positive for all required seeds without evidence of seeds 2 and 3. This does not necessarily narrow the tile-bootstrap interval: the defect is unsupported three-run identity and seed-level reporting, not resampling seeds independently.

**Fix:** at minimum reject duplicate resolved pair paths. Before operational scientific use, bind each file hash to a verified producer receipt carrying its actual seed/run and final inventory; differing filenames alone are insufficient. Legitimately identical numbers from three proven runs must remain allowed. If that receipt integration is intentionally deferred, mark **pair/run provenance**, not just count provenance, explicitly unverified in the machine-readable output and do not claim the three-seed policy is verified.

### B3 — MEDIUM operational provenance gap: reference/GFF identity is caller-asserted, not manifest-bound

**Location:** `revision/scripts/mvp/mvp_align_panel_longreads.py:130-137,159,191-198`.

Read identity is correctly checked against the canonical ROOT roles row. In contrast, reference/GFF identity is checked only against `--ref-md5` / `--gff-md5` supplied alongside those same files. Nothing authenticates those expected values against the frozen Athaliana species manifest required by protocol §§1.2/2. SCORING.md tells the operator to obtain the values there, but this is a manual precondition, not an implemented fail-closed gate.

**Repro by direct code inspection (not a full aligner run):** provide an alternative FASTA with the required chromosome names and sufficient lengths, any alternative GFF, and each file's own MD5; retain the valid panel and canonical read/roles inputs. The reference checks and FASTA geometry checks permit `VALIDATED_ONLY`. With successful external commands, the same inputs can receive `SCORING_ALIGNMENT_COMPLETE`. The actual consumed files are hashed, but their frozen identity is not established.

**Fix:** obtain/verify expected Athaliana hashes from a commit-bound species manifest (or a separately verified receipt binding that manifest), and record that anchor. Alternatively explicitly classify this driver as requiring an external reference-identity gate, and avoid treating `VALIDATED_ONLY` as full scientific input validation. This does not negate the correctly implemented actual-read MD5 must-fix.

## Checked behavior that is correct within the implemented boundary

| Requirement | Assessment / evidence |
|---|---|
| Centered window and prefilter | `mvp_windows.py:139-190`: half-open 30,720 nt, odd padding base to genomic right, decimal **10,000** nt prefilter padding. No gene±10,240 substitution. Window boundary errors fail rather than shift. Decoder reuse remains future work, not verified implementation. |
| Separate support/denominator scopes | `mvp_windows.py:193-236`, `mvp_score.py:29-57`: window-contained supported junctions; gene-overlapping transcripts retain all introns; gene-contained recall denominator. Wrong-strand predictions are retained for penalization. |
| Roles and actual read bytes | `mvp_align_panel_longreads.py:36-55`: exact canonical ROOT schema, one matching whitelisted run, Athaliana/reference/validation-only role, fresh MD5 of resolved consumed read path. No training-role bypass. Before/after input hashing detects persistent mutation during execution. |
| Orientation | `:58-94,171-178`: four-MD5 legacy key plus audit JSON MD5; unchanged scorer defaults MAPQ20/1000/.95; PASS alone enables `-uf`; FAIL/UNRESOLVED use no `-uf`, as required. Identity/count/status inconsistencies fail. Fresh disjoint output path and symlink-redirection check protect the legacy namespace. The sampler's actual CLI accepts the supplied arguments. |
| Pass 2 | `:97-103`: correct source-specific splice/splice:hq, ONT k14, conditional uf, G200000, secondary=no, cs=long, full reference. Thread count is configurable. No external-tool execution certification follows from inspection. |
| Strand and rejection | `mvp_junction_table.py:77-141` delegates to unchanged `59_evidence_support.py:69-160,272-288`: FLAG×ts composition, audited-uf consistency, MAPQ20, whole-read chimeric rejection. Raw exact intron coordinates and 3 ONT / 2 HiFi thresholds match this MVP definition. No unintended B1 coordinate-correction substitution. |
| Primary arithmetic | `mvp_bootstrap.py:56-151`: finite pure CE consistent with numerator/denominator, paired off/on denominators, matched RN inventories/evidence flags, duplicate RN rejection, evidence-positive tiles only, **off−on**, equal tiles/seeds rather than token pooling. Composite stays diagnostic. |
| Bootstrap | Tile carries all three columns; locus carries both fractions/all seeds and recomputes pooled ratios. 2000 draws, default_rng(20260914), 5th/95th percentiles. F1 is `200*TP/(predicted+reference)` percentage points, not average per-locus F1. Per-seed/pooled and all50/core20 are produced. Undefined ratios become UNASSESSABLE. |
| Mathematical decision | `mvp_bootstrap.py:219-295`: STOP first, threshold strictness correct, reversal at >=.005, positive primary lower bound plus all positive seed points for GO, missing structural/unassessable withholding prevents an otherwise prospective STOP from falling through to GO. Precision-point interpretation is disclosed, still pending author review. |
| Public safety gate | `mvp_score.py:23-26,110-116,138-149`: final publication blocked, decision null, training ineligible, count-provenance limitation in CLI output. No author-policy approval knob. Missing structural fields are absence, mixed availability rejected. |

## Omissions versus unsafe implemented behavior

**Acknowledged scope omissions, not disguised completed features:** GPU decoding/300 predictions, seed123 baseline column, restricted raw-GFF producer and GFFCompare/exact-CDS verification, end-to-end evidence/count provenance, real alignment execution smoke, complete-chain invocation, withholding receipts/equivalence review, and author coverage policy. The score adapter also reports metric deltas rather than the full absolute off/on readout table required for the eventual result report. These block full Track B acceptance but do not by themselves mean this partial CPU adapter is incorrectly implemented.

**Implemented acceptance behavior requiring correction:** B1 accepts internally impossible count summaries; B2 treats user labels as seed identity; B3 permits self-attested reference identity. None currently emits a final GO/STOP through the public CLI. They can nevertheless publish numerical readouts or alignment completion records that need further validation before scientific interpretation.

The documentation is unusually clear about the raw-GFF/decoder omissions and correctly says hashes do not prove count provenance. Together with unconditional final-decision blocking, it genuinely prevents an accidental GO/STOP claim through the provided score CLI. It does **not** make all intermediate outputs scientific evidence: junction TSV/QC is a low-level un-gated adapter, `structural_available` means only that count fields were supplied, pair run identity is unchecked, and `alignment.json` means command completion rather than coverage acceptance. Future automation must not promote any of these into an approval receipt merely by checking existence/status.

**Recommended disposition:** retain as a partial implementation for revision; fix B1, add seed provenance/rejection or explicit unverified-pair status for B2, and establish/document the external reference trust gate for B3 before operational alignment/scientific scoring. Keep the final-decision block until real provenance, coverage policy, and remaining protocol outputs are complete.

Graft reported approximately 1,027,359 tokens saved across this review's queries (tool estimate).
