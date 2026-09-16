# MVP score provenance implementation report

**Latest: round-1 review fixes are documented below. That section supersedes
initial provenance-check and synthetic-loader-fixture descriptions.**

## Identity / status

- Worktree `/tmp/b6-p2-scoring-1535144`, branch `b6-p2-scoring-1535144`.
- Verified requested base: `2276fcb1efdeb124a97a11dced8d73e4d48560e0`.
- Implementation: **`1a9253d02b0202106e29587ff689bb1e2e1f3e72`**.
- **Partial evaluation implementation; controller review required before use.**
  No empirical efficacy, real-tool certification, training authorization, or
  complete connected biological evaluation is claimed.
- Decoder interface published early at
  **`notes/MVP_EVALUATION_INPUT_CONTRACT.md`**. Producer:
  `revision/scripts/mvp/mvp_produce_counts.py`.
- No training code, approved bootstrap math, fixed thresholds, legacy scorer,
  main worktree, or approved protocol was edited. Training does not wait on this
  producer. No GPU, real alignment, remote action, job, subagent, merge or push.
  Pre-existing `.gitignore` change and `.ignore` remain outside task commits.

## What is implemented

1. **Existing finalizer receipts, not a parallel trust schema.** Reads the exact
   five-payload `manifest.json`, `TRAINING_DONE`, `metadata.json`, paired JSONL,
   paired summary and final safetensors produced by `mvp_finalize`. Reuses its
   metadata validation and `mvp_metrics.aggregate_paired_records`. Binds seed
   1/2/3, augmentation seed, fixed recipe, source checkpoint/audit file bytes,
   final checkpoint/config bytes, code-manifest hash and listed code bytes,
   corpus/evidence/panel identities, and exact complete valid RN inventory.
   Duplicate final RN records fail even if the manifest is rehashed.
2. **Actual candidate inventory reuse.** Reuses `validate_candidate_records`
   and `verify_candidate_artifacts`, including read-only corpus/evidence DB
   metadata comparisons. All seeds share data, consumed-code and initialization
   source identities. Code-manifest package comparisons use overlapping package
   fields: the existing finalizer runtime schema does not include duckdb.
3. **Commit-bound reference identity.** Reads the existing immutable git object
   `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`.
   Exact Athaliana FASTA MD5 `513e5ef30845ed754b00816a99abbf8e`; GFF MD5
   `0264c349bc61ccceb546648edec46ebb`. Missing git anchor or mismatched data fails
   closed. No caller-provided reference hash/approval can substitute. FASTA
   lengths are streamed and frozen panel windows checked against them.
4. **Raw reference-dependent count production.** Reuses script 24 restriction
   and script 07 half-open parser behind strict validation. Exact panel gene
   geometry/strand and gene-overlap prediction restriction; explicit exons,
   CDS containment, unique transcript IDs, known strands and parent hierarchy.
   Multi-exon comparisons invoke GFFCompare 0.12.6, accept `=` only after exact
   chain/strand cross-check; monoexonic matches require exact nonempty CDS.
   TP/FP/FN, commands, executable/version, inputs, restricted GTF and scorer
   output identities are retained. No caller counts are accepted as verified.
5. **350-output transport inventory.** All frozen 50 loci × three seeds × off/on,
   plus 50 seed-123 baseline outputs. Exact window, checkpoint, bytes and unique
   file identity required; partial panels and cross-seed checkpoint substitution
   fail. Empty header-only GFF is zero prediction; missing files are errors.
   Baseline reference F1 percentages are reported for all50/core20. Bootstrap
   computation calls the unchanged existing adapter in the same invocation.
6. **Boundaries preserved.** Reference F1 is not independent read-splice P/R.
   Structural metrics are absent, never zero-filled; withholding remains
   unassessable; coverage/HiFi author-policy block and `decision=null` remain.
   Old `mvp_score.py --input counts.json` is intentionally still an unverified
   count adapter; loading producer counts there does not transfer verification.

Local receipts establish byte/schema/identity consistency. They cannot prove
historical training or decoder execution against a party capable of rewriting
all artifacts, and are not scientific approval. Decoder commands are explicitly
recorded, not independently witnessed. Existing metadata `verification` labels
are not treated as substitute evidence for byte checks.

## Actual CPU verification

Main covering run:

```sh
PYTHONPATH=/tmp/b6-trackb-testdeps:src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  tests/test_mvp_score_producer.py tests/test_mvp_score_producer_integration.py \
  tests/test_mvp_scoring.py tests/test_mvp_scoring_driver.py \
  tests/test_mvp_metrics.py revision/scripts/tests/test_evidence_support.py -q
```

**115 passed, 21 subtests passed** (21.55 s final main run). New producer suites
account for **20 tests**. Coverage includes hand-count TP/FP/FN, strand and exact
mono CDS changes, duplicate-isoform rejection, malformed GFF, missing/empty
predictions, full350 inventory, partial panel, wrong windows/checkpoints,
actual finalizer publication, wrong seeds, missing RN, byte mutation, rehashed
bad summary/duplicate RN/test-scope/marker, missing authority anchor and wrong
reference bytes. The full350 integration uses the actual frozen panel and real
CPU finalizer publication, but **synthetic reference/candidate authority and an
explicit GFFCompare subprocess stand-in**; it is not biological/tool validation.

Additional reused candidate gate verification:

```sh
PYTHONPATH=/tmp/b6-p2-readiness-1535112/.testdeps:/tmp/b6-trackb-testdeps:src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  tests/test_mvp_cli.py -k candidate -q
```

**16 passed, 29 deselected** (17.08 s). Initial attempt using only Track B's
existing test dependencies had 6 import failures because duckdb was absent;
reused the already-installed local temporary dependency directory read-only.
No repository/system dependency installation was made. Initial new tests also
caught the real duckdb/runtime-package schema difference and script 24's final
attribute newline capture; both were fixed, with the latter confined to the
locally loaded restrictor instance rather than editing the frozen script.

`compileall` for the task modules/tests and `git diff --check` passed. Graft was
queried before source work and refreshed after implementation. This is targeted
CPU verification, not the complete repository test suite.

## Remaining gaps / required review

- **Real GFFCompare 0.12.6 is not installed on PATH here.** No real comparison
  binary was run. Exact tmap behavior, actual frozen-reference parsing, decoder
  GFF compatibility and successful full-panel empirical production still need
  an authorized local CPU smoke. Fake tool fixtures are explicitly not proof.
- **Duplicate-isoform multiplicity policy remains unresolved.** Identical
  multi-exon intron chains or monoexonic CDS structures in either selected set
  currently fail closed. This includes reference isoforms differing only in
  terminal boundaries. They are not silently deduplicated and cannot inflate
  TP. This may block real scoring. Controller/author must settle multiplicity
  semantics before extending this producer; the approved document does not
  specify a duplicate policy sufficient to justify inventing one here.
- **Independent read-evidence provenance is not connected to this producer.**
  The existing alignment driver/table remain as before, including their
  external-reference-gate boundary. The new reference gate does not certify old
  alignments. No HiFi/ONT coverage, independently verified splice P/R, complete
  chain support, or withholding-equivalence receipt was produced. This is a
  remaining requested integration, not completed work.
- The manifest binds declared decoder code/commands/output bytes/checkpoint,
  but the separately owned decoder must actually emit and use this contract.
  No decoder or training run was invoked and no real prediction was produced.
- The strict GFF adapter requires ordered gene/transcript/features and explicit
  exons; unsupported feature forms fail instead of silently normalizing. Real
  compatibility may require a reviewed normalization producer, not relaxation
  of identity or denominator checks.
- No resume/overwrite: failures may leave partial comparison directories, but
  no final receipt is published until verification and statistics finish.
- Author HiFi/ONT-only coverage and withholding policy remain unresolved. No
  new authority booleans or final-decision bypass were introduced.

Controller should review these gaps before operational promotion. The unchanged
training path can proceed through its own approved gates independently.

## Round 1: five independent-review corrections

- Fix base verified: `8803afb44189e7510e67e39976676dcad664c164`.
- Implementation commit: **`64b6d5bf2494bb5b23e061c6c0917b4bbafcdbb9`**.
- Read the complete controller review at
  `notes/recovery_20260916_1251/B.producer.independent_review.md` in main,
  read-only. No main changes, other-worker files, trainer/finalizer changes,
  metric math/policy changes, SSH, jobs, subagents, merges or pushes.
- **350 transport schema unchanged**. Controller independent re-review is
  required; no operational promotion is claimed.

### R1: exact bound training source/package inventories

`verify_training_code` locates the training checkout through the audited
`src/transgenic/training/b5_runtime.py` implementation path. It enforces the
existing training rule: all Python files beneath that checkout's
`src/transgenic`, plus `train/train_HyenaTransgenic_MVP.py`, exactly, with
canonical paths and matching bytes. This is **not** the current scoring
checkout's potentially expanded source set. It requires exactly the five
recorded external packages (accelerate, duckdb, safetensors, torch,
transformers), each a nonempty version string. All four shared package fields
must equal finalizer metadata. Duckdb is mandatory but cannot be cross-compared
because the unchanged finalizer runtime-package schema omits it. Historical
version strings are not compared with the installed scoring runtime.

The two initialization implementation paths/hashes must identify their exact
locations in that consumed code inventory. Semantic reconstruction uses the
existing weights_init implementation only when its own code, b5_runtime, and
configuration_transgenic bytes equal the bound training copies. A different
historical semantic implementation fails closed for a reviewed verifier adapter;
no arbitrary source from a supplied checkout is imported or executed.

Tests reject one-file and omitted-loader manifests and incomplete/extra/blank
package schemas. The positive relocated-checkout test includes an extra
training module absent from the scorer checkout and historical package strings;
it succeeds. Adding an unlisted module or deleting a listed module fails.

### R2: source bytes versus actual initialization claims

The verifier now requires canonical source checkpoint/run-config/optional-file
paths, a real embedded source recipe with the loader's required fields and
zero evidence channels, and the canonical digest of **run_config['recipe']**
using `weights_init._canonical`/`_sha256_bytes`. It reconstructs source semantic
fields and origins through the actual `model_kwargs`, HyenaTransgenicConfig,
`_manifest`, and `_manifest_record`; source recipe contradictions and present
optional source-config semantic fields are checked. Source safetensors must
open as a nonempty zero-channel file without an evidence projection, and their
metadata must equal the audit. Source/target semantic equality is rechecked.
This does not instantiate or reload a full-size model.

The positive fixture is no longer a shape-only hand-built initialization audit:
it creates actual tiny Transgenic/LED models, saves the zero-channel source,
and runs the **real `load_weights_only`**, then the real finalizer publisher.
Only the remote Hyena factory uses the existing local embedding test double;
there is no download, training, GPU, or forward pass. Production-labelled
hardware/update metadata remain explicit synthetic test declarations, not
measurements. The full350 test still stubs biological/candidate/tool authority.

Seed-123 remains a byte-consistent **declared initialization source**, not proof
of the intended historical **best** selection. A trusted concrete source pin
for that historical choice remains an operational prerequisite; the result's
provenance caveat now says this explicitly. No new self-attestation switch or
checkpoint-authority schema was added.

### R3/R4: no silent ID collapse or missing-decode-as-zero

Preflight rejects noncanonical attribute keys and ID/Parent tokens. The supported
canonical identifier alphabet is `[A-Za-z0-9_.:-]+`; Parent may be a comma-separated
list of unique such IDs. Literal whitespace (including nonbreaking spaces),
empty identifiers, and silent percent/other normalization are not repaired.
Parsed gene and per-gene transcript identity sets must exactly conserve the
preflight inventories. Tests include the reviewer's two-transcript whitespace
collision, additional whitespace forms, and simulated parser inventory loss.

A zero-byte, whitespace-only, or comment-only empty file now fails. A completed
zero-transcript output must include the exact `##gff-version 3` header; genuine
header-only GFF3 remains valid. Decoder owners must still publish transport only
after successful completion, not convert failed generation into header-only
biological zero. Transport field names/counts and existing nonempty GFF inputs
have not been changed.

### R5: final config JSON and audited target semantics

`config.json` is read through the duplicate-key-rejecting JSON reader, must be
an object containing audited semantic fields, and must reconstruct the target
semantic manifest exactly. Raw versus reconstructed values are checked to
prevent silent configuration normalization. The source/target relation is
checked with the existing loader helper, including the allowed 0-to-8 channel
transition. Tests reject invalid JSON, non-object JSON, duplicate keys, and
valid JSON with inconsistent target dimensions.

### Actual round-1 validation

Before implementation, the new review suite had **22 substantive missing-guard
failures** (invalid artifacts were accepted) and one additional expected-exception
type mismatch in the bound-checkout deletion test. The strengthened real-loader
positive fixture first passed the unchanged 20 producer/integration tests.

Final fresh commands, from the B worktree:

```sh
PYTHONPATH=/tmp/b6-trackb-testdeps:src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  tests/test_mvp_score_producer_review.py -q
# 27 passed, 42.05 s

PYTHONPATH=/tmp/b6-trackb-testdeps:src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  tests/test_mvp_score_producer.py tests/test_mvp_score_producer_integration.py \
  tests/test_mvp_scoring.py tests/test_mvp_scoring_driver.py \
  tests/test_mvp_metrics.py revision/scripts/tests/test_evidence_support.py -q
# 115 passed, 21 subtests passed, 29.19 s

PYTHONPATH=/tmp/b6-p2-readiness-1535112/.testdeps:/tmp/b6-trackb-testdeps:src \
  /home/wyim/miniconda3/envs/pdb/bin/python -m pytest \
  tests/test_mvp_cli.py -k candidate -q
# 16 passed, 29 deselected, 24.11 s
```

One intermediate full350 regression exposed nondeterministic safetensors
alias-metadata key ordering when independently saving equal source tensors.
Tensor payloads matched but file bytes did not. The fixture was corrected to
make all three runs consume **one serialized baseline**, as the real contract
requires; no production identity check was weakened. The complete regression
command then passed. Compileall for the producer and three task test modules
and `git diff --check` also passed. Graft queried before edits and refreshed.
Advisor was attempted but unavailable due to client/model-version mismatch;
no advisor review is claimed.

**Unchanged blockers:** real GFFCompare behavior and frozen-reference smoke,
duplicate-isoform multiplicity policy, independent read-splice provenance,
coverage/HiFi author policy, withholding, historical seed-123 best pin, and
operational decoder execution remain unverified/unresolved. All previous
non-authority and non-empirical limitations remain in force.
