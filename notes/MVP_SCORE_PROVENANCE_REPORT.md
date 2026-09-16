# MVP score provenance implementation report

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
