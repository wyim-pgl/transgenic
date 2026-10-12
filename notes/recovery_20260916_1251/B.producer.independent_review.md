# Track B producer — independent read-only review

## Verdict and scope

**Request changes before operational promotion.** The implementation substantially follows the requested partial reference-count producer design, but its positive provenance labels currently overstate the consistency checks performed. Five focused defects below need correction. This is **partial-scope code/spec review, not full scientific acceptance**, training authorization, decoder validation, or empirical evaluation.

Reviewed `/tmp/b6-p2-scoring-1535144`, HEAD `8803afb44189e7510e67e39976676dcad664c164`; implementation `1a9253d02b0202106e29587ff689bb1e2e1f3e72`, against `2276fcb1efdeb124a97a11dced8d73e4d48560e0`. Read the complete controller `B.producer.review.diff` (including its continuation), both producer reports/contracts, the complete `revision/protocols/B6_MVP_PILOT_v1.md`, and all 388 producer lines through the diff. Inspected relevant unchanged finalizer, candidate, loader, parser, panel, and scoring contracts via graft and exact source spans.

No code edits, SSH, jobs, agents, merges, pushes, dependency installation, or broad test reruns. Only this requested review document is written in main; focused probes created disposable local data fixtures and disabled bytecode writes. Advisor consultation was attempted but unavailable due to the configured client/model version mismatch; no advisor conclusion is being claimed.

### Immediate guidance for consumer/decoder planning

- Keep the **350 independent files** contract: frozen 50 loci × seeds 1/2/3 × off/on, plus 50 source seed-123 off files. Preserve exact window, checkpoint hash, raw GFF bytes, and declared decoder code/commands.
- Do not treat `three_seed_identity_verified`, `three_seed_policy_verified`, or `FINALIZER_BYTES_METADATA_INVENTORY_CONSISTENT` as an operational acceptance gate until R1/R2/R5 are repaired. Recorded decoder commands are not witnessed execution.
- Explicitly distinguish completed zero-transcript decode from empty/truncated output (R4); reject ambiguous IDs before legacy parsing (R3).
- Preserve explicit exon rows, genomic forward coordinates, and known strands. Do not silently deduplicate predictions or reference isoforms to get past the current duplicate gate.
- Real GFFCompare 0.12.6 behavior and actual frozen-reference compatibility remain unverified. Neither consumer implementation nor interface agreement closes that gap.

## Actionable findings

All producer line references below are in `revision/scripts/mvp/mvp_produce_counts.py` at the reviewed HEAD.

### R1 — High: incomplete consumed-code inventories pass as verified training provenance

**Location:** lines 89–95; positive labels at 351–352. Existing contract: `src/transgenic/training/mvp_cli_runtime.py:146–172`.

`verify_final` requires only a nonempty `files` mapping and compares the intersection of package names. The unchanged training runtime requires the exact project-owned source closure and all five external packages. A manifest containing one arbitrary existing test file passes the producer, although the training validator rejects it. An empty `external_packages` mapping also passes, making every package comparison vacuous. Cross-seed equality merely repeats the same incomplete declaration.

**Focused reproduction (executed):** import `finalized_fixture`, `p`, and `dump` from `tests/test_mvp_score_producer.py`; create its normal production-labelled fixture in a temporary directory; call `p.verify_final(final, 1, [1], hashes, code)`. It succeeds with one listed file. `mvp_cli_runtime._expected_code_files()` returns 63 here; `_manifest_files(code)` raises `consumed code manifest does not match the exact executed source set`. Set the fixture code manifest's `external_packages` to `{}`, refresh its hash in metadata and the metadata entry in the final manifest, and call `verify_final` again: it still succeeds.

**Fix:** enforce the existing source-inventory/package-schema contract, not merely nonemptiness/intersection. Bind required training module paths and initialization implementation identities to the recorded training checkout. Compare the required shared runtime package fields and require the complete recorded package set, while allowing the documented duckdb/finalizer-schema difference. Do not compare historical training versions blindly to the scoring machine's installed versions. Retain explicit local-consistency, not historical-execution, language.

### R2 — High: initialization bytes are hashed but their audit claims are not checked against those bytes

**Location:** lines 84, 96–100; positive labels at 351–352. Existing contracts: `mvp_finalize.py:84–178`; `weights_init.py:143–286`.

The reused finalizer validator validates an audit's shape and self-hash; it deliberately does not re-open the source files and reconstruct its claims. The producer opens the source files but checks only their recorded byte hashes and top-level `seed == 123`. It never checks that the source weights are safetensors, that `run_config.json` contains the recipe whose canonical digest is recorded, or that the two initialization implementation identities exist/match the consumed code. It also does not connect the recorded source/target semantics to the source/final configs.

**Focused reproduction (executed):** the existing `finalized_fixture` passes `verify_final` while its source weights are literally `b'explicit synthetic source'`, its run config is only `{'seed': 123}`, and both loader implementation paths (`/tmp/source/b5_runtime.py`, `/tmp/source/configuration_transgenic.py`) are nonexistent. This is a fixture limitation that directly exposes missing producer checks, not evidence that actual training emitted those files. These artifacts cannot be mutually consistent outputs of the unchanged `load_weights_only` path, which requires an embedded recipe and readable safetensors.

**Fix:** verify source safetensors readability/metadata as appropriate; require the embedded source recipe and recompute its canonical audit hash using the loader's actual canonicalization (the digest is of `run_config['recipe']`, not the entire run config); reconcile audited semantic fields and implementation hashes with the bound source/final configs and consumed-code manifest. Add a fixture created with the real weights-only loader, rather than a shape-only hand-built audit. No full model load or training rerun is needed merely to reject these demonstrated contradictions.

This is distinct from the acknowledged inability to prove historical execution against wholesale artifact forgery: the current receipt accepts directly checkable internal contradictions. An external trusted identity for the intended seed-123 **best** checkpoint also remains an operational prerequisite if the receipt is to establish that specific historical selection, rather than just agreement with the three initialization declarations.

### R3 — Medium: strict preflight and legacy parser disagree on IDs, silently dropping a transcript

**Location:** lines 147–158, 172–180. Legacy parser: `revision/scripts/07_selfconsistency_stats.py:76–90,93–130`.

The preflight does not strip attribute tokens, while the reused parser does. Preflight accepts `t` and `t ` as distinct transcript IDs. The parser then normalizes both to `t` and overwrites the first transcript. The postflight verifies geometry but never verifies conservation of transcript identities/counts. This violates the strict-wrapper promise of no silent omission and can change TP/FP/FN denominators.

**Focused reproduction (executed):** use a `Chr1` plus-strand gene spanning 1–100; first mRNA/exon span 10–20 with `ID=t;Parent=g` and `Parent=t`; second mRNA/exon span 30–40 with `Parent=g;ID=t ` and `Parent=t ` (trailing literal spaces). `read_gff` succeeds and returns only:

```text
{'t': {'exons': [(29, 40)], 'cdss': [], 'start': 29, 'end': 40,
       'strand': '+', 'chrom': 'Chr1'}}
```

**Fix:** reject whitespace/noncanonical ID/Parent values before calling the legacy parser, and compare parsed gene/transcript inventories with the validated preflight inventory. Do not silently normalize duplicate IDs. Add this adversarial input as a rejection test. This input is malformed; the required behavior is failure, not acceptance with a smaller denominator.

### R4 — Medium: zero-byte files are accepted as completed zero-prediction decodes

**Location:** lines 139–180, 266, 336–340. Contract: `notes/MVP_EVALUATION_INPUT_CONTRACT.md:27–34`.

The published interface specifies a **header-only GFF3** as a completed empty prediction. `read_gff` accepts an entirely empty file (also whitespace-only input) as `{}`. A crash-created or truncated empty file can therefore be assigned a hash and counted as a biological zero, with the producer emitting derived-count provenance. Hash agreement establishes bytes, not decode completion; inode uniqueness does not solve this.

**Focused reproduction (executed):** create a zero-byte temporary `pred.gff3`, then `p.read_gff(path)` returns `{}` rather than raising.

**Fix:** at minimum require the documented GFF3 header for empty output and reject zero-byte/whitespace-only files. Decoder-side transport should be published only after successful completion, with an explicit completion/error distinction; absent or failed decodes must never be converted into zero predictions. Keep valid header-only zero output supported.

### R5 — Medium: final config is byte-bound but not even validated as JSON

**Location:** lines 80–84, 107–109. Existing publisher contract: `mvp_finalize.py:517–557`.

All five payloads are hashed, but `config.json` is never parsed. The existing publisher writes a JSON object from the target config and validates JSON payloads. The producer accepts a final config that no unchanged publisher could have emitted, while labelling the finalizer artifacts consistent. A hash-only config declaration is also insufficient for downstream reconstruction of the decoder model.

**Focused reproduction (executed):** create `finalized_fixture`, replace `final/config.json` with `not JSON`, update only its size/hash in `manifest.json`, and call `verify_final(final, 1, [1], hashes, code)`: accepted.

**Fix:** parse the final config with the duplicate-key-rejecting reader, require a JSON object, and reconcile relevant target semantic fields with the initialization audit. Include invalid JSON and valid-but-inconsistent target config cases. This need not change the finalizer format.

## Checks that are sound within this limited review

- Exact five-payload manifest names and `TRAINING_DONE == b'complete\n'` match the unchanged publisher. Production scope, fixed normalized recipe, 250 updates/effective batch 96, seed/augmentation policy are validated through the existing metadata validator. Paired records are independently schema-checked, duplicate RNs rejected, and reaggregated against the complete candidate valid-RN inventory; paired summary equality is checked. Final weights are opened as nonempty safetensors. These are meaningful checks, but do not repair R1/R2/R5.
- Candidate gates reuse `validate_candidate_records` and `verify_candidate_artifacts`, including hash-bound acceptance/inventory/receipt records and read-only actual corpus/evidence tile metadata comparisons. This is not a new authority schema. It does not independently re-audit every biological candidate QC assertion, and no such new authority should be inferred.
- The reference anchor is read with `git show` from immutable commit `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`. I confirmed its Athaliana MD5 values: FASTA `513e5ef30845ed754b00816a99abbf8e`, GFF `0264c349bc61ccceb546648edec46ebb`. No caller override of that anchor exists. Panel bytes are separately pinned by `mvp_windows.load_panel`; FASTA lengths constrain windows.
- Script 24 restriction is reused with a local-only regex correction; selected reference gene geometry/strand is checked exactly. Predictions are limited to their window and selected by transcript-span overlap with the panel gene, without silently filtering opposite-strand predictions out of the denominator.
- Multi-exon structures include chromosome, strand, and full intron chain. A reported `=` is accepted only after equality of those structures. Monoexonic TP requires exact nonempty CDS intervals plus chromosome/strand. Noncoding monoexonic records remain in denominators but cannot produce exact-CDS TP. Opposite-strand structures cannot become TP. Duplicate non-null structures in either side fail closed, so the reviewed path does not silently inflate TP through many-to-one matching. Multiplicity remains an unresolved policy/operability issue, not a deduplication solution.
- The inventory requires all 350 distinct physical files, exact integer seeds/fractions, exact windows and expected checkpoint hashes; missing/extra/duplicate combinations and partial panels fail. Baseline is linked to the shared initialization source hash, not to fine-tuned fraction-off weights. This binds declared identities, not evidence-channel delivery or historical decoder execution.
- Counts are derived in this invocation; TP/FP/FN and per-comparison artifacts/commands are retained. Existing bootstrap/scoring code is called unchanged. The full diff changes only six task files, not training, bootstrap math, thresholds, legacy parser/restrictor, or protocol.
- Structural metrics stay absent rather than fabricated as zeros; `structural_available=False`, `independent_read_splice_status=ABSENT_NOT_NEGATIVE`, pending author coverage policy, unassessable withholding, and `decision=null` remain. No final decision bypass was found.

## Real-tool semantics and operational gaps

**GFFCompare correctness remains UNVERIFIED, not inferred from the test double.** `command -v gffcompare` found no executable in this environment; the worktree search found only analysis/docs/results, not tool source. I did not fetch source or install/run a real binary. Therefore I cannot certify exact 0.12.6 tmap classification, terminal-boundary tolerance, redundancy handling, unmatched-query enumeration, or output naming. The code checks `=` hits for exact chain identity but does not prove that every eligible exact chain will receive `=`. The fixture hardcodes that equivalence and cannot establish it for the real tool.

Before operational use, run an authorized CPU smoke against real 0.12.6 using hand-counted unique chains (including altered terminal boundaries, same/opposite strands, unmatched queries), mono-CDS cases, empty predictions, and then the actual frozen panel/reference. Resolve duplicate-chain/terminal-variant multiplicity with the controller/author before changing its fail-closed behavior. Do not invent tool semantics or count policy based on this review.

The full350 integration stubs candidate authority, reference authority/lengths, and GFFCompare; its predictions are header-only. It usefully validates transport/finalizer/statistics plumbing, not an empirical full-panel comparison or the actual candidate gate in the same invocation. The separately reported candidate tests do not remove that distinction.

Decoder command lists/code/output hashes do not yet bind per-row execution settings, grammar/beam behavior, actual fraction-1 tracks, model loading, or successful generation. Planning can adopt the transport contract, but the decoder owner must supply and test those execution guarantees. Independent read-splice alignment provenance is still unconnected; no source coverage, HiFi/ONT policy, chain support, or withholding-equivalence acceptance is supplied here.

Output behavior is non-overwriting for the new result directory and exclusive for final JSON files. Failures can leave partial directories and a crash can leave incomplete JSON: downstream must require a successfully parsed final receipt with verified output hashes, not simply `counts.json` or directory existence. No broad arbitrary-output overwrite was found in the normal single-process new-directory path. Input bytes are rechecked before final publication; comparison output hashes are recorded, not a historical-execution attestation.

## Verification evidence and limitations

- Controller-provided broad result: **115 passed, 21 subtests, 20.32 s**. Agent report additionally records **16 candidate tests passed**. These are supplied evidence, not reruns by this reviewer.
- Independently ran `git diff --check 2276fcb..8803afb`: clean. Verified HEAD and six-file diff scope. Worktree retained its pre-existing `.gitignore` modification and untracked `.ignore`; neither was changed by this review.
- Executed only focused local Python probes described in R1–R5 using `/home/wyim/miniconda3/envs/pdb/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and existing `/tmp/b6-trackb-testdeps` plus worktree `src`/`tests` paths. All reproduced the stated acceptance behavior. No real biological data/tool efficacy claim follows from these probes.
- Acceptance after correction should include negative tests for all five defects, a real-loader positive provenance fixture, and a real-tool smoke before promotion. Do not change the approved bootstrap or use missing independent evidence to authorize a final decision.
