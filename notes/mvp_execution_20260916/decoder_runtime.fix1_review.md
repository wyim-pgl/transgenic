# Decoder runtime fix 1 — independent scoped review

## Verdict

Reviewed detached `6366d8a` against `e6950b7`, the neutral integrated review's I1–I3 and fix diff, and `notes/MVP_DECODER_RUNTIME_FIX1_REPORT.md`. **ACCEPT the scoped code fixes for their explicitly pinned d99b1f8 interface; NOT production authorization or final-combined compatibility.** No new Critical/Important fix defect found within that supported interface. The final combined GPU code `772c38c` is outside it and is deterministically rejected (see operational blocker below).

This is code/interface acceptance, not proof that an actual d99b1f8 receipt and deployment pass every closure check. Diagnostic validation may proceed only within controller authorization; nothing here authorizes SSH, jobs, training, decoding production data, or scientific claims.

## I1–I3 disposition

### I1 — closed in implementation

`revision/scripts/mvp/mvp_decode_runtime.py:67–82,109–139,169–175,198–209` removes the ROOT exemption. Loaded current project files map by exact relative path to the checkout derived from the bound canonical probe path. Their hashes must equal the probed identity; unlisted local implementation fails. The concrete model/config/tokenizer/encoder-init modules are imported before construction, and model/config objects must be the concrete project classes. Original receipt-declared imports are all rehashed, even if not currently loaded; relocating a project checkout does not release the original files from retention.

The closed `DECODER_ONLY` list (`:25–39`) is precisely:

- `revision/scripts/mvp/{mvp_decode_panel,mvp_decode_runtime,mvp_verify_panel_inputs,mvp_prepare_panel_inputs,mvp_produce_counts}.py`;
- `src/transgenic/evidence/{mvp_decode,mvp_windows,mvp_tracks}.py`;
- `src/transgenic/model/constrained_decoding.py`;
- `src/transgenic/utils/{gsf,gsf_contract,gsf_grammar}.py`.

Its relative-path/hash mapping is canonically JSON-hashed with a trailing newline and compared to independently supplied `decoder_code_sha256`. It is neither a directory exemption nor a caller-chosen list. An existing probe import identity takes precedence over this decoder audit; the audit cannot authorize changed probe-consumed code. Model/config/tokenizer/encoder-init are not exceptions.

The three `PROBE_CODE` constants (`:18–22`) independently match SHA256 of the actual d99b1f8 Git objects for `real_probes.py`, `run_bounded.py`, and `make_launch_readiness.py`. Probe and preparation import identities are mandatory; all three sibling files are rehashed. Actual d99b1f8 `real_probes.py:652–679` snapshots imports before its final `run_bounded` import, so allowing that one inventory omission while independently pinning its bytes is justified. This fixes applicability, not execution attestation.

### I2 — closed for the actual d99b1f8 completed-wrapper interface

`mvp_decode_runtime.py:42–64,153–159` consumes the independently digest-pinned `generation_cache_parity.json`, exact sibling DONE digest plus newline, absence of FAILED (including dangling symlink), successful typed wrapper status, and matching 64-lowercase-hex wrapper/nested measurement binding. It then validates the nested measurement, not wrapper top-level measurement fields. Raw measurement alone is rejected. DONE joins the repeated identity audit.

Checked against actual d99b1f8 `run_bounded.py:151–166,187–207`: it verifies bindings, validates subprocess completion and measurement, publishes nested `measurement` with `pass=true`, and then digest-bearing DONE; failure removes DONE and publishes FAILED. The new `tests/test_mvp_decoder_runtime.py:26–98` exercises that real Git runner's subprocess/publication with explicitly synthetic CPU measurement and substituted allocation/tooling, not a real GPU receipt. No equivalence between an operator pin and cryptographic execution proof is asserted.

### I3 — predicate incompatibility closed; deployed-loader verification remains pending

`mvp_decode_runtime.py:119–128,189–197` handles both bare `transformers_modules` and dotted descendants. Every such loaded file requires exact probed path and hash under the actual module cache. The scaffold exception requires empty bytes (therefore the empty-file hash), exact `__init__.py`, and cache-relative components exactly matching the imported package name. Other implementation files still require immutable snapshot source-hash membership as well as exact probe import identity. External implementation remains exact-path/hash checked. No blanket remote-code waiver or immutable-asset placeholder is introduced.

Module-cache relocation is explicitly unsupported; original module cache/imports and model/tokenizer asset-tree identities must remain available. The real installed-loader test covers 4.46.3, including bare/generated package initializers. The 4.57.6 conditional test is skipped when that version is absent. This closes the demonstrated helper defect, **not** actual 4.57.6/SIF certification.

## Operational blocker: final combined code is not supported yet

The user reports independent combined ACCEPT for `772c38c`; this review does not independently re-review that branch. I verified its actual three Git source objects and their differences from d99b1f8. All three hashes differ:

| File | 772c38c SHA256 (not accepted by this decoder) |
|---|---|
| real_probes.py | `a40b46b3e2df1f5a80a4357d06c0e77bf1cdedd66c0fd22582df7de0541bbc58` |
| run_bounded.py | `8b2f4042e03732c16d633fb66444b1b91e87cef54d3c696104775beebf19a932` |
| make_launch_readiness.py | `73728aee67611575448722cb4180062917ee0c86330749ff1ca648346e75a6d5` |

Focused source-readonly reproduction: construct `probe_identity.imports` from `sha256(git show 772c38c:revision/scripts/mvp/<name>)` under `/retained/combined/revision/scripts/mvp/`, then call the unchanged `project_closure(identity, {})`. It raises **`ValueError: unreviewed probe/handshake implementation`** at `mvp_decode_runtime.py:75–76`, before evaluating the decoder operator pin. This is a helper-level compatibility reproduction, not an end-to-end fabricated GPU receipt.

This is an explicit limited supported-version contract, not a new false-acceptance defect: the fix report and source deliberately select d99b1f8. It is nevertheless a **production integration blocker** for using final-combined receipts, not a harmless diagnostic warning. Combined code acceptance does not make the hashes interchangeable. Follow-up must review and explicitly bind the selected final handshake (including its resource/tooling changes), add a positive completed-wrapper regression for that exact source, regenerate/review the affected decoder operator pin, and validate a genuine completed receipt in the intended retained deployment. Do not simply disable the pin or copy receipt-selected hashes into the allowlist. Updating three hashes alone is not a claim that the full live import/asset closure will pass.

## CUDA, unchanged contracts, and publication truthfulness

`mvp_decode_runtime.py:85–92` requires CUDA availability, a valid CUDA index, and a CUDA allocation attempt. `mvp_decode_panel.py:101–104,131–143` defaults to CUDA and passes the gated device to model loaders and tensors. CPU is no longer an operational default or public bypass. Runtime verification remains before loading, after each model load and at final publication (`:108,140,160–171`). Zero-sized allocation is an allocation/runtime check, not successful generation or a CUDA-kernel benchmark.

The actual base-to-fix diff leaves D1 rasterization, D2 publication implementation, COMPLETE-panel contract, and 350-row transport unchanged. No regression reopening those previously reviewed contracts was found in the caller change. The report correctly limits D2 to closed-temp atomic no-replace visibility, not directory-fsync crash durability: pre-link failure cannot publish; **post-link cleanup failure may leave a complete valid final file alongside a failure diagnostic**. That is not partial serialization. A diagnostic/FAILED state is not production/scientific approval.

## Evidence and limits

- Graph-first map/query/skeleton used in this isolated worktree; graph estimate total about 1,009,911 tokens saved. Actual changed runtime/caller/test/report diff reviewed; relevant truncated runtime/caller spans opened. Historical authority read from Git, not assumed from an old report.
- `git diff --check e6950b7..6366d8a` passed. Five changed paths only. The four concrete project model/config/tokenizer/encoder-init files have no diff between d99b1f8 and 6366d8a; this does not establish their runtime import inventories.
- Supplied controller evidence: **28 passed, 1 skipped in 16.34 s**. Committed implementer evidence: **135 passed, 1 skipped in 404.94 s**. These were not independently rerun here, as requested. The skip is actual Transformers 4.57.6 not installed, not SIF certification. No duplicate suite; only the focused final-combined compatibility reproduction above.
- Advisor consultation failed due to configured client/model-version incompatibility; no advisor conclusion was used.
- Genuine pinned-SIF/CUDA/remote-tokenizer/cache/grammar/8192-total-token execution, real frozen reference/evidence, completed three-seed runs and real 350-output/GFFCompare transport remain outstanding. A's short numerical parity probe and CPU fixture success do not establish these. Structural scoring, coverage/withholding policy and scientific approval remain separate.
- No source edits, SSH, jobs, subagents, merge or push. Only this review report is intended for commit.
