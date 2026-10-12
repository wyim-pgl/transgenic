# Eval decoder repin — 2026-09-21 PDT

Worktree `/tmp/mvp-seed123-integration-20260917`, branch `mvp-seed123-integration-20260917`,
base `9603e20`. Local only. **No commits, no merge/push, no Slurm submission, no remote
execution, no rename of existing receipts, no change to frozen release artifacts.** Changes
are left uncommitted in the worktree, beside the other owners' pre-existing uncommitted work.

Scope: `notes/TODO_FULL_RECONCILIATION_20260917.md` §1.1 and
`notes/TODO_ACTIVE_AUDIT_20260917.md` rows 05 / 14 —
"새9603 producer/manifest를 소비하는 decoder package/pin/closure를 검토·repin하고
positive/negative 계약시험 준비".

This is a bounded software fix to a byte/closure consumption contract. It is **not** a GPU
receipt, not a readiness proof, not an actual decode, and not any scientific, A11 or raster
acceptance. Those remain false/unexecuted.

---

## 1. What was already correct (checked, not changed)

The reviewed **runtime handshake** repin (precedent item, `seed123_recovery_20260917_1400/RECOVERY.md`)
is present and current in this worktree. `mvp_decode_runtime.PROBE_CODE` was rehashed against the
live source bytes and all three pins match exactly:

| file (`revision/scripts/mvp/`) | pinned = recomputed SHA256 |
|---|---|
| `real_probes.py` | `7f9507f4fced687e1886c659224d4b778b3486a2b390dac1e8f3c6954e46c7ae` |
| `run_bounded.py` | `8b2f4042e03732c16d633fb66444b1b91e87cef54d3c696104775beebf19a932` |
| `make_launch_readiness.py` | `a5b2c895bd76699718655993d67430b9459c5db07d5f47636feb6acc77ce6de2` |

`PROBE_CODE` was **not touched** by this work. `tasknotes/seed123_eval/REPORT.md` remaining-work
item 2 ("PROBE_CODE remains the reviewed legacy handshake pins") is therefore already superseded.

## 2. What was actually stale

The eval side (`9acd7db` reference-ID resolution through the accepted corpus gene map, `9603e20`
reference-source validation at the atomic publication boundary) changed
`revision/scripts/mvp/mvp_produce_counts.py`, and the panel owner added
`revision/scripts/mvp/mvp_panel_compatibility.py` to the operator-pinned `DECODER_ONLY` closure.
The **decoder consumption contract was never repinned to that closure**. Two concrete holes:

**(a) The producer imposed no closure requirement on the decoder package.**
`mvp_produce_counts.produce()` accepted `payload['decoder']['files']` as any non-empty mapping and
only rechecked the bytes of whatever was listed:

```python
decoder = payload['decoder']
require(decoder['files'] and decoder['commands'] and ..., 'decoder code/commands missing')
inputs.extend(check_file(p, {'sha256': h}) for p, h in decoder['files'].items())
```

A stale or partial decoder package — including a legacy/`d99`-era transport produced before
`mvp_panel_compatibility.py` and the 9603 reference-mapping producer existed — passed this gate
unchanged. The existing synthetic fixture in `tests/test_mvp_score_producer_integration.py:61`
demonstrated the hole literally: it declared the *test file itself* as the entire decoder package
and `produce()` accepted it.

**(b) The decoder recorded less than its own pinned closure.**
`mvp_decode_panel.decode_panel()` built `code_inputs` from a hand-maintained list that covered
13 of the 14 `DECODER_ONLY` entries. Missing: `src/transgenic/evidence/mvp_bootstrap.py` — a
module the decoder process actually loads (`mvp_produce_counts` imports `validate_paired_record`
from it at module scope) and which the operator `decoder_code_sha256` pin already covers. The
evaluation transport therefore published a decoder package that did not cover the reviewed
decoder boundary, and nothing downstream detected it.

## 3. The fix

Isolated diff, three files. No check was weakened and no expected value was adjusted.

**`revision/scripts/mvp/mvp_decode_panel.py`** — extracted the code inventory to a module-level
`decoder_code_paths()`, added the missing `src/transgenic/evidence/mvp_bootstrap.py`, and made the
decoder fail closed before any model load if its own inventory ever drifts below `DECODER_ONLY`:

```python
recorded = {str(Path(path).resolve(strict=True).relative_to(ROOT)) for path in paths}
d.require(set(DECODER_ONLY) <= recorded, 'decoder inventory omits reviewed decoder closure')
```

`decode_panel()` now does `code_inputs = [d.identity(p) for p in decoder_code_paths()]`.
A's loaded model implementation files stay listed beside the closure; probe-consumed identities
still take precedence (unchanged).

**`revision/scripts/mvp/mvp_produce_counts.py`** — extracted the decoder-package check to
`verify_decoder_package(decoder)` and bound it to the reviewed closure:

```python
recorded = {str(Path(path).resolve()) for path in decoder['files']}
missing = sorted(rel for rel in mvp_decode_runtime.DECODER_ONLY
                 if str((ROOT / rel).resolve()) not in recorded)
require(not missing, 'decoder package omits reviewed decoder closure: ' + ', '.join(missing))
```

Per-file byte identity (`check_file`) is still enforced on every recorded entry, as before.

**`tests/test_mvp_score_producer_integration.py`** — the synthetic full-350 fixture now supplies
the real reviewed decoder closure via a `reviewed_decoder_package()` helper instead of declaring
the test file as the decoder. This makes the synthetic transport conform to the contract; it does
not relax it.

## 4. Tests (TDD)

New: `tests/test_mvp_eval_decoder_repin.py` — CPU-only positive/negative contract tests
(SHA256 `46a6f32ce899bd9e0123ee1f49ff45f19f0681ed84a697495e1e4a245fffc4c3`).

All commands run from the worktree with the existing env:

```sh
cd /tmp/mvp-seed123-integration-20260917
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/mvp-final-source-env/bin/python -m pytest -q <targets> --tb=short
```

| step | targets | real result |
|---|---|---|
| baseline (before any edit) | `test_mvp_score_producer{,_integration,_review}.py` | **47 passed** in 65.53s |
| RED (test written, no impl) | `test_mvp_eval_decoder_repin.py` | **6 failed** in 4.24s (missing API) |
| RED (seams extracted verbatim, gate still absent) | `test_mvp_eval_decoder_repin.py` | **3 failed, 3 passed** in 2.56s — coverage test reports the extra item `src/transgenic/evidence/mvp_bootstrap.py`; both rejection tests report `DID NOT RAISE ValueError` |
| GREEN (fix applied) | `test_mvp_eval_decoder_repin.py` | **6 passed** in 2.90s |
| existing producer suites, gate biting | `test_mvp_score_producer{,_integration,_review}.py` | **1 failed, 46 passed** in 59.03s — the synthetic fixture correctly rejected with `decoder package omits reviewed decoder closure: <all 14 entries>` |
| after fixture repin | the same three + `test_mvp_eval_decoder_repin.py` | **53 passed** in 50.84s |
| wider affected scope | `test_mvp_seed123_eval.py test_mvp_decoder_runtime.py test_mvp_reference_mapping.py test_mvp_reference_publication.py test_mvp_panel_compatibility.py test_mvp_pinned_panel.py test_mvp_scoring.py` | **202 passed, 1 skipped** in 581.36s |
| end-to-end decoder | `test_mvp_decoder_integration.py` (includes both `full_publication` traversals) | **24 passed** in 331.96s |
| final confirmation after a whitespace-only indent tidy | `test_mvp_eval_decoder_repin.py test_mvp_score_producer{,_integration,_review}.py test_mvp_decoder_runtime.py test_mvp_seed123_eval.py test_mvp_decoder_integration.py` | **161 passed, 1 skipped** in 454.45s |

`git diff --check` clean. `graft build` completed (301 files, 6 reparsed).

The one skip is the pre-existing version-conditional test, not a waived failure. These suites are
CPU fixtures with stand-ins for the remote encoder/generation/GPU runtime and for GFFCompare.
**They are not GPU or scientific receipts.** No whole-suite green is claimed; suites outside the
scope above were not run.

## 5. Hashes verified by recomputation (2026-09-21 PDT, this worktree)

Recomputed with the production algorithm in `mvp_decode_runtime.project_closure`
(`sha256(json.dumps({relative: sha256}, sort_keys=True, separators=(',',':')) + '\n')`),
not read from any record:

```
decoder_code_sha256 = cec4611de54154ebb47d2914d76f2005d88595811c74a871b9a146365f0a1be3
```

| `DECODER_ONLY` entry | SHA256 |
|---|---|
| `revision/scripts/mvp/mvp_decode_panel.py` | `c1bde2959809b205781d1423e15e65017c0ad90daa87e1f1631ebc5906a6c132` |
| `revision/scripts/mvp/mvp_decode_runtime.py` | `b0ffac46cff61f23262f5f8455a2217f823278b7b0225f4b9a445f8957db7d03` |
| `revision/scripts/mvp/mvp_panel_compatibility.py` | `ffc67337496cae5c2b3cdcf3bfb708bac3d7d74f09ec96f4b618fc5c3e0a0aa2` |
| `revision/scripts/mvp/mvp_prepare_panel_inputs.py` | `3f7ae796f142c8e41221c79d9ea088b2d2c2c66a430f5b08c645df517c37e8ac` |
| `revision/scripts/mvp/mvp_produce_counts.py` | `1c8f3d4472c0f7a68954d3b8ff122a904a32e1a51a5587f7cdf8cf35880dd149` |
| `revision/scripts/mvp/mvp_verify_panel_inputs.py` | `8478a39077dcb7d94f65218f719b7e9380757f44a3c799dcc6dedf933b621bae` |
| `src/transgenic/evidence/mvp_bootstrap.py` | `21555321217318ec162dc52e528b5f6540513165ca9b469c5312969ad0908172` |
| `src/transgenic/evidence/mvp_decode.py` | `4b05e3737200d8ace7c19dd581dcc612c7b0abe2a42df8a07bb1bc80ec226201` |
| `src/transgenic/evidence/mvp_tracks.py` | `80a27c1909f41041c50af8d719856bbea914b80b804c6b6a4e07b715718922c6` |
| `src/transgenic/evidence/mvp_windows.py` | `7583e0886418c606b0f0fd5d34440553390e1064d233f5e2ad7200603d1a4775` |
| `src/transgenic/model/constrained_decoding.py` | `ea3229f7eb266a2c68a4f0c59fad10601ed6e35bbc3d325b22eed721055c32ff` |
| `src/transgenic/utils/gsf.py` | `d82eeb72dd868e1d0f97d2afd96874407284581740b2dbb0123850480a728ea9` |
| `src/transgenic/utils/gsf_contract.py` | `c71d2381f22faec1e4e7030ba7a991eb3f3c25b38c22334e5c60c91cd98bd8c9` |
| `src/transgenic/utils/gsf_grammar.py` | `4075dde4b837d3a94d21a60f7248f5308d5fa05b2c0a6bcf933020fbd1c098cb` |

Also recomputed: `tests/test_mvp_score_producer_integration.py` =
`65397de928cda4e470d9e72712f5fef7732652bebf05dad08defcdb661bfff8a`.

**`cec4611d…` is a computed value of the current worktree bytes, NOT an approved operator pin.**
The recorded historical values `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb`
(`notes/MVP_FINAL_SOURCE_INTEGRATION_REPORT.md`) and
`c0e8c141b1963abe1bd5b1b167bada5e3089057da7c4855dd61aa4fb4610b0d7`
(`notes/MVP_REFERENCE_MAP_FIX_REPORT.md`) are both stale against this source line and must not be
supplied as `runtime_preflight.decoder_code_sha256`. It will change again if any writer edits a
`DECODER_ONLY` file, so it must be frozen and independently reviewed at the controller's source
freeze, not copied from here.

## 6. Unresolved / needs author or controller decision

1. **Independent review of this change has not been obtained.** The precedent shape is
   repin → controller diff → real hash verification → focused tests → independent review.
   The first four are done here; the review is not.
2. **Operator `decoder_code_sha256` freeze.** §5 gives the current computed digest only. Freezing
   and reviewing it is controller work, and must happen after all source writers finish.
3. **Cross-checkout consumption is still unsupported.** `verify_decoder_package` matches
   `DECODER_ONLY` by absolute path under the producer's own `ROOT`, consistent with the existing
   exact same-checkout/same-packages readiness gate. A relocated or cross-version counts consumer
   needs a reviewed adapter; this change deliberately does not invent one.
4. **Out of scope and untouched:** independent junction/GFF → counts linkage, the ONT full
   wrapper and real-tool fixtures, HSP/phase policy, the extern OOM gate, and all seed123
   scientific/A11/raster acceptance — these remain unfinished or false.
5. No actual decode, training, GPU gate or panel regeneration was performed, and no existing
   seed123 artifact or prior result was modified.

Author: Won <wyim@unr.edu>.
