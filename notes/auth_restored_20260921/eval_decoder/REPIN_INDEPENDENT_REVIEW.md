# Independent review — eval decoder consumption-contract repin

Reviewer: independent, did not author the change. 2026-09-21 09:11–09:35 PDT.
Subject: `notes/auth_restored_20260921/eval_decoder/REPIN.md`, worktree
`/tmp/mvp-seed123-integration-20260917`, base `9603e20`.

Everything below marked **measured** was executed by me in this worktree. Nothing was
edited, committed, submitted or pinned. No Slurm, no remote, no GPU.

**Verdict: APPROVE.** Nothing CRITICAL or HIGH. Two MEDIUM items and four LOW items are
recorded below; all are record/landing-order concerns or residual-scope statements, not
defects in the gate itself.

---

## 1. Claims tested

### Claim 1 — the hole was real. **CONFIRMED (measured).**

`git show HEAD:revision/scripts/mvp/mvp_produce_counts.py` contains exactly one decoder
gate: `require(decoder['files'] and decoder['commands'] and ...)` followed by
`inputs.extend(check_file(p, {'sha256': h}) for p, h in decoder['files'].items())`. Any
non-empty mapping passed. There is no closure requirement at HEAD.

`git show HEAD:tests/test_mvp_score_producer_integration.py` line 61 is literally
`'decoder':{'files':{str(Path(__file__)):w.sha256_file(__file__)},'commands':[['synthetic-test-only']]}`
— the test file declared as the entire decoder package, accepted by `produce()`. The hole is
demonstrated by the repo's own fixture, as stated.

### Claim 2 — `mvp_bootstrap.py` belongs. **CONFIRMED (measured).**

`revision/scripts/mvp/mvp_produce_counts.py:19` — `from transgenic.evidence.mvp_bootstrap
import validate_paired_record`, module scope, used at line 193. The decoder process loads it.
`git show HEAD:revision/scripts/mvp/mvp_decode_runtime.py` shows
`src/transgenic/evidence/mvp_bootstrap.py` already inside `DECODER_ONLY` at HEAD, while
HEAD's `decode_panel()` hand-maintained `files` list does not contain it. The operator pin
covered it; the decoder's own inventory did not. Now present at
`mvp_decode_panel.py:decoder_code_paths()`.

### Claim 3 — the fixture tightens rather than relaxes. **CONFIRMED, with a caveat.**

`tests/test_mvp_score_producer_integration.py:8-14` replaces the single-self-file package
with `reviewed_decoder_package()`, which enumerates the real reviewed closure and hashes the
real bytes. It conforms to the contract; it does not weaken it, and it does not stub, skip or
monkeypatch `verify_decoder_package`. I confirmed the gate is live in that test by running
the suite (§2) — the implementer's own record shows the unrepinned fixture failing with
`decoder package omits reviewed decoder closure`, which is the expected direction.

Caveat (LOW-2 below): the fixture derives its package from `decoder.decoder_code_paths()`,
the same function the gate is defined against, so the positive path is self-consistent by
construction and can never detect drift. The independent bite comes from the two rejection
tests, which I verified separately.

### Claim 4 — fail-closed is real. **CONFIRMED by my own execution, not accepted on assertion.**

I did not rely on the implementer's test file for this. I injected drift and enumerated the
gate myself (`/tmp/mvp-final-source-env/bin/python`, worktree root):

- **Decoder side.** Monkeypatching `mvp_decode_runtime.DECODER_ONLY` to include two real
  files absent from the decoder's list makes `decoder_code_paths()` raise
  `ValueError: decoder inventory omits reviewed decoder closure`. **Measured.**
- **Ordering.** By `inspect.getsource(decode_panel)`, `decoder_code_paths` appears at source
  offset 1365, `load_baseline_model` at 2239 and `load_final_model` at 2564. The inventory
  gate runs before any model load. **Measured.** (Note `require_cuda_device` runs earlier
  still, so on a CPU host the CUDA gate fires first — that is pre-existing and correct.)
- **Producer side, exhaustively.** I dropped each of the 14 `DECODER_ONLY` entries from a
  full package one at a time and called `verify_decoder_package`. Entries not enforced:
  **none — all 14 raise.** **Measured.** This is stronger than the implementer's test, which
  drops only `DECODER_ONLY[-1]`.
- The full 19-path package (14 closure entries plus A's 5 model-implementation files) is
  accepted, and `check_file` byte identity is applied to all 19. **Measured.**

The implementer's reported intermediate state (`DID NOT RAISE ValueError` when the seams
existed but the gate did not) is consistent with the final state I tested: the gate is now
present and does raise.

### Claim 5 — no scope creep. **CONFIRMED for the gate; see MEDIUM-1 for entanglement.**

- `check_file` is untouched — no hunk in `git diff -- revision/scripts/mvp/mvp_produce_counts.py`
  reaches it, and byte-identity semantics are unchanged (still `check_file(path, {'sha256':
  digest})` per recorded entry).
- Runtime handshake: I recomputed all three `mvp_decode_runtime.PROBE_CODE` pins against live
  bytes. `real_probes.py`, `run_bounded.py`, `make_launch_readiness.py` — **3/3 MATCH**.
  `PROBE_CODE` was not touched by this change. **Measured.**
- Nothing on the GPU `b660` line is reachable from this worktree's uncommitted state.

---

## 2. Reported results — reproduced by me

| target | reported | my run | verdict |
|---|---|---|---|
| `test_mvp_eval_decoder_repin.py` | 6 passed | **6 passed** in 2.50s | confirmed |
| producer suites + repin (after fixture repin) | 53 passed | **53 passed** in 51.85s | confirmed |
| wider scope (7 suites) | 202 passed, 1 skipped | **202 passed, 1 skipped** in 573.21s | confirmed |
| `test_mvp_decoder_integration.py` | 24 passed | **24 passed** in 314.92s | confirmed |
| final combined (161 passed, 1 skipped) | — | not re-run; it is a strict subset of the four rows above, all of which I reproduced | accepted on overlap |

**The single skip is pre-existing and version-conditional. Confirmed.** `pytest -rs` names it:
`tests/test_mvp_decoder_runtime.py:203: pinned 4.57.6 not installed; CPU 4.46.3 test is not
SIF validation`. The same `pytest.skip` call exists at HEAD (`git show
HEAD:tests/test_mvp_decoder_runtime.py`, line 188). It is unrelated to this change and is not
a waived failure.

---

## 3. Pin status — the unmeasurable-staleness caveat

**The implementer's caveat is over-conservative. The conclusion is measurable, and I measured
it.** The implementer inferred pre-existing staleness from the `DECODER_ONLY` tuple change
without computing anything. Two independent measurements settle it:

**(a) Both recorded pins reproduce exactly from their own commits' trees.** Recomputing
`sha256(json.dumps({relative: sha256}, sort_keys=True, separators=(',',':')) + '\n')` over
each commit's own `DECODER_ONLY` and its own blob bytes:

| commit | tuple size | recomputed digest | recorded as |
|---|---|---|---|
| `b660bf1` (2026-09-16 20:04 PDT) | 13 | `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb` | `notes/MVP_FINAL_SOURCE_INTEGRATION_REPORT.md:12` |
| `66f130d` (2026-09-16 22:59 PDT) | 13 | `c0e8c141b1963abe1bd5b1b167bada5e3089057da7c4855dd61aa4fb4610b0d7` | `notes/MVP_REFERENCE_MAP_FIX_REPORT.md:64` |

Both are genuine operator pins recorded at their own reviewed source states, exactly as the
controller stated. Neither is stale *at its own state*.

**(b) Both pins were already stale against the committed base of this line, before any
uncommitted edit existed.** Recomputed at `9603e20` over its own 13-entry tuple and its own
blobs: **`b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088`** — equal to
neither pin. This is a direct measurement over git objects and needs no inference about
uncommitted bytes at all. It is a stronger result than the implementer established.

**(c) The tuple argument is a proof, not an inference.** Both pins were computed over a
13-entry `DECODER_ONLY` (measured above, and `mvp_panel_compatibility.py` is absent from the
tuple at both commits). The pre-edit worktree already carried the 14-entry tuple — the panel
owner's `mvp_decode_runtime.py` is dated 09-17 14:11, the repin 09-21 08:38/08:57. A digest
over a 14-key dict cannot equal one over a 13-key dict short of a SHA-256 collision. So
staleness at the pre-edit state holds independently of the pre-edit bytes.

**What genuinely remains unmeasurable, and permanently so:** the *numeric value* of the
pre-edit digest. It requires the 09-17 uncommitted bytes of `mvp_decode_panel.py` and
`mvp_produce_counts.py`, which are not in git (never committed), not in `git stash` (empty),
and not in any `.orig`/`.bak`/swap file (none present). Reconstructions using HEAD blobs for
those two files are not that state and must not be passed off as it — I computed such a
reconstruction (`57c82a09…`) only to confirm it differs from both pins, and it has no standing
as a measurement of the pre-edit tree. That value is gone. The *conclusion* that both pins
were stale, however, is established by (b) and (c) and needs no such value.

I also independently reproduced the implementer's §5 current-worktree digest:
`cec4611de54154ebb47d2914d76f2005d88595811c74a871b9a146365f0a1be3` — **matches exactly**. It
is correctly labelled a measurement and not a pin. **I issue, propose and compute no
replacement operator pin.** Freeze remains controller work at source freeze.

---

## 4. Entanglement with other owners' uncommitted work

The worktree carries 19 modified and 16 untracked paths from several owners. `git diff` against
`9603e20` for the two source files under review therefore shows a **mixture**: the reviewed
repin plus the seed123 profile owner's `evaluation_profile` / `verify_run_inventory` /
`verify_evaluation_readiness` / `prediction_inventory(profile=...)` work and the
development-profile job machinery in `decode_panel()`. The diff alone cannot attribute hunks.

Attribution is nevertheless clean, by mtime: the profile/panel work is dated 09-17
(`mvp_profiles.py` 11:28, `mvp_panel_compatibility.py` 11:38, `mvp_score.py` 14:03,
`mvp_decode_runtime.py` 14:11) and the reviewed change is dated 09-21 (08:36–08:57). The repin
is a delta applied on top of a settled earlier state, not concurrent co-editing of the same
logic. The reviewed additions — `decoder_code_paths()`, `verify_decoder_package()`, the
fixture helper, the new test file — are self-contained: `verify_decoder_package` uses only
pre-existing `ROOT`, `require`, `check_file` and `mvp_decode_runtime.DECODER_ONLY`, and touches
none of the profile code paths. **I can separate them, and I do.** Suite counts in §2 are
against the shared working state, as the controller noted, and are reported as such.

---

## 5. Findings

### MEDIUM-1 — the change is not independently applicable to `9603e20`

`revision/scripts/mvp/mvp_decode_panel.py`, `decoder_code_paths()`:
`here.with_name('mvp_panel_compatibility.py')` is resolved with `Path(...).resolve(strict=True)`,
and the closure assertion requires `set(DECODER_ONLY) <= recorded`. `mvp_panel_compatibility.py`
is **untracked** (another owner's work), and the 14th `DECODER_ONLY` entry is **uncommitted**.
Committing these four files onto a clean `9603e20` yields `FileNotFoundError` at
`decoder_code_paths()` on every decode — a hard failure before model load. Concrete failure:
any landing order that takes the repin without the panel owner's `mvp_panel_compatibility.py`
and the 14-entry tuple breaks the decoder outright. Action: the controller must land these
together; worth one line in the record.

### MEDIUM-2 — the producer gate is a completeness gate, not an approval gate

`revision/scripts/mvp/mvp_produce_counts.py`, `verify_decoder_package()`: presence is checked
against `DECODER_ONLY`, and bytes are checked with `check_file` against **this checkout's**
files. Nothing binds the recorded decoder digests to the operator `decoder_code_sha256` pin.
Concrete consequence: an unreviewed local edit to a `DECODER_ONLY` file is recorded by the
decoder and accepted by the producer, because local bytes agree with the record. The operator
pin is enforced on the decoder side in `mvp_decode_runtime` (`'operator decoder code pin
mismatch'`), not here. That split is defensible and matches the stated scope, but the record's
§3 framing ("bound it to the reviewed closure") overstates it. Action: state the limit in the
record — this gate proves completeness and same-checkout byte identity, not reviewed approval.

### LOW-1 — one contract test is vacuous as written

`tests/test_mvp_eval_decoder_repin.py:26-29`. `decoder_code_paths()` raises on drift before
`assert missing == []` is ever evaluated, so the assertion can never report a non-empty list;
the test would error, not fail with the diagnostic it implies. It still fails on drift, so it
is a working guard — just not the assertion it appears to be.

### LOW-2 — positive paths are self-consistent by construction

`tests/test_mvp_eval_decoder_repin.py:20-23` (`_full_package`) and
`tests/test_mvp_score_producer_integration.py:11-14` (`reviewed_decoder_package`) both build
the package from `decoder.decoder_code_paths()`. The producer can therefore never reject in
those paths while the decoder is self-consistent. The suite's real discriminating power is in
the two rejection tests — which I verified bite for all 14 closure entries, so the suite is
adequate in aggregate.

### LOW-3 — record §2(b) "13 of 14" is not verifiable from the diff

Against `9603e20`, `decode_panel()`'s list covers **12** of the current 14 entries (missing
both `mvp_bootstrap.py` and `mvp_panel_compatibility.py`), and 12 of HEAD's 13. "13 of 14"
holds only if the panel owner had already added `mvp_panel_compatibility.py` to the pre-edit
list — plausible, since they added it to `DECODER_ONLY` in the same 09-17 pass, but
unverifiable now for the same reason as §3. The record should say it is counting against the
pre-edit list, not against `9603e20`, and should note that `decoder_code_paths()` as written
adds two entries relative to HEAD, not one.

### LOW-4 — record §6 understates what is measurable

The staleness caveat can be, and now has been, converted to a measurement (§3b, §3c). Only the
pre-edit digest *value* is permanently lost.

---

## 6. What I verified myself vs accepted on the record

**Verified by my own execution:** the HEAD producer gate and HEAD fixture (git objects); the
module-scope `mvp_bootstrap` import and its absence from HEAD's decoder list; the drift gate
raising; gate ordering before model load; all 14 closure entries individually enforced; full
package acceptance with byte checks; 3/3 `PROBE_CODE` pins matching; `check_file` untouched;
4 of 5 reported suite results, including `-rs` identification of the skip and its presence at
HEAD; the current-worktree digest `cec4611d…`; both recorded pins recomputed from their own
commits; the `9603e20` digest; absence of stash/backup copies of the pre-edit bytes.

**Accepted on the record:** the TDD RED/intermediate sequence in §4 of `REPIN.md` (I cannot
re-observe a state that no longer exists, and the final state is consistent with it); the
`161 passed, 1 skipped` combined run (strict subset of runs I reproduced); `git diff --check`
clean and `graft build` completion.

**Not assessed:** anything on the `b660` GPU line; scientific, A11 or raster acceptance;
cross-checkout consumption (`REPIN.md` §6.3 correctly declines to invent it).

---

**APPROVE.** The hole was real, the added closure entry genuinely belongs, the fixture
tightens rather than dodges, and fail-closed behaviour is real on both sides — all confirmed
by my own execution. MEDIUM-1 and MEDIUM-2 are landing-order and record-accuracy items for
the controller to fold in; neither blocks. No operator pin is issued, proposed or computed
here.

Author: Won <wyim@unr.edu>
