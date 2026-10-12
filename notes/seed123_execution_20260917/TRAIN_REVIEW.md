# Independent training review — 2026-09-17

## Verdict: APPROVE (bounded training-code scope)

No Critical/Important defect reproduced in the finished training changes against `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558` in `/tmp/mvp-seed123-integration-20260917`. This is not deployment authorization, whole-suite approval, or real-GPU readiness certification. Evaluation/panel workers' unfinished changes were not reviewed. Controller still owns actual diff review, integration and source freeze.

Reviewed IMPLEMENTATION.md, main INTERFACE_AND_OWNERSHIP.md, the actual scoped diff, the new helper/tests, and targeted caller/context ranges. No source, Git index or HEAD changes; no remote jobs/subagents/dependency installations. Graft was queried first; its query command automatically refreshed graph cache. Advisor consultation failed with `claude_code_version_too_old`; no advisor approval is claimed.

## Contract findings

- `mvp_profiles.py:7-13`: exact integer seeds, explicit development-single/123 versus historical-three-seed/1,2,3. Missing legacy fields default only where intended; explicit null/unknown/cross-profile values reject.
- `mvp_cli_runtime.py:175-245,466-498,515-529`: CLI and direct execution/readiness/private training paths reject mismatches. Source config AND source binding require exact integer123; readiness profile and role are checked independently of rehashed JSON.
- `make_launch_readiness.py:145-193,248-368`: profile/role/seed enter canonical prepared bindings and receipt digest identity. Development receipts require matching measurement profile/seed/finetune role. Historical receipt identity fields may be absent; development receipts cannot be replayed as historical even with a newly hashed ready JSON.
- `real_probes.py:137-152,220-250`: validation precedes expensive CUDA/model access for invalid profiles, including direct model-loading API. Final measurements carry the scoped identity. Existing real-hardware checks remain, not substituted with fixture booleans.
- `mvp_finalize.py:252-265`: profile validation now applies even to test scope. Development final metadata requires finetune role and exact augmentation20261037. Publication rechecks metadata; role mutation after evaluation cannot publish. Test scope does not relax the new identity contract. The existing distributed fixture uses legacy seed1/aug20260915, so this tightening does not itself invalidate that fixture.
- `resource_probe.py:26-183`: schema2 requires explicit development-single/[123] and positive finite max_training_node_hours; no implicit cap or legacy-cap reinterpretation. Projection is one run, `(250*worst + entire_probe_elapsed)*margin + launcher_allowance`, divided by3600. Campaign adds prior smoke and measured probe time. Schema1 retains three-run arithmetic and output field. Probe ceiling, resource envelope, stage/coverage/memory guards are retained. `run_training.py` validates full schema and binds the authorization to the actual ready bindings before warmup/launch.
- Fixed250/12/four-rank/effective96 recipe and production hardware guard are unchanged. Cross-rank training identity now includes profile and seed. Finalizer metadata preflight/publication errors remain under existing collective coordination. New early profile guards run before process-group initialization; rank-asymmetric launch/file failures at that boundary still rely on the bounded launcher teardown, not a new promise of collective recovery. No reproduced coordination regression; only one excluded distributed success case was exercised, not all failure paths.

## Fresh test evidence

Working directory throughout: `/tmp/mvp-seed123-integration-20260917`.

```sh
PYTHONPATH=src:revision/scripts/mvp /tmp/mvp-final-source-env/bin/python3.12 -m pytest -q tests/test_mvp_seed123_training.py tests/test_mvp_resource_probe.py tests/test_mvp_resource_fix1.py tests/test_mvp_readiness_tools.py tests/test_mvp_readiness_review.py tests/test_mvp_real_probes.py tests/test_mvp_cli.py tests/test_mvp_finalize.py -k 'not collective and not two_rank and not huge_status and not one_rank_arbitrary' --tb=short
```

**277 passed,22 deselected in90.21s**. Log: `/tmp/train-review-focused.log`. Scoped `git diff --check` also passed.

Representative excluded test, original harness:

```sh
PYTHONPATH=src:revision/scripts/mvp /tmp/mvp-final-source-env/bin/python3.12 -m pytest -q 'tests/test_mvp_finalize.py::test_real_two_rank_gloo_padded_paired_evaluation_and_compiled_path[3]' --tb=short
```

**1 failed in11.74s**, exact dependency failure (both ranks):

```text
tests/mvp_finalize_probe.py:140 main -> make_accelerator(...)
src/transgenic/training/mvp_runtime.py:59: from accelerate import Accelerator
ModuleNotFoundError: No module named 'accelerate'
```

The unchanged harness invokes `/home/wyim/miniconda3/envs/pdb/bin/torchrun`/its Python, overwriting PYTHONPATH with external `.testdeps` paths. Failure occurs before changed finalizer validation. This is a harness dependency issue, not evidence of a new seed-profile regression. Full log: `/tmp/train-review-ddp.log`.

An isolated `/tmp` pytest plugin changed only the collected test module's TORCHRUN to a temporary shell wrapper executing `/tmp/mvp-final-source-env/bin/python3.12 -m torch.distributed.run "$@"`, and DEPS to `/tmp/mvp-final-source-env/lib/python3.12/site-packages`. No shared source/global environment edits:

```sh
PYTHONPATH=/tmp:src:revision/scripts/mvp /tmp/mvp-final-source-env/bin/python3.12 -m pytest -p train_review_plugin -q 'tests/test_mvp_finalize.py::test_real_two_rank_gloo_padded_paired_evaluation_and_compiled_path[3]' --tb=short
```

**1 passed in20.93s**. This runs the original test assertions: actual two-rank CPU/Gloo paired evaluation, compiled forwards, consistent summary, publication, clean saved keys and tied reload. Files: `/tmp/train_review_plugin.py`, `/tmp/train-review-torchrun`, `/tmp/train-review-ddp-isolated.log`. Both distributed invocations were supervised in a new process group with a145-second outer deadline and process-group cleanup; no leftover probe/wrapper process appeared in the final process check.

The original22-test exclusion is NOT covered wholesale: only the one representative success case above was recovered; the other21 excluded cases were not run here. No full suite, GPU training, model readiness probes or real receipts were produced. The implementation's reported first red invalid-probe test reached CUDA availability/native-bf16 detection only; this review does not relabel that as training or hardware readiness. Fresh development-bound real receipts and an explicit controller schema2 cap are expected later deployment gates, not missing-code defects.

## Exact reviewed SHA256 snapshots

Relative to integration worktree; HEAD remained the base revision above.

```text
77348d6ded60596023f6b29a15e94c1eda9b16b476173bf2f9908f0805685968  src/transgenic/training/mvp_profiles.py
29ec1682fdd5dbb8a02049f57c107e5e5d12d9851d9b2df38db6f5841d11a8e5  src/transgenic/training/mvp_cli.py
a34d11559ee38eee2a240d0ab720660cddec038eb46410c31d159ba96b078bf7  src/transgenic/training/mvp_cli_runtime.py
3cd6395911ae2e8618ca03f0210b74a01a225f6ee134573065a5cb5d12181239  src/transgenic/training/mvp_finalize.py
a5b2c895bd76699718655993d67430b9459c5db07d5f47636feb6acc77ce6de2  revision/scripts/mvp/make_launch_readiness.py
7f9507f4fced687e1886c659224d4b778b3486a2b390dac1e8f3c6954e46c7ae  revision/scripts/mvp/real_probes.py
81baf49407017b000966ef72be7ae9a6f35f3fc5f598517c74a470fadbe41e0e  revision/scripts/mvp/resource_probe.py
62d9499259645c29175698aab7fbb9eb56a1edd3bc6cc7ac6dc2dda1bdd48998  revision/scripts/mvp/run_training.py
33411c5eb8754b21db5f9713569aa7a914f0c71b61af98866b661fea4d9f4839  tests/test_mvp_seed123_training.py
4a0c58711a13ea65ba46711ff8e5acdefd6dc6149e7475fd2b4e7fa449c361df  tests/test_mvp_cli.py
2bcbcefd70235c267e61bddc3236728c354119dd5647d26927718754afeda454  tests/test_mvp_readiness_tools.py
11ce134b2a87c63aded62c8571ba09d795f731a674155affbf37fb8328786bd7  tests/test_mvp_finalize.py
671ef3a5b8660a8ddadd2dcdd24e431ff45f7fd80844892fb3dd20c678d35c4d  tests/mvp_finalize_probe.py
```
