# Track A: fail-closed launch tooling

CPU implementation only. No GPU/SIF/Hyena readiness is certified here. The
existing trainer and `mvp_cli_runtime.py:175-236` contract are unchanged.
`--validate-only` bypasses readiness; use this tool's **check** command instead.
The base consumer trusts operator receipt digests; this tooling additionally
checks the receipt files, DONE digests, measurements, sidecars and tool identity.
Do not bypass it with a handcrafted readiness record.

## Required decisions before deployment

* Author-selected **concrete checkpoint**, its weights/run_config/runtime_identity
  SHA256 pins. No implicit epoch05 choice and no `best`, `latest`, or symlink.
* Staging route, quota headroom, copy byte ceiling and reserve.
* Reviewed real-runtime P1–P4 probe programs and their executable/import/source
  identities. The bounded runner is infrastructure, **not a real Hyena probe**.
* Per-seed walltime authorization after measured P4 timings; submission approval.
* An immutable snapshot including these scripts plus the f847cb8 training code;
  an f847cb8-only archive does not include these tools.

No script submits jobs or transfers between hosts. Run inside the intended,
independently pinned SIF and compute allocation. `SLURM_JOB_ID` is an operational
safeguard, not authentication. Approved data movers can set
`MVP_DATA_MOVER_ALLOCATION_ID` to their real allocation identifier. Never set a
fictional identifier to run large hashing on a login node. Unit tests use tiny
synthetic files and explicitly labelled CPU fixture identifiers.

## Full staging inventory

Use a disjoint root `/work/nvme/bilv/wcyim/transgenic/mvp/` with
`inputs/`, `checkpoints/`, `preflight/`, `readiness/`, and `runs/`. Keep seed123
source runs and existing B5 jobs read-only.

| Input | Route requiring operator confirmation | Post-copy check |
|---|---|---|
| `--db`: B5 corpus (~23.5 GB) | pronghorn approved data mover to `inputs/` | acceptance `inputs.corpus.sha256` |
| `--evidence-db`: fullc0lite.db (~1.07 GB) | pronghorn approved data mover to `inputs/` | `ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d` |
| panel TSV | immutable snapshot to `inputs/` | `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243` |
| config JSON | selected immutable snapshot to `inputs/` | operator-approved config SHA256 |
| `--init-from` weights (~2.09 GB), run_config, runtime_identity | compute-local copy with `stage_checkpoint.py` | all three author-selected pins, destination rehash |
| SIF (~11.12 GB) | existing DeltaAI container read-only; no redundant copy | `1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6` |
| technical acceptance JSON | controller-approved bytes to `inputs/` | `faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e` |
| inventory JSON | controller-approved bytes to `inputs/` | `9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98` |
| receipts JSON | controller-approved bytes to `inputs/` | `73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac` |
| Hyena/tokenizer offline cache, probe source/imports | confirm existing cache or approved transfer | operator-reviewed cache/source manifest; no network fallback |
| code, probe argv, budget authorization, quota record | reviewed snapshot/operator records | fresh SHA256 on compute; retain provenance |

Before any copy, measure **user/project quota**, not merely `df` capacity.
Account for all missing inputs, checkpoint temporary copy, caches, compile
artifacts, preflights, three output checkpoints, logs and reserve. SIF already
present should not be double-counted. Do not invent a fixed headroom value.
`stage_checkpoint.py` checks only its own copy budget, not the whole inventory.

## Checkpoint staging

Required flags (all values supplied by operator):

```
python revision/scripts/mvp/stage_checkpoint.py \
  --source-epoch "$CONCRETE_EPOCH" --dest "$FRESH_STAGED_ROOT" \
  --expected-weights-sha256 "$WEIGHTS_SHA" \
  --expected-run-config-sha256 "$RUN_CONFIG_SHA" \
  --expected-runtime-identity-sha256 "$RUNTIME_SHA" \
  --quota-record "$QUOTA_JSON" --quota-sha256 "$QUOTA_SHA" \
  --max-copy-bytes "$COPY_CEILING" --reserve-bytes "$RESERVE"
```

Quota JSON: `path` is the resolved destination parent, `available_bytes` is the
measured quota headroom, `measured_at_unix` is its actual timestamp. It must be
at most one hour old. Keep the original quota command/output alongside it.
Filesystem free space is checked separately. Failures leave a `.partial.PID`
directory for inspection, never a published destination. Do not auto-delete it.
Success publishes parent sidecars plus `checkpoint/model.safetensors`, the exact
chosen loader inventory (`config.json`, `meta.json`, `runtime.json`, each present
with its digest/size or explicitly absent), optional `eval.json`, a staging
receipt and digest-bound `STAGING_DONE`. Sidecars must be canonical nonsymlink
regular files. Staging checks the chosen source and copied inventories before
publication; readiness prepare/generate/check bind presence as well as bytes.
Use `--init-from "$FRESH_STAGED_ROOT/checkpoint"`.

## Prepare → probe → generate → direct check

`make_launch_readiness.py prepare` and `generate` take the full input flags:
`--db --evidence-db --panel --sif --config --init-from --acceptance-record
--acceptance-sha256 --inventory-record --receipts-record --seed`, plus the three
`--expected-*-sha256` checkpoint pins, `--manifest-output`, and `--output`.
Seeds are exactly 1/2/3; prepare separate bindings/receipts per seed.

1. **prepare** writes a non-authorizing bindings JSON and exact consumed-code
   manifest, including the shipped `accelerate_mvp.json` config path and bytes.
   Config absence fails prepare; later addition/removal/modification cannot reuse
   prepared evidence. Save the bindings SHA256. It hashes all inputs on compute and compares the
   corpus/evidence/panel identities with the accepted candidate.
2. Run each real, reviewed probe with `run_bounded.py`:

   ```
   python revision/scripts/mvp/run_bounded.py \
     --name "$EVIDENCE_NAME" --bindings "$BINDINGS_JSON" \
     --bindings-file-sha256 "$BINDINGS_FILE_SHA" \
     --command-file "$ARGV_JSON" --command-sha256 "$ARGV_SHA" \
     --output-dir "$PREFLIGHT_ROOT/$EVIDENCE_NAME" --timeout-seconds "$BOUND"
   ```

   `ARGV_JSON` is an argv array, executable path absolute, never a shell string.
   Run only source-reviewed programs in the pinned SIF/cache. The argv hash
   does **not** certify mutable imported code or the executing SIF; those remain
   operator-enforced deployment gates. Six evidence names/stage mapping are in
   `EVIDENCE_RECEIPTS`. P4 can produce separate measured outputs for its three
   evidence names. The probe writes JSON to `MVP_MEASUREMENT_OUTPUT`, echoes
   `MVP_BINDINGS_SHA256` into `bindings_sha256`, and measures `device="cuda"`,
   `real_runtime=true` plus the fields checked by `validate_measurement`.
   Do not copy these flags from the CPU tests into a purported GPU receipt.

   Optional warmup: `--warmup-command-file --warmup-sha256 --warmup-attempts`
   (1–3), `--warmup-timeout-seconds`. Only exit 75 retries. The real probe is
   never retried. Logs cap at 1 MiB per attempt; timeouts terminate then kill
   the process group. All argv bounds must fit the authorized allocation.
3. **generate** uses the same input flags/manifest and adds `--preflight-dir`.
   It rehashes inputs and requires all six measured, bound receipts and
   hash-bearing DONE markers. It calls the existing consumer validator before
   atomically publishing the ready JSON. Outputs must be fresh; no overwrite.
4. **check** uses the full input flags (no checkpoint-pin flags), plus
   `--readiness` and `--readiness-sha256`. It directly exercises
   `validate_launch_readiness` and revalidates receipt/sidecar identities.

### Deterministic P2 policy

Predeclare **bitwise equality**, no post-hoc tolerance. Probe must call
`torch.use_deterministic_algorithms(True)`, set models to eval, fix Python,
NumPy and torch/CUDA RNG seeds to the bound seed, disable stochastic transforms,
and use the same measured batch. Runner exports
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python/CUDA startup. No training
semantics are changed by this preflight-only policy.

Measurement requires all tiers `30720`, `61440`, `129024`, each with
`input_sha256`, `finite=true`, and identical canonical tensor-byte digests for
`b5_logits_sha256`, `mvp_off_logits_sha256`,
`mvp_zero_projection_logits_sha256`, `repeat_logits_sha256`. Report
`deterministic_algorithms=true`, `eval_mode=true`, `comparison="bitwise"`, and
`cublas_workspace_config=":4096:8"` only after observing those settings.
Unsupported deterministic operations or any mismatch fail; changing to a
numerical tolerance requires a separately reviewed prospective policy.

## Bounded training wrapper

Run `run_training.py` inside the pinned SIF on the authorized single-node,
four-GPU allocation. Required: `--arguments` (JSON array of exact trainer CLI
flags), `--arguments-sha256`, `--budget-authorization`, `--budget-sha256`,
`--timeout-seconds`, `--status-dir` (fresh).

Budget JSON requires `authorized_scope="exploratory_mvp"`, explicitly approved
`seeds`, integer `max_wall_seconds`; retain the author's actual authorization.
Its SHA256 must match P4 resource measurement `authorization_sha256`; measured
projected runtime must fit the requested bound. Allow additional allocation
time for input rehashing and up to three 60-second CUDA-startup attempts.
The wrapper rejects `--validate-only`, missing readiness, existing output or
unauthorized seed/budget. It directly checks readiness, warms CUDA, then launches
exactly one fixed four-process bf16 Accelerate training attempt. It passes the exact
bound `accelerate_mvp.json` path through `--config_file` and rechecks its bytes
after warmup; home/default Accelerate config is not consulted. The config pins
single-machine rank 0, GPU IDs `0,1,2,3`, bf16, static loopback rendezvous
`127.0.0.1:29500`, and Accelerate Dynamo `NO`. `--max_restarts 0` is explicit on
the command line (it is not a supported ClusterConfig key in Accelerate 0.34.2).
Dynamo `NO` prevents a second Accelerate compiler; the trainer still compiles its
bound forward, with no eager fallback. Its four-distinct-non-MIG-GPU gate stays
unchanged. Port 29500 must be free in the isolated single-node allocation.
The schema/parser was CPU-tested with installed Accelerate 0.34.2 against a
conflicting default config. Verify the actual SIF dependency separately; this
is not production SIF certification. All original trainer argv are preserved.

SIGTERM/timeout writes atomic FAILED, never completion or resumable state.
The trainer has no public resume or graceful-save handler. Zero child exit is
recorded as `PROCESS_EXIT.json`, **not** a training completion certificate;
inspect the trainer's own finalized artifacts. SIGKILL/node loss may leave no
failure record: absence of a success artifact is not success. Never auto-resume,
resubmit, or fabricate a checkpoint after interruption.
