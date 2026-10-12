# Independent bindfix1 deployment-delta review

**APPROVE one corrected bounded CPU diagnostic attempt**, subject to the fresh-namespace/controller conditions below. No delta blocker found. This is not runtime or scientific approval.

## Exact release verified

Independently hashed both releases, compared all seven payloads byte-for-byte, and checked transport metadata and the new directory's exact eight-file inventory. The old transport still matches the previously reviewed `b4bc567d6bdae382318fd0061186bf83c5077b12ac024b5cc36a6dcf9bb54eef`.

**New `final_release_bindfix1/transport.json` SHA256:**
`1883df3c32c5b560030dbcb89543b3d2730d7c08518d47e4f932b62f88b18c0a`

| Payload | Verified SHA256 |
|---|---|
| authority.py | 0b3917b7fa33df0d1818bdbad992f664e8dc6b959639a09aa998f75a97e8b2dc |
| cpu_dependency.json | cf4d509e8914d6bcf44e6a96a79fe127e4521f7592e4a768a7e2659760ec3435 |
| cpu_site.tar | 939cd7ab2480c03c057be6817e5e9076d0bb6bd453f7a39ba849a58e7d6d6f79 |
| panel_cpu.sbatch | cbbf453ae97038f09a429b10dad3702f2166c9dd7d029f6e635f9cf026bc5fd9 |
| reference_authority.tar | 7002445d7007c6495d5fc00406faa2c7b373ef10b7c5ed93674bc9b04aae6ec3 |
| run_panel_cpu.py | 3e62ae91e83b874b2c32d453d2e7001a9834930304ca57159eb4194f21acf3bd |
| source.tar | afe0bf0cbe60397167bfe64f878db4c8472053f652272308e8772e3ba562fc2f |

The sole executable difference is exactly three bytes removed at `final_release_bindfix1/panel_cpu.sbatch:36`:

```diff
- -B /data/gpfs:/data/gpfs:ro -B "$PARENT:$PARENT:rw" \
+ -B /data/gpfs:/data/gpfs:ro -B "$PARENT:$PARENT" \
```

All other six payloads are identical to the prior reviewed release. Transport metadata outside the corresponding batch hash is unchanged. No source, authority object, dependency, gate or resource-budget change; the earlier full review remains applicable to those unchanged bytes. Fresh `bash -n` passed. Controller reports three narrow tests passing in 1.372 s; the agent report records its earlier narrow and original-nine-test runs. I did not repeat the full suite or full source review.

## Failure and version-specific reasoning

`first_result_status.txt` directly records the Singularity 3.6.1 module and `rw is not a valid bind option` fatal parser error. Job6165556's FAILED/127/zero-elapsed and empty-output details are controller-reported, not independently refreshed here. This failure precedes payload execution and supplies no Torch/data compatibility result.

Independently retrieved and hashed the two version-tagged upstream files at `https://raw.githubusercontent.com/sylabs/singularity/v3.6.1/` in memory, without installing or executing them:

- `pkg/runtime/engine/singularity/config/config.go`, SHA256 `dcbbeb69963616b8aab7a6dd106b443546c4ba0e72baf512c24e71383cc38ebb`: lines 283–287 do not permit `rw`; lines 369–408 accept omitted options and line 402 produces the observed error. Lines 98–101 identify read-only only when `ro` is present.
- `internal/pkg/runtime/engine/singularity/container_linux.go`, SHA256 `001df735a7f0e477204513952118a78e157207652d6d8c76aaf2fa802d00588c`: lines 1688–1709 reset default flags separately for each bind and add MS_RDONLY only for `ro`; lines 1750–1759 bind/remount with those flags.

Thus omission requests default RW for the output bind while preserving the separate explicit global `/data/gpfs:ro` bind. This directly addresses the demonstrated parser failure. It does not prove the installed binary's identity, effective nested mount permissions, native imports or successful processing on the host.

## Conditions and limits

Follow `bindfix1_REPORT.md`'s controller-only sequence: retain job6165556 and its logs/namespace untouched; verify live status before submission; stage only these pinned bytes into a **new, absent, canonical non-symlink** `/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_input_audit_20260916_bindfix1` namespace. Safely extract reviewed members without overwrites or extra executable/site files and create a dedicated empty outputs parent. Existing full-review staging requirements still apply.

Approve only **one new separately recorded attempt: 4 CPUs, 16 GB, 1 hour**, no GPU/requeue/automatic retry; retain timeout **3500 s +20 s kill grace**. Use approved account/partition and fresh logs. Further failure means stop and preserve evidence, not bypass gates, install packages, change source or increase resources.

Full CPU runtime/Torch compatibility remains **unproven**. No actual 50-locus results, reference acceptance, decoder compatibility or GPU readiness are claimed. The unchanged Task 3 sparse Git-authority interface retains its prior review verdict; no authority re-review or Task 3 edits were performed.

No SSH, jobs, subagents, historical-artifact edits or deployment changes were made by this review; only this separate review file was written. Graft automatically refreshed its context index during lookup. Graft-reported savings: approximately 10,622 tokens.
