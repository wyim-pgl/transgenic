# Task 2 bindfix1 — local-only release correction

**Ready for controller review; not submitted or runtime-certified.** New retained package: `final_release_bindfix1/` in this folder. Old `final_release/`, `REPORT.md`, independent review, submission receipt and failure evidence were not edited. No SSH, remote container, login-node work, job submission, subagents, source/environment installation, companion changes, merge or push occurred.

## Failure and exact correction

Controller/user reports job **6165556 FAILED, exit 127, elapsed 0**, before data processing, with no files found in its remote outputs directory. Those accounting/empty-directory observations were not independently refreshed here. Retained `first_result_status.txt` directly records Singularity 3.6.1 and:

```
FATAL:   while parsing bind path: while getting bind path: rw is not a valid bind option
```

Only `final_release_bindfix1/panel_cpu.sbatch:36` changes:

```diff
- -B /data/gpfs:/data/gpfs:ro -B "$PARENT:$PARENT:rw" \
+ -B /data/gpfs:/data/gpfs:ro -B "$PARENT:$PARENT" \
```

The only other changed payload is `transport.json`, replacing the batch SHA256. No rebuilding/repacking occurred. Only seven manifest-listed payload files plus transport were copied; no extracted source/site directories were copied. Six payloads remain byte-identical by SHA256, including source, reference authority, CPU site, dependency receipt and both helpers. The complete batch byte comparison proves all other runtime, path, output-exclusivity, gate, CPU/GPU and budget settings unchanged.

## Singularity 3.6.1 semantics verified from upstream source

Read-only HTTPS retrieval of existing version-tagged upstream source; no source was installed or executed:

- Base URL: `https://raw.githubusercontent.com/sylabs/singularity/v3.6.1/`
- `pkg/runtime/engine/singularity/config/config.go`, SHA256 `dcbbeb69963616b8aab7a6dd106b443546c4ba0e72baf512c24e71383cc38ebb`: lines 283–287 permit `ro`, `image-src`, `id`, **not `rw`**. Lines 369–408 split source/destination/options; omitted options bypass option validation; line 402 emits the exact observed error. Lines 98–101 implement `Readonly()` as `b.Options != nil && b.Options["ro"] != nil`.
- `internal/pkg/runtime/engine/singularity/container_linux.go`, SHA256 `001df735a7f0e477204513952118a78e157207652d6d8c76aaf2fa802d00588c`: lines 1688–1709 reset flags for each bind, defaulting to `MS_BIND | c.suidFlag | MS_NODEV | MS_REC`, adding `MS_RDONLY` only when `b.Readonly()`. Lines 1750–1759 add the bind/remount. Thus the output bind omitting options requests default RW while the separate global bind remains explicitly RO.

This is source-level verification, not certification of the installed binary, mount permissions, nested-bind behavior on the actual host, native CPU imports, or actual data results. Host permissions/read-only source mounts still constrain writes; lines 2498–2530 explicitly account for source mount flags. Initial guessed `config/bind.go` and `help-exec.txt` URLs returned HTTP 404; the actual source paths above succeeded. No help or runtime success is inferred from those failed lookups.

## New payload inventory

All paths below are under `final_release_bindfix1/`; SHA256 values were measured locally.

| File | Bytes | SHA256 |
|---|---:|---|
| authority.py |3433|`0b3917b7fa33df0d1818bdbad992f664e8dc6b959639a09aa998f75a97e8b2dc`|
| cpu_dependency.json |6463|`cf4d509e8914d6bcf44e6a96a79fe127e4521f7592e4a768a7e2659760ec3435`|
| cpu_site.tar |60928000|`939cd7ab2480c03c057be6817e5e9076d0bb6bd453f7a39ba849a58e7d6d6f79`|
| panel_cpu.sbatch |2235|`cbbf453ae97038f09a429b10dad3702f2166c9dd7d029f6e635f9cf026bc5fd9`|
| reference_authority.tar |51200|`7002445d7007c6495d5fc00406faa2c7b373ef10b7c5ed93674bc9b04aae6ec3`|
| run_panel_cpu.py |11627|`3e62ae91e83b874b2c32d453d2e7001a9834930304ca57159eb4194f21acf3bd`|
| source.tar |2897920|`afe0bf0cbe60397167bfe64f878db4c8472053f652272308e8772e3ba562fc2f`|
| transport.json |1653|`1883df3c32c5b560030dbcb89543b3d2730d7c08518d47e4f932b62f88b18c0a`|

Old transport remains `b4bc567d6bdae382318fd0061186bf83c5077b12ac024b5cc36a6dcf9bb54eef`; old batch remains `9f236df0ff39ce3b6282e729d577f932b89cd0fac301cb4bff406616ea1abb26`. The new release is content-frozen, separately named and hash-pinned. File write bits were removed (this mounted filesystem reports files as 0555 after chmod 0444). Directory chmod 0555 still reports 0777 on this mount: this is **not OS-enforced immutable storage**. Do not modify/reuse it; controller must independently rehash it and retain a frozen canonical remote copy.

## Tests and retained local evidence

From this folder, with `PYTHONDONTWRITEBYTECODE=1`:

- `python3 test_tools.py`: **9 original tests PASS, 19.894s**, still targeting the untouched old `final_release`. Log `tests.bindfix1.original.txt`.
- `python3 test_bindfix1.py`: **3 narrow tests PASS, 1.125s**. Log `tests.bindfix1.txt`. Verifies old reviewed transport/payload hashes, only the batch digest differs, transport metadata remains equal, only eight files exist in the new package, exact three-byte removal, `bash -n`, and actual shell-expanded argv. A local `timeout` stub captures argv and never invokes Singularity or Python payload. Old/new argv are identical except the output bind option; quoted paths with spaces are exercised; global RO and timeout limits retained.
- Red control before correction: `BIND_FIX_RELEASE="$PWD/final_release" python3 test_bindfix1.py`: expected **3 failures**, retained in `tests.bindfix1.red.txt` (old RW argv, unchanged batch, old extracted source directory). This is a test sensitivity check, not another remote attempt.

New verification files and SHA256:

| File | SHA256 |
|---|---|
| test_bindfix1.py |`cfc358709f4892d26013ec5ae9f738125dd00f6692e009f53d5681e59622ead4`|
| tests.bindfix1.original.txt |`2b097df6484eaad542f69c474964bc2abdf683b94fbe7cf14b4530ef635ffb59`|
| tests.bindfix1.txt |`09580ce1ac44dbd3fef242ad9f6fa3e34a981e2f70961506caaf8a8b8747b228`|
| tests.bindfix1.red.txt |`4f01d782f182df0c91526fe93a78b6c835521cee935d0438ce1f28b0053ea203`|

Unedited retained evidence measured after verification: `REPORT.md` SHA256 `133b2ebcf38d40e21d2c251e65d7f815bd3a5f93bd6f8538c63f1b2c4a135b67`; `independent_review.md` `05438d9c5360e5b299eefe915fe7b32e3ffc641bd24a8414cb27eebcfa577632`; `first_result_status.txt` `b94d0c0d651758e1006f3270a37ee65e6d808d4c2be04effcc1ce982a8f6e9a1`; `submission_receipt.txt` `b1ca89452f4239b1d1bf868d8703edda5007fe9974743a44b5c57acf9ea0b59e`; original `test_tools.py` `369348b94fd0e5c0c87b384bb49de15b9c58cd9c539e5cfff08773a07d542330`.

## Controller-only next attempt — not performed

1. Review the one-option diff, upstream evidence and new transport pin independently. The old independent review covers the old release; it is not automatically approval of this new package.
2. Recheck live job/accounting state and ownership to prevent duplicate work. Preserve job6165556, its logs and old namespace; never overlay, clean or retry there.
3. Use a **new canonical non-symlink** namespace: `ROOT/evidence/mvp50_input_audit_20260916_bindfix1`, where `ROOT=/data/gpfs/assoc/pgl/data/Transgenic`. Require it absent before staging. Transfer only the eight listed files after hash review; verify hashes and retain exact payloads. Safely extract only reviewed source/site members without overwrites; reject extra executable/site files. Create a new empty dedicated `outputs/`.
4. Set `PANEL_RELEASE` to that absolute namespace and `PANEL_CPU_SITE=$PANEL_RELEASE/cpu_site`; choose approved CPU account/partition and fresh absolute Slurm log paths. Only after controller review, submit **one separately recorded new attempt**: one node/task, **4 CPUs,16GB,1h**, no GPU, no requeue, no retry. Existing timeout remains **3500s +20s kill grace**. Do not automatically reuse any old attempt receipt or output path.
5. Record new job ID, transport pin, commands and actual allocation/batch/extern results separately. A remaining mount/runtime/import/gate failure stops the attempt and preserves all diagnostics; no bypass, install, source change, budget escalation or automatic resubmission. Missing observation-completion marker is not success.

No actual 50-locus processing, CPU runtime certification, reference acceptance, decoder compatibility or GPU readiness is claimed. All original report/review caveats remain. Graft context savings this task: approximately 959,816 tokens (tool estimates).
