# Reference-map CPU release — local packaging, 2026-09-16

**PROVISIONAL: independent deployment/repin review still required.** Source fix `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558` has scoped independent ACCEPT in `reference_publication.fix1_review.md` (review report `8508c4b`; implementation report `1de033a`). That closes the original late-WAL finding, not deployment, actual-data, decoder SIF or scientific approval. The independent review's 32 passing tests are attributed evidence, not this packaging run.

## Retained local package and exact pins

New package: `notes/wait_work_20260916/panel_inputs/reference_map_release/package/`.
Only the seven `transport.json` payloads plus transport are deployment inputs. `source/`, `SOURCE_VERIFICATION.json` and `entrypoint.diff` are local inspection evidence, not extra transport inputs. Do not copy the whole working folder remotely.

| Payload | Bytes | SHA256 |
|---|---:|---|
| `source.tar` |2754560|`d672732c756d93741d8e3edb075a0901802e29fa6082a5a128e85180c1521897`|
| `run_panel_cpu.py` |12982|`2db226c71352c55b5e5b48133a4785e304fe323a936dc5356c1c33094c9cdb18`|
| `authority.py` |3433|`0b3917b7fa33df0d1818bdbad992f664e8dc6b959639a09aa998f75a97e8b2dc`|
| `reference_authority.tar` |51200|`7002445d7007c6495d5fc00406faa2c7b373ef10b7c5ed93674bc9b04aae6ec3`|
| `cpu_site.tar` |60928000|`939cd7ab2480c03c057be6817e5e9076d0bb6bd453f7a39ba849a58e7d6d6f79`|
| `cpu_dependency.json` |6463|`cf4d509e8914d6bcf44e6a96a79fe127e4521f7592e4a768a7e2659760ec3435`|
| `panel_cpu.sbatch` |2235|`cbbf453ae97038f09a429b10dad3702f2166c9dd7d029f6e635f9cf026bc5fd9`|
| **`transport.json`** |2100|**`11d0550bd46c121c6b53e1cb51f0c528381ff41763e7e95e754d286a4d0448b5`**|

The final entrypoint hash above is NOT template hash `09281c0bd38304e2adc317f8f9216fb3686627c64b768985dd84586dd0c75a23`.

Archive is content-frozen and payload write bits removed (mount reports 0555). This is not OS-enforced immutable storage: directory semantics on this mount do not prevent replacement. Rehash before transfer and retain a canonical immutable copy; never rebuild over this package. Source tar metadata is normalized (UID/GID/time and modes); source bytes are exact Git bytes, not working-tree copies. No `.venv`, graft, cache, local site-packages or extra third-party imports are included. CPU site remains a separate unchanged payload.

## Minimal repin delta

`reference_map_release/build_release.py` retains the existing Task2 `build(repo,out)` and `--repo/--out` contract, refuses existing output, and uses unchanged authority verification. It is a local adaptation, not a new deployment framework. Original builder and releases were not overwritten.

`package/entrypoint.diff` is the complete template-to-entrypoint diff: exactly three literal substitutions, no path-name change, imports, mapping, publication, candidate, runtime or scientific logic changes:

1. `SOURCE_COMMIT`: `b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6` → `9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558`.
2. `SOURCE_PIN`: `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44` → `ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e` (86-file evaluation component).
3. `verify_source`'s `source_pins.json` digest: `7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689` → `5570f32378928ccd627494652988a0ec8cda3c657cff5d63e0c496ec8c97010b`.

Inventory contains exactly the prior 100 selected paths, freshly read using `git show 9603...:<path>`. It includes all prior src/train/MVP scripts/configs/panel and legacy 07/24; it is an evaluation inventory, not a stale copy of the GPU release ledger. Within those 100 paths, **only `revision/scripts/mvp/mvp_produce_counts.py` differs from b660**, now SHA256 `6069fb5fe0771015d044d820a4cbbff313d9f3d2982991e1c58c17bc8aa726dc`. All remaining 99 hashes match the prior reviewed inventory. Separate Git comparison proves no differences for all `src`, `train`, `configs`, `revision/protocols`, `real_probes.py`, `run_bounded.py`, `make_launch_readiness.py`, or `mvp_decode_runtime.py`. Training source/probe constants remain b660; no existing probe artifacts were rewritten.

Decoder operator closure recomputed from unchanged `DECODER_ONLY`: `b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088`. Component closure independently recomputed at build: 86 files, `ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e`. Panel remains `79367154bbf922fd0554285cb6cb47ae3d3c866b145aa1f51ead4f47b00cd243`.

The three candidate companions remain byte-identical: acceptance `faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e`, inventories `9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98`, receipts `73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac`.

Five original sparse Git authority objects, authority metadata/TSV, authority helper, cp310 DuckDB1.5.5 archive/wheel receipt and bindfix1 batch are reused unchanged. No authority reconstruction or synthetic commit. Authority remains commit `92d8020ee79eeedd7f1d50167b1c8dc94700fb16`, manifest `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`. Original runtime SIF path `/data/gpfs/assoc/pgl/singularity/micromamba.sif` and Python `/data/gpfs/home/wyim/micromamba/envs/signalp6/bin/python` are unchanged. The SIF itself was neither read nor changed locally; its runtime compatibility is not certified here.

## Local verification

Actual staged `verify_source(package/source)` passed, and unchanged `authority.verify` validated all five objects. Receipt `package/SOURCE_VERIFICATION.json` SHA256 `8d73fd2898bfd2c1569d4c7045bb2c1a03d28446ba4a689326d0d94375d514fa`.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /tmp/mvp-final-source-env/bin/python -m pytest -q -rs -p no:cacheprovider \
 notes/wait_work_20260916/panel_inputs/reference_map_release/test_release.py
```

**12 passed in 65.78s**, retained `reference_map_release/tests.final.txt`, SHA256 `a5a0478f8196caf75f16bd7bf270f4b5c8a29e837d7d5fb8778c304876c4484f`. `bash -n package/panel_cpu.sbatch` also passed. Earlier first run: 12 passed in 60.70s; final run adds stronger receipt-field assertions.

Tests execute staged source/authority verification; compare all 100 staged files to exact Git bytes; prove training/probe invariants; verify byte-identical reused payloads and literal-only wrapper diff; inspect archive bytes/member safety; run both b660 and 9acd template guards and require rejection of the new inventory; exercise the actual panel consumer's stale-producer inventory rejection; and run genuine tiny DuckDB mapping fixtures with full ordered mapping fields/source identities, rejecting NULL/empty IDs, wrong species/geometry, many-to-one mappings and reference hash mismatch. The stale-publication fixture stubs preflight only to isolate the genuine inventory-equality gate, not an actual historical job manifest. No old arrays or COMPLETE were loaded/reused.

Synthetic 50-key fixtures are **not actual 50-locus compatibility evidence**; no accepted corpus or reference was queried. Publication semantics remain byte-identical to the independently accepted source template rather than being reimplemented for packaging.

## Controller-only planned deployment — NOT executed

1. Obtain independent deployment-delta review of archive inventory, literal repins, staged entrypoint hash, ledger and preserved bind semantics. Source ACCEPT alone is insufficient.
2. Controller must verify live job status/ownership before any later attempt, preserving jobs 6165556/6165559, their arrays/manifests/COMPLETE/logs and all prior namespaces. This task did no SSH or job inspection.
3. New planned canonical absent namespace: `ROOT/evidence/mvp50_reference_map_20260916`, `ROOT=/data/gpfs/assoc/pgl/data/Transgenic`. Ledger records full planned root and expected `.../source`. Never overlay an existing namespace. Transfer only ledger payloads plus transport; verify every hash; reject symlinks, traversal, overwrites and extra executable/site files. Extract `source.tar` and separate `cpu_site.tar`; reference authority tar is retained evidence (authority already exists in source archive). Create fresh dedicated `outputs/`.
4. Batch is byte-identical to accepted bindfix1: global `/data/gpfs:/data/gpfs:ro`, default-RW output-parent bind without unsupported `:rw`. Candidate DB/reference paths are unchanged and read-only. Set `PANEL_RELEASE` to the new canonical root and `PANEL_CPU_SITE=$PANEL_RELEASE/cpu_site`; approved account/partition and fresh logs remain controller inputs.
5. One CPU-only attempt: **4 CPUs, 16GB, 1h**, timeout 3500s +20s kill grace, no GPU, no requeue, **no automatic retry**. Failure stops and preserves diagnostics; no installs, guard bypasses or budget escalation.
6. Default invocation remains **full mode** (batch contains no `--reference-only`): generate a **fresh 50-panel publication** under the new evaluation producer pin. Do not reuse job6165559 arrays or manifest even if biological inputs match. Optional `--reference-only` is explicitly partial diagnostic, never default/completion evidence. Require fresh OBSERVATIONS_COMPLETE plus validated compatibility and panel publication; a compatibility report alone is not completion.
7. Actual accepted-corpus 50-ID mapping, raw parser/geometry/duplicates, CPU imports/mounts and later decoder package/SIF compatibility remain unverified. Observational completion is not scientific/reference-policy/training approval.

No production source edits, source commit, subagents, SSH, jobs, remote writes, merge or push performed. Existing release/source/output histories preserved. Advisor consultation failed due client/model-version incompatibility and supplies no approval. Graft-reported context savings: approximately 1,014,685 tokens.
