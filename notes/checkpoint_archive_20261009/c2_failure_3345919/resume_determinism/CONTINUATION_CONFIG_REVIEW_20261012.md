# C2 version2 final deployment configuration review — 2026-10-12

**Disposition: approved for initialization and controller-gated validation400 submission using the exact final configuration below. No remaining material findings.** The concrete host preflight has passed. Full initial proof3369694 has been adjudicated and its accepted runtime fingerprint matches the configuration. Validation400, activation and ordinary production retain every existing scientific, scheduler, storage and identity gate; this approval does not bypass them.

## Exact reviewed configuration and runtime

Final file: `/tmp/c2-controller-config-v2.final.json`.

SHA256: `aa0b423844a59be4a690bf5218ad6598bbe4a0411a24fde7cd9a824ec3ee9951`.

Intended new operation: `/work/nvme/bilv/wcyim/transgenic/ops/c2_in_place_198ce14_20261012`, operation ID `history198ce14-20261012`, owner `wcyim`, UID97434. Initialization must use this final file, not the earlier draft builder, which writes a null fingerprint. No operation was initialized by this reviewer.

| Pin | Approved value |
|---|---|
| Runtime helper | `ops/c2_compile_history_20261012/runtime_v2_506bb11` |
| Runtime manifest | `551f45007645cee43cab3774feda174493df4c0682bfb1079030d2cb56d07407` |
| Runtime staging receipt | `ops/c2_compile_history_20261012/runtime_v2_506bb11_receipt.json`, SHA `ef1cdc867551221791fde6912c566848380a1526cf7dbf6380417b7f6473b254` |
| Controller bytes | `d4e59576141d7f08d08832383cf1f246828ad3c4df0f0c7e8517426fc9d1b96b`, including the independently approved WorkDir correction |
| Producer bytes | `5da60c8f994a7814ba2d0ac337f7f8320e59b887612bc6ec0622059e427d88c9` |
| Original checkpoint receipt | New runtime helper's `original_checkpoint_receipt.json`, unchanged SHA `c7ba61600300c8467f453c5476f42af80c795a217a6b9dab803966f5f37d44da` |
| Source | Commit `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4`, snapshot Git `4f12c97a3798e09200de1f012e963c193e68964f`, directory `repo_b6_198ce14` |
| Source manifest | `29206d99c14ce4260d4d6505bf1321ff494315a5ba97fe3220a013f0fd5969dc` |
| Validation helper | `ops/c2_compile_history_20261012/validation_v2_61d79f0`, manifest `a751eb193b607574df6ac6e44c224c59e57b75b1fd069a23bc6f816ffc1bbb91` |
| Adopted proof |3369694; immutable helper `ops/c2_compile_history_20261012/v1_188bf3e`, manifest `5d10eaad8598b12b964ed0f576fb345f0486f3bdd278aac63331688dc9e57012` |
| Runtime fingerprint canonical SHA | `5f7e184ca90fc18889cd9194952911808b9d570bc0c5272e7e5449bc87e6df6c` |

The final diff from the reviewed populated draft changes only the production runtime directory/script/manifest, its staging receipt pin, and the original-receipt reference into that successor directory. Source, validation, proof, resources, original/diagnostic/protected-job pins and fingerprint remain unchanged. All18 local successor manifest members were independently hashed and matched both the manifest and remote staging receipt. Relative to the prior runtime, only the three independently reviewed WorkDir-fix files changed; its review note was added. The exact configured production script is manifest-covered. The source/validation/proof receipt pins remain those already reviewed.

## Closed deployment blocker

The draft review found that old3345920's live absolute Command and terminal relative SubmitLine script were inconsistent in the original parser. The new controller normalizes the relative script against recorded absolute accounting WorkDir and checks equality with live scontrol Command, while retaining exact argv/SubmitLine, token, owner/UID and submission time. `CONTINUATION_WORKDIR_FIX_REVIEW_20261012.md` independently approved this narrow correction.

Actual host evidence `/tmp/c2-workdir-host-smoke.json` uses Python3.6.15 and the exact controller SHA above. It records PASS for historical active/terminal command agreement and completed-proof accounting. Actual historical terminal3345919 and held3345920 both resolve to `/work/nvme/bilv/wcyim/transgenic/repo_b6_233574a/revision/scripts/b6/wp5_pilot.sbatch`;3345920 remains PENDING/JobHeldUser. The obsolete job's correct absolute pin is preserved. The old runtime bundle is superseded, not edited in place. This closes the draft's P2 blocker.

## Accepted proof and effective environment

Parent acceptance is retained in `3369694/PARENT_ACCEPTANCE.json`. The reviewer independently ran the pinned pure numerical validator on the collected actual `comparison.json` and checked its SHA `33dfc993831b0325214693d3d6422e2231c5eaf9868cf4462c4ecea7b912fe89` against that acceptance record. The report has zero different model keys, all six optimizer/scheduler/rank-RNG equality fields true, all four ranks' eight mismatch lists empty, equal histories, and the exact historical210 loss `3.2567708492279053`.

Collected checkpoint metadata, identity and history bindings were independently checked:

- at210: epoch0/step5040/global210/pending0, history SHA `49028ddf20a382e4778d533680540cf900760a1371654fc44260c2fcb9135b17`.
- Continuous220 and resumed220: epoch0/step5280/global220/pending0, identical history SHA `db3b69f91efc5c8ca831a4453a32160036c684920538b3cd152c12155cac15a9`.
- All three fingerprints exactly equal the final configuration fingerprint; all meta/history/identity/bootstrap relationships checked match. Verified allocation record names3369694 and the collected batch log contains the terminal proof marker.
- Actual host parser evidence shows parent, batch and extern COMPLETED0:0 and no active execution. The exact configured SubmitLine/comment/command agree. Failed3369338 remains historical, not adopted.

Bootstrap SHA `894b62c2fa793b66ac23710ee32aecfcbab738cbf54b9460cbd0756ddf7c59a2` also matches independent reconstruction from the original checkpoint receipt, four prefix hashes, required slots and the helper's exact JSON serialization.

`/tmp/c2-production-env-match.json` contains exact equality for all **14** fingerprint cache/environment fields between the actual image clean environment plus production overrides and accepted proof context. Its actual and expected dictionaries independently equal `config.expected_runtime_fingerprint.cache_env`. This preserves observed TF32, loop-ordering, CUTLASS/Triton tool paths and enabled local FX/AOT caches; no runtime value was adjusted to an assumed preferred default. The environment smoke is not GPU training and does not replace the later allocation's full fingerprint check.

## Concrete host preflight

`/tmp/c2-config-host-preflight.json` is present and pins the same final config SHA `aa0b4238…ee9951`. It records:

- Python3.6.15 and PASS for source/helper/input/original pins.
- PASS for no competing writers, completed proof accounting and exact old-held-job identity.
- Project22169, pool `ddn_ssd`; quota remaining137,088,151,552 bytes and filesystem available320,458,334,208 bytes, both above107,374,182,400 bytes (100GiB).
- Read-only scope: no initialization, ownership claim, submission, cancellation or activation.

The first parent dry runner omitted its synthetic journal receipt field and stopped before storage; the corrected runner alone was rerun. No implementation or source bytes changed for that correction. The final concrete PASS record, not the initial dry-run attempt, supports this approval. Storage is still rechecked by the controller at every required transition.

## Unchanged reviewed routing, resources and protections

Proof root is `runs/c2_compile_history_proof`; validation root is `runs/c2_compile_history_validation`, matching the launcher's fixed ROOT/PRIOR values. Proof marker logs use `c2-history-proof-{job_id}.out`; explicit validation/production output patterns match their configured marker-log templates exactly. Production phase.root does not choose the training destination: activation derives the accepted validation job's `/resumed` path, and execution receipts belong under operation links.

Proof test counts remain NGC58/skip1 and harness7/skip0; validation counts remain NGC58/skip1 and harness18/skip0. Production `tests={}` is intentional because its allocated receipt is the acceptance boundary.

Validation uses account `bilv-dtai-gh`, partition `ghx4`, time04:00:00, one node, four GPUs,64 CPUs/task, exclusions `gh062,gh093` and no-requeue. Explicit argv overrides the two-hour interactive script header. This parent-authorized choice accommodates measured setup plus both training/replay/save phases without a fixed user two-hour cap. Production remains24:00:00 ghx4/four-GPU/64-CPU with its reviewed900-second graceful signal. No recipe or numerical setting changes accompany these resources.

Clone peak22,910,679,660 bytes is exactly four times5,727,669,915 bytes, covering three clones plus a publication temporary (about21.337GiB). It must fit independently of the100GiB minimum; the two requirements are not added. `/usr/bin/python3` and `repo_b6_198ce14/revision/scripts/b6/storage_preflight.py` remain the approved storage command.

Original200 remains `runs/b6_wp5_pilot_C2`, including its empty original failure marker; diagnostic256 remains `runs/b6_wp5_pilot_C2_recovery_20261011`. Their metadata/identity/provenance pins were checked against supplied remote evidence, and the available retained JSON bytes independently hash to those pins. Original position is epoch0/step4800/global200/pending0; diagnostic is epoch0/step6144/global256/pending0. Bindings SHA `64e7fa926ecbcc74062f9cff9c3978f3bdd9de461029fd4b32dd78ee9e08eab3` and training config SHA `e4e7d123fbda29a36a39b3ad4ed7fe4c505767aaf7e1085fcf6c63f9a8800d41` are unchanged and checked.

Protected B5 IDs3309786–3309790 retain their recorded absolute `repo_seeds/deploy/deltaai/train_b5.slurm` command with `held=false`. B6 IDs3345917/3345918 retain their recorded old-source launcher with `held=true`; no release is authorized by this configuration. Old C2 ID3345920 is the separate conditional retirement target. Every action still requires fresh scheduler owner/command/state evidence. Nice0 restoration remains gated on fully verified final4099, with B6 held.

No new review finding remains. This review performs no remote actions, source edits, operation initialization or delegation. It updates only this review note. Previously approved controller/producer suites were not repeated; the final work was confined to changed bundle pins, actual proof/control artifacts, effective environment equality and concrete host preflight evidence.
