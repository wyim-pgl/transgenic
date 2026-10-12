# Legacy checkpoint200 bootstrap review — 2026-10-12

Independent read-only review of implementation base `/tmp/b6-c2-recompile` at `d7b1a37`. No GPU jobs submitted, no remote files changed, no implementation edits. Reviewed current resume/TODO/handoff/quarantine, original draw logs and checkpoint metadata, 3360978 replay transitions, 3361666 traces/source receipts/comparisons, and 3364480 reports and diagnostic source. This is a proposed bootstrap with unresolved proof requirements, not an observed legacy compile ledger or production approval.

## Concrete descriptors recovered

Slots below are **1-based per-rank training forward positions**, epoch0, C2 evidence view, batch1, train mode, grad enabled, bf16, nonreentrant checkpointing, four-rank DDP, backward loss divided by24. All ranks advance together at a slot. This is not global flattened row number; its canonical flattened offset is `(slot-1)*4+rank`.

First actual input for each DNA tier:

| Rank | Slot | Corpus rn | DNA L | Labels |
|---|---:|---:|---:|---:|
|0|1|207602|61440|58|
|0|3|113494|129024|651|
|0|4|95461|30720|269|
|1|1|185278|61440|534|
|1|2|70488|30720|48|
|1|3|26212|129024|1106|
|2|1|162083|30720|245|
|2|4|210585|61440|708|
|2|15|85437|129024|202|
|3|1|91492|30720|33|
|3|3|146698|129024|80|
|3|7|117991|61440|1011|

Recovery basis: the local `rank00.tsv`–`rank03.tsv` in this directory supply rn/epoch/draw hash. Read-only SSH pronghorn→dtai ran NGC DuckDB SELECT of `rn, fin-start, gff` for the first20 rows per rank plus four slot3915 companions, opening the sealed corpus read-only, with its directory bound `:ro`. Labels were derived locally with implementation `gsf_contract.count_tokens_v3`, capped8192 as the production tokenizer does; these are metadata-derived lengths, not newly observed GPU tensor shapes/hashes. Rank3 values agree with actual `../3360978/replay.json` observations. The original corpus identity is SHA256 `6583b1b202c4c5b93b88f893e6c15f267a5ec54f0c2db675b29466071f15111c`; this review did not rehash the 23.8GB corpus. Exact collated input hashes remain to be produced in the validation allocation.

First-tier event-union candidate and actual companions, notation `rn / DNA L / labels`:

| Slot | Rank0 | Rank1 | Rank2 | Rank3 |
|---:|---|---|---|---|
|1|207602 / 61440 / 58|185278 / 61440 / 534|162083 / 30720 / 245|91492 / 30720 / 33|
|2|152317 / 61440 / 1539|70488 / 30720 / 48|234405 / 30720 / 222|104126 / 30720 / 560|
|3|113494 / 129024 / 651|26212 / 129024 / 1106|154375 / 30720 / 239|146698 / 129024 / 80|
|4|95461 / 30720 / 269|215823 / 61440 / 33|210585 / 61440 / 708|243815 / 129024 / 267|
|7|228306 / 61440 / 93|4653 / 30720 / 731|204661 / 30720 / 165|117991 / 61440 / 1011|
|15|187061 / 129024 / 82|34427 / 61440 / 194|85437 / 129024 / 202|91222 / 30720 / 200|
|3915|105675 / 30720 / 121|68854 / 30720 / 42|66958 / 61440 / 440|10431 / 129024 / 4501|

Rank3 slot3915 must be included: in 3360978 it is replay micro75, graph3→4. Its draw hash is `0ffabc7a57792cb6e4eb4c53ac0149709060383a6f2f144736c3ec85097a75c5`. Other rank companions at3915 have draw hashes:

- Rank0: `1b48f1d26f82fb3505fbef88cc41addecac69a49b7d5be12f3af471be2c5cb52`.
- Rank1: `ce701b2a0596a5f234525f8605427bed51ca2587ea19859acdd2615c17815b80`.
- Rank2: `1a486261d5c2fa6a3cac37f116759272bce3a51dbf70ce1127e854d2101f6164`.

Rank3 slot6718 rn6195 labels3924, graph4→5 in that replay, is **after checkpoint200/slot4800**. Do not preload it into a checkpoint200 bootstrap: it would add future history. Likewise, 3364480 primers4801–4805 establish diagnostic B history after200, not original pre200 history.

## Exhaustive metadata audit of the committed prefix

A second read-only NGC DuckDB SELECT restricted `geneList` to the **19200 distinct rn entries actually present in the four committed4800 draw prefixes**, selecting only `rn, fin-start, gff`; no DNA/evidence tensor rasterization, input SHA sweep or GPU execution. All19200 requested records were returned. Transferred GSF JSON was18,322,905bytes; SSH/container/query/count processing took5.77seconds. `SET threads=1` affected this read-only connection, not the database. The source corpus was bound read-only.

Checked per rank/tier chronological first occurrence of labels1, labels2, labels3..3413, labels3414..4095, labels>=4096:

- No labels1 or2 in any of the19200 actual inputs.
- Ranks0,1,2: every committed input stays in labels3..3413, for all three DNA tiers.
- Rank3: its initial three tier descriptors are in labels3..3413; only additional checked regime is129024/labels>=4096, first at3915 rn10431 labels4501. No labels3414..4095 case precedes checkpoint200 on any rank.
- Hence the additional chronological regime union beyond fullprefix1..15 is exactly **[3915]**. The first-tier union itself is `[1,2,3,4,7,15]`. This establishes the proposed16-slot sequence's coverage of the checked metadata regimes, not complete observed compiler events.

Boundary correction: actual `../3360978/replay.log:99` records `1536*labels.size()[1] < 5242880`, whose maximum integer label length is3413. `../3361110/replay.log:700` explicitly observes new graph at labels3414; `:712` observes another at4096, and `:706` has a guard ending4095. Thus use **<=3413 /3414..4095 />=4096**, not thresholds shifted by one. Guard interval overlaps (e.g. another branch has lower3072) show why these three metadata buckets are candidates, not a universal mathematical partition of all compiler specializations.

## Candidate sequence and its exact limits

1. Validate the original200 checkpoint payload against `3361666/B/DIAGNOSTIC_SOURCE.json` (all model/optimizer/scheduler/four RNG/meta/identity hashes), frozen corpus/binding/source identity and draw-prefix hashes. Model SHA256 is `722ee4e55b09cda023584c1bf4ba927624ce338c94184e5b803b04c45c7760b5`; meta says epoch0/slot4800/update200/pending0. Mark origin explicitly `legacy reconstructed bootstrap`, never `observed ledger`.
2. Reconstruct each rank's exact collated C2 inputs from the sealed corpus/evidence/cache at epoch0, checking recorded draw hashes and tensor bytes/shape/stride/dtype hashes. Preserve parameter objects, complete buffer registries including None/new/nonpersistent entries and alias identity, per-submodule modes, all RNG domains and every pending gradient. No optimizer/scheduler/global-step advance.
3. Start with **all actual synchronized slots1..15**, forward and backward on every rank. This preserves the first hints for every tier, including rank2's late first129024. The smaller union `[1,2,3,4,7,15]` is a separate compression candidate; no existing experiment validates that compression. Replay synchronized slot3915 with the four actual companions above. However, the jump15→3915 is also compression, so this is a minimum-known-history candidate, not established complete history.
4. Restore the protected numerical state, modes and RNG exactly while retaining the generated compiler dispatch history. Check complete digests and zero progress; do not call Dynamo reset afterward. Resume at4801. Record the bootstrap digest and validation receipt alongside future observed events in every checkpoint path.
5. If historical loss/checkpoint proof fails, inspect further metadata/compiler guards and expand **actual** synchronized prefixes/events. The correctness reference is actual full-prefix1..4800 replay, or an independently observed equivalent original history; complete-prefix replay itself requires state/context/cache proof and does not automatically reproduce original weights/RNG-dependent tracing. Do not declare a sparse candidate complete because it reaches three or four graphs.

Committed4800 draw-prefix SHA256 (computed from retained exact bytes in this review):

| Rank | Prefix SHA256 |
|---|---|
|0|f6c2eecc10921cf621479e005db1361e61dfc8e81d5aa8c5b6ca676c84650de5|
|1|ff893d0ba22c901d7e4f7e76934d5a041e884584205d2f7224d3abe3c6590ce5|
|2|8926cd9db4a34d5fb1d0f825d836e37bf052f3eacb94bcfeef367d99bbf6dfd3|
|3|d7a298d40fb53375a5fcfdc652592134856a40228de874f88528ddbad3fb2ca9|

The full rank3 draw file has6718 rows and a different SHA. Bind the committed prefix rather than the failed tail. `3364480/canonical_order.json` independently confirms all preserved original/recovery rows match canonical rank sharding.

## What existing evidence proves, and what is missing

`../checkpoint_metadata_checks.json` explicitly reports `recompile_trace_present=false`. Draw entries contain only rn, epoch and augmentation hash. They contain no graph event, first specialization guard, train/eval view, grad context, collated tensor bytes or runtime/compiler/cache fingerprint. They establish descriptors and ordering, not completeness of original compile history. Rank0–2's original draw logs stop at4800; their later order is canonically reconstructible, but original tensor/logit/gradient traces are absent. The rank3-only 190-sample replay observes selected events under world1 DDP and a reused cache; it cannot establish the other ranks' pre200 event histories or four-rank numerical identity. The metadata audit below rules out additional first occurrences of the checked label regimes for ranks0–2, but extra specializations from other guards, dispatch history or backward compilation remain unresolved. Decoder-label regions alone are not an exhaustive compiler guard inventory.

3361666 A/B prove old/new policy code equality from the **same cold** checkpoint200/cache starting conditions. They do not prove original uninterrupted history was reproduced. B/C prove resume divergence despite identical input/RNG and scheduler states (609/770 tensors and optimizer differ). 3364480's12 reports prove priming restores B's first-forward calculation and Dynamo reset removes it, with optimizerupdates0 and exact restored params/buffers/RNG. Its target is5041 with diagnostic B210 weights. It neither tests legacy200 bootstrap nor subsequent gradient/update/whole-checkpoint equality. Its five primers were not compressed and FX/AOT local/remote cache lookup was disabled uniformly.

Historical gate: original uninterrupted update210 logged rank0 last-micro loss `3.2567708492279053`, for slot5040/rn184594. The earlier recovery log gave `3.255345344543457`; **3361666 B's actual trace at5040 gives another value, `3.256150245666504`**, so neither successful diagnostic control reproduces that historical scalar. Require bootstrap200→210 to reproduce the original logged210 scalar exactly with verified input, LR and restored four-rank state; also compare available subsequent original logged losses. A single scalar match supports historical reproduction but cannot prove original210 weights because no original210 checkpoint/complete gradient trace is retained here. Required operational proof remains four-rank uninterrupted vs saved/resumed full model/optimizer/scheduler/all RNG/pending-gradient equality using the implemented history mechanism, through multiple updates and a future compile event. Include the400 checkpoint and a second resume, and epoch evaluation coverage. If the original loss still differs, the historical gate is unresolved even if newly created uninterrupted/resume pairs match.

## Original cache setting: supported inference and remaining uncertainty

Read-only `sacct -X -j3345919 --format=JobID,SubmitLine` returned:

```text
sbatch --parsable --dependency=afterok:3345916 --export=ALL,BINDINGS=/work/b6_prep_20260925/bindings/train_v1.deltaai.json,REPO_DIR=/work/nvme/bilv/wcyim/transgenic/repo_b6_233574a,ARM=C2 revision/scripts/b6/wp5_pilot.sbatch
```

The script retained in that original snapshot does not assign/unset `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, FX/AOT cache flags or rank-specific cache paths. One apptainer invocation launches all four accelerate ranks. Original `runtime_identity.txt` confirms job3345919, fresh `global_step=0/resumed_from=null` and original argv, but records no compiler-cache environment.

Reading installed NGC source without importing or invoking cache factories showed:

- `torch/_inductor/config.py:109`: local FX graph cache default=True (overridable by `TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT`/`TORCHINDUCTOR_FX_GRAPH_CACHE`).
- `torch/_functorch/config.py:71`: local autograd cache default=True (`TORCHINDUCTOR_AUTOGRAD_CACHE` override).
- `torch/_inductor/runtime/cache_dir_utils.py:14–35`: explicit Inductor environment path if set; otherwise `tempfile.gettempdir()/torchinductor_<sanitized username>`. This has **no rank suffix**.
- Same file`:37–44`: explicit Triton environment path if set; otherwise Inductor dir plus `triton/<device>`.

Therefore the supported default operating model is **rank-shared Inductor cache root**, with device-separated default Triton subdirectories, and enabled local FX/AOT lookup. This is an inference from source/deployment defaults, not a recovered original environment receipt. Submission used `--export=ALL`; inherited overrides, TMPDIR, actual cache contents/hits and startup script effects are unrecorded. Do not claim exact historical cache configuration is certified. Operational validation should use these production defaults/shared layout without the diagnostic cache disables and record actual effective values and paths. Exact historical210 reproduction remains its own gate.

## Main correctness hazards

- **DDP collective sequence:** never replay independent per-rank event lists. Each global slot must include actual companions and matching train backward/grad-enabled or eval no-grad behavior. Do not insert arbitrary padding examples, change no_sync behavior or skip backward only on ranks without events. A local replay failure must reach collective error handling before peers enter unmatched collectives. Reducer internal state and buffer broadcasts are not restored by restoring weights; existing `has_rebuilt_buckets=0` controls exclude that specific cause, not all replay side effects.
- **Eval and checkpoint boundaries:** production evaluation runs de_novo then evidence for every eval batch under model.eval/no_grad (`train_HyenaTransgenic_B6.py:354–373`). These need separate view/context descriptors and a synchronized position distinct from training slots. Original200 is epoch0 before its first epoch evaluation, so no legacy eval prefix is implied. Future latest_state/step/epoch checkpoint writers must save the same complete history prefix, including pending accumulation and mode transitions.
- **Parameters/buffers/RNG/gradients:** `first_forward.py:16–40` preserves object references, None entries, new-buffer deletion and nonpersistent registries, but its diagnostic target has pending0 and it clears all grads. That clear is not valid for pending checkpoints. Restore each existing gradient value/None and preserve references where necessary, per-submodule mode, all device RNG state and parameter/buffer aliasing. Guard reuse after restored buffer shapes/None registries must be verified, not presumed. No optimizer or scheduler step during replay.
- **Caches and hint-dependent fusion:** `compile_b6` disables automatic DNA dynamics and optimize_ddp (`b6_runtime.py:621–637`). `compiled_graph_count` counts Dynamo unique_graphs (`:640–642`), not every AOT/Inductor/Triton compile or dispatch detail. A forward-event-only ledger may miss lazy backward compilation or changed guarded dispatch without that counter increasing. Observe/validate after train backward too. Same input-tier counts do not prove same generated kernel fusion; first labels58/534/708/1011 etc matter. Test with production cache options and actual shared-rank cache layout; Dynamo reset does not erase disk caches. Bind runtime/container/compiler/hardware/autocast/checkpointing fingerprints and guard context, and reject mismatches.
- **Durability/identity:** binding a ledger to run identity alone does not bind a particular numerical checkpoint. Store/verify exact history prefix length/hash and checkpoint position, all rank companion records and tensor hashes atomically with checkpoint publication. Reject missing/tampered/incomplete rank histories or world-size mismatch before collective replay. Preserve original200 and failed-tail evidence.

Graph-first review used current worktree graph spans for trainer/runtime/dataset/context/tokenizer and main graph spans for diagnostic scripts. No production-readiness assertion is made.

## Small derived metadata receipt

The following exact UTF-8/LF CSV summarizes first occurrences of the checked regimes. It is derived from the read-only19200-row audit; it is not an observed compiler ledger or a hash of the entire raw DB query result. SHA256: `cc85a38a07d309c6afa12dbb9d0846cdfb0a0579d385866f3dd43d37d69b86c0`. No raw GSF/tensor payloads were written by this review.

```csv
rank,slot,rn,DNA_L,labels
0,1,207602,61440,58
0,3,113494,129024,651
0,4,95461,30720,269
1,1,185278,61440,534
1,2,70488,30720,48
1,3,26212,129024,1106
2,1,162083,30720,245
2,4,210585,61440,708
2,15,85437,129024,202
3,1,91492,30720,33
3,3,146698,129024,80
3,7,117991,61440,1011
3,3915,10431,129024,4501
```
