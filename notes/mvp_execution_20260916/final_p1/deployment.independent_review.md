# Independent final P1 deployment-delta review — 2026-09-16

## Disposition: APPROVE one bounded diagnostic

Approve this exact release for **at most one separately controller-authorized S0 + real_tokenizer diagnostic**, one exclusive ghx4 node/four GPUs, 30 minutes, subject to the no-overlap and evidence conditions below. No blocking deployment-delta or release-closure defect found. This is not submission permission, runtime PASS, six-gate readiness, resource-mode approval, training/decode authorization or scientific approval.

Read `../final_source.independent_review.md`: it independently ACCEPTS source **b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6**, executable integration **362eb899d8d43164ec1e0fb0f3b174ead9b1ba40**, for staging/export, fresh bindings and separately authorized bounded diagnostics only. Its unresolved whole-suite native crash and genuine SIF/runtime limitations remain unresolved; this review does not erase them.

## Exact approved ledger

Paths in this block are relative to `final_p1/release/`. Independently recomputed outer `release.sha256` SHA256:

`5b36bdf09a8b8eb548147654e23d6ad59fe18542642a05eb720f22be102fdfb8`

```text
0fb803cf0612623fa0023bb1f0f28b4b2f6a227a930833f50f74cd31a64e9cd2  code.tar
49922d81bc11eb17d651f0ea4132f0d1a8e6bcc3299dcc026d8bf6182c8b8cc6  deploy.py
950e2a54e40e8d84471b8f0907b1fb168b967cb283d1a1107f040d7d0b4f035f  p1.sbatch
faab8387cbbe87812c7f3faf5412d19004874a5cc439f5a3d182e2c976a5b51e  notes/b6_mvp_technical_candidate_20260915.json
9f79019679bfdd25b0a2bdac7cae63d01fa6d874be03fa3ef3f81cfd0a880a98  notes/b6_mvp_candidate_inventories_20260915.json
73204a4ed98624b1567a5eb26fd6b2bbafbb597e0bd4b807685e8c703c807bac  notes/b6_mvp_candidate_receipts_20260915.json
7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689  release.json
```

Other reviewed SHA256 identities (relative to `final_p1/` unless qualified):

| Artifact | SHA256 |
|---|---|
| `build_release.py` | `3c98184c17bc4e15130c75ebebd69cd0157895b3d015e6a7cb542e2634604405` |
| `test_deploy.py` | `4661cdba74bc4c016aaa1d720c36d990af6d51b4328798cda4513a657680a396` |
| `old_p1_preservation.json` | `d349410e62b0a48a8c480dfd0026639bd97fc8bf55916a4658e702f03aa6f1d1` |
| `review_status.json` (pre-review PENDING snapshot) | `6e02e609dabc0fd6948301c1baa01cff8f07db5bd8c9d9b77150758a80c1e645` |
| `notes/MVP_FINAL_P1_DEPLOYMENT_PLAN_20260916.md` (repo-relative) | `a3e43f579d820155438de86eb7d6cd246fe4547345a36c7a8c557834583c9d33` |
| `../final_source.independent_review.md` | `3462343dd16b80456191366398082cebcd307a347e0a9d90a48738efaa045f37` |

Top-level deploy/bootstrap bytes equal their release copies. This report records approval without editing the pre-review status JSON or plan.

## Independent delta and byte verification

Compared all four helper/builder/bootstrap/test files against the exact old `p1/` files approved in `p1/deployment.independent_review.md`. Recomputed the complete old preservation inventory: **all20 paths and hashes match**, including the old review, release and logs. Thus the old approved helper conclusions are reused against verified bytes, not presumed from filenames.

- `deploy.py:14,175–190`: only source commit, `staging/final_p1_release`, `final-p1-job-<ID>` and the early missing-evidence guard changed. Guard runs before fresh job-tree creation. All other helper bytes, including staging, tile checks, bindings and P1 execution, are unchanged.
- `build_release.py:12–14,26–63`: two exact historical producer dependencies join the existing fixed Git archive roots. Source bytes come from Git objects, not any worktree. Existing-output refusal and separately attributed candidate companions remain intact.
- `p1.sbatch:2,13–15`: only job name, chdir and log names changed. Exclusive marker, environment sanitization, checksum verification, allocation shape and timeout are unchanged. Job name is `mvp8-final-p1-0916`; log prefix is `staging/final-p1-`.
- Test delta adapts these paths, checks missing evidence before collision handling, and adds complete closure/pin/772-preservation and old-file-preservation checks. No readiness/staging algorithm or gate logic was changed by deployment preparation.

Independently recreated `git archive --format=tar b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6` in memory with the builder's seven path selections: **entire archive bytes equal release/code.tar**. Verified all **100 regular files / 1,832,960 archive bytes**, unique members, directories/regular files only, exact `git ls-tree` inventory, every `git show <commit>:<path>` payload and every manifest hash. No `.venv`, `.testdeps`, `site-packages`, `__pycache__` or graft payload; no mutable/dependency-worktree contamination. Both added producer helpers and resource/decoder/bootstrap/windows code are covered by this exact inventory.

Recomputed from Git-verified bytes using sorted compact JSON plus newline:

- **86-file source pin:** `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`.
- **13-file decoder pin:** `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb`; path list extracted from the actual `DECODER_ONLY` AST literal.
- All **74 pre-existing772 component files** equal their immutable772 Git bytes. No duplicate semantic review of unchanged hundred-file source was needed; final integration semantics rely on the separate accepted source review.

Seven ledger entries, manifest release-file mapping, three main/release companion identities, config/panel small pins, helper-copy identities, Python3.6 grammar/in-memory compile and `bash -n` independently PASS. These are local checks, not Python3.6/SIF execution. Controller-reported **11 tests PASS in15.03s**, syntax and seven-ledger PASS are attributed evidence; the unittest suite was deliberately not duplicated.

## Retained execution limits and controller conditions

`deploy.py:204–251` retains compute-only quota/large hashing/checkpoint/cache-copy work. Corpus and SIF are never copied; their existing paths are mounted read-only. B5 seed123 is mounted read-only only for staging, with concrete epoch05 and unchanged sidecar/cache pins; no best/latest fallback. Source/HF/release stay read-only, staged inputs are writable only during creation then frozen/read-only for P1. Expensive checks/copies remain inside the allocation, not login. This verifies command construction and byte preservation, not actual site mount enforcement.

`container_stage` remains fresh `prepare`, not readiness publication. Fresh full-unified consumed-code/package/tooling/input/SIF bindings must be generated inside the actual pinned SIF; no d99 receipt reuse. `deploy.py:381–403` runs only real_tokenizer, inner900 seconds. `p1.sbatch:11,38–40` keeps Slurm1800 seconds, combined outer1700 plus20-second termination grace. No warmup, retry, P2/resource/generation/four-rank/native-bf16 gate or training is authorized. Missing valid DONE or any timeout/drift is failure.

**Old job3163545 must not overlap this attempt.** No live state was observed here. After both source and deployment reviews, only the controller may obtain and retain fresh timestamped owner/state/accounting proof for3163545, own jobs and B5 chain. Only its own still-PENDING3163545 may be cancelled under the pending-only permission; recheck state and confirm terminal cancellation before replacement. A race to RUNNING means STOP under that permission: let it finish and review outputs/accounting. Finished/failed/cancelled jobs still require output, charged-allocation and evidence-location review. Unknown state means STOP. Reviewers perform no cancellation.

`deploy.py:179–180,217–226` requires the original staged evidence and verifies its identity before renaming it into the fresh job. Failure after that move leaves evidence in the old job; **STOP, preserve provenance, no automatic recovery**. Do not move/copy it back, search/reuse old partial inputs, delete markers or resubmit. Recovery needs separate review/authorization. Before any transfer/submission the controller must verify staging evidence, absent new release destination, live exclusions/quota, remaining cumulative budget and authorization; externally verify this ledger hash and all entries after transfer. At most one job, record its ID immediately, no queued follow-ups. Thirty minutes costs at most0.5 node-hour/2 GPU-hours within—not additional to—the cumulative5 node-hour/20 GPU-hour cap including prior attempts and overhead.

## Deferred Git authority and review boundaries

Later panel/evaluation `frozen_reference` consumers require actual Git object `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv` in their operational root. Transport remains **DEFERRED_UNVERIFIED**. This archive is neither Git-object transport nor authority evidence. P1 does not call that consumer: this later requirement is truthfully retained, not imposed as a P1 prerequisite.

No SSH, scheduler queries, jobs, cancellation, subagents, dependencies, release rebuild, source edits, merge or push occurred. Required graph orientation auto-refreshed local graft metadata (reported four files); no source was changed by this review. This report is the only intentional deliverable. All runtime uncertainties and future gates remain open; APPROVE here means only the bounded diagnostic scope above.
