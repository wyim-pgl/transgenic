# Independent static reconciliation review

## Decision: static reconciliation justified; DRAFT / BLOCKED for execution

Reviewed `EST_C1_REAL_MANIFEST_REPORT.md` and the complete `reconciliation_proposal/` bundle against code commit `f6bef565f0bb432256b7e596a81b896afd2266c5` in `/tmp/b6-p2-phase-1535433`.

Manifest reviewed: `90ce4404d3d0d0547c21a1ccf7db96344b97413e10d4bdd167e03976025a9ca9`.

No static wrong-root, expanded exception, inconsistent nested snapshot, unaccounted receipt bytes, or replacement of a pinned wrapper/lineage/pilot anchor was found. This conclusion is conditional on the operator's capture provenance, not cryptographic proof of remote execution or present filesystem state. It is NOT an operational auditor PASS, staging approval, or execution authorization.

The full queue correctly retains `6165461|RUNNING|est_c1_20260916_1530` at **2026-09-16 23:09:58.284713 UTC**. That row fails the reviewed validator. Do not filter it, treat the capture as quiescent, or deploy this manifest as approval.

## Independent verification performed

- Independently recomputed all **19 bundle SHA256SUMS entries**, with exact coverage of every bundle file except SHA256SUMS itself; all matched.
- Strictly decoded and hashed **32 unique receipt paths / 489,199 bytes**. Every byte count and SHA256 matched. The decoded-file set equals the union of snapshot file leaves: no missing or extra captured bytes. Repeated/nested snapshot references agree exactly.
- Checked exact manifest keys/schema/root, unique coverage of all 19 authoritative paths, all job lists, wrapper paths/pins, file/directory kinds, numeric UID/GID, child listings, and file hashes. Every node, not only top-level entries, records UID/GID **3092957/3092957**.
- Checked all four pinned local deployment source files both against their actual worktree bytes and `git show` bytes at the reviewed commit. This includes auditor, launcher, sbatch template and `64_orthodb_filter.py`; no reliance on HEAD label alone.
- Used reviewed auditor constants and the isolated `reconciliation_paths` definition, not proposal-derived authority. Source authority: `revision/scripts/evidence/est_blastx_audit.py:57–88,462–625,628–735`. No auditor main routine or collector/verifier utility was executed.
- Compared parsed `static.stdout.txt` to `static_facts.json`; compared accounting stdout byte-for-byte to `accounting.psv`, and queue stdout to `queue.json.stdout`. Both manifest capture hashes match the local bytes.
- Checked the saved remote program byte-for-byte against the collector's literal template instantiated with its saved configuration. Configuration is the closed 19-path map plus three wrappers and two pilot logs (preflight wrapper overlaps its tree). No source-tree, DB, FASTA or raw13 traversal is specified. Unknown directory listings, links, special files and files over 1 MiB fail capture.
- Transport records specify read-only `sacct`, bounded Python stdin capture, and unfiltered `squeue --noheader --format="%i|%T|%j"`; return codes are zero. Stderr is the preserved default-environment/module banner, not empty and not a reported command failure.
- Accounting has exactly 30 unique rows: pilot, eight array tasks and preflight, each allocation/batch COMPLETED 0:0 and extern OUT_OF_MEMORY 0:125. The extern anomaly is preserved, not waived.

## All 19 proposed paths

All paths below are relative to the exact root `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx`. D = directory; F = regular file. Every row's full recursive snapshot equals its captured counterpart. No wildcard or parent-root waiver is proposed.

| Path | Job binding | Kind / reviewed contents |
|---|---|---|
| `sitalica_pilot_20260914/tmp` | 6163851 | D, empty |
| `sitalica_pilot_20260914/execution.lock` | 6163851 | D, empty |
| `sitalica_pilot_20260914/submission.lock` | 6163851 | D, only jobid `6163851\n` |
| `remaining8_20260915/submission.lock` | 6164049_0 through _7 | D, only jobid `6164049\n` |
| `remaining8_20260915/Athaliana/tmp` | 6164049_0 | D, empty |
| `remaining8_20260915/Bdistachyon/tmp` | 6164049_1 | D, empty |
| `remaining8_20260915/Gmax/tmp` | 6164049_2 | D, empty |
| `remaining8_20260915/Osativa/tmp` | 6164049_3 | D, empty |
| `remaining8_20260915/Ppatens/tmp` | 6164049_4 | D, empty |
| `remaining8_20260915/Ptrichocarpa/tmp` | 6164049_5 | D, empty |
| `remaining8_20260915/Sbicolor/tmp` | 6164049_6 | D, empty |
| `remaining8_20260915/Vvinifera/tmp` | 6164049_7 | D, empty |
| `preflight_20260914` | 6163397 | D, exact closed 20-child layout |
| `preflight_20260914/logs` | 6163397 | D, stdout and stderr only |
| `preflight_20260914/receipts` | 6163397 | D, exact five receipt files |
| `preflight_20260914/submission.lock` | 6163397 | D, empty |
| `preflight_20260914/tmp` | 6163397 | D, empty |
| `preflight_20260914/xdg-cache` | 6163397 | D, empty |
| `preflight_20260914/execution.lock` | 6163397 | F, zero bytes; not a directory |

Pilot rows bind `run.sbatch` SHA256 `8e8f501ebaa0cbd15ee8d61aef127f5cb7ee3fdc3a070d3c7d18923440de2f4a`; remaining8 rows bind `run.sbatch` `4cdad18b443c13ad564be640859419e86def7fb255fe82cabe797888913fdd7c`; preflight rows bind `prepare.sbatch` `dfe295d76d9f4f15a5b153e6dcd4fb1e9fbd658aa1a6d59539c92bb8c33bf318`. Decoded bytes match all three code-authoritative pins.

The array species order matches reviewed code. The captured wrapper obtains the actual species from historical inputs.json; these 32 receipts do not include that file or the 16 remaining8 logs. This review verifies the prescribed map, not a new independent reconstruction of array runtime identity. Full-tree unit/input/log auditing remains necessary.

## All 32 receipts

Each row below was decoded and independently byte-count/hash checked. Full independently checked digests are in the bundle's `RECEIPTS.md` and `static_facts.json`. Prefix P = `preflight_20260914/`, R = `remaining8_20260915/`, S = `sitalica_pilot_20260914/`, under the root above.

| Receipt | Bytes |
|---|---:|
| P accounting-bounded-check.txt | 223 |
| P batch.exitcode | 2 |
| P duplicate-check-correct-user.txt | 820 |
| P duplicate-check.txt | 0 |
| P execution.lock | 0 |
| P expected-sources.sha256 | 322 |
| P independent_lineage_audit.json | 1946 |
| P jobid | 8 |
| P lineage_ncbi.baseline.tsv | 52988 |
| P logs/slurm-6163397.err | 0 |
| P logs/slurm-6163397.out | 2699 |
| P orthodb/DONE | 45 |
| P orthodb/lineage.tsv | 52988 |
| P orthodb/provenance.json | 8762 |
| P prepare.sbatch | 5088 |
| P receipts/DB.sha256 | 359 |
| P receipts/PROVENANCE.txt | 684 |
| P receipts/SOURCE_COMMIT.txt | 8 |
| P receipts/build_dbs.sh | 1839 |
| P receipts/swissprot_viridiplantae_reviewed.fa.gz.sha256 | 105 |
| P reviewed-small-files.sha256 | 516 |
| P sbatch-receipt.json | 62 |
| P source-hash-attestation.txt | 198 |
| P swissprot/DONE | 45 |
| P swissprot/lineage.tsv | 316260 |
| P swissprot/provenance.json | 36174 |
| R run.sbatch | 3549 |
| R submission.lock/jobid | 8 |
| S run.sbatch | 3103 |
| S slurm-6163851.err | 0 |
| S slurm-6163851.out | 390 |
| S submission.lock/jobid | 8 |

Preflight source DONE files contain exactly `Outputs and QC complete; see provenance.json\n`. Its root has no invented DONE. Jobid, zero batch exitcode, submission receipt and lock kinds/content satisfy the reviewed contract. Historical duplicate/accounting checks showing RUNNING in September 14 are retained historical observations, not current quiescence claims.

Both lineage hashes equal code pins and provenance/report references. Independently parsed complete TSVs and provenance taxid maps: OrthoDB has 412 cache rows, 408 observed taxa, 12,115,085 recorded sequences and 18 cached-ancestor suffix checks; Swiss-Prot has 2,427 rows, 2,208 observed taxa, 42,106 recorded sequences and 56 suffix checks. Recomputed self/root/Viridiplantae, uniqueness, cached-suffix consistency, baseline-overlap equality, counts and new-cache-key checks match the report. Source hashes agree across expected-sources, provenance and small source receipts; the log's embedded lineage report equals its JSON file. DB.sha256 is preserved inside PROVENANCE.txt; SOURCE_COMMIT is `950cf4c\n`. These are internally consistent records, not new reads/hashes of their referenced large sources or proof of authoritative ancestry/merged IDs.

Recovered pilot extract SHA256 `f45de648f3491133890420fdb4a374299a55a169d199eaf4dc79c608e7ed7b23` equals the auditor anchor and original operations-bundle copy. Also checked original local session records 540/541: their line-ending-stripped raw-record hashes match the extract. Call/result IDs pair; the command explicitly submits with the historical root, chdir and `slurm-%j.out/.err`, and the result returns 6163851. No assumption of default Slurm log naming or synthetic substitute was used.

## Limits and documentation qualification

1. Hashes authenticate equality to selected bytes/anchors, not that an unsigned SSH transcript truly came from the claimed host at that time. The transport JSON has no signed server attestation or independent stdin execution receipt. Saved collector/program/stdout consistency supports operator capture provenance; it cannot exclude a coordinated fabrication of unanchored records. No such internal inconsistency was detected.
2. Capture is not atomic. File before/after size/inode/mtime checks and bounded no-follow reads help, but directory snapshots and earlier path stat ownership can race; they do not prove continuous immutability, absence of external writers, or that a flock is unheld. An empty lock file is not lock-ownership evidence.
3. Timestamp wording in the main report needs qualification: remote `static_facts.captured_at` is 23:09:58.613982 UTC, whereas local static transport ends at 23:09:57.891530 and local final queue capture is 23:09:58.284713. Collector sequencing supports “queue after static command returned,” not synchronized-clock wall-time ordering. Host clock offset was not measured. This is not evidence of forgery, and does not affect the already-blocking RUNNING row, but do not claim demonstrated cross-host timestamp ordering.
4. `verify_local.py` writes report/pins/status/checksum files; it is not a read-only verification entrypoint despite being offline. Neither it nor `collect_local.py` was run here. Local checks used independently written in-memory decoding/assertions only. No raw DB/FASTA was accessed or replaced with a fixture.

## Precise outstanding conditions — no execution approval

- Preserve baseline job 6165461 and its outputs. After natural completion, independently inspect final accounting, exit status, manifests and results; do not cancel, resubmit or patch it to obtain a clean queue.
- Obtain new historical accounting and separate final baseline accounting, retaining anomalies. Obtain a **fresh full unfiltered successful queue capture after baseline completion**, with no relevant running/held/pending EST or historical producer jobs. The queue must be timezone-aware, nonfuture and at most one hour old at auditor startup. Recheck if the window expires or relevant jobs appear.
- Revalidate all 19 live path snapshots, every UID/GID/kind/listing/hash, all three wrappers, pilot receipt/logs and preflight evidence. A historical snapshot cannot establish present lock safety. Resolve any changed/unknown content fail-closed; never broaden the allowlist or fabricate DONE.
- Rebind the new queue and any refreshed accounting/static bytes into a **new immutable manifest**, record its new digest, and independently review that exact revision. This review and the old manifest digest do not authorize later edits.
- Independently confirm the planned external `est_followup_20260916_1530/c1_reconciled/inputs` and new run/output/snapshot destinations are exclusively available. Stage only under separate authorization, outside source and input trees, without overwrites; verify actual deployed code/input bytes and exact invocation against the reviewed commit/template. The proposed paths are not proven to exist or be absent by this bundle.
- Complete the real reviewed auditor's whole-tree/input/unit checks and final snapshot/hash rechecks. Static receipt reconstruction is not a substitute for real payload validation. Keep extern OOM, ancestry/merged-taxid authority, C2/C3, phase acceptance and production authorization unresolved.

Review used graft before source inspection (estimated 844,458 tokens saved). Advisor consultation failed due to client/model version incompatibility; no advisor review is claimed. No SSH, job action, source edit, subagent, merge or push was performed. Graft itself reported automatic graph refresh on initial orientation; only this review report was intentionally written.
