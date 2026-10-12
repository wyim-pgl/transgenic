# Independent C1 final artifact/provenance result review

## Verdict: TECHNICAL PASS — narrowly scoped C1 acceptance

Accept the reconciled **C1 artifact/provenance audit result**, job **6165525**, against reviewed code **`f6bef565f0bb432256b7e596a81b896afd2266c5`** and the exact approved refreshed manifest. No blocking inconsistency was found in the reviewed result, execution closure, or reconciliation delta.

**This does not resolve external-step OOM, complete C2 HSP acceptance, approve genomic phase policy/projection, authorize production, or establish scientific validity.** The three false policy flags remain false. This review independently checks reports and their execution/provenance consistency; it does not independently recompute raw-data biology.

Evidence directory throughout: [`est_execution_20260916_1530/`](est_execution_20260916_1530/). Primary inputs are `reconciled_result/`, `reconciled_final_status.txt`, `reconciled_submit_receipt.txt`, `reconciliation_refreshed/`, its adjacent `reconciliation_refreshed.independent_review.md`, and `c1_baseline_result/`.

## Identity and execution closure

| Item | Independently checked identity |
|---|---|
| Final rollup, 57,210 bytes | `eae5bdd060cc629aad254c24c41816e7c9fa5a0ea42d9780ef0f3b9675c7501c` |
| Exact refreshed manifest | `c8b03a35bcf527b70d69fa4c11eb0ab2d90da1da3dd0b487d4dd85c6bfe00a2e` |
| Actual wrapper | `763d655c2043b20bbe9882411589f6542112c8a65018e7eda638042a04080933` |
| Auditor | `b4392729aaff73feb61e4ffef8df7b6ddc66235ec6aa1fd988babb0da46bebfb` |
| Runtime launcher | `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291` |
| Runtime taxonomy helper | `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302` |
| Queue capture | `fc30d8cb7e0f85d96416ba7587c4981853a30a54e4ad2f83081b5797a3722a6e` |
| Historical accounting | `61120e921d3c8897e46608e02a8c46d9233812f71791c024a2626150bfbbb237` |

All **28 local bundle checksum entries** were recomputed successfully. All **four source/template files** match their pins, the existing C worktree, and `git show` bytes at f6bef56. Deployment tar members match the reviewed local files; its `deployment.sha256` is `c3c7d9533d8520e9579648ac9ff4b2acf07725e5c1b76973498617371e7e833b`. The rollup's three runtime-code hashes match those verified bytes.

A read-only SSH closure check independently queried `TZ=UTC sacct` and hashed **33 small files totaling 2,533,782 bytes** under the reconciled execution directory. **29 remote/local comparisons matched**: four code/template files, four inputs, wrapper, DONE, rollup, and all 18 unit reports. The remaining four observations were the deployment checksum listing, batch exit code, and two Slurm logs. No raw output, database, protein FASTA, or EST FASTA was read or hashed remotely by this reviewer.

Observed scheduler closure:

| Step | State | Exit | UTC start → end | Elapsed |
|---|---|---|---|---|
| 6165525 | COMPLETED | 0:0 | 2026-09-17 02:07:28 → 04:56:42 | 02:49:14 |
| .batch | COMPLETED | 0:0 | same | 02:49:14 |
| .extern | OUT_OF_MEMORY | 0:125 | same | 02:49:14 |

The persistent execution guard exists; `batch.exitcode` is exactly `0\n`. Remote audit inventory is exactly `DONE`, `rollup.json`, `units`; the local copy agrees. No final `FAILED.json` is present. All 18 unit inventories contain eight reported files and no FAILED record. The 416-byte stdout contains nine successful startup pin checks (including the interpreter); stderr is empty. Stdout SHA256 is `c24f60e14b24df11205a253cc56d1e1093132d0b5217ec7f0099bd4a2173bda3`. Empty stderr does **not** negate the independently observed extern OOM. DONE's generic text refers to provenance.json; it is not used as standalone proof of acceptance.

## Queue freshness: timezone correction and unchanged gate

The queue was captured at **2026-09-17 01:57:58.018983 UTC = September 16 18:57:58.018983 PDT**. Submission was **02:07:27.986641 UTC = 19:07:27.986641 PDT**; allocation/batch started at **02:07:28 UTC = 19:07:28 PDT**. Thus **19:07 PDT corresponds to 02:07 UTC, not 01:57 UTC**. These are a capture followed approximately ten minutes later by submission/start, not conflicting representations of one instant.

Capture age at Slurm start was **569.981017 seconds**, comfortably inside the unchanged nonfuture **0–3600-second** startup gate; expiry was 02:57:58.018983 UTC. The 68 saved queue rows exactly equal saved capture stdout and satisfy the original field, historical-job-prefix, and `blastx`/`lineage`/`est_` name predicates. No self-row removal or freshness extension was introduced. The later audit's own queue row is not retroactively added to this fixed capture.

The exact compute-side gate time is not recorded as a separate timestamp. Its successful execution is supported by the pinned code, nine startup pin checks, and PASS `historical_reconciliation`, whose embedded evidence equals the exact reviewed manifest. The check runs before the expensive unit loop, not at the audit's later completion: expiry during the 2h49m run is not an acceptance bypass. Relevant reviewed spans: bundled `est_blastx_audit.py:554–625,738–832`.

## Exhaustive result consistency checks

Both rollups and **all 18 complete unit JSON records** were parsed, including their full taxonomy/count maps; this was not a top-N or sampled unit review.

- Exact nine-species × two-source inventory, with Sitalica in pilot layout and the other eight in remaining8 layout.
- **18/18 PASS**, **324/324 unit checks PASS**, **117/117 tree checks PASS**, zero failed units/checks.
- **19/19 historical final snapshot checks** cover exactly the manifest entry paths. **20/20 producer-artifact final checks** cover the two producing wrappers and 18 stdout/stderr logs; final recorded hashes agree with their initial inventory bindings. These are executed-checker observations, not a fresh reviewer snapshot of the historical tree.
- Species batch-exit/timing checks, unit DONE/closed-directory inventories, eight-file manifests, producer identities, screening, query resolution, raw13 fields/counts, subject and query-ID membership, and frozen command bindings are all present and passing.
- Historical accounting text equals the supplied 30-row capture byte-for-byte. The rollup's 27 allocation/batch/extern checks concern the nine producing jobs; the capture additionally includes the preflight job's three rows. An accounting check marked PASS records the expected extern anomaly; it does not mean clean accounting.
- `clean_accounting=false`, `accepted_phase_evidence=false`, `production_authorized=false` in both baseline and final rollups, consistent with hard-coded false assignments at `est_blastx_audit.py:817–828`.

### Queries, raw counts, and unchanged baseline payloads

Each count below is an **executed auditor report**, checked for internal equality against recorded provenance and the baseline report, not independently recounted from raw data by this reviewer.

| Species | Screened query records | OrthoDB raw HSPs | Swiss-Prot raw HSPs | OrthoDB / Swiss HSPs <30 aa |
|---|---:|---:|---:|---:|
| Athaliana | 1,046,825 | 20,071,843 | 4,672,257 | 536,066 / 206,829 |
| Bdistachyon | 206,123 | 4,518,360 | 1,363,835 | 8,228 / 4,860 |
| Gmax | 1,374,461 | 22,401,265 | 6,825,751 | 370,000 / 130,494 |
| Osativa | 1,244,141 | 22,058,626 | 5,964,282 | 117,645 / 49,963 |
| Ppatens | 381,809 | 6,923,738 | 2,533,444 | 6,946 / 6,635 |
| Ptrichocarpa | 89,450 | 1,905,270 | 587,346 | 6,583 / 2,628 |
| Sbicolor | 209,441 | 4,402,714 | 1,625,119 | 14,109 / 7,459 |
| Sitalica | 66,040 | 1,371,252 | 290,009 | 3,868 / 1,290 |
| Vvinifera | 450,180 | 9,275,327 | 3,601,162 | 25,874 / 15,092 |

Totals: **120,391,600 raw HSPs**, including **1,514,569 <30-aa HSPs**. These are raw screening/search counts, not accepted HSPs or performance measures.

All **18 entire parsed unit reports equal their baseline counterparts**, a stronger check than raw-hash equality alone: all 144 reported file hashes, including all 18 raw13 hashes, all counts, query hashes, taxonomy maps, commands, and check details are unchanged. Full raw hashes remain in the linked unit records rather than being abbreviated as new authority here.

Pilot query resolution binds `query.sha256` to provenance and reported fresh SHA256. Remaining8 binds the hashed `inputs.json` species row and query path, then the report's fresh SHA256/MD5. All command query paths resolve to `est/<species>/univec/est.univec.fa.gz`; query-record counts equal screening counts and every reported orphan set is empty. The reviewed `production_inputs` implementation (`est_blastx.py:168–196`) additionally requires matching source/query/lineage/roles hashes, one training-eligible role row, A21 screening artifacts, and nucleotide records ≥100 nt. These are claims of the executed checker, not separate reviewer inspection of those large inputs.

### Source, taxonomy, and command checks

All nine OrthoDB units use the same A19-prefiltered source (`evidence/protein/odb12_Viridiplantae.filtered.fa.gz`), SHA256 `fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f`, with 12,115,085 input proteins. All nine Swiss-Prot units use `evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz`, SHA256 `8d66a18e2ad350fd640f1fb97a180a8091c544efd33351e5f451bb03e11379c9`, with 42,106 input proteins. Each lineage digest equals its reviewed source-specific pin.

For every unit, summed taxid input/output counts equal reported totals, excluded counts equal input minus output, retained-taxid lists exactly match positive output counts, excluded taxids have zero output, and retained counts equal their input counts. The focal species root is absent from every retained set; each regenerated-filter digest equals the unit's filtered-FASTA digest. Athaliana OrthoDB legitimately reports zero additional exclusions: taxid 3702 is already absent from the supplied prefiltered source. Osativa OrthoDB excludes six listed taxids, including subspecies, while Swiss-Prot excludes three present source taxids. Neither pattern is a missing-check failure. This establishes consistency with the frozen lineage implementation, **not** independent authoritative ancestry/merged-taxid validation.

All 18 command reports bind the same pinned DIAMOND identity (`0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`, recorded version 2.1.24), database/query/output paths, 16 threads, frameshift 15, e-value 1e-5, min-orf 30, max-target-seqs 25, max-hsps 1, no-parse-seqids, and the exact 13-field output contract ending in BTOP. The historical launcher/helper pins agree with the recorded identities. No new DIAMOND execution or independently obtained version-banner receipt was performed here. Source exclusion precedes search; no post-top25 exclusion was substituted (`est_blastx.py:117–143,293–315`; auditor `:298–402`).

## Why the baseline FAIL becomes PASS

Baseline job 6165461 remains a failed audit; its historical FAILED record is not erased or reclassified. Its **18 unit reports already passed**. Its **12 failed tree checks** were one pilot log-receipt binding and eleven historical directory-disposition checks. The final run supplies the reviewed recovered pilot submission receipt and exact historically bound reconciliation evidence, then repeats all unit auditing and the final small-artifact checks.

This changes the provenance/tree interpretation of explicitly enumerated historical sentinels, scratch, and preflight containers. It does **not** alter raw hashes/counts, fabricate historical DONE markers, skip unit auditing using baseline manifests, relax missing/failed unit gates, waive startup freshness, or turn scientific/production flags on. The final embedded manifest equals the approved manifest; final historical and producer-artifact rechecks all pass. Thus the observed delta supports narrowly scoped technical acceptance, not a general acceptance bypass.

## Trust limits and outstanding gates

- Remote SSH command output and local receipts are unsigned observations, not hostile-environment attestation. Scheduler clocks, filesystem state, and normal command execution are trusted within the recorded workflow. The on-disk wrapper hash is not a hash of Slurm's spool copy.
- The complete raw dataset (~117 GB) was **not** reread or rehashed by this reviewer. Report identity checks establish that the fetched records match remote records; they do not independently establish biological truth. Final small-artifact rechecks do not claim a second full-payload hash or atomic isolation from every possible external writer.
- OOM remains independently observed on 6165525.extern and unresolved on historical producer extern steps. Allocation/batch completion cannot authorize a clean-accounting claim or diagnose the cause.
- C2 HSP semantics/acceptance, BTOP/frameshift interpretation at scientific acceptance level, genomic phase policy/projection, C3, production/training authorization, and authoritative taxonomy limitations remain separate unresolved work. Raw short-HSP counts and min-orf settings must not be presented as downstream ≥30-aa acceptance.
- No subagents, scheduler submissions, remote writes, administrative actions, source edits, merge, or push were used. Only this requested report was intentionally written. The first optional remote file check found `python3` absent on PATH; it was rerun successfully using the already reviewed absolute interpreter. An advisor consultation was unavailable due to client/model incompatibility and supplies no review evidence. No test suite was rerun or newly certified.

**Final decision: accept C1 artifact/provenance TECHNICAL PASS only; keep extern-OOM, C2/HSP, phase-policy and production gates explicitly unresolved.**
