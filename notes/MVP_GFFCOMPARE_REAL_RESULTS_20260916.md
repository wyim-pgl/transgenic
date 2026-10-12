# GFFCompare 6165521 — local recovery of actual captured results

## Verdict and limits

**Recovered semantic checks PASS for all 16 frozen synthetic cases. Controller independent review of this recovery is PENDING.** This is a read-only results audit, not a new engine run or a declaration that the whole tool/pipeline is validated.

Three distinct observations must remain separate:

| Layer | Observed outcome |
|---|---|
| Captured GFFCompare/function semantics | Ten raw comparisons returned **0**; seven producer comparisons produced valid counts/maps; six mono/empty bypass cases and three fail-closed duplicate guards behaved as planned. All 16 expectations match. |
| Original Slurm job/allocation and batch | **FAILED, 1:0, 00:00:02**. Original analyzer raised `AssertionError: ordinary` at `smoke.py:167`. This status is not repaired or relabeled. |
| Original `.extern` | **OUT_OF_MEMORY, 0:125, 00:00:02**. Cause unresolved. Do not infer either an engine OOM or resolution from successful captured calls. |

No SSH, submissions, engine reruns, installs, producer/source-runtime edits, fixture/threshold changes, subagents, merge, or push were performed. Existing harness, source, staging manifest, plan/review, logs and raw outputs are preserved. No results or maps were synthesized or substituted. Tests modify only disposable local copies.

## Evidence and provenance verified

Read the original [smoke plan](MVP_GFFCOMPARE_SMOKE_PLAN_20260916.md) and [pre-execution independent review](mvp_execution_20260916/gffcompare/independent_review.md). Their approval was for the bounded diagnostic, not actual semantics; their raw-map lookup assumption proved incorrect.

Evidence base: `notes/mvp_execution_20260916/gffcompare/`.

- Captured bundle: `real_run/`.
- Logged remote origin: `/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_gffcompare_smoke_20260916/run`.
- `hashes.json`: **25,709 bytes**, SHA256 **`335bee5e27b65cfdd7eea1a32b3ef349ed9f31ee9853074245b2b64eb8f724af`**, identical to the remote digest in `result_status.txt`.
- **236/236 manifest-listed files** exist and match SHA256; no unlisted files apart from the manifest itself. Including the manifest: **237 files, 122,149 bytes**. This includes **34 map files**: ten raw tmap/refmap pairs and seven producer pairs, at their actual paths.
- Every producer output receipt was checked for its exact remote absolute path, complete output inventory, SHA256, and recorded byte size. Every version/raw/producer command was checked literally, not basename-normalized. The recovered JSON carries SHA256 and measured size for every bundle file, including the manifest, plus six historical source/log/staging artifacts.
- Frozen fixture identity, all emitted raw/producer exon-only GTFs (including half-open to 1-based coordinate conversion), all 16 result/error JSONs, seven producer tmap query inventories and accepted full chains, and raw/producer tmap/refmap row agreement were verified. No expected counts were changed.

Pinned original identities:

| Artifact | SHA256 |
|---|---|
| `smoke.py` | `3c022e41fa2759a33d8fee76452a55eca5ff2b226527a37fddecbba79c6098d1` |
| `smoke.sbatch` | `e6d18fcc732e183b7b3f40d3176e76889608f316453070ef4e12c5086ae0258b` |
| `mvp_produce_counts.py` | `bf7c8284804ee0a12c359a1fbe8f140bfc0f222abb522d6d7b7b4d3a691b417b` |
| `result_status.txt` | `96d8114b5c1c3b3d15a45a036a82d475b22048242c061d47738f30dc5b72a6da` |
| `submission_receipt.txt` | `7509d598c54cbdd2f1ddec015dc58742b88adf6728d2121e79a62895b5e99ccd` |

`stage.sha256` has exactly the three intended source/script entries and matches their bytes. Submission receipt records `submitted=6165521`; status capture is dated `2026-09-16T18:58:07-0700`. These are stored observations, not a new live scheduler query.

### Version and binary proof boundary

Recorded executable and resolved path are both:
`/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/gffcompare`.

The captured exact `--version` invocation returned0 and stderr `gffcompare v0.12.6`. Runtime records executable SHA256 **`a440e1533611a40a51cf2b5b35a2e15ec6b885660dd35f6b531e0d65601a963a`** and Python **3.11.14**, conda-forge/GCC14.3.0. The pinned harness records the binary hash before the case loop, rechecks binary/source/harness hashes afterward, then writes `hashes.json` (`smoke.py:96–121`). Its completed manifest and matching local source identities support that recorded provenance.

**Executable bytes are not in the copied bundle.** This recovery verifies the recorded binary identity and hash-bound runtime/logs, not an independent local rehash of the remote executable or adversarial attestation.

## Root cause; producer impact

Graft source spans: `smoke.py:124–175`, especially the raw glob at **line160** and assertion at **line167**; `mvp_produce_counts.py:313–346`.

The original raw command uses:

```text
gffcompare -r <origin>/<case>/reference.gtf \
  -o <origin>/<case>/raw/compare <origin>/<case>/predicted.gtf
```

The actual GFFCompare0.12.6 output locations are:

```text
<case>/compare.predicted.gtf.tmap
<case>/compare.predicted.gtf.refmap
<case>/raw/compare.{annotated.gtf,loci,stats,tracking}
```

Thus `list((d / 'raw').glob('*.tmap'))` returns zero maps even though the query-sibling maps exist. The raw `ordinary` exit is0 and its one row is `q -> r`, class`=`. Its producer receipt independently records TP1/FP0/FN0. The failure is the raw observation analyzer's directory assumption, not this producer result.

**Producer impact: no failure observed in any of the seven exercised tool-calling cases.** The producer writes both query and output prefix inside `<case>/producer/`; its actual `compare.predicted.gtf.tmap` is therefore in the directory searched by `directory.glob('*.tmap')`. Every exercised producer directory has exactly one captured tmap and matching refmap; receipts and counts agree. Mono/empty calls intentionally bypass the engine and duplicate guards intentionally reject before GTF output. This does not certify unexercised producer inputs or full `produce()`.

## All 16 observed outcomes and independent hand counts

Coordinates below are zero-based, half-open. Multi reference R has exons `[100,200),[300,400),[500,600)`, strand+, and chain **(200,300),(400,500)**. Mono M has exon `[1000,1200)`, CDS **[1020,1180)**, strand+.

`—` means no raw engine invocation, not a missing result. Count columns are literal values from captured producer receipts; required duplicate errors have no TP/FP/FN.

| Case | Observed TP/FP/FN | Raw exit; query→reference/class | Hand-count reason |
|---|---|---|---|
| ordinary | **1/0/0** | 0; q→r `=` | Same strand and both junctions. |
| terminal_short | **1/0/0** | 0; q→r `=` | First start120, last end580; both junctions unchanged. |
| terminal_extend | **1/0/0** | 0; q→r `=` | First start80, last end620; both junctions unchanged. |
| junction_change | **0/1/1** | 0; q→r `j` | Middle start310 changes chain to (200,310),(400,500). |
| opposite_strand | **0/1/1** | 0; q→r `s` | Same coordinates, opposite strand. |
| unmatched | **0/1/1** | 0; q→`-` `u` | Disjoint chain (2100,2200). |
| mono_exact | **1/0/0** | —; producer command null | Same nonempty CDS and strand. |
| mono_utr_variant | **1/0/0** | —; producer command null | Exon[990,1210), same CDS; UTR boundaries ignored. |
| mono_cds_change | **0/1/1** | —; producer command null | CDS start1021 differs. |
| mono_opposite | **0/1/1** | —; producer command null | Same CDS, opposite strand. |
| mono_no_cds | **0/1/1** | —; producer command null | Neither has a nonempty CDS key; no match counted. |
| empty_query | **0/0/1** | 0; zero rows | No predictions and one reference; producer bypass. |
| multiisoform_unique | **2/0/0** | 0; q→r `=`, q_alt→r_alt `=` | Two distinct chains, both present once on each side. |
| duplicate_query_terminal | **required predicted-duplicate ValueError** | 0; q→r `=`, q_terminal→r `=` | Query multiplicity ambiguous: same chain twice. |
| duplicate_reference_terminal | **required reference-duplicate ValueError** | 0; q→r `=` | Reference multiplicity ambiguous: same chain twice. |
| duplicate_mono_cds | **required predicted-duplicate ValueError** | — | Same CDS/strand twice despite different UTR exons. |

All three errors have exact suffix `duplicate isoform structures: author multiplicity policy needed`; all three producer directories are empty. They are expected fail-closed outcomes, not silently assigned counts.

### Duplicate and empty behavior, without policy inference

- **Duplicate query:** raw tmap retains both `q` (length300) and `q_terminal` (length260), each class`=` to `r`; `major_iso_id=q` for both. Raw refmap contains `g|q,g|q_terminal`. Queries were not collapsed from this tmap. Do not count these as two independent reference TPs.
- **Duplicate reference:** input reference GTF contains both `r` and `r_terminal`; raw tmap chooses `r` for `q`, and refmap lists only `r -> g|q`. This is a recorded choice, not a general reference tie-break guarantee or permission to choose an isoform in production. Original inputs remain intact.
- **Empty query:** raw subprocess returned0; stderr reports one reference and zero query transfrags. Both map files exist and contain headers only (tmap103B; refmap39B). Raw annotated/tracking/loci/stats outputs are retained too. Producer command is null and its predicted GTF is empty. No denominator or empty-input policy changed.
- ±20bp terminal observations apply only to the tested changes, not every possible terminal offset. Exact mono CDS behavior is producer logic, not a claim that GFFCompare compares CDS.

## Local recovery implementation and verification

New files only:

1. `notes/mvp_execution_20260916/gffcompare/local_recovery/recover.py` — separate read-only analyzer. Historical `smoke.py` retains its original hash. Loads only the pinned `tx`/`cases` AST definitions, never the runner or producer functions. No engine subprocess capability is used. Maps are read by exact captured query-sibling paths, with no fallback search. Full inventory hashing still rejects missing, extra, moved, or mutated maps.
2. `.../local_recovery/test_recover.py` — ten stdlib unittest regressions, using actual bundle copies.
3. `.../local_recovery/analysis.json` — complete recovered observations, commands, tmap/refmap rows, runtime, file hash/size inventories, original scheduler failure and unresolved extern status. SHA256 **`273865876c1861f474c3b95e28ebbc28c61c2c6be14993ae8b05c67a604fad4a`**.
4. `.../local_recovery/test_results.txt` — captured local verification output: **10 tests PASS**, including parameterized negatives.
5. This report.

The original runtime has no explicit run-root field. The new helper therefore requires `--origin` and binds it to the **pinned `result_status.txt` line associating the remote absolute `run/hashes.json` path with the pinned manifest SHA**, then checks all logged commands and receipt absolute paths against that origin. Local copy location is independent; arbitrary prefix stripping, relative origins, a different old path, and substituting the local path as origin are rejected. The inventory-bound `runtime.json` and pinned historical files provide the remaining identity chain. No historical path is rewritten.

Reproduce locally from repository root (no tool invocation):

```sh
python3 -I -B notes/mvp_execution_20260916/gffcompare/local_recovery/recover.py \
  --root notes/mvp_execution_20260916/gffcompare/real_run \
  --origin /data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_gffcompare_smoke_20260916/run
python3 -I -B notes/mvp_execution_20260916/gffcompare/local_recovery/test_recover.py
```

Tests cover all16 expected results, classes`j/s/u`, duplicate rows, empty maps, read-only before/after inventories, old raw-glob miss versus new exact path, missing raw/producer tmap/refmap, extra maps, same-size mutated raw/producer maps, relocated maps, rewritten manifest, changed result, symlink map, and wrong/relative/noncanonical origins. Initial positive regression failed before the new helper existed; it passed after implementation. No fake engine or synthetic replacement results were used. Local negative mutations are confined to temporary copies and never become accepted report evidence.

Advisor was attempted for a recovery risk check but returned a client/model-version error; no advisor approval is claimed. No independent reviewer was dispatched. The controller must independently inspect the new analysis before reporting tool-validation acceptance.

## Remaining claim boundary

This recovers actual **function-level synthetic GFFCompare0.12.6 diagnostic evidence**. It does not validate frozen biological GFF compatibility, full `produce()`, parser/decoder integration, independent-read provenance, MVP training/GPU readiness, or biological performance. Author duplicate-multiplicity policy remains **UNRESOLVED_FAIL_CLOSED**; `biological_compatibility=NOT_TESTED` and **`decision=null`** remain unchanged. There is no authorization for a new job or rerun from this report.
