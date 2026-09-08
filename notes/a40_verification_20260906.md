# A40 verification — 2026-09-06

**PASS: completion interface resolved, +924 rows reconciled, full verification completed. Not frozen or adopted.** No unexplained row additions, removals, or shared-row semantic differences remain in the tested attribution. All nine current-source replays agree with their databases; the merged validator has zero violations; every additional ledger check below passed.

New database on `gpu`: `/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1/merged/b5_full_a40_v1.db`. The output metadata is `b5_full_a40_v1.merge-receipt.json`, explicitly a merge receipt, not an adopted freeze. No `.DONE` markers were created. No commit, protocol edit, freeze/adoption, historical artifact edit, or alignment work occurred.

## Completion evidence — option (a)

The merger now accepts `--build-status <root>/provenance/build-status.json`. This is cleaner than deriving markers: the A40 driver already records completion there, and a second mutable completion object would add another freshness problem. The original build snapshot remains unchanged. The revised merger was executed from `<root>/audit/verification-source/`; only the merger and its tests differ from the copied snapshot. The actual argv, cwd, timestamps, exit 0, and every executed Python source hash are in [merge-status-v2.json](a40_audit/merge-status-v2.json).

The gate requires aggregate SUCCESS/exit 0, an exact species set with no missing/duplicate/extra records, integer species exit 0 and `completion_ok is true`, the correct DB path, the recorded rejection hash, and a separate manifest exactly matching the database manifest. It then validates each current DB afresh, requires `ok=true`, zero violations and that species only, and hashes the DB before/after validation and again before merging. Evidence hashes and complete per-species validation results are embedded in the [merge receipt](a40_audit/b5_full_a40_v1.merge-receipt.json). Invalid evidence cannot fall back to the legacy marker path. All nine passed. The legacy path remains for historical callers; the A40 invocation explicitly uses the status gate.

Completion tests: **20/20 passed**, including missing/duplicate species, failed/incomplete status, wrong database, changed rejection evidence, changed manifest evidence, and failed fresh validation. The original attempt exited 1 for missing `Athaliana.DONE`; that interface failure is resolved by the new gate, not by inventing markers. The confirmed original build duration remains 04:59:28–06:50:21 UTC (1h50m53s).

## The +924 rows — causal reconciliation

The allocation below uses an explicit order: **historical corpus → #55 with the old masking population → #55 + #56**. The intermediate is an in-memory counterfactual, not another built or adopted database. Its sole change from the snapshot replay is to restore unconditional cap-exclusion of no-CDS loci; seeds, input files, flags and all other rules are identical. Every intermediate row shared with the historical database has identical audited row semantics; every added intermediate row is one of the 97 recorded ordering-reject coordinates (or its RC). There are zero intermediate removals and zero other additions or shared-row changes. This is evidence for causation, not an assignment of an unexplained remainder to #56.

| Species | Historical | New | Δ rows | #55 forward | #55 RC | #56 net | Other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Athaliana | 10,454 | 10,470 | +16 | 8 | 8 | +0 | 0 |
| Bdistachyon | 21,967 | 21,967 | +0 | 0 | 0 | +0 | 0 |
| Gmax | 53,247 | 53,247 | +0 | 0 | 0 | +0 | 0 |
| Osativa | 28,057 | 28,057 | +0 | 0 | 0 | +0 | 0 |
| Ppatens | 39,009 | 39,029 | +20 | 10 | 10 | +0 | 0 |
| Ptrichocarpa | 30,076 | 30,078 | +2 | 1 | 1 | +0 | 0 |
| Sbicolor | 31,223 | 31,223 | +0 | 0 | 0 | +0 | 0 |
| Sitalica | 26,075 | 26,075 | +0 | 0 | 0 | +0 | 0 |
| Vvinifera | 32,116 | 33,002 | +886 | 78 | 44 | +764 | 0 |
| Total | 272,224 | 273,148 | +924 | 97 | 63 | +764 | 0 |

**#55 contributes +160, not +194:** all 97 ordering-only forward windows survive; only 63 qualify for isoform-only RC. Vvinifera contributes 78 forward +44 RC; the other three affected species contribute 19 forward +19 RC. **#56 contributes +829 forward −65 RC = +764**. Together: +926 forward −2 RC = +924.

### Direct recovery in the final corpus is a different count

| Species | Historical rejects | Final forward at old coordinate | Final RC at old coordinate | Final fate |
| --- | --- | --- | --- | --- |
| Athaliana | 8 | 8 | 8 | All retained |
| Ppatens | 10 | 10 | 10 | All retained |
| Ptrichocarpa | 1 | 1 | 1 | All retained |
| Vvinifera | 78 | 0 | 0 | All 78 coordinates absent from changed grid |

Thus **19 old rejected coordinates +19 RC rows are directly recovered in the final corpus**. All 78 Vvinifera coordinates disappear after #56 changes empty-sampling RNG consumption and subsequent offsets. The +122 ordering-only Vvinifera rows are among the rows removed by the population counterfactual; they are not claimed as physically retained final rows. Equivalently, using direct final recovery, +924 = +38 direct ordering recovery +886 Vvinifera population/order interaction. The table above exposes that interaction instead of double-counting it. All 97 individual fates and old rejection messages are in [reconciliation.json](a40_audit/reconciliation.json). The historical 13 ties +84 decreasing-span failures are all covered; new canonical-order rejections are zero.

### Vvinifera: what #56 actually changes

Old Vvinifera 32,116 → ordering-only 32,238 → final 33,002. The last step adds 32,627 row keys, removes 31,863, and changes the contents of 30 shared rows, net +764. The large coordinate turnover is measured RNG/grid movement, not 32,627 newly discovered biological loci. The complete row-level transition file is [Vvinifera.population-row-changes.tsv](a40_audit/Vvinifera.population-row-changes.tsv).

| #56 transition | Forward added | RC added | Forward removed | RC removed | Net |
| --- | --- | --- | --- | --- | --- |
| Different tile coordinates (grid absent in other replay) | 24919 | 7706 | 24091 | 7770 | +764 |
| Same coordinate: previously empty-sampled, now retained | 2 | 0 | 0 | 0 | +2 |
| Same coordinate: now rejected for mask fraction | 0 | 0 | 1 | 1 | −2 |
| Shared rows with changed sequence/label/counters | 0 | 0 | 0 | 0 | 0 (24 forward +6 RC changed) |
| Total | 24921 | 7706 | 24092 | 7771 | +764 |

| Forward-window accounting | #55 only | Final #55 + #56 | Δ |
| --- | --- | --- | --- |
| candidate_windows | 28,555 | 28,552 | -3 |
| empty_sampled_out | 3,801 | 2,923 | -878 |
| mask_fraction_rejected | 330 | 376 | +46 |
| window_cap_rejected | 0 | 0 | +0 |
| forward_retained | 24,424 | 25,253 | +829 |
| rc_rows | 7,814 | 7,749 | -65 |
| component_added | 2,281 | 3,978 | +1,697 |
| leak | 31,096 | 34,686 | +3,590 |
| hard | 1,985 | 1,969 | -16 |
| decoy | 2,542 | 2,602 | +60 |
| dup_collapsed | 0 | 0 | +0 |

Conservation: candidate windows −3, empty-sampling drops −878, mask-fraction drops +46, cap drops unchanged → +829 retained forward windows; RC −65 → +764 rows. Component propagation increases by 1,697 forward occurrences. These mechanisms interact: the report does not invent an additive “component-only row count” separate from RNG, flank masking, and sampling when they jointly determine a tile. The disjoint per-row fate transitions above and the one-change counterfactual establish the row cause.

All five species with no ordering or population change reproduce exactly. Athaliana, Ppatens and Ptrichocarpa have only their measured ordering additions. **Unexplained additions = 0; unexplained removals = 0; unexplained shared-row semantic changes = 0.** Each current replay matches row fingerprints, `window_genes`, `gene_key_map`, tile blocks and rejection records. Remaining provenance/static row fields and unchanged historical input references were checked separately in [supplement.json](a40_audit/supplement.json).

## No-CDS population and stored masks

**PASS: 6,205/6,205 original Vvinifera no-CDS loci**, assigned 4,600 train /634 valid /971 test; all 6,205 are eligible in the new population, with zero coding memberships and zero decoy memberships. None is hard-flagged. Denominator 41,766 →47,971 (+6,205); leak/hard-union numerator 15,346 →16,951 (+1,605 =634+971); realised rate `min(0.05, numerator/denominator/3)` is **0.05 before and after**, verified rather than presumed. Per-species denominators/numerators/rates are retained in reconciliation JSON.

| No-CDS occurrence check | Forward | RC |
| --- | --- | --- |
| Complete locus in retained tile | 17,776 | 6,445 |
| Leak seed | 3,632 | 1,178 |
| Hard (including overlaps with leak) | 0 | 0 |
| Hard only | 0 | 0 |
| Leak ∪ hard seed | 3,632 | 1,178 |
| Masked after component closure | 4,850 | 1,615 |

The independent audit examines all **25,253 Vvinifera forward windows**, reconstructs masking seeds and fixed-point interval-graph closure independently of the builder’s component sweep, and checks actual stored N coverage over every masked complete locus. No labelled coding member remains in a masked component. All 4,850 forward +1,615 RC masked no-CDS occurrences have coverage; the RC conclusion also uses exhaustive actual RC-sequence equality. Of these, closure adds 1,218 forward +437 RC occurrences beyond the leak/hard seed union. Hard and hard-only counts are both zero; union is not double-counted. See [no-cds-independent.json](a40_audit/no-cds-independent.json).

The existing complete-locus guarantee is preserved: partial loci crossing tile edges are outside its scope. Same-split non-hard no-CDS loci are not unconditional mask seeds; some are masked through closure. No stronger protection is claimed.

## Rows, empty targets and token ledger

Full species × split × tier × orientation counts are provided for both corpora: [historical TSV](a40_audit/old-rows-species-split-tier-orientation.tsv), [new TSV](a40_audit/new-rows-species-split-tier-orientation.tsv). These enumerate the full joint distribution, not only marginal sums.

| Split | Old rows | New rows | Δ rows | Old empty | New empty | Δ empty |
| --- | --- | --- | --- | --- | --- | --- |
| train | 198,171 | 198,828 | +657 | 22311 | 23005 | +694 |
| valid | 25,910 | 25,977 | +67 | 1809 | 1869 | +60 |
| test | 48,143 | 48,343 | +200 | 1039 | 1183 | +144 |

Rows: **272,224 →273,148 (+924)**. Forward: **188,145 →189,071 (+926)**; RC: **84,079 →84,077 (−2)**. Empty targets: **25,159 →26,057 (+898)**, 9.2420% →9.5395%. Every one of the **26,057 actual empty rows** was passed through the snapshot dataset’s decoder path and produced exactly `<s> <empty> </s>` (three tokens). The cached read adapter supplied the actual stored row; only the unrelated encoder tokenizer was stubbed to avoid model downloads.

| Species | Old token max | New token max | Δ max | Old stored tokens | New actual/stored tokens | Δ tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Athaliana | 5920 | 5920 | +0 | 9,159,879 | 9,184,600 | +24,721 |
| Bdistachyon | 4822 | 4822 | +0 | 12,286,244 | 12,286,244 | +0 |
| Gmax | 4230 | 4230 | +0 | 20,826,104 | 20,826,104 | +0 |
| Osativa | 3702 | 3702 | +0 | 12,544,998 | 12,544,998 | +0 |
| Ppatens | 4790 | 4790 | +0 | 19,173,260 | 19,188,541 | +15,281 |
| Ptrichocarpa | 3540 | 3540 | +0 | 14,158,118 | 14,158,674 | +556 |
| Sbicolor | 3639 | 3639 | +0 | 11,061,323 | 11,061,323 | +0 |
| Sitalica | 4251 | 4251 | +0 | 10,675,638 | 10,675,638 | +0 |
| Vvinifera | 2574 | 2785 | +211 | 9,696,905 | 9,667,570 | -29,335 |

All **273,148 actual tokenizer lengths equal stored counts**, with no unknown tokens, truncation, or cap violations. Maximum remains **5,920** (cap 8,192); Vvinifera’s species maximum moves 2,574 →2,785 (+211). Total stored/actual tokens: **119,582,469 →119,593,692 (+11,223)**. #58’s extra actual token on the same new corpus is **26,057**, not the old 25,159; stored counts already used three tokens. Using the old two-token empty convention, the historical actual total would be 119,557,310, so the combined old-actual →new-actual movement is +36,382. This last comparison is derived from the known old convention, not a claim that the old tokenizer was re-executed here.

## Mask counters, rejection counters and N totals

| Counter (RC included) | Old | New | Δ |
| --- | --- | --- | --- |
| component_masked | 29,367 | 31,711 | +2,344 |
| decoy_masked | 31,808 | 31,937 | +129 |
| dup_collapsed | 743 | 743 | +0 |
| edge_partial | 184,894 | 186,407 | +1,513 |
| hard_masked | 27,258 | 27,288 | +30 |
| leak_masked | 410,386 | 415,465 | +5,079 |

| Rejection class | Old | New | Δ |
| --- | --- | --- | --- |
| canonical_order | 97 | 0 | -97 |
| mask_fraction_a33 | 1,543 | 1,589 | +46 |
| no_cds | 6,205 | 6,205 | +0 |
| token_cap | 3 | 3 | +0 |
| transcript_cap | 333 | 333 | +0 |

No additional gene/window/feature-cap rejection class appears (gene-cap count 0); transcript-cap 333 and token-cap 3 are unchanged. No-CDS rejection diagnostics remain 6,205 coding exclusions, not masking exclusions. Empty-sampling and duplicate-collapse accounting are separately recorded by replay; Vvinifera accounts for the sampling change. Per-species mask sums, tile counts carrying each counter, and rejection breakdowns are in [population-v2.json](a40_audit/population-v2.json); forward-only replay counters avoid accidentally double-counting RC in causal comparisons.

Stored uppercase N bases, RC included: **1,688,716,560 →1,704,655,473 (+15,938,913)**; stored sequence bases **15,949,750,272 →15,989,194,752 (+39,444,480)**. N fraction **10.58773041% →10.66129658%**. Per-species values are in ledger JSON. These count natural Ns plus introduced masks; they are not a count of newly masked biological bases.

## Assignments, labelled populations, diagnostics and structural acceptance

**All three must-not-move quantities pass:** 334,642 nominal assignments; train 213,702 /valid 29,153 /test 91,787; strict 3,430. All source tables and the merged table retain gene-split content SHA256 `61766859a1d712b49f3fdbeb457067d00e604d53a0298af614c28bff2ea1deee`. All original gene-key maps match historical sources. Source FASTA/GFF references, hashes, split hash, QC families, RC mode, ordering version, window policy and DuckDB version are unchanged.

| Assigned split | Old ever-labelled | New ever-labelled | Δ | New coverage |
| --- | --- | --- | --- | --- |
| train | 196,592 | 196,521 | -71 | 91.96% |
| valid | 7,316 | 7,328 | +12 | 25.136% |
| test | 15,005 | 15,020 | +15 | 16.364% |

| Gene assignment | Train tiles | Valid tiles | Test tiles |
| --- | --- | --- | --- |
| train | 145,783 | 20,068 | 34,655 |
| valid | 0 | 2,777 | 4,590 |
| test | 0 | 0 | 15,020 |

Strict ever-labelled remains **431/3,430**, Δ0. Test-tile target universe **54,188 →54,265 (+77)** =34,655 train-assigned (+49) +4,590 valid-assigned (+13) +15,020 test-assigned (+15). These unions across tile splits must not be summed to infer ever-labelled totals because a gene can occur in several tile splits. Original reference test-gene length and transcript-count distributions were regenerated per species; missing reference gene keys =0. The detailed labelled/unlabelled distributions and hashed reference paths are in population JSON. No metric denominator or evaluation universe was redefined.

**#57 PASS:** all-original-gene denominator 334,642 and maximum length 196,414 nt are unchanged. Length-bound exceedances remain 3,418 /653 /86 for tiers 30,720 /61,440 /129,024. The historical no-edge-credit margin upper bounds remain 2,034 /434 /44. Exact FASTA-edge-aware counters were recomputed for all nine species from original coordinates and current FASTA and match every stored diagnostic; for these inputs their aggregate values also happen to be 2,034 /434 /44. Missing contigs =0. These are separate calculations, not an assumption that an upper bound equals an exact count. See [margins-recomputed.json](a40_audit/margins-recomputed.json).

| Structural invariant | Result |
| --- | --- |
| Authoritative orthogroup split integrity; strict assignments | PASS, nine fresh source validators and merged validator; zero violations |
| Strict holdouts and forbidden train/valid label leakage | PASS, orientation-aware membership joins; no gene labelled below its assigned split |
| Excluded maize species/models | PASS, none |
| Window/sequence lengths and caps | PASS; observed tiers 30,720 /61,440 /129,024; tokenizer cap 8,192 |
| Required labels, no orphan membership, row-key uniqueness | PASS; no NULL/invalid labels, no orphan window_genes, no duplicate keys |
| RC membership, sequence and split correspondence | PASS exhaustively; every RC has its forward row and identical gene membership |
| Canonical labels and RC involution | PASS for all 273,148 labels, not only paired RC labels |
| Merged rn contiguous and deterministic | PASS; 1..273,148, 273,148 distinct values; each equals species offset + source rank ordered by source rn |
| Source replay and historical row-delta conservation | PASS 9/9; zero unexplained row differences |

Full results: [ledger.json](a40_audit/ledger.json), [reconciliation.json](a40_audit/reconciliation.json), and the population report.

## Artifact identity and protected objects

| New artifact field | Value |
| --- | --- |
| Rows | 273,148 |
| Bytes | 23,534,252,032 |
| File MD5 | dda7e78c880c993435654b57dcd1c498 |
| geneList content SHA256 | 063c9f2f78549dd3d36c692dbc1cf2f6ff1d44816718a51865610f97f5e9e0d0 |
| gene_split content SHA256 | 61766859a1d712b49f3fdbeb457067d00e604d53a0298af614c28bff2ea1deee |

The old merged DB is unchanged: **23,491,260,416 bytes**, `mtime_ns=1788422329816219148`, freshly recomputed MD5 **ebf4fb511073c546d05e3fa0fc0e4258**. All nine old source DB MD5s match the old freeze. All original executed source hashes were rechecked; all 18 QC SHA256s still match the previous verified invocation (whose MD5s matched the historical freeze). See [protected-artifacts-v2.json](a40_audit/protected-artifacts-v2.json). The original A40 source snapshot and historical source objects were only read. The protocol, `data/freeze/b5_full_v1.freeze.json`, and alignment results were not edited.

## Limits and final disposition

No failing acceptance check was bypassed. No unexplained row delta remains. The #55/#56 additive allocation is explicitly order-dependent; the final-coordinate recovery count is reported separately. Component closure, RNG stream changes and masking can interact, so no unsupported independent component-only row increment is claimed. The N-coverage guarantee applies to complete eligible loci under the existing recipe, not edge-partial loci. File-layout hashes and recipe/manifest metadata are expected to move; the old artifact is preserved.

**Verified replacement artifact available for review; no commit, protocol amendment, freeze or adoption performed.**
