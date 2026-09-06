# #23 frozen addition export — 2026-09-05

Implemented, uncommitted: script 48 `--dump-structures NEW_DIRECTORY` exports both species and exits without rewriting its existing aggregate output. `revision/scripts/addition_dump.py` reuses script 36 predicates. JSONL records retain stable structure IDs, all collapsed prediction transcript IDs, locus, supplied primary ID/CDS, prediction CDS, inclusive intronic coordinates in genomic order, strand/contig, exact/chain reference matches, novelty or explicit ambiguity, and structural predicates. These are prediction inventory records, not C0 observations or direct script-59 input rows. An adapter still needs the scorer's assembly/locus-bound metadata and must reject unresolved novelty, not silently omit those rows.

Output: `revision/results/addition_dump_20260905_v2/`. `MANIFEST.json` records frozen-input MD5 verification, FASTA-index hashes, implementation hashes, JSONL hashes, duplicate audit and reconciled aggregates. Output directory creation is exclusive; manifest written last. No evidence, frozen corpus, protocol, sbatch or source annotations modified. No commits or issue updates sent.

| Species | Additions | Filter pass | Reference-alt | Junction-novel | Combination-novel | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| Zmays | 3,363 | 491 | 484 | 2,092 | 677 | 110 |
| Athaliana | 1,103 | 266 | 347 | 499 | 204 | 53 |

Every row contributes to these counts, including unresolved rows. Exact-CDS and alternative-chain match totals reconcile against script 48's aggregate function; Arabidopsis's 266 retained structures reproduce the frozen script-36 count. Counts describe predictions only: no real evidence alignment has been scored.

## Genuine ambiguity — stop before assigning remaining novelty classes

Frozen protocol §2 defines `junction-novel` as an intron absent from every reference transcript, `combination-novel` as a chain matching none, and `reference-alt` as exact CDS or intron-chain match to a reference alternative. An addition with changed outer CDS boundaries can match **only the supplied primary's chain**. It satisfies none of these definitions. Empty-chain reference matching also needs care: frozen script-28/48 alternative-chain matching explicitly requires at least one intron, so an unmatched monoexonic structure is not credited an alternative-chain match here.

There are 110 maize and 53 Arabidopsis unresolved rows; **11 maize and 8 Arabidopsis pass F**. Example maize locus `Zm00001d002418.RefGen_V4` changes the first CDS boundary while retaining the primary chain. The JSONL includes exact coordinates and reason on every unresolved row. No fourth biological class was invented, no primary-chain match credited as a reference alternative, and no rows dropped from the addition denominator. These counts are diagnostics, not a request to choose a favorable category after evidence scoring. Author clarification is required before a complete classified scorer input can be frozen; #23 remains partially blocked, not complete.

## Validation and execution

- 67 targeted tests passed on GPU host in isolated `/tmp/transgenic-issue23-YNe5JP`: export selection/strand/index/no-overwrite tests plus existing structural-filter and evidence-scorer tests.
- Six frozen script-48 selection/scoring functions are AST-identical to HEAD; script 28 and 36 unchanged. Arabidopsis reader was extracted without changing parsing statements.
- All seven protocol §1 prediction/reference/primary/genome MD5s matched.
- Initial v1 attempt was killed while script 36 loaded the full maize FASTA. Its directory is preserved without a completion manifest. Export now uses indexed slices through the unchanged predicates; v2 completed successfully. Neither attempt accessed alignment results.
- Reproduction command: `python3 revision/scripts/48_score_zmays_additions.py --dump-structures NEW_DIRECTORY`.

Next unblocked priority: #9 retroactive download integrity audit, then accurately record remaining #25 provenance gaps (including the fact that any newly completed audit occurred after alignment). This is an explicit dependency of #21 C0 ingestion. Do not backdate the audit or treat DONE markers as integrity evidence. B5 #58/#56/#55 recovery and the corpus remain blocked exactly as recorded in `author_decisions_20260905.md`; #61 also retains its separate paper-claim scoring-universe decision.

Final verification: real-input `--verify-athaliana` reproduced **200/1,103**; all maize ALL aggregate fields except the descriptive stratum label match the dump manifest. `git diff --check` passed. At final status, unrelated concurrent modifications appeared in `src/transgenic/datasets/build_b5.py` and `src/transgenic/model/tokenization_transgenic.py`; this task did not make or revert those edits. Output files are under the repository's ignored results directory and must be retained separately from a future code commit.
