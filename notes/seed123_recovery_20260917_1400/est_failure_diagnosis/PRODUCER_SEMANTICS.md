# Producer semantics follow-up

## Finding

Retained provenance identifies the **same DIAMOND executable/build, launcher, query and raw13 format for both production searches**. No retained evidence supports a source-name-dependent length rule or older-output conversion. The universal MID validator is demonstrably overgeneralized; the upstream reason for differing lengths remains unresolved.

The reproduced observations stand: 400 OrthoDB rows match MIDF, 62 SwissProt rows match MID; only shift-containing rows discriminate. OrthoDB has 41 such rows (31 minus-containing, 10 plus), versus five SwissProt minus-shift rows. Thus shift sign alone cannot explain the difference. Twenty-nine OrthoDB rows reach the length check after exact replay.

## Producer and byte binding

`notes/seed123_execution_20260917/est_source_metadata.txt` contains full historical commands/provenance. Its SHA256 is `f814b5665f60421826eec838bd259f3728ec793d58e1c6ea4a3aa4870769b998`. I recomputed the four embedded command/provenance hashes: all equal `est_result/raw/worker/sample/report.json` source receipts and `est_release_final/manifest.json` pins. That historical job reports whole-file raw/protein verification against provenance. This follow-up did not rehash original data.

Both searches record:

- Executable `/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond`.
- Version `2.1.24`, help-banner build `2.1.24.178`; executable SHA256 `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`.
- Launcher SHA256 `5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291`; taxonomy helper `32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302`.
- `makedb --in <filtered.fa> --db <source.dmnd> --threads 16 --no-parse-seqids`.
- `blastx --query <same est.univec.fa.gz> --db <source.dmnd> --out <source.raw13.tsv> --threads 16 --tmpdir <temporary-directory> --frameshift 15 --evalue 1e-5 --min-orf 30 --max-target-seqs 25 --max-hsps 1 --no-parse-seqids`.
- `--outfmt 6 qseqid qlen sseqid slen qstart qend sstart send length evalue bitscore qframe btop`.

Commands differ only in database/output/filter paths and temporary directories. This is historical recorded identity, not a fresh binary execution or an independent build-source attestation.

## Conversion and parser trace

The hash-matching launcher is retained at `notes/est_execution_20260916_1530/reconciliation_refreshed/code/revision/scripts/evidence/est_blastx.py`. Lines 199-227 and 275-315 write DIAMOND output directly, then read, validate and hash it; no length rewrite or raw17-to-raw13 conversion occurs. Lines 117-143 filter protein records by taxonomy, not output length.

Recovery `est_release_final/code/evidence/est_real_sample.py:343-365,450-466` preserves raw fields and passes them unchanged. `est_hsp_contract.py:76-126` appends reconstructed fields; lines 41-43 impose MID equality, while line 69 uses the reported column for >=30.

The approved replay script `/tmp/est-real-source-binding-20260917/tasknotes/hsp/replay_actual.py` takes the first 13 fields of **raw17 fixture output**, without rewriting length. Its 144-row success is not a separate raw13-mode producer test. The expanded fixture used the same binary, two threads, tiny synthetic inputs and four extra output fields. Earlier `fixture_result/btop_pin_report.json` also reports MID for both shift signs under both source labels. These fixtures refute a universal source-label rule, but do not establish the production execution path.

## Database and masking limits

Historical databases were freshly built from separately filtered sources: OrthoDB `odb12_Viridiplantae.filtered.fa.gz`, 12,092,922 retained proteins; SwissProt `swissprot_viridiplantae_reviewed.fa.gz`, 42,103. Provenance records separate source/filtered/database hashes. Size/content could influence automatic engine paths; causation is unproven.

Neither command explicitly sets masking or composition options. Absence does not mean masking disabled. Metadata stderr contains only environment-loading messages. Preflight lists logs but does not retain their contents. X and numeric-run masking uncertainty remains.

## Smallest missing evidence

Request, only with separate authorization:

1. Under `/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx/sitalica_pilot_20260914/`, both sources' `blastx.log` and `makedb.log`: four existing files, 16,736 bytes total per preflight. These may identify actual automatic algorithms, database build and masking settings.
2. Pinned DIAMOND 2.1.24.178 source excerpts covering outfmt `length`/BTOP serialization, every reachable frameshift traceback length accumulator, algorithm dispatch, and masking defaults/traceback residues. Require commit/build binding, not generic current documentation. If necessary, the small `evidencemodeler/conda-meta/diamond-2.1.24-*.json` package record identifies the build/recipe.

Completed: local provenance/conversion trace. Blocked: upstream implementation attribution. Preserve **reported-length >=30**, withheld failures and all acceptance gates. No new jobs, downloads, original-data rescans, source fixes or acceptance promotion occurred.
