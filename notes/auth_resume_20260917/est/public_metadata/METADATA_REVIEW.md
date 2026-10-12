# Public metadata recovery: selected Sitalica members

## Completed scope

Recovered only the 16 versioned nuccore records already represented in retained job 6165875 diagnostics. One HTTPS EFetch batch, no retries or redirects, completed on 2026-09-17 at 23:18:40 UTC. Response: 58,548 bytes; SHA256 `85af56da812d225e1c46e7c969b1de67ce76a01b1da51c5409f3d56a9b4249d0`.

All 16 requested accession versions were returned exactly once, with no extras. Every complete ORIGIN sequence matches its retained original sequence and SHA256, after removing GenBank position/format whitespace and making nucleotide letter case uppercase. No trimming, reverse-complementing, alignment, or nucleotide substitutions were permitted. This is nucleotide identity, not byte identity between GenBank formatting and retained uppercase sequence strings. LOCUS accession/length/date, ACCESSION, VERSION, retained header ID/hash, original length, and complete response parsing were checked. Metadata extraction starts only after the entire 16-record set passes.

Retained diagnostics SHA256: `8d8c8be00102a6ba7e7b5cdd875dbeca5a99b18290c024f3690cd849e670fb1d`. Whole parent FASTA hashes are retained provenance pointers, not independently rehashed whole files. No reference/data rescan was performed.

## Per-member observed fields

The table below is a readable summary. Library words appear as recorded; the exact multiline values, indentation, source preparation notes, comments, and all other source qualifiers are preserved in `metadata.json` and `metadata.tsv`. No alias identity is inferred from a shared BioSample.

| Accession.version | LOCUS date | nt | BioSample | Library text identifier | Clone qualifier | Tissue qualifier | Strain/cultivar qualifier |
|---|---|---:|---|---|---|---|---|
| JK596275.1 | 06-OCT-2011 | 648 | SAMN00736318 | CFGZ | CFGZ9178 | floral spike | strain Yugu1 |
| JK578847.1 | 06-OCT-2011 | 691 | SAMN00736317 | CFBX | CFBX9727 | whole seedlings | strain Yugu1 |
| JK570678.1 | 06-OCT-2011 | 787 | SAMN00736317 | CFBX | CFBX14764 | whole seedlings | strain Yugu1 |
| JK559843.1 | 06-OCT-2011 | 767 | SAMN00736317 | CFBX | CFBX8241 | whole seedlings | strain Yugu1 |
| JK552962.1 | 06-OCT-2011 | 328 | SAMN00736317 | CFBX | CFBX4385 | whole seedlings | strain Yugu1 |
| GW787538.1 | 12-JAN-2011 | 383 | SAMN00168307 | Salinity stress responsive genes from cv. PRASAD (Setaria italica L.) | not reported | not reported | cultivar PRASAD |
| JK608601.1 | 06-OCT-2011 | 499 | SAMN00736318 | CFGZ | CFGZ16439 | floral spike | strain Yugu1 |
| JK608600.1 | 06-OCT-2011 | 499 | SAMN00736318 | CFGZ | CFGZ16439 | floral spike | strain Yugu1 |
| JK608599.1 | 06-OCT-2011 | 580 | SAMN00736318 | CFGZ | CFGZ16438 | floral spike | strain Yugu1 |
| JK608598.1 | 06-OCT-2011 | 580 | SAMN00736318 | CFGZ | CFGZ16438 | floral spike | strain Yugu1 |
| JK608597.1 | 06-OCT-2011 | 435 | SAMN00736318 | CFGZ | CFGZ16437 | floral spike | strain Yugu1 |
| JK608596.1 | 06-OCT-2011 | 435 | SAMN00736318 | CFGZ | CFGZ16437 | floral spike | strain Yugu1 |
| JK608595.1 | 06-OCT-2011 | 716 | SAMN00736318 | CFGZ | CFGZ16436 | floral spike | strain Yugu1 |
| JK608594.1 | 06-OCT-2011 | 702 | SAMN00736318 | CFGZ | CFGZ16436 | floral spike | strain Yugu1 |
| JK608593.1 | 06-OCT-2011 | 716 | SAMN00736318 | CFGZ | CFGZ16435 | floral spike | strain Yugu1 |
| JK608592.1 | 06-OCT-2011 | 699 | SAMN00736318 | CFGZ | CFGZ16435 | floral spike | strain Yugu1 |

LOCUS dates are public record dates, not asserted collection dates, original submission dates, or metadata-history versions. Sequence `.1` is the observed VERSION identity; it does not independently version every annotation. Full reference/journal text is preserved in `header_raw` where present.

## Preparation evidence and limits

- Each selected CFGZ/CFBX record explicitly reports oligo(dT)-primed cDNA, SMART preparation, PCR second-strand synthesis, SfiI digestion, size selection greater than 2 kb, pDNR-LIB, and `Library is normalized.` in its source note. The complete multiline wording is preserved verbatim, rather than replaced by this summary.
- GW787538.1 instead reports `Suppression subtractive hybridization (SSH) library` of `6 hour salt stressed foxtail millet.` and `/dev_stage="21 days old seedlings"`. It has no clone or tissue_type qualifier. Its title/description is not promoted into either missing qualifier.
- BioProject is not reported in any of these 16 GenBank DBLINK sections. This does not establish that no related BioProject exists. Empty JSON lists mean not observed in this response, not a fabricated identifier or negative biological finding.
- Repeated clone qualifier values are observations only. They do not certify read pairing, all mates, complete molecules, complete library membership, or unique eligible-molecule totals.
- No BioSample batch was needed for the per-member GenBank fields and none was requested. Biological-preparation equivalence, all library aliases, and cross-library identity remain unresolved; shared BioSample alone is not used to collapse libraries.
- Actual A11 audit, secondary-hit/other required audit closure, exclusions, route A closure, and downstream acceptance remain blocked or unchanged. Metadata recovery is not training eligibility, raster acceptance, or full-source acceptance. No eligible-molecule counts were calculated or normalized.

## Files and reproduction

- `nuccore_16.genbank.raw`: original response bytes, including sequences only for the existing selected members; not a new downstream sequence dataset.
- `fetch_receipt.json`, `nuccore_headers.raw`, `curl_transfer.json`, `curl_stderr.txt`: exact URL, UTC interval, command/resource limits, transport details and hashes.
- `metadata.json`: every feature qualifier plus complete raw feature/header/comment sections, per-record byte spans and hashes. Qualifier values retain source line breaks/indentation after removing only outer quote delimiters. The raw source section also retains qualifier syntax and quotes.
- `metadata.tsv`: 16 per-member rows. Multiline fields and lists are JSON-encoded inside standard TSV quoting; use `csv.DictReader(..., delimiter='\t')`, then `json.loads` those fields. This avoids losing multiline provenance.
- `validation_receipt.json`: complete accession-set, sequence, parser and output hash bindings.
- `recover_metadata.py`: stdlib-only, offline replay by default; `--fetch` refuses any repeat attempt once the raw file or fetch receipt exists.
- `test_parser.py`: offline original-fixture and fail-closed checks for duplicated/missing/extra or wrong-version records, changed sequence, incomplete response, stray response bytes, length mismatch, and parent sequence-hash mismatch.

Run locally from this folder:

```sh
PYTHONDONTWRITEBYTECODE=1 python test_parser.py
PYTHONDONTWRITEBYTECODE=1 python recover_metadata.py
```

The initial tests failed before the parser existed; the implemented parser passed both test methods including their rejection subcases and completed the full real batch validation. `verification.log` records an additional check using already-installed BioPython 1.83: all 16 accession IDs, sequences, dates and source qualifiers agree with the independent parser (whitespace folded only for this semantic comparison). It also records 16 lossless TSV source/note round trips, per-record/output hash checks, no forbidden punctuation, and equality of the first record with the controller's earlier candidate. No dependency was installed. No source code, remote cluster, existing diagnostics, jobs, controller receipt, or acceptance settings were modified. Only this new folder was written.

## Work record

- Complete: exact bounded public batch recovery, lossless metadata extraction, identity validation and retained receipts.
- Complete: per-member field review and offline reproducibility checks.
- Blocked/unresolved: full A11 audit/exclusions, library/mate completeness, preparation-alias equivalence and missing BioProject linkage. No broad search or mate download was attempted.
- Tool limitations: `graft map` and the targeted `graft ask` timed out; the explicitly supplied retained paths and current controller handoff were used. Advisor consultation failed because the installed Claude client did not support its configured model; no advisor approval is claimed.
