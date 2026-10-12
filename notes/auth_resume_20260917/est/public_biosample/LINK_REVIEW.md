# Bounded BioSample linkage follow-up

## Blocked: network timeout, stopped without retry

One HTTPS NCBI EFetch BioSample batch was attempted for exactly SAMN00736318, SAMN00736317, and SAMN00168307 on 2026-09-18, 00:57:59 to 00:58:04 UTC. Curl exited 28: `SSL connection timeout`. No HTTP response, headers, or body were received. Limits: 30-second total, 5-second connection, 1 MiB maximum response. No retries, redirects, alternative endpoints, or broad searches were attempted.

`fetch_receipt.json` preserves the exact URL, command, UTC interval, and file status. `curl_transfer.json`, `curl_stderr.txt`, and the empty `headers.raw` retain transport evidence. There is no raw XML response, so its SHA256 is null, not an invented hash of an unreceived document. The empty header artifact has SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Relation to retained observations

The existing `../public_metadata/METADATA_REVIEW.md` and `metadata.json` remain the only evidence reviewed: 16 GenBank observations associate CFGZ with SAMN00736318, CFBX with SAMN00736317, and GW787538.1 with SAMN00168307. Those records report library wording and preparation descriptions but no BioProject in their DBLINK sections. This follow-up adds no BioSample fields or linkage evidence.

Exact returned sample-ID validation and XML extraction could not be performed. BioProject, BioSample dbxrefs, source preparation, library, donor, genotype, tissue attributes, and descriptions are unobserved in this attempt. Missing BioProject linkage and preparation-identity equivalence remain unresolved, not proven absent from NCBI. No identifiers were invented. Shared BioSample does not establish one library, all aliases, or biological-preparation equivalence.

## Scope and work record

- Complete: single bounded fetch attempt and failure receipts.
- Blocked: XML validation, field extraction, and additional linkage grounding because no response was obtained. Stopped as instructed.
- Unchanged: GPU/storage I/O and ONT gates, scientific eligibility, source policy, and all existing records. No cluster access, jobs, source edits, mate retrieval, new sequence dataset, or molecular counts.
- Writes confined to this new `public_biosample/` folder. Context graph orientation timed out; explicitly supplied retained notes were used. No production parser or framework was created.
