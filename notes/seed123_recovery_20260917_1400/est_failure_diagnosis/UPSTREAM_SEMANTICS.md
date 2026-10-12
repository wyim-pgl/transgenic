# Pinned upstream semantics: DIAMOND 2.1.24.178

## Scope and provenance

Completed static source investigation only. No DIAMOND build, installation or execution, real-data replay, raw/BAM scan, cluster call, job submission, acceptance change or threshold change occurred. The requested shorthand `recovery/est_failure_diagnosis` resolves here to `notes/seed123_recovery_20260917_1400/est_failure_diagnosis`.

Archive A: `https://github.com/bbuchfink/diamond/archive/v2.1.24.tar.gz`, SHA256 `4879c27a8fc96d84793c3239a314cc3ff78f3b26ee6a3f228cddab6338bc0990`. Download succeeded using HTTPS-only redirects, 60-second timeout and 64-MiB cap. Hash was verified before inspection/extraction. `upstream/v2.1.24.tar.gz` retains A; selected regular source files retain their original bytes under `upstream/diamond-2.1.24/`. `upstream/evidence.json` records file hashes and local provenance verification. All source citations below mean **A, diamond-2.1.24/src/<path>:lines**, not current upstream.

`producer_small_logs.txt` embeds the package record linking the installed executable hash `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d` to Bioconda `diamond-2.1.24-hf93d47f_0`. `producer_recipe.txt` pins A and contains no source patches or source-editing build steps. This is a package/provenance attribution, not an independently reproduced binary build. Both blastx logs identify build 2.1.24.178 and Double-indexed search. OrthoDB logs reference blocks 1/3 through 3/3, temporary output and joining; SwissProt logs 1/1 without joining.

## Finding: hypothesis survives attempted refutation

The difference is the **blocked intermediate-output reparse**, not database/source labels or frameshift sign. Define M as all residue-pair columns, including substitutions; I/D as amino-acid gap columns; F as frameshift operations. On these pinned frameshift/BTOP paths:

- Direct single-block output reports `M+I+D`.
- Joined multi-block output reports `M+I+D+F`.

The decoder does not corrupt or duplicate a shift. The joined parser resets length and counts each decoded transcript operation, including each shift, whereas the original traceback does not add length for its extra shift operation.

### Exact call chain and accumulator

1. `run/double_indexed.cpp:414-421` sets `blocked_processing` when the first loaded reference block contains fewer sequences than the database. `:215-241` selects temporary output and calls `align_queries`; `:491-551` joins blocked output.
2. `align/align.cpp:147-158,182-187` dispatches nonzero frameshift penalty through **legacy_pipeline**, constructs `ExtensionPipeline::BandedSwipe::Pipeline`, runs it, then calls `QueryMapper::generate_output`. The newer `Extension::generate_output` branch is bypassed here.
3. `align/legacy/banded_swipe_pipeline.cpp:158-170,259` calls `run_swipe(false)` and `banded_3frame_swipe` on both strands. `dp/swipe/banded_3frame_swipe.cpp:533-558` selects traceback; `:358-400` calls `push_match` once for each residue pair and separately pushes a forward/reverse shift, without a length increment for that shift. `basic/hssp.cpp:267-296` increments length once per pair or by gap length.
4. `align/legacy/query_mapper.cpp:256-285` sends blocked results to `IntermediateRecord::write`; the direct branch calls `print_match(HspContext(...))` **without parse**. `output/blast_tab_format.cpp:312-314` prints `r.length()`; `basic/match.h:400-401` returns stored HSP length.
5. BTOP requests `HspValues::TRANSCRIPT` (`output/blast_tab_format.cpp:81`). `output/output_format.cpp:105-131,43-75` writes/reads the transcript instead of the statistics, including length. `basic/packed_transcript.h:45-85,139-146,159-164` preserves shift sentinels, decoded as distinct operations of count one. `basic/hssp.cpp:300-340` reconstructs the HSP; no saved length is restored in transcript mode.
6. `output/join_blocks.cpp:154,215-275` decodes records, reconstructs HSPs and explicitly calls `HspContext::parse` before printing. `basic/hssp.cpp:59-108` cannot take the no-transcript shortcut for BTOP: it zeroes length and executes `++hsp_.length` before every operation switch. Neither shift is subtracted. `basic/match.h:125-152` consumes one iterator step per shift, changing query position but not subject position.

Refutations checked: both searches use the same algorithm/build; both shift signs follow the same counting logic; no source-label condition controls this branch; the direct path does not also reparse; the intermediate transcript branch does not preserve authoritative original length; compressed match runs expand per column. Without F the formulas coincide. This explains the retained 44-residue `8/-36` example reporting 45 after joining, without rerunning it. It is not a universal rule for other versions, output fields, DAA/view, or stats-only output.

## BTOP, masking and query clearing

`output/blast_tab_format.cpp:375-409` serializes numeric match runs, literal substitutions, gaps and shifts. `basic/match.h:323-359` obtains query letters from the query sequence, substitution/deletion subject letters from the packed transcript, and renders forward/reverse shifts as `\\-` and `/-`. Numeric matches contain no residue identities.

Default masking is tantan on query and target (`run/config.cpp:121-133`). `run/double_indexed.cpp:125-127,717-719` invokes hard masking on both; `:297` excludes this frameshift path from lazy masking. `masking/masking.cpp:161-170,220-239` invokes tantan mode 1 without a restoration table; `masking/tantan.cpp:189-210` replaces masked letters with the mask character. The logged “Clearing query masking” calls `FrequentSeeds::clear_masking` (`run/double_indexed.cpp:208-209`, `data/frequent_seeds.cpp:128-135`): `letter_mask` strips flag bits or is a no-op (`basic/value.h:70-88`), not reconstruction of residues replaced by X. Both output paths therefore can expose hard-masked query/subject symbols. `Hsp::push_match` compares the actual masked symbols (`basic/hssp.cpp:267-280`), so an X/X equality can become an information-losing numeric match run.

Logs plus source establish the mechanism, not the exact mask at each retained position. No masked query arrays, complete per-residue mask trace or producer masking replay were obtained. Thus exact residue replay for the X discrepancies and concealed numeric-run disagreements remains unavailable from these artifacts. Do not interpret X as a wildcard, canonical mismatch as proven source corruption, or “clearing” as proof of unmasked truth.

## Minimal producer-bound contract, not an implementation change

A future authorized contract must bind executable/package/archive identities; exact blastx command and raw13 field order including BTOP; query/database/output provenance receipts; and **the actual output path** established by the matching retained log. For this run, OrthoDB's logged blocked/joined path selects MIDF, SwissProt's direct single-block path selects MID. Source names alone, row-fit selection, or accepting either formula are insufficient. Unknown or conflicting path evidence remains unassessable.

Preserve the reported-column threshold `int(row[8]) >= 30`; never normalize away F for eligibility. Preserve identity/hash, endpoints, exact-residue and geometry/conflict gates. Any future change requires path-bound fixtures, both shift signs/strands, gaps, multiple/no shifts, and reported 29/30/31 boundaries. This static diagnosis supplies a length-contract basis only. X failures, five geometry failures, accounting/OOM uncertainty and all scientific eligibility restrictions remain withheld or unresolved as before.

Completed: archive attribution, static branch/serialization/masking trace. Blocked: exact per-column masking replay and any validation implementation, neither authorized here. Advisor consultation failed due to client/model incompatibility and supplied no review evidence. Graft context lookup saved approximately 35,001 tokens; parent-directory graph attempts timed out.
