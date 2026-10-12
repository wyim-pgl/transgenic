# Independent adversarial length-semantics review

## Decision

**The mechanism is sufficiently established by pinned static source for a narrowly producer-path-bound length contract. It is not authorization to implement, promote previously withheld rows, or generalize by database name.** The hypothesis survived inspection of dispatch, traceback, filtering, constructors, output callbacks, serialization and iterator semantics. M means all aligned residue pairs, including substitutions; I and D count amino-acid gap columns; F counts frameshift operations, not their signed sum. Direct legacy tabular output retains M+I+D; blocked transcript output reparsed by the joiner reports M+I+D+F.

This is static verification with historical provenance attribution, not experimental reproduction or a reproducible-build attestation. The reports were read, but the conclusion below was checked against source bytes rather than accepted from those reports.

## Rehash and provenance checks

Recomputed `upstream/v2.1.24.tar.gz` SHA256:

`4879c27a8fc96d84793c3239a314cc3ff78f3b26ee6a3f228cddab6338bc0990`

It equals the embedded recipe pin. All 18 retained source files are byte-identical to their regular archive members and match every entry in `upstream/evidence.json:selected_files_sha256`. The two enclosing receipt hashes and all ten embedded text-record hashes also match that manifest and their receipts. No upstream source was executed or changed.

Independently rehashed `notes/seed123_execution_20260917/est_source_metadata.txt`: `f814b5665f60421826eec838bd259f3728ec793d58e1c6ea4a3aa4870769b998`. Inspected its actual command/provenance payloads. Both record executable SHA256 `0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d`, frameshift 15, outfmt 6 and the same raw13 field order with length ninth and BTOP thirteenth. Commands differ in database/output/temp paths, not the relevant options. Their temporary directories match the respective retained logs; database paths, retained sequence counts and reported HSP counts also agree. OrthoDB explicitly logs three reference blocks, temporary output and joining. SwissProt logs one reference block with neither temporary-output nor joining messages.

The package record attributes that executable hash to Bioconda `diamond-2.1.24-hf93d47f_0`; the retained recipe pins the archive and has no source patches or source-editing build steps. This does not cryptographically prove that the historical executable was compiled from these exact sources. Logs themselves do not embed the raw-output hash or complete command; linkage to raw bytes relies on the historical provenance chain. This review did not rehash original executable, databases, queries or raw outputs, or independently revalidate release/sample manifests. A future contract must preserve those bindings rather than treat matching paths as sufficient identity.

## Source-level attempts to refute

All spans below are under archive `diamond-2.1.24/src/` and refer to the verified bytes.

1. **Wrong dispatch or score-only HSP leaking into output?** `align/align.cpp:147-158,182-187` selects the legacy mapper for nonzero frameshift and continues before the newer extension output branch. `align/legacy/banded_swipe_pipeline.cpp:158-170,250-263` clears score-only HSPs and runs traceback on both strands. Its `reset`/`finish` at 107-119 and `align/legacy/query_mapper.cpp:319-348` only clear, cull or filter; they do not reparse or rewrite length. The parallel, scalar and overflow routes in `dp/swipe/banded_3frame_swipe.cpp:533-558,575-639` preserve the score-only/traceback selection.

2. **Traceback already counts F?** `dp/swipe/banded_3frame_swipe.cpp:358-400` calls `push_match` for the residue pair and separately appends either shift, without incrementing length for the shift. `basic/hssp.cpp:267-296` increments length for each residue pair or by gap size. Reversal and termination of the transcript add no alignment columns.

3. **Direct output silently reparses?** `align/legacy/query_mapper.cpp:256-285` directly constructs and prints an HspContext. Its constructor only copies the HSP (`basic/match.h:287-315`). `output/blast_tab_format.cpp:700-734` dispatches callbacks; length reads the stored value at 312-314, and BTOP only iterates at 375-409. An archive-wide textual search of C++ parse call sites found the joiner and DAA decoder, not an additional direct legacy parse. The DAA decoder explicitly reparses at `output/daa/daa_record.cpp:74-95`, so DAA/view is a real alternative mechanism, excluded by the recorded raw tabular command. This search is not a formal compiler-generated call graph.

4. **Blocked output preserves authoritative length?** `output/blast_tab_format.cpp:81` requires TRANSCRIPT for BTOP. `output/output_format.cpp:43-75,105-131` selects transcript instead of statistics, even when length is requested too. `output/output.h:85-87` makes stats mode false whenever TRANSCRIPT is present. `basic/hssp.cpp:300-340` restores saved length only in stats mode. Saved query endpoints are not a substitute saved length. `output/join_blocks.cpp:215-275` then explicitly invokes parse. The no-transcript shortcut at `basic/hssp.cpp:59-71` is unavailable; lines 73-108 reset length and increment it before every operation switch, with no shift subtraction.

5. **Packing changes operation multiplicity?** `basic/packed_transcript.h:45-85,139-146,159-164` stores shifts as distinct substitution sentinels and decodes each with count one. `basic/match.h:113-158` expands match/insertion runs per column and advances once per shift, moving query position only. Both shift signs therefore contribute one each after parse, regardless of strand; a forward/reverse pair does not cancel. The terminator contributes nothing.

6. **One block necessarily means direct?** No. `run/double_indexed.cpp:414-421` determines blocking from loaded sequence counts; `:215-241,491-551` also exposes iterated/multiprocessing/global-ranking alternatives. The retained commands and matching complete logs, not block count or source name alone, delimit the ordinary direct-versus-blocked paths considered here. Other modes are outside this contract.

## Minimum future contract and limitations

- Bind exact executable/package/archive identities, command/options/raw13 schema, query/database/raw-output receipts and the matching retained execution log to one immutable producer record. Attach the independently established path to that record before evaluating rows.
- Permit exactly the path's formula: MID for the verified direct legacy path, MIDF for the verified blocked transcript join. Never choose from row fit, source name, shift sign, or acceptance of either formula. Unknown, mixed or conflicting path evidence is unassessable, including rows where F=0 and formulas coincide.
- Keep reported `int(row[8]) >= 30` unchanged. Do not subtract F for eligibility. Preserve identity/hash, endpoint, exact-residue, geometry/conflict and scientific eligibility gates.
- Before any separately authorized implementation, require path-bound fixtures covering both signs/strands, substitutions, gaps, zero/multiple shifts, cancelling signed shifts, reported 29/30/31, wrong/missing provenance and DAA/other-mode rejection. No such new runtime verification was performed here.
- Masking remains unassessable at the affected residue positions. Nothing in this length review authorizes X wildcard matching or promotion of concealed numeric-run disagreements. Other withheld failures and accounting/OOM uncertainties remain unresolved.

Completed: independent source/provenance inspection and local rehashes. Not performed: download, build, DIAMOND execution, remote calls, real-input scans or validator implementation. Advisor consultation failed due to client/model incompatibility and supplied no evidence. The required initial graft lookup automatically refreshed graph metadata; no upstream or application source edits were made. Graft reported approximately 1,188,781 tokens saved across successful outputs.
