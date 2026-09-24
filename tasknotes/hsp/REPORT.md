# Approved HSP contract bounded unit — 2026-09-17

Implemented only `revision/scripts/evidence/est_hsp_contract.py`, `tests/test_est_hsp_contract.py`, and this `tasknotes/hsp/` directory in the existing `/tmp/est-approved-contract-20260917` worktree. No commit, engine invocation, job, remote write, historical gate edit or production/MVP change.

## APIs and semantics

- `classify_rich17(row, dna, protein, *, expected_identities=None)` validates existing raw13 fields and canonical actual sequence lengths, calls the unchanged rich17 interpreter, requires reported col9 = M+I+D, then classifies exactly >=30. Invalid evidence raises ValueError, not an accepted diagnostic. F and per-segment cutoffs are excluded.
- Result names `DIAMOND_REPORTED_HSP_M_PLUS_I_PLUS_D_GE30`, `accepted_by_hsp_rule`, disposition, explicit diagnostic-only scope, q/s IDs and SHA256 identities. Hash convention: UTF-8 tab-joined input row fields without newline; exact supplied DNA/protein text. Optional expected hashes fail closed for mismatches, unknown keys and an empty identity assertion.
- Exact interpreted codons become M/I events with `q0` = first translated base, zero-based ORIGINAL screened query coordinate; length=3, direction=+1/-1, c0=0; subject amino-acid index zero-based or None for I. Reverse bases are q0, q0-1, q0-2. This is a local codon anchor, not projected phase inferred from qframe.
- Frameshift records retain interpreter oriented `at`, `delta`, marker, plus original-coordinate omitted (+1) or reused (-1) nucleotide. No codon event is invented for F or D.
- `adapt_raw13(row, dna, protein)` returns exact reconstructed rich17 fields, preserving the first13 fields. Uses existing `parse_row`/`btop_tokens`, actual DNA/protein equality checks, bounded match-run expansion and gapped-string-driven interpreter replay. It also enforces the approved col9 contract. No candidate_decompose use and no file-processing/acceptance CLI. Its returned fields remain diagnostic only as documented.

## TDD and actual verification

- RED before implementation: `python -m pytest -q tests/test_est_hsp_contract.py`; **65 failed**, expected assertion `approved HSP contract not implemented`. Full `red.log` preserved.
- GREEN: same command; **65 passed**, `green.log`.
- Regression command: `python -m pytest -q tests/test_est_hsp_contract.py revision/scripts/tests/test_est_hsp_expanded.py revision/scripts/tests/test_est_blastx_fixture.py revision/scripts/tests/test_est_phase_project.py`; **140 passed**, `regression.log`.
- Tests cover 29/30/31, all six frames, reverse events, D30(query29), I29(paired28), I30(paired29), substitutions, insertion/deletion, plus/minus shifts, whole-vs-segment, missing IDs, bounds/endpoints, col9 mismatch, malformed BTOP/CIGAR/aligned/translated strings, nonfinite scores, noncanonical sequences and failed expected identities.
- `PYTHONPATH=. python tasknotes/hsp/replay_actual.py > tasknotes/hsp/actual_replay.json`: preserved actual FASTAs and rows validated; **144/144 exact raw13→rich17 equality**, **126 HSP_RULE_ACCEPTED /18 BELOW_HSP_LENGTH_30**, observed counts M9294/I30/D30/F96. No expected disposition count was forced in the replay script. Full small per-row identity report is stored, not a copied diagnose artifact.
- Replay source is the absolute external fixture path named in the task. FASTA IDs are looked up exactly and duplicates rejected. Fixture input hashes, including diagnostic.json and manifest.json, were checked unchanged before/after. Original diagnostic statuses (including NOT_ELICITED) are neither modified nor relabeled.
- `python -m compileall -q revision/scripts/evidence/est_hsp_contract.py tests/test_est_hsp_contract.py` and `git diff --check`: success.

## Limitations / handoff

Per-row functions assume the caller correctly binds actual original-screened DNA/protein to q/s IDs. Content hashes cannot certify source-file mapping, FASTA lineage, A11, genomic projection, production eligibility or training eligibility. No fabricated FASTAs/lineage authority. Raw13 reconstruction and interpret share the genetic code and semantics: serialization replay is a cross-check, not independent biological proof. The adapter preserves raw13 fields but does not return a separate original13 identity; the replay report records that hash explicitly alongside the classifier's reconstructed rich17 identity.

Controller independent review/commit coordination remains pending. Other workers' `.gitignore`, `.ignore`, phase/closure and tasknote changes were untouched. Advisor planning consultation failed with client/model-version HTTP400; no advisory validation claimed. Graft was queried before source reads (covers interpret L49–126 and parse_row L110–126); no graph rebuild for this bounded addition to avoid touching files outside the assigned scope.
