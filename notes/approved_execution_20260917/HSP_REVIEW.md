# Independent HSP review — 2026-09-17

## Verdict: scoped APPROVE

No Critical or Important correctness/spec findings in the reviewed bounded API. Approval covers the finished `est_hsp_contract.py`, `tests/test_est_hsp_contract.py`, and `tasknotes/hsp/REPORT.md` against `notes/APPROVED_HSP_PHASE_A11_B5_20260917.md`. It does not approve production integration, source authority, phase/A11 implementations, or scientific certification.

Reviewed in `/tmp/est-approved-contract-20260917`. Graft map/query preceded source inspection. Unchanged dependencies `est_phase_accept.py` (`parse_row`, `btop_tokens`) and `est_hsp_expanded.py` (`interpret`) were examined; `git diff c9a3514 --exit-code -- <both paths>` returned 0. Other workers' disjoint unfinished implementations were not reviewed. No implementation edits, commits, remote jobs, or subagents. Advisor consultation failed with model/client-version HTTP400; no advisory validation is claimed.

## Contract checks

- Raw13 requires exactly 13 nonempty whitespace-free strings; validates coordinate bounds/orientation, finite nonnegative evalue <=1e-5, finite positive bitscore, signed frame, actual canonical sequence lengths, and BTOP syntax/consumption. Match runs are bounded before expansion; codon and subject truth are checked against supplied sequences. Rich replay rejects initial-frame, terminal-coordinate and serialization disagreement.
- Acceptance is exactly validated reported col9=M+I+D >=30. F does not count; neither paired/query residue count nor each constant-frame segment imposes a substitute threshold. D30 with only29 query residues and a split 15+15 HSP remain eligible.
- M/I events refer to actual translated codons in original query coordinates, with reverse direction -1, and local c0=0. D advances subject without inventing a query codon; I has no subject index. Subject start offsets and internal D/I were additionally exercised, not just terminal indels.
- +1 identifies the omitted oriented nucleotide; -1 identifies the reused nucleotide at pre-shift position minus1. Hand-specified forward/reverse mixed traces verify the corresponding original coordinates and overlapping codon starts.
- Returned IDs and SHA256 identities describe supplied row/sequences. The rich classifier's `raw_row_sha256` hashes all17 supplied fields, as documented; adapter preserves first13 exactly, and replay separately records raw13 hashes. Expected identities enforce supplied assertions; empty/unknown/mismatched assertions reject.

## Fresh execution evidence

Evidence directory: [`hsp_review_evidence/`](hsp_review_evidence/).

From worktree root (set `E=/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/approved_execution_20260917/hsp_review_evidence`):

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_est_hsp_contract.py \
  revision/scripts/tests/test_est_hsp_expanded.py \
  revision/scripts/tests/test_est_blastx_fixture.py \
  revision/scripts/tests/test_est_phase_project.py > "$E/pytest.log"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python tasknotes/hsp/replay_actual.py > "$E/actual_replay.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python "$E/adversarial.py" > "$E/adversarial.json"
git diff c9a3514 --exit-code -- revision/scripts/evidence/est_phase_accept.py revision/scripts/evidence/est_hsp_expanded.py
```

- **140 passed in 0.53s**, including the new65 focused tests.
- Read replay script before running. **144/144 exact raw13→rich17 reconstruction**, **126 HSP_RULE_ACCEPTED /18 BELOW_HSP_LENGTH_30**; aggregate M9294/I30/D30/F96. These are empirical outcomes for these144 rows, not a universal guarantee or new production gate.
- **63 independent assertions passed**: hand-specified mixed internal indels/shifts, nonzero query/subject starts, both strands, event positions/subject indices, identity checks, endpoint overrun, subject truth, nonfinite/negative scores, evalue boundary, invalid frame, noncanonical/zero/padded BTOP runs, and malformed shift tokens.
- Actual fixture replay hashes its five original inputs before/after and confirms equality. `diagnostic.json` still has **132 ELICITED /12 NOT_ELICITED** cases and historical `acceptance_policy: UNRESOLVED`. Nothing relabels the12 unelicited cases. Case counts and144 emitted rows are distinct units. Input hashes are in `actual_replay.json`; diagnostic SHA256 is `99fb4acaac0a46712d8723f608a39755926151e85edbf3dd99173d76bd1fbfcb`.
- The implementation report's green/regression/replay claims are reproduced. Its historical RED run was not re-executed by removing the implementation; this review does not independently certify the chronology of TDD.

## Explicit limitations (not blockers)

Caller must bind actual screened DNA/protein to qseqid/sseqid. Hashes cannot establish FASTA mapping, lineage, A11 or complete consumer binding. No production caller integration is claimed by these per-row pure functions. Shared genetic code/interpreter semantics mean exact replay is a serialization/consistency check, not independent biological validation. This scope is consistent with the approved bounded diagnostic implementation; no extra production-completeness requirement is imposed.

## Reviewed source SHA256

```text
fcb8a4b2413f12dc5c5b0c4d55676cedeac21b3fcb26cf849c089657650a3954  revision/scripts/evidence/est_hsp_contract.py
8d7cc3f0901c16569319a85baac3708d965f3f0c650d422b4684e9208ce3b39d  tests/test_est_hsp_contract.py
28dad36b296261cf11b3825cb19e1027fd52eb1e53f2061fbe819d720b0581aa  tasknotes/hsp/REPORT.md
709268740179afd312bbd0f6bf72e3ee532dc99af0d497715311b682a9c1fda7  tasknotes/hsp/replay_actual.py
0c6490c76d63463d945ee270f46a4ab8b96e929b57f4ac38dc52229f24c11ae3  revision/scripts/evidence/est_phase_accept.py
aaa8348b4e5d47d43c065857e3bb06509f63086a6d2c79f35f93e88bc211b3e4  revision/scripts/evidence/est_hsp_expanded.py
```

Graft reported approximately833,835 tokens saved across the three context calls (tool estimates).
