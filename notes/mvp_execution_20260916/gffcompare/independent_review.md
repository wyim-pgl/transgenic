# Independent execution/correctness review

## Verdict: APPROVE bounded synthetic smoke

No blocking correctness or execution-safety issue found for the exact reviewed `python -I`/Slurm invocation. This approves staging and one bounded diagnostic execution, **not** a semantic PASS, biological score, GPU action, merge, or author multiplicity policy. No actual GFFCompare comparisons have been observed by this reviewer.

Read the complete plan, `smoke.py`, and `smoke.sbatch`; inspected the exact producer definitions and `produce` version/provenance path. Independently confirmed `/tmp/b6-p2-scoring-1535144` HEAD `c024445` and producer SHA256 `bf7c8284804ee0a12c359a1fbe8f140bfc0f222abb522d6d7b7b4d3a691b417b`. Harness/template hashes match the plan:

- `smoke.py`: `3c022e41fa2759a33d8fee76452a55eca5ff2b226527a37fddecbba79c6098d1`
- `smoke.sbatch`: `e6d18fcc732e183b7b3f40d3176e76889608f316453070ef4e12c5086ae0258b`

## Correctness findings

- `smoke.py:29–42` extracts the five exact AST function definitions from SHA-pinned producer bytes. Its only dependency shim is SHA256; there is no replacement matcher or fabricated tmap. Producer `revision/scripts/mvp/mvp_produce_counts.py:297–346` executes the unchanged `-r reference.gtf -o compare predicted.gtf` command and enforces exact query inventory, class `=`, chromosome/strand/full-chain agreement, and unique reference matches.
- The 16 cases (`smoke.py:49–73`) agree with the hand calculations. Both ±20 terminal changes retain `(200,300),(400,500)`; moving the middle start to310 changes one junction; opposite strand and disjoint chains cannot match. The two distinct chains in `multiisoform_unique` have two unique matches. Mono exact nonempty CDS matches independently of UTR exon boundaries; changed CDS/strand and absent CDS do not. Empty query yields TP0/FP0/FN1 without a producer tool call.
- Duplicate query/reference terminal structures and duplicate mono CDS are correctly required to raise the exact producer guard error before GTF/tool output (`smoke.py:142–144`). Raw duplicate rows are observations only, including collapse/reference choice; no multiplicity convention is silently approved. Raw empty-file behavior is likewise observation-only.
- `smoke.py:124–175` is read-only: inventory/hash mismatch, absent manifests, incorrect fixture/runtime identity, missing result/command/version evidence, wrong counts/classes, or unexpected producer errors prevent normal PASS. Assertions remain enabled under the prescribed `-I` invocation; do not introduce `-O`/`-OO`. Complete, self-consistent evidence is provenance, not adversarial attestation.

## Execution boundary

`smoke.sbatch:2–6` requests one task, two CPUs, 4GB, 15minutes, no GPU. Python and GFFCompare paths are absolute. `generate` refuses an existing run directory; source/harness/binary hashes are checked across execution, and the completed raw-file inventory is hashed. The controller must verify the exact three-entry staging manifest, retain Slurm stdout/stderr plus scheduler allocation/exit status, and pin the printed `hashes.json` digest. `sha256sum -c` alone does not prove a manifest lists all three intended files; this remains the explicitly assigned controller gate.

Use the controller's supplied `R=/data/gpfs/assoc/pgl/data/Transgenic`, fresh namespace `evidence/mvp_gffcompare_smoke_20260916`, and explicit approved account/partition selection (`cpu-s1-pgl-0` as supplied by the controller). The template intentionally does not guess those submission settings. Check live queue before submission; do not overwrite or resume an existing run. Availability evidence establishes login-side 0.12.6/Python3.11.14 only, not compute execution.

## Nonblocking diagnostic limitation

**Low — incomplete logs on exceptional subprocess failure:** `smoke.py:86–93` writes captured stdout/stderr/status only after `subprocess.run` returns. A timeout or launch exception leaves the command record but not those files. Likewise, unchanged producer `mvp_produce_counts.py:328–330` uses `check=True` before writing captured logs; `smoke.py:115–116` retains exception type/text, not its captured streams. Reproduction path by inspection: a raw child exceeding60seconds raises `TimeoutExpired` before log writes; a producer child returning nonzero raises `CalledProcessError` before producer log writes. No reproduction/tool execution was performed. Both paths fail closed and cannot certify semantics, so this does not block this tiny bounded execution. The plan's “full raw logs/status” promise is accurate for returned raw calls and successful producer calls, not guaranteed on these failure paths. If encountered, preserve the incomplete evidence and scheduler failure logs and investigate rather than claiming PASS or silently rerunning in place.

## Review actions and limits

No SSH, jobs, source edits, installations, or subagents. No duplicate selfcheck/syntax suite: controller-reported passing checks were not reclassified as reviewer-run evidence. Only local reads, graph queries, identity checks, and this report were used. Graft auto-refreshed its index during the required context query. Advisor consultation was attempted but unavailable due to client/model version mismatch; no advisor approval is claimed.

This remains a function-level synthetic tool-semantics diagnostic. Full `produce`, parsing/frozen-panel compatibility, biological scoring, independent-read provenance, GPU execution, and author policy are outside approval; `decision=null` remains required.
