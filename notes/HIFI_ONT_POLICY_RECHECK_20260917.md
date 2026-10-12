# HiFi / ONT policy recheck — 2026-09-17

## Verdict

The blanket TODO **“HiFi acquisition or ONT-only policy pending author approval” is not an accurate statement of the remaining work.** Existing evidence-source rules and exploratory MVP execution were already approved. A hardcoded pending-policy exception in implementation is not evidence that the author withheld approval.

There is nevertheless a narrower unresolved **artifact identity/availability** issue: the named MVP A-HiFi accession ERR13994458 is not in the canonical role table or the candidate paths inspected. Other Arabidopsis PacBio data **were already acquired and converted to FASTA**. That is not proof that ERR13994458 FLNC was produced, nor that the downloaded other PacBio files are validated CCS/FLNC or independently aligned for this MVP. “No HiFi anywhere” and “no PacBio downloaded” are both unjustified conclusions.

**No renewed policy approval is needed to implement the approved evaluator, preserve per-source missingness, run authorized preparation within its operational budget, or apply the existing missing-structural STOP→INCONCLUSIVE safeguard.** A proposal to replace named A-HiFi with another accession, or to let ONT alone certify the complete structural availability required for STOP despite missing named A-HiFi, is a narrower potential specification change—not a prerequisite to all implementation or reporting.

## Authority, in order

1. **B1 source/role rules already exist.** `revision/protocols/PROTOCOL_B1_frozen_v1.md:78–110,139–192` declares ONT and PacBio equal tier 1, source-specific thresholds, per-run results plus pooling, per-source and union callability, and non-callable observations not negative evidence. It does not require both technologies to support each positive junction/chain. A14 (`:591–602`) designates A-ONT1 and A-HiFi validation-only; A15 (`:605–614`) additionally designates A-CCS23 validation-only. These are not policies waiting to be invented.
2. **Exact recorded MVP approval:** `notes/b6_execution_approval_20260914.md:3–16` records the author's **“mvp 승인할께 진행해줘”**, authorizing implementation and execution of MVP v1.2, SHA256 `734387b0a3251e7b0f47cf36bb0edc0dc51549a02f0c2d2579765d5445879c74`. This supersedes that draft's stale approval-pending header, without approving full A46/B6. The same record separately quotes “EST blastx 실행”.
3. **Follow-up authorization:** `notes/b6_parallel_execution_authorization_20260916.md:3–8` records the explicit request for MVP training/decoding/**independent long-read scoring/bootstrap** through subagents, keeping the finalized spec authoritative. This is not an ONT-only waiver, but is clear authorization to implement scoring.
4. **Approved MVP contract:** `revision/protocols/B6_MVP_PILOT_v1.md:99–150,240–285,295–297` names A-EST as input and A-ONT1/A-HiFi as scoring; defines supported junctions as ≥2 A-HiFi **or** ≥3 A-ONT1 reads; requires per-run plus pooled scores and full-reference pass 2. §5.3 already supplies the decision rule and missing-structural safeguard. Missing structural evidence downgrades prospective STOP to INCONCLUSIVE; it does not create a blanket ban on computing readouts or every GO. GO still requires its own primary criteria and the exact STOP-first ordering. No result is asserted here.
5. **Historical checks:** `.atl/archive/todo_before_20260916.md:104–105,217` records approval resolved and the earlier author's “동의” to WP-MVP; `:145,150,160,220` and `.atl/archive/resume_before_20260916.md:69–89` retain older draft/review/approval-pending stages. Those are not current vetoes. `notes/author_decisions_20260905.md` concerns B5 #55/#56/#58/#61 and #54; it supplies no new HiFi-versus-ONT approval gate.
6. **Not author authority:** `notes/b6_three_track_discovery_synthesis_20260916.md:29,60` proposed fetch-versus-ONT-only as a question. Later notes copied it. Graft's top match, `require_coverage_policy` in `notes/wait_work_20260916/longread/frozen/revision/scripts/mvp/mvp_score.py:26–29`, unconditionally raises “coverage policy pending author approval”. That proves a software block exists, not that the author's approved specification is absent. `REPORT.md:23,58` and its downstream review/gate statements similarly are implementation-owner constraints, not new author decisions.

The surviving ambiguity is **complete MVP structural availability with only a subset of the named sources**, not B1's already-set source hierarchy. B1's per-source/union rules do not automatically amend MVP §2/§5.3. Conversely, MVP's missing-evidence safeguard already permits a truthful incomplete result; no new user choice is necessary to implement that path.

## Accession reconciliation

| Name | Actual identity | What is established |
|---|---|---|
| A-HiFi | Zhong 2025, PRJEB77203, **ERR13994458**, `Col_isoseq.subreads.bam → ccs/lima/refine → FLNC` | Named explicitly in B1 §3.2/A13/A14 and MVP scoring. No matching canonical role row or candidate directory observed; no CCS/FLNC production receipt located in the reviewed records. Not an exhaustive absence proof. |
| A-CCS23 | Zhang 2023, **PRJNA911826**, WT **SRR22719002–07** | B1 A15 calls this ~2.9M CCS-level reads and approves validation-only role. Six downloaded `.fa.gz` files, DONE markers, MD5 receipts and download log really exist. Their **recorded ENA source URLs are `_subreads.fastq.gz`**, with empty submitted-file fields. Thus the protocol's CCS-level description is not sufficient product validation. |
| A-ONT1 | FLIC PRJNA1087576, **SRR28341943–45** | Canonical validation-only files/roles and receipts documented in `notes/wait_work_20260916/longread/readonly_actual_inputs.txt`; this recheck did not rehash or rerun alignment. The recorded orientation sample is not full-run structural scoring. |
| Other training long reads | e.g. Cui PRJNA594286 and other Athaliana sets | Existing training alignment directories are real but are not A-ONT1/A-HiFi validation BAMs. Do not infer validation readiness from “long-read alignments complete”. |

**Specific historical factual error:** root `quarantine.md:228–230` says `Athaliana_zhang2023_PRJNA911826` is the protocol's “subreads.bam → ccs → FLNC” set. That named BAM conversion belongs to **PRJEB77203 / ERR13994458**, not Zhang's PRJNA911826. `.atl/archive/todo_before_20260916.md:569` carries the same conflation as “Zhang CCS conversion not executed”. Do not restart a supposed pending Zhang CCS conversion from this note. Conversely A15's older “CCS-level” label must not override the source-product discrepancy now observed.

The author's 2026-09-03 exclusion of 51 **training** PacBio subread runs (`quarantine.md:211–230`; `.atl/quarantine.md:133,1466`) is not proof these validation files were excluded; the historical record explicitly distinguishes them. Role eligibility, successful download, FASTA conversion, CCS/FLNC production and accepted validation alignment are five different facts.

## Fresh bounded pronghorn metadata

Read-only SSH observations **2026-09-17 10:25:12 and 10:25:40 PDT**. Root: `/data/gpfs/assoc/pgl/data/Transgenic/evidence`. Only small role/report/log/MD5 text and bounded shallow directory/stat metadata were read. No raw-read scanning, hashes, downloads, jobs, or remote writes.

- `DATASET_ROLES.longread_v1.tsv`: six PRJNA911826 rows, all `Athaliana/reference/b1_validation_only`, expected run counts below; no ERR13994458/PRJEB77203 row. Local repo `data/manifests/DATASET_ROLES.longread_v1.tsv:5–10` independently contains the same six accessions.
- `pacbio/Athaliana_zhang2023_PRJNA911826/log` ends **2026-09-02T04:31:14-0700: “finished: 6 runs DONE, 0 skipped”**. It records `_subreads.fastq.gz` source URLs for all six, not CCS/BAM processing. `filereport.tsv` is 6,106 bytes and records these same URLs and blank `submitted_ftp`/`submitted_format`.

| Run | `.fa.gz` stat bytes | Expected reads (metadata, not recounted) | Small MD5 receipt (not fresh hash) |
|---|---:|---:|---|
| SRR22719002 | 382,910,131 | 346,107 | acb4a84edb36f924fd04990d6cc4a6dd |
| SRR22719003 | 647,887,565 | 558,941 | 60be29e3223eef5c3e1f2c8f1ac4d75a |
| SRR22719004 | 255,469,456 | 320,156 | 8a62439a411ee2b9df381af01c513719 |
| SRR22719005 | 335,459,261 | 419,128 | e28ee76c22835bfaa8d887ab510aa24b |
| SRR22719006 | 368,821,813 | 421,485 | 7899571b8bc321e9a26679a660eeacaa |
| SRR22719007 | 473,089,876 | 538,791 | ba50afe88780228b2cff7a06bc2556b4 |

All six empty historical `.DONE` markers exist. Metadata counts sum to **2,604,608**, not a fresh molecule/read count. The dataset directory contains those files/receipts, the log and filereport; no BAM/FLNC production output was present in this shallow listing. Source URL naming is a significant provenance discrepancy, not a substitute for molecular-product validation of the bytes.

- `pacbio/` immediate children: `Zmays_wang2020_zenodo2611319`, `Athaliana_zhang2023_PRJNA911826`, `Zmays_kinnex_hybrid_PRJNA1290227`, `Zmays_B73_ccs_PRJNA1470126`, `Zmays_wang2018_PRJEB22122`. No PRJEB77203 here.
- Explicit candidates `pacbio/Athaliana_PRJEB77203` and `hifi/Athaliana_PRJEB77203`: absent. `mvp_scoring_20260916`: absent at observation. This does not exclude other output namespaces.
- `longread_align_v1/Athaliana/` immediate children: `col0_stigma_PRJNA874115`, `inflorescence_PRJNA596358`, `cui2020_PRJNA594286`, `ck_control_PRJNA796315`, `seed_flcdna_PRJNA1155260`, `wt_only_PRJNA705300`, `col0_DRP009401`, `lncrna_survey_PRJNA765684`, `col0_nitrogen_PRJDB38182`, `buffer_control_PRJNA1393863`. None is FLIC, PRJNA911826 or PRJEB77203. We did not inspect their payloads or infer correctness from existence.

## Corrections and remaining work

Replace the broad policy TODO with:

> **Reconcile existing validation PacBio accession/product/provenance and any already-produced alignment; wire the already-approved MVP scoring and missing-structural safeguards. A-CCS23 acquisition is complete at the file/receipt level but CCS/FLNC product identity is unresolved. Named A-HiFi ERR13994458 was not found in the bounded canonical inventory. Do not redownload or substitute datasets before reconciling existing artifacts. A new author decision is needed only for an actual deviation from the approved source/decision contract, not because the implementation hardcodes PENDING.**

Implementation work can proceed without another broad HiFi-versus-ONT request: bind actual authorized runs and references, preserve per-source/union availability, separate descriptive ONT results from full MVP structural availability, and implement §5.3/§5.4 faithfully. The hardcoded always-reject policy path needs contract-driven implementation/review, not retroactive author authority. This note does not authorize code edits or remove actual resource/review gates.

Unknowns: whether ERR13994458 raw/processed data or a suitable validation alignment exists under another previously used host/path; whether a correctly identified CCS/FLNC product of PRJNA911826 exists elsewhere; and whether a separate exact author instruction explicitly permits substitution or ONT-only satisfaction of the full MVP STOP safeguard. **No such recorded instruction was located in this review.** That negative search result does not prove the user never gave it. Existing paths/receipts should be reconciled first; a question to the user, if still necessary, should identify the precise missing artifact or proposed rule change rather than repeat the discredited blanket pending-policy claim.

Scope: only this requested note was written. No source/TODO/protocol modification, Slurm action, acquisition, full scan/hash, or investigation of the separately owned extern issue. Advisor consultation failed due provider version incompatibility and contributed no review authority. Graft was consulted before searches; reported savings across map/ask were approximately 1,018,212 tokens.
