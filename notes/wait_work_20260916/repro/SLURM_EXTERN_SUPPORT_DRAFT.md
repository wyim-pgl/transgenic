> 🧊 **미발송 종료 (2026-09-21 저자결정 "extern OOM 은 그냥 무시한다").**
> 이 draft는 사이트 지원팀에 보내지 않는다. 기록으로만 보존하며, 내용은 발송되지 않은 초안이다.
> 정본: `notes/auth_restored_20260921/EXTERN_OOM_WAIVED_20260921.md`.

# Draft site-support request — not sent

Subject: Repeated extern OUT_OF_MEMORY 0:125 on successful CPU allocations

Please help distinguish an actual extern-process OOM from a Slurm/cgroup lifecycle or accounting issue on pronghorn. We have not changed site settings or retried completed workloads to suppress this status.

Observed examples:
- 6165460 and 6165462: tiny DIAMOND fixtures, allocation/batch COMPLETED 0:0, approximately 1–2 seconds; extern OUT_OF_MEMORY 0:125. Sampled extern RSS ~1480K.
- 6165555: 10,000-read ONT orientation diagnostic, allocation/batch COMPLETED 0:0, 2m28s; extern OUT_OF_MEMORY 0:125. Payload produced complete, hash-consistent outputs. Minimap2 reports peak RSS 1.732GB; Slurm batch MaxRSS is 51000K, so we do not assume accounting sampled the true peak.
- 6165525: successful artifact audit, allocation/batch COMPLETED 0:0, extern OUT_OF_MEMORY 0:125; UTC 2026-09-17 02:07:28–04:56:42, node cpu-50.
- Earlier fullDB6164118 on cpu-51 shows the same extern anomaly. It is not confined to a single workload or node.

Previously observed site configuration (not presumed current): Slurm20.11.7, proctrack/cgroup, task/affinity,task/cgroup, PrologFlags=Alloc,Contain, jobacct_gather/linux at30s, NoOverMemoryKill, ConstrainRAMSpace/ConstrainSwapSpace enabled. These facts do not establish a cause; short sampling and low RSS do not rule out a transient extern victim.

Requested evidence for these exact allocations:
1. slurmd/slurmstepd and kernel OOM records identifying job, step, cgroup, victim PID and time.
2. Effective extern memory/swap limits and memory events/failcnt at teardown.
3. Adopted process membership and executed prolog/epilog children and exit statuses.
4. Whether any known issue in this installed version produces extern0:125 despite successful batch completion, and how to verify it without rerunning scientific payloads.

Please preserve the distinction between allocation/batch success and extern failure. We currently retain clean_accounting=false and have not declared the issue harmless or resolved.

## Local evidence pointers

- `notes/EST_AUDIT_OPS_20260916_1530.md` and its command captures.
- `notes/EST_C1_FINAL_RESULT_REVIEW_20260916.md`.
- `notes/wait_work_20260916/longread/RESULT_REVIEW.md` and `result_review/remote_stat_accounting.txt`.

This is a draft only. No message, ticket, logs or biological data were sent externally. Native CPU pytest SIGSEGV is a separate local issue and is not attributed to this Slurm anomaly.
