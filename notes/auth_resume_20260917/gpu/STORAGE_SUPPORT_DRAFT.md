# Unsent support draft: DeltaAI small-file read stalled

Status: DRAFT ONLY. No external message or support ticket has been sent.

On2026-09-17 around19:49–19:54CDT, my owned process1944304 on the DeltaAI login host became blocked while reading a newly staged5688-byte code file on /work/nvme/bilv. Metadata listing succeeds. The process is python3- from a small deployment-verification script, not a model workload or GPUallocation.

Observed stateI,wchancl_syn; aarch64syscall63(read),fd4points to:
/work/nvme/bilv/wcyim/transgenic/mvp_est8_20260916_1545/staging/dev123_p1_release_v6_20260917/build_release.py

A TERM request was made after recording the state. SIGKILL andSIGTERM remainpending; process termination is not established. No new GPUjob was submitted, and no model/DBfiles are being scanned by this process. The incomplete staging directory and all prior artifacts are preserved.

Please investigate client/server I/O or lock state for this path/process and advise safe recovery. Do not delete or modify the existing dataset, checkpoints or failed-job artifacts. We can provide the exactsmall-script/metadata/syscall observations if useful. We have not diagnosed the underlying filesystem cause and are not treating a shell timeout as successful termination.
