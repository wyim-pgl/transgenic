# Controller panel transport decision

The read-only architecture investigation established the intended DeltaAI SIF package versions: Python3.12.3, NumPy2.1.0, DuckDB1.5.5, Torch2.14.0a0+4fdf77b940.nv26.08, Git2.43.0. No input/image hashes were renewed by that metadata call. DeltaAI is aarch64; Pronghorn observedhost isx86_64. No dedicated zero-GRESCPUpartition was established.

## Explicit proposed amendment, not a silent waiver

Prior scoped PANEL_REVIEW excluded copying historical inputs/sources. The newly proposed transport is a real change to that operating condition: named byte-identical replicas, not original GPFS files, would be staged on the existing owned DeltaAI allocation storage and exposed at original absolute paths by read-only bind mounts. Independent transport/mount/provenance review is required before any copy or execution. Canonical path equality alone is not proof of physical-source continuity.

No helper, package-version equality, original manifest, request, array, DB or source-byte acceptance check is to be weakened. Current and historical source files must remain separate. No symlinks, original rewrites, recomputation or receipt relabeling. Transferred inventory must exactly match15historicalinputs plus152publicationfiles. Reuse existing exact DeltaAI corpus/evidence; nonDBoriginal payload212152146B is the maximum candidate before deduplication, not a transfer already performed. Do not copy SIF/HF/checkpoints/DBs.

Current consumer uses the reviewed103-file snapshot at `/work/repo`, with five sparse original Git authority objects supporting the exact authority lookup. Historical sources remain at recorded GPFS-visible paths. Request bytes/pin unchanged; readonly mount for request, separate fresh external writable output/logs.

## Lifecycle and resources

P1 moves staged evidence. Final panel backing location must come from the actual reviewed P1 staging receipt; no searching partial runs, automatic fallback, or racing two owners. Therefore panel candidate preparation can run now, but panel execution must wait for a stable verified input lifecycle. This does not make panel completion a prerequisite for GPU P1.

Conditional panel ceiling: at most one node-hour CPU workload in an aarch64 GPU-backed diagnostic allocation, CUDA hidden/no --nv, inside the existing total five-node-hour diagnostic cap. No claim of zero GPU billing or dedicated CPU resources. This is not a submission, training authorization or measured performance guarantee. Fresh account/campaign/pending-reservation reconciliation and independent package/resource review are required. P1replacement ceiling remains0.5node-hour; other followups share the same cap.

Implementation owner e743b823-dfdb-4f9 prepares new transport_v1/metadata,validators,launcher/tests only. No remote transfer/job or existing-source changes authorized by that task. GPU package owner03f535de-21e3-460 continues independently.
