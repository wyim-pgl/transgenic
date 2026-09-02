#!/usr/bin/env bash
# Build the TransGenic NGC-based image. Run on a login node (or any host with apptainer >= 1.2).
set -euo pipefail
NGC_TAG=${NGC_TAG:-25.06}                       # pin the tag actually used; record it in issue #18
OUT=${OUT:-$SCRATCH/containers/transgenic-ngc.sif}
mkdir -p "$(dirname "$OUT")" "${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}"
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}
apptainer build --build-arg NGC_TAG="$NGC_TAG" "$OUT" "$(dirname "$0")/transgenic.def"
apptainer exec --nv "$OUT" cat /opt/transgenic/IMAGE_INFO
sha256sum "$OUT" | tee "$OUT.sha256"
