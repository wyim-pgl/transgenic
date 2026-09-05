#!/usr/bin/env bash
# Stable entry point, including the existing evidence/ symlink.
set -Eeuo pipefail
SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
exec "$(dirname "$SCRIPT")/longread_fetch_v2.sh" "$@"
