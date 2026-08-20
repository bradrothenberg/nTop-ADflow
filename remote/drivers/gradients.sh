#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
echo "START $(date +%H:%M:%S) on $NP ranks"
run_mpi gradients.py > gradients.log 2>&1
echo "END $(date +%H:%M:%S) exit=$?"
grep -aE "^(===|   )" gradients.log | grep -avE "KSP Residual" | tail -40 || true
