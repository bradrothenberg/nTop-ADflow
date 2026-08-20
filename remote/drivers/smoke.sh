#!/bin/bash
# Fast sanity check on the SAME code path as gradients.sh: loose tolerance, one design
# variable, no finite difference. Under 2 min. Run this after any change.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
export RB_L2=1e-5 RB_NOFD=1
DV="$(python -c "import json;print(json.load(open('case.json'))['gradients']['design_vars'][-1])")"
export RB_DVS="$DV"
echo "START $(date +%H:%M:%S)  dv=$DV  loose tolerance, no FD"
run_mpi gradients.py > smoke.log 2>&1
echo "END $(date +%H:%M:%S) exit=$?"
grep -aE "^(===|   )" smoke.log | grep -avE "KSP Residual" | tail -25 || true
