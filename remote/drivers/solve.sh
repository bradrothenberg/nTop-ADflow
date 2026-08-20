#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
echo "START $(date +%H:%M:%S) on $NP ranks"
run_mpi solve.py > solve.log 2>&1
echo "END $(date +%H:%M:%S) exit=$?"
grep -aE "^   (cl|cd|cmy|NOTE)|solve time|liftindex" solve.log | tail -10 || true
