#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
echo "START $(date +%H:%M:%S)"
python extrude.py > extrude.log 2>&1
echo "END $(date +%H:%M:%S) exit=$?"
grep -aiE "bad|quality|error|Warning" extrude.log | tail -20 || true
echo "--- read extrude.log: bad cells must be 0, min quality about 0.9 ---"
