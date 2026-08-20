#!/bin/bash
# Push a built case to the solver host: the surface, the profile table, the case snapshot,
# the remote scripts, and the package modules those scripts import.
#
# Usage: scripts/push.sh <case-name>
set -euo pipefail

CASE="${1:?usage: scripts/push.sh <case-name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/work/$CASE"
[ -f "$WORK/case.json" ] || { echo "no built case at $WORK; run scripts/build_case.py first"; exit 1; }

# Windows Python cannot read a /d/... MSYS path, so convert when cygpath exists.
winpath () { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
CASE_JSON="$(winpath "$WORK/case.json")"

# Host and remote dir come out of the snapshot, so they are configured in exactly one place.
HOST="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['remote']['host'])" "$CASE_JSON")"
RDIR="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['remote']['dir'])" "$CASE_JSON")"
[ -n "$HOST" ] && [ -n "$RDIR" ] || { echo "could not read remote.host / remote.dir from case.json"; exit 1; }

echo "pushing $CASE to $HOST:$RDIR"
# Do NOT single-quote: the remote shell must expand a leading ~ in remote.dir.
ssh "$HOST" "mkdir -p $RDIR/diagnostics"
scp -q "$WORK/surface.fmt" "$WORK/profile.npz" "$WORK/case.json" "$HOST:$RDIR/"
scp -q "$ROOT"/remote/*.py "$HOST:$RDIR/"
if compgen -G "$ROOT/remote/diagnostics/*.py" >/dev/null; then
  scp -q "$ROOT"/remote/diagnostics/*.py "$HOST:$RDIR/diagnostics/"
fi
scp -q "$ROOT"/remote/drivers/*.sh "$HOST:$RDIR/"
# The remote scripts import these directly, so there is only ever one copy of each.
scp -q "$ROOT"/ntop_adflow/config.py "$ROOT"/ntop_adflow/surface.py \
       "$ROOT"/ntop_adflow/plot3d.py "$HOST:$RDIR/"
ssh "$HOST" "chmod +x $RDIR/*.sh"

cat <<EOT
done. On the host:
  cd $RDIR
  ./extrude.sh      # surface -> volume.cgns; bad cells MUST be 0
  ./solve.sh        # flow solution -> results.json
  ./smoke.sh        # 2 min gradient sanity check on the same code path
  ./gradients.sh    # full adjoint gradients + K -> gradients.json
EOT
