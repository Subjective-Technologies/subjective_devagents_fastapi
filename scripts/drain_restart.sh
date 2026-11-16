#!/usr/bin/env bash
# Restart DevAgents without cutting off running agent turns.
#
# 1. Mark the runtime as draining: new coordinated dispatches are refused with
#    runtime_capacity, which the chat journal keeps queued (visibly), never failed.
# 2. Wait until no turn is queued or running (at most DRAIN_MAX seconds).
# 3. Restart the service; a fresh process clears the drain flag and takes the queue.
#
# Usage (on the VPS, as a user that may sudo):  scripts/drain_restart.sh [--force]
#   DRAIN_MAX=1500  seconds to wait before giving up (or restarting anyway with --force)
set -euo pipefail
SERVICE=${SERVICE:-subjective-devagents-fastapi.service}
RUN_AS=${RUN_AS:-subjective}
STATE_DIR=${STATE_DIR:-/home/subjective/.subjective_chat}
DRAIN_MAX=${DRAIN_MAX:-1500}   # below the chat journal's 1800 s capacity wait
FORCE=${1:-}

busy() {
  sudo -n -u "$RUN_AS" python3 - "$STATE_DIR/language_models.sqlite3" <<'PY'
import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(db.execute("SELECT count(*) FROM turns WHERE status IN ('queued','running')").fetchone()[0])
PY
}

sudo -n -u "$RUN_AS" touch "$STATE_DIR/DRAINING"
echo "draining $SERVICE: new coordinated work stays queued in the chat journal"
started=$(date +%s)
while :; do
  n=$(busy)
  [ "$n" = "0" ] && break
  waited=$(( $(date +%s) - started ))
  if [ "$waited" -ge "$DRAIN_MAX" ]; then
    if [ "$FORCE" = "--force" ]; then
      echo "still $n running turn(s) after ${waited}s; restarting anyway (their successors continue from checkpoints)"
      break
    fi
    sudo -n -u "$RUN_AS" rm -f "$STATE_DIR/DRAINING"
    echo "still $n running turn(s) after ${waited}s; drain cancelled, service not restarted (use --force)" >&2
    exit 1
  fi
  echo "waiting for $n running turn(s)... ${waited}s"
  sleep 15
done
sudo -n systemctl restart "$SERVICE"
sleep 5
systemctl is-active "$SERVICE"
