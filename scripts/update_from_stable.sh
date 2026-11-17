#!/usr/bin/env bash
# Bring this server's DevAgents to the newest TESTED release, then drain-restart it.
#
# CI moves the `stable` tag only after the pipeline's tests pass on main. Every company
# server (ours and each customer VPS) runs this from a systemd timer installed by
# company_setup.sh, so a release reaches the whole fleet without CI holding any server
# credentials.
#
# Safe by construction:
#   - fast-forward only: a server whose checkout diverged (local commits) is reported, not reset;
#   - local uncommitted edits are kept; if they conflict with the release, nothing changes;
#   - running agent turns finish first (scripts/drain_restart.sh); queued work waits;
#   - if the service is unhealthy after the restart, it goes back to the previous commit.
#
# Usage (root):  update_from_stable.sh [--check]
#   UPDATE_REF=stable   tag or branch to follow
#   STATUS_FILE=/var/lib/subjective-devagents/update.json   last outcome, for verify scripts
set -uo pipefail

API_DIR=${API_DIR:-/subjective/services/subjective_devagents_fastapi}
SERVICE=${SERVICE:-subjective-devagents-fastapi.service}
UPDATE_REF=${UPDATE_REF:-stable}
STATUS_FILE=${STATUS_FILE:-/var/lib/subjective-devagents/update.json}
CHECK_ONLY=${1:-}

log() { printf '%s %s\n' "$(date -u +%Y_%m_%d_%H_%M_%S)" "$*"; }

status() {
  local outcome="$1" detail="${2:-}"
  mkdir -p "$(dirname "$STATUS_FILE")"
  python3 - "$STATUS_FILE" "$outcome" "$detail" "${HEAD_SHA:-}" "${TARGET_SHA:-}" "$UPDATE_REF" <<'PY'
import json, sys, time
path, outcome, detail, head, target, ref = sys.argv[1:7]
json.dump({"outcome": outcome, "detail": detail, "head": head, "target": target, "ref": ref,
           "at": time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())}, open(path, "w"), indent=1)
PY
  log "$outcome${detail:+: $detail}"
}

[[ -d "$API_DIR/.git" || -f "$API_DIR/.git" ]] || { status failed "no checkout at $API_DIR"; exit 1; }
OWNER=$(stat -c '%U' "$API_DIR/.git")
as_owner() { if [[ "$(id -un)" == "$OWNER" ]]; then "$@"; else sudo -n -u "$OWNER" -H "$@"; fi; }
git_() { as_owner git -c safe.directory='*' -C "$API_DIR" "$@"; }

if ! git_ fetch -q --force origin "refs/tags/$UPDATE_REF:refs/tags/$UPDATE_REF" 2>/dev/null \
   && ! git_ fetch -q origin "$UPDATE_REF" 2>/dev/null; then
  status failed "cannot fetch $UPDATE_REF from origin"; exit 1
fi
HEAD_SHA=$(git_ rev-parse HEAD)
TARGET_SHA=$(git_ rev-parse "$UPDATE_REF^{commit}" 2>/dev/null || git_ rev-parse FETCH_HEAD)

if [[ "$HEAD_SHA" == "$TARGET_SHA" ]]; then
  status current "at ${TARGET_SHA:0:7}"; exit 0
fi
if git_ merge-base --is-ancestor "$TARGET_SHA" "$HEAD_SHA"; then
  status current "${HEAD_SHA:0:7} is already ahead of $UPDATE_REF ${TARGET_SHA:0:7}"; exit 0
fi
if ! git_ merge-base --is-ancestor "$HEAD_SHA" "$TARGET_SHA"; then
  status diverged "local ${HEAD_SHA:0:7} is not an ancestor of $UPDATE_REF ${TARGET_SHA:0:7}; not touching it"; exit 1
fi
if [[ "$CHECK_ONLY" == "--check" ]]; then
  status available "${HEAD_SHA:0:7} -> ${TARGET_SHA:0:7}"; exit 0
fi

log "updating ${HEAD_SHA:0:7} -> ${TARGET_SHA:0:7}"
if ! git_ merge -q --ff-only "$TARGET_SHA"; then
  status blocked "fast-forward refused (local edits conflict with the release); nothing changed"; exit 1
fi
git_ -c url.https://github.com/.insteadOf=git@github.com: submodule update -q --init --recursive \
  || log "warning: submodule update failed; continuing with the checked-out submodules"

if git_ diff --quiet "$HEAD_SHA" "$TARGET_SHA" -- pyproject.toml; then
  :
else
  log "pyproject.toml changed: reinstalling"
  as_owner "$API_DIR/.venv/bin/python" -m pip install -q -e "$API_DIR" || {
    git_ reset -q --keep "$HEAD_SHA"
    status failed "dependency install failed; kept ${HEAD_SHA:0:7}"; exit 1
  }
fi

healthy() {
  local url="${HEALTH_URL:-http://127.0.0.1:8080/healthz}" extra=()
  if [[ -z "${HEALTH_URL:-}" && -f /etc/subjective-devagents-fastapi/tls/fullchain.pem ]]; then url="https://127.0.0.1:8080/healthz"; extra=(-k); fi
  local i
  for i in $(seq 1 "${HEALTH_TRIES:-30}"); do
    curl -fsS --max-time 3 "${extra[@]}" "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if ! "${DRAIN_CMD:-$API_DIR/scripts/drain_restart.sh}" --force; then
  status failed "restart did not complete"; exit 1
fi
if ! healthy; then
  log "unhealthy after update: returning to ${HEAD_SHA:0:7}"
  git_ reset -q --keep "$HEAD_SHA"
  git_ -c url.https://github.com/.insteadOf=git@github.com: submodule update -q --init --recursive || true
  "${RESTART_CMD:-systemctl}" restart "$SERVICE" || true
  status rolled_back "service unhealthy on ${TARGET_SHA:0:7}; back on ${HEAD_SHA:0:7}"; exit 1
fi
status updated "${HEAD_SHA:0:7} -> ${TARGET_SHA:0:7}"
