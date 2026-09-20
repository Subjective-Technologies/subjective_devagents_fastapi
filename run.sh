#!/usr/bin/env bash
#
# Subjective DevAgents FastAPI — datasource-backed backend.
#
# Binds 0.0.0.0:8080 so it does not collide with the web UI
# (subjective-chat-web :80, subjective-devagents-web :8787).
#
# Usage:
#   ./run.sh
#   SUBJECTIVE_CHAT_WEB_PORT=8090 ./run.sh
#   ./run.sh --reload
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${SUBJECTIVE_CHAT_WEB_HOST:-${BIND_HOST:-0.0.0.0}}"
PORT="${SUBJECTIVE_CHAT_WEB_PORT:-${WEB_PORT:-8080}}"

# Agent CLIs (codex/claude) live under nvm, not system PATH.
if [ -d "${HOME}/.nvm/versions/node" ]; then
  latest="$(ls -1d "${HOME}/.nvm/versions/node"/v* 2>/dev/null | sort -V | tail -1 || true)"
  if [ -n "${latest:-}" ] && [ -d "$latest/bin" ]; then
    PATH="$latest/bin:${PATH}"
  fi
fi
PATH="${HOME}/.local/bin:${PATH}"
export PATH

if [ ! -f "${SCRIPT_DIR}/plugins/subjective_codex_datasource/SubjectiveCodexDataSource.py" ]; then
  echo "### datasource submodules missing — git submodule update --init --recursive" >&2
  git -C "${SCRIPT_DIR}" submodule update --init --recursive
fi

# Browser UI is a sibling client in the monorepo. Checkout may still be named
# subjective_chat. Override with SUBJECTIVE_DEVAGENTS_CLIENT_ROOT if needed.
if [ -z "${SUBJECTIVE_DEVAGENTS_CLIENT_ROOT:-}" ]; then
  for candidate in \
    "${SCRIPT_DIR}/../../clients/desktop/subjective_devagents" \
    "${SCRIPT_DIR}/../../clients/desktop/subjective_chat"
  do
    if [ -d "${candidate}/web" ]; then
      SUBJECTIVE_DEVAGENTS_CLIENT_ROOT="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi
if [ -n "${SUBJECTIVE_DEVAGENTS_CLIENT_ROOT:-}" ]; then
  export SUBJECTIVE_DEVAGENTS_CLIENT_ROOT
fi

py_path="${SCRIPT_DIR}/src"
py_path="${py_path}:${SCRIPT_DIR}/libs/dependencies/subjective-abstract-data-source-package"
py_path="${py_path}:${SCRIPT_DIR}/libs/dependencies/brainboost_configuration_package"
py_path="${py_path}:${SCRIPT_DIR}/libs/dependencies/brainboost_data_source_logger_package"
export PYTHONPATH="${py_path}${PYTHONPATH:+:${PYTHONPATH}}"

venv="${SCRIPT_DIR}/.venv"
py="${venv}/bin/python"
if [ ! -x "${py}" ]; then
  # Never build from a conda interpreter: conda is not used in this project and
  # a conda-based venv breaks when conda is removed or its base env deactivated.
  base_py="${SUBJECTIVE_BASE_PYTHON:-/usr/bin/python3}"
  [ -x "${base_py}" ] || base_py="$(command -v python3)"
  echo "### creating venv in ${SCRIPT_DIR} (base: ${base_py})" >&2
  "${base_py}" -m venv "${venv}"
  "${py}" -m pip install -q -U pip setuptools wheel
  "${py}" -m pip install -q -e "${SCRIPT_DIR}" "requests>=2.25.1"
elif ! "${py}" -c "import uvicorn, fastapi" 2>/dev/null; then
  "${py}" -m pip install -q -e "${SCRIPT_DIR}" "requests>=2.25.1"
fi

# Secrets live in the monorepo's .env, never in this script or in a unit file. Read
# only the keys this service needs, and only when the environment has not already set
# them, so an explicit override or a systemd unit still wins.
env_file="${SUBJECTIVE_ENV_FILE:-}"
if [ -z "${env_file}" ]; then
  for candidate in "${SCRIPT_DIR}/../../.env" "/subjective/.env"; do
    if [ -f "${candidate}" ]; then env_file="${candidate}"; break; fi
  done
fi
if [ -n "${env_file:-}" ] && [ -f "${env_file}" ]; then
  for key in SUBJECTIVE_DEVAGENTS_API_TOKEN; do
    if [ -z "$(eval "printf '%s' \"\${${key}:-}\"")" ]; then
      # `cut -d= -f2-` keeps a value that itself contains '='.
      value="$(grep -m1 "^${key}=" "${env_file}" | cut -d= -f2-)"
      if [ -n "${value}" ]; then export "${key}=${value}"; fi
    fi
  done
  unset value
fi

if [ -z "${SUBJECTIVE_DEVAGENTS_API_TOKEN:-}" ]; then
  echo "### no SUBJECTIVE_DEVAGENTS_API_TOKEN — the API is open and the SSH terminal will refuse to start" >&2
fi

export SUBJECTIVE_CHAT_WEB_HOST="${HOST}"
export SUBJECTIVE_CHAT_WEB_PORT="${PORT}"
export SUBJECTIVE_CHAT_PYTHON="${SUBJECTIVE_CHAT_PYTHON:-${py}}"

echo "### DevAgents FastAPI (datasources) on http://${HOST}:${PORT}/"
if [ "${SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED:-}" = "1" ]; then
  echo "### ssh console: http://${HOST}:${PORT}/ssh"
  # The command, never the token itself: this script's output is routinely
  # redirected to a log file, and the secret must not end up in one.
  echo "### open it with the token already filled in:"
  echo "###   xdg-open \"http://${HOST}:${PORT}/ssh?token=\$(grep -m1 '^SUBJECTIVE_DEVAGENTS_API_TOKEN=' ${env_file:-.env} | cut -d= -f2-)\""
fi
echo "### plugins=${SCRIPT_DIR}/plugins"
if [ -n "${SUBJECTIVE_DEVAGENTS_CLIENT_ROOT:-}" ]; then
  echo "### client=${SUBJECTIVE_DEVAGENTS_CLIENT_ROOT}"
fi

exec "${py}" -m uvicorn subjective_devagents_fastapi.app:app \
  --app-dir "${SCRIPT_DIR}/src" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
