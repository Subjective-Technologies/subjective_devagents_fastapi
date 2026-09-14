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
  echo "### creating venv in ${SCRIPT_DIR}" >&2
  python3 -m venv "${venv}"
  "${py}" -m pip install -q -U pip setuptools wheel
  "${py}" -m pip install -q -e "${SCRIPT_DIR}" "requests>=2.25.1"
elif ! "${py}" -c "import uvicorn, fastapi" 2>/dev/null; then
  "${py}" -m pip install -q -e "${SCRIPT_DIR}" "requests>=2.25.1"
fi

export SUBJECTIVE_CHAT_WEB_HOST="${HOST}"
export SUBJECTIVE_CHAT_WEB_PORT="${PORT}"
export SUBJECTIVE_CHAT_PYTHON="${SUBJECTIVE_CHAT_PYTHON:-${py}}"

echo "### DevAgents FastAPI (datasources) on http://${HOST}:${PORT}/"
echo "### plugins=${SCRIPT_DIR}/plugins"
if [ -n "${SUBJECTIVE_DEVAGENTS_CLIENT_ROOT:-}" ]; then
  echo "### client=${SUBJECTIVE_DEVAGENTS_CLIENT_ROOT}"
fi

exec "${py}" -m uvicorn subjective_devagents_fastapi.app:app \
  --app-dir "${SCRIPT_DIR}/src" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
