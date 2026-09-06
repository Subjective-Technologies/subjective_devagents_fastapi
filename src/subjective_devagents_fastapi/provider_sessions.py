"""Discovers which OS users look logged in to each agent CLI (Claude/Codex/Cursor).

Python port of clients/desktop/subjective_devagents/src/providerSessions.js. Only checks
that known credential files exist — never reads token contents.
"""
from __future__ import annotations

import os
from pathlib import Path

from .system_users import current_username, list_system_users

CREDENTIAL_MARKERS = {
    "claude": [
        ".claude/.credentials.json",
        ".claude/credentials.json",
        ".config/claude/.credentials.json",
        ".config/claude/credentials.json",
        ".config/claude-code/.credentials.json",
        ".local/share/claude/.credentials.json",
        ".claude.json",
    ],
    "codex": [
        ".codex/auth.json",
        ".codex/.auth.json",
    ],
    "cursor": [
        ".config/cursor/auth.json",
    ],
}


def _probe_marker(file_path: str) -> str:
    # Existence only — never read token contents. 0600 files owned by another
    # user fail R_OK even when the login file is present.
    try:
        if os.path.exists(file_path):
            return "present"
        # exists() returns False for both missing files and EACCES on a parent
        # directory; distinguish by probing the parent for search permission.
        parent = os.path.dirname(file_path)
        if os.path.isdir(parent) or not os.access(os.path.dirname(parent) or "/", os.X_OK):
            return "missing" if os.path.isdir(parent) else "unknown"
        return "missing"
    except PermissionError:
        return "unknown"
    except OSError:
        return "missing"


def probe_home(home: str) -> dict:
    providers = {}
    for provider, relatives in CREDENTIAL_MARKERS.items():
        states = [_probe_marker(str(Path(home) / relative)) for relative in relatives]
        providers[provider] = {
            "ready": "present" in states,
            "unknown": "present" not in states and "unknown" in states,
        }
    return providers


def list_provider_sessions() -> dict:
    me = current_username()
    listed = list_system_users(include_current=True)
    if not listed["ok"]:
        return {"ok": False, "currentUser": me, "users": [], "error": listed.get("error")}
    users = [
        {
            "username": user["username"],
            "home": user["home"],
            "current": user["username"] == me,
            "providers": probe_home(user["home"] or str(Path("/home") / user["username"])),
        }
        for user in listed["users"]
    ]
    return {"ok": True, "currentUser": me, "users": users}


def sessions_for_provider(listed: dict, provider: str) -> dict:
    key = str(provider or "").lower()
    users = [
        {
            "username": user["username"],
            "current": bool(user.get("current")),
            "ready": bool((user.get("providers") or {}).get(key, {}).get("ready")),
            "unknown": bool((user.get("providers") or {}).get(key, {}).get("unknown")),
        }
        for user in listed.get("users") or []
    ]
    users.sort(key=lambda user: (-int(user["ready"]), -int(user["unknown"]), user["username"]))
    return {
        "ok": listed.get("ok") is not False,
        "provider": key,
        "currentUser": listed.get("currentUser"),
        "users": users,
    }
