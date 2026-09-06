"""Enumerates real Linux user accounts so the "Continue as..." menu can offer them.

Python port of clients/desktop/subjective_devagents/src/systemUsers.js. Linux-only
(relies on `getent passwd` / /etc/passwd); on any other platform list_system_users()
reports itself unsupported rather than guessing.
"""
from __future__ import annotations

import getpass
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

NOLOGIN_SHELLS = {"/usr/sbin/nologin", "/sbin/nologin", "/bin/false", "/usr/bin/false", ""}
MIN_UID = 1000
MAX_UID = 60000  # excludes `nobody` (65534) and other high system/service UIDs
USERNAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


def is_valid_username(username: Any) -> bool:
    return isinstance(username, str) and bool(USERNAME_PATTERN.match(username))


def current_username() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        import os

        return str(os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()


@dataclass
class PasswdEntry:
    username: str
    uid: int
    gecos: str
    home: str
    shell: str


def parse_passwd_line(line: str) -> PasswdEntry | None:
    fields = str(line or "").split(":")
    if len(fields) < 7:
        return None
    username, _, uid_raw, _, gecos, home, shell = fields[:7]
    try:
        uid = int(uid_raw)
    except (TypeError, ValueError):
        return None
    if not username:
        return None
    return PasswdEntry(username=username, uid=uid, gecos=gecos or "", home=home or "", shell=(shell or "").strip())


def parse_passwd_text(text: str) -> list[PasswdEntry]:
    entries = []
    for line in str(text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        entry = parse_passwd_line(line)
        if entry:
            entries.append(entry)
    return entries


def filter_human_users(
    entries: list[PasswdEntry], exclude_username: str = "", include_excluded: bool = False
) -> list[PasswdEntry]:
    kept = [
        entry
        for entry in entries
        if MIN_UID <= entry.uid <= MAX_UID
        and entry.shell not in NOLOGIN_SHELLS
        and (include_excluded or entry.username != exclude_username)
    ]
    return sorted(kept, key=lambda entry: entry.username)


def _read_passwd_text() -> dict:
    # `getent passwd` resolves NSS-backed accounts too, not just local /etc/passwd.
    try:
        completed = subprocess.run(["getent", "passwd"], capture_output=True, text=True, timeout=10)
        if completed.returncode == 0 and completed.stdout.strip():
            return {"ok": True, "text": completed.stdout}
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        with open("/etc/passwd", encoding="utf-8") as handle:
            return {"ok": True, "text": handle.read()}
    except OSError as error:
        return {"ok": False, "error": f"Could not list system users: {error}"}


def list_system_users(include_current: bool = False, exclude_username: str = "") -> dict:
    if sys.platform != "linux":
        return {"ok": False, "users": [], "error": "Switching OS users is only supported on Linux."}
    me = current_username()
    exclude = "" if include_current else (exclude_username or me)
    read = _read_passwd_text()
    if not read["ok"]:
        return {"ok": False, "users": [], "error": read["error"]}
    entries = parse_passwd_text(read["text"])
    users = [
        {
            "username": entry.username,
            "uid": entry.uid,
            "home": entry.home,
            "gecos": entry.gecos,
            "current": entry.username == me,
        }
        for entry in filter_human_users(entries, exclude_username=exclude, include_excluded=include_current)
    ]
    return {"ok": True, "users": users}
