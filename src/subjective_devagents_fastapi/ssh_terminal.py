"""Terminal sessions over SSH, driven from the browser through one WebSocket.

The SSH datasource answers a pipeline turn: one command in, one result out. That is
the wrong shape for a terminal, where the operator expects a connection that stays up,
state that persists between commands, and output that appears while the command is
still running. `SubjectiveSshDataSource.open_shell()` provides that shape; this module
is the service-side half — the interlocks, the accepted parameters, and the bookkeeping
for the live sessions.

It is, plainly, remote code execution on a *third* machine, so it follows the same
interlocks as `shell_exec`:

1. **Off unless asked.** `SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED=1`.
2. **Never without authentication.** Enabling it on a service with no token configured
   is refused rather than warned about.
3. **Bounded.** A cap on concurrent sessions, and an idle timeout, so a forgotten tab
   cannot hold an SSH connection open forever.

Credentials come from the operator's own machine by default: naming a `Host` alias from
`~/.ssh/config` is enough, and then no secret passes through this service at all. Keys
and passwords may be supplied per session instead; they are handed to the datasource,
which keeps them out of argv, logs and results.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from . import auth, datasources

# Connection fields a browser may set. Everything else in the datasource's schema is
# either derived here or deliberately not remote-controllable.
ALLOWED_CONNECTION_KEYS = frozenset({
    "host",
    "port",
    "username",
    "auth_method",
    "password",
    "private_key_path",
    "private_key_content",
    "key_passphrase",
    "known_hosts_path",
    "host_key_fingerprint",
    "strict_host_key_checking",
    "working_directory",
    "shell",
    "login_shell",
    "connect_timeout",
    "term",
})
SECRET_KEYS = frozenset({"password", "private_key_content", "key_passphrase"})

DEFAULT_MAX_SESSIONS = 8
DEFAULT_IDLE_TIMEOUT_SECONDS = 1800
MAX_COMMAND_BYTES = 100_000
# A terminal's keystrokes arrive as small frames; anything this large is a paste of a
# size no shell input line has any business being.
MAX_INPUT_BYTES = 1_000_000
# Beyond these a browser is not rendering a terminal, it is sending nonsense.
MAX_COLS = 500
MAX_ROWS = 300


class TerminalDisabled(RuntimeError):
    """Raised when the endpoint is switched off, or unsafe to switch on."""


class TerminalBusy(RuntimeError):
    """Raised when the concurrent-session cap is already reached."""


def terminal_enabled() -> bool:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED") or "").strip() == "1"


def max_sessions() -> int:
    try:
        return max(1, int(os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_MAX_SESSIONS") or DEFAULT_MAX_SESSIONS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_SESSIONS


def idle_timeout_seconds() -> int:
    try:
        return max(
            30,
            int(os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_IDLE_TIMEOUT") or DEFAULT_IDLE_TIMEOUT_SECONDS),
        )
    except (TypeError, ValueError):
        return DEFAULT_IDLE_TIMEOUT_SECONDS


def ensure_available() -> None:
    """Fail closed, and say which interlock stopped it."""
    if not terminal_enabled():
        raise TerminalDisabled(
            "The SSH terminal is disabled. Set SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED=1 "
            "to enable it."
        )
    if not auth.auth_enabled():
        raise TerminalDisabled(
            "Refusing to enable the SSH terminal on an unauthenticated service: set "
            "SUBJECTIVE_DEVAGENTS_API_TOKEN first, or anyone who finds this host gets a "
            "shell on every machine it can reach."
        )


def connection_from_request(payload: dict | None) -> dict:
    """Take only the fields a browser is allowed to set, and require a host."""
    payload = payload or {}
    connection = {
        key: value
        for key, value in payload.items()
        if key in ALLOWED_CONNECTION_KEYS and value not in (None, "")
    }
    if not str(connection.get("host") or "").strip():
        raise ValueError("A host is required: a hostname, an address, or a ~/.ssh/config alias.")
    return connection


def describe(connection: dict) -> dict:
    """A log- and UI-safe view of a connection: never the secrets."""
    return {
        key: value
        for key, value in connection.items()
        if key not in SECRET_KEYS
    }


class TerminalSessions:
    """The live sessions, so they can be capped, reported and reaped."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}

    def register(self, session_id: str, session: Any, connection: dict) -> None:
        with self._lock:
            if len(self._sessions) >= max_sessions():
                raise TerminalBusy(
                    f"Too many open SSH sessions ({len(self._sessions)}). Close one and retry."
                )
            self._sessions[session_id] = {
                "session": session,
                "connection": describe(connection),
                "opened_at": time.time(),
                "last_active": time.time(),
            }

    def touch(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry["last_active"] = time.time()

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def idle_seconds(self, session_id: str) -> float:
        with self._lock:
            entry = self._sessions.get(session_id)
            return time.time() - entry["last_active"] if entry else 0.0

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": session_id,
                    "connection": entry["connection"],
                    "openedAt": entry["opened_at"],
                    "lastActive": entry["last_active"],
                    "alive": bool(getattr(entry["session"], "alive", False)),
                }
                for session_id, entry in self._sessions.items()
            ]

    def close_all(self) -> None:
        with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()
        for entry in entries:
            try:
                entry["session"].close()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                pass


sessions = TerminalSessions()


def clamp_size(cols, rows) -> tuple[int, int]:
    """A window size the terminal can actually be set to."""
    def bounded(value, default, ceiling):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, min(number, ceiling))

    return bounded(cols, 120, MAX_COLS), bounded(rows, 32, MAX_ROWS)


def open_session(connection: dict, on_event=None, *, pty: bool = False, cols=0, rows=0):
    """Build the datasource and open a live session on the named machine.

    Two shapes, chosen by the caller:

    - **command** (`pty=False`) — framed commands with exit codes, a tracked working
      directory and separated streams. What a UI or an agent can parse.
    - **pty** (`pty=True`) — a real terminal. `claude` and `codex` with no arguments,
      `top`, `vim` and `sudo` prompts work, Ctrl-C reaches the remote job, and in
      exchange there is no framing at all: only a byte stream for a terminal
      emulator to render.

    Raises LookupError when the plugin or the ssh client is missing or the connection
    is incomplete, and ValueError when a pinned host key does not match.
    """
    ensure_available()
    datasource_class = datasources.load("ssh")
    instance = datasources.instantiate(datasource_class, connection)
    name = "open_pty" if pty else "open_shell"
    opener = getattr(instance, name, None)
    if not callable(opener):
        raise LookupError(
            f"The installed SSH datasource has no {name}(); update the plugin to a "
            "version that supports this session type."
        )
    if pty:
        width, height = clamp_size(cols, rows)
        return instance, opener(cols=width, rows=height, on_event=on_event)
    return instance, opener(on_event=on_event)
