"""Manages one persistent, `su`'d-into pty per "switched-to" OS user.

Python port of clients/desktop/subjective_devagents/src/suSessionManager.js. Nothing here
ever collects or types an OS password — the pty is only ever driven interactively, by
whoever is at the browser, through the terminal panel (pty input arrives over the API).

State machine per username:
  idle -> connecting -> awaiting-password -> ready
                                           -> auth-failed (retry via ensure_session)
  ready -> disconnected (explicit disconnect, or the shell died) / error

"ready" means the su'd shell itself is authenticated — NOT that codex/claude is signed in
for that user. That is checked lazily via ensure_provider_login(), which reuses the same
visible pty for the OAuth device flow only if needed.

Once ready, one-shot pipeline calls for that user are routed through a small local Unix
socket bridge (pipeline_bridge_server.py) started as a background child of the
authenticated shell — see run_in_session().

All methods are blocking; call them from a worker thread (asyncio.to_thread).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import shlex
import socket
import struct
import subprocess
import termios
import threading
import time
from pathlib import Path
from typing import Callable

from .system_users import is_valid_username

STATUS_IDLE = "idle"
STATUS_CONNECTING = "connecting"
STATUS_AWAITING_PASSWORD = "awaiting-password"
STATUS_AUTH_FAILED = "auth-failed"
STATUS_READY = "ready"
STATUS_ERROR = "error"
STATUS_DISCONNECTED = "disconnected"

PROVIDER_CHECKING = "checking-login"
PROVIDER_AWAITING_OAUTH = "awaiting-oauth"
PROVIDER_READY = "ready"
PROVIDER_ERROR = "error"

BUFFER_TAIL_KEEP = 16384
BUFFER_TRIM_THRESHOLD = 262144
PASSWORD_PROMPT_RE = re.compile(r"[Pp]assword:\s*$")
AUTH_FAILURE_RE = re.compile(r"su:\s*Authentication failure|incorrect password", re.IGNORECASE)

LOGIN_COMMANDS = {
    "codex": {"status": "codex login status", "login": "codex login", "which": "codex"},
    "claude": {"status": "claude auth status", "login": "claude auth login", "which": "claude"},
    "cursor": {"status": "cursor-agent status", "login": "cursor-agent login", "which": "cursor-agent"},
}
LOGIN_SUCCESS_RE = re.compile(r"logged in|login successful|authentication successful|already logged in", re.IGNORECASE)

_HEADER = struct.Struct(">I")


class _Session:
    def __init__(self, username: str, viewers: set | None = None):
        self.username = username
        self.master_fd: int | None = None
        self.process: subprocess.Popen | None = None
        self.status = STATUS_IDLE
        self.last_error = ""
        self.buffer = ""
        self.buffer_start = 0
        self.total_written = 0
        self.condition = threading.Condition()
        self.password_line_submitted = False
        self.viewers: set[Callable[[str], None]] = viewers or set()
        self.exited = False
        self.exit_error = ""
        self.queue_lock = threading.Lock()  # serializes run_in_session per user
        self.connect_future: threading.Event | None = None
        self.connect_result: dict | None = None
        self.connect_error: Exception | None = None
        self.provider_ready: dict[str, bool] = {}
        self.provider_login_locks: dict[str, threading.Lock] = {}
        self.bridge: dict | None = None
        self.bridge_lock = threading.Lock()


class SuSessionManager:
    def __init__(
        self,
        bridge_server_path: str | None = None,
        python_path: str = "python3",
        tmp_dir: str = "/tmp",
        log: Callable[[str, str, dict], None] | None = None,
    ):
        self.sessions: dict[str, _Session] = {}
        self.pending_viewers: dict[str, set] = {}
        self._lock = threading.Lock()
        self.bridge_server_path = bridge_server_path or str(Path(__file__).parent / "pipeline_bridge_server.py")
        self.python_path = python_path
        # Deliberately NOT tempfile.gettempdir(): that can resolve to a private per-user
        # XDG_RUNTIME_DIR the su'd-into user cannot write. /tmp (mode 1777) is the one
        # location every local user can create a socket in; the random suffix plus the
        # shared-secret handshake gate access, not directory permissions.
        self.tmp_dir = tmp_dir
        self.log = log or (lambda level, message, details=None: None)
        # Event callbacks wired by ChatCore.
        self.on_status: Callable[[dict], None] = lambda event: None
        self.on_provider_status: Callable[[dict], None] = lambda event: None

    # --- public API -----------------------------------------------------------------

    def get_status(self, username: str) -> dict:
        session = self.sessions.get(username)
        if not session:
            return {"username": username, "status": STATUS_IDLE}
        return {"username": username, "status": session.status, "error": session.last_error}

    def list_statuses(self) -> list[dict]:
        return [
            {"username": session.username, "status": session.status, "error": session.last_error}
            for session in list(self.sessions.values())
        ]

    def is_provider_ready(self, username: str, provider: str) -> bool:
        session = self.sessions.get(username)
        return bool(session and session.status == STATUS_READY and session.provider_ready.get(provider) is True)

    def attach_viewer(self, username: str, on_data: Callable[[str], None]) -> None:
        with self._lock:
            session = self.sessions.get(username)
            if session:
                session.viewers.add(on_data)
                return
            self.pending_viewers.setdefault(username, set()).add(on_data)

    def detach_viewer(self, username: str, on_data: Callable[[str], None]) -> None:
        with self._lock:
            pending = self.pending_viewers.get(username)
            if pending:
                pending.discard(on_data)
                if not pending:
                    self.pending_viewers.pop(username, None)
            session = self.sessions.get(username)
            if session:
                session.viewers.discard(on_data)

    def write_input(self, username: str, data: str) -> None:
        session = self.sessions.get(username)
        if not session or session.master_fd is None:
            return
        try:
            os.write(session.master_fd, data.encode("utf-8", "ignore"))
        except OSError:
            return
        # Track whether the human actually submitted a line while a password prompt is
        # showing, so the auth-confirmation probe never races ahead of them.
        if session.status == STATUS_AWAITING_PASSWORD and re.search(r"[\r\n]", data):
            with session.condition:
                session.password_line_submitted = True
                session.condition.notify_all()

    def resize(self, username: str, cols: int, rows: int) -> None:
        session = self.sessions.get(username)
        if session and session.master_fd is not None and cols > 0 and rows > 0:
            try:
                packed = struct.pack("HHHH", max(rows, 1), max(cols, 1), 0, 0)
                fcntl.ioctl(session.master_fd, termios.TIOCSWINSZ, packed)
            except OSError:
                pass

    def ensure_session(self, username: str) -> dict:
        """Spawns `su - username` and drives it to an authenticated, ready shell.

        Safe to call repeatedly: a session that's already connecting/ready is reused.
        Blocks until ready or failed.
        """
        if not is_valid_username(username):
            raise ValueError(f"Invalid username: {username}")
        with self._lock:
            existing = self.sessions.get(username)
            if existing:
                if existing.status == STATUS_READY:
                    return {"status": STATUS_READY}
                if existing.connect_future and existing.status not in (
                    STATUS_AUTH_FAILED,
                    STATUS_ERROR,
                    STATUS_DISCONNECTED,
                ):
                    waiting = existing
                else:
                    waiting = None
            else:
                waiting = None
            if waiting is None:
                if existing:
                    self._disconnect_locked(username)
                session = _Session(username, viewers=self.pending_viewers.pop(username, None))
                session.connect_future = threading.Event()
                self.sessions[username] = session
                threading.Thread(target=self._connect_flow_thread, args=(session,), daemon=True).start()
                waiting = session
        waiting.connect_future.wait()
        if waiting.connect_error:
            raise waiting.connect_error
        return waiting.connect_result or {"status": waiting.status}

    def ensure_provider_login(self, username: str, provider: str) -> dict:
        session = self.sessions.get(username)
        if not session or session.status != STATUS_READY:
            raise RuntimeError(f"No ready shell session for {username}. Call ensure_session first.")
        if session.provider_ready.get(provider) is True:
            return {"status": PROVIDER_READY}
        commands = LOGIN_COMMANDS.get(provider)
        if not commands:
            raise ValueError(f"Unknown provider: {provider}")
        lock = session.provider_login_locks.setdefault(provider, threading.Lock())
        with lock:
            if session.provider_ready.get(provider) is True:
                return {"status": PROVIDER_READY}
            try:
                return self._run_provider_login(session, provider, commands)
            except Exception as error:
                session.provider_ready[provider] = False
                self._emit_provider(session, provider, PROVIDER_ERROR, str(error))
                raise

    def run_in_session(self, username: str, argv: list[str], stdin_payload: str, timeout_ms: int = 660000) -> dict:
        """Routes a one-shot pipeline invocation through this user's authenticated bridge.

        Calls for the same username are serialized so concurrent messages never interleave.
        """
        session = self.sessions.get(username)
        if not session or session.status != STATUS_READY:
            raise RuntimeError(f"No ready session for {username}. Reconnect as {username} first.")
        with session.queue_lock:
            return self._run_in_session_once(session, argv, stdin_payload, timeout_ms)

    def disconnect(self, username: str) -> None:
        with self._lock:
            self._disconnect_locked(username)

    def disconnect_all(self) -> None:
        with self._lock:
            for username in list(self.sessions.keys()):
                self._disconnect_locked(username)

    # --- internals --------------------------------------------------------------------

    def _disconnect_locked(self, username: str) -> None:
        session = self.sessions.pop(username, None)
        if not session:
            return
        with session.condition:
            session.condition.notify_all()

        def cleanup():
            # Sequence bridge cleanup BEFORE killing the pty: /tmp's sticky bit means only
            # the target user (a command through their own shell) can remove their socket
            # file, so cleanup must happen while that shell is still alive.
            try:
                self._shutdown_bridge(session)
            except Exception:  # noqa: BLE001
                pass
            if session.process:
                try:
                    session.process.kill()
                except OSError:
                    pass
            if session.master_fd is not None:
                try:
                    os.close(session.master_fd)
                except OSError:
                    pass

        threading.Thread(target=cleanup, daemon=True).start()
        self._set_status(session, STATUS_DISCONNECTED)

    def _su_spawn_env(self) -> dict:
        # Login `su -` must not inherit this process's HOME/USER/XDG — otherwise
        # cursor-agent writes sandbox-policies into the chat process owner's ~/.cursor
        # while running as a different uid, and that owner then gets EACCES.
        return {
            "TERM": os.environ.get("TERM") or "xterm-256color",
            "LANG": os.environ.get("LANG") or "C.UTF-8",
            "LC_ALL": os.environ.get("LC_ALL") or os.environ.get("LANG") or "C.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "DISPLAY": os.environ.get("DISPLAY") or "",
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY") or "",
            "XAUTHORITY": os.environ.get("XAUTHORITY") or "",
        }

    def _set_status(self, session: _Session, status: str, error: str = "") -> None:
        session.status = status
        session.last_error = error
        try:
            self.on_status({"username": session.username, "status": status, "error": error})
        except Exception:  # noqa: BLE001
            pass

    def _emit_provider(self, session: _Session, provider: str, status: str, error: str = "") -> None:
        try:
            self.on_provider_status({"username": session.username, "provider": provider, "status": status, "error": error})
        except Exception:  # noqa: BLE001
            pass

    def _connect_flow_thread(self, session: _Session) -> None:
        try:
            session.connect_result = self._run_connect_flow(session)
        except Exception as error:  # noqa: BLE001
            if session.status != STATUS_AUTH_FAILED:
                self._set_status(session, STATUS_ERROR, str(error))
            session.connect_error = error
        finally:
            session.connect_future.set()

    def _run_connect_flow(self, session: _Session) -> dict:
        self._set_status(session, STATUS_CONNECTING)
        master_fd, slave_fd = os.openpty()
        packed = struct.pack("HHHH", 30, 100, 0, 0)
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, packed)
        except OSError:
            pass
        process = subprocess.Popen(
            ["su", "-", session.username],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            # NOT the target user's home: subprocess applies cwd= in the forked child
            # before exec, so that chdir runs as the server account and dies with EACCES on
            # any normal drwx------ home. `su -` is a login shell and enters the home
            # itself, after the uid switch - the only point where it is readable.
            cwd="/",
            env=self._su_spawn_env(),
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        session.master_fd = master_fd
        session.process = process
        threading.Thread(target=self._reader_thread, args=(session,), daemon=True).start()

        prompt_seen = False
        try:
            self._wait_for_pattern(session, PASSWORD_PROMPT_RE, timeout=15.0)
            prompt_seen = True
            self._set_status(session, STATUS_AWAITING_PASSWORD)
        except TimeoutError:
            # No password prompt within 15s — unusual (e.g. passwordless trust); fall
            # through to the auth-result poll, which resolves either way.
            pass

        if prompt_seen:
            # Wait for the human to actually submit a line before probing anything —
            # writing into the pty earlier would corrupt the password su is reading.
            if not self._wait_for_flag(session, timeout=120.0):
                message = "Timed out waiting for password submission."
                self._set_status(session, STATUS_AUTH_FAILED, message)
                raise TimeoutError(message)

        result = self._poll_auth_result(session)
        if not result["success"]:
            message = result.get("error") or "su did not authenticate."
            self._set_status(session, STATUS_AUTH_FAILED, message)
            raise RuntimeError(message)
        self._set_status(session, STATUS_READY)
        return {"status": STATUS_READY}

    def _reader_thread(self, session: _Session) -> None:
        fd = session.master_fd
        while True:
            try:
                chunk = os.read(fd, 8192)
            except OSError:
                chunk = b""
            if not chunk:
                break
            text = chunk.decode("utf-8", "replace")
            with session.condition:
                session.buffer += text
                session.total_written += len(text)
                # Trim only when nobody is waiting on the buffer (matches Node's
                # "no waiters" rule loosely: waiting threads re-check via offsets).
                if len(session.buffer) > BUFFER_TRIM_THRESHOLD:
                    drop = len(session.buffer) - BUFFER_TAIL_KEEP
                    session.buffer = session.buffer[drop:]
                    session.buffer_start += drop
                session.condition.notify_all()
            for viewer in list(session.viewers):
                try:
                    viewer(text)
                except Exception:  # noqa: BLE001
                    pass
        self._handle_exit(session)

    def _handle_exit(self, session: _Session) -> None:
        exit_code = None
        if session.process:
            try:
                exit_code = session.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                exit_code = None
        with session.condition:
            session.exited = True
            tail = session.buffer[-4000:]
            session.exit_error = (
                "su: Authentication failure"
                if AUTH_FAILURE_RE.search(tail)
                else f"su exited (code {exit_code}) before completing authentication."
            )
            session.condition.notify_all()
        if session.status == STATUS_READY:
            self.sessions.pop(session.username, None)
            self._set_status(session, STATUS_DISCONNECTED, "Shell process exited unexpectedly.")

    def _wait_for_pattern(self, session: _Session, regex: re.Pattern, timeout: float = 15.0, since: int | None = None):
        deadline = time.monotonic() + timeout
        with session.condition:
            start = session.total_written if since is None else since
            while True:
                offset = max(0, start - session.buffer_start)
                haystack = session.buffer[offset:]
                match = regex.search(haystack)
                if match:
                    return match, haystack
                if session.exited:
                    raise RuntimeError(session.exit_error or "su exited.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for pattern.")
                session.condition.wait(timeout=min(remaining, 1.0))

    def _wait_for_flag(self, session: _Session, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        with session.condition:
            while not session.password_line_submitted:
                if session.exited:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                session.condition.wait(timeout=min(remaining, 1.0))
            return True

    def _poll_auth_result(self, session: _Session, timeout: float = 120.0, poll_interval: float = 0.7) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if session.exited:
                return {"success": False, "error": session.exit_error or "su exited before authenticating."}
            token = secrets.token_hex(6)
            marker = f"SUBJ_WHOAMI_{token}"
            since = session.total_written
            self._pty_write(session, f'echo "{marker}:$(whoami):$?"\n')
            try:
                match, _ = self._wait_for_pattern(
                    session, re.compile(rf"{marker}:([^:\r\n]*):(-?\d+)"), timeout=3.0, since=since
                )
                if match.group(1) == session.username:
                    return {"success": True}
            except (TimeoutError, RuntimeError):
                # whoami never echoed back — probably still at the password prompt.
                pass
            recent_tail = session.buffer[-4000:]
            if AUTH_FAILURE_RE.search(recent_tail):
                return {"success": False, "error": "su: Authentication failure"}
            if session.exited:
                return {"success": False, "error": session.exit_error or "su exited before authenticating."}
            time.sleep(poll_interval)
        return {"success": False, "error": "Timed out waiting for su to authenticate."}

    def _pty_write(self, session: _Session, data: str) -> None:
        if session.master_fd is None:
            raise RuntimeError(f"No active shell for {session.username}.")
        os.write(session.master_fd, data.encode("utf-8"))

    def _run_framed(self, session: _Session, shell_command: str, timeout: float = 660.0) -> dict:
        """Runs a single-line command through the pty, waiting for a unique sentinel.

        The command is wrapped in a subshell so cd/env changes never leak, and command +
        marker echo are written as ONE line so the echoed input stays unambiguous.
        """
        if session.master_fd is None:
            raise RuntimeError(f"No active shell for {session.username}.")
        token = secrets.token_hex(8)
        end_re = re.compile(rf"SUBJ_END_{token}:(-?\d+)===")
        since = session.total_written
        self._pty_write(session, f'({shell_command}); echo "SUBJ_END_{token}:$?==="\n')
        match, haystack = self._wait_for_pattern(session, end_re, timeout=timeout, since=since)
        exit_code = int(match.group(1))
        body = haystack[: match.start()]
        first_newline = body.find("\n")
        body = "" if first_newline == -1 else body[first_newline + 1 :]
        return {"stdout": body.replace("\r\n", "\n"), "exitCode": exit_code}

    def _resolve_provider_cli(self, session: _Session, which_name: str) -> dict | None:
        which = self._run_framed(session, f"command -v {which_name}", timeout=10)
        lines = [line for line in str(which["stdout"] or "").strip().split("\n") if line]
        if which["exitCode"] == 0 and lines:
            return {"cli": which_name, "onPath": True}
        fallbacks = [
            f'test -x "$HOME/.local/bin/{which_name}" && echo "$HOME/.local/bin/{which_name}"',
            f'ls -1 "$HOME/.nvm/versions/node/"*/bin/{which_name} 2>/dev/null | tail -1',
            f"test -x /usr/local/bin/{which_name} && echo /usr/local/bin/{which_name}",
            f"test -x /usr/bin/{which_name} && echo /usr/bin/{which_name}",
        ]
        for command in fallbacks:
            result = self._run_framed(session, command, timeout=10)
            found = [line for line in str(result["stdout"] or "").strip().split("\n") if line]
            if found:
                return {"cli": found[-1], "onPath": False}
        return None

    def _provider_commands_for_cli(self, commands: dict, resolved: dict | None) -> dict:
        if not resolved or resolved["onPath"]:
            return commands
        quoted = shlex.quote(resolved["cli"])
        return {
            "which": commands["which"],
            "status": commands["status"].replace(commands["which"], quoted),
            "login": commands["login"].replace(commands["which"], quoted),
        }

    def _run_provider_login(self, session: _Session, provider: str, commands: dict) -> dict:
        self._emit_provider(session, provider, PROVIDER_CHECKING)
        resolved_cli = self._resolve_provider_cli(session, commands["which"])
        if not resolved_cli:
            message = f"{commands['which']} was not found on {session.username}'s PATH (also checked ~/.local/bin and nvm)."
            self._pty_write(session, f"echo {shlex.quote(message)}\n")
            raise RuntimeError(message)
        resolved = self._provider_commands_for_cli(commands, resolved_cli)
        status = self._run_framed(session, resolved["status"], timeout=30)
        if status["exitCode"] == 0:
            session.provider_ready[provider] = True
            self._emit_provider(session, provider, PROVIDER_READY)
            return {"status": PROVIDER_READY}

        self._emit_provider(session, provider, PROVIDER_AWAITING_OAUTH)
        banner = f"=== Sign in to {provider} as {session.username}. Complete the URL/device flow below. ==="
        self._pty_write(session, f"echo {shlex.quote(banner)}\n")
        self._run_framed(session, resolved["login"], timeout=5 * 60)
        recheck = self._run_framed(session, resolved["status"], timeout=15)
        blob = f"{status['stdout'] or ''}\n{recheck['stdout'] or ''}\n{session.buffer[-8000:]}"
        if recheck["exitCode"] == 0 or LOGIN_SUCCESS_RE.search(blob):
            session.provider_ready[provider] = True
            self._emit_provider(session, provider, PROVIDER_READY)
            return {"status": PROVIDER_READY}
        raise RuntimeError(
            f"{provider} login did not complete. Finish the {provider} sign-in in the terminal window, then retry."
        )

    # --- bridge -----------------------------------------------------------------------

    def _ensure_bridge(self, session: _Session) -> dict:
        with session.bridge_lock:
            if session.bridge:
                return session.bridge
            session.bridge = self._start_bridge(session)
            return session.bridge

    def _start_bridge(self, session: _Session) -> dict:
        secret = secrets.token_hex(16)
        socket_path = str(Path(self.tmp_dir) / f".subjective_chat_bridge_{session.username}_{secrets.token_hex(4)}.sock")
        log_path = f"{socket_path}.log"
        # SUBJECTIVE_CHAT_PIPELINE_PARENT_HOME must resolve against the TARGET user's own
        # $HOME (expanded by their login shell), never the invoking user's app home.
        start_command = (
            f'SUBJECTIVE_CHAT_PIPELINE_PARENT_HOME="$HOME/.subjective_chat" '
            f"{self.python_path} {shlex.quote(self.bridge_server_path)} "
            f"--socket {shlex.quote(socket_path)} --secret {secret} "
            f">{shlex.quote(log_path)} 2>&1 &"
        )
        self._run_framed(session, start_command, timeout=10)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            probe = self._run_framed(session, f"test -S {shlex.quote(socket_path)}", timeout=5)
            if probe["exitCode"] == 0:
                break
            time.sleep(0.25)

        # `test -S` only confirms the file exists, not that listen() was called yet.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe_socket:
            probe_socket.settimeout(5)
            probe_socket.connect(socket_path)
        return {"secret": secret, "socketPath": socket_path}

    def _shutdown_bridge(self, session: _Session) -> None:
        if not session.bridge:
            return
        try:
            self._call_bridge(session.bridge, {"action": "shutdown"}, timeout=3.0)
        finally:
            session.bridge = None

    def _run_in_session_once(self, session: _Session, argv: list[str], stdin_payload: str, timeout_ms: int) -> dict:
        if session.status != STATUS_READY:
            raise RuntimeError(f"No ready session for {session.username}.")
        bridge = self._ensure_bridge(session)
        response = self._call_bridge(
            bridge,
            {
                "argv": argv,
                "stdin": stdin_payload if isinstance(stdin_payload, str) else json.dumps(stdin_payload),
                "timeout_seconds": (timeout_ms + 999) // 1000,
            },
            timeout=(timeout_ms + 5000) / 1000,
        )
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Bridge command failed.")
        return {
            "stdout": response.get("stdout") or "",
            "stderr": response.get("stderr") or "",
            "exitCode": response.get("returncode"),
        }

    def _call_bridge(self, bridge: dict, payload: dict, timeout: float) -> dict:
        # Fresh connection per call: the server handles exactly one request per accepted
        # connection. Unix socket connects are cheap enough this costs nothing.
        body = json.dumps({"secret": bridge["secret"], **payload}).encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(bridge["socketPath"])
            conn.sendall(_HEADER.pack(len(body)) + body)
            header = self._recv_exact(conn, _HEADER.size)
            (length,) = _HEADER.unpack(header)
            data = self._recv_exact(conn, length)
        return json.loads(data.decode("utf-8"))

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = conn.recv(min(remaining, 65536))
            if not chunk:
                raise ConnectionError("Bridge connection closed before responding.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
