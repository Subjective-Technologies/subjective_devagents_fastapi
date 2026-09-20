"""The /ws/ssh terminal: interlocks, protocol, and the live session against a real host.

Everything except the last class runs against a stand-in shell, so the protocol is
pinned without needing a machine to connect to. The last class is the real thing —
a WebSocket to a real SSH host — and skips itself when that host is not reachable.
"""
from __future__ import annotations

import contextlib
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

TOKEN = "test-terminal-token"
# The host this repo's operators use for agent work. Present in ~/.ssh/config with its
# account, key and options; nothing here hard-codes credentials.
LIVE_HOST = os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_TEST_HOST", "subjective-agents")


@pytest.fixture(autouse=True)
def no_leaked_sessions():
    """The session registry is process-wide; a leak in one test must not fail another."""
    from subjective_devagents_fastapi import ssh_terminal

    ssh_terminal.sessions.close_all()
    yield
    ssh_terminal.sessions.close_all()


def wait_for(predicate, timeout=10.0, interval=0.05):
    """Server-side cleanup runs after the socket closes, so it is polled, not assumed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture()
def terminal_client(tmp_path, monkeypatch):
    """A client with the terminal switched on and both interlocks satisfied."""
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "chat-home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_SYNC", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SERVE_UI", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED", "1")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "token")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", TOKEN)

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as client:
        yield client


def authenticate(socket) -> dict:
    """Send the first-frame credential and return the server's hello."""
    socket.send_json({"type": "auth", "token": TOKEN})
    return socket.receive_json()


def drain_until(socket, kinds, limit=200):
    """Collect frames until one of `kinds` arrives. Returns (frames, final)."""
    frames = []
    for _ in range(limit):
        frame = socket.receive_json()
        if frame.get("type") in kinds:
            return frames, frame
        frames.append(frame)
    raise AssertionError(f"none of {kinds} arrived; saw {[f.get('type') for f in frames]}")


class TerminalReader:
    """One background reader for a socket, shared by every wait in a test.

    Two things force this. TestClient's websocket read blocks with no timeout, so
    reading inline hangs the moment the terminal goes quiet — which is its normal
    state between commands. And a second reader would sit blocked on the same socket
    and swallow frames meant for the first, so there must be exactly one.
    """

    def __init__(self, socket):
        self.frames: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(socket,), daemon=True).start()

    def _pump(self, socket):
        try:
            while True:
                self.frames.put(socket.receive_json())
        except Exception:
            self.frames.put(None)

    def wait_for(self, kinds, seconds=40):
        """Return the first frame of one of `kinds`, or fail."""
        seen, deadline = [], time.time() + seconds
        while time.time() < deadline:
            try:
                frame = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            if frame.get("type") in kinds:
                return frame
            seen.append(frame.get("type"))
        raise AssertionError(f"none of {kinds} arrived; saw {seen}")

    def collect(self, until=None, seconds=15) -> str:
        """Drain terminal output until `until` appears or the clock runs out."""
        chunks, deadline = [], time.time() + seconds
        while time.time() < deadline:
            try:
                frame = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            if frame.get("type") == "data":
                chunks.append(frame["data"])
            elif frame.get("type") in {"closed", "error"}:
                chunks.append(f"<{frame['type']}>")
                break
            if until and until in strip_ansi("".join(chunks)):
                break
        return strip_ansi("".join(chunks))


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


def text_of(frames, stream="stdout") -> str:
    return "".join(
        f.get("data", "") for f in frames if f.get("type") == "output" and f.get("stream") == stream
    )


# --- interlocks ------------------------------------------------------------------------


def test_the_terminal_is_off_unless_switched_on(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "chat-home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_SYNC", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SERVE_UI", "0")
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "development")

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/ssh") as socket:
            frame = socket.receive_json()
    assert frame["type"] == "error"
    assert "disabled" in frame["error"].lower()


def test_enabling_it_without_authentication_is_refused(monkeypatch):
    """A shell on every reachable machine must not be one open port away."""
    from subjective_devagents_fastapi import ssh_terminal

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED", "1")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "development")
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", raising=False)
    with pytest.raises(ssh_terminal.TerminalDisabled) as caught:
        ssh_terminal.ensure_available()
    assert "unauthenticated" in str(caught.value)


def test_an_unauthenticated_socket_is_closed(terminal_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with terminal_client.websocket_connect("/ws/ssh") as socket:
            socket.send_json({"type": "auth", "token": "wrong"})
            socket.receive_json()


# --- accepted parameters ---------------------------------------------------------------


def test_only_connection_fields_are_taken_from_the_browser():
    from subjective_devagents_fastapi import ssh_terminal

    connection = ssh_terminal.connection_from_request({
        "host": "server",
        "username": "deploy",
        "ssh_path": "/tmp/evil",      # choosing the local binary is not the client's call
        "persistent_session": True,   # derived here, not remote-controllable
        "type": "open",
    })
    assert connection == {"host": "server", "username": "deploy"}


def test_a_session_without_a_host_is_refused():
    from subjective_devagents_fastapi import ssh_terminal

    with pytest.raises(ValueError):
        ssh_terminal.connection_from_request({"username": "deploy"})


def test_secrets_never_appear_in_the_described_connection():
    from subjective_devagents_fastapi import ssh_terminal

    described = ssh_terminal.describe({
        "host": "server",
        "password": "hunter2",
        "private_key_content": "-----BEGIN-----",
        "key_passphrase": "letmein",
    })
    assert described == {"host": "server"}
    assert "hunter2" not in str(described)


# --- protocol, against a stand-in session ------------------------------------------------


class FakeSession:
    """A shell that answers without an SSH connection."""

    def __init__(self):
        self.commands = []
        self.closed = False
        self.interrupted = False
        self.alive = True
        self._on_event = None

    def start(self, **_kwargs):
        return {"type": "ready", "cwd": "/home/tester"}

    def attach(self, on_event):
        self._on_event = on_event

    def send_command(self, command):
        self.commands.append(command)
        if not self.alive:
            return False
        self._on_event({"type": "output", "stream": "stdout", "data": f"ran:{command}\n"})
        self._on_event({"type": "exit", "code": 0, "cwd": "/home/tester"})
        return True

    def interrupt(self):
        self.interrupted = True
        return True

    def close(self):
        self.closed = True
        self.alive = False


@pytest.fixture()
def fake_session(monkeypatch):
    from subjective_devagents_fastapi import ssh_terminal

    session = FakeSession()
    monkeypatch.setattr(
        ssh_terminal,
        "open_session",
        lambda connection, on_event=None, *, pty=False, cols=0, rows=0: (object(), session),
    )
    return session


def test_a_command_round_trips_over_the_socket(terminal_client, fake_session):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        hello = authenticate(socket)
        assert hello["type"] == "hello"
        socket.send_json({"type": "open", "host": "server"})
        _, ready = drain_until(socket, {"ready"})
        assert ready["cwd"] == "/home/tester"

        socket.send_json({"type": "input", "command": "echo hi"})
        frames, exited = drain_until(socket, {"exit"})
        assert text_of(frames) == "ran:echo hi\n"
        assert exited["code"] == 0
    assert fake_session.commands == ["echo hi"]


def test_the_opening_frame_never_carries_the_password(terminal_client, fake_session):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "host": "server", "password": "hunter2"})
        frames, _ = drain_until(socket, {"ready"})
        opening = [f for f in frames if f["type"] == "opening"]
        assert opening and "password" not in opening[0]["connection"]
        assert "hunter2" not in str(frames)


def test_opening_without_a_host_reports_the_problem(terminal_client, fake_session):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "username": "deploy"})
        frame = socket.receive_json()
    assert frame["type"] == "error"
    assert "host is required" in frame["error"]


def test_interrupt_reaches_the_session(terminal_client, fake_session):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "host": "server"})
        drain_until(socket, {"ready"})
        socket.send_json({"type": "interrupt"})
        socket.send_json({"type": "input", "command": "echo after"})
        drain_until(socket, {"exit"})
    assert fake_session.interrupted


def test_closing_the_socket_closes_the_remote_shell(terminal_client, fake_session):
    """A forgotten tab must not leave an SSH connection and a shell running."""
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "host": "server"})
        drain_until(socket, {"ready"})
    assert wait_for(lambda: fake_session.closed)


def test_a_session_is_registered_while_open_and_gone_after(terminal_client, fake_session):
    from subjective_devagents_fastapi import ssh_terminal

    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "host": "server"})
        _, ready = drain_until(socket, {"ready"})
        listed = ssh_terminal.sessions.snapshot()
        assert [entry["id"] for entry in listed] == [ready["sessionId"]]
        assert listed[0]["connection"] == {"host": "server"}
    assert wait_for(lambda: ssh_terminal.sessions.snapshot() == [])


def test_the_session_cap_is_enforced(terminal_client, fake_session, monkeypatch):
    from subjective_devagents_fastapi import ssh_terminal

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_MAX_SESSIONS", "1")
    with terminal_client.websocket_connect("/ws/ssh") as first:
        authenticate(first)
        first.send_json({"type": "open", "host": "server"})
        drain_until(first, {"ready"})
        with terminal_client.websocket_connect("/ws/ssh") as second:
            authenticate(second)
            second.send_json({"type": "open", "host": "server"})
            _, refused = drain_until(second, {"error"})
    assert "Too many open SSH sessions" in refused["error"]
    assert wait_for(lambda: ssh_terminal.sessions.snapshot() == [])


def test_an_over_long_command_is_refused_rather_than_sent(terminal_client, fake_session):
    from subjective_devagents_fastapi import ssh_terminal

    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "host": "server"})
        drain_until(socket, {"ready"})
        socket.send_json({"type": "input", "command": "x" * (ssh_terminal.MAX_COMMAND_BYTES + 1)})
        _, error = drain_until(socket, {"error"})
    assert "too long" in error["error"]
    assert fake_session.commands == []


def test_the_sessions_endpoint_reports_state_without_credentials(terminal_client):
    body = terminal_client.get("/api/ssh/sessions", headers={"X-DevAgents-Token": TOKEN}).json()
    assert body["enabled"] is True
    assert body["sessions"] == []


def test_the_sessions_endpoint_needs_the_token(terminal_client):
    assert terminal_client.get("/api/ssh/sessions").status_code == 401


# --- the real thing --------------------------------------------------------------------


def live_host_reachable() -> bool:
    try:
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", LIVE_HOST, "true"],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(
    not live_host_reachable(), reason=f"{LIVE_HOST} is not reachable from this machine"
)
class TestAgainstARealHost:
    """End to end: browser socket -> FastAPI -> ssh -> a real remote shell."""

    def test_a_command_runs_on_the_remote_machine(self, terminal_client):
        with terminal_client.websocket_connect("/ws/ssh") as socket:
            authenticate(socket)
            socket.send_json({"type": "open", "host": LIVE_HOST})
            _, ready = drain_until(socket, {"ready", "error"})
            assert ready["type"] == "ready", ready

            socket.send_json({"type": "input", "command": "hostname"})
            frames, exited = drain_until(socket, {"exit"})
            assert exited["code"] == 0
            assert text_of(frames).strip() == LIVE_HOST

    def test_shell_state_survives_between_commands(self, terminal_client):
        with terminal_client.websocket_connect("/ws/ssh") as socket:
            authenticate(socket)
            socket.send_json({"type": "open", "host": LIVE_HOST})
            drain_until(socket, {"ready"})

            socket.send_json({"type": "input", "command": "cd /tmp && export MARK=set"})
            _, exited = drain_until(socket, {"exit"})
            assert exited["cwd"] == "/tmp"

            socket.send_json({"type": "input", "command": "pwd; echo $MARK"})
            frames, exited = drain_until(socket, {"exit"})
            assert text_of(frames) == "/tmp\nset\n"

    def test_a_failing_command_reports_its_code_and_stderr(self, terminal_client):
        with terminal_client.websocket_connect("/ws/ssh") as socket:
            authenticate(socket)
            socket.send_json({"type": "open", "host": LIVE_HOST})
            drain_until(socket, {"ready"})

            socket.send_json({"type": "input", "command": "ls /nonexistent-path"})
            frames, exited = drain_until(socket, {"exit"})
            assert exited["code"] != 0
            assert "No such file" in text_of(frames, "stderr")

    def test_output_arrives_while_the_command_is_still_running(self, terminal_client):
        """Streaming, not batching: the first line must not wait for the last."""
        with terminal_client.websocket_connect("/ws/ssh") as socket:
            authenticate(socket)
            socket.send_json({"type": "open", "host": LIVE_HOST})
            drain_until(socket, {"ready"})

            started = time.time()
            socket.send_json({
                "type": "input",
                "command": "echo first; sleep 3; echo last",
            })
            first_seen = None
            while True:
                frame = socket.receive_json()
                if frame.get("type") == "output" and "first" in frame.get("data", ""):
                    first_seen = time.time() - started
                if frame.get("type") == "exit":
                    total = time.time() - started
                    break
            assert first_seen is not None
            assert total >= 3, "the command should have taken at least its sleep"
            assert first_seen < total - 1.5, (
                f"first line arrived at {first_seen:.2f}s of a {total:.2f}s command: "
                "output is being batched, not streamed"
            )

    @staticmethod
    @contextlib.contextmanager
    def open_terminal(client, **extra):
        """Open a pty session, yielding the socket and its single reader.

        A context manager rather than a plain call: TestClient's socket has to be
        exited properly or its portal thread outlives the test and the suite hangs
        at teardown.
        """
        with client.websocket_connect("/ws/ssh") as socket:
            socket.send_json({"type": "auth", "token": TOKEN})
            reader = TerminalReader(socket)
            reader.wait_for({"hello"})
            socket.send_json({
                "type": "open", "host": LIVE_HOST, "pty": True,
                "cols": 100, "rows": 30, **extra,
            })
            ready = reader.wait_for({"ready", "error"})
            assert ready["type"] == "ready", ready
            assert ready["mode"] == "pty"
            # `ready` means ssh started, not that the remote shell is reading yet.
            # Anything sent before it is, including a Ctrl-C, reaches the login
            # process instead of the shell — and a Ctrl-C there ends the session.
            # So sync on a round trip before handing the terminal over.
            socket.send_json({"type": "data", "data": 'echo "PTY""_READY"\r'})
            handshake = reader.collect(until="PTY_READY", seconds=25)
            assert "PTY_READY" in handshake, handshake[-400:]
            yield socket, reader, ready

    def test_a_pty_session_gets_a_real_tty_on_the_remote_host(self, terminal_client):
        """What command mode cannot do: claude and codex check for this and refuse."""
        with self.open_terminal(terminal_client) as (socket, reader, _):
            # Split markers: the tty echoes input, so a whole marker inside the
            # command would match before the command has run.
            socket.send_json({"type": "data", "data":
                              'test -t 0 && echo "IS_A""_TTY"; stty size; echo "T=""$TERM"\r'})
            transcript = reader.collect(until="T=xterm-256color", seconds=25)
            assert "IS_A_TTY" in transcript, transcript[-400:]
            assert "30 100" in transcript, transcript[-400:]
            assert "T=xterm-256color" in transcript, transcript[-400:]

    def test_resizing_re_flows_the_remote_terminal(self, terminal_client):
        """Only the local pty makes this possible: it is what carries SIGWINCH."""
        with self.open_terminal(terminal_client) as (socket, reader, _):
            socket.send_json({"type": "resize", "cols": 80, "rows": 24})
            socket.send_json({"type": "data", "data": "stty size\r"})
            transcript = reader.collect(until="24 80", seconds=25)
            assert "24 80" in transcript, transcript[-400:]

    def test_control_c_reaches_the_remote_job_and_the_session_survives(self, terminal_client):
        """Command mode's interrupt drops the connection; this one signals the job."""
        with self.open_terminal(terminal_client) as (socket, reader, _):
            socket.send_json({"type": "data", "data": 'sleep 30; echo "SHOULD_NOT""_PRINT"\r'})
            time.sleep(2.0)
            socket.send_json({"type": "interrupt"})
            socket.send_json({"type": "data", "data": 'echo "STILL""_HERE"\r'})
            transcript = reader.collect(until="STILL_HERE", seconds=25)
            assert "STILL_HERE" in transcript, transcript[-400:]
            # The sleep was killed, so its echo never ran — and it could not have
            # finished on its own inside this window.
            assert "SHOULD_NOT_PRINT" not in transcript, transcript[-400:]

    def test_the_remote_shell_is_gone_once_the_socket_closes(self, terminal_client):
        from subjective_devagents_fastapi import ssh_terminal

        with terminal_client.websocket_connect("/ws/ssh") as socket:
            authenticate(socket)
            socket.send_json({"type": "open", "host": LIVE_HOST})
            drain_until(socket, {"ready"})
            socket.send_json({"type": "input", "command": "echo $$ > /tmp/subjective-ssh-test-pid"})
            drain_until(socket, {"exit"})
        assert wait_for(lambda: ssh_terminal.sessions.snapshot() == [])
        # The shell that wrote that pid file is no longer running on the remote host.
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", LIVE_HOST,
             "pid=$(cat /tmp/subjective-ssh-test-pid); rm -f /tmp/subjective-ssh-test-pid; "
             "kill -0 $pid 2>/dev/null && echo ALIVE || echo GONE"],
            capture_output=True, text=True, timeout=30,
        )
        assert probe.stdout.strip() == "GONE", probe.stdout


# --- the standalone console ------------------------------------------------------------


def test_the_console_page_is_served_when_the_terminal_is_on(terminal_client):
    """The DevAgents client pins the production API and Firebase, so it cannot drive a
    local backend; this page is how the service is exercised on its own."""
    response = terminal_client.get("/ssh")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/ws/ssh" in response.text
    # Static like the rest of the UI: the socket still demands the token.
    assert "auth" in response.text


def test_the_console_page_is_absent_when_the_terminal_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "chat-home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_SYNC", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SERVE_UI", "0")
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "development")

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as client:
        assert client.get("/ssh").status_code == 404


# --- pty mode --------------------------------------------------------------------------


class FakePtySession:
    """A terminal that answers without an SSH connection."""

    def __init__(self):
        self.written = []
        self.sizes = []
        self.closed = False
        self.alive = True
        self._on_event = None

    def start(self, **_kwargs):
        return {"type": "ready", "cols": 100, "rows": 30, "term": "xterm-256color"}

    def attach(self, on_event):
        self._on_event = on_event

    def write(self, data):
        self.written.append(data)
        if not self.alive:
            return False
        self._on_event({"type": "data", "data": data})
        return True

    def resize(self, cols, rows):
        self.sizes.append((cols, rows))
        return True

    def interrupt(self):
        return self.write("\x03")

    def close(self):
        self.closed = True
        self.alive = False


@pytest.fixture()
def fake_pty(monkeypatch):
    from subjective_devagents_fastapi import ssh_terminal

    session = FakePtySession()
    captured = {}

    def opener(connection, on_event=None, *, pty=False, cols=0, rows=0):
        captured.update({"pty": pty, "cols": cols, "rows": rows})
        return object(), session

    monkeypatch.setattr(ssh_terminal, "open_session", opener)
    session.captured = captured
    return session


def open_pty_socket(socket, **extra):
    authenticate(socket)
    socket.send_json({"type": "open", "host": "server", "pty": True, "cols": 100, "rows": 30, **extra})
    return drain_until(socket, {"ready"})


def test_a_pty_session_reports_its_mode_and_size(terminal_client, fake_pty):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        _, ready = open_pty_socket(socket)
    assert ready["mode"] == "pty"
    assert ready["term"] == "xterm-256color"
    assert fake_pty.captured == {"pty": True, "cols": 100, "rows": 30}


def test_keystrokes_reach_the_terminal_untouched(terminal_client, fake_pty):
    """Raw input: no newline added, nothing quoted — the remote tty does all of it."""
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        open_pty_socket(socket)
        socket.send_json({"type": "data", "data": "l"})
        drain_until(socket, {"data"})
        socket.send_json({"type": "data", "data": "s\r"})
        drain_until(socket, {"data"})
    assert fake_pty.written == ["l", "s\r"]


def test_a_command_in_pty_mode_becomes_keystrokes_plus_return(terminal_client, fake_pty):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        open_pty_socket(socket)
        socket.send_json({"type": "input", "command": "uptime"})
        drain_until(socket, {"data"})
    assert fake_pty.written == ["uptime\r"]


def test_resizing_reaches_the_terminal(terminal_client, fake_pty):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        open_pty_socket(socket)
        socket.send_json({"type": "resize", "cols": 80, "rows": 24})
        socket.send_json({"type": "data", "data": "x"})
        drain_until(socket, {"data"})
    assert fake_pty.sizes == [(80, 24)]


def test_an_absurd_window_size_is_clamped(terminal_client, fake_pty):
    from subjective_devagents_fastapi import ssh_terminal

    with terminal_client.websocket_connect("/ws/ssh") as socket:
        open_pty_socket(socket)
        socket.send_json({"type": "resize", "cols": 99999, "rows": -4})
        socket.send_json({"type": "data", "data": "x"})
        drain_until(socket, {"data"})
    assert fake_pty.sizes == [(ssh_terminal.MAX_COLS, 1)]


def test_interrupt_sends_a_real_control_c(terminal_client, fake_pty):
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        open_pty_socket(socket)
        socket.send_json({"type": "interrupt"})
        drain_until(socket, {"data"})
    assert fake_pty.written == ["\x03"]


def test_raw_input_is_refused_on_a_command_session(terminal_client, fake_session):
    """The two modes are not interchangeable; saying so beats writing to the void."""
    with terminal_client.websocket_connect("/ws/ssh") as socket:
        authenticate(socket)
        socket.send_json({"type": "open", "host": "server"})
        drain_until(socket, {"ready"})
        socket.send_json({"type": "data", "data": "ls\r"})
        _, error = drain_until(socket, {"error"})
    assert "needs a pty session" in error["error"]


def test_the_console_serves_the_vendored_terminal_emulator(terminal_client):
    """Vendored, not from a CDN: the page must work with no internet."""
    response = terminal_client.get("/ssh/assets/xterm.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert terminal_client.get("/ssh/assets/xterm.css").status_code == 200


def test_the_asset_route_serves_nothing_but_those_files(terminal_client):
    """A path parameter over a directory would be a file-read primitive."""
    for name in ("ssh_terminal.html", "../app.py", "../../pyproject.toml"):
        assert terminal_client.get(f"/ssh/assets/{name}").status_code == 404
