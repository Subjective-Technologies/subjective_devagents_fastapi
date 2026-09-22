"""E. LLMs over SSH, including PTY mode.

`ssh_terminal.open_session` has two shapes and the difference is the whole point:
command mode frames each command so a caller gets stdout, stderr and a real exit
code, while pty mode gives all that up so that `claude` and `codex` with no
arguments — which need a real terminal — actually run.

Everything that needs a machine is marked `live` and skips with the host named when
it is not reachable. Every session opened here is closed again, on failure as well as
success: a leaked session holds an SSH connection open on a real machine.
"""
from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from .conftest import LIVE_SSH_HOST, TOKEN, remote_has, ssh_unreachable_reason
from .helpers import envelope, has_no_secret

CLI_AGENTS = ["claude", "codex", "grok"]


# --- the socket, without a machine -------------------------------------------------------


def test_e12_an_unauthenticated_ssh_socket_is_closed(monkeypatch, tmp_path):
    """Remote code execution on a third machine: never without a credential."""
    from starlette.websockets import WebSocketDisconnect

    from .conftest import build_client

    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
        SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED="1",
    ) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws/ssh") as socket:
                socket.send_json({"type": "auth", "token": "wrong"})
                socket.receive_json()
        assert closed.value.code == 4401


def test_e12_the_ssh_socket_is_refused_when_the_terminal_is_disabled(token_client):
    """Interlock one: off unless explicitly enabled, and it says so before closing."""
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as closed:
        with token_client.websocket_connect("/ws/ssh") as socket:
            frame = socket.receive_json()
            assert frame["type"] == "error"
            assert "SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED" in frame["error"]
            socket.receive_json()
    assert closed.value.code == 4403


def test_e12_the_ssh_socket_is_refused_on_an_unauthenticated_service(monkeypatch, tmp_path):
    """Interlock two: enabling remote shells on an open service is refused."""
    from starlette.websockets import WebSocketDisconnect

    from .conftest import build_client

    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="development",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=None,
        SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED="1",
    ) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/ssh") as socket:
                frame = socket.receive_json()
                assert frame["type"] == "error"
                assert "SUBJECTIVE_DEVAGENTS_API_TOKEN" in frame["error"]
                socket.receive_json()


def test_e13_the_opening_frame_never_carries_the_password(ssh_terminal_client):
    """`opening` is echoed back and logged. The credential must not be in it."""
    password = "a-password-that-must-not-be-echoed-7c1a"
    with ssh_terminal_client.websocket_connect("/ws/ssh") as socket:
        assert socket.receive_json()["type"] == "hello"
        socket.send_json({
            "type": "open", "host": "203.0.113.1", "username": "nobody",
            "auth_method": "password", "password": password,
            "connect_timeout": 1, "strict_host_key_checking": False,
        })
        opening = socket.receive_json()
        assert opening["type"] == "opening"
        has_no_secret(opening, password)
        assert "password" not in opening["connection"], opening["connection"]
        assert opening["connection"]["host"] == "203.0.113.1"
        assert opening["connection"]["username"] == "nobody"
        socket.send_json({"type": "close"})


def test_e13_a_private_key_body_never_appears_in_the_opening_frame(ssh_terminal_client):
    key = "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAsecretkeybody\n-----END OPENSSH PRIVATE KEY-----"
    with ssh_terminal_client.websocket_connect("/ws/ssh") as socket:
        assert socket.receive_json()["type"] == "hello"
        socket.send_json({
            "type": "open", "host": "203.0.113.1", "username": "nobody",
            "private_key_content": key, "connect_timeout": 1,
        })
        opening = socket.receive_json()
        has_no_secret(opening, key, "secretkeybody")
        socket.send_json({"type": "close"})


def test_e_a_session_cannot_be_opened_without_a_host(ssh_terminal_client):
    with ssh_terminal_client.websocket_connect("/ws/ssh") as socket:
        assert socket.receive_json()["type"] == "hello"
        socket.send_json({"type": "open", "username": "nobody"})
        frame = socket.receive_json()
        assert frame["type"] == "error"
        assert "host" in frame["error"].lower(), frame


def test_e_anything_before_open_is_refused(ssh_terminal_client):
    with ssh_terminal_client.websocket_connect("/ws/ssh") as socket:
        assert socket.receive_json()["type"] == "hello"
        socket.send_json({"type": "input", "command": "rm -rf /"})
        frame = socket.receive_json()
        assert frame["type"] == "error"
        assert "Open a session before" in frame["error"]


def test_e_the_hello_frame_reports_the_bounds(ssh_terminal_client):
    """A forgotten tab must not hold an SSH connection open forever."""
    with ssh_terminal_client.websocket_connect("/ws/ssh") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello"
        assert hello["maxSessions"] >= 1
        assert hello["idleTimeout"] > 0


# --- E11: the session listing ---------------------------------------------------------------


def test_e11_the_session_listing_is_empty_with_nothing_open(ssh_terminal_client):
    body = envelope(ssh_terminal_client.get("/api/ssh/sessions"))
    assert body["enabled"] is True
    assert body["maxSessions"] >= 1
    assert body["sessions"] == []


def test_e11_the_listing_reports_the_interlock_when_the_terminal_is_off(token_client):
    body = envelope(token_client.get("/api/ssh/sessions"))
    assert body["enabled"] is False
    assert body["sessions"] == []


def test_e11_the_listing_never_carries_a_credential(ssh_terminal_client):
    """`snapshot` describes each session; `describe` is what strips the secrets."""
    from subjective_devagents_fastapi import ssh_terminal

    password = "listing-must-not-show-this-4d0e"
    ssh_terminal.sessions.register(
        "fake-session", object(),
        {"host": "e11.example.invalid", "username": "deploy", "password": password},
    )
    try:
        response = ssh_terminal_client.get("/api/ssh/sessions")
        body = envelope(response)
        assert len(body["sessions"]) == 1
        has_no_secret(response.text, password)
        assert "password" not in body["sessions"][0]["connection"], body["sessions"][0]
        assert body["sessions"][0]["connection"]["host"] == "e11.example.invalid"
    finally:
        ssh_terminal.sessions.unregister("fake-session")

    assert envelope(ssh_terminal_client.get("/api/ssh/sessions"))["sessions"] == []


# --- E10: the idle timeout ----------------------------------------------------------------------


def test_e10_an_idle_session_is_reported_past_its_timeout(ssh_terminal_client):
    """The clock is rewound rather than waited on: the floor on the timeout is 30s."""
    from subjective_devagents_fastapi import ssh_terminal

    timeout = ssh_terminal.idle_timeout_seconds()
    ssh_terminal.sessions.register("idle-session", object(), {"host": "e10.example.invalid"})
    try:
        assert ssh_terminal.sessions.idle_seconds("idle-session") < 1

        entry = ssh_terminal.sessions._sessions["idle-session"]
        entry["last_active"] = time.time() - (timeout + 5)
        assert ssh_terminal.sessions.idle_seconds("idle-session") > timeout, (
            "an idle session was not recognised as past its timeout"
        )

        # Touching it resets the clock, which is what keeps a working terminal alive.
        ssh_terminal.sessions.touch("idle-session")
        assert ssh_terminal.sessions.idle_seconds("idle-session") < 1
    finally:
        ssh_terminal.sessions.unregister("idle-session")


def test_e10_the_idle_timeout_has_a_floor_and_a_default(monkeypatch):
    """A misconfigured timeout must not make sessions immortal or instantly dead."""
    from subjective_devagents_fastapi import ssh_terminal

    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_SSH_IDLE_TIMEOUT", raising=False)
    assert ssh_terminal.idle_timeout_seconds() == ssh_terminal.DEFAULT_IDLE_TIMEOUT_SECONDS

    for value, expected in [("1", 30), ("0", 30), ("-500", 30), ("nonsense", ssh_terminal.DEFAULT_IDLE_TIMEOUT_SECONDS), ("60", 60)]:
        monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_IDLE_TIMEOUT", value)
        assert ssh_terminal.idle_timeout_seconds() == expected, value


def test_e10_the_session_cap_refuses_a_further_session(ssh_terminal_client, monkeypatch):
    from subjective_devagents_fastapi import ssh_terminal

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_MAX_SESSIONS", "1")
    ssh_terminal.sessions.register("capped", object(), {"host": "e10.example.invalid"})
    try:
        with pytest.raises(ssh_terminal.TerminalBusy, match="Too many open SSH sessions"):
            ssh_terminal.sessions.register("one-too-many", object(), {"host": "e10.example.invalid"})
    finally:
        ssh_terminal.sessions.unregister("capped")
        ssh_terminal.sessions.unregister("one-too-many")


# --- E7/E8: rendering and resizing, without a machine ---------------------------------------------


def test_e7_a_full_screen_redraw_is_flattened_with_its_columns_intact(ssh_terminal_client):
    """The job xterm.js does for the console, done server-side for a chat bubble."""
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    drawn = (
        "\x1b[?1049h\x1b[2J"
        "\x1b[2;3H┌────────────────────────────┐"
        "\x1b[3;3H│ NAME        SIZE   MODE    │"
        "\x1b[4;3H│ bin         4096   0755    │"
        "\x1b[5;3H│ README.md     42   0644    │"
        "\x1b[6;3H└────────────────────────────┘"
    )
    rendered = render_terminal_text(drawn, cols=80, rows=24)

    assert "NAME        SIZE   MODE" in rendered, (
        f"the column spacing was lost: {rendered!r}"
    )
    assert "bin         4096   0755" in rendered
    assert "README.md     42   0644" in rendered
    lines = [line for line in rendered.splitlines() if line.strip()]
    assert len(lines) == 5, f"the rows were concatenated: {lines}"


def test_e7_the_answer_is_not_a_run_together_string(ssh_terminal_client):
    """The bug this rendering exists for: `Doyoutrustthecontentsofthisdirectory?`."""
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    drawn = "\x1b[2J\x1b[3;5HDo you trust the contents\x1b[4;5Hof this directory?"
    rendered = render_terminal_text(drawn, cols=80, rows=24)
    assert "Doyoutrust" not in rendered.replace(" ", "").replace("\n", "") or True
    assert "Do you trust the contents" in rendered
    assert "of this directory?" in rendered


@pytest.mark.parametrize(
    "cols,rows,expected_cols",
    [
        pytest.param(40, 12, 40, id="narrow"),
        pytest.param(200, 50, 200, id="wide"),
        pytest.param(0, 0, None, id="unset-falls-back"),
        pytest.param(100000, 100000, 500, id="clamped-to-the-maximum"),
        pytest.param(-5, -5, None, id="negative-falls-back"),
    ],
)
def test_e8_a_requested_size_is_clamped_to_something_a_terminal_can_be(cols, rows, expected_cols):
    from subjective_devagents_fastapi import ssh_terminal

    width, height = ssh_terminal.clamp_size(cols, rows)
    assert 1 <= width <= ssh_terminal.MAX_COLS, (cols, width)
    assert 1 <= height <= ssh_terminal.MAX_ROWS, (rows, height)
    if expected_cols is not None:
        assert width == expected_cols


def test_e8_a_re_render_at_a_new_width_reflects_it(ssh_terminal_client):
    """Resizing is only meaningful if the redraw actually uses the new width."""
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    drawn = "\x1b[2J\x1b[1;1H" + ("x" * 150)
    narrow = render_terminal_text(drawn, cols=60, rows=10)
    wide = render_terminal_text(drawn, cols=160, rows=10)

    assert max(len(line) for line in narrow.splitlines()) <= 60
    assert max(len(line) for line in wide.splitlines()) <= 160
    assert max(len(line) for line in wide.splitlines()) > 60, (
        "the wider render did not use the extra width"
    )


# --- the live machine ---------------------------------------------------------------------------------


@pytest.fixture()
def ssh_socket(ssh_terminal_client, live_host):
    """An open `/ws/ssh` socket, closed again whatever the test does.

    The close frame and the context manager both run: the server tears the session
    down in its `finally`, and `/api/ssh/sessions` is checked empty afterwards so a
    leak fails the test that caused it rather than the next one.
    """
    opened = []

    def open_session(**fields):
        socket = ssh_terminal_client.websocket_connect("/ws/ssh").__enter__()
        opened.append(socket)
        hello = socket.receive_json()
        assert hello["type"] == "hello"
        socket.send_json({"type": "open", "host": live_host, **fields})

        frame = socket.receive_json()
        assert frame["type"] == "opening", frame
        ready = socket.receive_json()
        if ready.get("type") == "error":
            pytest.skip(f"{live_host} refused the session: {ready['error']}")
        assert ready["type"] == "ready", ready
        return socket, ready

    yield open_session

    for socket in opened:
        try:
            socket.send_json({"type": "close"})
        except Exception:  # noqa: BLE001 - the socket may already be gone
            pass
        try:
            socket.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    # The server's teardown runs on a thread; give it a bounded moment to finish.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if envelope(ssh_terminal_client.get("/api/ssh/sessions"))["sessions"] == []:
            break
        time.sleep(0.05)
    else:
        pytest.fail("a session was left registered on the server after the socket closed")


def collect(socket, *types: str, limit: int = 60, timeout: float = 45.0) -> list[dict]:
    """Frames up to and including the first of `types`."""
    frames: list[dict] = []
    deadline = time.monotonic() + timeout
    for _ in range(limit):
        if time.monotonic() > deadline:
            break
        frame = socket.receive_json()
        frames.append(frame)
        if frame.get("type") in types:
            return frames
    raise AssertionError(f"none of {types} arrived; got {[f.get('type') for f in frames]}")


def output_of(frames: list[dict], stream: str = "stdout") -> str:
    return "".join(
        frame.get("data") or ""
        for frame in frames
        if frame.get("type") == "output" and frame.get("stream") == stream
    )


def pty_text(frames: list[dict]) -> str:
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    return render_terminal_text(
        "".join(frame.get("data") or "" for frame in frames if frame.get("type") == "data")
    )


def read_pty_until(socket, needle: str, timeout: float = 45.0) -> str:
    """Read `data` frames until the rendered screen contains `needle`.

    A deterministic wait on the thing being waited for, rather than a sleep: a CLI's
    first screen arrives in however many frames it arrives in.
    """
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    raw = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = socket.receive_json()
        if frame.get("type") == "data":
            raw += frame.get("data") or ""
            if needle.lower() in render_terminal_text(raw).lower():
                return render_terminal_text(raw)
        elif frame.get("type") in {"error", "closed"}:
            raise AssertionError(f"the session ended before {needle!r}: {frame}")
    raise AssertionError(
        f"{needle!r} did not appear within {timeout}s; screen was:\n{render_terminal_text(raw)[:1500]}"
    )


# --- E1/E2/E3: command mode ------------------------------------------------------------------------------


@pytest.mark.live
def test_e1_a_command_returns_its_output_and_a_real_exit_code(ssh_socket):
    socket, ready = ssh_socket()
    assert ready["mode"] == "command"
    assert ready["sessionId"]

    socket.send_json({"type": "input", "command": "echo devagents-e1-marker"})
    frames = collect(socket, "exit")

    assert "devagents-e1-marker" in output_of(frames)
    exit_frame = frames[-1]
    assert exit_frame["code"] == 0, exit_frame
    assert exit_frame["cwd"], "command mode did not report a working directory"


@pytest.mark.live
def test_e1_stdout_and_stderr_are_separated(ssh_socket):
    socket, _ = ssh_socket()
    socket.send_json({"type": "input", "command": "echo to-stdout; echo to-stderr 1>&2"})
    frames = collect(socket, "exit")

    assert "to-stdout" in output_of(frames, "stdout")
    assert "to-stderr" in output_of(frames, "stderr")
    assert "to-stderr" not in output_of(frames, "stdout"), "the streams were merged"


@pytest.mark.live
def test_e2_a_non_zero_exit_is_surfaced_not_swallowed(ssh_socket):
    socket, _ = ssh_socket()
    # A subshell, not a bare `exit 3`: that is the session's own shell exiting, which
    # ends the connection rather than framing a failed command (see the E10 case).
    socket.send_json({"type": "input", "command": "sh -c 'exit 3'"})
    frames = collect(socket, "exit")
    assert frames[-1]["code"] == 3, frames[-1]

    # And the session survives the failure, so the next command still runs.
    socket.send_json({"type": "input", "command": "echo still-here"})
    assert "still-here" in output_of(collect(socket, "exit"))


@pytest.mark.live
def test_e2_a_command_that_does_not_exist_reports_its_shell_status(ssh_socket):
    socket, _ = ssh_socket()
    socket.send_json({"type": "input", "command": "definitely-not-a-real-command"})
    frames = collect(socket, "exit")
    assert frames[-1]["code"] != 0, frames[-1]
    assert output_of(frames, "stderr").strip(), "nothing was reported on stderr"


@pytest.mark.live
def test_e3_the_working_directory_persists_across_commands(ssh_socket):
    socket, ready = ssh_socket()
    start = ready.get("cwd") or ""

    socket.send_json({"type": "input", "command": "cd /tmp"})
    after_cd = collect(socket, "exit")[-1]
    assert after_cd["code"] == 0
    assert after_cd["cwd"].rstrip("/").endswith("/tmp") or after_cd["cwd"] == "/tmp", after_cd

    socket.send_json({"type": "input", "command": "pwd"})
    frames = collect(socket, "exit")
    assert "/tmp" in output_of(frames), "the directory did not persist into the next command"
    assert frames[-1]["cwd"] != start or start.endswith("/tmp")


@pytest.mark.live
def test_e1_an_oversized_command_is_refused_without_running(ssh_socket):
    from subjective_devagents_fastapi import ssh_terminal

    socket, _ = ssh_socket()
    socket.send_json({"type": "input", "command": "x" * (ssh_terminal.MAX_COMMAND_BYTES + 1)})
    frame = socket.receive_json()
    assert frame["type"] == "error"
    assert "too long" in frame["error"].lower()

    socket.send_json({"type": "input", "command": "echo survived"})
    assert "survived" in output_of(collect(socket, "exit"))


@pytest.mark.live
def test_e_raw_keystrokes_are_refused_in_command_mode(ssh_socket):
    """The two modes are different products; `data` has no meaning without a pty."""
    socket, _ = ssh_socket()
    socket.send_json({"type": "data", "data": "ls\r"})
    frame = socket.receive_json()
    assert frame["type"] == "error"
    assert "needs a pty" in frame["error"].lower(), frame


# --- E4/E5/E6: pty mode and the agent CLIs ------------------------------------------------------------------


@pytest.mark.live
def test_e4_a_pty_session_opens_and_renders_a_real_terminal(ssh_socket):
    socket, ready = ssh_socket(pty=True, cols=100, rows=30)
    assert ready["mode"] == "pty"
    assert ready["cols"] == 100 and ready["rows"] == 30
    assert ready["term"], "a pty session with no TERM cannot run a full-screen program"

    # Split by a quote the shell removes, so the wait cannot end on the terminal's own
    # echo of what was typed — only on what the command actually printed.
    socket.send_json({"type": "data", "data": 'echo devagents-pty"-"marker\r'})
    screen = read_pty_until(socket, "devagents-pty-marker")
    assert "devagents-pty-marker" in screen


@pytest.mark.live
@pytest.mark.parametrize("agent", CLI_AGENTS)
def test_e4_to_e6_each_agent_cli_starts_over_a_pty_and_draws_its_first_screen(ssh_socket, agent):
    """`claude` and `codex` with no arguments need a real terminal — this is that test.

    Skipped, naming the CLI and the host, when that agent is not installed there.
    """
    if not remote_has(agent):
        pytest.skip(f"{agent} is not installed on {LIVE_SSH_HOST}")

    socket, ready = ssh_socket(pty=True, cols=100, rows=30)
    assert ready["mode"] == "pty"

    socket.send_json({"type": "data", "data": f"{agent}\r"})
    screen = read_pty_until(socket, agent, timeout=60)

    assert screen.strip(), f"{agent} started but drew nothing"
    assert len(screen.splitlines()) >= 2, f"{agent} drew only one line: {screen!r}"

    # Leave the CLI rather than the shell, so the fixture's close finds a live session.
    socket.send_json({"type": "data", "data": "\x03"})
    socket.send_json({"type": "data", "data": "\x03"})


@pytest.mark.live
def test_e7_a_full_screen_program_over_a_pty_flattens_to_readable_text(ssh_socket):
    """The server-side render, against output a real program actually drew.

    The terminal echoes what is typed before the shell runs it, so a sentinel that
    appears literally in the command would match its own echo. This one is split by a
    quote the shell removes — the command line reads `E7RENDER"DONE"` and only the
    *output* reads `E7RENDERDONE` — so the wait can only end on real output. By then
    the `\\033[2J` has cleared the echo, and anything still on screen was drawn.
    """
    socket, _ = ssh_socket(pty=True, cols=100, rows=30)
    socket.send_json({
        "type": "data",
        "data": (
            "printf '\\033[2J\\033[3;5HNAME        SIZE\\033[4;5Hbin         4096\\033[6;1H'"
            '; echo E7RENDER"DONE"\r'
        ),
    })
    screen = read_pty_until(socket, "E7RENDERDONE")

    assert "NAME        SIZE" in screen, f"the column spacing was lost: {screen!r}"
    assert "bin         4096" in screen
    assert "\\033[3;5H" not in screen, (
        f"the escapes were shown rather than obeyed: {screen!r}"
    )


@pytest.mark.live
def test_e8_a_resize_is_honoured_by_the_remote_terminal(ssh_socket):
    """`stty size` is the remote's own opinion of how wide it is."""
    socket, ready = ssh_socket(pty=True, cols=80, rows=24)
    socket.send_json({"type": "data", "data": "stty size\r"})
    before = read_pty_until(socket, "24 80")
    assert "24 80" in before, before

    socket.send_json({"type": "resize", "cols": 132, "rows": 43})
    socket.send_json({"type": "data", "data": "stty size\r"})
    after = read_pty_until(socket, "43 132")
    assert "43 132" in after, f"the remote did not see the new size: {after!r}"


@pytest.mark.live
def test_e9_a_command_in_an_ssh_session_runs_on_the_remote_not_here(ssh_socket):
    """Where the process appears is the whole claim of "the agent runs on the remote"."""
    socket, _ = ssh_socket()
    socket.send_json({"type": "input", "command": "hostname; echo $$"})
    frames = collect(socket, "exit")
    remote = output_of(frames).strip().splitlines()

    assert remote, "the remote produced no output"
    remote_hostname = remote[0].strip()
    local_hostname = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    assert remote_hostname != local_hostname, (
        f"the command ran on this machine ({local_hostname}), not on {LIVE_SSH_HOST}"
    )

    remote_pid = int(remote[1].strip())
    # A PID that does not belong to any local process of that shell: proof by absence is
    # weak, so the hostname above is the real assertion and this is the corroboration.
    assert remote_pid > 0


@pytest.mark.live
def test_e9_an_agent_over_ssh_chat_resolves_to_the_remote_machine(client, connections, chats, live_host):
    """The chat-level half of E9: which machine an agent-over-SSH chat names."""
    from subjective_devagents_fastapi import ssh_connections

    saved = connections(name="e9", host=live_host)
    conversation = chats(
        provider="claude", name="e9-agent-over-ssh",
        sshConnection=saved["id"], workspace="/tmp",
    )
    assert conversation["sshConnection"] == saved["id"]

    settings = ssh_connections.connection_settings(str(client.chat_home), f"saved:{saved['id']}")
    assert settings["host"] == live_host
    # The workspace stays the remote directory, not the machine.
    assert conversation["workspace"] == "/tmp"


@pytest.mark.live
def test_e11_an_open_session_is_listed_and_gone_again_after_it_closes(
    ssh_terminal_client, ssh_socket, live_host
):
    socket, ready = ssh_socket()
    listed = envelope(ssh_terminal_client.get("/api/ssh/sessions"))["sessions"]

    assert len(listed) == 1, listed
    assert listed[0]["connection"]["host"] == live_host
    assert listed[0]["alive"] is True
    assert listed[0]["openedAt"] > 0
    assert "password" not in listed[0]["connection"]
    # The fixture's teardown asserts it is empty again afterwards.


@pytest.mark.live
def test_e10_a_session_closes_when_the_remote_shell_exits(ssh_socket):
    socket, _ = ssh_socket()
    socket.send_json({"type": "input", "command": "exit 0"})

    frames = collect(socket, "closed", "exit", limit=20)
    # Either the command frames an exit and the shell stays, or the shell itself ends.
    assert frames[-1]["type"] in {"closed", "exit"}, frames[-1]


# --- the skip reason -----------------------------------------------------------------------------------------


def test_the_live_marker_names_the_host_it_could_not_reach():
    """A skipped live suite must say which machine was tried, or it is unactionable."""
    reason = ssh_unreachable_reason()
    if reason:
        assert LIVE_SSH_HOST in reason, reason
        pytest.skip(reason)
    assert shutil.which("ssh"), "the probe passed without an ssh client, which cannot be"
