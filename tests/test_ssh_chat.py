"""Tests for SSH chats — a conversation whose agent is a shell on another machine.

Three bugs shipped here before there were tests, and every one of them looked like
success: the send reported `ok`, the transcript grew, and the reply was empty or a
turn stale. They are the reason each section below exists.

- The setup handshake matched the tty's *echo* of its own commands, so every turn
  read the previous turn's output.
- A leftover prompt marker ended the next turn instantly, before the command had
  produced anything.
- The result was returned without the `result` envelope `_pipeline_result` unwraps,
  so every key including `response` was dropped on the way to the transcript.

The shell under test is real. A fake ssh client runs `bash` locally, so turn framing,
state persistence and prompt handling are exercised against an actual shell rather
than a mock of one.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import ssh_chat  # noqa: E402
from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402

LIVE_HOST = os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_TEST_HOST", "subjective-agents")

# ssh argv: -o/-i/-p take a value, the first bare word is the target, anything after
# is the remote command. A terminal session sends no command, so the login shell runs.
FAKE_SSH = r"""#!/bin/sh
target=""
remote=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o|-i|-p) shift 2 ;;
    -*) shift ;;
    *) if [ -z "$target" ]; then target="$1"; else remote="$1"; fi; shift ;;
  esac
done
if [ -n "$remote" ]; then exec bash --norc --noprofile -c "$remote"; fi
exec bash --norc --noprofile -i
"""


@pytest.fixture(autouse=True)
def terminal_enabled(monkeypatch):
    """ssh_chat goes through ssh_terminal, which fails closed by design."""
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED", "1")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", "test-token")
    yield
    ssh_chat.sessions.close_all()


@pytest.fixture()
def fake_ssh(tmp_path):
    script = tmp_path / "fake-ssh"
    script.write_text(FAKE_SSH, encoding="utf-8")
    script.chmod(0o755)
    return str(script)


@pytest.fixture()
def connection(fake_ssh):
    # login_shell is irrelevant here (a terminal session runs the login shell either
    # way); ssh_path is what points the datasource at the stand-in.
    return {"host": "example.test", "ssh_path": fake_ssh}


def say(target, connection, text, timeout=30):
    return ssh_chat.sessions.send(target, connection, text, timeout=timeout)


# --- cleaning ---------------------------------------------------------------------


def test_escape_sequences_are_stripped_for_a_chat_bubble():
    dirty = "\x1b[32mgreen\x1b[0m\x1b]0;a title\x07 plain"
    assert ssh_chat.clean_terminal_text(dirty) == "green plain"


def test_crlf_becomes_lf():
    assert ssh_chat.clean_terminal_text("a\r\nb\r\n") == "a\nb\n"


def test_a_line_redrawn_in_place_keeps_only_its_final_state():
    """A bare \\r is a program overwriting the line it is already on."""
    assert ssh_chat.clean_terminal_text("50%\r100%\ndone\n") == "100%\ndone\n"


# --- turns ------------------------------------------------------------------------


def test_a_turn_returns_only_the_commands_output(connection):
    """No echo of what was typed, no prompt, no marker — just the answer.

    This is the echo regression: with the markers written whole, setup matched its
    own echo and this came back as the text of the command instead.
    """
    result = say("t", connection, "echo hello-from-chat")
    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["response"] == "hello-from-chat"


def test_no_internal_marker_ever_reaches_the_reply(connection):
    result = say("t", connection, "echo marker-check")
    assert ssh_chat.TURN_MARK not in result["response"]
    assert ssh_chat.READY_MARK not in result["response"]


def test_each_turn_answers_its_own_message(connection):
    """The stale-turn regression: replies were one message behind."""
    assert say("t", connection, "echo first")["response"] == "first"
    assert say("t", connection, "echo second")["response"] == "second"
    assert say("t", connection, "echo third")["response"] == "third"


def test_shell_state_survives_between_messages(connection):
    """The whole point of a chat-shaped session rather than a command per turn."""
    say("t", connection, "MARKER=persisted")
    assert say("t", connection, "echo $MARKER")["response"] == "persisted"


def test_the_working_directory_survives_between_messages(connection, tmp_path):
    target_dir = tmp_path / "somewhere"
    target_dir.mkdir()
    say("t", connection, f"cd {target_dir}")
    assert os.path.realpath(say("t", connection, "pwd")["response"]) == os.path.realpath(target_dir)


def test_a_command_with_no_output_says_so_rather_than_nothing(connection):
    """An empty bubble reads as a failure; this is the state the bugs produced."""
    assert say("t", connection, "true")["response"] == "(no output)"


def test_a_failing_command_still_answers(connection):
    result = say("t", connection, "ls /nonexistent-path-for-tests")
    assert result["success"] is True  # the chat worked; the command did not
    assert "No such file" in result["response"]


def test_multiline_output_is_kept_whole(connection):
    assert say("t", connection, "printf 'a\\nb\\nc\\n'")["response"] == "a\nb\nc"


def test_a_slow_command_is_waited_for(connection):
    started = time.time()
    result = say("t", connection, "sleep 2; echo after-the-wait")
    assert result["response"] == "after-the-wait"
    assert time.time() - started >= 2


# --- sessions ---------------------------------------------------------------------


def test_one_shell_per_conversation(connection):
    """Two chats to the same machine are two shells, not one shared one."""
    say("one", connection, "WHO=first")
    say("two", connection, "WHO=second")
    assert say("one", connection, "echo $WHO")["response"] == "first"
    assert say("two", connection, "echo $WHO")["response"] == "second"


def test_a_session_is_reused_rather_than_reopened(connection):
    say("t", connection, "true")
    opened = ssh_chat.sessions.snapshot()[0]["openedAt"]
    say("t", connection, "true")
    assert ssh_chat.sessions.snapshot()[0]["openedAt"] == opened


def test_closing_a_chat_ends_its_shell(connection):
    say("t", connection, "MARKER=gone")
    ssh_chat.sessions.close(target := "t")
    assert ssh_chat.sessions.snapshot() == []
    # A new shell, so nothing from the old one survives.
    assert say(target, connection, "echo $MARKER")["response"] == "(no output)"


def test_the_snapshot_never_carries_credentials(connection):
    say("t", {**connection, "password": "hunter2"}, "true")
    listed = ssh_chat.sessions.snapshot()
    assert listed and "password" not in listed[0]["connection"]
    assert "hunter2" not in str(listed)


def test_a_host_that_cannot_be_reached_is_reported_not_raised(tmp_path):
    failing = tmp_path / "failing-ssh"
    failing.write_text(
        "#!/bin/sh\necho 'ssh: Could not resolve hostname nope' >&2\nexit 255\n", encoding="utf-8"
    )
    failing.chmod(0o755)
    result = say("t", {"host": "nope", "ssh_path": str(failing)}, "echo hi")
    assert result["success"] is False
    assert result["status"] == "connection_failed"
    assert result["error"]


# --- where the host comes from ------------------------------------------------------


def test_the_target_is_read_from_the_conversations_workspace():
    connection = ssh_chat.connection_for_conversation({"workspace": "deploy@server:/srv/app"})
    assert connection["host"] == "server"
    assert connection["username"] == "deploy"
    assert connection["working_directory"] == "/srv/app"


def test_a_bare_alias_is_a_complete_target():
    assert ssh_chat.connection_for_conversation({"workspace": "subjective-agents"})["host"] == "subjective-agents"


def test_a_chat_with_no_host_says_what_to_do():
    with pytest.raises(ValueError) as caught:
        ssh_chat.connection_for_conversation({"workspace": ""})
    assert "workspace" in str(caught.value)


# --- the envelope -------------------------------------------------------------------


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return LlmService(root=str(tmp_path / "workspace"), app_home=str(tmp_path / "home"))


def test_the_reply_reaches_the_transcript(service, monkeypatch):
    """The envelope regression, pinned end to end through send_message.

    `_pipeline_result` unwraps a `result` key. Returning the shell's result without
    that envelope dropped every field it carried, so the send reported success and
    wrote an empty assistant message — which is indistinguishable, in the UI, from a
    command that printed nothing.
    """
    monkeypatch.setattr(
        ssh_chat.sessions,
        "send",
        lambda target, connection, text, timeout=None: {
            "success": True,
            "status": "completed",
            "response": f"answer to {text}",
            "error": "",
        },
    )
    conversation = service.create_conversation(provider="ssh", name="Box", workspace="subjective-agents")
    target = conversation["target"]

    outcome = service.send_message(target, "hostname")
    assert outcome["ok"] is True, outcome
    assert outcome["assistantMessage"]["text"] == "answer to hostname"


def test_a_chat_without_a_host_fails_before_connecting(service, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ssh_chat.sessions,
        "send",
        lambda *args, **kwargs: calls.append(args) or {"success": True, "response": ""},
    )
    conversation = service.create_conversation(provider="ssh", name="Box", workspace="")
    outcome = service.send_message(conversation["target"], "hostname")
    assert outcome["ok"] is False
    assert outcome["status"] == "configuration_required"
    assert calls == []


def test_ssh_is_offered_as_a_creatable_agent():
    from subjective_devagents_fastapi.llm_service import WRITABLE_PROVIDERS

    assert WRITABLE_PROVIDERS["ssh"]["label"] == "SSH"


# --- the real thing -------------------------------------------------------------------


def live_host_reachable() -> bool:
    import subprocess

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
    def test_a_conversation_with_a_real_machine(self):
        connection = ssh_chat.connection_for_conversation({"workspace": LIVE_HOST})
        assert say("live", connection, "hostname")["response"] == LIVE_HOST
        say("live", connection, "cd /tmp && MARKER=kept")
        assert say("live", connection, "echo $MARKER; pwd")["response"] == "kept\n/tmp"
        assert "No such file" in say("live", connection, "ls /nonexistent")["response"]


# --- what a failure says --------------------------------------------------------------


def test_a_refused_connection_reports_what_ssh_said(tmp_path):
    """A generic "did not respond" threw away the one thing worth knowing.

    ssh has almost always already explained itself — a rejected key, an unknown host
    key, a refused port — and reporting a timeout instead left nothing to act on.
    Production hit exactly this: the real message was "Host key verification failed"
    and the operator saw "The remote shell did not respond."
    """
    failing = tmp_path / "failing-ssh"
    failing.write_text(
        "#!/bin/sh\n"
        "echo 'No ED25519 host key is known for 10.0.0.1 and you have requested strict checking.' >&2\n"
        "echo 'Host key verification failed.' >&2\n"
        "exit 255\n",
        encoding="utf-8",
    )
    failing.chmod(0o755)
    result = say("t", {"host": "10.0.0.1", "ssh_path": str(failing)}, "hostname")
    assert result["status"] == "connection_failed"
    assert "Host key verification failed" in result["error"]
    assert "did not respond" not in result["error"]


def test_the_diagnostic_does_not_quote_our_own_setup_commands(tmp_path):
    """Echo is on during setup, so the transcript contains the marker commands the
    shell was sent — as split spellings no whole-marker filter would catch."""
    failing = tmp_path / "failing-ssh"
    failing.write_text(
        "#!/bin/sh\nread line\necho \"$line\"\necho 'Permission denied (publickey).' >&2\nexit 255\n",
        encoding="utf-8",
    )
    failing.chmod(0o755)
    result = say("t", {"host": "h", "ssh_path": str(failing)}, "hostname")
    assert "Permission denied" in result["error"]
    assert "SUBJECTIVE_SSH" not in result["error"]


# --- reading back what the terminal drew ------------------------------------------------


def test_ordinary_output_keeps_its_columns():
    """`ls` lines up with runs of spaces; anything that reflows them destroys it."""
    listing = "AGENTS.md    libs         README.md\nartifacts    pipelines    run.sh\n"
    assert ssh_chat.render_terminal_text(listing) == listing


def test_colour_is_removed_without_disturbing_the_layout():
    coloured = "\x1b[34mdir\x1b[0m        \x1b[32mfile\x1b[0m\n"
    assert ssh_chat.render_terminal_text(coloured) == "dir        file\n"


def test_long_output_is_never_truncated_to_a_screen():
    """A build log is not a screen; emulating it would drop everything that scrolled."""
    long_output = "".join(f"line {index}\n" for index in range(500))
    rendered = ssh_chat.render_terminal_text(long_output, cols=80, rows=24)
    assert rendered.count("\n") == 500
    assert "line 0" in rendered and "line 499" in rendered


def test_positioned_text_is_drawn_rather_than_concatenated():
    """The bug this fixes: stripping the escapes ran the words together, so a
    prompt arrived as `Doyoutrustthecontentsofthisdirectory?`."""
    drawn = "\x1b[2J\x1b[1;1HDo you trust\x1b[3;3Hthe contents\x1b[5;1H1. Yes"
    rendered = ssh_chat.render_terminal_text(drawn, cols=40, rows=10)
    assert rendered.splitlines() == ["Do you trust", "", "  the contents", "", "1. Yes"]


def test_text_overwritten_in_place_shows_its_final_state():
    """A TUI redraws the same cell; the last write is what was on screen."""
    redrawn = "\x1b[2J\x1b[1;1Hloading\x1b[1;1Hdone   "
    assert ssh_chat.render_terminal_text(redrawn, cols=20, rows=5).strip() == "done"


def test_the_alternate_screen_counts_as_drawing():
    full_screen = "\x1b[?1049h\x1b[2;4Hinside the full-screen app"
    rendered = ssh_chat.render_terminal_text(full_screen, cols=40, rows=6)
    assert "inside the full-screen app" in rendered


def test_blank_padding_around_a_drawn_screen_is_dropped():
    """Most of a drawn screen is the terminal's empty rows, not the program's."""
    drawn = "\x1b[2J\x1b[4;1Honly this line"
    assert ssh_chat.render_terminal_text(drawn, cols=30, rows=20) == "only this line"


def test_malformed_output_still_produces_a_reply():
    """A turn must not be lost to a byte sequence no terminal would emit."""
    broken = "\x1b[2J\x1b[999999999999;1Hstill here\x1b[\x1b[\x1b"
    assert "still here" in ssh_chat.render_terminal_text(broken, cols=40, rows=10)


# --- answering a password prompt ---------------------------------------------------


@pytest.mark.parametrize("line", [
    "[sudo] password for worktwins:",
    "worktwins@host's password:",
    "Enter passphrase for key '/home/x/.ssh/id_ed25519':",
    "Password:",
    "password: ",
    "Verification code:",
])
def test_a_prompt_for_a_secret_is_recognised(line):
    assert ssh_chat.secret_prompt(line) == line.strip()


@pytest.mark.parametrize("line", [
    "total 48",
    "The password is stored in the vault",
    "PASSWORD_FILE=/etc/secret",
    "",
])
def test_ordinary_output_is_not_mistaken_for_a_prompt(line):
    assert ssh_chat.secret_prompt(line) == ""


def test_only_the_last_line_counts():
    """The prompt is the last thing written before the program blocks; the word
    appearing earlier in a log means nothing."""
    assert ssh_chat.secret_prompt("Password:\nall done\n") == ""
    assert ssh_chat.secret_prompt("checking\nPassword:") == "Password:"


def test_a_completed_command_never_asks_for_a_secret(connection):
    """The shell being back at its prompt means nothing is waiting, whatever the
    output happens to say — otherwise `echo Password:` would open the dialog."""
    result = say("t", connection, "echo 'Password:'")
    assert result["response"] == "Password:"
    assert result["awaitingSecret"] == ""


def test_a_blocked_prompt_is_reported(connection):
    result = say("t", connection, 'read -s -p "Password: " pw')
    assert result["awaitingProgram"] is True
    assert result["awaitingSecret"] == "Password:"


def test_the_secret_reaches_the_program_and_is_not_echoed(connection):
    say("t", connection, 'read -s -p "Password: " pw')
    result = ssh_chat.sessions.send_secret("t", "hunter2", timeout=30)
    assert result["success"] is True
    # The program chose to reveal the length; nothing echoed the secret itself.
    assert "hunter2" not in result["response"]
    check = say("t", connection, 'echo "${#pw}"')
    assert check["response"] == "7"


def test_a_secret_is_never_sent_to_a_chat_with_no_shell():
    """A secret is an answer to something already asking; it must not open a
    session and type a password into a fresh login shell."""
    result = ssh_chat.sessions.send_secret("never-opened", "hunter2", timeout=5)
    assert result["success"] is False
    assert result["status"] == "no_session"


def test_the_transcript_gets_a_mask_not_the_secret(service, monkeypatch):
    monkeypatch.setattr(
        ssh_chat.sessions, "send",
        lambda target, connection, text, timeout=None: {
            "success": True, "status": "completed", "response": "Password:",
            "error": "", "awaitingSecret": "Password:",
        },
    )
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: {
            "success": True, "status": "completed", "response": "Welcome.", "error": "",
        },
    )
    conversation = service.create_conversation(provider="ssh", name="Box", workspace="host")
    target = conversation["target"]
    first = service.send_message(target, "sudo -S id")
    assert first["assistantMessage"]["awaitingSecret"] == "Password:"

    outcome = service.send_ssh_secret(target, "hunter2")
    assert outcome["ok"] is True
    assert outcome["assistantMessage"]["text"] == "Welcome."
    stored = service.find_conversation(target, include_messages=True)["messages"]
    assert "hunter2" not in str(stored)
    assert stored[-2]["text"] == "••••••••"
    assert stored[-2]["secretInput"] is True


def test_a_secret_is_refused_for_a_chat_that_is_not_ssh(service):
    conversation = service.create_conversation(provider="claude", name="Not ssh")
    outcome = service.send_ssh_secret(conversation["target"], "hunter2")
    assert outcome["ok"] is False
    assert outcome["status"] == "read_only"
