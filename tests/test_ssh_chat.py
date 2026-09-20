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
