"""Tests for the agent-turn path — the code that runs when someone actually sends a message.

This is the riskiest logic in the service and it had no coverage at all, which is how a
real failure reached production reporting `{"status": "", "error": ""}` while the datasource
had said `execution_error` one log line earlier.

Nothing here launches a CLI or talks to a model: a stub datasource stands in, so the tests
exercise the turn lifecycle (transcript bookkeeping, rollback, busy-guard, session id,
fallback) rather than any provider's behaviour.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402


@pytest.fixture(autouse=True)
def restore_datasources():
    """datasources.run is module-level state; put the real one back after each test."""
    from subjective_devagents_fastapi import datasources

    original = datasources.run
    yield
    datasources.run = original


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    instance = LlmService(root=str(tmp_path / "workspace"), app_home=str(tmp_path / "home"))
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return instance


def stub_datasource(service, result, *, record=None):
    """Stand in for the datasource package, returning exactly what its `run()` returns.

    Patched at `datasources.run` rather than at `_run_datasource`: the envelope handling
    that turns a provider reply into what the caller sees lives *inside* `_run_datasource`,
    so stubbing that method would test the stub instead of the code — which is how the
    empty-error bug survived a green suite the first time these tests were written.
    """
    from subjective_devagents_fastapi import datasources

    def fake_run(provider, payload, connection=None):
        if record is not None:
            record.append({"provider": provider, "payload": payload, "connection": connection})
        return dict(result)

    datasources.run = fake_run  # type: ignore[assignment]
    # Any fall-through to the bundle would be a silent subprocess; make it loud instead.
    service._run_pipeline = lambda *a, **k: pytest.fail("fell through to the exported bundle")  # type: ignore[method-assign]
    return service


# --- the bug this file exists for -----------------------------------------------------------


def test_a_failed_turn_reports_why(service):
    """Regression: the datasource said `execution_error`, the caller reported an empty string.

    _run_datasource wrapped every reply in {success, result}, but _pipeline_result only
    unwraps that envelope on success — so a failure buried status/error one level down and
    the user saw "Unknown error"."""
    conversation = service.create_conversation(provider="claude", name="Failing chat")
    stub_datasource(service, {
        "success": False,
        "status": "execution_error",
        "error": "claude exited 1: context deadline exceeded",
    })

    result = service.send_message(conversation["target"], "hello")

    assert result["ok"] is False
    assert result["status"] == "execution_error", "the datasource's status must survive"
    assert "context deadline exceeded" in result["error"], "the datasource's error must survive"


def test_a_successful_turn_still_unwraps_the_envelope(service):
    """The other half of that fix: success must keep reaching through the wrapper."""
    conversation = service.create_conversation(provider="claude", name="Working chat")
    stub_datasource(service, {
        "success": True,
        "status": "completed",
        "response": "Here is your answer.",
        "session_id": "sess-123",
    })

    result = service.send_message(conversation["target"], "hello")

    assert result["ok"] is True
    assert result["assistantMessage"]["text"] == "Here is your answer."
    assert result["result"]["session_id"] == "sess-123"


# --- transcript bookkeeping ------------------------------------------------------------------


def test_a_failed_send_is_rolled_out_of_the_transcript(service):
    """An unsent message left in the transcript reads exactly like a delivered one."""
    conversation = service.create_conversation(provider="claude", name="Rollback")
    target = conversation["target"]
    stub_datasource(service, {"success": False, "status": "runtime_error", "error": "boom"})

    service.send_message(target, "this never lands")

    stored = service.find_conversation(target, include_messages=True)
    assert [m for m in stored["messages"] if m["text"] == "this never lands"] == []


def test_a_failed_send_with_an_attachment_is_kept_and_marked(service):
    """The opposite rule for attachments: the screenshot and its failure must stay visible,
    or the user cannot tell the image was ever sent."""
    conversation = service.create_conversation(provider="claude", name="Attachment")
    target = conversation["target"]
    stub_datasource(service, {"success": False, "status": "runtime_error", "error": "boom"})

    service.send_message(target, "look at this", files=[
        {"filename": "shot.png", "mime_type": "image/png", "content": "aGk="},
    ])

    stored = service.find_conversation(target, include_messages=True)
    kept = [m for m in stored["messages"] if m["text"] == "look at this"]
    assert len(kept) == 1
    assert kept[0]["sendFailed"] is True
    assert kept[0]["error"]


def test_a_successful_turn_appends_both_sides(service):
    conversation = service.create_conversation(provider="claude", name="Both sides")
    target = conversation["target"]
    stub_datasource(service, {"success": True, "response": "pong", "session_id": "s1"})

    service.send_message(target, "ping")

    messages = service.find_conversation(target, include_messages=True)["messages"]
    assert [(m["role"], m["text"]) for m in messages] == [("user", "ping"), ("assistant", "pong")]


def test_the_session_id_is_persisted_for_the_next_turn(service):
    """Without this every turn starts a new provider session and the agent loses context."""
    conversation = service.create_conversation(provider="claude", name="Session")
    target = conversation["target"]
    stub_datasource(service, {"success": True, "response": "ok", "session_id": "sess-abc"})

    service.send_message(target, "first")

    assert service.find_conversation(target)["sessionId"] == "sess-abc"


# --- guards ------------------------------------------------------------------------------------


def test_an_empty_message_is_rejected_before_anything_is_recorded(service):
    conversation = service.create_conversation(provider="claude", name="Empty")
    result = service.send_message(conversation["target"], "   ")
    assert result == {"ok": False, "status": "invalid_request", "error": "Message is empty."}


def test_an_unknown_target_is_rejected(service):
    result = service.send_message("llm:claude:local:nope", "hello")
    assert result["status"] == "not_found"


def test_a_read_only_provider_is_rejected(service):
    """History from a provider with no writable datasource cannot be continued."""
    conversation = service._normalize_conversation({
        "target": "llm:gemini:local:read-only",
        "provider": "gemini",
        "label": "Imported gemini chat",
        "messages": [],
    }, "local")
    service.local.insert(0, conversation)

    result = service.send_message("llm:gemini:local:read-only", "hello")
    assert result["status"] == "read_only"


def test_a_second_send_while_busy_is_refused(service):
    """Two turns in one conversation would interleave in the provider's session."""
    conversation = service.create_conversation(provider="claude", name="Busy")
    target = conversation["target"]
    service.busy.add(target)
    try:
        result = service.send_message(target, "hello")
    finally:
        service.busy.discard(target)
    assert result["status"] == "busy"


def test_continue_as_a_user_whose_session_is_not_ready_is_refused(service):
    """A "Continue as <user>" chat must not fall back to running as the service account —
    and must be refused before the message is written to the transcript."""
    conversation = service.create_conversation(provider="claude", name="As other user")
    target = conversation["target"]
    service.local[0]["osUser"] = "someone-else"

    class NotReady:
        def get_status(self, username):
            return {"username": username, "status": "idle"}

    service.su_session_manager = NotReady()

    result = service.send_message(target, "run this as them")

    assert result["status"] == "os_user_not_connected"
    stored = service.find_conversation(target, include_messages=True)
    assert [m for m in stored["messages"] if m["text"] == "run this as them"] == []


# --- the request handed to the provider ----------------------------------------------------------


def test_the_request_carries_what_the_provider_needs(service):
    calls: list[dict] = []
    conversation = service.create_conversation(provider="claude", name="Request shape")
    stub_datasource(service, {"success": True, "response": "ok"}, record=calls)

    service.send_message(conversation["target"], "do the thing")

    payload = calls[0]["payload"]
    assert calls[0]["provider"] == "claude"
    assert payload["action"] == "send"
    assert payload["prompt"] == "do the thing"
    assert payload["new_session"] is True, "a fresh chat has no session to resume"
    # Claude takes a permission mode; codex takes a sandbox mode. Sending the wrong one is
    # silently ignored by the CLI, which is worse than failing.
    assert payload["permission_mode"] == "acceptEdits"
    assert "sandbox_mode" not in payload


def test_codex_gets_a_sandbox_mode_instead(service):
    calls: list[dict] = []
    conversation = service.create_conversation(provider="codex", name="Codex request")
    stub_datasource(service, {"success": True, "response": "ok"}, record=calls)

    service.send_message(conversation["target"], "do the thing")

    payload = calls[0]["payload"]
    assert payload["sandbox_mode"] == "workspace-write"
    assert "permission_mode" not in payload


def test_the_bundle_runs_when_the_datasource_module_is_unavailable(service):
    """The resilience claim: a missing plugin must degrade to the exported bundle rather
    than fail the turn. Cursor does exactly this in production today."""
    conversation = service.create_conversation(provider="codex", name="Fallback")
    ran: list[str] = []

    service._run_datasource = lambda *a, **k: None  # type: ignore[method-assign]

    def fake_bundle(runtime_path, payload, timeout_seconds):
        ran.append(runtime_path)
        return {"success": True, "result": {"success": True, "response": "from the bundle"}}

    service._run_pipeline = fake_bundle  # type: ignore[method-assign]

    result = service.send_message(conversation["target"], "hello")

    assert ran, "the bundle was never reached"
    assert result["ok"] is True
    assert result["assistantMessage"]["text"] == "from the bundle"


def test_the_busy_flag_is_released_after_a_failure(service):
    """A leaked busy flag makes the conversation permanently unusable."""
    conversation = service.create_conversation(provider="claude", name="Busy release")
    target = conversation["target"]
    stub_datasource(service, {"success": False, "status": "execution_error", "error": "nope"})

    service.send_message(target, "hello")

    assert target not in service.busy
