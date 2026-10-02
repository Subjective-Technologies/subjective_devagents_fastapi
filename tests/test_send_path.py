"""Tests for the agent-turn path — the code that runs when someone actually sends a message.

This is the riskiest logic in the service and it had no coverage at all, which is how a
real failure reached production reporting `{"status": "", "error": ""}` while the datasource
had said `execution_error` one log line earlier.

Nothing here launches a CLI or talks to a model: a stub standalone backend stands in, so the tests
exercise the turn lifecycle (transcript bookkeeping, rollback, busy-guard, session id,
failure reporting) rather than any provider's behaviour.
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    instance = LlmService(root=str(tmp_path / "workspace"), app_home=str(tmp_path / "home"))
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return instance


def stub_backend(service, result, *, record=None):
    """Stub the standalone execution boundary, retaining real turn bookkeeping."""
    def fake_send(conversation, source, request, register, clear, progress):
        if record is not None:
            record.append({"provider": conversation["provider"], "payload": request,
                           "conversation": conversation})
        return dict(result)
    service.backend.send = fake_send
    return service


# --- the bug this file exists for -----------------------------------------------------------


def test_a_failed_turn_reports_why(service):
    """Regression: the datasource said `execution_error`, the caller reported an empty string.

    _run_datasource wrapped every reply in {success, result}, but _pipeline_result only
    unwraps that envelope on success — so a failure buried status/error one level down and
    the user saw "Unknown error"."""
    conversation = service.create_conversation(provider="claude", name="Failing chat")
    stub_backend(service, {
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
    stub_backend(service, {
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
    stub_backend(service, {"success": False, "status": "runtime_error", "error": "boom"})

    service.send_message(target, "this never lands")

    stored = service.find_conversation(target, include_messages=True)
    assert [m for m in stored["messages"] if m["text"] == "this never lands"] == []


def test_a_failed_send_with_an_attachment_is_kept_and_marked(service):
    """The opposite rule for attachments: the screenshot and its failure must stay visible,
    or the user cannot tell the image was ever sent."""
    conversation = service.create_conversation(provider="claude", name="Attachment")
    target = conversation["target"]
    stub_backend(service, {"success": False, "status": "runtime_error", "error": "boom"})

    image = {"filename": "shot.png", "mime_type": "image/png", "content": "aGk="}
    service.send_message(target, "look at this", files=[image])

    stored = service.find_conversation(target, include_messages=True)
    kept = [m for m in stored["messages"] if m["text"] == "look at this"]
    assert len(kept) == 1
    assert kept[0]["sendFailed"] is True
    assert kept[0]["error"]
    assert kept[0]["attachments"] == [image]


def test_a_successful_turn_appends_both_sides(service):
    conversation = service.create_conversation(provider="claude", name="Both sides")
    target = conversation["target"]
    stub_backend(service, {"success": True, "response": "pong", "session_id": "s1"})

    service.send_message(target, "ping")

    messages = service.find_conversation(target, include_messages=True)["messages"]
    assert [(m["role"], m["text"]) for m in messages] == [("user", "ping"), ("assistant", "pong")]


def test_the_session_id_is_persisted_for_the_next_turn(service):
    """Without this every turn starts a new provider session and the agent loses context."""
    conversation = service.create_conversation(provider="claude", name="Session")
    target = conversation["target"]
    stub_backend(service, {"success": True, "response": "ok", "session_id": "sess-abc"})

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


def test_continue_as_a_user_preserves_execution_identity(service):
    conversation = service.create_conversation(provider="claude", name="As other user")
    service.local[0]["osUser"] = "someone-else"
    calls = []
    stub_backend(service, {"success": False, "status": "configuration_required",
                           "error": "Account is unavailable"}, record=calls)
    result = service.send_message(conversation["target"], "run this as them")
    assert result["status"] == "configuration_required"
    assert calls[0]["conversation"]["osUser"] == "someone-else"


# --- the request handed to the provider ----------------------------------------------------------


def test_the_request_carries_what_the_provider_needs(service):
    calls: list[dict] = []
    conversation = service.create_conversation(provider="claude", name="Request shape")
    stub_backend(service, {"success": True, "response": "ok"}, record=calls)

    service.send_message(conversation["target"], "do the thing")

    payload = calls[0]["payload"]
    assert calls[0]["provider"] == "claude"
    assert payload["action"] == "send"
    assert payload["prompt"] == "do the thing"
    assert payload["new_session"] is True, "a fresh chat has no session to resume"
    # Claude takes a permission mode; codex takes a sandbox mode. Sending the wrong one is
    # silently ignored by the CLI, which is worse than failing. The mode is
    # bypassPermissions, not acceptEdits: the CLI runs headless under -p, so a Bash
    # approval prompt has nobody to answer it and the whole turn fails closed.
    assert payload["permission_mode"] == "bypassPermissions"
    assert "sandbox_mode" not in payload


def test_codex_gets_a_sandbox_mode_instead(service):
    calls: list[dict] = []
    conversation = service.create_conversation(provider="codex", name="Codex request")
    stub_backend(service, {"success": True, "response": "ok"}, record=calls)

    service.send_message(conversation["target"], "do the thing")

    payload = calls[0]["payload"]
    assert payload["sandbox_mode"] == "workspace-write"
    assert "permission_mode" not in payload


def _dashboard_file(filename: str, mime_type: str, raw: bytes) -> dict:
    return {
        "filename": filename,
        "mime_type": mime_type,
        "content": base64.b64encode(raw).decode("ascii"),
    }


def _user_message(service, target: str, text: str) -> dict:
    stored = service.find_conversation(target, include_messages=True)
    matches = [message for message in stored["messages"] if message["text"] == text]
    assert len(matches) == 1
    return matches[0]


def test_image_and_text_files_reach_the_pipeline_request(service):
    """The datasource request keeps every file. Stored attachments keep only image/*."""
    calls: list[dict] = []
    conversation = service.create_conversation(provider="claude", name="Files")
    target = conversation["target"]
    image = _dashboard_file("pixel.png", "image/png", b"\x89PNG\r\n\x1a\nfake-image")
    notes = _dashboard_file("notes.txt", "text/plain", b"not an image")
    files = [image, notes]
    stub_backend(service, {"success": True, "response": "seen", "session_id": "s-files"}, record=calls)

    result = service.send_message(target, "see these", files)

    assert result["ok"] is True
    request = calls[0]["payload"]
    assert request["files"] == files
    assert [item["mime_type"] for item in request["files"]] == ["image/png", "text/plain"]
    stored = _user_message(service, target, "see these")
    assert stored["text"] == "see these"
    assert stored["attachments"] == [image]
    assert notes not in stored["attachments"]


def test_a_pdf_reaches_the_pipeline_request_but_not_message_attachments(service):
    calls: list[dict] = []
    conversation = service.create_conversation(provider="claude", name="PDF")
    target = conversation["target"]
    pdf = _dashboard_file("spec.pdf", "application/pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    stub_backend(service, {"success": True, "response": "seen", "session_id": "s-pdf"}, record=calls)

    result = service.send_message(target, "read the spec", [pdf])

    assert result["ok"] is True
    request = calls[0]["payload"]
    assert request["files"] == [pdf]
    stored = _user_message(service, target, "read the spec")
    assert stored["text"] == "read the spec"
    assert pdf not in stored.get("attachments", [])
    assert all(item.get("mime_type") != "application/pdf" for item in stored.get("attachments", []))


def test_backend_failure_does_not_fall_back_to_datasources_or_bundle(service, monkeypatch):
    from subjective_devagents_fastapi import datasources
    monkeypatch.setattr(datasources, "run", lambda *a, **k: pytest.fail("legacy datasource invoked"))
    service._run_pipeline = lambda *a, **k: pytest.fail("legacy bundle invoked")
    calls = []
    stub_backend(service, {"success": False, "status": "unsupported_provider",
                           "error": "Driver unavailable"}, record=calls)
    chat = service.create_conversation(provider="codex", name="No fallback")
    result = service.send_message(chat["target"], "hello")
    assert result["ok"] is False
    assert result["status"] == "unsupported_provider"
    assert len(calls) == 1


def test_the_busy_flag_is_released_after_a_failure(service):
    """A leaked busy flag makes the conversation permanently unusable."""
    conversation = service.create_conversation(provider="claude", name="Busy release")
    target = conversation["target"]
    stub_backend(service, {"success": False, "status": "execution_error", "error": "nope"})

    service.send_message(target, "hello")

    assert target not in service.busy


def test_status_uses_standalone_backend(service, monkeypatch):
    calls = []
    monkeypatch.setattr(service.backend, "status", lambda provider: calls.append(provider) or {"success": True, "status": "ready"})
    service.status("claude")
    assert calls == ["claude"]


def test_each_provider_gets_its_own_auto_approve_flag(service, monkeypatch):
    """Each CLI spells 'let the agent run commands' differently, and sending the wrong flag
    is silently ignored by that CLI — so the request must carry the right one per provider.
    Cursor is the one that bit us: without force_commands its sandbox blocks every command."""
    seen: dict[str, dict] = {}
    def capture(conversation, source, request, register, clear, progress):
        seen[conversation["provider"]] = request
        return {"success": True, "response": "ok"}
    monkeypatch.setattr(service.backend, "send", capture)

    for provider, key, value in [
        ("claude", "permission_mode", "bypassPermissions"),
        ("codex", "sandbox_mode", "workspace-write"),
        ("cursor", "force_commands", True),
        ("grok", "permission_mode", "bypassPermissions"),
    ]:
        conversation = service.create_conversation(provider=provider, name=f"{provider} approve")
        service.send_message(conversation["target"], "run a command")
        assert seen[provider].get(key) == value, f"{provider} did not get {key}={value}"
        # and only its own: sandbox_mode and force_commands are single-provider, so their
        # presence on the wrong provider is the real mistake. permission_mode is shared by
        # claude and grok, so it is not exclusive.
        exclusive = {"sandbox_mode", "force_commands"} - {key}
        assert not (exclusive & seen[provider].keys()), f"{provider} carried a flag meant for another CLI"


def test_a_transient_oauth_refresh_collision_is_not_replayed(service, monkeypatch):
    """Even transient provider errors must surface without replaying a user turn."""
    real_error = ("Failed to refresh OAuth token: another Claude Code process is refreshing "
                  "it or exited mid-refresh. This is usually transient; retry in a minute.")
    attempts = {"n": 0}

    def flaky(conversation, source, request, register, clear, progress):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"success": False, "status": "execution_error", "error": real_error}
        return {"success": True, "response": "goldenthinker"}

    monkeypatch.setattr(service.backend, "send", flaky)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    conversation = service.create_conversation(provider="claude", name="OAuth retry")
    result = service.send_message(conversation["target"], "try ls .. whoami")

    assert attempts["n"] == 1, "failed turns must not replay automatically"
    assert result["ok"] is False
    assert result["error"] == real_error
    # Failed text-only sends retain the existing transcript rollback behavior.
    messages = service.find_conversation(conversation["target"], include_messages=True)["messages"]
    assert messages == []


def test_a_persistent_oauth_error_gives_up_and_reports_it(service, monkeypatch):
    """Persistent authentication errors surface after one execution attempt."""
    real_error = "Failed to refresh OAuth token: another Claude Code process is refreshing it"
    attempts = {"n": 0}

    def always_failing(conversation, source, request, register, clear, progress):
        attempts["n"] += 1
        return {"success": False, "status": "execution_error", "error": real_error}

    monkeypatch.setattr(service.backend, "send", always_failing)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    conversation = service.create_conversation(provider="claude", name="OAuth give up")
    result = service.send_message(conversation["target"], "try ls")

    assert attempts["n"] == 1, "failed turns must not replay automatically"
    assert result["ok"] is False
    assert "refresh OAuth token" in result["error"]


def test_a_slow_turn_emits_heartbeat_progress(service, monkeypatch):
    """A turn is one blocking CLI subprocess call with no live signal of its own — the
    heartbeat is what keeps the UI's "thinking steps" list (and therefore the busy
    indicator) from looking frozen on a turn that legitimately takes a while."""
    from subjective_devagents_fastapi import llm_service

    monkeypatch.setattr(llm_service, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    def slow(conversation, source, request, register, clear, progress):
        time.sleep(0.08)
        return {"success": True, "response": "done"}

    monkeypatch.setattr(service.backend, "send", slow)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    events: list[dict] = []
    service.on_progress = events.append

    conversation = service.create_conversation(provider="claude", name="Heartbeat check")
    result = service.send_message(conversation["target"], "do something slow")

    assert result["ok"] is True
    assert events, "expected at least one heartbeat while the turn was in flight"
    assert all(event["target"] == conversation["target"] for event in events)
    assert all("still working" in event["text"] for event in events)

    # The heartbeat thread must not keep firing (or crash on a torn-down service) after
    # the turn already returned.
    count_after_return = len(events)
    time.sleep(0.05)
    assert len(events) == count_after_return


def test_real_steps_suppress_the_heartbeat_and_reach_the_progress_feed(service, monkeypatch):
    """While the agent reports what it is doing, timer rows would only interleave noise."""
    from subjective_devagents_fastapi import llm_service

    monkeypatch.setattr(llm_service, "HEARTBEAT_INTERVAL_SECONDS", 0.05)

    def busy(conversation, source, request, register, clear, progress):
        for index in range(8):
            progress({"target": conversation["target"], "text": f"Running: step {index}"})
            time.sleep(0.02)
        return {"success": True, "response": "done"}

    monkeypatch.setattr(service.backend, "send", busy)
    events: list[dict] = []
    service.on_progress = events.append

    conversation = service.create_conversation(provider="codex", name="Steps")
    assert service.send_message(conversation["target"], "work")["ok"] is True
    assert [event["text"] for event in events] == [f"Running: step {index}" for index in range(8)]


def test_an_ordinary_error_is_not_retried(service, monkeypatch):
    """Ordinary execution errors also fail once without replay."""
    attempts = {"n": 0}

    def failing(conversation, source, request, register, clear, progress):
        attempts["n"] += 1
        return {"success": False, "status": "execution_error", "error": "command not found: frobnicate"}

    monkeypatch.setattr(service.backend, "send", failing)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    conversation = service.create_conversation(provider="claude", name="Ordinary error")
    service.send_message(conversation["target"], "run frobnicate")

    assert attempts["n"] == 1, "a non-transient error must not be retried"
