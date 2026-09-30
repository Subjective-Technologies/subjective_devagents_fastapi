"""Tests for the agent-turn path — the code that runs when someone actually sends a message.

This is the riskiest logic in the service and it had no coverage at all, which is how a
real failure reached production reporting `{"status": "", "error": ""}` while the datasource
had said `execution_error` one log line earlier.

Nothing here launches a CLI or talks to a model: a stub datasource stands in, so the tests
exercise the turn lifecycle (transcript bookkeeping, rollback, busy-guard, session id,
fallback) rather than any provider's behaviour.
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


@pytest.fixture(autouse=True)
def restore_datasources():
    """datasources.run is module-level state; put the real one back after each test."""
    from subjective_devagents_fastapi import datasources

    original_run, original_supports = datasources.run, datasources.supports_chat
    yield
    datasources.run, datasources.supports_chat = original_run, original_supports


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

    def fake_run(provider, payload, connection=None, on_instance=None):
        if record is not None:
            record.append({"provider": provider, "payload": payload, "connection": connection})
        return dict(result)

    datasources.run = fake_run  # type: ignore[assignment]
    # These tests exercise the in-process module path deliberately, so assert the datasource
    # is chat-capable regardless of which generation happens to be pinned. The routing rule
    # itself is covered by test_a_chat_incapable_datasource_falls_back_to_the_bundle.
    datasources.supports_chat = lambda klass: True  # type: ignore[assignment]
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
    # silently ignored by the CLI, which is worse than failing. The mode is
    # bypassPermissions, not acceptEdits: the CLI runs headless under -p, so a Bash
    # approval prompt has nobody to answer it and the whole turn fails closed.
    assert payload["permission_mode"] == "bypassPermissions"
    assert "sandbox_mode" not in payload


def test_codex_gets_a_sandbox_mode_instead(service):
    calls: list[dict] = []
    conversation = service.create_conversation(provider="codex", name="Codex request")
    stub_datasource(service, {"success": True, "response": "ok"}, record=calls)

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
    stub_datasource(service, {"success": True, "response": "seen", "session_id": "s-files"}, record=calls)

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
    stub_datasource(service, {"success": True, "response": "seen", "session_id": "s-pdf"}, record=calls)

    result = service.send_message(target, "read the spec", [pdf])

    assert result["ok"] is True
    request = calls[0]["payload"]
    assert request["files"] == [pdf]
    stored = _user_message(service, target, "read the spec")
    assert stored["text"] == "read the spec"
    assert pdf not in stored.get("attachments", [])
    assert all(item.get("mime_type") != "application/pdf" for item in stored.get("attachments", []))


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


def test_a_chat_incapable_datasource_falls_back_to_the_bundle(service, monkeypatch):
    """Regression: the pinned codex datasource loads fine but cannot run a turn. Preferring
    it over the working bundle made every codex send return a status object and do nothing —
    the reason codex turns silently stopped working in production."""
    from subjective_devagents_fastapi import datasources

    class ChatIncapable:
        def run(self, request):
            return {"success": True, "status": "ready"}  # a status object, not a turn

    monkeypatch.setattr(datasources, "load", lambda provider: ChatIncapable)
    monkeypatch.setattr(datasources, "supports_chat", lambda klass: False)
    ran: list[str] = []

    def fake_bundle(runtime_path, payload, timeout_seconds):
        ran.append(runtime_path)
        return {"success": True, "result": {"success": True, "response": "ran the command"}}

    service._run_pipeline = fake_bundle  # type: ignore[method-assign]
    conversation = service.create_conversation(provider="codex", name="Codex fallback")

    result = service.send_message(conversation["target"], "run whoami")

    assert ran, "a chat-incapable datasource was used instead of the bundle"
    assert result["ok"] is True
    assert result["assistantMessage"]["text"] == "ran the command"


def test_a_status_call_still_uses_the_module(service, monkeypatch):
    """Only `send` needs the chat contract — status works on both generations, and routing
    it to a subprocess would cost an interpreter start for every poll."""
    from subjective_devagents_fastapi import datasources

    calls: list[str] = []
    monkeypatch.setattr(datasources, "run",
                        lambda provider, payload, connection=None, on_instance=None: calls.append(payload.get("action")) or {"success": True, "status": "ready"})
    service._run_pipeline = lambda *a, **k: pytest.fail("status should not spawn a bundle")  # type: ignore[method-assign]

    service.status("claude")

    assert calls == ["status"]


def test_each_provider_gets_its_own_auto_approve_flag(service, monkeypatch):
    """Each CLI spells 'let the agent run commands' differently, and sending the wrong flag
    is silently ignored by that CLI — so the request must carry the right one per provider.
    Cursor is the one that bit us: without force_commands its sandbox blocks every command."""
    from subjective_devagents_fastapi import datasources

    seen: dict[str, dict] = {}

    # claude runs in-process; codex and cursor route to the bundle (no chat-capable
    # module). The request dict is built the same way for both, so capture at each path
    # rather than assume which one a provider takes.
    def capture_module(provider, payload, connection=None, on_instance=None):
        seen[payload["target"].split(":")[1]] = payload
        return {"success": True, "response": "ok"}

    def capture_bundle(runtime_path, payload, timeout_seconds):
        seen[payload["target"].split(":")[1]] = payload
        return {"success": True, "result": {"success": True, "response": "ok"}}

    monkeypatch.setattr(datasources, "run", capture_module)
    service._run_pipeline = capture_bundle  # type: ignore[method-assign]

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


def test_a_transient_oauth_refresh_collision_is_retried(service, monkeypatch):
    """The Claude CLI refreshes a shared OAuth token and two invocations can collide with
    "another Claude Code process is refreshing it". The CLI calls this transient and asks
    for a retry, so a turn must retry rather than surface a spurious failure — and must not
    double the transcript when it does."""
    from subjective_devagents_fastapi import datasources

    real_error = ("Failed to refresh OAuth token: another Claude Code process is refreshing "
                  "it or exited mid-refresh. This is usually transient; retry in a minute.")
    attempts = {"n": 0}

    def flaky(provider, payload, connection=None, on_instance=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"success": False, "status": "execution_error", "error": real_error}
        return {"success": True, "response": "goldenthinker"}

    monkeypatch.setattr(datasources, "run", flaky)
    monkeypatch.setattr(datasources, "supports_chat", lambda klass: True)
    monkeypatch.setattr("subjective_devagents_fastapi.llm_service.time.sleep", lambda *_: None)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    conversation = service.create_conversation(provider="claude", name="OAuth retry")
    result = service.send_message(conversation["target"], "try ls .. whoami")

    assert attempts["n"] == 2, "the turn was not retried after a transient refresh collision"
    assert result["ok"] is True
    assert result["assistantMessage"]["text"] == "goldenthinker"
    # Exactly one user line and one assistant line — the retry must not double the transcript.
    messages = service.find_conversation(conversation["target"], include_messages=True)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_a_persistent_oauth_error_gives_up_and_reports_it(service, monkeypatch):
    """Retry is bounded: if it never clears, the real error must still reach the user
    rather than looping or vanishing."""
    from subjective_devagents_fastapi import datasources

    real_error = "Failed to refresh OAuth token: another Claude Code process is refreshing it"
    attempts = {"n": 0}

    def always_failing(provider, payload, connection=None, on_instance=None):
        attempts["n"] += 1
        return {"success": False, "status": "execution_error", "error": real_error}

    monkeypatch.setattr(datasources, "run", always_failing)
    monkeypatch.setattr(datasources, "supports_chat", lambda klass: True)
    monkeypatch.setattr("subjective_devagents_fastapi.llm_service.time.sleep", lambda *_: None)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    conversation = service.create_conversation(provider="claude", name="OAuth give up")
    result = service.send_message(conversation["target"], "try ls")

    assert attempts["n"] == 4, "expected one attempt plus three bounded retries"
    assert result["ok"] is False
    assert "refresh OAuth token" in result["error"]


def test_a_slow_turn_emits_heartbeat_progress(service, monkeypatch):
    """A turn is one blocking CLI subprocess call with no live signal of its own — the
    heartbeat is what keeps the UI's "thinking steps" list (and therefore the busy
    indicator) from looking frozen on a turn that legitimately takes a while."""
    from subjective_devagents_fastapi import datasources, llm_service

    monkeypatch.setattr(llm_service, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    def slow(provider, payload, connection=None, on_instance=None):
        time.sleep(0.08)
        return {"success": True, "response": "done"}

    monkeypatch.setattr(datasources, "run", slow)
    monkeypatch.setattr(datasources, "supports_chat", lambda klass: True)
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


def test_an_ordinary_error_is_not_retried(service, monkeypatch):
    """Only the transient refresh signature retries; a real failure must fail once."""
    from subjective_devagents_fastapi import datasources

    attempts = {"n": 0}

    def failing(provider, payload, connection=None, on_instance=None):
        attempts["n"] += 1
        return {"success": False, "status": "execution_error", "error": "command not found: frobnicate"}

    monkeypatch.setattr(datasources, "run", failing)
    monkeypatch.setattr(datasources, "supports_chat", lambda klass: True)
    service._run_pipeline = lambda *a, **k: pytest.fail("should not reach the bundle")  # type: ignore[method-assign]

    conversation = service.create_conversation(provider="claude", name="Ordinary error")
    service.send_message(conversation["target"], "run frobnicate")

    assert attempts["n"] == 1, "a non-transient error must not be retried"
