"""H. The routes no test exercised at all.

Twelve handlers that no unit test called and no browser run reached. Each gets a
success and a refusal, because the interesting half of an untested route is what it
does with a body it did not expect — a missing field, a string where a number belongs,
a target that is not there.

`/api/llm/reload-conversation` (12 statements) and `/api/switchuser/pty-resize` (7)
carry the most logic and get more than a smoke test.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from .helpers import envelope, refusal

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


# --- /api/agents/role (GET and POST) ----------------------------------------------------


def test_h_agent_role_is_read_back(client):
    """The fixture registry holds one agent; its role file is what comes back."""
    body = envelope(client.get("/api/agents/role", params={"agent": "pytest_agent"}), ok=True)
    assert body["id"] == "pytest_agent"
    assert "Own automated verification" in body["text"]
    assert "model" in body


@pytest.mark.parametrize(
    "agent,expect",
    [
        pytest.param("", "invalid", id="empty"),
        pytest.param("../../../etc/passwd", "invalid", id="traversal"),
        pytest.param("has spaces", "invalid", id="spaces"),
        pytest.param("no_such_agent", "no role file", id="unknown"),
    ],
)
def test_h_reading_a_bad_agent_role_is_refused(client, agent, expect):
    error = refusal(client.get("/api/agents/role", params={"agent": agent}))
    assert expect in error.lower(), error


def test_h_an_agent_role_can_be_saved_and_read_back(client):
    replacement = "# pytest_agent\n\n## Mission\n\nRewritten by the API suite.\n"
    saved = envelope(
        client.post("/api/agents/role", json={"id": "pytest_agent", "text": replacement}), ok=True
    )
    assert saved["text"] == replacement

    reread = envelope(client.get("/api/agents/role", params={"agent": "pytest_agent"}), ok=True)
    assert reread["text"] == replacement


def test_h_saving_a_role_writes_the_file_not_just_the_response(client):
    roles = Path(client.tmp_path) / "workflow-agents" / "roles" / "pytest_agent.md"
    envelope(client.post("/api/agents/role", json={"id": "pytest_agent", "text": "# changed\n"}), ok=True)
    assert roles.read_text(encoding="utf-8") == "# changed\n"


@pytest.mark.parametrize(
    "body,expect",
    [
        pytest.param({"id": "pytest_agent", "text": ""}, "cannot be empty", id="empty-text"),
        pytest.param({"id": "pytest_agent", "text": "   \n "}, "cannot be empty", id="blank-text"),
        pytest.param({"id": "never_registered", "text": "hello"}, "no role file", id="unregistered"),
        pytest.param({"id": "../escape", "text": "hello"}, "invalid", id="traversal"),
        pytest.param({}, "invalid", id="nothing-at-all"),
    ],
)
def test_h_a_bad_role_save_is_refused_and_writes_nothing(client, body, expect):
    roles = Path(client.tmp_path) / "workflow-agents" / "roles" / "pytest_agent.md"
    before = roles.read_text(encoding="utf-8")
    error = refusal(client.post("/api/agents/role", json=body))
    assert expect in error.lower(), error
    assert roles.read_text(encoding="utf-8") == before, "a refused save changed the file"


def test_h_saving_a_role_never_creates_a_new_file(client):
    """A typo'd id must not leave an orphan with no registry row."""
    roles_dir = Path(client.tmp_path) / "workflow-agents" / "roles"
    before = sorted(path.name for path in roles_dir.iterdir())
    refusal(client.post("/api/agents/role", json={"id": "typoed_agent", "text": "hello"}))
    assert sorted(path.name for path in roles_dir.iterdir()) == before


# --- /api/connect and /api/disconnect ----------------------------------------------------


def test_h_connect_answers_immediately_and_asks_core_to_reconnect(client, monkeypatch):
    """The handler must not wait on a TCP handshake: it is an HTTP request.

    `reconnect_now` is spied rather than left to dial, so the test makes no outbound
    connection of its own.
    """
    from subjective_devagents_fastapi.app import state

    called = []
    monkeypatch.setattr(state.core, "reconnect_now", lambda: called.append(True))

    assert envelope(client.post("/api/connect"), ok=True) == {"ok": True}
    assert called == [True], "the route did not ask core to reconnect"


def test_h_connect_takes_no_body_and_ignores_one(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    monkeypatch.setattr(state.core, "reconnect_now", lambda: None)
    assert envelope(client.post("/api/connect", json={"host": "ignored"}), ok=True) == {"ok": True}


def test_h_disconnect_is_a_user_requested_disconnect(client, monkeypatch):
    """The flag matters: a user-requested disconnect must not auto-reconnect."""
    from subjective_devagents_fastapi.app import state

    seen = []
    monkeypatch.setattr(state.core, "disconnect_irc", lambda user_requested: seen.append(user_requested))

    assert envelope(client.post("/api/disconnect"), ok=True) == {"ok": True}
    assert seen == [True], f"disconnect was not marked user-requested: {seen}"


def test_h_disconnect_on_an_idle_service_is_harmless(client):
    """Nothing is connected — IRC is off in this fixture — and it still answers cleanly."""
    assert envelope(client.post("/api/disconnect"), ok=True) == {"ok": True}
    assert envelope(client.post("/api/disconnect"), ok=True) == {"ok": True}


# --- /api/d1/sync (GET and POST) ----------------------------------------------------------


def test_h_d1_sync_status_starts_empty(client):
    body = envelope(client.get("/api/d1/sync"), ok=True)
    assert "last" in body, body


def test_h_d1_sync_is_skipped_without_credentials_rather_than_failing(client):
    """Local JSON is the live store; D1 is a replica, so its absence is not an error."""
    body = envelope(client.post("/api/d1/sync"), ok=True)
    assert body["skipped"] is True
    assert body["reason"] == "disabled_or_no_credentials"


def test_h_d1_sync_status_reports_the_last_run(client):
    envelope(client.post("/api/d1/sync"), ok=True)
    body = envelope(client.get("/api/d1/sync"), ok=True)
    assert body["last"]["skipped"] is True, body


# --- /api/dcc/send-image-data ---------------------------------------------------------------


def test_h_sending_an_image_while_unregistered_is_refused(client):
    """No IRC registration, so there is nobody to send to — and it must say so, not throw."""
    body = envelope(
        client.post("/api/dcc/send-image-data", json={
            "target": "#somewhere",
            "bytes": base64.b64encode(PNG_BYTES).decode("ascii"),
            "fileName": "pasted.png",
            "mimeType": "image/png",
        }),
        ok=False,
    )
    assert body == {"ok": False}


def test_h_image_bytes_are_accepted_as_base64_or_as_a_byte_array(client, monkeypatch):
    """The browser sends one or the other; both have to arrive as the same bytes."""
    from subjective_devagents_fastapi.app import state

    seen: list[bytes] = []
    monkeypatch.setattr(
        state.core, "send_dcc_buffer",
        lambda target, buffer, file_name, mime_type: seen.append(buffer) or {"ok": True},
    )

    envelope(client.post("/api/dcc/send-image-data", json={
        "target": "#t", "bytes": base64.b64encode(PNG_BYTES).decode("ascii"), "fileName": "a.png",
    }), ok=True)
    envelope(client.post("/api/dcc/send-image-data", json={
        "target": "#t", "bytes": list(PNG_BYTES), "fileName": "b.png",
    }), ok=True)

    assert seen == [PNG_BYTES, PNG_BYTES], "the two encodings did not arrive identically"


def test_h_an_image_with_no_bytes_sends_an_empty_buffer_not_a_crash(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    seen: list[bytes] = []
    monkeypatch.setattr(
        state.core, "send_dcc_buffer",
        lambda target, buffer, file_name, mime_type: seen.append(buffer) or {"ok": True},
    )
    envelope(client.post("/api/dcc/send-image-data", json={"target": "#t"}), ok=True)
    assert seen == [b""]


@pytest.mark.parametrize("payload", ["!!!not base64!!!", "aaaaa", "not base64 at all"])
def test_h_an_image_with_undecodable_bytes_is_a_400_envelope_not_a_500(browser_client, payload):
    """The route does not validate the field; the app's catch-all is what saves it.

    That is still a 4xx envelope rather than a 500, which is the contract the browser
    depends on — but the error text is `binascii`'s ("Incorrect padding", "Invalid
    base64-encoded string"), so the operator is never told which field was wrong.
    """
    response = browser_client.post("/api/dcc/send-image-data", json={"target": "#t", "bytes": payload})
    refusal(response, status=400)


def test_h_an_image_falls_back_to_the_configured_channel(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    seen: list[str] = []
    monkeypatch.setattr(
        state.core, "send_dcc_buffer",
        lambda target, buffer, file_name, mime_type: seen.append(target) or {"ok": True},
    )
    envelope(client.post("/api/dcc/send-image-data", json={"bytes": ""}), ok=True)
    assert seen == [state.core.config["channel"]]


def test_h_an_image_with_no_name_is_given_a_generated_one(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    seen: list[str] = []
    monkeypatch.setattr(
        state.core, "send_dcc_buffer",
        lambda target, buffer, file_name, mime_type: seen.append(file_name) or {"ok": True},
    )
    envelope(client.post("/api/dcc/send-image-data", json={"target": "#t", "bytes": ""}), ok=True)
    assert seen[0].startswith("pasted-") and seen[0].endswith(".png"), seen


# --- /api/execute-command --------------------------------------------------------------------


def test_h_an_empty_slash_command_is_refused(client):
    body = envelope(client.post("/api/execute-command", json={"command": ""}), ok=False)
    assert body["focusTarget"] == ""


def test_h_a_slash_command_reaches_core_with_the_default_target(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    seen: list[tuple] = []
    monkeypatch.setattr(
        state.core, "execute_slash_command",
        lambda command, target: seen.append((command, target)) or {"ok": True, "focusTarget": target},
    )
    body = envelope(client.post("/api/execute-command", json={"command": "/join #room"}), ok=True)
    assert seen == [("/join #room", state.core.config["channel"])]
    assert body["focusTarget"] == state.core.config["channel"]


def test_h_a_slash_command_honours_an_explicit_target(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    seen: list[tuple] = []
    monkeypatch.setattr(
        state.core, "execute_slash_command",
        lambda command, target: seen.append((command, target)) or {"ok": True, "focusTarget": "#focus"},
    )
    body = envelope(client.post("/api/execute-command", json={"command": "/me waves", "target": "#here"}), ok=True)
    assert seen == [("/me waves", "#here")]
    assert body["focusTarget"] == "#focus"


def test_h_an_unknown_slash_command_is_refused_without_raising(client):
    response = client.post("/api/execute-command", json={"command": "/definitely-not-a-command"})
    body = envelope(response)
    assert set(body) == {"ok", "focusTarget"}, body


def test_h_execute_command_always_answers_the_same_two_fields(client):
    """The UI reads both unconditionally; a missing `focusTarget` is a client crash."""
    for command in ["", "/join", "not-a-slash-command", "/", "   "]:
        body = envelope(client.post("/api/execute-command", json={"command": command}))
        assert set(body) == {"ok", "focusTarget"}, (command, body)
        assert isinstance(body["focusTarget"], str)


# --- /api/llm/refresh-history ------------------------------------------------------------------


def test_h_refresh_history_reports_a_missing_pipeline_rather_than_claiming_success(client):
    """The fixture client root has no history pipeline, and that has to surface."""
    body = envelope(settle_then_refresh(client))
    assert body["ok"] is False
    assert body["status"] == "runtime_missing", body
    assert "run.py" in body["error"], body


def test_h_refresh_history_is_serialised_not_stacked(client, monkeypatch):
    """A second refresh while one is running must be told so, not start another scan.

    The busy flag is held by a refresh that is genuinely in flight rather than set by
    hand: the startup refresh clears the flag in its own `finally`, so a hand-set one
    is a race with whatever the app is still doing.
    """
    import threading

    from subjective_devagents_fastapi.app import state

    entered = threading.Event()
    release = threading.Event()

    def blocking_refresh():
        entered.set()
        assert release.wait(30), "the test never released the blocked refresh"
        return {"success": False, "status": "runtime_missing", "error": "fixture has no pipeline"}

    monkeypatch.setattr(state.core.llm_service, "refresh_history", blocking_refresh)

    first = threading.Thread(target=lambda: client.post("/api/llm/refresh-history"), daemon=True)
    first.start()
    try:
        assert entered.wait(10), "the first refresh never started"
        assert envelope(client.post("/api/llm/refresh-history"), ok=True) == {
            "ok": True, "deferred": True, "status": "busy",
        }
    finally:
        release.set()
        first.join(30)

    # And once it finishes the gate reopens, rather than latching busy forever.
    assert state.history_refreshing is False
    assert envelope(client.post("/api/llm/refresh-history"))["status"] == "runtime_missing"


# --- /api/llm/reload-conversation --------------------------------------------------------------


def test_h_reloading_an_unknown_conversation_is_refused(client):
    error = refusal(client.post("/api/llm/reload-conversation", json={"target": "llm:claude:local:nope"}))
    assert "not found" in error.lower(), error


def test_h_reloading_with_no_target_is_refused(client):
    refusal(client.post("/api/llm/reload-conversation", json={}))
    refusal(client.post("/api/llm/reload-conversation", json={"target": ""}))


def test_h_reloading_a_real_conversation_returns_it(client, chats):
    conversation = chats(provider="claude", name="h-reload")
    body = envelope(
        client.post("/api/llm/reload-conversation", json={"target": conversation["target"]}), ok=True
    )
    assert body["conversation"]["target"] == conversation["target"]
    assert body["conversation"]["provider"] == "claude"
    assert "messages" in body["conversation"]


def test_h_reloading_broadcasts_the_conversation_to_every_tab(client, chats):
    """The route's whole purpose beyond returning it: other tabs see the reload."""
    conversation = chats(provider="claude", name="h-reload-broadcast")
    with client.websocket_connect("/ws") as socket:
        for expected in OPENING_FRAMES:
            assert socket.receive_json()["type"] == expected
        envelope(
            client.post("/api/llm/reload-conversation", json={"target": conversation["target"]}), ok=True
        )
        event = drain_until(socket, "llm-conversation")
        assert event["conversation"]["target"] == conversation["target"]


def test_h_a_reload_that_raises_is_reported_as_an_error_envelope(client, chats, monkeypatch):
    from subjective_devagents_fastapi.app import state

    def explode(target):
        raise RuntimeError("the history store is unreadable")

    monkeypatch.setattr(state.core.llm_service, "reload_conversation", explode)
    error = refusal(client.post("/api/llm/reload-conversation", json={"target": "llm:claude:local:x"}))
    assert "unreadable" in error, error


# --- /api/switchuser/* ---------------------------------------------------------------------------


def test_h_continue_as_an_invalid_username_is_refused(client, chats):
    conversation = chats(provider="claude", name="h-switch")
    for username in ["", "root; rm -rf /", "../etc", "a" * 200]:
        error = refusal(client.post("/api/switchuser/continue-as", json={
            "target": conversation["target"], "username": username,
        }))
        assert "username" in error.lower() or "not found" in error.lower(), (username, error)


def test_h_continue_as_on_an_unknown_conversation_is_refused(client):
    error = refusal(client.post("/api/switchuser/continue-as", json={
        "target": "llm:claude:local:nope", "username": "someone",
    }))
    assert "not found" in error.lower(), error


def test_h_continue_as_switches_owner_when_the_session_is_already_ready(client, chats, monkeypatch):
    """The synchronous branch — no `su` is spawned, which is what makes this testable."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="h-switch-ready")
    monkeypatch.setattr(state.core.su_session_manager, "is_provider_ready", lambda username, provider: True)
    switched = []
    monkeypatch.setattr(
        state.core, "switch_conversation_owner",
        lambda target, username, provider: switched.append((target, username, provider)) or {"switched": True},
    )

    body = envelope(client.post("/api/switchuser/continue-as", json={
        "target": conversation["target"], "username": "someone", "provider": "claude",
    }), ok=True)
    assert body["switched"] is True
    assert switched == [(conversation["target"], "someone", "claude")]


def test_h_continue_as_defaults_to_the_conversations_own_provider(client, chats, monkeypatch):
    """The auth-retry picker calls this without a provider and must not switch the agent."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="codex", name="h-switch-default")
    monkeypatch.setattr(state.core.su_session_manager, "is_provider_ready", lambda username, provider: True)
    switched = []
    monkeypatch.setattr(
        state.core, "switch_conversation_owner",
        lambda target, username, provider: switched.append(provider) or {"ok": True},
    )

    envelope(client.post("/api/switchuser/continue-as", json={
        "target": conversation["target"], "username": "someone",
    }), ok=True)
    assert switched == ["codex"], switched


def test_h_disconnecting_a_user_with_no_session_is_harmless(client):
    assert envelope(client.post("/api/switchuser/disconnect", json={"username": "nobody"}), ok=True) == {"ok": True}
    assert envelope(client.post("/api/switchuser/disconnect", json={}), ok=True) == {"ok": True}


def test_h_disconnect_reaches_the_session_manager(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    seen = []
    monkeypatch.setattr(state.core.su_session_manager, "disconnect", lambda username: seen.append(username))
    envelope(client.post("/api/switchuser/disconnect", json={"username": "someone"}), ok=True)
    assert seen == ["someone"]


def test_h_pty_input_for_an_unknown_user_is_a_no_op(client):
    assert envelope(client.post("/api/switchuser/pty-input", json={
        "username": "nobody", "data": "ls\n",
    }), ok=True) == {"ok": True}


def test_h_pty_input_is_passed_through_verbatim(client, monkeypatch):
    """Keystrokes are a byte stream: control characters must not be mangled or stripped."""
    from subjective_devagents_fastapi.app import state

    seen: list[tuple] = []
    monkeypatch.setattr(
        state.core.su_session_manager, "write_input",
        lambda username, data: seen.append((username, data)),
    )
    for payload in ["ls -la\n", "\x03", "\x1b[A", "a\tb", "", "é"]:
        envelope(client.post("/api/switchuser/pty-input", json={
            "username": "someone", "data": payload,
        }), ok=True)
    assert [data for _, data in seen] == ["ls -la\n", "\x03", "\x1b[A", "a\tb", "", "é"]


# --- /api/switchuser/pty-resize ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param({"cols": 120, "rows": 40}, (120, 40), id="numbers"),
        pytest.param({"cols": "120", "rows": "40"}, (120, 40), id="strings-from-a-form"),
        # Any single unparseable field zeroes both, so resize() ignores the call rather
        # than acting on half a size.
        pytest.param({"cols": "wide", "rows": 40}, (0, 0), id="unparseable-cols"),
        pytest.param({"cols": 120, "rows": None}, (120, 0), id="null-rows"),
        pytest.param({}, (0, 0), id="nothing"),
        pytest.param({"cols": 0, "rows": 0}, (0, 0), id="zero"),
        pytest.param({"cols": -5, "rows": -5}, (-5, -5), id="negative"),
        pytest.param({"cols": 1.9, "rows": 2.9}, (1, 2), id="floats-truncate"),
    ],
)
def test_h_pty_resize_coerces_what_the_browser_sends(client, monkeypatch, body, expected):
    from subjective_devagents_fastapi.app import state

    seen: list[tuple] = []
    monkeypatch.setattr(
        state.core.su_session_manager, "resize",
        lambda username, cols, rows: seen.append((cols, rows)),
    )
    assert envelope(client.post("/api/switchuser/pty-resize", json={
        "username": "someone", **body,
    }), ok=True) == {"ok": True}
    assert seen == [expected], f"{body} coerced to {seen}, expected {expected}"


def test_h_pty_resize_ignores_a_nonpositive_size(client):
    """`resize` itself refuses a size it cannot set, so the route may pass it through."""
    for body in [{"cols": 0, "rows": 0}, {"cols": -1, "rows": 40}, {}]:
        assert envelope(client.post("/api/switchuser/pty-resize", json={
            "username": "nobody", **body,
        }), ok=True) == {"ok": True}


def test_h_pty_resize_for_an_unknown_user_is_a_no_op(client):
    assert envelope(client.post("/api/switchuser/pty-resize", json={
        "username": "nobody", "cols": 80, "rows": 24,
    }), ok=True) == {"ok": True}


# --- helpers ---------------------------------------------------------------------------------------


OPENING_FRAMES = ("config", "status")


def settle_then_refresh(client, timeout: float = 10.0):
    """Ask for a refresh once the startup one has finished.

    Lifespan fires a history refresh of its own, and it holds the same busy flag the
    manual route checks. Without waiting for it, this test intermittently gets
    `deferred` instead of the pipeline's real answer — a race with the app's own
    startup, not with anything the test is trying to assert.
    """
    import time

    from subjective_devagents_fastapi.app import state

    deadline = time.monotonic() + timeout
    while state.history_refreshing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not state.history_refreshing, "the startup history refresh never finished"
    return client.post("/api/llm/refresh-history")


def drain_until(socket, event_type: str, limit: int = 12) -> dict:
    """The next event of this type, past the opening frames and the idle heartbeat.

    `/ws` sends `config` then `status` on connect and a `ping` every 25 seconds while
    idle, so a bounded read that skips those is what "the event arrived" means here —
    and the heartbeat is what stops a missing event blocking forever.
    """
    for _ in range(limit):
        event = socket.receive_json()
        if event.get("type") == event_type:
            return event
    raise AssertionError(f"no {event_type} event arrived in {limit} frames")


def open_events_socket(client):
    """A connected `/ws` with the two opening frames already consumed."""
    socket = client.websocket_connect("/ws").__enter__()
    for expected in OPENING_FRAMES:
        frame = socket.receive_json()
        assert frame["type"] == expected, f"expected {expected} first, got {frame['type']}"
    return socket
