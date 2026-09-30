"""F. Sending and receiving messages.

`POST /api/llm/send-message` is the route production defects live on, because it is
the one that must answer *before* the work it started has finished: Cloudflare closes
an idle HTTP request at ~100s with a 524, while an agent turn is allowed 30 minutes.
So the contract is "accept, then deliver on the socket", and F2–F4 are the cases that
pin the difference between "still working" and "failed".

`POST /api/send-message` — the non-LLM IRC path — had no test at all before F5.
"""
from __future__ import annotations

import base64
import time

import pytest

from .helpers import envelope, refusal

OPENING_FRAMES = ("config", "status")


def open_socket(client):
    socket = client.websocket_connect("/ws").__enter__()
    for expected in OPENING_FRAMES:
        frame = socket.receive_json()
        assert frame["type"] == expected, f"expected {expected}, got {frame['type']}"
    return socket


def drain_until(socket, *types: str, limit: int = 12) -> dict:
    for _ in range(limit):
        event = socket.receive_json()
        if event.get("type") in types:
            return event
    raise AssertionError(f"none of {types} arrived in {limit} frames")


def wait_for(condition, timeout: float = 5.0, what: str = "condition") -> None:
    """A deterministic wait, not a sleep on a guess."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"{what} was not met within {timeout}s")


@pytest.fixture()
def spy_send(monkeypatch):
    """Record what reaches `send_llm_message`, and decide what it answers."""
    from subjective_devagents_fastapi.app import state

    sent: list[dict] = []
    outcome = {"result": {"ok": True}}

    def fake(target, text, files):
        sent.append({"target": target, "text": text, "files": files})
        result = outcome["result"]
        return result() if callable(result) else result

    monkeypatch.setattr(state.core, "send_llm_message", fake)
    return sent, outcome


# --- F1: a message to a live chat -------------------------------------------------------------


def test_f1_a_send_is_accepted_and_the_reply_arrives_on_the_socket(client, chats, monkeypatch):
    """Accepted here, delivered there: the two halves of the contract in one test."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f1")

    def fake(target, text, files):
        state.core.broadcast("llm-message", {
            "target": target,
            "message": {"role": "assistant", "text": f"Answering: {text}", "time": time.time() * 1000},
            "provider": "claude",
        })
        return {"ok": True}

    monkeypatch.setattr(state.core, "send_llm_message", fake)

    with client.websocket_connect("/ws") as socket:
        for expected in OPENING_FRAMES:
            socket.receive_json()

        accepted = envelope(client.post("/api/llm/send-message", json={
            "target": conversation["target"], "text": "Hello there",
        }), ok=True)
        assert accepted["pending"] is True
        assert accepted["target"] == conversation["target"]

        event = drain_until(socket, "llm-message")
        assert event["target"] == conversation["target"]
        assert event["message"]["text"] == "Answering: Hello there"
        assert event["message"]["role"] == "assistant"


def test_f1_the_prompt_reaches_the_backend_unchanged(client, chats, spy_send):
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f1-verbatim")

    envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": "  Refactor the parser  ",
    }), ok=True)

    wait_for(lambda: sent, what="the send reaching the backend")
    # Leading and trailing space is trimmed at the route; the text itself is untouched.
    assert sent[0]["text"] == "Refactor the parser"
    assert sent[0]["target"] == conversation["target"]


def test_f1_attachments_are_passed_through(client, chats, spy_send):
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f1-files")
    files = [{"name": "diff.patch", "mimeType": "text/plain", "bytes": "aGk="}]

    envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": "Apply this", "files": files,
    }), ok=True)

    wait_for(lambda: sent, what="the send reaching the backend")
    assert sent[0]["files"] == files


def _dashboard_file(filename: str, mime_type: str, raw: bytes) -> dict:
    """The dashboard body: filename, mime_type, and standard base64 content."""
    return {
        "filename": filename,
        "mime_type": mime_type,
        "content": base64.b64encode(raw).decode("ascii"),
    }


def test_f1_dashboard_image_and_pdf_reach_send_llm_message(client, chats, spy_send):
    """POST /api/llm/send-message must hand the dashboard files list to the pipeline
    as the third argument, image and non-image together, unchanged."""
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f1-dashboard-files")
    files = [
        _dashboard_file("pixel.png", "image/png", b"\x89PNG\r\n\x1a\nfake-image"),
        _dashboard_file("spec.pdf", "application/pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"),
    ]

    body = envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": "Look at both", "files": files,
    }), ok=True)

    assert body["pending"] is True
    assert body["ok"] is True
    wait_for(lambda: sent, what="the send reaching the backend")
    assert sent[0]["files"] == files
    assert [item["mime_type"] for item in sent[0]["files"]] == ["image/png", "application/pdf"]
    assert all(isinstance(item, dict) and "content" in item for item in sent[0]["files"])


@pytest.mark.parametrize("text", ["", " ", "   ", "\n", "\t\n", "\t\n  \r\n"])
def test_f7_whitespace_text_with_files_is_rejected_before_the_backend(client, chats, spy_send, text):
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f7-files-empty")
    files = [_dashboard_file("pixel.png", "image/png", b"\x89PNG\r\n\x1a\nfake-image")]

    body = envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": text, "files": files,
    }), ok=False)

    assert body["status"] == "invalid_request"
    assert sent == [], f"{text!r} with files reached the agent"


def test_f6_an_unknown_target_with_files_is_not_found(client, spy_send):
    sent, _ = spy_send
    files = [
        _dashboard_file("pixel.png", "image/png", b"\x89PNG\r\n\x1a\nfake-image"),
        _dashboard_file("spec.pdf", "application/pdf", b"%PDF-1.4"),
    ]

    body = envelope(client.post("/api/llm/send-message", json={
        "target": "llm:claude:local:not-a-real-chat",
        "text": "Look at both",
        "files": files,
    }), ok=False)

    assert body["status"] == "not_found"
    assert sent == [], "files for a missing chat reached the backend"


# --- F2/F3/F4: pending is not failed --------------------------------------------------------------


def test_f2_a_long_turn_returns_pending_immediately(client, chats, monkeypatch):
    """The POST must not stay open for the turn, or Cloudflare 524s it at ~100s."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f2-long")
    started = {"at": None}
    release = {"go": False}

    def slow(target, text, files):
        started["at"] = time.monotonic()
        while not release["go"]:
            time.sleep(0.01)
        return {"ok": True}

    monkeypatch.setattr(state.core, "send_llm_message", slow)
    try:
        before = time.monotonic()
        body = envelope(client.post("/api/llm/send-message", json={
            "target": conversation["target"], "text": "A very long turn",
        }), ok=True)
        elapsed = time.monotonic() - before

        assert body == {"ok": True, "pending": True, "target": conversation["target"]}
        assert elapsed < 2.0, f"the POST waited {elapsed:.1f}s for the turn"
        wait_for(lambda: started["at"] is not None, what="the turn starting in the background")
    finally:
        release["go"] = True


def test_f3_a_proxy_timeout_on_a_long_call_is_reported_as_pending_not_failed(client, chats, spy_send):
    """The route answers before the turn can fail, so a 524 downstream cannot make the
    send look failed: the caller was already told `pending`, and the real outcome comes
    over the socket."""
    sent, outcome = spy_send

    def proxy_timeout(*_args, **_kwargs):
        raise RuntimeError("504 Gateway Time-out from the proxy")

    outcome["result"] = proxy_timeout
    conversation = chats(provider="claude", name="f3-proxy")

    body = envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": "Something slow",
    }), ok=True)
    assert body["pending"] is True, "a downstream timeout was reported as a failed send"

    wait_for(lambda: sent, what="the background turn running")
    # The background crash is logged and does not retroactively fail the accepted POST.
    assert envelope(client.get("/api/llm/conversation", params={
        "target": conversation["target"],
    }), ok=True)["conversation"]["busy"] is False


def test_f4_a_failure_on_a_short_read_is_an_error_not_an_empty_success(client, monkeypatch):
    """`GET /api/llm/conversations` is a short read: a crash must not look like "no chats"."""
    from subjective_devagents_fastapi.app import state

    def explode(needle, limit):
        raise RuntimeError("the conversation store is unreadable")

    monkeypatch.setattr(state.core.llm_service, "list_conversations", explode)

    body = envelope(client.get("/api/llm/conversations"), ok=False)
    assert "unreadable" in body["error"]
    assert body["conversations"] == []
    assert "total" not in body or body.get("total") == 0


def test_f4_a_failed_listing_is_distinguishable_from_an_empty_one(client):
    """The empty case, for contrast: `ok` true and an honest zero."""
    body = envelope(client.get("/api/llm/conversations"), ok=True)
    assert body["conversations"] == []
    assert body["total"] == 0


def test_f4_a_failed_hydrate_reports_the_reason(client, chats, monkeypatch):
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f4-hydrate")

    def explode(target):
        raise RuntimeError("the transcript is corrupt")

    monkeypatch.setattr(state.core.llm_service, "reload_conversation", explode)
    error = refusal(client.get("/api/llm/conversation", params={"target": conversation["target"]}))
    assert "corrupt" in error, error


# --- F5: the non-LLM send path --------------------------------------------------------------------


def test_f5_the_irc_send_path_refuses_when_not_connected(client):
    """`POST /api/send-message` had no test at all. Nothing is connected here."""
    body = envelope(client.post("/api/send-message", json={"target": "#room", "text": "hello"}), ok=False)
    assert body == {"ok": False}


def test_f5_the_irc_send_path_accepts_when_the_socket_takes_it(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    written: list[str] = []
    monkeypatch.setattr(state.core, "write_irc", lambda command: written.append(command) or True)
    broadcast: list[tuple] = []
    monkeypatch.setattr(state.core, "broadcast", lambda event, payload: broadcast.append((event, payload)))

    assert envelope(client.post("/api/send-message", json={
        "target": "#room", "text": "hello there",
    }), ok=True) == {"ok": True}

    assert written == ["PRIVMSG #room :hello there"]
    assert broadcast[0][0] == "message"
    assert broadcast[0][1]["text"] == "hello there"
    assert broadcast[0][1]["mine"] is True
    assert broadcast[0][1]["private"] is False


def test_f5_a_private_target_is_marked_private(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    monkeypatch.setattr(state.core, "write_irc", lambda command: True)
    broadcast: list[tuple] = []
    monkeypatch.setattr(state.core, "broadcast", lambda event, payload: broadcast.append((event, payload)))

    envelope(client.post("/api/send-message", json={"target": "someone", "text": "hi"}), ok=True)
    assert broadcast[0][1]["private"] is True


def test_f5_the_irc_path_falls_back_to_the_configured_channel(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    written: list[str] = []
    monkeypatch.setattr(state.core, "write_irc", lambda command: written.append(command) or True)
    monkeypatch.setattr(state.core, "broadcast", lambda event, payload: None)

    envelope(client.post("/api/send-message", json={"text": "no target"}), ok=True)
    assert written == [f"PRIVMSG {state.core.config['channel']} :no target"]


# --- F6/F7: refusals ------------------------------------------------------------------------------------


def test_f6_sending_to_an_unknown_target_creates_nothing(client, spy_send):
    sent, _ = spy_send
    body = envelope(client.post("/api/llm/send-message", json={
        "target": "llm:claude:local:not-a-real-chat", "text": "hello",
    }), ok=False)

    assert body["status"] == "not_found"
    assert sent == [], "a message for a chat that does not exist reached the backend"
    assert envelope(client.get("/api/llm/conversations"), ok=True)["total"] == 0


def test_f6_sending_with_no_target_is_refused(client, spy_send):
    sent, _ = spy_send
    body = envelope(client.post("/api/llm/send-message", json={"text": "hello"}), ok=False)
    assert body["status"] == "invalid_request"
    assert sent == []


@pytest.mark.parametrize("text", ["", "   ", "\n", "\t\n  \r\n", None])
def test_f7_an_empty_message_is_rejected_before_the_backend(client, chats, spy_send, text):
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f7-empty")

    body = envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": text,
    }), ok=False)
    assert body["status"] == "invalid_request"
    assert "Missing target or text" in body["error"]
    assert sent == [], f"{text!r} reached the agent"


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_f7_the_irc_path_also_rejects_an_empty_message(client, monkeypatch, text):
    from subjective_devagents_fastapi.app import state

    written: list[str] = []
    monkeypatch.setattr(state.core, "write_irc", lambda command: written.append(command) or True)

    assert envelope(client.post("/api/send-message", json={
        "target": "#room", "text": text,
    }), ok=False) == {"ok": False}
    assert written == []


# --- F8/F9: cancelling, and authoritative busy -------------------------------------------------------------


def test_f8_cancelling_an_idle_chat_is_a_no_op_not_an_error(client, chats):
    """Idempotent by design: `running` says whether a turn was actually in flight."""
    conversation = chats(provider="claude", name="f8-idle")
    body = envelope(client.post("/api/llm/cancel-message", json={
        "target": conversation["target"],
    }), ok=True)
    assert body["running"] is False


def test_f8_cancelling_mid_turn_stops_it_and_clears_busy(client, chats, monkeypatch):
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f8-cancel")
    target = conversation["target"]
    service = state.core.llm_service

    stopped = {"count": 0}

    def canceller() -> bool:
        stopped["count"] += 1
        service.busy.discard(target)
        return True

    service.busy.add(target)
    service._cancellers[target] = canceller

    assert envelope(client.get("/api/llm/conversation", params={
        "target": target,
    }), ok=True)["conversation"]["busy"] is True

    body = envelope(client.post("/api/llm/cancel-message", json={"target": target}), ok=True)
    assert body["running"] is True
    assert stopped["count"] == 1

    assert envelope(client.get("/api/llm/conversation", params={
        "target": target,
    }), ok=True)["conversation"]["busy"] is False


def test_f8_cancelling_broadcasts_the_busy_release(client, chats, monkeypatch):
    """A tab that did not press Stop still has to re-enable its composer."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f8-broadcast")
    target = conversation["target"]
    service = state.core.llm_service
    service.busy.add(target)
    service._cancellers[target] = lambda: service.busy.discard(target) or True

    with client.websocket_connect("/ws") as socket:
        for expected in OPENING_FRAMES:
            socket.receive_json()
        envelope(client.post("/api/llm/cancel-message", json={"target": target}), ok=True)

        event = drain_until(socket, "llm-busy")
        assert event["target"] == target
        assert event["busy"] is False


def test_f8_cancelling_with_no_target_is_refused(client):
    error = refusal(client.post("/api/llm/cancel-message", json={}))
    assert "target" in error.lower(), error


def test_f9_busy_is_authoritative_in_both_the_hydrate_and_the_listing(client, chats):
    """The recovery path from a dropped socket: whatever the tab missed, these agree."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f9-busy")
    target = conversation["target"]
    service = state.core.llm_service

    service.busy.add(target)
    try:
        hydrated = envelope(client.get("/api/llm/conversation", params={"target": target}), ok=True)
        listed = envelope(client.get("/api/llm/conversations"), ok=True)["conversations"]
        activated = envelope(client.post("/api/llm/activate-conversation", json={"target": target}), ok=True)

        assert hydrated["conversation"]["busy"] is True
        assert next(item for item in listed if item["target"] == target)["busy"] is True
        assert activated["busy"] is True, (
            "activation is what recovers a wedged chat on every open, and it did not report busy"
        )
    finally:
        service.busy.discard(target)

    assert envelope(client.get("/api/llm/conversation", params={
        "target": target,
    }), ok=True)["conversation"]["busy"] is False


def test_f9_busy_survives_a_dropped_socket(client, chats):
    """The socket carries the *event*; these endpoints carry the truth."""
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f9-dropped")
    target = conversation["target"]
    state.core.llm_service.busy.add(target)
    try:
        with client.websocket_connect("/ws") as socket:
            for expected in OPENING_FRAMES:
                socket.receive_json()
        # Socket closed — the tab missed everything. Reopening still learns the truth.
        assert envelope(client.post("/api/llm/activate-conversation", json={
            "target": target,
        }), ok=True)["busy"] is True
    finally:
        state.core.llm_service.busy.discard(target)


# --- F10/F11: what the composer sends ------------------------------------------------------------------------


def test_f10_a_multi_line_message_is_one_send(client, chats, spy_send):
    """Splitting it would deliver a half-written prompt as its own turn."""
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f10-multiline")
    text = "First line\nSecond line\n\nFourth line after a blank one"

    envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": text,
    }), ok=True)

    wait_for(lambda: sent, what="the send reaching the backend")
    assert len(sent) == 1, f"one message became {len(sent)} sends"
    assert sent[0]["text"] == text, "the newlines were not preserved"


def test_f10_indentation_inside_a_code_block_survives(client, chats, spy_send):
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f10-code")
    text = "Fix this:\n\n```python\ndef f():\n    return 1\n```\n"

    envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": text,
    }), ok=True)
    wait_for(lambda: sent, what="the send reaching the backend")
    assert sent[0]["text"] == text.strip()
    assert "    return 1" in sent[0]["text"]


def test_f11_a_slash_command_goes_down_the_command_path(client, monkeypatch):
    """`/join #room` is a command, not a prompt, and has its own endpoint."""
    from subjective_devagents_fastapi.app import state

    seen: list[tuple] = []
    monkeypatch.setattr(
        state.core, "execute_slash_command",
        lambda command, target: seen.append((command, target)) or {"ok": True, "focusTarget": "#room"},
    )
    envelope(client.post("/api/execute-command", json={"command": "/join #room", "target": "#lobby"}), ok=True)
    assert seen == [("/join #room", "#lobby")]


def test_f11_a_multi_line_llm_message_beginning_with_a_slash_is_still_a_message(
    client, chats, spy_send
):
    """The exception to the slash rule: a prompt may legitimately open with one."""
    sent, _ = spy_send
    conversation = chats(provider="claude", name="f11-slash")
    text = "/usr/local/bin is on PATH.\nWhy does the installer not find it?"

    envelope(client.post("/api/llm/send-message", json={
        "target": conversation["target"], "text": text,
    }), ok=True)

    wait_for(lambda: sent, what="the send reaching the backend")
    assert sent[0]["text"] == text, "a message starting with / was mangled into a command"


# --- F12/F13: the socket --------------------------------------------------------------------------------------


def test_f12_an_unauthenticated_socket_is_closed(bare_token_client):
    """No credential in the headers, and none in the first frame, so it closes."""
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as closed:
        with bare_token_client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "not-auth"})
            socket.receive_json()
    assert closed.value.code == 4401, closed.value.code


def test_f12_a_socket_with_a_wrong_token_in_the_first_frame_is_closed(bare_token_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as closed:
        with bare_token_client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "auth", "token": "not-the-token"})
            socket.receive_json()
    assert closed.value.code == 4401


def test_f12_a_socket_with_the_right_token_in_a_header_is_accepted(bare_token_client):
    from .conftest import TOKEN

    with bare_token_client.websocket_connect("/ws", headers={"X-DevAgents-Token": TOKEN}) as socket:
        assert socket.receive_json()["type"] == "config"


def test_f12_a_socket_may_authenticate_in_its_first_frame(bare_token_client):
    """Owner browsers do it this way: a URL is logged, and a token in one outlives it."""
    from .conftest import TOKEN

    with bare_token_client.websocket_connect("/ws") as socket:
        socket.send_json({"type": "auth", "token": TOKEN})
        assert socket.receive_json()["type"] == "config"


def test_f13_the_socket_delivers_each_event_in_the_shape_the_ui_expects(client, chats):
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="claude", name="f13")
    target = conversation["target"]

    with client.websocket_connect("/ws") as socket:
        for expected in OPENING_FRAMES:
            socket.receive_json()

        # llm-message
        state.core.broadcast("llm-message", {
            "target": target,
            "message": {"role": "assistant", "text": "Hello", "time": 1.0},
            "provider": "claude",
        })
        message = drain_until(socket, "llm-message")
        assert message["target"] == target
        assert message["message"]["role"] == "assistant"
        assert message["message"]["text"] == "Hello"
        assert message["provider"] == "claude"

        # llm-busy
        state.core.broadcast("llm-busy", {"target": target, "busy": True})
        busy = drain_until(socket, "llm-busy")
        assert busy["target"] == target and busy["busy"] is True

        # llm-conversation
        state.core.broadcast("llm-conversation", {"conversation": conversation})
        whole = drain_until(socket, "llm-conversation")
        assert whole["conversation"]["target"] == target
        assert whole["conversation"]["provider"] == "claude"


def test_f13_creating_a_chat_announces_it_on_the_socket(client, cleanup):
    with client.websocket_connect("/ws") as socket:
        for expected in OPENING_FRAMES:
            socket.receive_json()

        created = envelope(client.post("/api/llm/create-conversation", json={
            "provider": "claude", "name": "f13-announced",
        }), ok=True)["conversation"]
        cleanup(lambda: client.post("/api/llm/delete-conversation", json={"target": created["target"]}))

        event = drain_until(socket, "llm-conversation")
        assert event["conversation"]["target"] == created["target"]
        assert event["conversation"]["label"] == "f13-announced"


def test_f13_a_rename_and_a_delete_reach_every_tab(client):
    created = envelope(client.post("/api/llm/create-conversation", json={
        "provider": "claude", "name": "f13-before",
    }), ok=True)["conversation"]

    with client.websocket_connect("/ws") as socket:
        for expected in OPENING_FRAMES:
            socket.receive_json()

        envelope(client.post("/api/llm/rename-conversation", json={
            "target": created["target"], "name": "f13-after",
        }), ok=True)
        renamed = drain_until(socket, "llm-conversation-renamed")
        assert renamed["target"] == created["target"]
        assert renamed["label"] == "f13-after"

        envelope(client.post("/api/llm/delete-conversation", json={"target": created["target"]}), ok=True)
        deleted = drain_until(socket, "llm-conversation-deleted")
        assert deleted["target"] == created["target"]

    assert envelope(client.get("/api/llm/conversations"), ok=True)["total"] == 0


# --- F14: an SSH chat's reply -------------------------------------------------------------------------------------


def test_f14_an_ssh_reply_keeps_its_column_spacing(client, chats, monkeypatch):
    """A chat bubble has to show what the terminal showed: `ls -l` columns must line up."""
    from subjective_devagents_fastapi import ssh_chat

    conversation = chats(provider="ssh", name="f14", workspace="deploy@f14.example.invalid")
    listing = (
        "total 12\n"
        "drwxr-xr-x  2 deploy deploy 4096 Sep 21 10:00 bin\n"
        "-rw-r--r--  1 deploy deploy   42 Sep 21 10:01 README.md\n"
    )
    monkeypatch.setattr(
        ssh_chat.sessions, "send",
        lambda target, connection, text, timeout=None: {
            "success": True, "status": "completed", "response": listing, "error": "",
        },
    )

    rendered = ssh_chat.render_terminal_text(listing)
    assert "drwxr-xr-x  2 deploy deploy 4096" in rendered, "runs of spaces were collapsed"
    assert rendered.count("\n") == listing.count("\n") - 1 or listing.strip() in rendered


@pytest.mark.parametrize(
    "raw,expected_in",
    [
        pytest.param("\x1b[32mgreen\x1b[0m text\n", "green text", id="colour-stripped"),
        pytest.param("a    b\tc\n", "a    b\tc", id="spacing-kept"),
        pytest.param("progress 10%\rprogress 100%\n", "progress 100%", id="redrawn-line"),
        pytest.param("line one\r\nline two\r\n", "line one\nline two", id="crlf-normalised"),
    ],
)
def test_f14_ordinary_command_output_only_has_its_escapes_removed(raw, expected_in):
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    assert expected_in in render_terminal_text(raw)


def test_f14_a_full_screen_redraw_is_drawn_rather_than_concatenated(client):
    """Stripping escapes from a positioned screen gives `Doyoutrustthisdirectory?`."""
    from subjective_devagents_fastapi.ssh_chat import render_terminal_text

    drawn = (
        "\x1b[?1049h\x1b[2J"
        "\x1b[3;5HDo you trust the contents"
        "\x1b[4;5Hof this directory?"
        "\x1b[6;5H1. Yes"
        "\x1b[7;5H2. No"
    )
    rendered = render_terminal_text(drawn, cols=80, rows=24)

    assert "Do you trust the contents" in rendered
    assert "of this directory?" in rendered
    assert "1. Yes" in rendered and "2. No" in rendered
    # The lines landed on separate rows, not concatenated into one.
    assert "contentsof" not in rendered.replace(" ", "")[:200] or "\n" in rendered
    assert rendered.count("\n") >= 3, rendered
