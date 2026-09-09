"""Tests for the quick-reply path — clicking Yes/No on an agent's question.

The button does exactly one thing: POST /api/llm/send-message with the canned reply text
("Yes, proceed."). These exercise that endpoint at the API surface — the same request the
browser sends — and prove the reply is accepted and handed to the backend, without spawning
a real agent turn (send_llm_message is stubbed so no CLI runs).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

TOKEN = "quick-reply-token"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True)
    (tmp_path / "client" / "src").mkdir(parents=True)
    (tmp_path / "client" / "web" / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", raising=False)

    from subjective_devagents_fastapi.app import app, state

    with TestClient(app) as test_client:
        test_client.app_state = state
        yield test_client


def capture_send(state):
    """Replace ChatCore.send_llm_message with a stub that records the call and signals it,
    so the background task never touches a real CLI. Returns (record, done_event)."""
    record: dict = {}
    done = threading.Event()

    def stub(target, text, files=None):
        record["target"] = target
        record["text"] = text
        record["files"] = files
        done.set()
        return {"ok": True, "status": "completed", "assistantMessage": {"role": "assistant", "text": "ok"}}

    state.core.send_llm_message = stub  # type: ignore[method-assign]
    return record, done


# --- the click's request --------------------------------------------------------------------


# Every writable provider a Yes button can appear under. The button behaves identically
# for each, so the acceptance-and-dispatch contract is checked for all of them.
WRITABLE_PROVIDERS = ["codex", "claude", "cursor", "grok"]


@pytest.mark.parametrize("provider", WRITABLE_PROVIDERS)
def test_clicking_yes_sends_the_reply_to_the_backend(client, provider):
    """The whole feature, for each CLI: the Yes button POSTs the canned reply, the endpoint
    accepts it, and the reply text reaches send_llm_message for that provider — not dropped,
    not altered, not misrouted. The send is stubbed, so no CLI actually runs here; a real
    per-CLI turn is the opt-in live test below."""
    created = client.post("/api/llm/create-conversation", json={"provider": provider, "name": f"Quick reply {provider}"}).json()
    conversation = created["conversation"]
    target = conversation["target"]
    assert conversation["provider"] == provider
    record, done = capture_send(client.app_state)

    response = client.post("/api/llm/send-message", json={"target": target, "text": "Yes, proceed."})

    # The endpoint accepts immediately and delivers the answer on the WebSocket later.
    assert response.status_code == 200
    assert response.json() == {"ok": True, "pending": True, "target": target}

    # The background task must actually run the send with the exact reply, for this provider.
    assert done.wait(timeout=5), f"send_llm_message was never called for a {provider} quick reply"
    assert record["target"] == target
    assert record["text"] == "Yes, proceed."


def test_a_no_reply_is_delivered_verbatim(client):
    created = client.post("/api/llm/create-conversation", json={"provider": "codex", "name": "No reply"}).json()
    target = created["conversation"]["target"]
    record, done = capture_send(client.app_state)

    client.post("/api/llm/send-message", json={"target": target, "text": "No, do not proceed."})

    assert done.wait(timeout=5)
    assert record["text"] == "No, do not proceed."


def test_an_empty_reply_is_rejected_without_reaching_the_backend(client):
    """The "Something else…" button never posts, but a defensive empty text must be refused
    at the API rather than dispatched."""
    created = client.post("/api/llm/create-conversation", json={"provider": "claude", "name": "Empty"}).json()
    target = created["conversation"]["target"]
    record, done = capture_send(client.app_state)

    body = client.post("/api/llm/send-message", json={"target": target, "text": "   "}).json()

    assert body["ok"] is False
    assert body["status"] == "invalid_request"
    assert not done.wait(timeout=1), "an empty reply must not reach send_llm_message"


def test_a_reply_to_an_unknown_conversation_is_refused(client):
    record, done = capture_send(client.app_state)
    body = client.post("/api/llm/send-message", json={"target": "llm:claude:local:gone", "text": "Yes, proceed."}).json()
    assert body == {"ok": False, "status": "not_found", "error": "Conversation was not found."}
    assert not done.wait(timeout=1)


def test_the_send_endpoint_is_gated_when_a_token_is_set(tmp_path, monkeypatch):
    """The quick reply carries the token like any other call; without it the API refuses,
    which is what the bridge reports as 'unauthorized' to the reader."""
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True)
    (tmp_path / "client" / "src").mkdir(parents=True)
    (tmp_path / "client" / "web" / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", TOKEN)

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as guarded:
        anon = guarded.post("/api/llm/send-message", json={"target": "x", "text": "Yes, proceed."})
        assert anon.status_code == 401
        # With the token it gets past the gate (then fails on the unknown target, not on auth).
        ok = guarded.post(
            "/api/llm/send-message",
            json={"target": "llm:claude:local:x", "text": "Yes, proceed."},
            headers={"X-DevAgents-Token": TOKEN},
        )
        assert ok.status_code == 200
        assert ok.json()["status"] == "not_found"
