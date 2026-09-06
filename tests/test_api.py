"""API-surface tests: every route the browser UI calls answers in the shape it expects.

Runs against a throwaway SUBJECTIVE_CHAT_HOME so no real conversation state is touched,
and with IRC disabled (the default) so nothing dials out.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    chat_home = tmp_path / "chat-home"
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(chat_home))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True)
    (tmp_path / "client" / "src").mkdir(parents=True)
    (tmp_path / "client" / "web" / "index.html").write_text(
        "<!doctype html>\n<html lang=\"en\"><head><title>DevAgents</title></head><body>DEVAGENTS_DEPLOY_VERSION</body></html>",
        encoding="utf-8",
    )

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as test_client:
        test_client.chat_home = chat_home
        yield test_client


def test_version(client):
    body = client.get("/api/version").json()
    assert body["ok"] is True
    assert body["release"]
    assert body["version"].startswith(body["release"])


def test_config_round_trip(client):
    config = client.get("/api/config").json()
    assert {"host", "port", "channel", "nick", "bookmarks"} <= set(config)

    saved = client.post(
        "/api/save-config",
        json={"active": {"host": "10.0.0.9", "port": 6697, "channel": "#ported", "nick": "tester", "tls": True}},
    ).json()
    assert saved["ok"] is True
    assert saved["config"]["host"] == "10.0.0.9"
    assert saved["config"]["port"] == 6697
    assert saved["config"]["channel"] == "#ported"
    assert saved["config"]["tls"] is True
    assert client.get("/api/config").json()["nick"] == "tester"

    # Settings persist to disk in the same file layout the Node backend used, so an
    # existing ~/.subjective_chat/settings.json keeps working across the migration.
    settings = json.loads((Path(client.chat_home) / "settings.json").read_text(encoding="utf-8"))
    assert settings["active"]["host"] == "10.0.0.9"
    assert settings["active"]["channel"] == "#ported"
    assert any(bookmark["host"] == "10.0.0.9" for bookmark in settings["bookmarks"])


def test_config_save_does_not_block_on_a_dead_irc_host(client):
    # 10.255.255.1 is a black hole: the TCP handshake hangs until it times out. The
    # handshake belongs on its own thread, so the HTTP response must not wait for it.
    started = time.monotonic()
    response = client.post(
        "/api/save-config",
        json={"active": {"host": "10.255.255.1", "port": 6667, "channel": "#slow", "nick": "tester"}},
    )
    assert response.status_code == 200
    assert time.monotonic() - started < 5


def test_conversation_lifecycle(client):
    created = client.post("/api/llm/create-conversation", json={"provider": "codex", "name": "Port check"}).json()
    assert created["ok"] is True
    conversation = created["conversation"]
    target = conversation["target"]
    assert conversation["provider"] == "codex"
    assert conversation["label"] == "Port check"
    assert conversation["writable"] is True
    assert conversation["ircChannel"].startswith("#codex-")

    listed = client.get("/api/llm/conversations", params={"limit": 50}).json()
    assert listed["ok"] is True
    assert listed["total"] >= 1
    assert any(item["target"] == target for item in listed["conversations"])

    renamed = client.post("/api/llm/rename-conversation", json={"target": target, "name": "Renamed chat"}).json()
    assert renamed["ok"] is True
    assert renamed["conversation"]["label"] == "Renamed chat"
    # The override survives a fresh listing, the whole point of labelOverrides.
    relisted = client.get("/api/llm/conversations", params={"limit": 50}).json()["conversations"]
    assert next(item for item in relisted if item["target"] == target)["label"] == "Renamed chat"

    searched = client.get("/api/llm/conversations", params={"q": "renamed"}).json()
    assert any(item["target"] == target for item in searched["conversations"])
    assert not client.get("/api/llm/conversations", params={"q": "zzz-no-such-chat"}).json()["conversations"]


def test_rename_rejects_empty(client):
    created = client.post("/api/llm/create-conversation", json={"provider": "claude"}).json()["conversation"]
    result = client.post("/api/llm/rename-conversation", json={"target": created["target"], "name": "  "}).json()
    assert result["ok"] is False
    assert "empty" in result["error"].lower()


def test_missing_conversation_is_reported(client):
    body = client.get("/api/llm/conversation", params={"target": "llm:codex:local:does-not-exist"}).json()
    assert body["ok"] is False
    assert body["error"] == "Conversation was not found."


def test_send_message_requires_known_target(client):
    body = client.post("/api/llm/send-message", json={"target": "llm:codex:local:nope", "text": "hi"}).json()
    assert body == {"ok": False, "status": "not_found", "error": "Conversation was not found."}

    empty = client.post("/api/llm/send-message", json={"target": "", "text": ""}).json()
    assert empty["status"] == "invalid_request"


def test_switchuser_status_and_system_users(client):
    statuses = client.get("/api/switchuser/status").json()
    assert statuses["ok"] is True
    assert statuses["statuses"] == []

    single = client.get("/api/switchuser/status", params={"username": "nobody-here"}).json()
    assert single["status"]["status"] == "idle"

    users = client.get("/api/system/users").json()
    assert "ok" in users and "users" in users


def test_desktop_only_routes_report_unavailable(client):
    for path in ("/api/file/open", "/api/file/reveal"):
        body = client.post(path, json={}).json()
        assert body["ok"] is False
        assert "desktop app" in body["error"]


def test_context_menu_for_unknown_target(client):
    body = client.get("/api/context-menu", params={"target": "nope"}).json()
    assert body == {"ok": True, "renamable": False, "continueAs": None}


def test_browser_log_is_accepted(client):
    assert client.post("/api/log", json={"level": "info", "message": "hello", "details": {"a": 1}}).json() == {"ok": True}


def test_api_responses_are_not_cached(client):
    response = client.get("/api/config")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-devagents-version"]


def test_static_index_is_stamped_and_framable(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "DEVAGENTS_DEPLOY_VERSION" not in response.text
    assert 'data-devagents-version="' in response.text
    assert "frame-ancestors" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"


def test_asset_traversal_is_blocked(client):
    assert client.get("/assets/../../../etc/passwd").status_code == 404
    assert client.get("/definitely-not-a-route").status_code == 404


def test_websocket_sends_config_and_status_then_events(client):
    with client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        assert first["type"] == "config"
        assert "channel" in first
        second = socket.receive_json()
        assert second["type"] == "status"
        assert second["state"] in ("idle", "connected")

        # A conversation created over HTTP must reach the socket as a live event —
        # this is the path every agent reply travels.
        created = client.post("/api/llm/create-conversation", json={"provider": "codex", "name": "WS check"}).json()
        event = socket.receive_json()
        assert event["type"] == "llm-conversation"
        assert event["conversation"]["target"] == created["conversation"]["target"]
