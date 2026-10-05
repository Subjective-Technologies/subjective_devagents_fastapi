"""API-surface tests: every route the browser UI calls answers in the shape it expects.

Runs against a throwaway SUBJECTIVE_CHAT_HOME so no real conversation state is touched,
and with IRC disabled (the default) so nothing dials out.
"""
from __future__ import annotations

import json
import re
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
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_SYNC", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True)
    (tmp_path / "client" / "src").mkdir(parents=True)
    # Mirrors the real page's head: the version placeholder, the API origin meta tag and a
    # CSP with connect-src, since serve-time stamping rewrites all three.
    (tmp_path / "client" / "web" / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><title>DevAgents</title>\n'
        '<meta name="devagents-api" content="https://devagents-api.subjectivetechnologies.com:8080">\n'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; '
        "connect-src 'self' ws: wss: https://devagents-api.subjectivetechnologies.com:8080; "
        'img-src \'self\' data:;">\n'
        "</head><body>DEVAGENTS_DEPLOY_VERSION</body></html>",
        encoding="utf-8",
    )

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as test_client:
        test_client.chat_home = chat_home
        yield test_client


def test_version(client):
    from subjective_devagents_fastapi.app import resolve_deploy_version

    body = client.get("/api/version").json()
    assert body["ok"] is True
    assert body["release"]
    assert body["version"].startswith(body["release"])
    assert body["backendVersion"] == resolve_deploy_version(SERVICE_ROOT)
    assert body["backendVersion"] != body["version"]  # fixture UI has no Git checkout
    assert client.get("/healthz").headers["x-devagents-backend-version"] == body["backendVersion"]


def test_root_is_api_index_when_ui_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "chat-home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SERVE_UI", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_SYNC", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_WEB_URL", "https://devagents.subjectivetechnologies.com/")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True)
    (tmp_path / "client" / "src").mkdir(parents=True)

    from subjective_devagents_fastapi.app import app

    with TestClient(app) as test_client:
        body = test_client.get("/").json()
        assert body["ok"] is True
        assert body["docs"] == "/docs"
        assert body["web"] == "https://devagents.subjectivetechnologies.com/"
        assert test_client.get("/index.html").status_code == 404
        assert test_client.get("/docs").status_code == 200


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


def test_test_endpoints_are_disabled_by_default(client):
    result = client.post(
        "/api/llm/test/inject-assistant-message", json={"target": "llm:codex:local:x", "text": "hi"}
    ).json()
    assert result == {"ok": False, "error": "Test endpoints are disabled on this deployment."}


def test_injected_assistant_message_carries_prompt_options(client, monkeypatch):
    from subjective_devagents_fastapi import app as app_module

    monkeypatch.setattr(app_module, "TEST_ENDPOINTS_ENABLED", True)

    created = client.post("/api/llm/create-conversation", json={"provider": "codex", "name": "E2E target"}).json()
    target = created["conversation"]["target"]

    injected = client.post(
        "/api/llm/test/inject-assistant-message",
        json={"target": target, "text": "Should I proceed with this change?"},
    ).json()
    assert injected["ok"] is True
    assert injected["assistantMessage"]["role"] == "assistant"

    fetched = client.get("/api/llm/conversation", params={"target": target}).json()
    last = fetched["conversation"]["messages"][-1]
    assert last["text"] == "Should I proceed with this change?"
    options = last["promptOptions"]["options"]
    assert [option["id"] for option in options] == ["yes", "no"]
    assert last["promptOptions"]["allows_other"] is True


def test_activate_conversation_reports_authoritative_busy_state(client):
    """activate-conversation runs on every single conversation open (unconditionally, not
    gated on first-hydrate like the /api/llm/conversation recovery path) — it is what
    actually self-heals a conversation whose prompt-option/code-run buttons got stuck
    disabled by a missed llm-busy:false (e.g. a dropped websocket mid-turn), for as long as
    the tab stays open, not just the first time that conversation is ever opened."""
    from subjective_devagents_fastapi.app import core

    created = client.post("/api/llm/create-conversation", json={"provider": "codex", "name": "Busy check"}).json()
    target = created["conversation"]["target"]

    idle = client.post("/api/llm/activate-conversation", json={"target": target}).json()
    assert idle["ok"] is True
    assert idle["busy"] is False

    # Simulate a turn stuck in flight (or, the actual failure mode: a turn that finished
    # but whose llm-busy:false the client never saw) without needing a real CLI call.
    core().llm_service.busy.add(target)
    try:
        busy = client.post("/api/llm/activate-conversation", json={"target": target}).json()
        assert busy["ok"] is True
        assert busy["busy"] is True
    finally:
        core().llm_service.busy.discard(target)

    recovered = client.post("/api/llm/activate-conversation", json={"target": target}).json()
    assert recovered["ok"] is True
    assert recovered["busy"] is False


def test_import_chats_zip(client):
    import io
    import zipfile

    chat = {
        "type": "localchat",
        "title": "Imported from laptop",
        "product": "claude",
        "kind": "chat",
        "source_file": "/home/other/.claude/projects/x/session-abc.jsonl",
        "messages": [
            {"role": "user", "text": "hello from another machine", "timestamp": 1000},
            {"role": "assistant", "text": "hi there", "timestamp": 2000},
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("chat-1.json", json.dumps(chat))
        archive.writestr("not-a-chat.txt", "ignored")
    buffer.seek(0)

    result = client.post(
        "/api/llm/import-chats", files={"file": ("chats.zip", buffer.getvalue(), "application/zip")}
    ).json()
    assert result == {"ok": True, "imported": 1, "skipped": 0, "errors": []}

    listed = client.get("/api/llm/conversations", params={"q": "imported from laptop"}).json()
    assert listed["total"] == 1
    target = listed["conversations"][0]["target"]
    fetched = client.get("/api/llm/conversation", params={"target": target}).json()
    assert fetched["ok"] is True
    assert [m["text"] for m in fetched["conversation"]["messages"]] == ["hello from another machine", "hi there"]

    # Re-importing the same zip is a no-op — deduped by (provider, sessionId, sourceFile).
    buffer.seek(0)
    again = client.post(
        "/api/llm/import-chats", files={"file": ("chats.zip", buffer.getvalue(), "application/zip")}
    ).json()
    assert again == {"ok": True, "imported": 0, "skipped": 1, "errors": []}


def test_import_chats_rejects_bad_zip(client):
    result = client.post(
        "/api/llm/import-chats", files={"file": ("chats.zip", b"not a zip", "application/zip")}
    ).json()
    assert result["ok"] is False
    assert "zip" in result["error"].lower()


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
    from subjective_devagents_fastapi.app import state

    response = client.get("/")
    assert response.status_code == 200
    assert "DEVAGENTS_DEPLOY_VERSION" not in response.text
    assert 'data-devagents-version="' in response.text
    assert 'data-devagents-instance="' in response.text
    assert "frame-ancestors" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"
    # The shipped page already contains the placeholder, so the comment must
    # still be injected — otherwise curl of / cannot read the commit.
    match = re.search(
        r"<!-- devagents-deploy version=(?P<version>\S+) backend=(?P<backend>\S+) instance=(?P<instance>\S+) served=\S+ -->",
        response.text,
    )
    assert match, response.text.splitlines()[:8]
    assert match.group("version") == state.deploy_version
    assert match.group("backend") == state.backend_version
    assert match.group("instance") == state.instance_id
    assert response.headers["x-devagents-version"] == state.deploy_version
    assert response.headers["x-devagents-backend-version"] == state.backend_version


def test_instance_id_is_stable_within_a_run_and_matches_the_stamped_page(client):
    """instanceId exists precisely to catch a plain restart with no new commit — the most
    common case during active development, where the git-SHA-based version string does not
    change but every already-open tab is still running pre-restart JS."""
    import re

    first = client.get("/api/version").json()["instanceId"]
    second = client.get("/api/version").json()["instanceId"]
    assert first and first == second

    page = client.get("/").text
    match = re.search(r'data-devagents-instance="([^"]+)"', page)
    assert match and match.group(1) == first


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


def test_history_manifest_with_real_sources_is_parsed(tmp_path, monkeypatch):
    """Regression: the sidebar listing walks a manifest, and every earlier test ran against
    an empty archive — so a crash reading a populated manifest reached a deploy. Build one
    with a session uuid in the path and assert the conversation comes back."""
    import sys

    sys.path.insert(0, str(SERVICE_ROOT / "src"))
    from subjective_devagents_fastapi.llm_service import LlmService

    home = tmp_path / "chat-home"
    context = home / ".subjective_chat_history" / "context"
    context.mkdir(parents=True)
    export = context / "2026_04_26_10_38_51_Localchats-abc-context.json"
    export.write_text(json.dumps({
        "type": "localchat", "title": "Imported chat", "product": "claude",
        "messages": [{"role": "user", "text": "hello", "time": 1_760_000_000_000}],
    }), encoding="utf-8")
    (context / ".subjective_localchats_manifest.json").write_text(json.dumps({
        "sources": {
            "a": {
                "source_file": "/home/goldenthinker/.claude/projects/2f0f9c2e-1111-2222-3333-444455556666.jsonl",
                "export_filename": export.name,
                "title": "Imported chat",
                "message_count": 1,
                "last_activity_at": 1_760_000_000_000,
                "fingerprint": {"product": "claude", "source_user": "goldenthinker", "mtime_ns": 1_760_000_000_000_000_000},
            }
        }
    }), encoding="utf-8")

    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(home))
    service = LlmService(root=str(tmp_path), app_home=str(home))
    listed = service.list_conversations(limit=10)
    assert listed["total"] == 1
    conversation = listed["conversations"][0]
    assert conversation["label"] == "Imported chat"
    assert conversation["provider"] == "claude"
    # The session uuid must be lifted out of the source path, not dropped or crashed on.
    assert conversation["sessionId"] == "2f0f9c2e-1111-2222-3333-444455556666"


def test_api_origin_is_stamped_at_serve_time(client, monkeypatch):
    """Which backend a page talks to belongs to the deployment, not the checkout. Serving
    the UI from this process must be able to point the page at itself, or it would drive
    another machine's agent CLIs."""
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_PUBLIC_API", "https://api.example.test:8080")
    html = client.get("/").text
    assert '<meta name="devagents-api" content="https://api.example.test:8080">' in html
    # CSP has to name the same origin or the browser blocks the call before sending it.
    csp = [line for line in html.splitlines() if "connect-src" in line][0]
    assert "https://api.example.test:8080" in csp


def test_empty_api_origin_means_same_origin(client, monkeypatch):
    """The default when this process serves the UI: the page talks to whoever served it."""
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_PUBLIC_API", "")
    html = client.get("/").text
    assert '<meta name="devagents-api" content="">' in html
    csp = [line for line in html.splitlines() if "connect-src" in line][0]
    assert "https://" not in csp.split("connect-src")[1].split(";")[0]


def test_unset_api_origin_leaves_the_shipped_default(client, monkeypatch):
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_PUBLIC_API", raising=False)
    html = client.get("/").text
    assert 'name="devagents-api"' in html


def test_stamping_rewrites_the_real_pages_policy_not_its_prose(monkeypatch):
    """Regression: index.html explains connect-src in a comment above the tag, and a regex
    anchored on the first 'connect-src' rewrote that prose while leaving the real policy
    intact — the page then looks configured while the browser still blocks every call.
    Runs against the shipped page, since a synthetic fixture has no such comment."""
    import sys

    sys.path.insert(0, str(SERVICE_ROOT / "src"))
    from subjective_devagents_fastapi.app import _stamp_api_base

    page = SERVICE_ROOT / ".." / ".." / "clients" / "desktop" / "subjective_devagents" / "web" / "index.html"
    if not page.exists():
        pytest.skip("DevAgents client checkout not present")
    html = page.read_text(encoding="utf-8")

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_PUBLIC_API", "https://api.example.test:9000")
    out = _stamp_api_base(html)
    policy = re.search(r'<meta\s+http-equiv="Content-Security-Policy"[^>]*>', out).group(0)
    connect = re.search(r"connect-src[^;\"]*;", policy).group(0)
    assert "https://api.example.test:9000" in connect
    assert "devagents-api.subjectivetechnologies.com" not in connect
    # The explanatory comment must survive untouched.
    assert "connect-src must name the API origin explicitly" in out

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_PUBLIC_API", "")
    same_origin = _stamp_api_base(html)
    policy = re.search(r'<meta\s+http-equiv="Content-Security-Policy"[^>]*>', same_origin).group(0)
    connect = re.search(r"connect-src[^;\"]*;", policy).group(0)
    # Same-origin means the page names no *API* host. The fixed allowlist the page needs
    # whatever the deployment is — Firebase auth, and the Cloudflare Web Analytics beacon
    # the zone injects into every HTML response — is kept on purpose, so assert on which
    # hosts survive rather than on "no host at all".
    allowed = {
        "https://identitytoolkit.googleapis.com",
        "https://securetoken.googleapis.com",
        "https://www.googleapis.com",
        "https://subjectivetechnologies-login.firebaseapp.com",
        "https://cloudflareinsights.com",
    }
    hosts = set(re.findall(r"https?://[^\s;]+", connect))
    assert hosts <= allowed, f"same-origin policy names an API host: {sorted(hosts - allowed)}"
    assert '<meta name="devagents-api" content="">' in same_origin


def test_every_creatable_agent_is_listed_for_the_picker(client):
    """The new-chat picker hard-coded codex/claude/cursor in two index.html files and
    drifted: Grok was creatable on the backend but absent from the UI, and Cursor was
    offered even where its datasource module is missing. The list has to come from the
    same WRITABLE_PROVIDERS the create call validates against."""
    from subjective_devagents_fastapi.llm_service import WRITABLE_PROVIDERS

    payload = client.get("/api/llm/providers").json()

    assert payload["ok"] is True
    listed = {item["key"]: item for item in payload["providers"]}
    assert set(listed) == set(WRITABLE_PROVIDERS), "picker list drifted from the backend"
    assert "grok" in listed, "Grok is creatable and must be offered"
    assert payload["default"] in listed
    for key, provider in WRITABLE_PROVIDERS.items():
        assert listed[key]["label"] == provider["label"]
        assert isinstance(listed[key]["available"], bool)


def test_every_listed_agent_can_actually_be_created(client):
    """A provider offered by the picker must survive create-conversation, so the list can
    never advertise something the backend rejects."""
    for item in client.get("/api/llm/providers").json()["providers"]:
        created = client.post(
            "/api/llm/create-conversation",
            json={"provider": item["key"], "name": f"{item['key']} smoke"},
        ).json()
        assert created["ok"] is True, (item["key"], created)
        assert created["conversation"]["provider"] == item["key"]


def test_stop_endpoint_is_safe_on_an_idle_chat(client):
    """Stop must be idempotent: pressing it when nothing is running is a no-op, not an
    error, because the UI cannot know the turn just finished."""
    created = client.post(
        "/api/llm/create-conversation", json={"provider": "claude", "name": "idle"}
    ).json()

    result = client.post(
        "/api/llm/cancel-message", json={"target": created["conversation"]["target"]}
    ).json()

    assert result["ok"] is True
    assert result["running"] is False
    assert result["stopped"] is False


def test_stop_without_a_target_is_rejected(client):
    assert client.post("/api/llm/cancel-message", json={}).json()["ok"] is False


def test_a_chat_can_be_created_against_another_users_cli_login(client):
    """Each agent's credentials live in a home directory, so "Codex" really means "Codex
    as <someone>". Naming the account at creation lets a chat start against any of them
    instead of only the service account's, then being switched afterwards."""
    created = client.post(
        "/api/llm/create-conversation",
        json={"provider": "codex", "name": "as someone else", "osUser": "otheruser"},
    ).json()

    assert created["ok"] is True, created
    conversation = created["conversation"]
    assert conversation["osUser"] == "otheruser"
    assert conversation["sourceUser"] == "otheruser"
    # send_message reads osUser to route the turn through that user's session.
    listed = client.get("/api/llm/conversations").json()["conversations"]
    assert any(item["target"] == conversation["target"] and item["osUser"] == "otheruser" for item in listed)


def test_a_chat_for_the_current_user_carries_no_os_user(client):
    """A set osUser means "route through someone else's su session", so the service
    account's own chats must leave it empty or every turn would demand a connection."""
    created = client.post(
        "/api/llm/create-conversation", json={"provider": "codex", "name": "mine"}
    ).json()

    assert not created["conversation"]["osUser"]


def test_an_invalid_os_user_is_rejected(client):
    created = client.post(
        "/api/llm/create-conversation",
        json={"provider": "codex", "name": "bad", "osUser": "../../etc/passwd"},
    ).json()

    assert created["ok"] is False


def test_the_agent_registry_is_served(client, monkeypatch, tmp_path):
    monkeypatch.setenv("SUBJECTIVE_AGENTS_DIR", str(tmp_path))
    (tmp_path / "registry.md").write_text(
        "| Agent ID | Role file | Primary capability |\n|---|---|---|\n"
        "| `manager_agent` | `roles/manager_agent.md` | orchestration |\n",
        encoding="utf-8",
    )
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles" / "manager_agent.md").write_text("# manager_agent\n", encoding="utf-8")

    listed = client.get("/api/agents").json()

    assert listed["ok"] is True
    assert [item["id"] for item in listed["agents"]] == ["manager_agent"]


def test_an_agent_can_be_created_and_re_pointed_at_another_model(client, monkeypatch, tmp_path):
    monkeypatch.setenv("SUBJECTIVE_AGENTS_DIR", str(tmp_path))
    (tmp_path / "registry.md").write_text(
        "| Agent ID | Role file | Primary capability |\n|---|---|---|\n"
        "| `manager_agent` | `roles/manager_agent.md` | orchestration |\n",
        encoding="utf-8",
    )
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles" / "manager_agent.md").write_text("# manager_agent\n", encoding="utf-8")

    created = client.post(
        "/api/agents", json={"id": "triage_agent", "capability": "triage", "model": "claude"}
    ).json()
    assert created["ok"] is True, created

    changed = client.post("/api/agents/model", json={"id": "triage_agent", "model": "grok"}).json()
    assert changed["ok"] is True

    listed = {item["id"]: item for item in client.get("/api/agents").json()["agents"]}
    assert listed["triage_agent"]["model"] == "grok"
    # Never in the registry table: models are runtime metadata, not identity.
    assert "grok" not in (tmp_path / "registry.md").read_text(encoding="utf-8")


def test_creating_an_agent_with_a_bad_id_is_refused(client, monkeypatch, tmp_path):
    monkeypatch.setenv("SUBJECTIVE_AGENTS_DIR", str(tmp_path))
    (tmp_path / "registry.md").write_text("| Agent ID | Role file | Primary capability |\n|---|---|---|\n", encoding="utf-8")
    assert client.post("/api/agents", json={"id": "../evil", "capability": "x"}).json()["ok"] is False
