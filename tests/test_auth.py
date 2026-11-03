"""Tests for the API token gate.

The service reads every conversation on the machine, lists its OS accounts, and can start
an agent CLI turn. It is reachable from the public internet through a Cloudflare tunnel and
internet scanners already probe it, so "an anonymous caller gets nothing" has to be a tested
property rather than an intention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

TOKEN = "test-token-6b1f9c"

# Everything a scanner would try, plus the endpoints that actually expose data or act.
SENSITIVE_PATHS = [
    "/api/version",
    "/api/config",
    "/api/llm/conversations",
    "/api/llm/conversation?target=x",
    "/api/system/users",
    "/api/switchuser/status",
    "/api/datasources",
    "/api/context-menu?target=x",
    "/openapi.json",
    "/docs",
]


def build_client(monkeypatch, tmp_path, token: str | None):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "client" / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "client" / "web" / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_OWNER_EMAIL", "owner@example.com")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_OWNER_UID", "uid-owner-test")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "firebase-test-project")
    if token is None:
        monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", token)

    from subjective_devagents_fastapi.app import app

    return TestClient(app)


@pytest.fixture()
def guarded(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path, TOKEN) as client:
        yield client


@pytest.fixture()
def open_gate(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path, None) as client:
        yield client


# --- the gate ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path", SENSITIVE_PATHS)
def test_anonymous_callers_are_refused(guarded, path):
    response = guarded.get(path)
    assert response.status_code == 401, f"{path} answered an anonymous caller"
    assert "conversations" not in response.text.lower() or response.json() == {"ok": False, "error": "Unauthorized."}


@pytest.mark.parametrize("path", SENSITIVE_PATHS)
def test_a_valid_token_is_let_through(guarded, path):
    response = guarded.get(path, headers={"X-DevAgents-Token": TOKEN})
    assert response.status_code != 401, f"{path} rejected a valid token"


def test_a_wrong_token_is_refused(guarded):
    assert guarded.get("/api/config", headers={"X-DevAgents-Token": "wrong"}).status_code == 401


def test_a_token_prefix_is_not_enough(guarded):
    """Guards against a substring/startswith comparison creeping in."""
    assert guarded.get("/api/config", headers={"X-DevAgents-Token": TOKEN[:-1]}).status_code == 401
    assert guarded.get("/api/config", headers={"X-DevAgents-Token": TOKEN + "x"}).status_code == 401


def test_bearer_authorization_is_accepted(guarded):
    assert guarded.get("/api/config", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_writes_are_gated_too(guarded):
    """A scanner that only POSTs must not be able to start a turn or change config."""
    for path, body in [
        ("/api/llm/send-message", {"target": "x", "text": "hi"}),
        ("/api/llm/create-conversation", {"provider": "codex"}),
        ("/api/save-config", {"active": {"host": "evil.example"}}),
        ("/api/switchuser/continue-as", {"target": "x", "username": "root"}),
        ("/api/execute-command", {"command": "/raw QUIT"}),
    ]:
        assert guarded.post(path, json=body).status_code == 401, f"{path} accepted an anonymous write"


def test_the_scanner_paths_from_the_logs_get_nothing(guarded):
    """The exact probes seen in production: /mcp, /jsonrpc, /sse, security.txt."""
    for path in ("/mcp", "/mcp/", "/jsonrpc", "/sse", "/.well-known/security.txt"):
        assert guarded.get(path).status_code in (401, 404, 405)


def test_the_health_probe_stays_open_and_says_nothing(guarded):
    """Monitoring must not need the secret, and must not leak anything either."""
    response = guarded.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_static_ui_stays_public(guarded):
    """The page is an empty shell without the API; gating it would break the iframe for no
    security gain, since every byte of it is already in a public git repo."""
    assert guarded.get("/").status_code == 200


def test_an_unset_token_leaves_the_gate_open(open_gate):
    """Backwards compatible on purpose: an existing deployment must not lock itself out on
    restart. The startup log says loudly that it is open."""
    assert open_gate.get("/api/config").status_code == 200


# --- the websocket ------------------------------------------------------------------------------


def test_the_event_socket_is_gated(guarded):
    """The socket streams every message of every conversation, so it needs the same gate —
    and HTTP middleware does not run for a WebSocket scope, so it is checked separately."""
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with guarded.websocket_connect("/ws") as socket:
            socket.receive_json()


def test_the_event_socket_accepts_a_token_in_the_query(guarded):
    """A browser cannot set headers on a WebSocket handshake, so ?token= is the only place
    it can go."""
    with guarded.websocket_connect(f"/ws?token={TOKEN}") as socket:
        assert socket.receive_json()["type"] == "config"


def test_the_event_socket_refuses_a_wrong_query_token(guarded):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with guarded.websocket_connect("/ws?token=wrong") as socket:
            socket.receive_json()


# --- the unit itself ------------------------------------------------------------------------------


def test_protected_prefix_matching():
    from subjective_devagents_fastapi import auth

    for path in ("/api/config", "/api/llm/send-message", "/docs", "/openapi.json", "/redoc"):
        assert auth.path_requires_token(path), f"{path} should be gated"
    for path in ("/", "/healthz", "/renderer.js", "/assets/vendor/xterm.js", "/terminal.html"):
        assert not auth.path_requires_token(path), f"{path} should stay public"


def test_the_token_is_never_stamped_into_the_served_page(guarded):
    """The page is public. A token baked into it would be public too."""
    html = guarded.get("/").text
    assert TOKEN not in html


# --- Firebase owner sign-in, alongside the static secret --------------------------------------
#
# firebase_auth.verify_firebase_id_token is mocked throughout: it is a verbatim port of the
# same RS256/JWKS verification already used (and already exercised against the real Firebase
# project) by subjective_pipeline_fastapi and friends. What is new and actually needs testing
# here is auth.py's own logic layered on top of it — the owner allowlist, and that a bad or
# mismatched token cannot substitute for the real operator.


def test_a_firebase_token_for_the_owner_is_accepted(guarded, monkeypatch):
    from subjective_devagents_fastapi import auth, firebase_auth

    monkeypatch.setattr(
        firebase_auth,
        "verify_firebase_id_token",
        lambda token: {"sub": auth.owner_uid(), "email": auth.owner_email(), "email_verified": True},
    )
    response = guarded.get("/api/config", headers={"Authorization": "Bearer not-checked-by-the-mock"})
    assert response.status_code == 200


def test_a_firebase_token_for_the_verified_owner_email_is_accepted(guarded, monkeypatch):
    from subjective_devagents_fastapi import auth, firebase_auth

    monkeypatch.setattr(
        firebase_auth,
        "verify_firebase_id_token",
        lambda token: {"sub": "someone-else-entirely", "email": auth.owner_email(), "email_verified": True},
    )
    response = guarded.get("/api/config", headers={"Authorization": "Bearer whatever"})
    assert response.status_code == 200


def test_a_firebase_token_for_the_owner_uid_is_accepted_with_a_different_email(guarded, monkeypatch):
    """The configured UID remains sufficient when the owner changes email."""
    from subjective_devagents_fastapi import auth, firebase_auth

    monkeypatch.setattr(
        firebase_auth,
        "verify_firebase_id_token",
        lambda token: {"sub": auth.owner_uid(), "email": "someone-else@example.com"},
    )
    response = guarded.get("/api/config", headers={"Authorization": "Bearer whatever"})
    assert response.status_code == 200


def test_a_token_that_does_not_verify_as_firebase_is_refused(guarded, monkeypatch):
    """A wrong guess at the static secret must not accidentally fall through to the Firebase
    path and be treated as a signature-verification failure worth investigating — it is just
    not a JWT, and verify_firebase_id_token's own shape check rejects it fast."""
    from subjective_devagents_fastapi import firebase_auth

    def _reject(token):
        raise firebase_auth.FirebaseTokenError("bad_token")

    monkeypatch.setattr(firebase_auth, "verify_firebase_id_token", _reject)
    response = guarded.get("/api/config", headers={"Authorization": "Bearer garbage-not-a-jwt"})
    assert response.status_code == 401


def test_the_static_secret_still_works_once_firebase_auth_exists(guarded):
    """Adding the Firebase path must not have displaced the original credential."""
    assert guarded.get("/api/config", headers={"X-DevAgents-Token": TOKEN}).status_code == 200


def test_an_unset_static_token_still_leaves_the_gate_open_even_with_a_bad_firebase_token(open_gate):
    """Backwards compatibility must hold regardless of what a caller sends: a deployment that
    has never configured SUBJECTIVE_DEVAGENTS_API_TOKEN stays fully open, exactly as before
    Firebase auth was added — token_is_valid's "gate disabled" branch returns before the
    Firebase check ever runs."""
    assert open_gate.get("/api/config", headers={"Authorization": "Bearer nonsense"}).status_code == 200


# --- agents' read-only workflow access (/api/workflow/*) ---------------------

WORKFLOW_TOKEN = "w" * 40
EXPORT = {
    "version": {"version": "20261009190101"},
    "files": [
        {"path": "AGENTS.md", "category": "contract", "revision": 1, "content": "# Contract"},
        {"path": "skills/browser-verification/SKILL.md", "category": "skill", "revision": 2, "content": "# Browser"},
    ],
}


@pytest.fixture()
def workflow_client(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import agent_workflow

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_WORKFLOW_READ_TOKEN", WORKFLOW_TOKEN)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_WORKFLOW_COMPANY", "company_54961bc50fe94bbab712")
    calls = []
    monkeypatch.setattr(agent_workflow, "fetch_export_as_service", lambda company: calls.append(company) or EXPORT)
    client = build_client(monkeypatch, tmp_path, "owner-only-secret-token")
    client.calls = calls
    return client


def test_workflow_files_need_the_read_token(workflow_client):
    assert workflow_client.get("/api/workflow/files").status_code == 401
    assert workflow_client.get("/api/workflow/files", headers={"X-Workflow-Token": "x" * 40}).status_code == 401


def test_the_read_token_lists_and_reads_published_files(workflow_client):
    headers = {"X-Workflow-Token": WORKFLOW_TOKEN}
    listed = workflow_client.get("/api/workflow/files", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["version"] == "20261009190101"
    assert [f["path"] for f in listed.json()["files"]] == ["AGENTS.md", "skills/browser-verification/SKILL.md"]
    assert workflow_client.calls == ["company_54961bc50fe94bbab712"]

    one = workflow_client.get("/api/workflow/file", params={"path": "workflow/skills/browser-verification/SKILL.md"}, headers=headers)
    assert one.status_code == 200
    assert one.text == "# Browser"
    assert one.headers["X-Workflow-Version"] == "20261009190101"
    assert workflow_client.get("/api/workflow/file", params={"path": "nope.md"}, headers=headers).status_code == 404


def test_the_read_token_opens_nothing_else(workflow_client):
    headers = {"X-Workflow-Token": WORKFLOW_TOKEN}
    assert workflow_client.post("/api/workflow/files", headers=headers).status_code == 401
    assert workflow_client.get("/api/llm/conversations", headers=headers).status_code == 401


def test_without_a_configured_read_token_the_routes_stay_shut(monkeypatch, tmp_path):
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_WORKFLOW_READ_TOKEN", raising=False)
    client = build_client(monkeypatch, tmp_path, "owner-only-secret-token")
    assert client.get("/api/workflow/files", headers={"X-Workflow-Token": ""}).status_code == 401
