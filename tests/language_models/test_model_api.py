"""Stateful HTTP-to-process tests: no datasource imports, subscriptions or UI needed."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from subjective_devagents_fastapi.language_models import (
    Identity, IdentityRegistry, LanguageModel, ModelError, ModelResult, ModelStrategy, TurnStore,
)
from subjective_devagents_fastapi.language_models.api import router


def await_turn(store, identifier, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = store.turn(identifier)
        if turn["status"] not in {"queued", "running"}:
            return turn
        time.sleep(0.01)
    pytest.fail(f"Turn did not finish: {identifier}")


@pytest.fixture
def engine(tmp_path):
    # The fake executable enforces account-scoped session ids and emits the actual
    # Claude/Codex wire envelopes, exercising HTTP -> SQLite -> subprocess -> parser.
    script = tmp_path / "provider"
    script.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
prompt = sys.stdin.read()
account = os.path.basename(os.environ["HOME"])
session = account + "-session"
if "--resume" in sys.argv:
 assert sys.argv[sys.argv.index("--resume") + 1] == session
if "resume" in sys.argv:
 assert sys.argv[sys.argv.index("resume") + 1] == session
text = "reply:" + account
if "old marker" in prompt:
 text += ":old marker"
if "exec" in sys.argv:
 events = [{"type":"thread.started","thread_id":session},
           {"type":"item.completed","item":{"type":"agent_message","text":text}},
           {"type":"turn.completed"}]
else:
 events = [{"type":"system","session_id":session},
           {"type":"result","subtype":"success","session_id":session,"result":text}]
for event in events:
 print(json.dumps(event), flush=True)
''')
    script.chmod(0o700)
    identities = [Identity(key, provider, home=str(tmp_path / key), config_dir=str(tmp_path / key / "config"),
                           workspace=str(tmp_path), executable=str(script))
                  for key, provider in [("claude-a", "claude"), ("claude-b", "claude"), ("codex-a", "codex")]]
    for identity in identities:
        Path(identity.home).mkdir()
    model = LanguageModel(TurnStore(tmp_path / "turns.sqlite3"), IdentityRegistry(identities))
    try:
        yield model
    finally:
        model.close()


@pytest.fixture
def client(engine):
    app = FastAPI()
    app.state.language_model = engine
    app.include_router(router)
    with TestClient(app) as client:
        yield client


def test_http_account_and_provider_handoff_keeps_context_and_isolates_native_sessions(client, engine):
    created = client.post("/api/language-model/conversations", json={"identity_id": "claude-a"})
    assert created.status_code == 201
    conversation_id = created.json()["id"]
    root = f"/api/language-model/conversations/{conversation_id}"
    first = client.post(root + "/turns", json={"prompt": "old marker", "request_id": "first"})
    assert first.status_code == 202
    turn = await_turn(engine.store, first.json()["id"])
    assert turn["session_id"] == "claude-a-session" and turn["status"] == "completed"

    duplicate = client.post(root + "/turns", json={"prompt": "old marker", "request_id": "first"})
    assert duplicate.json()["id"] == turn["id"]
    assert client.post(root + "/turns", json={"prompt": "different", "request_id": "first"}).status_code == 409

    for index, identity in enumerate(("claude-b", "codex-a", "claude-a")):
        handoff = client.post(root + "/handoff", json={"identity_id": identity})
        assert handoff.status_code == 200 and handoff.json()["session_id"] is None
        sent = client.post(root + "/turns", json={"prompt": "remember previous context", "request_id": f"switch-{index}"})
        turn = await_turn(engine.store, sent.json()["id"])
        assert turn["status"] == "completed"
        assert turn["session_id"] == identity + "-session"
        assert "old marker" in turn["text"]
    conversation = client.get(root).json()
    assert len(conversation["messages"]) == 8
    assert len(conversation["handoffs"]) == 3
    # Durable cursors survive a client reconnect; no socket event is required for results.
    events = client.get(f"/api/language-model/turns/{turn['id']}/events").json()["events"]
    assert events[-1]["data"]["status"] == "completed"
    assert client.get(f"/api/language-model/turns/{turn['id']}/events?after={events[-1]['sequence']}").json() == {"events": []}
    assert TurnStore(engine.store.path).turn(turn["id"])["text"] == turn["text"]


def test_same_identity_resumes_native_session(client, engine):
    conversation = engine.create("claude-a")
    first = engine.submit(conversation["id"], "first", "first")
    assert await_turn(engine.store, first["id"])["status"] == "completed"
    second = engine.submit(conversation["id"], "second", "second")
    result = await_turn(engine.store, second["id"])
    assert result["status"] == "completed" and result["session_id"] == "claude-a-session"


class Waiting(ModelStrategy):
    def __init__(self):
        self.started = threading.Event()
        self.calls = 0

    def run(self, request, emit, cancelled):
        self.calls += 1
        self.started.set()
        deadline = time.monotonic() + 4
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        return ModelResult("cancelled", error_code="cancelled")


def test_concurrent_requests_are_rejected_and_cancel_persists_across_instances(engine):
    waiting = Waiting()
    engine.strategies["cli:claude"] = waiting
    conversation = engine.create("claude-a")
    first = engine.submit(conversation["id"], "first", "first")
    assert waiting.started.wait(2)
    with pytest.raises(ModelError) as error:
        engine.submit(conversation["id"], "overlap", "second")
    assert error.value.code == "busy"
    with pytest.raises(ModelError, match="Finish or cancel"):
        engine.handoff(conversation["id"], "codex-a")
    assert engine.submit(conversation["id"], "first", "first")["id"] == first["id"]
    TurnStore(engine.store.path).cancel(first["id"])
    assert await_turn(engine.store, first["id"])["status"] == "cancelled"
    assert waiting.calls == 1


def test_strategy_exception_is_not_replayed_and_requires_reconciliation(engine):
    class Crash(ModelStrategy):
        calls = 0
        def run(self, *_):
            self.calls += 1
            raise RuntimeError("sensitive provider diagnostic")
    crash = Crash()
    engine.strategies["cli:claude"] = crash
    conversation = engine.create("claude-a")
    turn = engine.submit(conversation["id"], "run once", "one")
    result = await_turn(engine.store, turn["id"])
    assert result["status"] == "interrupted" and result["error_code"] == "worker_exception"
    assert "sensitive" not in json.dumps(result)
    assert crash.calls == 1
    with pytest.raises(ModelError) as error:
        engine.submit(conversation["id"], "another", "two")
    assert error.value.code == "reconciliation_required"
    with pytest.raises(ModelError):
        engine.reconcile(turn["id"], False)
    engine.reconcile(turn["id"], True)
    assert not engine.store.conversation(conversation["id"])["needs_reconciliation"]


def test_restart_does_not_replay_abandoned_turn(engine):
    conversation = engine.create("claude-a")
    identity = engine.identities.get("claude-a")
    turn, _ = engine.store.reserve(conversation["id"], identity, "abandoned", "do not replay", 30)
    engine.store.claim(turn["id"])
    replacement = LanguageModel(TurnStore(engine.store.path), engine.identities)
    try:
        replay = replacement.submit(conversation["id"], "do not replay", "abandoned", 30)
        assert replay["id"] == turn["id"] and replay["status"] == "running"
        assert replacement._futures == {}
        replacement.reconcile(turn["id"], True)
        assert replacement.store.turn(turn["id"])["status"] == "interrupted"
    finally:
        replacement.close()


def test_shutdown_cancels_active_turn_and_refuses_new_work(engine):
    waiting = Waiting()
    engine.strategies["cli:claude"] = waiting
    conversation = engine.create("claude-a")
    turn = engine.submit(conversation["id"], "wait", "one")
    assert waiting.started.wait(2)
    engine.close()
    assert engine.store.turn(turn["id"])["status"] == "cancelled"
    with pytest.raises(ModelError) as error:
        engine.submit(conversation["id"], "new", "two")
    assert error.value.code == "unavailable"


def test_http_rejects_unsupported_features_and_unknown_identity(client):
    assert client.post("/api/language-model/conversations", json={"identity_id": "unknown"}).status_code == 404
    created = client.post("/api/language-model/conversations", json={"identity_id": "claude-a"}).json()
    assert client.post(f"/api/language-model/conversations/{created['id']}/turns",
                       json={"prompt": "x", "request_id": "x", "attachments": []}).status_code == 422


def test_existing_application_auth_gate_protects_new_routes(monkeypatch):
    from subjective_devagents_fastapi.app import app
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "token")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", "fixture-token")
    # No lifespan: this verifies middleware without starting legacy datasource runtime.
    client = TestClient(app)
    assert client.get("/api/language-model/identities").status_code == 401
    response = client.get("/api/language-model/identities", headers={"X-DevAgents-Token": "fixture-token"})
    assert response.status_code == 503


def test_existing_app_starts_configured_engine_and_closes_it(engine, tmp_path, monkeypatch):
    from subjective_devagents_fastapi.app import app
    identities = [engine.identities.get(name).snapshot() for name in ("claude-a", "claude-b", "codex-a")]
    config = tmp_path / "identities.json"
    config.write_text(json.dumps({"identities": identities}))
    root = tmp_path / "client"
    (root / "web").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "web" / "index.html").write_text("<!doctype html><title>Fixture</title>")
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_CONFIG", str(config))
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_DB", str(tmp_path / "configured.sqlite3"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "legacy-home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(root))
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_SYNC", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", raising=False)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert len(client.get("/api/language-model/identities").json()["identities"]) == 3
        conversation = client.post("/api/language-model/conversations", json={"identity_id": "claude-a"}).json()
        response = client.post(f"/api/language-model/conversations/{conversation['id']}/turns",
                               json={"prompt": "startup test", "request_id": "startup"})
        assert response.status_code == 202
        result = await_turn(app.state.language_model.store, response.json()["id"])
        assert result["status"] == "completed"
    assert app.state.language_model is None
