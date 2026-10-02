"""Cutover acceptance using the real HTTP adapter and persistent execution store."""
import builtins
import json
import subprocess
import threading

from fastapi.testclient import TestClient
import pytest

from subjective_devagents_fastapi.language_models import ModelResult, ModelStrategy
from subjective_devagents_fastapi.llm_service import LlmService


class RecordingProvider(ModelStrategy):
    def __init__(self):
        self.requests = []

    def run(self, request, emit, cancelled):
        self.requests.append(request)
        answer = "answer:" + request.prompt
        emit("message", {"text": answer})
        return ModelResult("completed", answer, request.identity.id + "-session")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "client"
    (root / "web").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "web" / "index.html").write_text("<!doctype html><title>Cutover fixture</title>")
    config = tmp_path / "identities.json"
    config.write_text(json.dumps({"identities": [
        {"id": name, "provider": "codex", "os_user": name,
         "workspace": str(root), "home": str(tmp_path / name),
         "config_dir": str(tmp_path / name / ".codex"), "executable": "/unused/codex"}
        for name in ("alice", "bob")
    ]}))
    for key, value in {
        "SUBJECTIVE_LANGUAGE_MODELS_CONFIG": config,
        "SUBJECTIVE_LANGUAGE_MODELS_DB": tmp_path / "turns.sqlite3",
        "SUBJECTIVE_CHAT_HOME": tmp_path / "chat-home",
        "SUBJECTIVE_DEVAGENTS_CLIENT_ROOT": root,
        "SUBJECTIVE_CHAT_HISTORY_REFRESH_MS": "0",
        "SUBJECTIVE_DEVAGENTS_D1_SYNC": "0",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cutover fixtures must not execute an external process or legacy bundle")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(LlmService, "_run_pipeline", forbidden)
    monkeypatch.setattr(LlmService, "_run_json_script", forbidden)
    return root, tmp_path / "chat-home"


@pytest.fixture
def services(environment):
    root, home = environment
    instances = []
    provider = RecordingProvider()

    def create():
        instance = LlmService(root=root, app_home=home)
        instance.backend.model.strategies = {"cli:codex": provider}
        instances.append(instance)
        return instance

    yield create, provider
    for instance in instances:
        instance.backend.close()


def transcript(service, target):
    return [(message["role"], message["text"])
            for message in service.find_conversation(target, include_messages=True)["messages"]]


def fail_completed_projection(service, monkeypatch):
    save = service._save_state

    def failing_save():
        if any(message.get("role") == "assistant" and message.get("turnId")
               for conversation in service.local for message in conversation["messages"]):
            raise OSError("injected JSON projection failure")
        return save()

    monkeypatch.setattr(service, "_save_state", failing_save)
    return save


def test_legacy_http_send_uses_standalone_execution_and_preserves_events(environment, monkeypatch):
    from subjective_devagents_fastapi.app import app, core

    monkeypatch.setattr("subjective_devagents_fastapi.app.resolve_deploy_version", lambda root: "fixture")
    provider = RecordingProvider()
    with TestClient(app) as client:
        instance = core()
        instance.llm_service.backend.model.strategies = {"cli:codex": provider}
        original_import = builtins.__import__

        def no_datasource_import(name, *args, **kwargs):
            assert "datasource" not in name.lower() and "data_source" not in name.lower(), name
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_datasource_import)
        events = []
        original_emit = instance._emit
        monkeypatch.setattr(instance, "_emit", lambda event: (events.append(event), original_emit(event)))
        finished = threading.Event()
        outcomes = []
        original_send = instance.send_llm_message

        def record_send(*args):
            try:
                result = original_send(*args)
                outcomes.append(result)
                return result
            finally:
                finished.set()

        monkeypatch.setattr(instance, "send_llm_message", record_send)
        created = client.post("/api/llm/create-conversation", json={
            "provider": "codex", "name": "Legacy client", "osUser": "alice"}).json()
        assert created["ok"] is True
        target = created["conversation"]["target"]
        accepted = client.post("/api/llm/send-message", json={"target": target, "text": "hello"})
        assert accepted.status_code == 200
        assert accepted.json() == {"ok": True, "pending": True, "target": target}
        assert finished.wait(5), "Legacy HTTP send did not finish"
        assert outcomes[0]["ok"] is True
        assert len(provider.requests) == 1
        hydrated = client.get("/api/llm/conversation", params={"target": target}).json()
        assert hydrated["ok"] is True
        assert [(m["role"], m["text"]) for m in hydrated["conversation"]["messages"]] == [
            ("user", "hello"), ("assistant", "answer:hello")]
        assert hydrated["conversation"]["busy"] is False
        assert {"llm-message", "llm-conversation", "llm-busy"} <= {event["type"] for event in events}
        assert instance.llm_service.backend.model.store.turn(outcomes[0]["result"]["turn_id"])["status"] == "completed"


@pytest.mark.parametrize("restart_before_reload", [False, True], ids=["same-process", "after-restart"])
def test_completed_turn_recovers_projection_without_replay(services, monkeypatch, restart_before_reload):
    create, provider = services
    service = create()
    target = service.create_conversation(provider="codex", os_user="alice")["target"]
    save = fail_completed_projection(service, monkeypatch)
    result = service.send_message(target, "run once")
    assert result["ok"] is True and result["projectionPending"] is True
    turn_id = result["result"]["turn_id"]
    assert service.backend.pending(target)["id"] == turn_id
    assert len(provider.requests) == 1
    if restart_before_reload:
        service.backend.close()
        service = create()
    else:
        # Failed reload must not acknowledge the only durable recovery pointer.
        with pytest.raises(OSError, match="injected JSON projection failure"):
            service.reload_conversation(target)
        assert service.backend.pending(target)["id"] == turn_id
        monkeypatch.setattr(service, "_save_state", save)
    service.reload_conversation(target)
    assert service.backend.pending(target) is None
    service.backend.close()
    service = create()
    expected = [("user", "run once"), ("assistant", "answer:run once")]
    assert transcript(service, target) == expected
    assert len(provider.requests) == 1
    result = service.send_message(target, "next")
    assert result["ok"] is True
    assert len(provider.requests) == 2
    assert [(m.role, m.content) for m in provider.requests[1].history] == expected
    assert provider.requests[1].session_id == "alice-session"


def test_bound_historical_reload_preserves_new_turns(services, tmp_path):
    create, provider = services
    service = create()
    export = tmp_path / "old-history.json"
    old_messages = [{"role": "user", "text": "old question", "time": 1000},
                    {"role": "assistant", "text": "old answer", "time": 2000}]
    export.write_text(json.dumps({"type": "localchat", "provider": "codex", "messages": old_messages}))
    conversation = service.create_conversation(provider="codex", os_user="alice")
    conversation.update({"sourceFile": str(tmp_path / "historical-native-session.jsonl"),
                         "historyExportPath": str(export), "sessionId": "unverified-old-session"})
    service._save_state()
    target = conversation["target"]
    result = service.send_message(target, "new question")
    assert result["ok"] is True
    assert provider.requests[0].session_id is None
    service.reload_conversation(target)
    expected = [("user", "old question"), ("assistant", "old answer"),
                ("user", "new question"), ("assistant", "answer:new question")]
    assert transcript(service, target) == expected
    assert json.loads(export.read_text())["messages"] == old_messages
    service.backend.close()
    assert transcript(create(), target) == expected
    assert len(provider.requests) == 1


@pytest.mark.parametrize("restart_before_retry", [False, True], ids=["same-process", "after-restart"])
def test_retry_after_projection_failure_returns_durable_result_without_duplicate_transcript(
    services, monkeypatch, restart_before_retry,
):
    create, provider = services
    service = create()
    target = service.create_conversation(provider="codex", os_user="alice")["target"]
    save = fail_completed_projection(service, monkeypatch)
    first = service.send_message(target, "run once")
    assert first["ok"] is True and first["projectionPending"] is True
    if restart_before_retry:
        service.backend.close()
        service = create()
    else:
        monkeypatch.setattr(service, "_save_state", save)
    retried = service.send_message(target, "run once")
    assert retried["ok"] is True
    assert retried["result"]["turn_id"] == first["result"]["turn_id"]
    assert len(provider.requests) == 1
    assert transcript(service, target) == [("user", "run once"), ("assistant", "answer:run once")]
    service.backend.close()
    assert transcript(create(), target) == [("user", "run once"), ("assistant", "answer:run once")]


def test_same_provider_account_handoff_preserves_history_without_resuming_other_account(services):
    create, provider = services
    service = create()
    target = service.create_conversation(provider="codex", os_user="alice")["target"]
    assert service.send_message(target, "remember me")["ok"] is True
    sibling = service.create_sibling_for_user(target, "bob")
    assert sibling["provider"] == "codex" and sibling["target"] != target
    assert not sibling.get("sessionId")
    assert transcript(service, sibling["target"]) == transcript(service, target)
    result = service.send_message(sibling["target"], "continue")
    assert result["ok"] is True
    assert len(provider.requests) == 2
    assert provider.requests[1].identity.id == "bob"
    assert provider.requests[1].session_id is None
    assert [(message.role, message.content) for message in provider.requests[1].history] == [
        ("user", "remember me"), ("assistant", "answer:remember me")]
    assert service.find_conversation(target)["sessionId"] == "alice-session"
    assert service.find_conversation(sibling["target"])["sessionId"] == "bob-session"
