"""Independent compatibility checks with real storage and fake providers."""
import base64
import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from subjective_devagents_fastapi.language_models import legacy
from subjective_devagents_fastapi.language_models.contracts import ModelResult, ModelStrategy


@pytest.fixture(autouse=True)
def prohibit_real_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Compatibility tests must never launch an external process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


class RecordingStrategy(ModelStrategy):
    def __init__(self, result=None):
        self.requests = []
        self.result = result or ModelResult("completed", "answer", "fresh-session")

    def run(self, request, emit, cancelled):
        self.requests.append(request)
        emit("message", {"text": "answer"})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBJECTIVE_LANGUAGE_MODELS_CONFIG", raising=False)
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_DB", str(tmp_path / "turns.sqlite3"))
    monkeypatch.setattr(legacy.pwd, "getpwnam", lambda name: SimpleNamespace(pw_dir=str(tmp_path / name)))
    monkeypatch.setattr(legacy.shutil, "which", lambda *args, **kwargs: "/usr/bin/codex")
    adapter = legacy.LegacyLanguageModel(str(tmp_path))
    yield adapter
    adapter.close()


@pytest.fixture
def conversation(tmp_path):
    return {"target": "legacy-target", "provider": "codex", "osUser": "alice",
            "workspace": str(tmp_path), "sessionId": "unverified-old-session"}


def send(adapter, conversation, source=None, **request):
    return adapter.send(conversation, source or {}, {"prompt": "question", "timeout": 5, **request},
                        lambda target, cancel: None, lambda target: None, None)


def test_transcript_handoff_and_binding_survive_restart(backend, conversation, tmp_path):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:codex": strategy}
    source = {"sessionId": "unverified-old-session", "messages": [
        {"role": "user", "text": "old question"}, {"role": "assistant", "text": "old answer"}]}
    first = send(backend, conversation, source)
    assert first["success"]
    assert strategy.requests[0].session_id is None
    assert [m.content for m in strategy.requests[0].history] == ["old question", "old answer"]
    backend.projected(conversation["target"], first["turn_id"])
    backend.close()
    reopened = legacy.LegacyLanguageModel(str(tmp_path))
    try:
        reopened.model.strategies = {"cli:codex": strategy}
        second = send(reopened, conversation, source, prompt="follow-up")
        assert second["success"]
        assert second["conversation_id"] == first["conversation_id"]
        assert strategy.requests[1].session_id == "fresh-session"
        assert [m.content for m in strategy.requests[1].history] == [
            "old question", "old answer", "question", "answer"]
    finally:
        reopened.close()


@pytest.mark.parametrize("outcome,status", [
    (ModelResult("failed", error_code="provider_error"), "failed"),
    (RuntimeError("provider already launched"), "interrupted"),
])
def test_failure_or_uncertainty_never_replays(backend, conversation, outcome, status):
    strategy = RecordingStrategy(outcome)
    backend.model.strategies = {"cli:codex": strategy}
    result = send(backend, conversation)
    assert result["success"] is False
    assert result["status"] == status
    assert len(strategy.requests) == 1
    assert backend.model.store.turn(result["turn_id"])["status"] == status
    if status == "interrupted":
        assert send(backend, conversation)["success"] is False
        assert len(strategy.requests) == 1


def test_cancel_callback_stops_execution_and_clears_registration(backend, conversation):
    class CancelStrategy(ModelStrategy):
        def run(self, request, emit, cancelled):
            for _ in range(500):
                if cancelled():
                    return ModelResult("cancelled")
                threading.Event().wait(0.01)
            return ModelResult("failed", error_code="cancel_not_received")

    backend.model.strategies = {"cli:codex": CancelStrategy()}
    cleared = []
    result = backend.send(conversation, {}, {"prompt": "cancel me", "timeout": 5},
                          lambda target, cancel: cancel(), cleared.append, None)
    assert result["status"] == "cancelled"
    assert result["success"] is False
    assert cleared == [conversation["target"]]


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def image_file(name="shot.png", mime_type="image/png", content=None):
    return {"filename": name, "mime_type": mime_type,
            "content": content if content is not None else base64.b64encode(PNG).decode()}


def test_images_reach_the_strategy_decoded_and_only_metadata_is_stored(backend, conversation):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:codex": strategy}
    assert send(backend, conversation, files=[image_file()])["success"] is True
    [request] = strategy.requests
    assert [(item.name, item.mime_type, item.data) for item in request.attachments] == [("shot.png", "image/png", PNG)]
    with backend.model.store.connection() as db:
        stored = " ".join(str(tuple(row)) for row in db.execute("SELECT * FROM events"))
        stored += " ".join(str(tuple(row)) for row in db.execute("SELECT * FROM turns"))
    assert "shot.png" in stored and base64.b64encode(PNG).decode() not in stored


@pytest.mark.parametrize("files,code", [
    ([image_file(mime_type="application/pdf")], "unsupported_attachments"),
    ([image_file(content="not base64!")], "invalid_attachment"),
    ([image_file(content="")], "attachment_too_large"),
    (["image.png"], "unsupported_attachments"),
])
def test_invalid_attachments_are_refused_before_any_turn(backend, conversation, files, code):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:codex": strategy}
    assert send(backend, conversation, files=files)["status"] == code
    assert not strategy.requests
    with backend.model.store.connection() as db:
        assert db.execute("SELECT count(*) FROM turns").fetchone()[0] == 0


def test_a_provider_without_image_input_refuses_before_any_turn(backend, conversation):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:grok": strategy}
    result = send(backend, {**conversation, "provider": "grok"}, files=[image_file()])
    assert result["status"] == "unsupported_attachments" and "Grok" in result["error"]
    assert not strategy.requests


def test_history_with_earlier_images_continues_with_a_note_instead_of_refusing(backend, conversation):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:codex": strategy}
    source = {"messages": [{"role": "user", "text": "see this", "attachments": [image_file("old.png")]},
                           {"role": "assistant", "text": "seen"}]}
    assert send(backend, conversation, source)["success"] is True
    history = strategy.requests[0].history
    assert history[0].content == "see this\n[Attached image(s), not re-sent: old.png]"
    assert history[1].content == "seen"


def test_existing_binding_rejects_account_change(backend, conversation):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:codex": strategy}
    assert send(backend, conversation)["success"]
    assert send(backend, {**conversation, "osUser": "bob"})["status"] == "identity_changed"
    assert len(strategy.requests) == 1


def test_configured_identity_requires_exact_account_and_workspace(tmp_path, monkeypatch, conversation):
    config = tmp_path / "identities.json"
    config.write_text(json.dumps({"identities": [{
        "id": "configured-alice", "provider": "codex", "os_user": "alice",
        "workspace": str(tmp_path), "home": str(tmp_path / "alice"),
        "config_dir": str(tmp_path / "alice/.codex"), "executable": "/usr/bin/codex"}]}))
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_CONFIG", str(config))
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_DB", str(tmp_path / "configured.sqlite3"))
    adapter = legacy.LegacyLanguageModel(str(tmp_path))
    strategy = RecordingStrategy()
    adapter.model.strategies = {"cli:codex": strategy}
    try:
        assert send(adapter, conversation)["success"]
        for change in ({"osUser": "bob"}, {"workspace": str(tmp_path / "other")},
                       {"sshConnection": "unconfigured-host"}):
            result = send(adapter, {**conversation, "target": "new-target", **change})
            assert result["status"] == "configuration_required"
        assert len(strategy.requests) == 1
        assert strategy.requests[0].identity.id == "configured-alice"
    finally:
        adapter.close()


def test_execution_required_prefers_api_agent_and_rejects_inference_only_identity(tmp_path, monkeypatch, conversation):
    base = {
        "provider": "codex",
        "model": "test-model",
        "workspace": str(tmp_path),
        "api_base_url": "http://127.0.0.1:9",
        "api_key_env": "TEST_AGENT_KEY",
    }
    config = tmp_path / "agentic-identities.json"
    config.write_text(json.dumps({"identities": [
        {**base, "id": "inference", "strategy": "api"},
        {**base, "id": "agentic", "strategy": "api_agent",
         "home": str(tmp_path / "agent-home"), "sandbox": "workspace-write"},
    ]}))
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_CONFIG", str(config))
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_DB", str(tmp_path / "configured-agentic.sqlite3"))
    adapter = legacy.LegacyLanguageModel(str(tmp_path))
    try:
        chosen = adapter.identity({**conversation, "osUser": None}, execution_required=True)
        assert chosen.id == "agentic"
        assert chosen.strategy == "api_agent"
    finally:
        adapter.close()

    config.write_text(json.dumps({"identities": [{**base, "id": "inference", "strategy": "api"}]}))
    monkeypatch.setenv("SUBJECTIVE_LANGUAGE_MODELS_DB", str(tmp_path / "configured-inference.sqlite3"))
    inference_only = legacy.LegacyLanguageModel(str(tmp_path))
    try:
        with pytest.raises(legacy.ModelError, match="tool-capable") as error:
            inference_only.identity({**conversation, "osUser": None}, execution_required=True)
        assert error.value.code == "execution_runtime_required"
    finally:
        inference_only.close()


def test_completed_turn_recovers_without_post_dispatch_binding_write(backend, conversation):
    strategy = RecordingStrategy()
    backend.model.strategies = {"cli:codex": strategy}
    with backend.model.store.connection() as db:
        db.execute("""CREATE TRIGGER refuse_pending BEFORE UPDATE OF pending_turn ON legacy_bindings
            BEGIN SELECT RAISE(ABORT, 'write failed'); END""")
    first = send(backend, conversation)
    second = send(backend, conversation)
    assert first['success'] and second['turn_id'] == first['turn_id']
    assert len(strategy.requests) == 1
    assert backend.pending(conversation['target'])['id'] == first['turn_id']


def test_remote_identity_uses_remote_user_not_service_user(tmp_path, monkeypatch):
    config = tmp_path / 'remote.json'
    config.write_text(json.dumps({'identities': [{
        'id': 'remote-connection', 'provider': 'codex', 'os_user': 'remoteoperator',
        'ssh_host': 'server-alias', 'home': '/home/remoteoperator',
        'config_dir': '/home/remoteoperator/.codex', 'workspace': '/srv/work'}]}))
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODELS_CONFIG', str(config))
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODELS_DB', str(tmp_path / 'remote.sqlite3'))
    adapter = legacy.LegacyLanguageModel(str(tmp_path))
    try:
        identity = adapter.identity({'provider': 'codex', 'osUser': 'localoperator',
                                     'workspace': '/srv/work', 'sshConnection': 'remote-connection'})
        assert identity.os_user == 'remoteoperator'
        assert identity.ssh_host == 'server-alias'
    finally:
        adapter.close()


def test_only_whole_steps_reach_the_live_progress_feed(backend, conversation):
    class Chatty(ModelStrategy):
        def run(self, request, emit, cancelled):
            emit("text_delta", {"text": "Hel"})
            emit("activity", {"text": "Running: ls"})
            emit("text_delta", {"text": "lo"})
            emit("message", {"text": "Hello"})
            return ModelResult("completed", "Hello", "session")

    backend.model.strategies = {"cli:codex": Chatty()}
    seen = []
    result = backend.send(conversation, {}, {"prompt": "q", "timeout": 5},
                          lambda target, cancel: None, lambda target: None, seen.append)
    assert result["success"] is True
    assert [item["text"] for item in seen] == ["Running: ls", "Hello"]
    assert {item["target"] for item in seen} == {conversation["target"]}


def test_file_diff_events_reach_the_progress_feed_with_their_payload(backend, conversation):
    class Editor(ModelStrategy):
        def run(self, request, emit, cancelled):
            emit("file_diff", {"path": "/repo/app.py", "status": "modified", "diff": "--- a\n+++ b\n-old\n+new"})
            emit("message", {"text": "Done."})
            return ModelResult("completed", "Done.", "session")

    backend.model.strategies = {"cli:codex": Editor()}
    seen = []
    result = backend.send(conversation, {}, {"prompt": "q", "timeout": 5},
                          lambda target, cancel: None, lambda target: None, seen.append)
    assert result["success"] is True
    diff_rows = [item for item in seen if "diff" in item]
    assert len(diff_rows) == 1
    assert diff_rows[0]["target"] == conversation["target"]
    assert diff_rows[0]["text"] == "Edited /repo/app.py"
    assert diff_rows[0]["diff"] == {"path": "/repo/app.py", "status": "modified", "diff": "--- a\n+++ b\n-old\n+new"}


def test_steps_and_diffs_are_kept_per_turn_for_a_reopened_chat(backend, conversation):
    class Editor(ModelStrategy):
        def run(self, request, emit, cancelled):
            emit("activity", {"text": "Running: git status"})
            emit("activity", {"text": "Running: git status"})
            emit("file_diff", {"path": "/repo/a.py", "status": "created", "diff": "+++ a\n+one"})
            emit("activity", {"text": "Writing /repo/b.py"})
            emit("file_diff", {"path": "/repo/b.py", "status": "modified", "diff": "--- b\n+++ b\n-x\n+y"})
            return ModelResult("completed", "Done.", "session")

    backend.model.strategies = {"cli:codex": Editor()}
    edited = send(backend, conversation)
    backend.model.strategies = {"cli:codex": RecordingStrategy()}
    plain = send(backend, conversation)
    details = backend.turn_details([edited["turn_id"], plain["turn_id"], ""])
    assert list(details) == [edited["turn_id"]]
    assert details[edited["turn_id"]]["steps"] == ["Running: git status", "Writing /repo/b.py"]
    assert [item["path"] for item in details[edited["turn_id"]]["diffs"]] == ["/repo/a.py", "/repo/b.py"]
    assert details[edited["turn_id"]]["diffs"][0] == {"path": "/repo/a.py", "status": "created", "diff": "+++ a\n+one"}
    assert backend.turn_details([]) == {}
