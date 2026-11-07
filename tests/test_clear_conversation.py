"""Clear means saved history disappears while the conversation remains usable."""
import json
import threading

import pytest

from test_delete_conversation import service
from test_api import client
from subjective_devagents_fastapi.llm_service import LlmService


def seed(service):
    chat = service.create_conversation(provider="codex", name="Keep this pane")
    service.inject_test_assistant_message(chat["target"], "History to clear")
    return chat


def test_clear_survives_reload_and_restart_and_preserves_other_chats(service):
    chat, other = seed(service), seed(service)
    result = service.clear_conversation_messages(chat["target"])
    assert result["ok"]
    assert service.reload_conversation(chat["target"])["messages"] == []
    restarted = LlmService(root=service.root, app_home=service.app_home)
    try:
        assert restarted.reload_conversation(chat["target"])["messages"] == []
        assert restarted.find_conversation(chat["target"])["label"] == "Keep this pane"
        assert restarted.find_conversation(other["target"], include_messages=True)["messages"]
        restarted.inject_test_assistant_message(chat["target"], "New history")
        assert len(restarted.reload_conversation(chat["target"])["messages"]) == 1
    finally:
        restarted.backend.close()


def test_clear_removes_recoverable_turns_and_native_session(service):
    chat = seed(service)
    store = service.backend.model.store
    identity = service.backend.identity(chat)
    from subjective_devagents_fastapi.language_models.contracts import Message
    bound = store.create(identity, (Message("user", "old"), Message("assistant", "old answer")))
    with store.connection() as db:
        db.execute("INSERT INTO legacy_bindings(target,conversation_id) VALUES (?,?)", (chat["target"], bound["id"]))
        db.execute("UPDATE conversations SET session_id='old-native-session' WHERE id=?", (bound["id"],))
        db.execute("UPDATE turns SET request_id='recover-me' WHERE conversation_id=?", (bound["id"],))
    assert service.backend.pending(chat["target"])
    assert service.clear_conversation_messages(chat["target"])["ok"]
    assert service.backend.pending(chat["target"]) is None
    assert store.conversation(bound["id"])["session_id"] is None
    assert service.reload_conversation(chat["target"])["messages"] == []


def test_clear_refuses_busy_and_imported_chats_without_changing_history(service):
    chat = seed(service)
    service.busy.add(chat["target"])
    assert service.clear_conversation_messages(chat["target"])["status"] == "busy"
    service.busy.clear()
    chat["historyExportPath"] = "/provider/transcript.json"
    assert service.clear_conversation_messages(chat["target"])["status"] == "read_only"
    assert chat["messages"]
    assert service.clear_conversation_messages("")["status"] == "invalid_request"
    assert service.clear_conversation_messages("missing")["status"] == "not_found"


def test_failed_save_keeps_history(service, monkeypatch):
    chat = seed(service)
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(service, "_save_state", fail)
    with pytest.raises(OSError):
        service.clear_conversation_messages(chat["target"])
    assert chat["messages"]
    assert json.loads(open(service.state_path).read())["conversations"][0]["messages"]


def test_clear_refuses_concurrent_send_or_reload(service):
    chat = seed(service)
    outcome = []
    with service._conversation_lock(chat["target"]):
        worker = threading.Thread(target=lambda: outcome.append(service.clear_conversation_messages(chat["target"])))
        worker.start()
        worker.join(timeout=2)
    assert outcome[0]["status"] == "busy"
    assert chat["messages"]


def test_clear_api_persists_and_broadcasts(client, monkeypatch):
    from subjective_devagents_fastapi.app import core
    chat = seed(core().llm_service)
    events = []
    monkeypatch.setattr(core(), "broadcast", lambda kind, payload: events.append({"type": kind, **payload}))
    response = client.post("/api/llm/clear-conversation", json={"target": chat["target"]}).json()
    assert response["ok"], response
    assert response["conversation"]["messages"] == []
    assert client.get("/api/llm/conversation", params={"target": chat["target"]}).json()["conversation"]["messages"] == []
    assert any(event["type"] == "llm-conversation" and event["conversation"]["messages"] == [] for event in events)


def test_cleared_chat_replaces_d1_replica_messages(service):
    from test_d1_sync import _db
    from subjective_devagents_fastapi.d1_client import SqliteExecutor
    from subjective_devagents_fastapi.d1_sync import sync_conversations
    chat = seed(service)
    db = _db()
    executor = SqliteExecutor(db)
    assert sync_conversations(executor, [chat])["failed"] == 0
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    result = service.clear_conversation_messages(chat["target"])
    assert sync_conversations(executor, [result["conversation"]])["failed"] == 0
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
    db.close()
