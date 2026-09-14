"""Assigned names must survive imported copies of the same provider session."""
import json
from pathlib import Path

import pytest

from subjective_devagents_fastapi.llm_service import LlmService


@pytest.fixture
def named_session(tmp_path):
    service = LlmService(root=str(tmp_path), app_home=str(tmp_path / "home"))
    local = service.create_conversation("claude", "manager")
    local["sessionId"] = "146d58f3-8d29-4e85-bdb7-a582fb4fd5c9"
    # Existing chats were named at creation before labelOverrides recorded names.
    service.label_overrides.clear()
    service._save_state()
    context = Path(service.context_dir)
    context.mkdir(parents=True, exist_ok=True)
    export = context / "2026_09_14_14_05_17-Localchats-test-context.json"
    export.write_text(json.dumps({
        "type": "localchat", "product": "claude", "thread_id": local["sessionId"],
        "title": "see agents protocol, you are the manager agent",
        "source_file": f"/home/example/.claude/projects/project/{local['sessionId']}.jsonl",
        "messages": [{"role": "user", "text": "Original prompt", "timestamp": 1700000000}],
    }))
    return service, local, export


def test_existing_assigned_name_wins_list_search_detail_and_restart(named_session):
    service, local, export = named_session
    for instance in (service, LlmService(root=service.root, app_home=service.app_home)):
        history = instance._history_conversations()[0]
        listed = instance.list_conversations()["conversations"]
        assert len(listed) == 1
        assert listed[0]["target"] == local["target"]
        assert listed[0]["label"] == "manager"
        assert instance.list_conversations(query="manager")["total"] == 1
        assert instance.list_conversations(query="see agents protocol")["total"] == 0
        for target in (local["target"], history["target"]):
            for hydrated in (False, True):
                assert instance.find_conversation(target, include_messages=hydrated)["label"] == "manager"
        # Live reload can attach sourceFile/historyExportPath to originally local chats.
        instance.local[0]["sourceFile"] = history["sourceFile"]
        instance.local[0]["historyExportPath"] = str(export)
        assert instance.find_conversation(local["target"], include_messages=True)["label"] == "manager"


@pytest.mark.parametrize("rename_alias", ["local", "history"])
def test_rename_updates_session_aliases_and_survives_reimport(named_session, rename_alias):
    service, local, export = named_session
    history = service._history_conversations()[0]
    target = local["target"] if rename_alias == "local" else history["target"]
    assert service.rename_conversation(target, "Architecture team")["ok"]
    payload = json.loads(export.read_text())
    payload["title"] = "Freshly imported first prompt"
    export.write_text(json.dumps(payload))
    restarted = LlmService(root=service.root, app_home=service.app_home)
    assert restarted.list_conversations(query="Architecture team")["conversations"][0]["label"] == "Architecture team"
    for alias in (local["target"], history["target"]):
        assert restarted.find_conversation(alias, include_messages=True)["label"] == "Architecture team"


def test_history_only_rename_is_searchable(named_session):
    service, _, _ = named_session
    service.local.clear()
    history = service._history_conversations()[0]
    service.rename_conversation(history["target"], "Release plan")
    assert service.list_conversations(query="Release plan")["total"] == 1
    assert service.list_conversations(query="see agents protocol")["total"] == 0


@pytest.mark.parametrize("name", ["Research <team>", "Claude conversation", "New Claude chat", "x" * 150])
def test_creation_records_explicit_name_as_override(tmp_path, name):
    service = LlmService(root=str(tmp_path), app_home=str(tmp_path / "home"))
    chat = service.create_conversation("claude", name)
    restarted = LlmService(root=service.root, app_home=service.app_home)
    assert restarted.find_conversation(chat["target"])["label"] == name


def test_unnamed_local_chat_can_use_imported_title(named_session):
    service, local, _ = named_session
    local["label"] = "New Claude chat"
    assert service.list_conversations()["conversations"][0]["label"].startswith("see agents protocol")
