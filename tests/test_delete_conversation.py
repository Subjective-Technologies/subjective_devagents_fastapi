"""Deleting a chat.

The browser has called /api/llm/delete-conversation since the Node server this
service replaced. The port never implemented it, so every delete answered 405 and
the UI could only say "Could not delete this conversation." These cover the route
existing, and the several things a delete has to clean up besides the row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))
for dependency in ("subjective-abstract-data-source-package",
                   "brainboost_configuration_package",
                   "brainboost_data_source_logger_package"):
    sys.path.insert(0, str(SERVICE_ROOT / "libs" / "dependencies" / dependency))

from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return LlmService(root=str(tmp_path / "workspace"), app_home=str(tmp_path / "home"))


def test_a_chat_created_here_can_be_deleted(service):
    conversation = service.create_conversation(provider="claude", name="Throwaway")
    target = conversation["target"]
    assert service.find_conversation(target) is not None

    outcome = service.delete_conversation(target)
    assert outcome["ok"] is True
    assert outcome["target"] == target
    assert service.find_conversation(target) is None


def test_it_survives_a_reload(service, tmp_path):
    """The row must not come back the next time the service starts."""
    conversation = service.create_conversation(provider="claude", name="Throwaway")
    service.delete_conversation(conversation["target"])
    reloaded = LlmService(root=str(tmp_path / "workspace"), app_home=str(tmp_path / "home"))
    assert reloaded.find_conversation(conversation["target"]) is None


def test_the_renamed_label_goes_with_it(service):
    """A label override outliving its chat would reattach to a recycled target."""
    conversation = service.create_conversation(provider="claude", name="Named")
    target = conversation["target"]
    service.rename_conversation(target, "Renamed")
    assert target in service.label_overrides
    service.delete_conversation(target)
    assert target not in service.label_overrides


def test_it_leaves_the_active_list(service):
    conversation = service.create_conversation(provider="claude", name="Active")
    target = conversation["target"]
    assert target in service.active_targets
    service.delete_conversation(target)
    assert target not in service.active_targets


def test_an_ssh_chat_takes_its_shell_with_it(service, monkeypatch):
    """An SSH chat holds a shell on another machine for as long as it lives."""
    from subjective_devagents_fastapi import ssh_chat

    closed = []
    monkeypatch.setattr(ssh_chat.sessions, "close", closed.append)
    conversation = service.create_conversation(
        provider="ssh", name="Box", workspace="deploy@host",
    )
    service.delete_conversation(conversation["target"])
    assert closed == [conversation["target"]]


def test_a_running_turn_is_stopped_first(service, monkeypatch):
    """Otherwise the turn writes to a conversation that no longer exists, and its
    CLI keeps running unwatched."""
    conversation = service.create_conversation(provider="claude", name="Busy")
    target = conversation["target"]
    service.busy.add(target)
    cancelled = []
    monkeypatch.setattr(service, "cancel_message", lambda key: cancelled.append(key) or {"ok": True})

    service.delete_conversation(target)
    assert cancelled == [target]
    assert target not in service.busy


def test_deleting_something_that_is_not_there_says_so(service):
    outcome = service.delete_conversation("llm:claude:local:nope")
    assert outcome["ok"] is False
    assert outcome["status"] == "not_found"


def test_a_missing_target_is_refused(service):
    outcome = service.delete_conversation("")
    assert outcome["ok"] is False
    assert outcome["status"] == "invalid_request"


def test_a_cli_transcript_on_disk_is_not_ours_to_delete(service, monkeypatch):
    """Those rows are files Claude or Codex wrote. Deleting one would delete the
    agent's own history behind its back, which is not what the button is asking."""
    history_row = {"target": "llm:claude:history:abc", "provider": "claude",
                   "label": "From disk", "origin": "history"}
    monkeypatch.setattr(service, "find_conversation", lambda target, **kwargs: history_row)
    outcome = service.delete_conversation("llm:claude:history:abc")
    assert outcome["ok"] is False
    assert outcome["status"] == "read_only"
    assert "CLI" in outcome["error"]


def test_a_chat_without_a_workspace_works_in_the_configured_repository(service, monkeypatch, tmp_path):
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_LLM_WORKSPACE", str(tmp_path / "repo"))
    conversation = service.create_conversation(provider="claude", name="Agent")
    assert conversation["workspace"] == str(tmp_path / "repo")

    explicit = service.create_conversation(provider="claude", name="Agent", workspace=str(tmp_path / "other"))
    assert explicit["workspace"] == str(tmp_path / "other")


def test_without_the_setting_a_chat_keeps_the_service_root(service, monkeypatch, tmp_path):
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_LLM_WORKSPACE", raising=False)
    conversation = service.create_conversation(provider="claude", name="Agent")
    assert conversation["workspace"] == str(tmp_path / "workspace")
