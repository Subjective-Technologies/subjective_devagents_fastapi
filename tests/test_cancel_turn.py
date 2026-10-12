"""Stopping an agent mid-turn.

Before this there was no way to stop a running agent at all: a turn ran until the CLI
returned or the 30-minute timeout expired, and the UI had no control to interrupt it.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402


@pytest.fixture
def service(tmp_path):
    return LlmService(root=str(tmp_path), app_home=str(tmp_path / "home"))


def test_stop_reaches_the_backend_running_the_turn(service, monkeypatch):
    chat = service.create_conversation("claude", "stoppable")
    running = threading.Event()
    cancelled = threading.Event()

    class SlowDatasource:
        def cancel(self):
            cancelled.set()
            return True

        def run(self, _request):
            running.set()
            # Stand in for the blocking CLI call; a real cancel() kills its process group.
            cancelled.wait(timeout=10)
            return {"success": False, "status": "cancelled", "error": "Stopped."}

    monkeypatch.setattr(
        service.backend, "send",
        lambda conversation, source, request, register, clear, progress: (
            register(conversation["target"], SlowDatasource().cancel)
        ) or SlowDatasource().run(request),
    )

    outcome: dict = {}
    turn = threading.Thread(target=lambda: outcome.update(service.send_message(chat["target"], "go")))
    turn.start()
    assert running.wait(timeout=10), "the turn never started"

    result = service.cancel_message(chat["target"])
    turn.join(timeout=15)

    assert result["ok"] is True
    assert result["running"] is True, "the turn was in flight and should be reported as such"
    assert result["stopped"] is True, "the backend's cancel() was never called"
    assert cancelled.is_set()
    assert outcome.get("stopped") is True
    assert outcome.get("status") == "cancelled"


def test_the_stopped_prompt_is_kept_not_rolled_back(service, monkeypatch):
    """Rolling the message back would make Stop look like the prompt was never sent."""
    chat = service.create_conversation("claude", "keeps prompt")

    class Stopped:
        def cancel(self):
            return True

        def run(self, _request):
            return {"success": False, "status": "cancelled", "error": "Stopped."}

    monkeypatch.setattr(
        service.backend, "send",
        lambda conversation, source, request, register, clear, progress: Stopped().run(request),
    )

    outcome = service.send_message(chat["target"], "remember this prompt")

    assert outcome["stopped"] is True
    kept = service.find_conversation(chat["target"], include_messages=True)["messages"]
    assert any("remember this prompt" in str(m.get("text") or "") for m in kept), kept
    assert any(m.get("stopped") for m in kept), "the prompt should be marked stopped"


def test_stopping_an_idle_chat_is_a_no_op_not_an_error(service):
    chat = service.create_conversation("claude", "idle")

    result = service.cancel_message(chat["target"])

    assert result["ok"] is True
    assert result["running"] is False
    assert result["stopped"] is False


def test_stop_missing_target_is_rejected(service):
    assert service.cancel_message("")["ok"] is False


def test_a_stop_racing_the_spawn_is_not_lost(service, monkeypatch):
    """Stop can land after the request is accepted but before the CLI process exists.
    Latching it means the turn is cancelled as soon as it registers, rather than running
    on unstoppable."""
    chat = service.create_conversation("claude", "racy")
    target = chat["target"]
    service.busy.add(target)
    service.cancel_message(target)  # arrives first

    cancelled = threading.Event()

    class Late:
        def cancel(self):
            cancelled.set()
            return True

    service._register_canceller(target, Late().cancel)

    assert cancelled.is_set(), "a cancel that arrived before the process was dropped"
    service.busy.discard(target)


def test_stop_while_the_send_is_still_being_prepared_never_starts_the_turn(service, monkeypatch):
    """The browser posts, DevAgents prepares the workflow, and only then is the chat busy.
    A Stop in that gap used to be dropped as "not running", and the turn ran on unwatched."""
    chat = service.create_conversation("claude", "early stop")
    started = []
    monkeypatch.setattr(
        service.backend, "send",
        lambda conversation, source, request, register, clear, progress: started.append(1) or {
            "success": True, "status": "completed", "response": "answer", "session_id": ""},
    )
    service.accept_send(chat["target"])
    result = service.cancel_message(chat["target"])
    assert result["running"] is True
    outcome = service.send_message(chat["target"], "go")
    service.settle_send(chat["target"])
    assert outcome.get("stopped") is True and outcome.get("status") == "cancelled"
    assert not started, "the stopped turn must not reach the model"


def test_a_stop_latched_for_one_send_does_not_stop_the_next(service, monkeypatch):
    chat = service.create_conversation("claude", "next send runs")
    started = []
    monkeypatch.setattr(
        service.backend, "send",
        lambda conversation, source, request, register, clear, progress: started.append(1) or {
            "success": True, "status": "completed", "response": "answer", "session_id": ""},
    )
    service.accept_send(chat["target"])
    service.cancel_message(chat["target"])
    service.settle_send(chat["target"])  # e.g. the workflow could not be loaded
    assert not service.was_cancelled(chat["target"])
    # an idle chat: Stop is a no-op and latches nothing
    assert service.cancel_message(chat["target"])["running"] is False
    service.send_message(chat["target"], "go")
    assert started == [1]
