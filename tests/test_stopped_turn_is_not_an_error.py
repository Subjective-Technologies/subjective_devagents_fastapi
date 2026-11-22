"""Operator Stop is not a failure (production chat llm:codex:local:e2227e1b, 2026-11-21 11:24:59 -0300).

`POST /api/llm/cancel-message` returned 200, the turn stopped cleanly after 42.9 s and
`LLM turn stopped by operator` was logged at info — and then the chat layer logged
`[error] LLM answer failed` / `[error] LLM send background finished` and broadcast
`Codex error (cancelled): Stopped.` to the conversation. The cancel worked; only the
reporting was wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi.chat_core import ChatCore  # noqa: E402


@pytest.fixture
def core(tmp_path):
    records: list[tuple[str, str, dict]] = []
    events: list[dict] = []
    instance = ChatCore(
        root=str(tmp_path),
        app_home=str(tmp_path / "home"),
        log=lambda level, message, details=None: records.append((level, message, details or {})),
        emit=events.append,
    )
    instance.logged, instance.events = records, events
    return instance


def answers(core, response):
    """Run answer_llm_message against a backend that returns exactly `response`."""
    chat = core.llm_service.create_conversation("codex", "stop reporting")
    core.llm_service.send_message = lambda target, text, files: response
    return chat, core.answer_llm_message(chat["target"], "do the work", "tester", False)


def system_lines(core):
    return [str(event.get("text") or "") for event in core.events if event.get("type") == "system"]


def test_a_stopped_turn_is_reported_at_info_with_the_backend_stop_text(core):
    """`status: cancelled` must not reach the error log and must not be shown as an error."""
    stop = {"ok": False, "stopped": True, "status": "cancelled", "error": "Stopped.",
            "userMessage": {"role": "user", "text": "do the work"}}
    _, response = answers(core, stop)

    assert response is stop
    assert not [row for row in core.logged if row[0] == "error"], core.logged
    assert [row[1] for row in core.logged if row[0] == "info"].count("LLM answer not delivered") == 1
    assert system_lines(core) == ["Stopped."]
    assert not any("error (" in line for line in system_lines(core))


def test_a_failed_turn_is_still_an_error_with_its_label(core):
    _, response = answers(core, {"ok": False, "status": "failed", "error": "provider_unreachable"})

    assert [row[1] for row in core.logged if row[0] == "error"] == ["LLM answer failed"]
    assert system_lines(core) == ["Codex error (failed): provider_unreachable"]


def test_a_context_handoff_or_restart_is_not_shown_as_an_error(core):
    """The same rule for the other turns that end without an answer but did not fail."""
    from subjective_devagents_fastapi.llm_service import CONTINUED_TURN_CODES

    for code, text in CONTINUED_TURN_CODES.items():
        core.logged.clear()
        core.events.clear()
        _, response = answers(core, {"ok": False, "continued": True, "status": code, "error": text})
        assert not [row for row in core.logged if row[0] == "error"], (code, core.logged)
        assert system_lines(core) == [text]


def test_the_background_send_wrapper_classifies_a_stop_the_same_way(core):
    """app.run_send logs the same outcome; both strings in the journal came from a stop."""
    assert ChatCore.llm_turn_not_failed({"ok": False, "status": "cancelled", "error": "Stopped."})
    assert ChatCore.llm_turn_not_failed({"ok": False, "stopped": True, "status": "cancelled"})
    assert ChatCore.llm_turn_not_failed({"ok": False, "continued": True, "status": "context_handoff"})
    assert not ChatCore.llm_turn_not_failed({"ok": False, "status": "failed", "error": "boom"})


def test_the_stop_text_is_never_prefixed_with_the_provider_label(core):
    stop = {"ok": False, "stopped": True, "status": "cancelled", "error": "Stopped."}
    assert ChatCore.llm_error_text("Codex", stop) == "Stopped."
    assert ChatCore.llm_error_text("Codex", {"ok": False, "status": "timeout", "error": "took too long"}).startswith(
        "Codex error (timeout): took too long"
    )
