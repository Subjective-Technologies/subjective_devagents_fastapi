"""Tests for backend-computed prompt options.

The backend is now the authoritative source of the choices an assistant is offering: it
returns structured options (label + description + the exact reply to send) so the UI does
not have to guess from prose. These pin the detection and the shape the UI consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import prompt_options  # noqa: E402


def test_a_statement_offers_no_options():
    assert prompt_options.extract_prompt_options("The build passed. Everything is green.")["asks_question"] is False


def test_a_plain_question_yields_yes_no_and_other():
    result = prompt_options.extract_prompt_options("I can do that. Shall I proceed?")
    assert result["asks_question"] is True
    assert result["kind"] == "yes_no"
    assert [o["id"] for o in result["options"]] == ["yes", "no"]
    assert result["options"][0] == {"id": "yes", "label": "Yes", "description": "Proceed.", "reply": "Yes, proceed."}
    # Every set lets the user answer in their own words.
    assert result["allows_other"] is True
    assert result["other_hint"]


def test_keep_or_revert_is_detected_with_descriptions():
    result = prompt_options.extract_prompt_options("I rewrote the parser. Keep the code or revert the changes?")
    assert result["kind"] == "keep_revert"
    assert {o["id"] for o in result["options"]} == {"keep", "revert"}
    assert all(o["description"] for o in result["options"]), "every option needs a description"


def test_a_permission_request_becomes_allow_deny():
    result = prompt_options.extract_prompt_options("The task needs network access. Do you approve?")
    assert result["kind"] == "allow_deny"
    assert result["options"][0]["reply"] == "Allow it and continue."


def test_a_run_request_becomes_run_or_not():
    result = prompt_options.extract_prompt_options("The migration is ready. Should I run it?")
    assert result["kind"] == "run"
    assert {o["id"] for o in result["options"]} == {"run", "do-not-run"}


def test_enumerated_choices_carry_their_own_descriptions_and_replies():
    text = (
        "Here are the options:\n"
        "A. Fast but risky\n"
        "B. Safe but slow\n"
        "C. A balance of both\n\n"
        "Which do you prefer — A, B, or C?"
    )
    result = prompt_options.extract_prompt_options(text)
    assert result["kind"] == "enumerated"
    assert [o["reply"] for o in result["options"]] == ["A", "B", "C"]
    assert [o["description"] for o in result["options"]] == ["Fast but risky", "Safe but slow", "A balance of both"]


def test_a_question_inside_a_code_block_is_ignored():
    """A code sample that happens to contain a question must not raise buttons."""
    text = "Here is a snippet:\n```\nif ok:\n    print('proceed?')\n```\nThat is the fix."
    assert prompt_options.extract_prompt_options(text)["asks_question"] is False


def test_annotate_only_touches_a_questioning_assistant_message():
    # A user message is never annotated.
    user = {"role": "user", "text": "Shall I proceed?"}
    assert prompt_options.annotate_message(user) is user
    # An assistant statement is returned unchanged (no promptOptions key).
    statement = {"role": "assistant", "text": "Done."}
    assert "promptOptions" not in prompt_options.annotate_message(statement)
    # An assistant question gets a promptOptions field, on a copy (original untouched).
    question = {"role": "assistant", "text": "Shall I proceed?"}
    annotated = prompt_options.annotate_message(question)
    assert "promptOptions" not in question
    assert annotated["promptOptions"]["kind"] == "yes_no"


# --- the options actually reach the client through the API ------------------------------------


def _api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True)
    (tmp_path / "client" / "src").mkdir(parents=True)
    (tmp_path / "client" / "web" / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", raising=False)
    from fastapi.testclient import TestClient
    from subjective_devagents_fastapi.app import app, state
    client = TestClient(app)
    return client, state


def test_conversation_endpoint_returns_prompt_options_on_the_last_assistant_message(tmp_path, monkeypatch):
    """The whole point: the UI reads promptOptions off the served conversation rather than
    parsing prose itself."""
    client, state = _api_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/llm/create-conversation", json={"provider": "claude", "name": "Options"}).json()
        target = created["conversation"]["target"]
        # Seed a stored assistant question directly, then hydrate through the API.
        conv = next(c for c in state.core.llm_service.local if c["target"] == target)
        conv["messages"] = [
            {"role": "user", "text": "Refactor it", "time": 1},
            {"role": "assistant", "text": "I rewrote the parser. Shall I proceed with the tests?", "time": 2},
        ]
        body = client.get("/api/llm/conversation", params={"target": target}).json()

        assert body["ok"] is True
        messages = body["conversation"]["messages"]
        assert "promptOptions" not in messages[0], "a user message must not carry options"
        options = messages[-1]["promptOptions"]
        assert options["kind"] == "yes_no"
        assert [o["reply"] for o in options["options"]] == ["Yes, proceed.", "No, do not proceed."]
        assert options["allows_other"] is True


def test_a_non_question_reply_carries_no_options_field(tmp_path, monkeypatch):
    client, state = _api_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/llm/create-conversation", json={"provider": "codex", "name": "NoOptions"}).json()
        target = created["conversation"]["target"]
        conv = next(c for c in state.core.llm_service.local if c["target"] == target)
        conv["messages"] = [{"role": "assistant", "text": "All tests pass. Done.", "time": 1}]
        body = client.get("/api/llm/conversation", params={"target": target}).json()
        assert "promptOptions" not in body["conversation"]["messages"][-1]
