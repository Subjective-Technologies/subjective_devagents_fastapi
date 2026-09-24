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


# --- menus written as a list under the question ------------------------------------------
#
# What agents actually produce when asked for a multiple-choice prompt: the choices are
# words on bullet lines, not "A."/"B." labels, and a message may ask several questions at
# once. Both shapes rendered as no buttons at all before this was detected.


MULTI_QUESTION_REPLY = """1. Should we proceed?

- Yes
- No
- Need more details

2. Which response style?

- Brief
- Detailed
- Step by step
- Examples

3. What matters most?

- Speed
- Accuracy
- Simplicity

3 questions ready. Each has 3-5 options. Use free-text input for "Other."
"""


def test_a_bulleted_list_under_a_question_becomes_that_questions_menu():
    result = prompt_options.extract_prompt_options(
        "Which response style do you prefer?\n\n- Brief\n- Detailed\n- Step by step\n"
    )
    assert result["kind"] == "menu"
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["question"] == "Which response style do you prefer?"
    assert [o["label"] for o in group["options"]] == ["Brief", "Detailed", "Step by step"]
    # The agent's own words are the answer: it wrote them, so it understands them.
    assert [o["reply"] for o in group["options"]] == ["Brief", "Detailed", "Step by step"]


def test_a_short_aside_after_the_question_mark_does_not_hide_the_menu():
    """Agents append "Optional: add context" to the question line. The question is still a
    question, and its list is still its menu."""
    result = prompt_options.extract_prompt_options(
        "Which response style do you prefer? Optional: add context using free-text input.\n"
        "\n- Brief\n- Detailed\n- Step by step\n"
    )
    assert result["kind"] == "menu"
    assert result["groups"][0]["question"] == "Which response style do you prefer?"


def test_every_question_in_one_reply_gets_its_own_group():
    result = prompt_options.extract_prompt_options(MULTI_QUESTION_REPLY)
    assert result["asks_question"] is True
    assert [g["question"] for g in result["groups"]] == [
        "Should we proceed?", "Which response style?", "What matters most?",
    ]
    assert [len(g["options"]) for g in result["groups"]] == [3, 4, 3]
    assert [g["id"] for g in result["groups"]] == ["q1", "q2", "q3"]


def test_a_multi_question_reply_is_detected_although_it_ends_in_a_statement():
    """The closing line is "3 questions ready.", not a question. Requiring a trailing "?"
    is what left this message with no buttons at all."""
    assert MULTI_QUESTION_REPLY.strip().endswith('Use free-text input for "Other."')
    assert prompt_options.extract_prompt_options(MULTI_QUESTION_REPLY)["asks_question"] is True


def test_the_first_group_is_repeated_at_the_top_level_for_older_clients():
    result = prompt_options.extract_prompt_options(MULTI_QUESTION_REPLY)
    assert result["options"] == result["groups"][0]["options"]
    assert result["kind"] == result["groups"][0]["kind"]


def test_every_result_offers_a_free_text_box():
    for text in (MULTI_QUESTION_REPLY, "I can do that. Shall I proceed?"):
        free_text = prompt_options.extract_prompt_options(text)["free_text"]
        assert free_text["enabled"] is True
        assert free_text["label"] and free_text["placeholder"]


def test_an_option_line_splits_into_label_and_description():
    result = prompt_options.extract_prompt_options(
        "Which deploy target?\n\n- staging - safe, resets nightly\n- production: serves real users\n"
    )
    options = result["groups"][0]["options"]
    assert [o["label"] for o in options] == ["staging", "production"]
    assert options[0]["description"] == "safe, resets nightly"
    assert options[0]["reply"] == "staging", "the reply is the choice, not the gloss"


def test_a_list_of_findings_followed_by_a_question_is_not_a_menu():
    """Regression guard: the list has to sit *under* the question to be its menu. A report
    that happens to end with a yes/no question is a yes/no question."""
    result = prompt_options.extract_prompt_options(
        "I found three problems:\n"
        "1. The parser drops CRLF\n"
        "2. The cache never expires\n"
        "3. The retry loop is unbounded\n\n"
        "Want me to fix them?"
    )
    assert result["kind"] == "yes_no"
    assert len(result["groups"]) == 1


def test_a_list_of_questions_is_not_mistaken_for_options():
    """Items that are themselves questions are not choices."""
    result = prompt_options.extract_prompt_options(
        "A few things are unclear. Can you clarify?\n\n- Which branch?\n- Which environment?\n"
    )
    assert result["kind"] != "menu"


def test_a_lettered_list_before_its_question_still_answers_with_the_letter():
    """The older shape keeps its older semantics: reply "A", not "A. Fast but risky"."""
    result = prompt_options.extract_prompt_options(
        "Here are the options:\nA. Fast but risky\nB. Safe but slow\n\nWhich do you prefer - A or B?"
    )
    assert result["kind"] == "enumerated"
    assert [o["reply"] for o in result["groups"][0]["options"]] == ["A", "B"]


# --- the same feature over an SSH pty ------------------------------------------------------
#
# An SSH chat is a shell. It reads what a program expects — the number a `select` printed,
# or y/n — so the reply cannot be the English sentence a model would want.


SHELL_MENU = """Which distribution?
1) Debian
2) Ubuntu
3) Fedora
"""


def test_an_ssh_menu_answers_with_the_number_the_shell_printed():
    result = prompt_options.extract_prompt_options(SHELL_MENU, provider="ssh")
    options = result["groups"][0]["options"]
    assert [o["label"] for o in options] == ["Debian", "Ubuntu", "Fedora"]
    assert [o["reply"] for o in options] == ["1", "2", "3"], "a shell select reads the number"


def test_the_same_menu_from_a_model_answers_in_words():
    options = prompt_options.extract_prompt_options(SHELL_MENU, provider="claude")["groups"][0]["options"]
    assert [o["reply"] for o in options] == ["Debian", "Ubuntu", "Fedora"]


def test_an_ssh_yes_no_prompt_answers_with_y_or_n():
    result = prompt_options.extract_prompt_options("Do you want to continue? [Y/n]", provider="ssh")
    assert [o["reply"] for o in result["options"]] == ["y", "n"]


def test_a_model_yes_no_prompt_still_answers_in_words():
    result = prompt_options.extract_prompt_options("Do you want to continue?", provider="codex")
    assert [o["reply"] for o in result["options"]] == ["Yes, proceed.", "No, do not proceed."]


@pytest.mark.parametrize("provider", ["claude", "codex", "grok", "cursor", "ssh"])
def test_every_datasource_gets_options_on_its_assistant_messages(provider):
    """The annotation happens in one place for every provider — local, or with the agent
    running on another machine over SSH — so no datasource is left without buttons."""
    message = {"role": "assistant", "text": "Which response style?\n\n- Brief\n- Detailed\n"}
    annotated = prompt_options.annotate_message(message, provider)
    assert annotated["promptOptions"]["kind"] == "menu"
    assert len(annotated["promptOptions"]["groups"][0]["options"]) == 2
