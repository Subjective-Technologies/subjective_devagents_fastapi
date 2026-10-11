"""A reopened chat shows each agent answer's steps and the files it changed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subjective_devagents_fastapi import app as app_module  # noqa: E402

DIFF = {"path": "/repo/a.py", "status": "modified", "diff": "--- a\n+++ a\n-x\n+y"}


def _instance(details, asked):
    def turn_details(turn_ids):
        asked.append(list(turn_ids))
        return details

    return SimpleNamespace(llm_service=SimpleNamespace(backend=SimpleNamespace(turn_details=turn_details)))


def test_answers_carry_the_diffs_of_their_turn():
    asked = []
    messages = [
        {"role": "user", "text": "edit", "turnId": "t1"},
        {"role": "assistant", "text": "done", "turnId": "t1"},
        {"role": "user", "text": "hi", "turnId": "t2"},
        {"role": "assistant", "text": "hello", "turnId": "t2"},
        {"role": "assistant", "text": "imported"},
    ]
    conversation = {"target": "x", "messages": messages}
    details = {"t1": {"steps": ["Running: ls", "Editing a.py"], "diffs": [DIFF]}, "t2": {"steps": ["Reading x"], "diffs": []}}
    result = app_module._with_turn_details(_instance(details, asked), conversation)
    assert asked == [["t1", "t2"]]
    assert result["messages"][1]["diffs"] == [DIFF]
    assert result["messages"][1]["steps"] == ["Running: ls", "Editing a.py"]
    assert result["messages"][3]["steps"] == ["Reading x"] and "diffs" not in result["messages"][3]
    assert "diffs" not in result["messages"][0] and "steps" not in result["messages"][0]
    # the stored conversation is not changed
    assert "diffs" not in messages[1]


def test_a_failing_lookup_still_returns_the_conversation():
    def broken(_ids):
        raise RuntimeError("db locked")

    instance = SimpleNamespace(llm_service=SimpleNamespace(backend=SimpleNamespace(turn_details=broken)))
    conversation = {"messages": [{"role": "assistant", "text": "a", "turnId": "t1"}]}
    assert app_module._with_turn_details(instance, conversation) is conversation
