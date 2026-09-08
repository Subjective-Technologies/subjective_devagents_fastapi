"""Tests for agent-status parsing.

The status stream leaked raw structures to the UI — a status line reading
`{'role': 'assistant', 'content': [{'type': 'text', 'text': '+'}]}` instead of the text a
person should see. These pin that every shape a datasource emits is flattened to readable
text, and that empty deltas produce nothing rather than a blank status.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi.llm_service import LlmService, _extract_readable_text  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("Checking the logs for errors", "Checking the logs for errors"),
    # The bug: a Python-repr'd message dict smuggled into the text field.
    ("{'role': 'assistant', 'content': [{'type': 'text', 'text': 'Reading the file'}]}", "Reading the file"),
    # The same, as real JSON.
    ('{"role": "assistant", "content": [{"type": "text", "text": "Writing a reply"}]}', "Writing a reply"),
    # A tool call reads as a verb, not its arguments.
    ("{'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'ls -la'}}", "Using Bash"),
    # Nothing worth showing.
    ("{'role': 'assistant', 'content': [{'type': 'text', 'text': ''}]}", ""),
    ("", ""),
    ("   ", ""),
])
def test_readable_text_extraction(raw, expected):
    assert _extract_readable_text(raw) == expected


def test_a_structure_that_is_not_a_message_is_left_alone():
    # Prose that merely starts with punctuation must not be mistaken for a structure.
    assert _extract_readable_text("(thinking about the plan)") == "(thinking about the plan)"


def test_consume_progress_emits_readable_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    service = LlmService(root=str(tmp_path), app_home=str(tmp_path / "home"))
    seen = []
    service.on_progress = lambda event: seen.append(event["text"])

    chunk = (
        'SUBJECTIVE_PROGRESS {"subjective_progress": true, "text": "Checking the logs"}\n'
        'SUBJECTIVE_PROGRESS {"subjective_progress": true, '
        '"text": "{\'role\': \'assistant\', \'content\': [{\'type\': \'text\', \'text\': \'Found it\'}]}"}\n'
        'SUBJECTIVE_PROGRESS {"subjective_progress": true, "text": ""}\n'
    )
    service._consume_progress(chunk, {"target": "llm:codex:x", "session_id": "s"})

    # Both readable lines come through as text; the empty delta is dropped, not blanked.
    assert seen == ["Checking the logs", "Found it"]


def test_consume_progress_ignores_lines_without_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    service = LlmService(root=str(tmp_path), app_home=str(tmp_path / "home"))
    seen = []
    service.on_progress = lambda event: seen.append(event["text"])
    service._consume_progress("ordinary stderr line\nanother line\n", {"target": "t"})
    assert seen == []
