"""Agent definitions built from the company's published workflow (agent_workflow.py)."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import agent_workflow as aw  # noqa: E402

COMPANY = "company_54961bc50fe94bbab712"
STAMPED = (
    "# coder_agent\n\n<!-- workflow-header:begin (generated) -->\nWorkflow version: `20261008225854`\n"
    "<!-- workflow-header:end -->\n\n## Mission\n\nShip code.\n"
)


def _export(version="20261009150839", role=STAMPED):
    return {
        "version": {"version": version},
        "files": [
            {"path": "AGENTS.md", "name": "AGENTS", "category": "contract", "content": "# Contract\nBe exact."},
            {"path": "agents/roles/coder_agent.md", "name": "coder_agent", "category": "agent_role", "content": role},
            {"path": "skills/browser-verification/SKILL.md", "name": "browser-verification", "category": "skill", "content": "# Browser"},
            {"path": "../escape.md", "name": "escape", "category": "other", "content": "nope"},
        ],
    }


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch, tmp_path):
    aw._cache.clear()
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_WORKFLOW_ROOT", str(tmp_path / "workflows"))
    yield
    aw._cache.clear()


def _serve(monkeypatch, export, seen=None):
    def opener(request, timeout=0):
        if seen is not None:
            seen.append(request)
        return _Response(json.dumps(export).encode())

    monkeypatch.setattr(aw.urllib.request, "urlopen", opener)
    monkeypatch.setattr(aw, "fetch_export", lambda company, auth, opener=opener: _real_fetch(company, auth, opener=opener))


_real_fetch = aw.fetch_export

AGENT = {"role_file": "agents/roles/coder_agent.md", "company_id": COMPANY, "display_name": "Coder Agent", "handle": "coder_agent"}


def test_the_first_turn_carries_the_definition_with_the_database_version(monkeypatch, tmp_path):
    seen = []
    _serve(monkeypatch, _export(), seen)
    state = {}
    prompt = aw.prepare_turn(state, "llm:t1", AGENT, "Bearer tok", "hello", str(tmp_path))
    assert prompt.startswith(aw.AGENT_DEFINITION_MARKER)
    assert prompt.endswith(f"{aw.USER_MESSAGE_MARKER}\nhello")
    assert "[WF:20261009150839][agent:coder_agent]" in prompt
    # the stamped git version is gone; only the database version remains
    assert "20261008225854" not in prompt
    assert "## Mission" in prompt and "# Contract" in prompt and "browser-verification" in prompt
    assert seen[0].full_url.endswith(f"/companies/{COMPANY}/workflow/export")
    assert seen[0].get_header("Authorization") == "Bearer tok"
    assert state["llm:t1"]["header"] == ("20261009150839", "coder_agent")


def test_later_turns_are_plain_until_the_workflow_changes(monkeypatch, tmp_path):
    _serve(monkeypatch, _export())
    state = {}
    aw.prepare_turn(state, "llm:t1", AGENT, "Bearer tok", "one", str(tmp_path))
    assert aw.prepare_turn(state, "llm:t1", AGENT, "Bearer tok", "two", str(tmp_path)) == "two"
    aw._cache.clear()
    _serve(monkeypatch, _export(version="20261010000000", role="# coder_agent\n\nNew mission.\n"))
    changed = aw.prepare_turn(state, "llm:t1", AGENT, "Bearer tok", "three", str(tmp_path))
    assert "New mission." in changed and "[WF:20261010000000]" in changed


def test_the_published_files_are_on_disk_and_paths_cannot_escape(monkeypatch, tmp_path):
    _serve(monkeypatch, _export())
    aw.prepare_turn({}, "llm:t1", AGENT, "Bearer tok", "hi", str(tmp_path))
    root = tmp_path / "workflows" / COMPANY / "20261009150839"
    assert (root / "skills" / "browser-verification" / "SKILL.md").read_text() == "# Browser"
    assert not (tmp_path / "workflows" / COMPANY / "escape.md").exists()


def test_a_missing_role_or_refused_read_stops_the_turn(monkeypatch, tmp_path):
    _serve(monkeypatch, _export())
    with pytest.raises(aw.WorkflowUnavailable):
        aw.prepare_turn({}, "llm:t1", {**AGENT, "role_file": "agents/roles/nobody.md"}, "Bearer tok", "hi", str(tmp_path))

    def refused(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    aw._cache.clear()
    with pytest.raises(aw.WorkflowUnavailable, match="cannot read"):
        aw.fetch_export(COMPANY, "Bearer tok", opener=refused)
    with pytest.raises(aw.WorkflowUnavailable):
        aw.fetch_export("../x", "Bearer tok")
    with pytest.raises(aw.WorkflowUnavailable):
        aw.fetch_export(COMPANY, "")


def test_replies_without_the_header_get_it():
    assert aw.ensure_header("Done.", "20261009150839", "coder_agent") == "[WF:20261009150839][agent:coder_agent]\nDone."
    already = "[WF:20261009150839][agent:coder_agent]\nDone."
    assert aw.ensure_header(already, "20261009150839", "coder_agent") == already
    assert aw.ensure_header("", "20261009150839", "coder_agent") == ""
