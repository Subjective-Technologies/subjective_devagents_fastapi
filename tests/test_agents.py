"""The agent registry: reading it, and editing it without breaking it.

An agent is defined by two files that must stay in step — a row in registry.md and a role
file in roles/. Drift between them is the failure these tests guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import agents  # noqa: E402

REGISTRY = """# Agent Registry

This registry lists reusable logical roles. Models/providers are runtime metadata and MUST NOT be part of agent identity.

| Agent ID | Role file | Primary capability |
|---|---|---|
| `manager_agent` | `roles/manager_agent.md` | orchestration |
| `coder_agent` | `roles/coder_agent.md` | feature implementation |

## Instance convention

Do not create a new role file per feature.
"""


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_AGENTS_DIR", str(tmp_path))
    (tmp_path / "registry.md").write_text(REGISTRY, encoding="utf-8")
    roles = tmp_path / "roles"
    roles.mkdir()
    for name in ("manager_agent", "coder_agent"):
        (roles / f"{name}.md").write_text(f"# {name}\n\n## Mission\n\nDo the thing.\n", encoding="utf-8")
    return tmp_path


def test_lists_every_registered_agent(registry):
    listed = agents.list_agents()

    assert listed["ok"] is True
    assert [item["id"] for item in listed["agents"]] == ["manager_agent", "coder_agent"]
    assert all(item["exists"] for item in listed["agents"])
    assert listed["agents"][0]["capability"] == "orchestration"


def test_a_registry_row_without_its_role_file_is_flagged_not_hidden(registry):
    """Dropping it silently would make a broken registry look healthy."""
    (registry / "roles" / "coder_agent.md").unlink()

    listed = {item["id"]: item for item in agents.list_agents()["agents"]}

    assert listed["coder_agent"]["exists"] is False


def test_creating_an_agent_writes_both_the_role_file_and_the_registry_row(registry):
    result = agents.create_agent("triage_agent", capability="incident triage", model="claude")

    assert result["ok"] is True, result
    role = (registry / "roles" / "triage_agent.md").read_text(encoding="utf-8")
    assert role.startswith("# triage_agent")
    assert "Model: `claude`" in role
    text = (registry / "registry.md").read_text(encoding="utf-8")
    assert "| `triage_agent` | `roles/triage_agent.md` | incident triage |" in text
    # Appended to the table, not to the end of the document.
    assert text.index("triage_agent") < text.index("## Instance convention")
    assert [item["id"] for item in agents.list_agents()["agents"]][-1] == "triage_agent"


def test_the_model_is_runtime_metadata_not_identity(registry):
    """registry.md's own contract: models must not be part of agent identity, so the
    model belongs in the role file's Runtime section, never in the registry table."""
    agents.create_agent("triage_agent", capability="incident triage", model="grok")

    table_row = [
        line for line in (registry / "registry.md").read_text(encoding="utf-8").splitlines()
        if "triage_agent" in line
    ][0]

    assert "grok" not in table_row
    assert "## Runtime" in (registry / "roles" / "triage_agent.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", ["", "Has Caps", "../escape", "a", "with-dash", "x" * 80, "9leading"])
def test_invalid_agent_ids_are_rejected(registry, bad):
    """The id becomes a filename and a registry key."""
    assert agents.create_agent(bad, capability="x")["ok"] is False


def test_duplicate_agents_are_refused(registry):
    assert agents.create_agent("manager_agent", capability="again")["ok"] is False


def test_changing_the_model_rewrites_only_the_runtime_section(registry):
    role = registry / "roles" / "coder_agent.md"
    role.write_text(
        "# coder_agent\n\n## Mission\n\nCarefully worded prose.\n\n## Runtime\n\nModel: `codex`\n",
        encoding="utf-8",
    )

    assert agents.set_agent_model("coder_agent", "claude")["ok"] is True

    updated = role.read_text(encoding="utf-8")
    assert "Model: `claude`" in updated
    assert "Model: `codex`" not in updated
    assert "Carefully worded prose." in updated, "a model change must not touch the role's own prose"
    assert agents.read_model("coder_agent") == "claude"


def test_a_model_can_be_set_on_a_role_that_had_none(registry):
    assert agents.read_model("manager_agent") == ""

    agents.set_agent_model("manager_agent", "claude")

    assert agents.read_model("manager_agent") == "claude"


def test_model_and_agent_inputs_are_validated(registry):
    assert agents.set_agent_model("coder_agent", "../../etc/passwd")["ok"] is False
    assert agents.set_agent_model("no_such_agent", "claude")["ok"] is False


def test_editing_a_role_overwrites_its_full_prose(registry):
    result = agents.write_role("coder_agent", "# coder_agent\n\n## Mission\n\nRewritten by the operator.\n")

    assert result["ok"] is True, result
    role = (registry / "roles" / "coder_agent.md").read_text(encoding="utf-8")
    assert role == "# coder_agent\n\n## Mission\n\nRewritten by the operator.\n"
    assert agents.read_role("coder_agent")["text"] == role


def test_editing_preserves_the_runtime_section_since_it_is_not_special_cased(registry):
    """Unlike set_agent_model, write_role replaces the file verbatim — the operator is
    editing on purpose, Runtime section included."""
    agents.set_agent_model("coder_agent", "claude")

    agents.write_role("coder_agent", "# coder_agent\n\n## Mission\n\nNo Runtime section anymore.\n")

    updated = (registry / "roles" / "coder_agent.md").read_text(encoding="utf-8")
    assert "## Runtime" not in updated
    assert agents.read_model("coder_agent") == ""


def test_editing_a_nonexistent_or_empty_role_is_refused(registry):
    assert agents.write_role("no_such_agent", "# no_such_agent\n")["ok"] is False
    assert agents.write_role("coder_agent", "   \n")["ok"] is False
    assert agents.write_role("../../etc/passwd", "x")["ok"] is False
    # Refused, so the original file must be untouched.
    assert "Do the thing." in (registry / "roles" / "coder_agent.md").read_text(encoding="utf-8")


def test_the_real_repository_registry_parses(monkeypatch):
    """The shipped registry must actually be readable by this parser."""
    monkeypatch.delenv("SUBJECTIVE_AGENTS_DIR", raising=False)
    listed = agents.list_agents()
    if not listed["ok"]:
        pytest.skip("workflow/agents is not present in this checkout")
    assert len(listed["agents"]) >= 20
    assert all(item["exists"] for item in listed["agents"]), "registry points at a missing role file"
