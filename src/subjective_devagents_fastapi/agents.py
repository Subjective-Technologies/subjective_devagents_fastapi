"""The agent registry in `workflow/agents/` — reading it, and editing it safely.

Two files define an agent and both must stay in step: a row in `registry.md` and a role
file in `roles/`. Drift between them is the failure this module exists to prevent, so a
create writes both or neither.

`registry.md` states that models/providers are runtime metadata and must not be part of
agent identity. The model therefore lives in its own "## Runtime" section of the role
file, clearly labelled as such, rather than in the registry table or the agent id: which
CLI answers for a role is a deployment choice, not what makes it that role.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# The id becomes a filename and a registry key, so it is deliberately narrow.
AGENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}$")
_REGISTRY_ROW_RE = re.compile(r"^\|\s*`(?P<id>[^`]+)`\s*\|\s*`(?P<file>[^`]+)`\s*\|\s*(?P<capability>[^|]*?)\s*\|\s*$")
_MODEL_RE = re.compile(r"^\s*Model:\s*`?(?P<model>[A-Za-z0-9_.-]+)`?\s*$", re.MULTILINE)
_RUNTIME_SECTION_RE = re.compile(r"\n## Runtime\b.*?(?=\n## |\Z)", re.DOTALL)

REGISTRY_NAME = "registry.md"
ROLES_DIRNAME = "roles"


def agents_dir() -> Path:
    """Where the registry lives.

    SUBJECTIVE_AGENTS_DIR wins, so tests and other checkouts are not pinned to one path.
    Otherwise walk up from this file looking for workflow/agents: the service's own root
    is the client bundle, not the monorepo, so it cannot be derived from that.
    """
    configured = os.environ.get("SUBJECTIVE_AGENTS_DIR")
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "workflow" / "agents"
        if (candidate / REGISTRY_NAME).is_file():
            return candidate
    return Path("/subjective/workflow/agents")


def _registry_path() -> Path:
    return agents_dir() / REGISTRY_NAME


def _role_path(agent_id: str) -> Path:
    return agents_dir() / ROLES_DIRNAME / f"{agent_id}.md"


def is_valid_agent_id(agent_id: str) -> bool:
    return bool(AGENT_ID_RE.match(str(agent_id or "")))


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_atomic(path: Path, text: str) -> None:
    """Never leave a half-written registry: a truncated table would orphan every agent
    below the truncation point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_model(agent_id: str) -> str:
    """The runtime model recorded in the role file, or "" when the role leaves it open."""
    match = _MODEL_RE.search(_read(_role_path(agent_id)))
    return match.group("model").strip().lower() if match else ""


def list_agents() -> dict:
    """Every agent in the registry, with its role file and runtime model.

    A registry row whose role file is missing is still listed, flagged rather than hidden:
    silently dropping it would make a broken registry look healthy.
    """
    text = _read(_registry_path())
    if not text:
        return {"ok": False, "error": f"No agent registry at {_registry_path()}", "agents": []}
    agents = []
    for line in text.splitlines():
        match = _REGISTRY_ROW_RE.match(line)
        if not match:
            continue
        agent_id = match.group("id").strip()
        if agent_id.lower() in {"agent id", "---"}:
            continue
        role_file = match.group("file").strip()
        path = agents_dir() / role_file
        agents.append({
            "id": agent_id,
            "roleFile": role_file,
            "capability": match.group("capability").strip(),
            "model": read_model(agent_id),
            "exists": path.is_file(),
        })
    return {"ok": True, "agents": agents, "dir": str(agents_dir())}


def _role_template(agent_id: str, capability: str, mission: str, model: str) -> str:
    return f"""# {agent_id}

## Mission

{mission or capability or "Describe what this agent is for."}

## Authority

Describe what this agent decides on its own.

## Responsibilities

- {capability or "Describe the work this agent owns."}

## Inputs

Describe what this agent needs to do its work.

## Outputs

Describe what this agent produces.

## Forbidden

Describe what this agent must not do.

## Done

Describe the evidence that closes this agent's work.

{runtime_section(model)}"""


def runtime_section(model: str) -> str:
    """Model as runtime metadata, kept out of the registry table and the agent id.

    The registry's contract is that a role is defined by what it does, not by which model
    happens to run it — so re-pointing this line at a different CLI does not make it a
    different agent.
    """
    chosen = str(model or "").strip().lower()
    return (
        "## Runtime\n\n"
        f"Model: `{chosen}`\n\n"
        "Runtime metadata, not identity: which CLI answers for this role is a deployment\n"
        "choice. Changing it re-points the agent; it does not make it a different agent.\n"
        if chosen
        else ""
    )


def set_agent_model(agent_id: str, model: str) -> dict:
    """Point an existing agent at a different CLI, rewriting only its Runtime section."""
    if not is_valid_agent_id(agent_id):
        return {"ok": False, "error": "Invalid agent id."}
    path = _role_path(agent_id)
    if not path.is_file():
        return {"ok": False, "error": f"No role file for {agent_id}."}
    chosen = str(model or "").strip().lower()
    if chosen and not re.match(r"^[a-z0-9_.-]{1,40}$", chosen):
        return {"ok": False, "error": "Invalid model."}
    text = _read(path)
    section = runtime_section(chosen)
    if _RUNTIME_SECTION_RE.search(text):
        # Replace in place so an agent's own prose is never rewritten by a model change.
        updated = _RUNTIME_SECTION_RE.sub("\n" + section if section else "\n", text).rstrip() + "\n"
    elif section:
        updated = text.rstrip() + "\n\n" + section
    else:
        updated = text
    _write_atomic(path, updated)
    return {"ok": True, "id": agent_id, "model": chosen}


def create_agent(agent_id: str, capability: str = "", model: str = "", mission: str = "") -> dict:
    """Add an agent: role file plus registry row, or neither.

    The registry row is written last, so a failure part-way leaves an unreferenced file
    rather than a registry pointing at something that does not exist.
    """
    agent_id = str(agent_id or "").strip().lower()
    if not is_valid_agent_id(agent_id):
        return {"ok": False, "error": "Agent id must be lowercase letters, digits and underscores."}
    registry_path = _registry_path()
    registry = _read(registry_path)
    if not registry:
        return {"ok": False, "error": f"No agent registry at {registry_path}"}
    if any(agent["id"] == agent_id for agent in (list_agents().get("agents") or [])):
        return {"ok": False, "error": f"{agent_id} is already registered."}
    role_path = _role_path(agent_id)
    if role_path.exists():
        return {"ok": False, "error": f"{role_path.name} already exists."}

    capability = str(capability or "").strip()
    chosen = str(model or "").strip().lower()
    if chosen and not re.match(r"^[a-z0-9_.-]{1,40}$", chosen):
        return {"ok": False, "error": "Invalid model."}

    _write_atomic(role_path, _role_template(agent_id, capability, str(mission or "").strip(), chosen))

    row = f"| `{agent_id}` | `{ROLES_DIRNAME}/{agent_id}.md` | {capability or 'unspecified'} |"
    lines = registry.splitlines()
    last_row = max(
        (index for index, line in enumerate(lines) if _REGISTRY_ROW_RE.match(line)),
        default=-1,
    )
    if last_row < 0:
        role_path.unlink(missing_ok=True)
        return {"ok": False, "error": "Could not find the registry table."}
    lines.insert(last_row + 1, row)
    _write_atomic(registry_path, "\n".join(lines).rstrip() + "\n")
    return {
        "ok": True,
        "agent": {
            "id": agent_id,
            "roleFile": f"{ROLES_DIRNAME}/{agent_id}.md",
            "capability": capability,
            "model": chosen,
            "exists": True,
        },
    }


def read_role(agent_id: str) -> dict:
    """The role file's prose, for seeding a chat with what the agent is supposed to do."""
    if not is_valid_agent_id(agent_id):
        return {"ok": False, "error": "Invalid agent id."}
    path = _role_path(agent_id)
    if not path.is_file():
        return {"ok": False, "error": f"No role file for {agent_id}."}
    return {"ok": True, "id": agent_id, "model": read_model(agent_id), "text": _read(path)}
