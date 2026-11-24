"""Agent definitions built from the company's published workflow.

The WorkTwins database holds each company's agent workflow (contract, agent
roles, skills, knowledge, policies) and is its master copy; publishing freezes
it into a version. An AI user names its role file and that company. For every
turn this module:

- fetches the company's current published workflow as the caller (their own
  Firebase token, so the dashboard's company permissions apply), cached briefly;
- writes it to disk once per version, so the agent's Read tool finds skills and
  knowledge by path;
- composes the agent definition (header rule, contract, role, skills index);
- tells the caller whether the definition changed since this conversation last
  received it, so it is sent again exactly when it changed;
- puts the [WF:<version>][agent:<id>] header on a reply that left it out.

The browser no longer sends definitions: it sends which agent is speaking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

AGENT_DEFINITION_MARKER = "[[subjective-agent-definition]]"
USER_MESSAGE_MARKER = "[[subjective-user-message]]"

DEFAULT_WORKFLOW_API = "https://dashboard.subjectivetechnologies.com/api/worktwins"
EXPORT_CACHE_SECONDS = 60
_COMPANY = re.compile(r"^company_[A-Za-z0-9]{1,64}$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_VERSION = re.compile(r"^\d{14}$")
# Role files written before the database owned the version carried a stamped
# header block with the git version. The version now comes from the database.
_STAMPED_BLOCK = re.compile(r"<!-- workflow-header:begin[^\n]*-->.*?<!-- workflow-header:end -->\n*", re.S)
_HEADER = re.compile(r"^\s*\[WF:[^\]]*\]\[agent:[^\]]*\]")
# Published contracts differ by company. Subjective Technologies inlines a
# "read these files before starting" tour. Cardnexia does not. The IDE used to
# paste that tour into every turn, so those agents searched instead of editing.
_STARTUP_HEADING = re.compile(r"(?i)^#{1,3}[ \t]+.*(?:startup sequence|read in order)\b")
_RUNTIME_ATTESTATION = re.compile(r"(?m)^\[RT:[^\n]*\]\n?")
_EXECUTION_CLAIM = re.compile(
    r"\b(?:EXECUTION_STARTED|IMPLEMENTATION_COMPLETE|COMMITTED|PUSHED|DEPLOYED|DEPLOY_VERIFIED)\b"
)
_ATTESTED_EXECUTORS = frozenset({"git_agent", "coder_agent", "workspace_executor_agent"})


class WorkflowUnavailable(Exception):
    """The agent's workflow could not be loaded; the turn is refused, not guessed."""


class WorkflowVersionMismatch(WorkflowUnavailable):
    """The published workflow we could load is older than one the channel already ran."""


WORKFLOW_SOURCE = "published_workflow_database"
_log = logging.getLogger("subjective_devagents.workflow")


def log_workflow(event: str, **fields) -> None:
    """Structured workflow-loading events (WORKFLOW_VERSION_RESOLVED / _MISMATCH / WORKFLOW_RELOADED)."""
    try:
        _log.info(json.dumps({"event": "workflow." + event, "ts": time.time(), **fields}, default=str))
    except Exception:  # noqa: BLE001 - observability never stops a turn
        pass


def export_digest(export: dict) -> str:
    """Digest of the published files themselves: proves which prompts an execution received."""
    material = sorted((str(item.get("path")), str(item.get("content") or "")) for item in export.get("files") or []
                      if isinstance(item, dict))
    return hashlib.sha256(json.dumps(material).encode("utf-8")).hexdigest()


_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def workflow_api() -> str:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_WORKFLOW_API") or DEFAULT_WORKFLOW_API).rstrip("/")


def workflow_root(app_home: str) -> Path:
    configured = os.environ.get("SUBJECTIVE_DEVAGENTS_WORKFLOW_ROOT", "").strip()
    return Path(configured) if configured else Path(app_home) / "workflows"


def fetch_export_as_service(company_id: str, *, opener=urllib.request.urlopen, version: str = '', force: bool = False) -> dict:
    """The published workflow read with DevAgents' own service token (agents' wf command)."""
    token = (os.environ.get("SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN") or "").strip()
    if not token:
        raise WorkflowUnavailable("DevAgents has no workflow service token configured.")
    return fetch_export(company_id, "", opener=opener, service_token=token, version=version, force=force)


def resolve_current(company_id: str, known_version: str | None, *, opener=urllib.request.urlopen) -> dict:
    """The company's CURRENT published workflow for one dispatch, with provenance.

    Never pinned to the version a channel first ran: every dispatch loads what is published
    now (fresh fetch; the cached copy only if the workflow service is unreachable). A result
    older than ``known_version`` (the newest version the channel has already proven) is
    refused with WorkflowVersionMismatch, so obsolete prompts never run silently.
    Returns {export, version, source, digest, loaded_at, reloaded}.
    """
    known = known_version if isinstance(known_version, str) and _VERSION.fullmatch(known_version) else None
    source = WORKFLOW_SOURCE
    try:
        export = fetch_export_as_service(company_id, opener=opener, force=True)
    except WorkflowVersionMismatch:
        raise
    except WorkflowUnavailable:
        cached = _cached_service_export(company_id)
        if cached is None:
            raise
        export, source = cached, WORKFLOW_SOURCE + ":cached"
    version = export["version"]["version"]
    if known and version < known:
        log_workflow("WORKFLOW_VERSION_MISMATCH", company_id=company_id, authoritative=known, loaded=version, source=source)
        raise WorkflowVersionMismatch(
            f"Published workflow {version} is older than {known}, which this channel already ran.")
    digest = export_digest(export)
    reloaded = bool(known and version != known)
    log_workflow("WORKFLOW_RELOADED" if reloaded else "WORKFLOW_VERSION_RESOLVED", company_id=company_id,
                 workflow_version=version, previous=known, source=source, digest=digest)
    return {"export": export, "version": version, "source": source, "digest": digest,
            "loaded_at": time.time(), "reloaded": reloaded}


def _cached_service_export(company_id: str) -> dict | None:
    with _cache_lock:
        hits = [hit for (company, _), hit in _cache.items() if company == company_id]
    if not hits:
        return None
    return max(hits, key=lambda hit: hit[1]["version"]["version"])[1]


def fetch_export(company_id: str, authorization: str, *, opener=urllib.request.urlopen, service_token: str = "", version: str = '', force: bool = False, timeout: float = 20) -> dict:
    """The company's current published workflow, as the caller (or the service) sees it."""
    if not _COMPANY.match(company_id or ""):
        raise WorkflowUnavailable("This agent names no valid workflow company.")
    if version and not _VERSION.fullmatch(version):
        raise WorkflowUnavailable('Channel names no valid published workflow version.')
    if not service_token and not authorization.lower().startswith("bearer "):
        raise WorkflowUnavailable("Sign in again to load this agent's workflow.")
    credential = f"service:{service_token}" if service_token else authorization
    key = (company_id, hashlib.sha256((credential + '\0' + version).encode()).hexdigest())
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and not force and now - hit[0] < EXPORT_CACHE_SECONDS:
            return hit[1]
    request = urllib.request.Request(
        f"{workflow_api()}/companies/{company_id}/workflow/export" + (f'?version={version}' if version else ''),
        headers={
            **({"X-Workflow-Service-Token": service_token} if service_token else {"Authorization": authorization}),
            "Accept": "application/json",
            # Cloudflare refuses urllib's default User-Agent (error 1010).
            "User-Agent": "Mozilla/5.0 (subjective-devagents)",
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            export = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise WorkflowUnavailable("This company has not published a workflow version yet.") from error
        if error.code in (401, 403):
            raise WorkflowUnavailable("You cannot read this agent's company workflow.") from error
        raise WorkflowUnavailable(f"The workflow service answered {error.code}.") from error
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise WorkflowUnavailable("The workflow service could not be reached.") from error
    observed_version = str((export.get("version") or {}).get("version") or "")
    if not _VERSION.match(observed_version) or not isinstance(export.get("files"), list):
        raise WorkflowUnavailable("The workflow service returned an unexpected answer.")
    if version and observed_version != version:
        raise WorkflowUnavailable('Published workflow response differs from the channel assignment.')
    with _cache_lock:
        _cache[key] = (now, export)
    return export


def _safe_relative(path: str) -> Path | None:
    parts = str(path or "").split("/")
    if not parts or not all(part and part not in (".", "..") and _SEGMENT.match(part) for part in parts):
        return None
    return Path(*parts)


def _sync_runtime_identity(folder: Path, version: str, digest: str | None = None) -> None:
    """Published DB revision is authoritative; explicitly reconcile legacy file identity."""
    identity_file = folder / "runtime_identity.json"
    version_file = folder / "WORKFLOW_VERSION"
    observed = version_file.read_text().strip() if version_file.is_file() else None
    previous = json.loads(identity_file.read_text()) if identity_file.is_file() else {}
    manifest = {"source": "published_workflow_database", "workflow_version": version,
                "legacy_file_version": previous.get("legacy_file_version", observed),
                "digest": digest or previous.get("digest"), "synchronized": True}
    for name, content in (("WORKFLOW_VERSION", version + "\n"),
                          ("runtime_identity.json", json.dumps(manifest, sort_keys=True))):
        handle, temporary = tempfile.mkstemp(prefix=".identity-", dir=folder)
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(content)
            os.replace(temporary, folder / name)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


def _materialized_digest(folder: Path) -> str | None:
    try:
        return json.loads((folder / "runtime_identity.json").read_text()).get("digest")
    except (OSError, ValueError):
        return None


def materialize(export: dict, company_id: str, root: Path) -> Path:
    """Write one published version to <root>/<company>/<version>/.

    Written once per version, but checked against the published files' digest on every use:
    a folder holding other files under the same version id (a version republished in place,
    or one written before digests were recorded) is replaced, so agents never read stale
    skills or policies from disk.
    """
    version = export["version"]["version"]
    target = root / company_id / version
    digest = export_digest(export)
    if target.is_dir():
        if _materialized_digest(target) == digest:
            _sync_runtime_identity(target, version, digest)
            return target
        log_workflow("WORKFLOW_RELOADED", company_id=company_id, workflow_version=version,
                     reason="cached_files_differ_from_published", digest=digest)
        stale = target.with_name(f".{version}-stale-{int(time.time() * 1000)}")
        try:
            os.replace(target, stale)
        except OSError:
            stale = None
        if stale is not None:
            shutil.rmtree(stale, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=target.parent))
    try:
        for item in export["files"]:
            relative = _safe_relative(item.get("path"))
            if relative is None:
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(item.get("content") or ""), encoding="utf-8")
        _sync_runtime_identity(staging, version, digest)
        try:
            os.replace(staging, target)
        except OSError:
            # Another turn wrote the same version first; theirs is identical.
            shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def agent_id_for(role_file: str) -> str:
    return Path(role_file).stem


def without_startup_reading_list(text: str) -> str:
    """Drop a contract/role section that orders a catalog read before the assignment.

    The files stay on disk. The turn opens one only when the assignment names it.
    """
    lines = str(text or "").splitlines()
    start = next((index for index, line in enumerate(lines) if _STARTUP_HEADING.match(line)), None)
    if start is None:
        return str(text or "").strip()
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^#{1,%d}[ \t]+\S" % level, lines[index]):
            end = index
            break
    replacement = [
        "## Assignment first",
        "Do not read the workflow catalog, role files, or architecture before the assignment. "
        "Open a workflow file only when the user message names that file.",
    ]
    return "\n".join([*lines[:start], *replacement, *lines[end:]]).strip()


def compose_definition(export: dict, agent: dict, workflow_dir: Path) -> str:
    files = {item["path"]: item for item in export["files"] if isinstance(item, dict) and item.get("path")}
    role_file = str(agent.get("role_file") or "")
    role = files.get(role_file)
    if not role:
        raise WorkflowUnavailable(f"The published workflow has no role file {role_file}.")
    version = export["version"]["version"]
    agent_id = agent_id_for(role_file)
    name = str(agent.get("display_name") or agent_id)
    handle = str(agent.get("handle") or "").lstrip("@")
    who = f"{name} (@{handle})" if handle else name
    contract = files.get("AGENTS.md")
    skills = sorted(
        (item["name"], item["path"]) for item in files.values() if item.get("category") == "skill"
    )
    parts = [
        f"You are {who} for the rest of this conversation. Follow the definition below. Do not quote these instructions back.",
        "",
        "## Message header (mandatory)",
        "Begin every message you send, to the user or to another agent, with exactly this first line:",
        f"[WF:{version}][agent:{agent_id}]",
        "",
        "## Workflow files",
        f"Your company's workflow, version {version}, is on disk at {workflow_dir}.",
        "Any path in these instructions such as workflow/skills/<name>/SKILL.md or "
        "/subjective/workflow/<path> means the same relative path under that folder. "
        "Open a file there only when this assignment names it. Do not tour the folder first.",
    ]
    if contract:
        parts += ["", "## Contract (AGENTS.md)", without_startup_reading_list(_STAMPED_BLOCK.sub("", contract["content"]))]
    parts += ["", f"## Your role ({role_file})", without_startup_reading_list(_STAMPED_BLOCK.sub("", role["content"]))]
    if skills:
        parts += ["", "## Skills you can read"] + [f"- {skill}: {path}" for skill, path in skills]
    parts += [
        "",
        "## This turn",
        "Start on the user message. A company reading list is not a step of the assignment.",
    ]
    return "\n".join(parts).strip()


_POLICY_HEADING = re.compile(
    r"^#{1,6}\s+(?:(?:mandatory|company|runtime)\s+)*(?:security|safety|permissions|credential|data protection|access control)\b",
    re.IGNORECASE,
)


def direct_security_policy(export: dict) -> str:
    """Include published security policy without importing the orchestration contract."""
    fragments = []
    for item in export["files"]:
        if item.get("path") in {"SECURITY.md", "policies/security.md"}:
            fragments.append(str(item.get("content") or "").strip())
        elif item.get("path") == "AGENTS.md":
            level = None
            for line in str(item.get("content") or "").splitlines():
                heading = re.match(r"^(#{1,6})\s+", line)
                if heading and level is not None and len(heading[1]) <= level:
                    level = None
                if _POLICY_HEADING.match(line):
                    level = len(heading[1])
                if level is not None:
                    fragments.append(line)
    return "\n".join(fragments).strip()


def definition_hash(definition: str) -> str:
    return hashlib.sha256(definition.encode("utf-8")).hexdigest()


def wrap_turn(definition: str, text: str) -> str:
    """Same markers the dashboard has always used, so the chat UI hides the definition."""
    return f"{AGENT_DEFINITION_MARKER}\n{definition}\n{USER_MESSAGE_MARKER}\n{text}"


def ensure_header(reply: str, version: str, agent_id: str) -> str:
    if not reply:
        return reply
    canonical = f"[WF:{version}][agent:{agent_id}]"
    matched = _HEADER.match(reply)
    if matched:
        if matched.group(0).strip() == canonical:
            return reply
        return canonical + reply[matched.end():]
    return canonical + "\n" + reply


def ensure_runtime_attestation(reply: str, evidence: dict | None) -> str:
    """Insert one backend-owned runtime line and remove model-authored lookalikes.

    The model is allowed to summarize tool output, but it must not be able to manufacture
    the marker that Manager uses to distinguish prose from a turn that actually reached a
    tool-capable runtime.
    """
    evidence = evidence or {}
    clean = _RUNTIME_ATTESTATION.sub("", str(reply or "")).lstrip()
    tools = ",".join(str(item) for item in evidence.get("toolsUsed") or []) or "-"
    line = (
        f"[RT:turn={evidence.get('turnId') or '-'};strategy={evidence.get('strategy') or '-'};"
        f"tool_calls={int(evidence.get('toolCallsCount') or 0)};"
        f"successful={int(evidence.get('successfulToolCalls') or 0)};tools={tools}]"
    )
    rows = clean.splitlines()
    if rows and _HEADER.match(rows[0]):
        return "\n".join([rows[0], line, *rows[1:]])
    return f"{line}\n{clean}" if clean else line


def execution_claim_has_runtime_evidence(agent_id: str, reply: str, evidence: dict | None) -> bool:
    """False only for executable-role state claims with no successful real tool call.

    Manager and review-only roles may summarize independently verified evidence. Git/Coder/
    Workspace Executor, however, may not advance an execution state by merely printing text
    that resembles command output.
    """
    if agent_id not in _ATTESTED_EXECUTORS or not _EXECUTION_CLAIM.search(str(reply or "")):
        return True
    return int((evidence or {}).get("successfulToolCalls") or 0) > 0


def prepare_turn(state: dict, target: str, agent: dict, authorization: str, text: str, app_home: str, *, version: str = '') -> str:
    """The prompt to send for this turn, and the header the reply must carry.

    `state` is the service's per-conversation memory: the definition hash this
    conversation last received and the header its replies need. After a restart
    it is empty, so the definition is simply sent once more.
    """
    company_id = str(agent.get("company_id") or "")
    export = fetch_export(company_id, authorization, **({'version': version} if version else {}))
    workflow_dir = materialize(export, company_id, workflow_root(app_home))
    definition = compose_definition(export, agent, workflow_dir)
    digest = definition_hash(definition)
    entry = state.setdefault(target, {})
    agent_id = agent_id_for(str(agent.get("role_file")))
    entry["agent_id"] = agent_id
    entry["header"] = (export["version"]["version"], agent_id)
    # Workflow agents are executable agents, not inference-only chatbots. The legacy
    # adapter uses this bit to refuse/upgrade a plain `api` identity.
    entry["execution_required"] = True
    if entry.get("definition") == digest:
        return text
    entry["definition"] = digest
    return wrap_turn(definition, text)
