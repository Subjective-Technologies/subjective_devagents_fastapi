"""Compatibility boundary for existing chat routes; execution belongs to LanguageModel."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import pwd
import shutil
import threading
import time
import uuid
from pathlib import Path

from .contracts import Attachment, Identity, Message, ModelError
from .model import LanguageModel
from .registry import IdentityRegistry, image_support
from .store import TurnStore, TERMINAL


PROXY_URL_ENV = "SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL"
PROXY_KEY_ENV = "SUBJECTIVE_DEVAGENTS_LLM_PROXY_API_KEY"
PROXY_MODEL_ENV = {"claude": "SUBJECTIVE_DEVAGENTS_LLM_MODEL_CLAUDE",
                   "codex": "SUBJECTIVE_DEVAGENTS_LLM_MODEL_CODEX",
                   "grok": "SUBJECTIVE_DEVAGENTS_LLM_MODEL_GROK",
                   "kimi": "SUBJECTIVE_DEVAGENTS_LLM_MODEL_KIMI",
                   "antigravity": "SUBJECTIVE_DEVAGENTS_LLM_MODEL_ANTIGRAVITY"}
# Providers with no CLI of their own: they run only as api_agent identities behind the proxy.
PROXY_ONLY_PROVIDERS = frozenset({"kimi", "antigravity"})


def proxy_default_model(provider: str, models: dict | None = None) -> str:
    """Use a configured default, or discover Antigravity's available model IDs."""
    configured = os.environ.get(PROXY_MODEL_ENV.get(provider, ""), "").strip()
    if configured or provider != "antigravity":
        return configured
    if models is None:
        from .catalog import proxy_models
        try:
            models = proxy_models()
        except ModelError:
            return ""
    return next(iter(models.get(provider, [])), "")


def proxy_configured(provider: str) -> bool:
    """Whether the inference proxy has a configured or discoverable model."""
    model_env = PROXY_MODEL_ENV.get(provider)
    return bool(model_env and os.environ.get(PROXY_URL_ENV, "").strip() and proxy_default_model(provider))
EXECUTION_STRATEGIES = frozenset({"cli", "api_agent"})


def proxy_identity(provider: str, username: str, workspace: str, home: str, selected_model: str = "") -> Identity | None:
    """An ad hoc api_agent identity when the inference proxy is configured for this provider.

    Returns None (caller falls back to the CLI identity below) when the operator has not set
    SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL and the per-provider model id: this keeps every existing
    deployment on its current CLI behavior until it explicitly opts in. Antigravity can
    discover its default from the proxy catalog. Cursor has no backend
    behind the proxy, so it is never offered one here.
    """
    model_env = PROXY_MODEL_ENV.get(provider)
    if model_env is None:
        return None
    base_url = os.environ.get(PROXY_URL_ENV, "").strip()
    model = selected_model or proxy_default_model(provider)
    if not base_url or not model:
        return None
    key = hashlib.sha256(json.dumps(["api_agent", provider, username, workspace, base_url, model]).encode()).hexdigest()[:32]
    return Identity(id=f"legacy-api-{key}", provider=provider, strategy="api_agent", model=model,
                    workspace=workspace, home=home, os_user=username,
                    api_base_url=base_url, api_key_env=PROXY_KEY_ENV, sandbox="workspace-write")


def find_executable(binary: str, home: str) -> str:
    """Find installed code without sourcing a shell or inheriting its credentials."""
    directories = [f"{home}/.local/bin", f"{home}/.npm-global/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    for base in dict.fromkeys((home, pwd.getpwuid(os.geteuid()).pw_dir)):
        try:
            versions = sorted((Path(base) / ".nvm/versions/node").glob("*/bin"),
                              key=lambda p: tuple(int(v) for v in p.parent.name.lstrip("v").split(".")), reverse=True)
            directories.extend(str(p) for p in versions)
        except (OSError, ValueError):
            pass
    return shutil.which(binary, path=":".join(directories)) or shutil.which(binary) or binary


class LegacyLanguageModel:
    def __init__(self, home: str):
        config = os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_CONFIG")
        self.configured = bool(config)
        registry = IdentityRegistry.load(config) if config else IdentityRegistry([])
        database = os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_DB") or str(Path(home) / "language_models.sqlite3")
        self.model = LanguageModel(TurnStore(database), registry)
        self._lock = threading.RLock()
        with self.model.store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS legacy_bindings (target TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, pending_turn TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS legacy_projections (target TEXT NOT NULL, turn_id TEXT NOT NULL, PRIMARY KEY(target,turn_id))")
            if "pending_turn" not in {row[1] for row in db.execute("PRAGMA table_info(legacy_bindings)")}:
                db.execute("ALTER TABLE legacy_bindings ADD COLUMN pending_turn TEXT")
        # One process owns this journal and has just started: no worker of an HTTP turn
        # survived. Left "running", such a turn refused every later send as busy.
        self.model.store.release_orphaned_http_turns()

    def identity(self, conversation: dict, execution_required: bool = False) -> Identity:
        provider = conversation["provider"]
        selected_model = conversation.get("model") or ""
        username = conversation.get("osUser") or pwd.getpwuid(os.geteuid()).pw_name
        workspace = str(Path(conversation.get("workspace") or "/tmp").absolute())
        if self.configured:
            choices = [self.model.identities.get(item["id"]) for item in self.model.identities.public()
                       if item["provider"] == provider]
            choices = [item for item in choices if item.workspace == workspace
                       and bool(item.ssh_host) == bool(conversation.get("sshConnection"))]
            if selected_model:
                choices = [item for item in choices if item.model == selected_model]
            # Remote accounts require an explicit saved-connection identity mapping.
            if conversation.get("sshConnection"):
                choices = [item for item in choices if item.id == conversation["sshConnection"]]
            else:
                choices = [item for item in choices
                           if (item.os_user or pwd.getpwuid(os.geteuid()).pw_name) == username]
            if execution_required:
                choices = [item for item in choices if item.strategy in EXECUTION_STRATEGIES]
                # Prefer the backend-owned tool loop when both a CLI identity and ApiAgent
                # identity point at the same workspace/account. It gives us durable tool events.
                api_agents = [item for item in choices if item.strategy == "api_agent"]
                if api_agents:
                    choices = api_agents
            if len(choices) != 1:
                if execution_required and not choices:
                    raise ModelError("execution_runtime_required", "This workflow agent requires a tool-capable cli/api_agent identity; inference-only api identities are not executable.")
                raise ModelError("configuration_required", "Configure exactly one matching provider/account/workspace identity; remote identity id must match the saved connection id.")
            return choices[0]
        if conversation.get("sshConnection"):
            raise ModelError("configuration_required", "Remote model execution requires an explicitly configured SSH identity.")
        if provider not in {"codex", "claude", "cursor", "grok"} | PROXY_ONLY_PROVIDERS:
            raise ModelError("unsupported_provider", "No standalone driver is registered for this provider.")
        account = pwd.getpwnam(username)
        home = account.pw_dir
        if selected_model:
            from .catalog import validate_model
            validate_model(provider, selected_model)
        api_identity = proxy_identity(provider, username, workspace, home, selected_model)
        if api_identity is not None:
            self.model.identities.register_legacy(api_identity)
            return api_identity
        if provider in PROXY_ONLY_PROVIDERS:
            raise ModelError("configuration_required",
                             f"{provider.title()} runs only through the LLM proxy: set {PROXY_URL_ENV} and {PROXY_MODEL_ENV[provider]}.")
        config_dir = str(Path(home) / {"codex": ".codex", "claude": ".claude", "cursor": ".config/cursor", "grok": ".grok"}[provider])
        binary = {"cursor": "cursor-agent"}.get(provider, provider)
        executable = find_executable(binary, home)
        key = hashlib.sha256(json.dumps([provider, username, home, workspace, executable]).encode()).hexdigest()[:32]
        identity = Identity(id=f"legacy-{key}", provider=provider, workspace=workspace, home=home,
                            config_dir=config_dir, os_user=username, executable=executable,
                            sandbox="workspace-write")
        self.model.identities.register_legacy(identity)
        return identity

    @staticmethod
    def history(source: dict) -> tuple[Message, ...]:
        messages = []
        for item in source.get("messages", []):
            if item.get("sendFailed") or item.get("stopped"):
                continue
            role = item.get("role")
            if role not in {"user", "assistant"}:
                raise ModelError("unsupported_history", "Conversation contains non-text turns; explicit migration is required.")
            text = str(item.get("text") or item.get("content") or "")
            images = [str((file or {}).get("filename") or "image") for file in item.get("attachments") or []]
            if images:
                # Earlier images are not re-sent to a new session; say so rather than
                # letting the transcript read as though the user referred to nothing.
                text = (text + "\n" if text else "") + f"[Attached image(s), not re-sent: {', '.join(images)}]"
            messages.append(Message(role, text))
        return tuple(messages)

    @staticmethod
    def attachments(files: list) -> tuple[Attachment, ...]:
        """The UI's {filename, mime_type, content: base64} images, decoded and validated."""
        decoded = []
        for item in files or []:
            if not isinstance(item, dict):
                raise ModelError("unsupported_attachments", "Attachments must be image objects.")
            try:
                data = base64.b64decode(str(item.get("content") or ""), validate=True)
            except (binascii.Error, ValueError) as error:
                raise ModelError("invalid_attachment", "An attached image is not valid base64.") from error
            decoded.append(Attachment(str(item.get("filename") or "image")[:200],
                                      str(item.get("mime_type") or "").lower(), data))
        return tuple(decoded)

    def send(self, conversation: dict, source: dict, request: dict, register, clear, progress) -> dict:
        try:
            images = self.attachments(request.get("files"))
            with self._lock:
                execution_required = bool(request.get("execution_required"))
                identity = self.identity(conversation, execution_required=execution_required)
                if images and not image_support(identity):
                    where = " over SSH" if identity.ssh_host else ""
                    raise ModelError("unsupported_attachments",
                                     f"{identity.provider.title()}{where} cannot receive images; send text only.")
                with self.model.store.connection() as db:
                    row = db.execute("SELECT conversation_id FROM legacy_bindings WHERE target=?", (conversation["target"],)).fetchone()
                if row is None:
                    created = self.model.create(identity.id, self.history(source))
                    with self.model.store.connection() as db:
                        db.execute("INSERT OR IGNORE INTO legacy_bindings(target,conversation_id) VALUES (?,?)", (conversation["target"], created["id"]))
                        row = db.execute("SELECT conversation_id FROM legacy_bindings WHERE target=?", (conversation["target"],)).fetchone()
                bound = self.model.store.conversation(row[0])
                if bound["identity"] != identity.snapshot():
                    previous_strategy = str((bound.get("identity") or {}).get("strategy") or "")
                    if execution_required and previous_strategy == "api" and identity.strategy in EXECUTION_STRATEGIES:
                        # Existing agent chats created before this fix may already be bound to
                        # an inference-only identity. Upgrade them safely while retaining history.
                        self.model.handoff(row[0], identity.id)
                        bound = self.model.store.conversation(row[0])
                    else:
                        raise ModelError("identity_changed", "Execution identity changed. Use explicit account handoff before sending.")
                pending = self.pending(conversation["target"])
                if pending and pending["status"] == "completed":
                    if pending["prompt"] == request["prompt"]:
                        return self.outcome(pending)
                    else:
                        raise ModelError("projection_pending", "Reload the completed previous turn before sending another prompt.")
                turn = self.model.submit(row[0], request["prompt"], str(uuid.uuid4()), request["timeout"], images, request.get("coordination"))
            register(conversation["target"], lambda: bool(self.model.cancel(turn["id"])))
            cursor = 0
            try:
                while True:
                    # Status first, then events: whatever the turn emitted before it ended
                    # (the end-of-turn diffs) is still read in this pass.
                    finished = self.model.store.turn(turn["id"])["status"] in TERMINAL
                    events = self.model.store.events(turn["id"], cursor)
                    for event in events:
                        cursor = event["sequence"]
                        # Whole steps only. Deltas are word fragments; the UI lists each
                        # progress text as its own row, so they stay in the event log.
                        if event["kind"] in {"activity", "message"} and progress:
                            progress({"target": conversation["target"], "text": event["data"].get("text", "")})
                        elif event["kind"] == "file_diff" and progress:
                            data = event["data"]
                            progress({"target": conversation["target"], "text": f"Edited {data.get('path', '')}",
                                     "diff": {"path": data.get("path"), "status": data.get("status"),
                                             "diff": data.get("diff")}})
                    if finished and len(events) < 1000:  # a full page may have more behind it
                        turn = self.model.store.turn(turn["id"])
                        break
                    time.sleep(0.05)
            finally:
                clear(conversation["target"])
            return self.outcome(turn)
        except ModelError as error:
            return {"success": False, "status": error.code, "error": str(error)}

    def execution_evidence(self, turn: dict) -> dict:
        """Backend-owned summary of the actual runtime/tool events for one durable turn."""
        events = self.model.store.events(turn["id"])
        runtime = next((item["data"] for item in events if item["kind"] == "runtime"), {})
        calls = [item["data"] for item in events if item["kind"] == "tool_call"]
        results = [item["data"] for item in events if item["kind"] == "tool_result"]
        successful = sum(1 for item in results if item.get("ok") is True)
        identity = turn.get("identity") if isinstance(turn.get("identity"), dict) else {}
        return {
            "turnId": turn["id"],
            "identityId": runtime.get("identity_id") or identity.get("id") or "",
            "strategy": runtime.get("strategy") or identity.get("strategy") or "",
            "workspace": runtime.get("workspace") or identity.get("workspace") or "",
            "effectiveTools": list(runtime.get("effective_tools") or []),
            "toolCallsCount": len(calls),
            "successfulToolCalls": successful,
            "toolsUsed": sorted({str(item.get("name") or "") for item in calls if item.get("name")}),
        }

    def outcome(self, turn: dict) -> dict:
        # A coordination block is the agent's answer, with its runtime evidence retained.
        return {"success": turn["status"] in {"completed", "blocked"}, "status": turn["status"],
                "response": turn["text"], "session_id": turn["session_id"] or "",
                "error": turn["error_code"] or "", "turn_id": turn["id"],
                "conversation_id": turn["conversation_id"],
                "execution_evidence": self.execution_evidence(turn)}

    def active_turn(self, target: str) -> bool:
        """The conversation bound to `target` has a queued or running turn."""
        with self.model.store.connection() as db:
            return db.execute("""SELECT 1 FROM turns t JOIN legacy_bindings b ON b.conversation_id=t.conversation_id
                WHERE b.target=? AND t.status IN ('queued','running') LIMIT 1""", (target,)).fetchone() is not None

    def pending(self, target: str) -> dict | None:
        with self.model.store.connection() as db:
            # Reservation already links every turn to the bound conversation before
            # dispatch. Recovery must not depend on a second post-dispatch write.
            row = db.execute("""SELECT t.id FROM turns t
                JOIN legacy_bindings b ON b.conversation_id=t.conversation_id
                WHERE b.target=? AND t.status='completed' AND t.request_id NOT LIKE 'import:%'
                AND NOT EXISTS (SELECT 1 FROM legacy_projections p WHERE p.target=b.target AND p.turn_id=t.id)
                ORDER BY t.rowid LIMIT 1""", (target,)).fetchone()
        return self.model.store.turn(row[0]) if row and row[0] else None

    def projected(self, target: str, turn_id: str):
        with self.model.store.connection() as db:
            db.execute("INSERT OR IGNORE INTO legacy_projections(target,turn_id) VALUES (?,?)", (target, turn_id))
            db.execute("UPDATE legacy_bindings SET pending_turn=NULL WHERE target=? AND pending_turn=?", (target, turn_id))

    def turn_details(self, turn_ids: list[str]) -> dict[str, dict]:
        """Per turn, steps/diffs plus backend-owned runtime execution evidence."""
        ids = sorted({str(item) for item in turn_ids if item})
        found: dict[str, dict] = {}
        if not ids:
            return found
        with self.model.store.connection() as db:
            for start in range(0, len(ids), 200):
                chunk = ids[start:start + 200]
                rows = db.execute(
                    f"SELECT turn_id, kind, data FROM events WHERE kind IN ('file_diff','activity') "
                    f"AND turn_id IN ({','.join('?' * len(chunk))}) ORDER BY sequence", chunk).fetchall()
                for turn_id, kind, raw in rows:
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        continue
                    entry = found.setdefault(turn_id, {"steps": [], "diffs": []})
                    if kind == "file_diff" and data.get("path") and data.get("diff"):
                        entry["diffs"].append(
                            {"path": data["path"], "status": data.get("status") or "modified", "diff": data["diff"]})
                    elif kind == "activity" and data.get("text"):
                        text = str(data["text"])
                        if (not entry["steps"] or entry["steps"][-1] != text) and len(entry["steps"]) < 200:
                            entry["steps"].append(text)
        for turn_id in ids:
            turn = self.model.store.turn(turn_id)
            evidence = self.execution_evidence(turn)
            found.setdefault(turn_id, {"steps": [], "diffs": []})["executionEvidence"] = evidence
        return found

    def is_bound(self, target: str) -> bool:
        with self.model.store.connection() as db:
            return db.execute("SELECT 1 FROM legacy_bindings WHERE target=?", (target,)).fetchone() is not None

    def clear_history(self, target: str, persist) -> None:
        """Reset local execution history and its native session without touching CLI files."""
        with self._lock, self.model.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            binding = db.execute("SELECT conversation_id FROM legacy_bindings WHERE target=?", (target,)).fetchone()
            if binding:
                identifier = binding[0]
                active = db.execute("SELECT 1 FROM turns WHERE conversation_id=? AND status IN ('queued','running')",
                                    (identifier,)).fetchone()
                if active:
                    raise ModelError("busy", "Wait for the current response to finish before clearing messages.")
                db.execute("DELETE FROM events WHERE turn_id IN (SELECT id FROM turns WHERE conversation_id=?)", (identifier,))
                db.execute("DELETE FROM turns WHERE conversation_id=?", (identifier,))
                db.execute("DELETE FROM legacy_projections WHERE target=?", (target,))
                db.execute("UPDATE legacy_bindings SET pending_turn=NULL WHERE target=?", (target,))
                db.execute("UPDATE conversations SET session_id=NULL, needs_reconciliation=0 WHERE id=?", (identifier,))
            persist()

    def status(self, provider: str) -> dict:
        candidates = [item for item in self.model.identities.public() if item["provider"] == provider]
        return {"provider": provider, "success": True, "status": "configured" if candidates else "not_checked",
                "backend": "language-model", "identities": candidates,
                **({} if candidates else {"error": f"No configured {provider} identity."})}

    def close(self):
        self.model.close()
