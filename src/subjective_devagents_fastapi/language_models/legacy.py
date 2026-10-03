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
                   "grok": "SUBJECTIVE_DEVAGENTS_LLM_MODEL_GROK"}


def proxy_identity(provider: str, username: str, workspace: str, home: str) -> Identity | None:
    """An ad hoc api_agent identity when the inference proxy is configured for this provider.

    Returns None (caller falls back to the CLI identity below) when the operator has not set
    SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL and the per-provider model id: this keeps every existing
    deployment on its current CLI behavior until it explicitly opts in. Cursor has no backend
    behind the proxy, so it is never offered one here.
    """
    model_env = PROXY_MODEL_ENV.get(provider)
    if model_env is None:
        return None
    base_url = os.environ.get(PROXY_URL_ENV, "").strip()
    model = os.environ.get(model_env, "").strip()
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

    def identity(self, conversation: dict) -> Identity:
        provider = conversation["provider"]
        username = conversation.get("osUser") or pwd.getpwuid(os.geteuid()).pw_name
        workspace = str(Path(conversation.get("workspace") or "/tmp").absolute())
        if self.configured:
            choices = [self.model.identities.get(item["id"]) for item in self.model.identities.public()
                       if item["provider"] == provider]
            choices = [item for item in choices if item.workspace == workspace
                       and bool(item.ssh_host) == bool(conversation.get("sshConnection"))]
            # Remote accounts require an explicit saved-connection identity mapping.
            if conversation.get("sshConnection"):
                choices = [item for item in choices if item.id == conversation["sshConnection"]]
            else:
                choices = [item for item in choices
                           if (item.os_user or pwd.getpwuid(os.geteuid()).pw_name) == username]
            if len(choices) != 1:
                raise ModelError("configuration_required", "Configure exactly one matching provider/account/workspace identity; remote identity id must match the saved connection id.")
            return choices[0]
        if conversation.get("sshConnection"):
            raise ModelError("configuration_required", "Remote model execution requires an explicitly configured SSH identity.")
        if provider not in {"codex", "claude", "cursor", "grok"}:
            raise ModelError("unsupported_provider", "No standalone driver is registered for this provider.")
        account = pwd.getpwnam(username)
        home = account.pw_dir
        api_identity = proxy_identity(provider, username, workspace, home)
        if api_identity is not None:
            self.model.identities.register_legacy(api_identity)
            return api_identity
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
                identity = self.identity(conversation)
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
                    raise ModelError("identity_changed", "Execution identity changed. Use explicit account handoff before sending.")
                pending = self.pending(conversation["target"])
                if pending and pending["status"] == "completed":
                    if pending["prompt"] == request["prompt"]:
                        return self.outcome(pending)
                    else:
                        raise ModelError("projection_pending", "Reload the completed previous turn before sending another prompt.")
                turn = self.model.submit(row[0], request["prompt"], str(uuid.uuid4()), request["timeout"], images)
            register(conversation["target"], lambda: bool(self.model.cancel(turn["id"])))
            cursor = 0
            try:
                while True:
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
                    turn = self.model.store.turn(turn["id"])
                    if turn["status"] in TERMINAL:
                        break
                    time.sleep(0.05)
            finally:
                clear(conversation["target"])
            return self.outcome(turn)
        except ModelError as error:
            return {"success": False, "status": error.code, "error": str(error)}

    @staticmethod
    def outcome(turn: dict) -> dict:
        return {"success": turn["status"] == "completed", "status": turn["status"],
                "response": turn["text"], "session_id": turn["session_id"] or "",
                "error": turn["error_code"] or "", "turn_id": turn["id"],
                "conversation_id": turn["conversation_id"]}

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

    def is_bound(self, target: str) -> bool:
        with self.model.store.connection() as db:
            return db.execute("SELECT 1 FROM legacy_bindings WHERE target=?", (target,)).fetchone() is not None

    def status(self, provider: str) -> dict:
        candidates = [item for item in self.model.identities.public() if item["provider"] == provider]
        return {"provider": provider, "success": True, "status": "configured" if candidates else "not_checked",
                "backend": "language-model", "identities": candidates,
                **({} if candidates else {"error": f"No configured {provider} identity."})}

    def close(self):
        self.model.close()
