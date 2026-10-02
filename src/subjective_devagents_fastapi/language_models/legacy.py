"""Compatibility boundary for existing chat routes; execution belongs to LanguageModel."""
from __future__ import annotations

import hashlib
import json
import os
import pwd
import shutil
import threading
import time
import uuid
from pathlib import Path

from .contracts import Identity, Message, ModelError
from .model import LanguageModel
from .registry import IdentityRegistry
from .store import TurnStore, TERMINAL


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
            if item.get("attachments"):
                raise ModelError("unsupported_history", "Conversation contains attachments; explicit migration is required.")
            role = item.get("role")
            if role not in {"user", "assistant"}:
                raise ModelError("unsupported_history", "Conversation contains non-text turns; explicit migration is required.")
            messages.append(Message(role, str(item.get("text") or item.get("content") or "")))
        return tuple(messages)

    def send(self, conversation: dict, source: dict, request: dict, register, clear, progress) -> dict:
        try:
            if request.get("files"):
                raise ModelError("unsupported_attachments", "This provider driver does not support attachments yet.")
            with self._lock:
                identity = self.identity(conversation)
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
                turn = self.model.submit(row[0], request["prompt"], str(uuid.uuid4()), request["timeout"])
            register(conversation["target"], lambda: bool(self.model.cancel(turn["id"])))
            cursor = 0
            try:
                while True:
                    events = self.model.store.events(turn["id"], cursor)
                    for event in events:
                        cursor = event["sequence"]
                        if event["kind"] in {"message", "text_delta"} and progress:
                            progress({"target": conversation["target"], "text": event["data"].get("text", "")})
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
