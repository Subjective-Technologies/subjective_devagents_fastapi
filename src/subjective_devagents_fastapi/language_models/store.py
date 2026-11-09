"""SQLite turn journal: durable acceptance, idempotency and conversation ownership.

No automatic replay or lease stealing. An interrupted worker can have a surviving
CLI or remote job; explicit operator reconciliation is required before releasing it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts import Identity, Message, ModelError, ModelResult

ACTIVE = ("queued", "running")
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}


class TurnStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS channel_bindings (target TEXT PRIMARY KEY, channel_id TEXT NOT NULL, actor_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, identity TEXT NOT NULL,
                    session_id TEXT, created REAL NOT NULL, needs_reconciliation INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    request_id TEXT NOT NULL, identity TEXT NOT NULL, prompt TEXT NOT NULL,
                    timeout REAL NOT NULL, status TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
                    session_id TEXT, error_code TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL, updated REAL NOT NULL,
                    UNIQUE(conversation_id, request_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_turn ON turns(conversation_id)
                    WHERE status IN ('queued', 'running');
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id TEXT NOT NULL REFERENCES turns(id), kind TEXT NOT NULL,
                    data TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS handoffs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                    previous_identity TEXT NOT NULL, next_identity TEXT NOT NULL,
                    previous_session_id TEXT, created REAL NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _event(db, turn_id: str, kind: str, data: dict):
        db.execute("INSERT INTO events(turn_id,kind,data,created) VALUES(?,?,?,?)",
                   (turn_id, kind, json.dumps(data), time.time()))

    def event(self, turn_id: str, kind: str, data: dict):
        with self.connection() as db:
            self._event(db, turn_id, kind, data)

    def create(self, identity: Identity, history: tuple[Message, ...] = ()) -> dict:
        identifier = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("INSERT INTO conversations(id,identity,session_id,created) VALUES(?,?,NULL,?)",
                       (identifier, json.dumps(identity.snapshot(), sort_keys=True), time.time()))
            # Imports use the same completed-turn representation, without native sessions.
            for index in range(0, len(history), 2):
                if (index + 1 >= len(history) or history[index].role != "user"
                        or history[index + 1].role != "assistant"):
                    raise ModelError("invalid_history", "Import requires complete user/assistant pairs.")
                turn_id = str(uuid.uuid4())
                now = time.time()
                db.execute("""INSERT INTO turns
                    (id,conversation_id,request_id,identity,prompt,timeout,status,text,created,updated)
                    VALUES(?,?,?,?,?,0,'completed',?,?,?)""",
                    (turn_id, identifier, f"import:{index}", json.dumps(identity.snapshot()),
                     history[index].content, history[index + 1].content, now, now))
        return self.conversation(identifier)

    @staticmethod
    def _conversation(db, identifier):
        row = db.execute("SELECT * FROM conversations WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ModelError("not_found", "Conversation does not exist.")
        return dict(row)

    def conversation(self, identifier: str) -> dict:
        with self.connection() as db:
            row = self._conversation(db, identifier)
            row["identity"] = json.loads(row["identity"])
            row["messages"] = [
                {"role": role, "content": turn[column]}
                for turn in db.execute("SELECT prompt,text FROM turns WHERE conversation_id=? "
                                       "AND status='completed' ORDER BY rowid", (identifier,))
                for role, column in (("user", "prompt"), ("assistant", "text"))
            ]
            row["handoffs"] = [dict(item) for item in db.execute(
                "SELECT * FROM handoffs WHERE conversation_id=? ORDER BY id", (identifier,))]
            row["active_turn_id"] = next((item[0] for item in db.execute(
                "SELECT id FROM turns WHERE conversation_id=? AND status IN ('queued','running')",
                (identifier,))), None)
            return row

    def reserve(self, conversation_id: str, identity: Identity, request_id: str,
                prompt: str, timeout: float) -> tuple[dict, bool]:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            conversation = self._conversation(db, conversation_id)
            previous = db.execute("SELECT * FROM turns WHERE conversation_id=? AND request_id=?",
                                  (conversation_id, request_id)).fetchone()
            if previous:
                if previous["prompt"] != prompt or previous["timeout"] != timeout:
                    raise ModelError("idempotency_conflict", "Request id already names a different request.")
                return self._decode(previous), False
            if conversation["needs_reconciliation"]:
                raise ModelError("reconciliation_required", "Previous execution outcome is unknown; reconcile it first.")
            if db.execute("SELECT count(*) FROM turns WHERE status IN ('queued','running')").fetchone()[0] >= 64:
                raise ModelError("overloaded", "Too many accepted turns; try again later.")
            snapshot = json.dumps(identity.snapshot(), sort_keys=True)
            if conversation["identity"] != snapshot:
                raise ModelError("identity_changed", "Identity configuration changed; create an explicit handoff.")
            # Serializing a CLI account also prevents concurrent OAuth refreshes.
            # BEGIN IMMEDIATE makes this admission check atomic across API workers.
            occupied = db.execute("SELECT conversation_id,identity,status FROM turns WHERE "
                                  "status IN ('queued','running') OR "
                                  "(status='interrupted' AND COALESCE(error_code,'') != 'operator_reconciled')")
            for active in occupied:
                bound = json.loads(active["identity"])
                same_account = bound["id"] == identity.id or (
                    identity.strategy == "cli" and bound.get("strategy") == "cli"
                    and all(bound.get(key) == getattr(identity, key)
                            for key in ("provider", "ssh_host", "home", "config_dir")))
                if active["conversation_id"] == conversation_id and active["status"] in ACTIVE:
                    raise ModelError("busy", "Conversation already has an active turn.")
                # Account-wide serialisation and the unknown-outcome lock protect a CLI's own
                # session/credential files. An HTTP identity (api, api_agent) has neither, and
                # every dashboard agent of a provider shares one: an interrupted turn in one
                # conversation must not lock all of them.
                if same_account and identity.strategy == "cli":
                    if active["status"] == "interrupted":
                        raise ModelError("reconciliation_required", "This account has an unresolved execution outcome.")
                    raise ModelError("identity_busy", "This CLI account is executing another conversation.")
            identifier, now = str(uuid.uuid4()), time.time()
            try:
                db.execute("""INSERT INTO turns
                    (id,conversation_id,request_id,identity,prompt,timeout,status,session_id,created,updated)
                    VALUES(?,?,?,?,?,?,'queued',?,?,?)""",
                    (identifier, conversation_id, request_id, snapshot, prompt, timeout,
                     conversation["session_id"], now, now))
            except sqlite3.IntegrityError as exc:
                raise ModelError("busy", "Conversation already has an active turn.") from exc
            self._event(db, identifier, "state", {"status": "queued"})
            return self._decode(db.execute("SELECT * FROM turns WHERE id=?", (identifier,)).fetchone()), True

    @staticmethod
    def _decode(row):
        if row is None:
            raise ModelError("not_found", "Turn does not exist.")
        result = dict(row)
        result["identity"] = json.loads(result["identity"])
        return result

    def turn(self, identifier: str) -> dict:
        with self.connection() as db:
            return self._decode(db.execute("SELECT * FROM turns WHERE id=?", (identifier,)).fetchone())

    def claim(self, identifier: str) -> bool:
        with self.connection() as db:
            claimed = db.execute("UPDATE turns SET status='running',updated=? WHERE id=? AND status='queued'",
                                 (time.time(), identifier)).rowcount == 1
            if claimed:
                self._event(db, identifier, "state", {"status": "running"})
            return claimed

    def finish(self, identifier: str, result: ModelResult):
        if result.status not in TERMINAL:
            raise ValueError("Invalid terminal state")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._decode(db.execute("SELECT * FROM turns WHERE id=?", (identifier,)).fetchone())
            if row["status"] not in ACTIVE:
                return
            db.execute("UPDATE turns SET status=?,text=?,session_id=?,error_code=?,updated=? WHERE id=?",
                       (result.status, result.text, result.session_id, result.error_code, time.time(), identifier))
            # Failed/ambiguous turns may have mutated native history. Never resume it implicitly.
            db.execute("UPDATE conversations SET session_id=?,needs_reconciliation=? WHERE id=?",
                       (result.session_id if result.status == "completed" else None,
                        int(result.status == "interrupted"), row["conversation_id"]))
            self._event(db, identifier, "state", {"status": result.status, "error_code": result.error_code})

    def cancel(self, identifier: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._decode(db.execute("SELECT * FROM turns WHERE id=?", (identifier,)).fetchone())
            if row["status"] in ACTIVE:
                db.execute("UPDATE turns SET cancel_requested=1,updated=? WHERE id=?", (time.time(), identifier))
                self._event(db, identifier, "cancel_requested", {})
        return self.turn(identifier)

    def handoff(self, identifier: str, identity: Identity) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._conversation(db, identifier)
            if previous["needs_reconciliation"]:
                raise ModelError("reconciliation_required", "Reconcile the previous execution before switching identity.")
            if db.execute("SELECT 1 FROM turns WHERE conversation_id=? AND status IN ('queued','running')",
                          (identifier,)).fetchone():
                raise ModelError("busy", "Finish or cancel the active turn before switching identity.")
            snapshot = json.dumps(identity.snapshot(), sort_keys=True)
            db.execute("INSERT INTO handoffs(conversation_id,previous_identity,next_identity,previous_session_id,created) "
                       "VALUES(?,?,?,?,?)", (identifier, previous["identity"], snapshot, previous["session_id"], time.time()))
            db.execute("UPDATE conversations SET identity=?,session_id=NULL WHERE id=?", (snapshot, identifier))
        return self.conversation(identifier)

    def reconcile(self, identifier: str) -> dict:
        """Operator has independently verified the old worker/remote job is stopped."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._decode(db.execute("SELECT * FROM turns WHERE id=?", (identifier,)).fetchone())
            if row["status"] not in {*ACTIVE, "interrupted"}:
                raise ModelError("invalid_state", "Only abandoned or ambiguous turns need reconciliation.")
            newest = db.execute("SELECT id FROM turns WHERE conversation_id=? ORDER BY rowid DESC LIMIT 1",
                                (row["conversation_id"],)).fetchone()[0]
            if newest != identifier:
                raise ModelError("invalid_state", "Only the latest turn can be reconciled.")
            db.execute("UPDATE turns SET status='interrupted',error_code='operator_reconciled',updated=? WHERE id=?",
                       (time.time(), identifier))
            db.execute("UPDATE conversations SET needs_reconciliation=0,session_id=NULL WHERE id=?",
                       (row["conversation_id"],))
            self._event(db, identifier, "reconciled", {"status": "interrupted"})
        return self.turn(identifier)

    def events(self, identifier: str, after: int = 0) -> list[dict]:
        self.turn(identifier)
        with self.connection() as db:
            return [{**dict(row), "data": json.loads(row["data"])} for row in db.execute(
                "SELECT * FROM events WHERE turn_id=? AND sequence>? ORDER BY sequence LIMIT 1000", (identifier, after))]
