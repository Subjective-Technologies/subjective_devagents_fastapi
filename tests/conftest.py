"""Shared fixtures for the devagents schema tests.

D1 is SQLite, so the schema is exercised against a real in-memory SQLite connection with
foreign keys enforced — the same enforcement D1 applies. Nothing here talks to Cloudflare.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = SERVICE_ROOT / "schema" / "devagents.sql"
SEED_PATH = SERVICE_ROOT / "seeds" / "devagents_seed.sql"

NOW = "2026-09-06T12:00:00Z"


@pytest.fixture()
def db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    # executescript commits and resets pragmas, so enforce keys after loading the schema.
    connection.execute("PRAGMA foreign_keys = ON")
    yield connection
    connection.close()


@pytest.fixture()
def seeded(db: sqlite3.Connection) -> sqlite3.Connection:
    db.executescript(SEED_PATH.read_text(encoding="utf-8"))
    db.execute("PRAGMA foreign_keys = ON")
    return db


def new_id() -> str:
    return str(uuid.uuid4())


def insert(connection: sqlite3.Connection, table: str, **columns) -> str:
    """Insert a row, filling the timestamp columns every table shares."""
    columns.setdefault("id", new_id())
    for column in ("created_at", "updated_at"):
        if _has_column(connection, table, column):
            columns.setdefault(column, NOW)
    names = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(f"INSERT INTO {table} ({names}) VALUES ({placeholders})", tuple(columns.values()))
    return columns["id"]


def _has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row["name"] == column for row in connection.execute(f"PRAGMA table_info({table})"))


def make_conversation(connection: sqlite3.Connection, **overrides) -> str:
    return insert(
        connection,
        "conversations",
        title=overrides.pop("title", "Test chat"),
        last_activity_at=overrides.pop("last_activity_at", NOW),
        **overrides,
    )


def make_session(connection: sqlite3.Connection, conversation_id: str, **overrides) -> str:
    return insert(
        connection,
        "conversation_sessions",
        conversation_id=conversation_id,
        seq=overrides.pop("seq", 1),
        provider_id=overrides.pop("provider_id", "claude"),
        started_at=overrides.pop("started_at", NOW),
        **overrides,
    )


def make_turn(connection: sqlite3.Connection, conversation_id: str, session_id: str, **overrides) -> str:
    return insert(
        connection,
        "turns",
        conversation_id=conversation_id,
        session_id=session_id,
        seq=overrides.pop("seq", 1),
        started_at=overrides.pop("started_at", NOW),
        **overrides,
    )


@pytest.fixture(autouse=True)
def explicit_test_auth_mode(monkeypatch):
    """Credential-free unit fixtures opt into development mode, never production."""
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "development")


@pytest.fixture(autouse=True)
def no_implicit_live_model_calls(monkeypatch):
    """A stale mock must fail loudly rather than consume a real CLI subscription."""
    import os
    from subjective_devagents_fastapi.language_models.transports import LocalProcess
    if os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_LIVE_URL"):
        return
    original = LocalProcess.run
    def guarded(self, command, **kwargs):
        binaries = {"codex", "claude", "cursor-agent", "grok"}
        if any(Path(arg).name in binaries for arg in command.argv):
            raise AssertionError("Real model invocation requires the explicit live acceptance gate")
        return original(self, command, **kwargs)
    monkeypatch.setattr(LocalProcess, "run", guarded)
