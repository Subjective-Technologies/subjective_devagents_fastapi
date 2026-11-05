"""Conversation replica: local JSON chats upsert into the devagents schema."""
from __future__ import annotations

import sqlite3

import pytest
from pathlib import Path

from conftest import SCHEMA_PATH, SEED_PATH

from subjective_devagents_fastapi.d1_client import SqliteExecutor
from subjective_devagents_fastapi.d1_sync import fingerprint_for, map_provider, sync_conversations

SERVICE_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def isolated_d1_namespace(monkeypatch):
    """The in-memory replica must not depend on a production database ID."""
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_D1_NAMESPACE", "00000000-0000-4000-8000-000000000001")


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.executescript(SEED_PATH.read_text(encoding="utf-8"))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _chat(**overrides) -> dict:
    base = {
        "target": "llm:claude:abc123",
        "label": "Plan the sync",
        "provider": "claude",
        "sessionId": "2f0f9c2e-1111-2222-3333-444455556666",
        "sourceFile": "/home/goldenthinker/.claude/projects/a.jsonl",
        "exportPath": "/tmp/export.json",
        "origin": "history",
        "osUser": "goldenthinker",
        "workspace": "/subjective",
        "ircChannel": "#plan-the-sync",
        "messageCount": 2,
        "createdAt": 1_760_000_000_000,
        "updatedAt": 1_760_000_100_000,
        "messages": [
            {"role": "user", "text": "hello", "time": 1_760_000_000_000},
            {"role": "assistant", "text": "hi", "time": 1_760_000_100_000},
        ],
    }
    base.update(overrides)
    return base


def test_provider_aliases():
    assert map_provider("Claude") == "claude"
    assert map_provider("chatgpt") == "codex"
    assert map_provider("not-a-vendor") is None


def test_sync_upserts_conversation_session_messages_and_import_source():
    db = _db()
    stats = sync_conversations(SqliteExecutor(db), [_chat()], host="subjective-agents")
    assert stats == {"seen": 1, "upserted": 1, "skipped": 0, "failed": 0, "messages": 2}

    row = db.execute("SELECT target, title, origin, current_provider_id, message_count FROM conversations").fetchone()
    assert row["target"] == "llm:claude:abc123"
    assert row["title"] == "Plan the sync"
    assert row["origin"] == "imported"
    assert row["current_provider_id"] == "claude"
    assert row["message_count"] == 2

    session = db.execute("SELECT provider_id, provider_session_id, start_reason FROM conversation_sessions").fetchone()
    assert session["provider_id"] == "claude"
    assert session["provider_session_id"] == "2f0f9c2e-1111-2222-3333-444455556666"
    assert session["start_reason"] == "import"

    messages = db.execute("SELECT seq, role, content FROM messages ORDER BY seq").fetchall()
    assert [(m["seq"], m["role"], m["content"]) for m in messages] == [
        (1, "user", "hello"),
        (2, "assistant", "hi"),
    ]

    source = db.execute("SELECT source_uri, import_status, host FROM import_sources").fetchone()
    assert source["source_uri"] == "/home/goldenthinker/.claude/projects/a.jsonl"
    assert source["import_status"] == "imported"
    assert source["host"] == "subjective-agents"


def test_sync_is_incremental_on_unchanged_fingerprint():
    db = _db()
    executor = SqliteExecutor(db)
    chat = _chat()
    sync_conversations(executor, [chat], host="h1")
    stats = sync_conversations(executor, [chat], host="h1")
    assert stats["skipped"] == 1
    assert stats["upserted"] == 0
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_sync_rewrites_messages_when_transcript_grows():
    db = _db()
    executor = SqliteExecutor(db)
    sync_conversations(executor, [_chat()], host="h1")
    grown = _chat(messageCount=3, updatedAt=1_760_000_200_000, messages=[
        {"role": "user", "text": "hello", "time": 1_760_000_000_000},
        {"role": "assistant", "text": "hi", "time": 1_760_000_100_000},
        {"role": "user", "text": "more", "time": 1_760_000_200_000},
    ])
    assert fingerprint_for(grown) != fingerprint_for(_chat())
    stats = sync_conversations(executor, [grown], host="h1")
    assert stats["upserted"] == 1
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    assert db.execute("SELECT message_count FROM conversations").fetchone()[0] == 3


def test_unknown_provider_is_skipped_not_inserted():
    db = _db()
    stats = sync_conversations(SqliteExecutor(db), [_chat(provider="mystery")], host="h1")
    assert stats["upserted"] == 0
    assert db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
