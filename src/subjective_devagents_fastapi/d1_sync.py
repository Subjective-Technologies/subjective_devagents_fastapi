"""Mirror local DevAgents conversations into the `devagents` D1 schema.

Incremental: `import_sources.fingerprint` skips unchanged transcripts so a history
refresh does not rewrite tens of thousands of rows. Failures here must never fail
the local history pipeline — D1 is a replica, not the live store.
"""
from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .d1_client import SqlExecutor

NAMESPACE = uuid.UUID("2348859b-1168-4ba9-81ba-044442898904")

PROVIDER_ALIASES = {
    "codex": "codex",
    "chatgpt": "codex",
    "gpt": "codex",
    "openai": "codex",
    "claude": "claude",
    "claude-code": "claude",
    "anthropic": "claude",
    "cursor": "cursor",
    "gemini": "gemini",
    "google": "gemini",
    "grok": "grok",
    "xai": "grok",
    "ssh": "ssh",
}

KNOWN_PROVIDERS = {"codex", "claude", "cursor", "anthropic", "openai", "gemini", "grok", "ssh"}
ALLOWED_ROLES = {"user", "assistant", "system", "tool", "tool_output"}


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, ":".join(str(part) for part in parts)))


def iso_from_ms(value: Any) -> str:
    try:
        millis = float(value or 0)
    except (TypeError, ValueError):
        millis = 0
    if millis > 1e12:
        millis = millis / 1000.0
    if millis <= 0:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.fromtimestamp(millis, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def map_provider(raw: Any) -> str | None:
    key = str(raw or "").strip().lower()
    mapped = PROVIDER_ALIASES.get(key, key)
    if mapped in KNOWN_PROVIDERS:
        return mapped
    return None


def map_role(raw: Any) -> str:
    role = str(raw or "user").strip().lower()
    if role in ALLOWED_ROLES:
        return role
    if role in {"human"}:
        return "user"
    if role in {"bot", "model", "ai"}:
        return "assistant"
    return "user"


def source_uri_for(conversation: dict) -> str:
    return str(
        conversation.get("sourceFile")
        or conversation.get("exportPath")
        or conversation.get("historyExportPath")
        or conversation.get("target")
        or ""
    )


def fingerprint_for(conversation: dict) -> str:
    export = str(conversation.get("exportPath") or conversation.get("historyExportPath") or "")
    stat = ""
    path = Path(export) if export else None
    if path and path.is_file():
        info = path.stat()
        stat = f"{info.st_mtime_ns}:{info.st_size}"
    return "|".join(
        [
            str(conversation.get("messageCount") or len(conversation.get("messages") or [])),
            str(int(float(conversation.get("updatedAt") or 0))),
            str(conversation.get("sessionId") or ""),
            stat,
        ]
    )


def load_export_messages(conversation: dict) -> list[dict]:
    messages = conversation.get("messages") if isinstance(conversation.get("messages"), list) else []
    if messages:
        return messages
    export = str(conversation.get("exportPath") or conversation.get("historyExportPath") or "")
    if not export or not Path(export).is_file():
        return []
    try:
        payload = json.loads(Path(export).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw = payload.get("messages") if isinstance(payload, dict) else None
    return raw if isinstance(raw, list) else []


def conversations_for_sync(llm_service: Any) -> list[dict]:
    listed = llm_service.list_conversations(include_messages=False).get("conversations") or []
    out: list[dict] = []
    for conversation in listed:
        full = dict(conversation)
        full["messages"] = load_export_messages(conversation)
        if not full["messages"]:
            found = llm_service.find_conversation(conversation["target"], include_messages=True)
            if found and found.get("messages"):
                full["messages"] = found["messages"]
        out.append(full)
    return out


def _one(executor: SqlExecutor, sql: str, params: Iterable | None = None) -> dict[str, Any] | None:
    rows = executor.query(sql, params)
    return rows[0] if rows else None


def sync_conversations(
    executor: SqlExecutor,
    conversations: list[dict],
    *,
    host: str | None = None,
) -> dict[str, int]:
    host = host or os.environ.get("SUBJECTIVE_DEVAGENTS_HOST") or socket.gethostname()
    stamp = now_iso()
    stats = {"seen": 0, "upserted": 0, "skipped": 0, "failed": 0, "messages": 0}
    for conversation in conversations:
        stats["seen"] += 1
        try:
            if _sync_one(executor, conversation, host=host, stamp=stamp):
                stats["upserted"] += 1
                stats["messages"] += len(conversation.get("messages") or [])
            else:
                stats["skipped"] += 1
        except Exception as error:  # noqa: BLE001 — one bad chat must not abort the replica
            stats["failed"] += 1
            if stats["failed"] == 1:
                stats["first_error"] = f"{type(error).__name__}: {error}"[:400]
    return stats


def _sync_one(executor: SqlExecutor, conversation: dict, *, host: str, stamp: str) -> bool:
    target = str(conversation.get("target") or "").strip()
    if not target:
        return False
    provider = map_provider(conversation.get("provider"))
    if not provider:
        return False
    uri = source_uri_for(conversation)
    fingerprint = fingerprint_for(conversation)
    existing = _one(executor, "SELECT fingerprint, import_status FROM import_sources WHERE source_uri = ?", (uri,)) if uri else None
    if existing and existing.get("fingerprint") == fingerprint and existing.get("import_status") == "imported":
        return False

    messages = conversation.get("messages") if isinstance(conversation.get("messages"), list) else []
    conversation_id = stable_id("conv", target)
    session_key = str(conversation.get("sessionId") or "")
    session_id = stable_id("sess", conversation_id, provider, session_key or "none")
    turn_id = stable_id("turn", session_id, "1")
    origin = "imported" if conversation.get("origin") == "history" else (
        conversation.get("origin") if conversation.get("origin") in {"local", "imported", "api", "web"} else "local"
    )
    title = str(conversation.get("label") or conversation.get("title") or f"{provider} conversation")[:500]
    created_at = iso_from_ms(conversation.get("createdAt"))
    updated_at = iso_from_ms(conversation.get("updatedAt") or conversation.get("sortAt"))
    first_at = iso_from_ms(messages[0].get("time") if messages else conversation.get("createdAt"))
    os_user = conversation.get("osUser") or conversation.get("sourceUser")
    message_count = int(conversation.get("messageCount") or len(messages) or 0)

    executor.execute(
        """
        INSERT INTO conversations (
          id, target, title, title_source, os_user, workspace, current_session_id,
          current_provider_id, origin, status, irc_channel, message_count, turn_count,
          first_message_at, last_activity_at, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, 'derived', ?, ?, ?, ?, ?, 'active', ?, ?, 1, ?, ?, ?, ?, ?)
        ON CONFLICT(target) DO UPDATE SET
          title = excluded.title,
          os_user = excluded.os_user,
          workspace = excluded.workspace,
          current_session_id = excluded.current_session_id,
          current_provider_id = excluded.current_provider_id,
          origin = excluded.origin,
          irc_channel = excluded.irc_channel,
          message_count = excluded.message_count,
          last_activity_at = excluded.last_activity_at,
          metadata_json = excluded.metadata_json,
          updated_at = excluded.updated_at
        """,
        (
            conversation_id,
            target,
            title,
            os_user,
            conversation.get("workspace"),
            session_id,
            provider,
            origin,
            conversation.get("ircChannel"),
            message_count,
            first_at,
            updated_at,
            json.dumps({"sourceFile": conversation.get("sourceFile"), "exportPath": conversation.get("exportPath")}, default=str),
            created_at,
            stamp,
        ),
    )

    executor.execute(
        """
        INSERT INTO conversation_sessions (
          id, conversation_id, seq, provider_id, provider_session_id, start_reason,
          os_user, workspace, source_file, status, started_at, metadata_json, created_at, updated_at
        ) VALUES (?, ?, 1, ?, ?, 'import', ?, ?, ?, 'open', ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          provider_session_id = excluded.provider_session_id,
          os_user = excluded.os_user,
          workspace = excluded.workspace,
          source_file = excluded.source_file,
          updated_at = excluded.updated_at
        """,
        (
            session_id,
            conversation_id,
            provider,
            session_key or None,
            os_user,
            conversation.get("workspace"),
            conversation.get("sourceFile"),
            created_at,
            json.dumps({"origin": conversation.get("origin")}, default=str),
            created_at,
            stamp,
        ),
    )

    executor.execute(
        """
        INSERT INTO turns (
          id, conversation_id, session_id, seq, initiator, status,
          started_at, ended_at, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, 1, 'user', 'completed', ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          ended_at = excluded.ended_at,
          updated_at = excluded.updated_at
        """,
        (turn_id, conversation_id, session_id, created_at, updated_at, json.dumps({"imported": True}), created_at, stamp),
    )

    executor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    rows: list[tuple] = []
    for index, message in enumerate(messages, start=1):
        role = map_role(message.get("role"))
        content = str(message.get("text") or message.get("content") or message.get("command") or "")
        sent_at = iso_from_ms(message.get("time") or message.get("timestamp") or conversation.get("updatedAt"))
        metadata = {key: message.get(key) for key in ("kind", "toolName", "workdir", "shell") if message.get(key)}
        if message.get("sendFailed"):
            metadata["sendFailed"] = True
        rows.append(
            (
                stable_id("msg", turn_id, str(index)),
                conversation_id,
                session_id,
                turn_id,
                index,
                role,
                content,
                "You" if role == "user" else provider,
                1 if message.get("sendFailed") else 0,
                str(message.get("error") or "") or None,
                sent_at,
                json.dumps(metadata, default=str) if metadata else None,
                sent_at,
                stamp,
            )
        )
    # D1 caps bound parameters around 100; 14 columns => 6 rows per insert.
    insert_sql = """
        INSERT INTO messages (
          id, conversation_id, session_id, turn_id, seq, role, content, content_format,
          author_label, send_failed, error_message, sent_at, metadata_json, created_at, updated_at
        ) VALUES {values}
    """
    chunk_size = 6
    for offset in range(0, len(rows), chunk_size):
        chunk = rows[offset : offset + chunk_size]
        values = ",".join(["(?, ?, ?, ?, ?, ?, ?, 'markdown', ?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
        params: list[Any] = []
        for row in chunk:
            params.extend(row)
        executor.execute(insert_sql.format(values=values), params)

    if uri:
        executor.execute(
            """
            INSERT INTO import_sources (
              id, conversation_id, session_id, provider_id, source_kind, source_uri,
              source_user, host, fingerprint, message_count, last_activity_at,
              imported_at, import_status, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'file', ?, ?, ?, ?, ?, ?, ?, 'imported', ?, ?, ?)
            ON CONFLICT(source_uri) DO UPDATE SET
              conversation_id = excluded.conversation_id,
              session_id = excluded.session_id,
              provider_id = excluded.provider_id,
              fingerprint = excluded.fingerprint,
              message_count = excluded.message_count,
              last_activity_at = excluded.last_activity_at,
              imported_at = excluded.imported_at,
              import_status = 'imported',
              error_message = NULL,
              updated_at = excluded.updated_at
            """,
            (
                stable_id("src", uri),
                conversation_id,
                session_id,
                provider,
                uri,
                os_user,
                host,
                fingerprint,
                message_count,
                updated_at,
                stamp,
                json.dumps({"target": target}, default=str),
                stamp,
                stamp,
            ),
        )
    return True
