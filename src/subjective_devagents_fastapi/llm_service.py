"""LLM agent conversations: history archive, local sends via CLI pipeline bundles.

Python port of clients/desktop/subjective_devagents/src/llmService.js. Synchronous —
route handlers call it via asyncio.to_thread. State file layout, target format, IRC
channel naming and the export/manifest formats are byte-compatible with the Node
implementation so both backends can share the same SUBJECTIVE_CHAT_HOME.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .cursor_inject import inject_into_cursor_dom
from .provider_sessions import list_provider_sessions, sessions_for_provider
from .system_users import current_username

PROVIDER_ICONS = {
    "codex": "../assets/providers/codex.svg",
    "cursor": "../assets/providers/cursor.svg",
    "claude": "../assets/providers/claude.svg",
    "anthropic": "../assets/providers/claude.svg",
    "anthropic-claude": "../assets/providers/claude.svg",
    "gemini": "../assets/providers/gemini.svg",
    "google-gemini": "../assets/providers/gemini.svg",
    "grok": "../assets/providers/grok.svg",
    "xai": "../assets/providers/grok.svg",
    "chatgpt": "../assets/providers/chatgpt.svg",
    "chat-gpt": "../assets/providers/chatgpt.svg",
    "openai": "../assets/providers/chatgpt.svg",
    "openai-chat-gpt": "../assets/providers/chatgpt.svg",
}

PROVIDER_LABELS = {
    "codex": "Codex",
    "cursor": "Cursor",
    "claude": "Claude",
    "anthropic": "Claude",
    "anthropic-claude": "Claude",
    "gemini": "Gemini",
    "google-gemini": "Gemini",
    "grok": "Grok",
    "xai": "Grok",
    "chatgpt": "ChatGPT",
    "chat-gpt": "ChatGPT",
    "openai": "ChatGPT",
    "openai-chat-gpt": "ChatGPT",
}

STATE_VERSION = 2

# Providers with an exported CLI datasource bundle that can create and resume sessions.
WRITABLE_PROVIDERS = {
    "codex": {"bundle": "codex", "label": "Codex"},
    "claude": {"bundle": "claude", "label": "Claude"},
    "cursor": {"bundle": "cursor", "label": "Cursor"},
}
DEFAULT_PROVIDER = "codex"

# A single agent turn can legitimately run far longer than the datasource's own
# 10-minute default — the per-turn timeout rides in the request, and the outer process
# wrapper gets slightly more so the datasource's own timeout reports cleanly first.
AGENT_TURN_TIMEOUT_SECONDS = 1800
AGENT_TURN_WRAPPER_TIMEOUT_SECONDS = AGENT_TURN_TIMEOUT_SECONDS + 60
CURSOR_INJECT_POLL_MS = int(os.environ.get("CURSOR_INJECT_POLL_MS") or 2000)
CURSOR_INJECT_POLL_MAX_MS = int(os.environ.get("CURSOR_INJECT_POLL_MAX_MS") or AGENT_TURN_TIMEOUT_SECONDS * 1000)

HYDRATE_MESSAGE_LIMIT = 80
HANDOFF_TRANSCRIPT_MAX_CHARS = 20000

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)

ENVELOPE_TAG_NAMES = [
    "timestamp",
    "user_query",
    "user_info",
    "user_instructions",
    "system_notification",
    "system-reminder",
    "system_reminder",
    "environment_context",
    "git_status",
    "open_and_recently_viewed_files",
    "agent_skills",
    "image_files",
    "external_links",
    "attached_files",
]
ENVELOPE_TAG_RE = re.compile(r"<(?:%s)_?\b" % "|".join(ENVELOPE_TAG_NAMES), re.IGNORECASE)
TASK_BLOCK_RE = re.compile(r"<task_?\b[^>]*>([\s\S]*?)</task_?>", re.IGNORECASE)
TASK_TITLE_RE = re.compile(r"^\s*title:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
INTERNAL_QUERY_RE = re.compile(
    r"^(?:briefly inform the user about the task result\b|the (?:above|following) task (?:has finished|is complete)\b)",
    re.IGNORECASE,
)
ENVELOPE_BLOCK_RE = re.compile(
    r"<(%s)_?\b[^>]*>[\s\S]*?</\1_?>" % "|".join(ENVELOPE_TAG_NAMES),
    re.IGNORECASE,
)
USER_QUERY_RE = re.compile(r"<user_query>\s*([\s\S]*?)\s*</user_query>", re.IGNORECASE)


def provider_key(provider: Any) -> str:
    return re.sub(r"[\s_]+", "-", str(provider or "unknown").strip().lower())


def provider_icon(provider: Any) -> str:
    return PROVIDER_ICONS.get(provider_key(provider), "")


def provider_label(provider: Any) -> str:
    return PROVIDER_LABELS.get(provider_key(provider)) or str(provider or "Unknown")


def writable_provider(provider: Any) -> dict | None:
    return WRITABLE_PROVIDERS.get(provider_key(provider))


def epoch_ms(value: Any) -> float:
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return value if value > 1e12 else value * 1000
    text = str(value)
    # ISO 8601 first, epoch-in-a-string second.
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000
    except ValueError:
        pass
    try:
        return epoch_ms(float(text))
    except ValueError:
        return 0


def safe_json(file_path: str | Path) -> Any:
    try:
        with open(file_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def normalize_role(role: Any) -> str:
    value = str(role or "").lower()
    if value in ("assistant", "agent"):
        return "assistant"
    if value in ("user", "human"):
        return "user"
    if value == "tool":
        return "tool"
    if value in ("tool_output", "output"):
        return "tool_output"
    return "system"


def latest_message_time(messages: list[dict]) -> float:
    latest = 0.0
    for message in messages:
        latest = max(latest, epoch_ms(message.get("time") or message.get("timestamp")))
    return latest


def export_filename_time(file_path: Any) -> float:
    match = re.match(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_", os.path.basename(str(file_path or "")))
    if not match:
        return 0
    from datetime import datetime, timezone

    year, month, day, hour, minute, second = (int(part) for part in match.groups())
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp() * 1000


def looks_wrapped(text: Any) -> bool:
    return bool(ENVELOPE_TAG_RE.search(str(text or "")))


def extract_task_title(text: Any) -> str:
    source = str(text or "")
    block = TASK_BLOCK_RE.search(source)
    body = block.group(1) if block else source
    title = TASK_TITLE_RE.search(body)
    return title.group(1).strip() if title else ""


def is_agent_internal_query(text: Any) -> bool:
    return bool(INTERNAL_QUERY_RE.match(re.sub(r"\s+", " ", str(text or "")).strip()))


def extract_user_queries(text: Any) -> list[str]:
    return [match.strip() for match in USER_QUERY_RE.findall(str(text or "")) if match.strip()]


def strip_envelope_blocks(text: Any) -> str:
    return ENVELOPE_BLOCK_RE.sub(" ", str(text or "")).strip()


def clean_visible_message(raw: Any, role: Any = "user") -> dict | None:
    original_role = normalize_role(role)
    source = str(raw or "").strip()
    if not source:
        return None
    if not looks_wrapped(source):
        return {"role": original_role, "text": source}

    queries = extract_user_queries(source)
    if queries:
        visible = [query for query in queries if not is_agent_internal_query(query)]
        if not visible:
            title = extract_task_title(source)
            return {"role": "system", "text": f"Task finished: {title}"} if title else None
        return {"role": original_role, "text": "\n\n".join(visible)}

    stripped = strip_envelope_blocks(source)
    stripped = re.sub(r"^\s*</?[a-z][\w:-]*_?\s*>\s*$", "", stripped, flags=re.IGNORECASE | re.MULTILINE).strip()
    if not stripped or is_agent_internal_query(stripped):
        title = extract_task_title(source)
        return {"role": "system", "text": f"Task finished: {title}"} if title else None
    return {"role": original_role, "text": stripped}


def is_generic_provider_label(label: Any, provider: Any) -> bool:
    text = str(label or "").strip()
    if not text:
        return True
    if re.match(r"^(cursor|codex|claude|chatgpt|gemini|grok|chat)\s+conversation$", text, re.IGNORECASE):
        return True
    name = str(provider or "").strip()
    return bool(name) and bool(re.match(rf"^{re.escape(name)}\s+conversation$", text, re.IGNORECASE))


def prefer_conversation_label(existing: Any, incoming: Any, provider: Any) -> str:
    nxt = str(incoming or "").strip()
    prev = str(existing or "").strip()
    if not nxt:
        return prev
    if prev and is_generic_provider_label(nxt, provider) and not is_generic_provider_label(prev, provider):
        return prev
    return nxt


def tail_conversation_for_client(conversation: dict | None, limit: int = HYDRATE_MESSAGE_LIMIT) -> dict | None:
    if not conversation or not isinstance(conversation.get("messages"), list):
        return conversation
    total = len(conversation["messages"])
    if total <= limit:
        return conversation
    return {
        **conversation,
        "messages": conversation["messages"][-limit:],
        "messageCount": int(conversation.get("messageCount") or total),
        "truncated": True,
        "hiddenBefore": total - limit,
    }


def clean_chat_title(raw: Any) -> str:
    cleaned = clean_visible_message(raw, "user")
    text = cleaned["text"] if cleaned and cleaned["role"] != "system" else str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<[^>]*$", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:80]


def clip_text(value: Any, maximum: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= maximum:
        return text
    return f"{text[:maximum]}…"


def format_transcript_for_handoff(conversation: dict | None) -> str:
    messages = conversation.get("messages") if conversation else None
    messages = messages if isinstance(messages, list) else []
    lines = [
        f"{'Assistant' if message.get('role') == 'assistant' else 'User'}: {str(message.get('text')).strip()}"
        for message in messages
        if message and message.get("text") and message.get("role") in ("user", "assistant")
    ]
    if not lines:
        return ""

    body = "\n\n".join(lines)
    truncated = False
    if len(body) > HANDOFF_TRANSCRIPT_MAX_CHARS:
        body = body[-HANDOFF_TRANSCRIPT_MAX_CHARS:]
        truncated = True

    from_label = provider_label(conversation.get("provider"))
    as_user = f" as {conversation['sourceUser']}" if conversation.get("sourceUser") else ""
    suffix = " (truncated to the most recent messages)" if truncated else ""
    return "\n\n".join(
        [
            f"Continuing a prior conversation from {from_label}{as_user}. Here is the transcript so far{suffix}:",
            "---",
            body,
            "---",
            "Please continue from here.",
        ]
    )


def uuid_from_path(file_path: Any) -> str:
    match = UUID_RE.search(str(file_path or ""))
    return match.group(0).lower() if match else ""


def composer_id_from_source(file_path: Any) -> str:
    source = str(file_path or "")
    composer = re.search(r"#composer:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", source, re.IGNORECASE)
    if composer:
        return composer.group(1).lower()
    if "agent-transcripts" in source:
        return uuid_from_path(source)
    return ""


def session_identity(item: dict) -> str:
    session_id = str(item.get("sessionId") or "").strip().lower()
    if session_id:
        return session_id
    from_path = uuid_from_path(item.get("sourceFile") or item.get("source_file") or "")
    if from_path:
        return from_path
    return str(item.get("target") or "")


def conversation_key(item: dict) -> str:
    return f"{item.get('provider')}:{session_identity(item)}"


def source_user_from_path(file_path: Any) -> str:
    match = re.search(r"(?:^|/)home/([^/]+)/", str(file_path or ""))
    user = match.group(1).strip() if match else ""
    if not user or user in (".", ".."):
        return ""
    return user


def resolve_source_user(payload: dict | None, source_file: Any, origin: str) -> str:
    fingerprint = payload.get("fingerprint") if payload and isinstance(payload.get("fingerprint"), dict) else {}
    named = str(
        (payload or {}).get("osUser")
        or (payload or {}).get("sourceUser")
        or (payload or {}).get("source_user")
        or fingerprint.get("source_user")
        or ""
    ).strip()
    if named:
        return named
    from_path = source_user_from_path(source_file or (payload or {}).get("sourceFile") or (payload or {}).get("source_file"))
    if from_path:
        return from_path
    if origin == "local":
        return current_username()
    return ""


def irc_channel_name(label: Any, identity: Any, provider: Any = DEFAULT_PROVIDER) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(label or "chat").lower()).strip("-")[:24] or "chat"
    suffix = hashlib.sha1(str(identity).encode("utf-8")).hexdigest()[:8]
    return f"#{provider_key(provider) or DEFAULT_PROVIDER}-{slug}-{suffix}"


def _b64url(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


class LlmService:
    def __init__(
        self,
        root: str | Path,
        app_home: str | Path | None = None,
        su_session_manager=None,
        python_path: str | None = None,
        log: Callable[..., None] | None = None,
    ):
        self.root = str(root)
        self.app_home = str(
            app_home
            or os.environ.get("SUBJECTIVE_CHAT_HOME")
            or Path.home() / ".subjective_chat"
        )
        self.codex_home = str(Path(self.app_home) / ".subjective_chat_codex")
        self.history_home = str(Path(self.app_home) / ".subjective_chat_history")
        self.context_dir = str(Path(self.history_home) / "context")
        self.state_path = str(Path(self.app_home) / "llm-conversations.json")
        virtualenv_python = Path(self.root) / ".venv" / "bin" / "python"
        self.python_path = (
            python_path
            or os.environ.get("SUBJECTIVE_CHAT_PYTHON")
            or (str(virtualenv_python) if virtualenv_python.exists() else "python3")
        )
        self.codex_runtime = str(Path(self.root) / "pipelines" / "codex" / "run.py")
        self.claude_runtime = str(Path(self.root) / "pipelines" / "claude" / "run.py")
        self.cursor_runtime = str(Path(self.root) / "pipelines" / "cursor" / "run.py")
        self.history_runtime = str(Path(self.root) / "pipelines" / "history" / "run.py")
        self.reload_runtime = str(Path(self.root) / "pipelines" / "history" / "reload_one.py")
        self.pipeline_runtime = str(Path(__file__).parent / "pipeline_runtime.py")
        self.su_session_manager = su_session_manager
        # Call the subjective_*_datasource submodules in-process instead of spawning an
        # exported bundle. Set SUBJECTIVE_DEVAGENTS_USE_BUNDLES=1 to force the old path.
        self.use_datasource_modules = os.environ.get("SUBJECTIVE_DEVAGENTS_USE_BUNDLES") != "1"
        self.log = log or (lambda level, message, details=None: None)
        self.busy: set[str] = set()
        self._history_refresh_deferred = False
        self._history_cache: list[dict] | None = None
        self._sorted_cache: dict | None = None
        self._state_lock = threading.RLock()
        self.on_progress: Callable[[dict], None] | None = None
        Path(self.app_home).mkdir(parents=True, exist_ok=True)
        saved = self._load_state()
        self.local: list[dict] = saved["conversations"]
        self.active_targets: list[str] = list(dict.fromkeys(saved["activeTargets"]))
        # User-assigned chat names, keyed by target. Kept separate from the conversation
        # records so a rename survives history re-imports.
        self.label_overrides: dict[str, str] = saved.get("labelOverrides") or {}
        if saved["migrated"]:
            backup_path = f"{self.state_path}.v1-backup"
            if Path(self.state_path).exists() and not Path(backup_path).exists():
                import shutil

                shutil.copyfile(self.state_path, backup_path)
            self._save_state()

    # --- state -----------------------------------------------------------------------

    def _load_state(self) -> dict:
        payload = safe_json(self.state_path)
        if not payload or not isinstance(payload.get("conversations"), list):
            return {"conversations": [], "activeTargets": [], "migrated": False, "labelOverrides": {}}
        old_version = int(payload.get("version") or 1)
        migrated = False
        conversations = []
        for item in payload["conversations"]:
            if old_version < STATE_VERSION and item and item.get("sourceFile") and item.get("exportPath"):
                messages = item.get("messages") if isinstance(item.get("messages"), list) else []
                imported_count = max(0, int(item.get("messageCount") or 0))
                if imported_count <= len(messages):
                    migrated = True
                    item = {
                        **item,
                        "historyExportPath": item["exportPath"],
                        "messages": messages[imported_count:],
                    }
            conversations.append(item)
        normalized = [
            conversation
            for conversation in (self._normalize_conversation(item, "local") for item in conversations)
            if conversation
        ]
        overrides = payload.get("labelOverrides")
        return {
            "conversations": normalized,
            "activeTargets": [str(target) for target in payload.get("activeTargets") or []],
            "migrated": migrated,
            "labelOverrides": dict(overrides) if isinstance(overrides, dict) else {},
        }

    def _save_state(self) -> None:
        with self._state_lock:
            self._sorted_cache = None
            Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
            temporary = f"{self.state_path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": STATE_VERSION,
                        "conversations": self.local,
                        "activeTargets": self.active_targets[-20:],
                        "labelOverrides": self.label_overrides,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write("\n")
            os.replace(temporary, self.state_path)

    # --- normalization ----------------------------------------------------------------

    def _normalize_conversation(self, payload: dict | None, origin: str = "history", file_path: str = "") -> dict | None:
        if not payload or (origin == "history" and payload.get("type") != "localchat"):
            return None
        provider = str(payload.get("provider") or payload.get("product") or "unknown").lower()
        source_file = str(payload.get("sourceFile") or payload.get("source_file") or file_path or "")
        session_id = str(
            payload.get("sessionId")
            or payload.get("thread_id")
            or payload.get("session_id")
            or uuid_from_path(source_file)
            or ""
        )
        identity = session_id or source_file or str(uuid.uuid4())
        target = str(payload.get("target") or f"llm:{provider}:{_b64url(identity)}")
        messages = []
        for message in payload.get("messages") or []:
            role = normalize_role(message.get("role"))
            command = str(message.get("command") or "")
            cleaned = clean_visible_message(str(message.get("text") or message.get("content") or ""), role)
            if not cleaned and not command:
                continue
            attachments = [
                {
                    "filename": str(file.get("filename") or "pasted-image"),
                    "mime_type": str(file.get("mime_type")),
                    "content": str(file.get("content")),
                }
                for file in (message.get("attachments") if isinstance(message.get("attachments"), list) else [])
                if re.match(r"^image/", str((file or {}).get("mime_type") or ""), re.IGNORECASE) and (file or {}).get("content")
            ]
            record = {
                "role": cleaned["role"] if cleaned else role,
                "kind": str(message.get("kind") or "message"),
                "text": cleaned["text"] if cleaned else command,
                "command": command,
                "shell": str(message.get("shell") or ""),
                "toolName": str(message.get("toolName") or message.get("tool_name") or ""),
                "workdir": str(message.get("workdir") or ""),
                "time": epoch_ms(message.get("time") or message.get("timestamp")) or time.time() * 1000,
            }
            if attachments:
                record["attachments"] = attachments
            if message.get("sendFailed"):
                record["sendFailed"] = True
                record["error"] = str(message.get("error") or "")
            if record["text"] or record["command"]:
                messages.append(record)
        message_updated_at = latest_message_time(messages)
        content_updated_at = max(
            epoch_ms(payload.get("updatedAt") or payload.get("source_mtime") or payload.get("source_timestamp")),
            message_updated_at,
        )
        updated_at = content_updated_at or (
            Path(file_path).stat().st_mtime * 1000 if file_path and Path(file_path).exists() else 0
        )
        workspace = str(payload.get("workspace") or payload.get("cwd") or self.root)
        created_at = (
            epoch_ms(payload.get("createdAt") or payload.get("chat_started_at") or payload.get("chat_started_timestamp"))
            or updated_at
            or time.time() * 1000
        )
        sort_at = (
            epoch_ms(payload.get("sortAt") or payload.get("sort_at") or payload.get("last_activity_at") or payload.get("lastActivityAt"))
            or message_updated_at
            or updated_at
            or created_at
        )
        raw_label = str(payload.get("label") or payload.get("title") or payload.get("chat_name") or f"{provider} conversation")
        cleaned_label = clean_chat_title(raw_label)
        return {
            "target": target,
            "label": cleaned_label or (f"{provider} conversation" if raw_label.strip().startswith("<") else raw_label),
            "description": str(payload.get("description") or f"{provider} {payload.get('kind') or 'chat'}"),
            "kind": "llm",
            "provider": provider,
            "sessionId": session_id,
            "sourceFile": source_file,
            "exportPath": str(payload.get("exportPath") or file_path or ""),
            "historyExportPath": str(payload.get("historyExportPath") or payload.get("history_export_path") or ""),
            "messageCount": int(payload.get("messageCount", payload.get("message_count", len(messages))) or 0),
            "workspace": workspace,
            "ircChannel": str(
                payload.get("ircChannel")
                or irc_channel_name(payload.get("label") or payload.get("title") or payload.get("chat_name"), identity, provider)
            ),
            "messages": messages,
            "updatedAt": updated_at or time.time() * 1000,
            "createdAt": created_at,
            "sortAt": sort_at,
            "writable": bool(writable_provider(provider)),
            "origin": origin,
            # Which OS user this conversation's CLI calls run as. None means the OS user
            # currently running the service — purely additive "Continue as…" field.
            "osUser": str(payload["osUser"]) if payload.get("osUser") else None,
            "sourceUser": resolve_source_user(payload, source_file, origin),
            "icon": provider_icon(provider),
            "participants": [
                {"nick": "You", "modes": ""},
                {"nick": provider_label(provider), "modes": "bot"},
            ],
            "loaded": payload.get("loaded") is not False,
        }

    # --- history ---------------------------------------------------------------------

    def _history_conversations(self) -> list[dict]:
        if self._history_cache is not None:
            return self._history_cache
        context = Path(self.context_dir)
        if not context.exists():
            return []
        manifest = safe_json(context / ".subjective_localchats_manifest.json")
        if manifest and isinstance(manifest.get("sources"), dict):
            # Trust the manifest — stat'ing tens of thousands of files blocks long
            # enough that the sidebar fetch times out.
            conversations = []
            for record in manifest["sources"].values():
                fingerprint = record.get("fingerprint") or {}
                source_file = str(record.get("source_file") or fingerprint.get("source_file") or "")
                session_match = UUID_RE.search(source_file)
                export_path = str(context / os.path.basename(record.get("export_filename") or record.get("export_path") or ""))
                filename_timestamp = export_filename_time(export_path)
                last_activity_ms = (
                    float(record.get("last_activity_at") or 0)
                    or (0 if "state.vscdb" in source_file else float(fingerprint.get("mtime_ns") or 0) / 1e6)
                    or filename_timestamp
                )
                raw_title = record.get("title") or os.path.basename(source_file)
                cleaned_title = clean_chat_title(raw_title)
                conversation = self._normalize_conversation(
                    {
                        "type": "localchat",
                        "title": cleaned_title
                        or (f"{fingerprint.get('product') or 'chat'} conversation" if str(raw_title).strip().startswith("<") else raw_title),
                        "product": fingerprint.get("product") or "unknown",
                        "kind": fingerprint.get("kind") or "chat",
                        # group(0): UUID_RE has no capture group, unlike the JS original.
                        "thread_id": session_match.group(0).lower() if session_match else "",
                        "source_file": source_file,
                        "source_user": fingerprint.get("source_user") or record.get("source_user") or "",
                        "fingerprint": fingerprint,
                        "source_mtime": float(fingerprint.get("mtime_ns") or 0) / 1e9,
                        "chat_started_at": filename_timestamp / 1000,
                        "last_activity_at": last_activity_ms,
                        "sort_at": last_activity_ms,
                        "messages": [],
                        "messageCount": int(record.get("message_count") or 0),
                        "exportPath": export_path,
                        "loaded": False,
                    },
                    "history",
                    export_path,
                )
                if conversation:
                    conversations.append(conversation)
            self._history_cache = conversations
            self._sorted_cache = None
            return conversations
        conversations = []
        try:
            names = os.listdir(context)
        except OSError:
            names = []
        for name in names:
            if "-Localchats-" not in name or not name.endswith("-context.json"):
                continue
            file_path = str(context / name)
            payload = safe_json(file_path)
            conversation = self._normalize_conversation(payload, "history", file_path)
            if conversation:
                conversations.append(conversation)
        self._history_cache = conversations
        self._sorted_cache = None
        return conversations

    def _sorted_conversations(self) -> list[dict]:
        if self._sorted_cache is not None:
            return self._sorted_cache["list"]
        listed = self._build_sorted_conversations()
        self._sorted_cache = {"list": listed, "haystacks": None}
        return listed

    def _search_haystacks(self) -> list[str]:
        self._sorted_conversations()
        if self._sorted_cache["haystacks"] is None:
            self._sorted_cache["haystacks"] = [
                (
                    f"{conversation.get('label') or ''} {conversation.get('provider') or ''} "
                    f"{conversation.get('description') or ''} {conversation.get('sourceUser') or ''} "
                    f"{conversation.get('osUser') or ''}"
                ).lower()
                for conversation in self._sorted_cache["list"]
            ]
        return self._sorted_cache["haystacks"]

    def count_conversations(self) -> int:
        return len(self._sorted_conversations())

    def _build_sorted_conversations(self) -> list[dict]:
        deduplicated: dict[str, dict] = {}
        for conversation in [*self._history_conversations(), *self.local]:
            key = conversation_key(conversation)
            previous = deduplicated.get(key)
            if not previous:
                deduplicated[key] = conversation
                continue
            history = previous if previous.get("origin") == "history" else (conversation if conversation.get("origin") == "history" else None)
            local = previous if previous.get("origin") == "local" else (conversation if conversation.get("origin") == "local" else None)
            if history and local:
                override = self.label_overrides.get(local["target"]) or self.label_overrides.get(history["target"])
                deduplicated[key] = {
                    **local,
                    "label": override or history.get("label") or local.get("label"),
                    "sourceUser": local.get("sourceUser") or history.get("sourceUser"),
                    "sessionId": local.get("sessionId") or history.get("sessionId"),
                    "sortAt": max(float(local.get("sortAt") or 0), float(history.get("sortAt") or 0)),
                    "updatedAt": max(float(local.get("updatedAt") or 0), float(history.get("updatedAt") or 0)),
                }
                continue
            if float(conversation.get("updatedAt") or 0) >= float(previous.get("updatedAt") or 0):
                deduplicated[key] = conversation

        def last_activity(conversation: dict) -> float:
            return float(conversation.get("sortAt") or conversation.get("updatedAt") or conversation.get("createdAt") or 0)

        return sorted(deduplicated.values(), key=last_activity, reverse=True)

    # --- listing ---------------------------------------------------------------------

    def list_conversations(self, query: str = "", limit: int | None = None, include_messages: bool = False) -> dict:
        needle = str(query or "").strip().lower()
        listed = self._sorted_conversations()
        if needle:
            haystacks = self._search_haystacks()
            listed = [conversation for index, conversation in enumerate(listed) if needle in haystacks[index]]
        total = len(listed)
        if isinstance(limit, int) and limit > 0:
            listed = listed[:limit]
        conversations = []
        for conversation in listed:
            if include_messages:
                conversations.append(self._with_label_override(self._hydrate_local_conversation(conversation)))
                continue
            conversations.append(
                self._with_label_override(
                    {
                        "target": conversation["target"],
                        "label": conversation["label"],
                        "kind": conversation["kind"],
                        "provider": conversation["provider"],
                        "icon": conversation["icon"],
                        "writable": conversation["writable"],
                        "origin": conversation["origin"],
                        "updatedAt": conversation["updatedAt"],
                        "sortAt": conversation["sortAt"],
                        "createdAt": conversation["createdAt"],
                        "messageCount": conversation["messageCount"],
                        "sessionId": conversation["sessionId"],
                        "osUser": conversation["osUser"],
                        "sourceUser": conversation["sourceUser"],
                        "workspace": conversation["workspace"],
                        "ircChannel": conversation["ircChannel"],
                        "exportPath": conversation["exportPath"],
                        "historyExportPath": conversation["historyExportPath"],
                        "sourceFile": conversation["sourceFile"],
                        "messages": [],
                        "loaded": conversation["origin"] == "local" and len(conversation["messages"]) == 0,
                        "activeOnIrc": conversation["target"] in self.active_targets,
                    }
                )
            )
        return {"conversations": conversations, "total": total}

    def _with_label_override(self, conversation: dict | None) -> dict | None:
        if not conversation:
            return conversation
        override = self.label_overrides.get(conversation["target"])
        if not override or override == conversation.get("label"):
            return conversation
        return {**conversation, "label": override}

    def rename_conversation(self, target: str, name: str) -> dict:
        key = str(target or "")
        if not key:
            return {"ok": False, "error": "Missing conversation target."}
        clean = str(name or "").strip()[:200]
        if not clean:
            return {"ok": False, "error": "Name cannot be empty."}
        local = next((conversation for conversation in self.local if conversation["target"] == key), None)
        if local:
            local["label"] = clean
        self.label_overrides[key] = clean
        self._save_state()
        self.log("info", "LLM conversation renamed", {"target": key, "label": clean, "local": bool(local)})
        return {"ok": True, "conversation": self.find_conversation(key)}

    def _runtime_for(self, provider: Any) -> str:
        key = provider_key(provider)
        if key == "claude":
            return self.claude_runtime
        if key == "cursor":
            return self.cursor_runtime
        return self.codex_runtime

    def create_conversation(self, provider: Any = None, name: str = "", workspace: str = "") -> dict:
        now = time.time() * 1000
        key = provider_key(provider) if writable_provider(provider) else DEFAULT_PROVIDER
        label = provider_label(key)
        conversation = self._normalize_conversation(
            {
                "target": f"llm:{key}:local:{uuid.uuid4()}",
                "provider": key,
                "label": str(name or f"New {label} chat").strip() or f"New {label} chat",
                "description": f"{label} CLI conversation",
                "workspace": str(workspace or self.root).strip() or self.root,
                "sourceUser": current_username(),
                "messages": [],
                "createdAt": now,
                "updatedAt": now,
                "loaded": True,
            },
            "local",
        )
        self.local.insert(0, conversation)
        self.mark_active(conversation["target"])
        self._save_state()
        return conversation

    def create_sibling_for_user(self, source_target: str, username: str, provider: Any = None) -> dict:
        """Fresh sibling conversation for "Continue as <username>" — a sessionId created
        as one OS user cannot be resumed as another ($HOME-scoped session stores)."""
        source = self.find_conversation(source_target, include_messages=False)
        if not source:
            raise ValueError("Conversation was not found.")
        key = provider_key(provider) if provider else provider_key(source["provider"])
        if not writable_provider(key):
            raise ValueError(f"{key} has no writable datasource.")
        cross_provider = key != provider_key(source["provider"])
        label = provider_label(key)
        now = time.time() * 1000
        conversation = self._normalize_conversation(
            {
                "target": f"llm:{key}:local:{uuid.uuid4()}",
                "provider": key,
                "label": f"{source['label']} ({label} as {username})" if cross_provider else f"{source['label']} (as {username})",
                "description": (
                    f"{label} CLI conversation — continued from {provider_label(source['provider'])} as {username}"
                    if cross_provider
                    else f"{source['description']} — continued as {username}"
                ),
                "workspace": source["workspace"],
                "osUser": username,
                "sourceUser": username,
                "messages": [],
                "createdAt": now,
                "updatedAt": now,
                "loaded": True,
            },
            "local",
        )
        self.local.insert(0, conversation)
        self.mark_active(conversation["target"])
        self._save_state()
        return conversation

    def find_conversation(self, target: str, include_messages: bool = False) -> dict | None:
        local = next((conversation for conversation in self.local if conversation["target"] == target), None)
        if local:
            return self._with_label_override(self._hydrate_local_conversation(local) if include_messages else local)
        metadata = next(
            (conversation for conversation in self._history_conversations() if conversation["target"] == target), None
        )
        if not metadata:
            if include_messages:
                self.log("error", "LLM conversation not found", {"target": target})
            return None
        if not include_messages or not metadata.get("exportPath"):
            return self._with_label_override(metadata)
        payload = safe_json(metadata["exportPath"])
        if not payload:
            return self._with_label_override(metadata)
        return self._with_label_override(
            self._normalize_conversation(
                {
                    **payload,
                    "target": metadata["target"],
                    "label": prefer_conversation_label(
                        metadata["label"],
                        payload.get("label") or payload.get("title") or payload.get("chat_name"),
                        metadata["provider"],
                    ),
                    "sessionId": metadata["sessionId"]
                    or payload.get("sessionId")
                    or payload.get("thread_id")
                    or payload.get("session_id"),
                    "sourceFile": metadata["sourceFile"] or payload.get("source_file"),
                    "exportPath": metadata["exportPath"],
                    "loaded": True,
                },
                "history",
                metadata["exportPath"],
            )
        )

    def reload_conversation(self, target: str) -> dict | None:
        metadata = self.find_conversation(target, include_messages=False)
        if not metadata:
            return None
        source_file = str(metadata.get("sourceFile") or "").strip()
        session_id = str(metadata.get("sessionId") or "").strip()
        if not source_file and not session_id:
            return self.find_conversation(target, include_messages=True)
        started = time.time()
        self.log("info", "LLM live reload start", {
            "target": target, "provider": metadata["provider"], "sourceFile": source_file, "sessionId": session_id,
        })
        response = self._run_json_script(
            self.reload_runtime,
            {"source_file": source_file, "session_id": session_id, "provider": metadata.get("provider") or ""},
            45,
        )
        transcript = response.get("transcript") if response and response.get("success") else None
        if not transcript or not isinstance(transcript.get("messages"), list):
            self.log("warn", "LLM live reload empty", {
                "target": target, "error": (response or {}).get("error") or "", "ms": int((time.time() - started) * 1000),
            })
            return self.find_conversation(target, include_messages=True)
        self._write_live_export(metadata, transcript)
        self._history_cache = None
        self._sorted_cache = None
        normalized = self._normalize_conversation(
            {
                **transcript,
                "type": "localchat",
                "target": metadata["target"],
                "label": prefer_conversation_label(
                    metadata["label"], transcript.get("title") or transcript.get("chat_name"), metadata["provider"]
                ),
                "sessionId": metadata["sessionId"] or session_id or uuid_from_path(transcript.get("source_file") or source_file),
                "sourceFile": metadata["sourceFile"] or transcript.get("source_file") or source_file,
                "exportPath": metadata["exportPath"],
                "loaded": True,
            },
            "history",
            metadata["exportPath"],
        )
        local = next((item for item in self.local if item["target"] == target), None)
        if local and normalized:
            # Keep locally recorded sends the provider transcript does not know about yet
            # (e.g. an attachment blocked by an active-writer error).
            merged = self._hydrate_local_conversation({**local, "messages": normalized["messages"]})
            local["messages"] = merged["messages"]
            local["messageCount"] = len(normalized["messages"])
            local["updatedAt"] = normalized["updatedAt"]
            local["sortAt"] = normalized["sortAt"]
            local["historyExportPath"] = metadata["exportPath"] or local.get("historyExportPath")
            local["sourceFile"] = normalized["sourceFile"] or local.get("sourceFile")
            composer_id = composer_id_from_source(local.get("sourceFile") or "")
            if composer_id:
                local["sessionId"] = composer_id
            self._save_state()
        self.log("info", "LLM live reload ok", {
            "target": target,
            "messageCount": len(normalized["messages"]) if normalized else 0,
            "ms": int((time.time() - started) * 1000),
        })
        return self.find_conversation(target, include_messages=True)

    def _write_live_export(self, metadata: dict, transcript: dict) -> None:
        export_path = str(metadata.get("exportPath") or "").strip()
        if not export_path:
            return
        existing = safe_json(export_path) or {"type": "localchat"}
        messages = transcript.get("messages") or []
        record = {
            **existing,
            "type": "localchat",
            "title": transcript.get("title") or existing.get("title"),
            "chat_name": transcript.get("title") or existing.get("chat_name"),
            "product": transcript.get("product") or existing.get("product"),
            "source_file": transcript.get("source_file") or existing.get("source_file") or metadata.get("sourceFile"),
            "last_activity_at": transcript.get("last_activity_at") or time.time(),
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "messages": messages,
        }
        try:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as error:
            self.log("error", "LLM live reload export write failed", {"exportPath": export_path, "message": str(error)})

    def _run_json_script(self, script_path: str, payload: dict, timeout_seconds: float) -> dict:
        if not Path(script_path).exists():
            return {"success": False, "error": f"Reload script missing: {script_path}"}
        try:
            completed = subprocess.run(
                [self.python_path, script_path],
                input=json.dumps(payload or {}),
                capture_output=True,
                text=True,
                cwd=self.root,
                env={**os.environ},
                timeout=max(5, timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Reload script timed out."}
        except OSError as error:
            return {"success": False, "error": str(error)}
        lines = [line for line in str(completed.stdout or "").strip().split("\n") if line]
        try:
            return json.loads(lines[-1] if lines else "{}")
        except ValueError:
            return {
                "success": False,
                "error": str(completed.stderr or completed.stdout or f"exit {completed.returncode}").strip(),
            }

    def _hydrate_local_conversation(self, conversation: dict | None) -> dict | None:
        if not conversation or conversation.get("origin") != "local" or not conversation.get("historyExportPath"):
            return conversation
        payload = safe_json(conversation["historyExportPath"])
        if not payload:
            return conversation
        history = self._normalize_conversation(
            {
                **payload,
                "target": conversation["target"],
                "label": prefer_conversation_label(
                    conversation["label"],
                    payload.get("label") or payload.get("title") or payload.get("chat_name"),
                    conversation["provider"],
                ),
                "exportPath": conversation["historyExportPath"],
                "loaded": True,
            },
            "history",
            conversation["historyExportPath"],
        )
        if not history:
            return conversation

        history_times: dict[str, list[float]] = {}
        for message in history["messages"]:
            signature = f"{message['role']}\0{message.get('command') or message.get('text')}"
            history_times.setdefault(signature, []).append(float(message.get("time") or 0))
        local_only = []
        for message in conversation["messages"]:
            signature = f"{message['role']}\0{message.get('command') or message.get('text')}"
            matches = history_times.get(signature) or []
            if not any(abs(stamp - float(message.get("time") or 0)) <= 300000 for stamp in matches):
                local_only.append(message)
        messages = sorted([*history["messages"], *local_only], key=lambda message: float(message.get("time") or 0))
        return {**conversation, "messages": messages, "messageCount": len(messages), "loaded": True}

    def mark_active(self, target: str) -> None:
        if target in self.active_targets:
            self.active_targets.remove(target)
        self.active_targets.append(target)
        while len(self.active_targets) > 20:
            self.active_targets.pop(0)
        self._save_state()

    # --- provider status / sessions ---------------------------------------------------

    def status(self, provider: Any = DEFAULT_PROVIDER) -> dict:
        key = provider_key(provider) if writable_provider(provider) else DEFAULT_PROVIDER
        request = {"action": "status"}
        response = self._run_datasource(key, request, 30) or self._run_pipeline(self._runtime_for(key), request, 30)
        return {"provider": key, **self._pipeline_result(response)}

    def list_provider_sessions(self, provider: Any = "") -> dict:
        listed = list_provider_sessions()
        if not provider:
            return listed
        return sessions_for_provider(listed, provider)

    def is_auth_failure(self, result: dict | None) -> bool:
        if not result:
            return False
        if result.get("status") == "authentication_required":
            return True
        blob = f"{result.get('error') or ''} {result.get('response') or ''}".lower()
        return (
            "authentication_required" in blob
            or 'loggedin": false' in blob
            or 'logged in": false' in blob
            or "authentication required" in blob
            or "agent login" in blob
            or "auth status" in blob
        )

    # --- history refresh --------------------------------------------------------------

    def refresh_history(self) -> dict:
        if self.busy:
            self._history_refresh_deferred = True
            self.log("info", "LLM history refresh deferred (send in flight)", {"busy": list(self.busy)})
            return {"success": True, "status": "deferred"}
        self._history_cache = None
        self._sorted_cache = None
        started = time.time()
        self.log("info", "LLM history refresh start", {"contextDir": self.context_dir})
        response = self._run_pipeline(self.history_runtime, {"context_folder": self.context_dir, "max_files": 500}, 1800)
        self._history_cache = None
        self._sorted_cache = None
        result = self._pipeline_result(response)
        self.log("info" if result.get("success") else "error", "LLM history refresh done", {
            "ok": bool(result.get("success")),
            "status": result.get("status") or "",
            "error": result.get("error") or "",
            "ms": int((time.time() - started) * 1000),
            "conversations": self.count_conversations(),
        })
        return result

    # --- sending ----------------------------------------------------------------------

    def _wait_for_cursor_ide_reply(self, target: str, user_message: dict, timeout_ms: int | None = None) -> dict:
        max_ms = int(timeout_ms or CURSOR_INJECT_POLL_MAX_MS)
        interval = CURSOR_INJECT_POLL_MS / 1000
        deadline = time.time() + max_ms / 1000
        while True:
            try:
                live = self.reload_conversation(target)
                if live and isinstance(live.get("messages"), list):
                    reply = next(
                        (
                            message
                            for message in reversed(live["messages"])
                            if message.get("role") == "assistant"
                            and float(message.get("time") or 0) >= float(user_message.get("time") or 0)
                            and str(message.get("text") or "").strip()
                        ),
                        None,
                    )
                    if reply:
                        return {"ok": True, "conversation": live, "assistantMessage": reply}
            except Exception as error:  # noqa: BLE001
                self.log("warn", "Cursor DOM inject poll failed", {"target": target, "message": str(error)})
            if time.time() >= deadline:
                break
            time.sleep(interval)
        return {"ok": False, "error": "Cursor tab did not produce a reply yet"}

    def send_message(self, target: str, text: str, files: list | None = None) -> dict:
        files = files or []
        clean = str(text or "").strip()
        if not clean:
            return {"ok": False, "status": "invalid_request", "error": "Message is empty."}
        source = self.find_conversation(target, include_messages=True)
        if not source:
            self.log("error", "LLM send aborted: conversation not found", {"target": target, "promptChars": len(clean)})
            return {"ok": False, "status": "not_found", "error": "Conversation was not found."}
        provider = writable_provider(source["provider"])
        if not provider:
            self.log("error", "LLM send aborted: read-only provider", {"target": target, "provider": source["provider"]})
            return {"ok": False, "status": "read_only", "error": f"{source['provider']} history has no writable datasource."}
        if target in self.busy:
            self.log("warn", "LLM send aborted: already busy", {"target": target, "provider": source["provider"]})
            return {"ok": False, "status": "busy", "error": f"{provider['label']} is already responding in this conversation."}

        conversation = next((item for item in self.local if item["target"] == target), None)
        if not conversation:
            conversation = {
                **source,
                "origin": "local",
                "historyExportPath": source["exportPath"],
                "messageCount": len(source["messages"]),
                "messages": [],
            }
            self.local.insert(0, conversation)

        # "Continue as <osUser>" conversations must run through that user's own connected
        # session. Checked before touching transcript/busy state, so a not-connected user
        # never leaves a dangling "sent" message behind.
        running_as_other_user = bool(conversation.get("osUser")) and conversation["osUser"] != current_username()
        if running_as_other_user:
            session_status = self.su_session_manager.get_status(conversation["osUser"]) if self.su_session_manager else {"status": "idle"}
            if session_status.get("status") != "ready":
                self.log("error", "LLM send aborted: OS user session not ready", {
                    "target": target, "osUser": conversation["osUser"], "sessionStatus": session_status.get("status"),
                })
                return {
                    "ok": False,
                    "status": "os_user_not_connected",
                    "error": f"Reconnect as {conversation['osUser']} first (right-click the conversation).",
                }

        # Record the user's line before the turn runs — but it must not survive a turn
        # that never happened (an unsent message reads exactly like a delivered one).
        previous_updated_at = conversation.get("updatedAt")
        previous_sort_at = conversation.get("sortAt")
        user_message: dict = {"role": "user", "text": clean, "time": time.time() * 1000}
        if files:
            user_message["attachments"] = [
                {
                    "filename": str(file.get("filename") or "pasted-image"),
                    "mime_type": str(file.get("mime_type")),
                    "content": str(file.get("content")),
                }
                for file in files
                if re.match(r"^image/", str((file or {}).get("mime_type") or ""), re.IGNORECASE) and (file or {}).get("content")
            ]
        conversation["messages"].append(user_message)
        conversation["updatedAt"] = user_message["time"]
        conversation["sortAt"] = user_message["time"]
        self._save_state()
        self.busy.add(target)
        started = time.time()
        original_session_id = str(conversation.get("sessionId") or source.get("sessionId") or "").strip()
        # IDE composer UUID vs later cursor-agent chat id differ. Resuming the CLI id
        # talks to a fork Cursor's open tab never sees — prefer the composer id.
        if provider["bundle"] == "cursor":
            composer_id = composer_id_from_source(conversation.get("sourceFile") or source.get("sourceFile") or "")
            if composer_id and composer_id != original_session_id.lower():
                self.log("warn", "LLM send rebound Cursor session to IDE composer", {
                    "target": target, "from": original_session_id, "to": composer_id,
                })
            if composer_id:
                original_session_id = composer_id
                conversation["sessionId"] = composer_id
        self.log("info", "LLM send start", {
            "target": target,
            "provider": source["provider"],
            "sessionId": original_session_id,
            "newSession": not original_session_id,
            "osUser": conversation.get("osUser") or current_username(),
            "files": len(files),
            "promptChars": len(clean),
            "promptPreview": clip_text(clean),
        })

        def rollback_user_message() -> None:
            try:
                index = len(conversation["messages"]) - 1 - conversation["messages"][::-1].index(user_message)
            except ValueError:
                return
            conversation["messages"].pop(index)
            conversation["updatedAt"] = previous_updated_at
            conversation["sortAt"] = previous_sort_at
            self._save_state()

        try:
            composer_id = (
                composer_id_from_source(conversation.get("sourceFile") or source.get("sourceFile") or "")
                if provider["bundle"] == "cursor"
                else ""
            )
            if composer_id:
                injected = inject_into_cursor_dom(clean)
                self.log("info", "Cursor DOM inject", {
                    "target": target, "ok": injected.get("ok"), "error": injected.get("error") or "", "via": injected.get("via") or "",
                })
                if injected.get("ok"):
                    waited = self._wait_for_cursor_ide_reply(target, user_message)
                    conversation["sessionId"] = original_session_id
                    if waited.get("ok"):
                        self._save_state()
                        self.log("info", "LLM send ok via Cursor DOM", {
                            "target": target, "sessionId": conversation["sessionId"], "ms": int((time.time() - started) * 1000),
                        })
                        return {
                            "ok": True,
                            "result": {"success": True, "response": waited["assistantMessage"]["text"], "via": "cdp-dom"},
                            "conversation": waited.get("conversation") or self._hydrate_local_conversation(conversation),
                            "userMessage": user_message,
                            "assistantMessage": waited["assistantMessage"],
                            "cursorIdeHint": False,
                            "cursorDomInjected": True,
                        }
                    pending = {
                        "role": "assistant",
                        "text": "Typed into the open Cursor tab. Click ↻ when the reply lands.",
                        "time": time.time() * 1000,
                    }
                    conversation["messages"].append(pending)
                    conversation["updatedAt"] = pending["time"]
                    conversation["sortAt"] = pending["time"]
                    self._save_state()
                    return {
                        "ok": True,
                        "result": {"success": True, "response": pending["text"], "via": "cdp-dom-pending"},
                        "conversation": self._hydrate_local_conversation(conversation),
                        "userMessage": user_message,
                        "assistantMessage": pending,
                        "cursorIdeHint": False,
                        "cursorDomInjected": True,
                    }
            request = {
                "action": "send",
                "prompt": clean,
                "files": files,
                "target": target,
                "session_id": original_session_id,
                "new_session": not original_session_id,
                "working_directory": conversation.get("workspace"),
                "source_file": conversation.get("sourceFile") or source.get("sourceFile") or "",
                "timeout": AGENT_TURN_TIMEOUT_SECONDS,
            }
            # Codex sandboxes per request; Claude takes an equivalent permission mode;
            # Cursor's CLI manages its own sandboxing.
            if provider["bundle"] == "claude":
                request["permission_mode"] = "acceptEdits"
            elif provider["bundle"] == "codex":
                request["sandbox_mode"] = "workspace-write"
            runtime_path = self._runtime_for(source["provider"])
            if running_as_other_user:
                # Must stay a subprocess: the whole point is running under a different uid,
                # and an import cannot change the uid of this process.
                pipeline_response = self._run_pipeline_for_user(
                    conversation["osUser"], runtime_path, request, AGENT_TURN_WRAPPER_TIMEOUT_SECONDS
                )
            else:
                pipeline_response = self._run_datasource(
                    source["provider"], request, AGENT_TURN_WRAPPER_TIMEOUT_SECONDS
                ) or self._run_pipeline(runtime_path, request, AGENT_TURN_WRAPPER_TIMEOUT_SECONDS)
            result = self._pipeline_result(pipeline_response)
            if not result.get("success"):
                # Preserve attachment messages so the screenshot and its failure stay
                # visible after reload; plain failed text keeps legacy rollback behavior.
                if files:
                    user_message["sendFailed"] = True
                    user_message["error"] = str(result.get("error") or result.get("status") or "Send failed")
                    self._save_state()
                else:
                    rollback_user_message()
                self.log("error", "LLM send failed", {
                    "target": target,
                    "provider": source["provider"],
                    "status": result.get("status") or "",
                    "error": clip_text(result.get("error"), 400),
                    "authFailure": self.is_auth_failure(result),
                    "ms": int((time.time() - started) * 1000),
                })
                failure = {"ok": False, **result, "userMessage": user_message}
                if self.is_auth_failure(failure):
                    failure["sessions"] = self.list_provider_sessions(source["provider"])
                    # This process just failed auth — don't advertise the current user as ready.
                    if failure["sessions"].get("users"):
                        failure["sessions"]["users"] = [
                            {**user, "ready": False} if user.get("current") else user
                            for user in failure["sessions"]["users"]
                        ]
                        failure["sessions"]["users"].sort(key=lambda user: (-int(bool(user["ready"])), user["username"]))
                return failure
            conversation["sessionId"] = original_session_id or str(result.get("session_id") or conversation.get("sessionId") or "")
            assistant_message = {"role": "assistant", "text": str(result.get("response") or ""), "time": time.time() * 1000}
            conversation["messages"].append(assistant_message)
            conversation["updatedAt"] = assistant_message["time"]
            conversation["sortAt"] = assistant_message["time"]
            self._save_state()
            self.log("info", "LLM send ok", {
                "target": target,
                "provider": source["provider"],
                "sessionId": conversation["sessionId"],
                "responseChars": len(str(result.get("response") or "")),
                "responsePreview": clip_text(result.get("response")),
                "ms": int((time.time() - started) * 1000),
            })
            shown = self._hydrate_local_conversation(conversation)
            try:
                reloaded = self.reload_conversation(target)
                if reloaded and isinstance(reloaded.get("messages"), list) and len(reloaded["messages"]) >= len(conversation["messages"]):
                    shown = reloaded
            except Exception as error:  # noqa: BLE001
                self.log("warn", "LLM post-send live reload failed", {"target": target, "message": str(error)})
            return {
                "ok": True,
                "result": result,
                "conversation": shown,
                "userMessage": user_message,
                "assistantMessage": assistant_message,
                "cursorIdeHint": bool(
                    provider["bundle"] == "cursor"
                    and composer_id_from_source(conversation.get("sourceFile") or source.get("sourceFile") or "")
                ),
            }
        except Exception as error:
            if files:
                user_message["sendFailed"] = True
                user_message["error"] = str(error)
                self._save_state()
            else:
                rollback_user_message()
            self.log("error", "LLM send crashed", {"target": target, "provider": source["provider"], "message": str(error)})
            raise
        finally:
            self.busy.discard(target)
            if self._history_refresh_deferred and not self.busy:
                self._history_refresh_deferred = False
                threading.Thread(target=self._deferred_refresh, daemon=True).start()

    def _deferred_refresh(self) -> None:
        try:
            self.refresh_history()
        except Exception as error:  # noqa: BLE001
            self.log("error", "LLM deferred history refresh failed", {"message": str(error)})

    # --- pipeline plumbing ------------------------------------------------------------

    def _pipeline_result(self, response: dict | None) -> dict:
        if not response or not response.get("success"):
            return response or {"success": False, "status": "runtime_error", "error": "Pipeline returned no response."}
        result = response.get("result")
        if isinstance(result, dict) and "pipeline_result" in result:
            result = result["pipeline_result"]
        if isinstance(result, dict):
            return result
        return {"success": True, "status": "completed", "result": result}

    def _run_datasource(self, provider: str, payload: dict, timeout_seconds: float) -> dict | None:
        """Call the datasource module directly, or return None to fall back to the bundle.

        These are the same `subjective_*_datasource` packages the monolithic web app uses,
        carried here as submodules under plugins/. Calling one in-process skips a Python
        interpreter start and an exported-bundle unpack per turn, and turns a crash into a
        real traceback instead of parsed stderr.

        Returns None — never raises — when the datasource is missing or the import fails,
        so the exported bundle stays a working fallback and the service keeps answering.
        """
        if not self.use_datasource_modules:
            return None
        try:
            from . import datasources
        except ImportError:
            return None
        # A datasource that predates the chat contract loads fine but cannot run a turn:
        # its run() reaches the abstract base and returns a status object, so the send
        # silently does nothing. The exported bundle can still do it, so treat "loaded but
        # cannot chat" exactly like "not installed" and fall through to it.
        if str(payload.get("action") or "") == "send":
            try:
                if not datasources.supports_chat(datasources.load(provider)):
                    self.log("info", "Datasource module cannot run a turn; using exported bundle", {
                        "provider": provider,
                        "reason": "no handle_message/run of its own — datasource predates the chat contract",
                    })
                    return None
            except LookupError:
                return None

        started = time.time()
        try:
            connection = datasources.connection_for(
                provider,
                workspace=str(payload.get("working_directory") or ""),
                session_id=str(payload.get("session_id") or ""),
                timeout=int(timeout_seconds),
            )
            result = datasources.run(provider, payload, connection)
        except LookupError as error:
            # Not installed for this provider — the bundle path handles it.
            self.log("info", "Datasource module unavailable; using exported bundle", {
                "provider": provider, "message": str(error)[:200],
            })
            return None
        except Exception as error:  # noqa: BLE001
            self.log("error", "Datasource module call failed; using exported bundle", {
                "provider": provider, "message": str(error)[:300],
            })
            return None
        self.log("info", "Datasource module call", {
            "provider": provider,
            "action": str(payload.get("action") or ""),
            "status": result.get("status") or "",
            "ok": result.get("success") is not False,
            "ms": int((time.time() - started) * 1000),
        })
        # _pipeline_result unwraps the bundle's {success, result:{pipeline_result}} envelope,
        # but only on success — its failure branch returns the envelope untouched. Wrapping a
        # failure would therefore bury `status` and `error` one level down where nothing looks
        # for them, and the caller reports an empty "Unknown error" while the datasource knew
        # exactly what went wrong. A failed datasource result already carries both fields at
        # the top level, which is the shape that branch expects, so hand it back unwrapped.
        if result.get("success") is False:
            return result
        return {"success": True, "result": result}

    def _run_pipeline(self, runtime_path: str, payload: dict, timeout_seconds: float) -> dict:
        action = str(payload.get("action") or "history") if isinstance(payload, dict) else "history"
        started = time.time()
        self.log("info", "LLM pipeline spawn", {
            "action": action,
            "runtime": os.path.basename(runtime_path),
            "timeoutSeconds": timeout_seconds,
            "sessionId": (payload or {}).get("session_id") or "",
            "workspace": (payload or {}).get("working_directory") or (payload or {}).get("context_folder") or "",
        })
        if not Path(runtime_path).exists() or not Path(self.pipeline_runtime).exists():
            missing = runtime_path if not Path(runtime_path).exists() else self.pipeline_runtime
            self.log("error", "LLM pipeline runtime missing", {"runtime": missing, "action": action})
            return {"success": False, "status": "runtime_missing", "error": f"Pipeline runtime is missing: {missing}"}
        runtime_env = {**os.environ, "SUBJECTIVE_CHAT_PIPELINE_PARENT_HOME": self.app_home}
        for key in (
            "USERDATA_PATH",
            "LOG_PATH",
            "CONTEXT_STORAGE",
            "DATASOURCES_TEMP_STORAGE",
            "DS_CONNECTION_TMP_SPACE",
            "DATASOURCES_PLUGIN_PATH",
            "SUBJECTIVE_PIPELINE_HOME",
        ):
            runtime_env.pop(key, None)
        try:
            process = subprocess.Popen(
                [self.python_path, self.pipeline_runtime, runtime_path, "--timeout", str(timeout_seconds)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.root,
                env=runtime_env,
                text=True,
            )
        except OSError as error:
            self.log("error", "LLM pipeline spawn error", {"action": action, "message": str(error)})
            return {"success": False, "status": "runtime_error", "error": str(error)}
        stderr_chunks: list[str] = []
        stderr_lock = threading.Lock()

        def read_stderr() -> None:
            for line in process.stderr:  # type: ignore[union-attr]
                with stderr_lock:
                    stderr_chunks.append(line)
                self._consume_progress(line, payload)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        try:
            stdout, _ = process.communicate(input=json.dumps(payload), timeout=timeout_seconds + 120)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
        stderr_thread.join(timeout=5)
        with stderr_lock:
            stderr = "".join(stderr_chunks)
        parsed = self._parse_pipeline_output(stdout, stderr)
        self.log("error" if parsed.get("success") is False else "info", "LLM pipeline exit", {
            "action": action,
            "runtime": os.path.basename(runtime_path),
            "exitCode": process.returncode,
            "ok": parsed.get("success") is not False,
            "status": parsed.get("status") or "",
            "stderrChars": len(stderr),
            "stderrPreview": clip_text(stderr, 240),
            "ms": int((time.time() - started) * 1000),
        })
        return parsed

    def _consume_progress(self, chunk: str, payload: dict | None) -> None:
        if not chunk or not callable(self.on_progress):
            return
        prefix = "SUBJECTIVE_PROGRESS "
        for line in str(chunk).splitlines():
            index = line.find(prefix)
            if index == -1:
                continue
            try:
                body = json.loads(line[index + len(prefix):])
                text = str(body.get("text") or "").strip()
                if text:
                    self.on_progress({
                        "text": text,
                        "target": (payload or {}).get("target") or "",
                        "sessionId": (payload or {}).get("session_id") or "",
                    })
            except ValueError:
                pass  # partial JSON

    def _parse_pipeline_output(self, stdout: str, stderr: str) -> dict:
        try:
            result = json.loads(stdout or "{}")
            if not isinstance(result, dict):
                raise ValueError("not a dict")
            if not result.get("error") and str(stderr or "").strip() and not result.get("success"):
                result["error"] = str(stderr).strip()
            return result
        except ValueError:
            return {
                "success": False,
                "status": "runtime_error",
                "error": str(stderr or "").strip() or str(stdout or "").strip() or "Exported pipeline returned invalid JSON.",
            }

    def _run_pipeline_for_user(self, os_user: str, runtime_path: str, payload: dict, timeout_seconds: float) -> dict:
        """Same one-shot invocation as _run_pipeline, but through `os_user`'s
        already-authenticated bridge session. Only called once send_message() has
        confirmed that session is ready."""
        if not Path(runtime_path).exists() or not Path(self.pipeline_runtime).exists():
            missing = runtime_path if not Path(runtime_path).exists() else self.pipeline_runtime
            return {"success": False, "status": "runtime_missing", "error": f"Pipeline runtime is missing: {missing}"}
        started = time.time()
        self.log("info", "LLM pipeline as OS user", {
            "osUser": os_user,
            "action": str(payload.get("action") or "") if isinstance(payload, dict) else "",
            "runtime": os.path.basename(runtime_path),
            "timeoutSeconds": timeout_seconds,
        })
        try:
            argv = [self.python_path, self.pipeline_runtime, runtime_path, "--timeout", str(timeout_seconds)]
            bridged = self.su_session_manager.run_in_session(
                os_user, argv, json.dumps(payload), timeout_ms=int(timeout_seconds * 1000 + 10000)
            )
            parsed = self._parse_pipeline_output(bridged["stdout"], bridged["stderr"])
            self.log("error" if parsed.get("success") is False else "info", "LLM pipeline as OS user done", {
                "osUser": os_user,
                "ok": parsed.get("success") is not False,
                "status": parsed.get("status") or "",
                "ms": int((time.time() - started) * 1000),
            })
            return parsed
        except Exception as error:  # noqa: BLE001
            self.log("error", "LLM pipeline as OS user failed", {"osUser": os_user, "message": str(error)})
            return {"success": False, "status": "runtime_error", "error": str(error)}
