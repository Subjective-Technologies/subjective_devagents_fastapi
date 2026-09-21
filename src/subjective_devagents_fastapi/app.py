"""FastAPI adapter over ChatCore — the backend half of Subjective DevAgents.

Replaces clients/desktop/subjective_devagents/src/webServer.js. Route paths, request
bodies, response shapes and the WebSocket event stream are identical to that server, so
the existing browser UI (clients/desktop/subjective_devagents/web + src renderer) runs
against this unchanged.

Blocking work (CLI subprocesses, pty sessions, history scans) runs in a thread pool via
asyncio.to_thread so one long agent turn never stalls the event loop.

Run locally:

    uvicorn subjective_devagents_fastapi.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import auth, shell_exec, ssh_terminal
from .chat_core import ChatCore
from .llm_service import tail_conversation_for_client
from .system_users import is_valid_username, list_system_users

# Bumped by hand when shipping web UI layout/composer fixes. Also stamped at serve time
# with the live git SHA so View Source proves which build the browser loaded.
WEB_UI_RELEASE = "20260920b"

MAX_BODY_BYTES = 32 * 1024 * 1024  # room for a handful of base64 image/file attachments

# Who may embed this app in an iframe. The dashboard's ST_BACKEND -> My Agents page frames
# it from a different origin, while a zone-wide Cloudflare rule stamps
# `X-Frame-Options: SAMEORIGIN` onto every HTML response in subjectivetechnologies.com —
# which alone forbids that. CSP frame-ancestors takes precedence over X-Frame-Options in
# any browser that supports it, and it is ignored when delivered via <meta http-equiv>,
# so it has to be this header.
FRAME_ANCESTORS = (
    "frame-ancestors 'self' https://dashboard.subjectivetechnologies.com https://dashboard.worktwins.com"
)

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def default_log(level: str, message: str, details: dict | None = None) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    detail_text = json.dumps(details, default=str) if details else ""
    print(f"{stamp} [{level}] {message} {detail_text}", flush=True)


def _client_root() -> Path:
    """Where the browser UI files live.

    Defaults to the DevAgents client next to this repo's services/ directory; override
    with SUBJECTIVE_DEVAGENTS_CLIENT_ROOT when the frontend is deployed elsewhere.
    """
    override = os.environ.get("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT")
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parents[3] / ".." / "clients" / "desktop" / "subjective_devagents").resolve()


def resolve_deploy_version(root: Path) -> str:
    sha = "nogit"
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            or "nogit"
        )
    except (OSError, subprocess.SubprocessError):
        pass  # bare checkout / no git — keep fallback
    dirty = ""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=2
        ).stdout.strip()
        if status:
            dirty = "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{WEB_UI_RELEASE}+{sha}{dirty}"


class EventHub:
    """Fans ChatCore's event stream out to every connected browser tab.

    ChatCore emits from worker threads; each socket owns a bounded queue drained by its
    own task, so a slow tab can never block an agent turn.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queues: set[asyncio.Queue] = set()

    def register(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.queues.add(queue)
        return queue

    def unregister(self, queue: asyncio.Queue) -> None:
        self.queues.discard(queue)

    def publish(self, event: dict) -> None:
        """Thread-safe: called from ChatCore's threads, not the event loop."""
        try:
            self.loop.call_soon_threadsafe(self._publish_now, event)
        except RuntimeError:
            pass  # loop already closed during shutdown

    def _publish_now(self, event: dict) -> None:
        for queue in list(self.queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.queues.discard(queue)


class AppState:
    def __init__(self):
        self.core: ChatCore | None = None
        self.hub: EventHub | None = None
        self.client_root: Path = _client_root()
        self.deploy_version: str = WEB_UI_RELEASE
        self.backend_version: str = WEB_UI_RELEASE
        # deploy_version is a git SHA and only changes on a commit — this process restarts
        # far more often than it commits (a plain reload of an uncommitted edit), so it is
        # the wrong signal for "does an open tab need to reload". A fresh id every startup
        # catches that case too.
        self.instance_id: str = str(uuid.uuid4())
        self.history_refreshing = False
        self._refresh_lock = asyncio.Lock()
        self.d1_last_sync: dict | None = None

    async def refresh_history(self, reason: str) -> dict:
        """Shared by the startup/interval refresh and the manual refresh API so a button
        click cannot stack a second LocalChats pipeline on top of an in-flight run."""
        if self.history_refreshing:
            return {"ok": True, "deferred": True, "status": "busy"}
        async with self._refresh_lock:
            self.history_refreshing = True
            core = self.core
            try:
                started = time.time()
                result = await asyncio.to_thread(core.llm_service.refresh_history)
                if result.get("status") == "deferred":
                    default_log("info", "Local chat history refresh deferred", {"reason": reason})
                    return {"ok": True, "deferred": True, "status": "deferred"}
                if not result.get("success"):
                    default_log("error", "Local chat history pipeline failed", {
                        "reason": reason, "status": result.get("status"), "error": result.get("error"),
                        "ms": int((time.time() - started) * 1000),
                    })
                    return {"ok": False, "status": result.get("status"), "error": result.get("error") or "History refresh failed."}
                count = core.llm_service.count_conversations()
                default_log("info", "Local chat history refreshed", {
                    "reason": reason, "count": count, "ms": int((time.time() - started) * 1000),
                })
                core.broadcast("llm-history-refreshed", {"count": count})
                d1_result = await asyncio.to_thread(self._sync_d1, reason)
                return {"ok": True, "count": count, "d1": d1_result}
            except Exception as error:  # noqa: BLE001
                default_log("error", "Local chat history pipeline crashed", {"reason": reason, "message": str(error)})
                return {"ok": False, "error": str(error)}
            finally:
                self.history_refreshing = False

    def _sync_d1(self, reason: str) -> dict:
        """Best-effort replica. Local JSON stays the live store if D1 is down."""
        from .d1_client import D1Client, D1Error, d1_sync_enabled
        from .d1_sync import conversations_for_sync, sync_conversations

        if not d1_sync_enabled():
            result = {"ok": True, "skipped": True, "reason": "disabled_or_no_credentials"}
            self.d1_last_sync = result
            return result
        if self.core is None:
            return {"ok": False, "error": "ChatCore is not initialized."}
        started = time.time()
        try:
            conversations = conversations_for_sync(self.core.llm_service)
            default_log("info", "D1 conversation sync start", {"reason": reason, "conversations": len(conversations)})
            stats = sync_conversations(D1Client(), conversations)
            stats.update({"ok": True, "reason": reason, "ms": int((time.time() - started) * 1000)})
            default_log("info", "D1 conversation sync done", stats)
            self.d1_last_sync = stats
            return stats
        except D1Error as error:
            result = {"ok": False, "error": str(error)[:400], "reason": reason}
            default_log("error", "D1 conversation sync failed", result)
            self.d1_last_sync = result
            return result
        except Exception as error:  # noqa: BLE001
            result = {"ok": False, "error": str(error)[:400], "reason": reason}
            default_log("error", "D1 conversation sync crashed", result)
            self.d1_last_sync = result
            return result


state = AppState()


@asynccontextmanager
async def lifespan(application: FastAPI):
    loop = asyncio.get_running_loop()
    state.hub = EventHub(loop)
    state.client_root = _client_root()
    state.deploy_version = resolve_deploy_version(state.client_root)
    # The UI can live in a different repository. Its SHA cannot identify this API.
    state.backend_version = resolve_deploy_version(Path(__file__).resolve().parents[2])
    state.instance_id = str(uuid.uuid4())
    core = ChatCore(
        root=str(state.client_root),
        log=default_log,
        emit=lambda event: state.hub.publish(event),
    )
    core.load_settings()
    state.core = core

    # Every "Continue as..." session's pty output is broadcast to whichever tab opened its
    # terminal panel, the moment a login is actually needed.
    original_emit = core._emit

    def emit(event: dict) -> None:
        if event.get("type") == "switchuser-terminal-needed":
            core.attach_su_viewer_broadcast(event.get("username"))
        original_emit(event)

    core._emit = emit

    # This service can read every conversation, list OS accounts and start an agent turn,
    # and it is reachable from the public internet. Say so on every boot when it is open.
    if not auth.auth_enabled():
        default_log("error", "Development authentication mode: anonymous API access enabled.", {})
    else:
        default_log("info", "API authentication enabled", {"mode": auth.auth_mode(), "protects": list(auth.PROTECTED_PREFIXES)})

    default_log("info", "Subjective DevAgents FastAPI starting", {
        "appHome": core.application_home,
        "clientRoot": str(state.client_root),
        "deployVersion": state.deploy_version,
        "webUiRelease": WEB_UI_RELEASE,
    })

    # Report the datasource submodules at startup. A plugin that fails to import degrades
    # silently to the exported bundle, and in a container the bundles are not present at
    # all — so an unreported import failure surfaces much later as a failed agent turn.
    from . import datasources

    report = await asyncio.to_thread(datasources.available)
    loaded = sorted(key for key, entry in report.items() if entry["loaded"])
    missing = {key: entry.get("error", "")[:160] for key, entry in report.items() if not entry["loaded"]}
    default_log("info" if loaded else "error", "Datasource modules resolved", {
        "loaded": loaded,
        "missing": missing,
        "pluginRoots": [str(root) for root in datasources.plugin_roots()],
    })

    # My Agents is the LLM UI. IRC is optional and the default host is often down;
    # reconnect storms painted the status bar as errors on every send.
    if os.environ.get("SUBJECTIVE_CHAT_IRC") == "1":
        await asyncio.to_thread(core.connect_irc)
    else:
        default_log("info", "IRC skipped (set SUBJECTIVE_CHAT_IRC=1 to enable)")

    # Serve whatever the manifest already has right away (no blank sidebar), then run the
    # history pipeline. It is incremental — newest-first, skipping fingerprinted unchanged
    # files — and re-runs on a timer so chats written while the server is up appear
    # without a restart.
    core.broadcast("llm-history-refreshed", {"count": core.llm_service.count_conversations()})
    refresh_every_ms = int(os.environ.get("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS") or 10 * 60 * 1000)

    async def refresh_loop() -> None:
        await state.refresh_history("startup")
        if refresh_every_ms <= 0:
            return
        while True:
            await asyncio.sleep(refresh_every_ms / 1000)
            await state.refresh_history("interval")

    refresh_task = asyncio.create_task(refresh_loop())
    try:
        yield
    finally:
        refresh_task.cancel()
        default_log("info", "Shutting down")
        await asyncio.to_thread(core.shutdown)


app = FastAPI(title="Subjective DevAgents", lifespan=lifespan)

def _serve_ui() -> bool:
    """False on the Hetzner API host. Set SUBJECTIVE_DEVAGENTS_SERVE_UI=1 to ship HTML too."""
    return os.environ.get("SUBJECTIVE_DEVAGENTS_SERVE_UI", "1").strip().lower() in {"1", "true", "yes"}


def _web_ui_url() -> str:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_WEB_URL") or "https://devagents.subjectivetechnologies.com/").rstrip("/") + "/"


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    """Reject unauthenticated callers before any handler runs.

    Registered after the version middleware below, which means it runs *first* — an
    unauthenticated request must not reach a handler that reads conversations or starts a
    turn. Static UI files stay public: they are an empty shell without the API.
    """
    if auth.path_requires_token(request.url.path):
        supplied = auth.token_from_request(request.headers)
        if not await asyncio.to_thread(auth.token_is_valid, supplied):
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": "Unauthorized."},
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


@app.middleware("http")
async def add_version_and_no_store(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-DevAgents-Version", state.deploy_version)
    response.headers.setdefault("X-DevAgents-Backend-Version", state.backend_version)
    if request.url.path.startswith("/api/"):
        # Without this a plain reload can serve a stale cached GET
        # /api/llm/conversations (e.g. a chat name from before a rename).
        response.headers["Cache-Control"] = "no-store"
    return response


# CORS must wrap authentication: preflight needs no credential, and 401s need CORS headers.
# The UI may be served from this process or hosted separately; allow both.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in (os.environ.get("SUBJECTIVE_DEVAGENTS_CORS_ORIGINS") or "*").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



def core() -> ChatCore:
    if state.core is None:
        raise RuntimeError("ChatCore is not initialized.")
    return state.core


# --- API routes (1:1 with webServer.js) ------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness only - deliberately says nothing about conversations, users or the host."""
    return {"ok": True}


@app.get("/api/auth/config")
async def api_auth_config() -> dict:
    return {"mode": auth.auth_mode(), "required": auth.auth_enabled()}


@app.get("/api/auth/session")
async def api_auth_session() -> dict:
    return {"ok": True}


@app.get("/api/version")
async def api_version() -> dict:
    return {
        "ok": True,
        "release": WEB_UI_RELEASE,
        "version": state.deploy_version,
        "backendVersion": state.backend_version,
        "instanceId": state.instance_id,
        "servedAt": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/config")
async def api_config() -> dict:
    return core().public_config()


@app.post("/api/save-config")
async def api_save_config(body: dict = Body(default={})) -> dict:
    connection = body.get("active") or body or {}
    await asyncio.to_thread(core().apply_connection_and_reconnect, connection, body.get("saveBookmark") is not False)
    return {"ok": True, "config": core().public_config()}


@app.post("/api/connect")
async def api_connect() -> dict:
    await asyncio.to_thread(core().reconnect_now)
    return {"ok": True}


@app.post("/api/disconnect")
async def api_disconnect() -> dict:
    await asyncio.to_thread(core().disconnect_irc, True)
    return {"ok": True}


@app.post("/api/send-message")
async def api_send_message(body: dict = Body(default={})) -> dict:
    instance = core()
    target = body.get("target") or instance.config["channel"]
    clean = " ".join(str(body.get("text") or "").split())
    if not clean:
        return {"ok": False}
    if instance.write_irc(f"PRIVMSG {target} :{clean}"):
        instance.broadcast("message", {
            "target": target,
            "from": instance.current_nick,
            "text": clean,
            "mine": True,
            "time": time.time() * 1000,
            "private": not str(target).startswith("#"),
        })
        return {"ok": True}
    return {"ok": False}


@app.post("/api/execute-command")
async def api_execute_command(body: dict = Body(default={})) -> dict:
    instance = core()
    target = body.get("target") or instance.config["channel"]
    result = await asyncio.to_thread(instance.execute_slash_command, body.get("command") or "", target)
    return {"ok": bool(result.get("ok")), "focusTarget": result.get("focusTarget") or ""}


@app.get("/api/llm/conversations")
async def api_llm_conversations(limit: int = Query(default=200), q: str = Query(default="")) -> dict:
    try:
        capped = min(500, max(1, int(limit or 200)))
    except (TypeError, ValueError):
        capped = 200
    try:
        listed = await asyncio.to_thread(core().llm_service.list_conversations, str(q or "").strip(), capped)
        return {"ok": True, "conversations": listed["conversations"], "total": listed["total"]}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": str(error), "conversations": []}


@app.get("/api/llm/conversation")
async def api_llm_conversation(target: str = Query(default="")) -> dict:
    instance = core()
    try:
        live = await asyncio.to_thread(instance.llm_service.reload_conversation, str(target))
        conversation = tail_conversation_for_client(
            live or await asyncio.to_thread(instance.llm_service.find_conversation, str(target), True)
        )
        default_log("info" if conversation else "error", "LLM conversation hydrate", {
            "target": target,
            "ok": bool(conversation),
            "provider": (conversation or {}).get("provider") or "",
            "label": (conversation or {}).get("label") or "",
            "sessionId": (conversation or {}).get("sessionId") or "",
            "messageCount": len((conversation or {}).get("messages") or []),
            "totalMessages": (conversation or {}).get("messageCount") or 0,
            "truncated": bool((conversation or {}).get("truncated")),
            "origin": (conversation or {}).get("origin") or "",
        })
        if conversation:
            # Authoritative busy state on hydrate, so opening a chat recovers from a missed
            # llm-busy:false (dropped socket) rather than inheriting a stale wedge.
            conversation["busy"] = str(target) in instance.llm_service.busy
            return {"ok": True, "conversation": conversation}
        return {"ok": False, "error": "Conversation was not found."}
    except Exception as error:  # noqa: BLE001
        default_log("error", "LLM conversation hydrate crashed", {"target": target, "message": str(error)})
        return {"ok": False, "error": str(error)}


@app.post("/api/llm/create-conversation")
async def api_llm_create_conversation(body: dict = Body(default={})) -> dict:
    instance = core()
    try:
        conversation = await asyncio.to_thread(
            instance.llm_service.create_conversation,
            body.get("provider"),
            body.get("name") or "",
            body.get("workspace") or "",
            body.get("osUser") or "",
        )
        instance.activate_llm_irc_channel(conversation)
        instance.broadcast("llm-conversation", {"conversation": conversation})
        return {"ok": True, "conversation": conversation}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": str(error)}


@app.post("/api/llm/rename-conversation")
async def api_llm_rename_conversation(body: dict = Body(default={})) -> dict:
    instance = core()
    try:
        result = await asyncio.to_thread(
            instance.llm_service.rename_conversation, str(body.get("target") or ""), str(body.get("name") or "")
        )
        if result.get("ok") and result.get("conversation"):
            conversation = result["conversation"]
            default_log("info", "LLM conversation renamed", {"target": conversation["target"], "label": conversation["label"]})
            instance.broadcast("llm-conversation-renamed", {"target": conversation["target"], "label": conversation["label"]})
        return result
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": str(error)}


@app.post("/api/llm/activate-conversation")
async def api_llm_activate_conversation(body: dict = Body(default={})) -> dict:
    instance = core()
    conversation = await asyncio.to_thread(instance.llm_service.find_conversation, str(body.get("target") or ""))
    if not conversation:
        return {"ok": False, "error": "Conversation was not found."}
    await asyncio.to_thread(instance.llm_service.mark_active, conversation["target"])
    result = instance.activate_llm_irc_channel(conversation)
    # Authoritative busy state here too, not just on the one-time hydrate fetch: this call
    # runs on every activation (setActiveTarget calls it unconditionally), so it is what
    # actually recovers a conversation whose buttons got stuck disabled by a missed
    # llm-busy:false — e.g. a dropped websocket during a turn — for as long as the tab
    # stays open, not just the first time that conversation is ever opened.
    if isinstance(result, dict):
        result["busy"] = conversation["target"] in instance.llm_service.busy
    return result


@app.post("/api/llm/refresh-history")
async def api_llm_refresh_history() -> dict:
    return await state.refresh_history("manual")


@app.post("/api/llm/import-chats")
async def api_llm_import_chats(file: UploadFile = File(...)) -> dict:
    instance = core()
    try:
        data = await file.read()
        result = await asyncio.to_thread(instance.llm_service.import_chats_zip, data)
        if result.get("imported"):
            instance.broadcast("llm-history-refreshed", {"count": instance.llm_service.count_conversations()})
        return result
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": str(error)}


# Off by default: this lets a caller put words in an assistant's mouth without running a
# CLI turn, which is exactly what an end-to-end UI test wants and exactly what production
# must never expose. Set SUBJECTIVE_DEVAGENTS_ENABLE_TEST_ENDPOINTS=1 only on a test
# deployment, never on the public one.
TEST_ENDPOINTS_ENABLED = os.environ.get("SUBJECTIVE_DEVAGENTS_ENABLE_TEST_ENDPOINTS") == "1"


@app.post("/api/llm/test/inject-assistant-message")
async def api_llm_test_inject_assistant_message(body: dict = Body(default={})) -> dict:
    if not TEST_ENDPOINTS_ENABLED:
        return {"ok": False, "error": "Test endpoints are disabled on this deployment."}
    instance = core()
    target = str(body.get("target") or "")
    text = str(body.get("text") or "")
    result = await asyncio.to_thread(instance.llm_service.inject_test_assistant_message, target, text)
    if result.get("ok"):
        from .prompt_options import annotate_message

        provider = (result.get("conversation") or {}).get("provider") or ""
        instance.broadcast("llm-message", {
            "target": target,
            "message": annotate_message(result["assistantMessage"]),
            "provider": provider,
        })
    return result


@app.get("/api/d1/sync")
async def api_d1_sync_status() -> dict:
    return {"ok": True, "last": state.d1_last_sync}


@app.post("/api/d1/sync")
async def api_d1_sync_now() -> dict:
    return await asyncio.to_thread(state._sync_d1, "manual")


@app.post("/api/llm/reload-conversation")
async def api_llm_reload_conversation(body: dict = Body(default={})) -> dict:
    instance = core()
    target = str(body.get("target") or "")
    try:
        conversation = tail_conversation_for_client(await asyncio.to_thread(instance.llm_service.reload_conversation, target))
        default_log("info" if conversation else "error", "LLM conversation live reload", {
            "target": target, "ok": bool(conversation), "messageCount": len((conversation or {}).get("messages") or []),
        })
        if conversation:
            instance.broadcast("llm-conversation", {"conversation": conversation})
            return {"ok": True, "conversation": conversation}
        return {"ok": False, "error": "Conversation was not found."}
    except Exception as error:  # noqa: BLE001
        default_log("error", "LLM conversation live reload crashed", {"target": target, "message": str(error)})
        return {"ok": False, "error": str(error)}


@app.get("/api/agents")
async def api_agents() -> dict:
    """The agent registry in workflow/agents — one entry per row of registry.md."""
    from . import agents as agents_module

    return await asyncio.to_thread(agents_module.list_agents)


@app.get("/api/agents/role")
async def api_agent_role(agent: str = Query(default="")) -> dict:
    from . import agents as agents_module

    return await asyncio.to_thread(agents_module.read_role, agent)


@app.post("/api/agents/role")
async def api_save_agent_role(body: dict = Body(default={})) -> dict:
    """Save the full text of an existing agent's role file (the "Edit agent" dialog)."""
    from . import agents as agents_module

    return await asyncio.to_thread(
        agents_module.write_role, str(body.get("id") or ""), str(body.get("text") or "")
    )


@app.post("/api/agents")
async def api_create_agent(body: dict = Body(default={})) -> dict:
    """Add an agent: a role file in roles/ and a row in registry.md, or neither."""
    from . import agents as agents_module

    return await asyncio.to_thread(
        agents_module.create_agent,
        str(body.get("id") or ""),
        str(body.get("capability") or ""),
        str(body.get("model") or ""),
        str(body.get("mission") or ""),
    )


@app.post("/api/agents/model")
async def api_set_agent_model(body: dict = Body(default={})) -> dict:
    """Re-point an agent at a different CLI.

    Written to the role file's Runtime section, never the registry table: the registry's
    contract is that models are runtime metadata and not part of agent identity.
    """
    from . import agents as agents_module

    return await asyncio.to_thread(
        agents_module.set_agent_model, str(body.get("id") or ""), str(body.get("model") or "")
    )


@app.post("/api/llm/cancel-message")
async def api_llm_cancel_message(body: dict = Body(default={})) -> dict:
    """Stop the turn in flight for a conversation.

    Idempotent and safe on an idle chat: `running` says whether a turn was actually in
    flight, `stopped` whether a process was signalled. A Stop pressed in the gap before
    the CLI exists is latched and applied as soon as the process appears.
    """
    instance = core()
    target = str(body.get("target") or "")
    result = await asyncio.to_thread(instance.llm_service.cancel_message, target)
    if result.get("ok") and result.get("running"):
        instance.broadcast("llm-busy", {"target": target, "busy": False})
        instance.broadcast("system", {"target": target, "text": "Stopped."})
    return result


@app.get("/api/llm/providers")
async def api_llm_providers() -> dict:
    """The agents a new chat can actually be created with.

    The picker used to hard-code its options in two index.html files, which drifted from
    the backend: Grok was creatable here but absent from the UI, while Cursor was offered
    despite its datasource module being missing on this host. Serving the list is the only
    way the two stay in step.

    `available` reports whether the in-process datasource module imports. It is a hint, not
    a gate — a provider whose module is missing can still run through its exported bundle,
    so every writable provider is listed either way.
    """
    from . import datasources
    from .llm_service import DEFAULT_PROVIDER, WRITABLE_PROVIDERS

    report = await asyncio.to_thread(datasources.available)
    return {
        "ok": True,
        "default": DEFAULT_PROVIDER,
        "providers": [
            {
                "key": key,
                "label": provider["label"],
                "available": bool((report.get(key) or {}).get("loaded")),
            }
            for key, provider in WRITABLE_PROVIDERS.items()
        ],
    }


@app.get("/api/llm/status")
async def api_llm_status(provider: str = Query(default="codex")) -> dict:
    result = await asyncio.to_thread(core().llm_service.status, provider)
    return {"ok": bool(result.get("success")), **result}


@app.get("/api/llm/sessions")
async def api_llm_sessions(provider: str = Query(default="")) -> dict:
    listed = await asyncio.to_thread(core().llm_service.list_provider_sessions, provider)
    return {"ok": True, **listed}


@app.post("/api/llm/send-message")
async def api_llm_send_message(body: dict = Body(default={})) -> dict:
    instance = core()
    target = str(body.get("target") or "")
    text = str(body.get("text") or "").strip()
    files = body.get("files") or []
    if not target or not text:
        return {"ok": False, "status": "invalid_request", "error": "Missing target or text."}
    conversation = await asyncio.to_thread(instance.llm_service.find_conversation, target)
    if not conversation:
        return {"ok": False, "status": "not_found", "error": "Conversation was not found."}
    # Cloudflare (and most reverse proxies) close idle HTTP at ~100s with 524, while
    # Cursor/Claude/Codex turns are allowed 30 minutes. Accept here and deliver the
    # assistant reply on the WebSocket (llm-message / llm-busy / system).
    default_log("info", "LLM send accepted (async)", {
        "target": target, "provider": conversation["provider"], "promptChars": len(text), "files": len(files),
    })

    async def run_send() -> None:
        try:
            result = await asyncio.to_thread(instance.send_llm_message, target, text, files)
            default_log("info" if result.get("ok") else "error", "LLM send background finished", {
                "target": target, "ok": bool(result.get("ok")),
                "status": result.get("status") or "", "error": result.get("error") or "",
            })
        except Exception as error:  # noqa: BLE001
            default_log("error", "LLM send background crash", {"target": target, "message": str(error)})

    asyncio.create_task(run_send())
    return {"ok": True, "pending": True, "target": target}


@app.get("/api/datasources")
async def api_datasources() -> dict:
    """Which datasource submodules resolved, and from where.

    The first thing to check when a provider misbehaves: it says whether the service is
    calling the plugin package in-process or has fallen back to the exported bundle.
    """
    from . import datasources

    report = await asyncio.to_thread(datasources.available)
    return {
        "ok": True,
        "datasources": report,
        "pluginRoots": [str(root) for root in datasources.plugin_roots()],
        "usingModules": core().llm_service.use_datasource_modules,
    }


@app.get("/api/system/users")
async def api_system_users() -> dict:
    try:
        return await asyncio.to_thread(list_system_users)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "users": [], "error": str(error)}


@app.get("/api/switchuser/status")
async def api_switchuser_status(username: str = Query(default="")) -> dict:
    instance = core()
    if username:
        return {"ok": True, "status": instance.su_session_manager.get_status(str(username))}
    return {"ok": True, "statuses": instance.su_session_manager.list_statuses()}


@app.get("/api/context-menu")
async def api_context_menu(target: str = Query(default="")) -> dict:
    model = await asyncio.to_thread(core().build_context_menu_model, str(target))
    return {"ok": True, **model}


@app.post("/api/switchuser/continue-as")
async def api_switchuser_continue_as(body: dict = Body(default={})) -> dict:
    instance = core()
    username = str(body.get("username") or "")
    result = await asyncio.to_thread(
        instance.continue_conversation_as_user,
        str(body.get("target") or ""),
        username,
        str(body["provider"]) if body.get("provider") else None,
    )
    # Attach the broadcast viewer eagerly: continue_conversation_as_user only does this via
    # the switchuser-terminal-needed event when a login is actually needed, but attaching is
    # a cheap deduped no-op, and this way a tab that opens the terminal panel right after
    # this call never misses a race on the first chunk.
    if is_valid_username(username):
        instance.attach_su_viewer_broadcast(username)
    return result


@app.post("/api/switchuser/disconnect")
async def api_switchuser_disconnect(body: dict = Body(default={})) -> dict:
    instance = core()
    username = str(body.get("username") or "")
    await asyncio.to_thread(instance.su_session_manager.disconnect, username)
    instance.detach_su_viewer_broadcast(username)
    return {"ok": True}


@app.post("/api/switchuser/pty-input")
async def api_switchuser_pty_input(body: dict = Body(default={})) -> dict:
    core().su_session_manager.write_input(str(body.get("username") or ""), str(body.get("data") or ""))
    return {"ok": True}


@app.post("/api/switchuser/pty-resize")
async def api_switchuser_pty_resize(body: dict = Body(default={})) -> dict:
    try:
        cols = int(body.get("cols") or 0)
        rows = int(body.get("rows") or 0)
    except (TypeError, ValueError):
        cols = rows = 0
    core().su_session_manager.resize(str(body.get("username") or ""), cols, rows)
    return {"ok": True}


@app.post("/api/dcc/send-image-data")
async def api_dcc_send_image_data(body: dict = Body(default={})) -> dict:
    instance = core()
    raw = body.get("bytes")
    if isinstance(raw, str):
        buffer = base64.b64decode(raw)
    else:
        buffer = bytes(bytearray(raw or []))
    return await asyncio.to_thread(
        instance.send_dcc_buffer,
        body.get("target") or instance.config["channel"],
        buffer,
        body.get("fileName") or f"pasted-{int(time.time() * 1000)}.png",
        body.get("mimeType") or "",
    )


@app.post("/api/file/open")
async def api_file_open() -> dict:
    return {"ok": False, "error": "Opening local files is only available in the Subjective Chat desktop app."}


@app.post("/api/file/reveal")
async def api_file_reveal() -> dict:
    return {"ok": False, "error": "Show in folder is only available in the Subjective Chat desktop app."}


@app.post("/api/log")
async def api_log(body: dict = Body(default={})) -> dict:
    default_log(body.get("level") or "info", body.get("message") or "Browser log", body.get("details") or {})
    return {"ok": True}


@app.get("/api/exec/status")
async def api_exec_status() -> dict:
    """Whether the phone can run commands here, and where they run from.

    The UI reads this to decide whether to show the terminal input at all, so it reflects
    both interlocks — enabled *and* authenticated.
    """
    try:
        shell_exec.ensure_available()
        return {"ok": True, "enabled": True, "cwd": shell_exec.default_workdir()}
    except shell_exec.ExecDisabled as reason:
        return {"ok": True, "enabled": False, "reason": str(reason)}


@app.post("/api/exec")
async def api_exec(body: dict = Body(default={})) -> dict:
    """Run one shell command on this host and return its output.

    Already behind the API token (the require_api_token middleware ran before this), and
    refuses outright unless execution is explicitly enabled — see shell_exec for the
    interlocks. Every invocation is logged with its command, so the journal is the audit
    trail.
    """
    command = str(body.get("command") or "")
    try:
        shell_exec.ensure_available()
    except shell_exec.ExecDisabled as reason:
        # 403, not 401: the caller authenticated fine, the capability is switched off.
        return JSONResponse(status_code=403, content={"ok": False, "error": str(reason)})

    default_log("info", "Exec request", {
        "command": shell_exec.summarize_for_log(command),
        "cwd": body.get("cwd") or shell_exec.default_workdir(),
    })
    result = await asyncio.to_thread(
        shell_exec.run_command, command, str(body.get("cwd") or ""), body.get("timeout")
    )
    default_log("info" if result.get("ok") else "warn", "Exec result", {
        "status": result.get("status") or "", "exitCode": result.get("exit_code"),
        "ms": result.get("duration_ms"), "truncated": result.get("truncated"),
    })
    return result


# --- WebSocket -------------------------------------------------------------------------


async def _authenticate_socket(socket: WebSocket) -> tuple[bool, dict | None]:
    """Accept and authenticate a socket, or close it and report failure.

    Returns `(ok, claims)`. Owner browsers send credentials in the first frame rather
    than the query string, because URLs are logged and a token in one outlives the
    request. Shared by every socket this service exposes so a new one cannot quietly
    ship with weaker authentication than /ws.
    """
    accepted = False
    claims = None
    if auth.auth_enabled():
        supplied = auth.token_from_request(socket.headers, socket.query_params)
        if auth.auth_mode() in {"owner", "token"} and not supplied:
            await socket.accept()
            accepted = True
            try:
                message = await asyncio.wait_for(socket.receive_json(), timeout=5)
                if not isinstance(message, dict) or message.get("type") != "auth":
                    raise ValueError("Expected authentication")
                supplied = message.get("token", "")
            except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
                await socket.close(code=4401)
                return False, None
        claims = await asyncio.to_thread(auth.authenticated_claims, supplied)
        if claims is None:
            await socket.close(code=4401 if accepted else 1008)
            return False, None
    if not accepted:
        await socket.accept()
    return True, claims


@app.websocket("/ws")
async def websocket_events(socket: WebSocket) -> None:
    # Authenticate before registering with the broadcast hub or sending any data.
    ok, claims = await _authenticate_socket(socket)
    if not ok:
        return
    expires_at = claims.get("exp") if claims else None
    hub = state.hub
    instance = core()
    queue = hub.register()
    default_log("info", "WebSocket client connected", {"clients": len(hub.queues)})
    try:
        await socket.send_text(json.dumps({"type": "config", **instance.public_config()}))
        await socket.send_text(json.dumps({
            "type": "status",
            "state": "connected" if instance.connected else "idle",
            "text": f"Connected to {instance.config['host']}:{instance.config['port']}" if instance.connected else "Ready",
        }))

        async def pump_incoming() -> None:
            # The UI never sends over the socket; draining keeps disconnects prompt.
            while True:
                await socket.receive_text()

        reader = asyncio.create_task(pump_incoming())
        try:
            while True:
                try:
                    # A heartbeat while idle: Cloudflare closes a WebSocket with no traffic
                    # for ~100s, and when it does mid-turn the client loses the reply and the
                    # llm-busy:false that re-enables the UI. A periodic frame keeps the
                    # connection alive through the proxy so events actually arrive.
                    remaining = expires_at - time.time() if expires_at else 25
                    if remaining <= 0:
                        await socket.close(code=4401)
                        break
                    event = await asyncio.wait_for(queue.get(), timeout=min(25, remaining))
                except asyncio.TimeoutError:
                    if expires_at and time.time() >= expires_at:
                        await socket.close(code=4401)
                        break
                    event = {"type": "ping", "t": int(time.time() * 1000)}
                await socket.send_text(json.dumps(event, default=str))
                if reader.done():
                    break
        finally:
            reader.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unregister(queue)
        default_log("info", "WebSocket client disconnected", {"clients": len(hub.queues)})


@app.websocket("/ws/ssh")
async def websocket_ssh_terminal(socket: WebSocket) -> None:
    """A live shell on another machine, for the duration of this socket.

    Protocol, client to server:
      {"type": "auth",      "token": "..."}          (only when the socket needs it)
      {"type": "open",      "host": "...", ...}      connection fields; host required
                            "pty": true, "cols": 120, "rows": 32   for terminal mode
      {"type": "input",     "command": "ls -la"}     one command (both modes)
      {"type": "data",      "data": "ls\r"}          raw keystrokes    (pty mode only)
      {"type": "resize",    "cols": 120, "rows": 32}                   (pty mode only)
      {"type": "interrupt"}
      {"type": "close"}

    Server to client:
      {"type": "hello",   "maxSessions": n, "idleTimeout": n}
      {"type": "opening", "connection": {...}}       secrets never appear here
      {"type": "ready",   "sessionId": "...", "mode": "command"|"pty", ...}
      {"type": "closed",  "code": 0}                 the remote shell is gone
      {"type": "error",   "error": "..."}
    command mode adds:
      {"type": "output",  "stream": "stdout"|"stderr", "data": "..."}
      {"type": "exit",    "code": 0, "cwd": "..."}   one per command
    pty mode adds:
      {"type": "data",    "data": "..."}             raw terminal output

    **The two modes are different products, not a flag.** Command mode frames each
    command, so a UI or an agent gets exit codes, a working directory and separated
    streams. A terminal has no machine-readable command boundary — the shell's prompt
    is cosmetic and a sentinel would be echoed back to the user — so pty mode gives
    up all of that, and in return `claude` and `codex` with no arguments, `top`, `vim`
    and `sudo` prompts work, and Ctrl-C reaches the remote job instead of dropping
    the connection.

    One session per socket: a terminal is a connection, and multiplexing several onto
    one socket would only move the bookkeeping into the client.
    """
    ok, claims = await _authenticate_socket(socket)
    if not ok:
        return

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    session = None
    session_id = ""
    expires_at = claims.get("exp") if claims else None

    async def say(payload: dict) -> None:
        await socket.send_text(json.dumps(payload, default=str))

    try:
        ssh_terminal.ensure_available()
    except ssh_terminal.TerminalDisabled as error:
        await say({"type": "error", "error": str(error)})
        await socket.close(code=4403)
        return

    try:
        await say({
            "type": "hello",
            "maxSessions": ssh_terminal.max_sessions(),
            "idleTimeout": ssh_terminal.idle_timeout_seconds(),
        })

        while True:
            # The socket may carry an auth frame first; keep reading until `open`.
            message = await socket.receive_json()
            if isinstance(message, dict) and message.get("type") == "open":
                break
            if isinstance(message, dict) and message.get("type") in {"auth", "ping"}:
                continue
            await say({"type": "error", "error": "Open a session before sending anything else."})

        try:
            connection = ssh_terminal.connection_from_request(message)
        except ValueError as error:
            await say({"type": "error", "error": str(error)})
            await socket.close(code=4400)
            return

        use_pty = bool(message.get("pty"))
        cols, rows = ssh_terminal.clamp_size(message.get("cols"), message.get("rows"))
        described = ssh_terminal.describe(connection)
        await say({"type": "opening", "connection": described, "mode": "pty" if use_pty else "command"})
        default_log("info", "SSH terminal opening", {**described, "pty": use_pty})

        # Every event the shell produces is pushed onto an asyncio queue from the
        # reader threads, so the socket never polls and output reaches the browser as
        # soon as the remote writes it.
        def forward(event: dict) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        try:
            instance, session = await asyncio.to_thread(
                lambda: ssh_terminal.open_session(connection, pty=use_pty, cols=cols, rows=rows)
            )
        except (LookupError, ValueError) as error:
            await say({"type": "error", "error": str(error)})
            await socket.close(code=4400)
            return

        started = await asyncio.to_thread(session.start)
        if started.get("type") != "ready":
            await say({"type": "error", "error": started.get("error") or "Could not open the session."})
            await asyncio.to_thread(session.close)
            default_log("error", "SSH terminal failed to open", {
                **described, "error": started.get("error") or "",
            })
            return

        session_id = uuid.uuid4().hex
        try:
            ssh_terminal.sessions.register(session_id, session, connection)
        except ssh_terminal.TerminalBusy as error:
            await say({"type": "error", "error": str(error)})
            await asyncio.to_thread(session.close)
            await socket.close(code=4429)
            return

        session.attach(forward)
        await say({
            "type": "ready",
            "sessionId": session_id,
            "mode": "pty" if use_pty else "command",
            "cwd": started.get("cwd") or "",
            "cols": started.get("cols"),
            "rows": started.get("rows"),
            "term": started.get("term"),
        })
        default_log("info", "SSH terminal ready", {
            **described, "sessionId": session_id, "pty": use_pty,
        })

        async def pump_events() -> None:
            while True:
                event = await events.get()
                await say(event)
                if event.get("type") == "closed":
                    return

        async def pump_input() -> None:
            while True:
                frame = await socket.receive_json()
                if not isinstance(frame, dict):
                    continue
                kind = frame.get("type")
                ssh_terminal.sessions.touch(session_id)
                if kind == "input":
                    command = str(frame.get("command") or "")
                    if len(command.encode("utf-8", "replace")) > ssh_terminal.MAX_COMMAND_BYTES:
                        await say({"type": "error", "error": "That command is too long."})
                        continue
                    # In pty mode there is no command concept — a line is keystrokes
                    # plus Return, which is what a terminal's own input would be.
                    ok = await (
                        asyncio.to_thread(session.write, command + "\r")
                        if use_pty
                        else asyncio.to_thread(session.send_command, command)
                    )
                    if not ok:
                        await say({"type": "error", "error": "The remote shell is no longer accepting input."})
                        return
                elif kind == "data":
                    if not use_pty:
                        await say({"type": "error", "error": "Raw input needs a pty session."})
                        continue
                    data = str(frame.get("data") or "")
                    if len(data.encode("utf-8", "replace")) > ssh_terminal.MAX_INPUT_BYTES:
                        await say({"type": "error", "error": "That input is too large."})
                        continue
                    if not await asyncio.to_thread(session.write, data):
                        await say({"type": "error", "error": "The terminal is no longer accepting input."})
                        return
                elif kind == "resize":
                    if not use_pty:
                        continue
                    width, height = ssh_terminal.clamp_size(frame.get("cols"), frame.get("rows"))
                    await asyncio.to_thread(session.resize, width, height)
                elif kind == "interrupt":
                    await asyncio.to_thread(session.interrupt)
                elif kind == "close":
                    return
                elif kind == "ping":
                    await say({"type": "pong", "t": int(time.time() * 1000)})

        async def watch_deadlines() -> None:
            while True:
                await asyncio.sleep(15)
                if expires_at and time.time() >= expires_at:
                    await say({"type": "error", "error": "The session credential expired."})
                    return
                if ssh_terminal.sessions.idle_seconds(session_id) > ssh_terminal.idle_timeout_seconds():
                    await say({"type": "error", "error": "Closed after being idle."})
                    return

        tasks = [
            asyncio.create_task(pump_events()),
            asyncio.create_task(pump_input()),
            asyncio.create_task(watch_deadlines()),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        # Nothing here may `await` before the session is dealt with. A client
        # disconnect cancels this coroutine, and the first await in a finally then
        # raises CancelledError immediately — which would leave a registered session
        # and a live SSH connection behind for every closed tab.
        if session_id:
            ssh_terminal.sessions.unregister(session_id)
            default_log("info", "SSH terminal closed", {"sessionId": session_id})
        if session is not None:
            # Closing waits for ssh to exit, so it runs on a thread that outlives
            # this task rather than blocking the event loop.
            threading.Thread(target=session.close, daemon=True).start()
        try:
            await socket.close()
        except RuntimeError:
            pass


@app.get("/ssh")
async def ssh_terminal_page() -> Response:
    """A standalone console for /ws/ssh, for driving this service on its own.

    The DevAgents client cannot do it: its page pins the production API origin in a
    meta tag and a CSP and gates itself behind Firebase, so it will not talk to a
    local backend. This page is same-origin with the socket and has no dependencies.

    Served only while the terminal is enabled, so it cannot turn up on a deployment
    that has the feature switched off.
    """
    if not ssh_terminal.terminal_enabled():
        return JSONResponse(
            {
                "ok": False,
                "error": "The SSH terminal is disabled. Set "
                         "SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED=1 to enable it.",
            },
            status_code=404,
        )
    page = Path(__file__).parent / "static" / "ssh_terminal.html"
    return Response(
        content=page.read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
        # A console for opening shells is never worth serving from a cache.
        headers={"Cache-Control": "no-store"},
    )


# Only these, by exact name: the directory holds a page and a vendored terminal
# emulator, and a path parameter here would be a file-read primitive.
_SSH_ASSETS = {
    "xterm.js": "text/javascript; charset=utf-8",
    "xterm.css": "text/css; charset=utf-8",
    "xterm.LICENSE": "text/plain; charset=utf-8",
}


@app.get("/ssh/assets/{name}")
async def ssh_terminal_asset(name: str) -> Response:
    """xterm.js and its stylesheet, for the console's terminal mode.

    Vendored into this service rather than loaded from a CDN: the page must work on a
    machine with no internet, and a terminal is not something to hand to a third
    party's script host.
    """
    media_type = _SSH_ASSETS.get(name)
    if not media_type or not ssh_terminal.terminal_enabled():
        return JSONResponse({"ok": False, "error": "Not found."}, status_code=404)
    asset = Path(__file__).parent / "static" / name
    return Response(
        content=asset.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --- saved SSH connections -------------------------------------------------------------


def _ssh_app_home() -> str:
    """Where the connection store lives — the same app home the transcript uses."""
    return core().llm_service.app_home


@app.get("/api/ssh/connections")
async def api_ssh_connections() -> dict:
    """Saved connections, the operator's ~/.ssh keys, and their config aliases.

    All three in one response because the picker shows all three together, and two
    of them are free to read. Passwords and passphrases are never included — only
    booleans saying whether one is stored.
    """
    from . import ssh_connections

    home = _ssh_app_home()
    connections, keys, hosts = await asyncio.gather(
        asyncio.to_thread(ssh_connections.list_connections, home),
        asyncio.to_thread(ssh_connections.list_keys, home),
        asyncio.to_thread(ssh_connections.list_config_hosts),
    )
    return {
        "ok": True,
        "connections": connections,
        "keys": keys,
        "configHosts": hosts,
        "authMethods": list(ssh_connections.AUTH_METHODS),
        "hostKeyPolicies": list(ssh_connections.HOST_KEY_POLICIES),
    }


@app.post("/api/ssh/connections")
async def api_create_ssh_connection(body: dict = Body(default={})) -> Response:
    from . import ssh_connections

    try:
        created = await asyncio.to_thread(
            ssh_connections.create, _ssh_app_home(), body or {}
        )
    except ValueError as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=400)
    default_log("info", "SSH connection saved", {"id": created["id"], "host": created["host"]})
    return JSONResponse({"ok": True, "connection": created})


@app.post("/api/ssh/connections/update")
async def api_update_ssh_connection(body: dict = Body(default={})) -> Response:
    from . import ssh_connections

    try:
        updated = await asyncio.to_thread(
            ssh_connections.update, _ssh_app_home(), str(body.get("id") or ""), body or {}
        )
    except ValueError as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=400)
    except LookupError as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=404)
    return JSONResponse({"ok": True, "connection": updated})


@app.post("/api/ssh/connections/delete")
async def api_delete_ssh_connection(body: dict = Body(default={})) -> Response:
    from . import ssh_connections

    removed = await asyncio.to_thread(
        ssh_connections.delete, _ssh_app_home(), str(body.get("id") or "")
    )
    if not removed:
        return JSONResponse({"ok": False, "error": "No such connection."}, status_code=404)
    default_log("info", "SSH connection deleted", {"id": str(body.get("id") or "")})
    return JSONResponse({"ok": True})


@app.post("/api/ssh/keys")
async def api_upload_ssh_key(body: dict = Body(default={})) -> Response:
    """Take a private key from the browser and keep it for future connections.

    The server's own ~/.ssh is the *server's*: on a deployed instance it holds the
    box's keys, not the operator's, so without this there is no way to reach a
    machine the box was never set up for. The uploaded key is written private and
    then behaves exactly like one already in ~/.ssh — named by path, never read
    back, never returned.
    """
    from . import ssh_connections

    try:
        saved = await asyncio.to_thread(
            ssh_connections.save_uploaded_key,
            _ssh_app_home(),
            str(body.get("name") or ""),
            str(body.get("content") or ""),
        )
    except ValueError as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=400)
    except OSError as error:
        return JSONResponse({"ok": False, "error": f"Could not store the key: {error}"}, status_code=500)
    # The name, never the key. This line goes to a log file.
    default_log("info", "SSH key uploaded", {"name": saved["name"]})
    return JSONResponse({"ok": True, "key": saved})


@app.post("/api/ssh/keys/delete")
async def api_delete_ssh_key(body: dict = Body(default={})) -> Response:
    """Delete an uploaded key. Keys in the server's own ~/.ssh are never touched."""
    from . import ssh_connections

    removed = await asyncio.to_thread(
        ssh_connections.delete_uploaded_key, _ssh_app_home(), str(body.get("path") or "")
    )
    if not removed:
        return JSONResponse(
            {"ok": False, "error": "No such uploaded key. Keys from the server's own "
                                   "~/.ssh cannot be deleted from here."},
            status_code=404,
        )
    return JSONResponse({"ok": True})


@app.post("/api/ssh/connections/test")
async def api_test_ssh_connection(body: dict = Body(default={})) -> Response:
    """Try the connection and report what happened, without opening a chat.

    The equivalent of an SSH client's "Test" button: the failures worth catching
    here — a wrong account, an unreadable key, a changed host key — are all ones the
    operator can only otherwise discover by starting a chat and watching it fail.
    """
    from . import ssh_connections, ssh_terminal

    try:
        ssh_terminal.ensure_available()
    except ssh_terminal.TerminalDisabled as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=403)

    identifier = str(body.get("id") or "")
    try:
        if identifier:
            settings = ssh_connections.connection_settings(_ssh_app_home(), f"saved:{identifier}")
        else:
            settings = ssh_connections._clean(body or {})
            settings = {
                key: value for key, value in settings.items()
                if key in ssh_connections.EDITABLE_FIELDS and key != "name"
            }
    except ValueError as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=400)

    def probe() -> dict:
        from . import datasources

        instance = datasources.instantiate(datasources.load("ssh"), settings)
        return instance.get_status()

    try:
        status = await asyncio.to_thread(probe)
    except LookupError as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=400)
    return JSONResponse({
        "ok": bool(status.get("success")),
        "status": status.get("status") or "",
        "error": status.get("error") or "",
    })


@app.get("/api/ssh/sessions")
async def api_ssh_sessions(request: Request) -> Response:
    """What SSH terminals are open right now. Never includes credentials."""
    return JSONResponse({
        "enabled": ssh_terminal.terminal_enabled(),
        "maxSessions": ssh_terminal.max_sessions(),
        "sessions": ssh_terminal.sessions.snapshot(),
    })


@app.get("/")
async def api_root() -> Response:
    """API index. Browser UI lives on the original web endpoint, not this process."""
    if _serve_ui():
        return _send_file(_static_files()["/"])
    return JSONResponse(
        {
            "ok": True,
            "service": "subjective_devagents_fastapi",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "web": _web_ui_url(),
        }
    )


# --- static UI -------------------------------------------------------------------------


def _static_files() -> dict[str, Path]:
    """Path -> absolute file, an explicit allowlist rather than a generic "serve anything
    under these directories" rule, so a request can never walk outside the handful of
    files this app actually ships to the browser."""
    root = state.client_root
    web = root / "web"
    src = root / "src"
    return {
        "/": web / "index.html",
        "/index.html": web / "index.html",
        "/webBridge.js": web / "webBridge.js",
        "/firebaseAuth.js": web / "firebaseAuth.js",
        "/bootstrap.js": web / "bootstrap.js",
        "/auth.css": web / "auth.css",
        "/fit-app.js": web / "fit-app.js",
        "/contextMenu.css": web / "contextMenu.css",
        "/layout.css": web / "layout.css",
        "/terminal.html": web / "terminal.html",
        "/terminalWebBridge.js": web / "terminalWebBridge.js",
        "/styles.css": src / "styles.css",
        "/terminal.css": src / "terminal.css",
        "/messageRenderer.js": src / "messageRenderer.js",
        "/renderer.js": src / "renderer.js",
        "/terminal-renderer.js": src / "terminal-renderer.js",
    }


def _stamp_api_base(html: str) -> str:
    """Rewrite the page's API origin at serve time.

    The bundle ships a default in <meta name="devagents-api">, but which backend a page
    should talk to is a property of the deployment, not of the checkout: the same files are
    served both by a split deployment (page and API on different hosts) and by this process
    when it serves the UI itself. Baking one hostname into the file makes the second case
    wrong in the worst way — the page would drive some *other* machine's agent CLIs.

    SUBJECTIVE_DEVAGENTS_PUBLIC_API sets it explicitly. Empty (the default when this process
    serves the UI) means same-origin, so the page talks to whichever backend served it,
    which is this one. The CSP is rewritten to match, since connect-src must name the origin
    or the browser blocks the call before sending it.
    """
    import re

    configured = os.environ.get("SUBJECTIVE_DEVAGENTS_PUBLIC_API")
    if configured is None:
        return html
    base = configured.strip().rstrip("/")
    out = re.sub(
        r'(<meta\s+name="devagents-api"\s+content=")[^"]*(")',
        lambda match: f"{match.group(1)}{base}{match.group(2)}",
        html,
        count=1,
    )
    # Keep connect-src in step with whatever the meta tag now names. Anchor this to the CSP
    # meta tag rather than to the first "connect-src" in the file: the page explains the
    # directive in a comment above it, and matching that instead rewrote the prose and left
    # the real policy untouched — which fails open, since the page then looks configured
    # while the browser still blocks the call.
    def fix_csp_tag(tag_match: "re.Match[str]") -> str:
        def fix_directive(directive_match: "re.Match[str]") -> str:
            directive = re.sub(
                r"\s*https?://[^\s;]+",
                lambda match: match.group(0) if match.group(0).strip() in {
                    "https://identitytoolkit.googleapis.com", "https://securetoken.googleapis.com",
                    "https://www.googleapis.com", "https://subjectivetechnologies-login.firebaseapp.com",
                } else "", directive_match.group(1),
            ).rstrip()
            return f"{directive} {base};" if base else f"{directive};"

        return re.sub(r"(connect-src[^;\"]*);", fix_directive, tag_match.group(0), count=1)

    return re.sub(
        r'<meta\s+http-equiv="Content-Security-Policy"[^>]*>',
        fix_csp_tag,
        out,
        count=1,
        flags=re.IGNORECASE,
    )


def _stamp_html(html: str) -> str:
    """Replace the static placeholder so View Source shows the live release + git SHA."""
    import re

    out = _stamp_api_base(str(html))
    stamp_comment = f"<!-- devagents-deploy {state.deploy_version} served={datetime.now(timezone.utc).isoformat()} -->"
    if "DEVAGENTS_DEPLOY_VERSION" in out:
        out = out.replace("DEVAGENTS_DEPLOY_VERSION", state.deploy_version)
    if "DEVAGENTS_INSTANCE_ID" in out:
        out = out.replace("DEVAGENTS_INSTANCE_ID", state.instance_id)
    if "DEVAGENTS_DEPLOY_VERSION" not in html and "<!doctype html>" in out.lower():
        out = re.sub(r"<!doctype html>", lambda match: f"{match.group(0)}\n{stamp_comment}", out, count=1, flags=re.IGNORECASE)

    def stamp_tag(match: "re.Match[str]") -> str:
        attrs = match.group(1)
        if "data-devagents-version=" in attrs:
            attrs = re.sub(r'\sdata-devagents-version="[^"]*"', f' data-devagents-version="{state.deploy_version}"', attrs)
        else:
            attrs = f'{attrs} data-devagents-version="{state.deploy_version}"'
        if "data-devagents-instance=" in attrs:
            attrs = re.sub(r'\sdata-devagents-instance="[^"]*"', f' data-devagents-instance="{state.instance_id}"', attrs)
        else:
            attrs = f'{attrs} data-devagents-instance="{state.instance_id}"'
        return f"<html{attrs}>"

    return re.sub(r"<html\b([^>]*)>", stamp_tag, out, count=1, flags=re.IGNORECASE)


def _send_file(file_path: Path, is_asset: bool = False) -> Response:
    try:
        data = file_path.read_bytes()
    except OSError:
        return Response(content="Not found", status_code=404, media_type="text/plain")
    extension = file_path.suffix.lower()
    body = data
    if extension == ".html":
        body = _stamp_html(data.decode("utf-8")).encode("utf-8")
    elif extension == ".js" and b"DEVAGENTS_DEPLOY_VERSION" in data:
        # bootstrap.js carries the ?v= tags for the scripts it loads. Those were
        # hand-maintained constants that had not been bumped since 2026-09-11, so every
        # renderer change after that reached the origin and was still served from
        # Cloudflare's edge and from browsers' own caches. Stamping the live deploy
        # version makes each deploy a new URL, which is the only thing those caches key on.
        body = data.decode("utf-8").replace("DEVAGENTS_DEPLOY_VERSION", state.deploy_version).encode("utf-8")
    headers = {
        # no-store everywhere: index.html, renderer.js and layout.css ship as a matched
        # set; a cached renderer.js against a newer index.html renders nothing.
        "Cache-Control": "public, max-age=300" if is_asset else "no-store",
        "X-DevAgents-Version": state.deploy_version,
    }
    if extension == ".html":
        headers["Content-Security-Policy"] = FRAME_ANCESTORS
    media_type = MIME_TYPES.get(extension) or mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return Response(content=body, media_type=media_type, headers=headers)


@app.get("/{path:path}")
async def serve_static(path: str) -> Response:
    if not _serve_ui():
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "Not found", "docs": "/docs", "web": _web_ui_url()},
        )
    pathname = f"/{path}"
    target = _static_files().get(pathname)
    if target:
        return _send_file(target)

    # /assets/** mirrors the client's assets/ directory (vendored libs, icons) — the only
    # tree served generically, fenced to assets_dir by a resolved-path containment check
    # so `/assets/../../etc/passwd`-style traversal cannot escape it.
    if pathname.startswith("/assets/"):
        assets_dir = (state.client_root / "assets").resolve()
        resolved = (assets_dir / pathname[len("/assets/") :]).resolve()
        if resolved == assets_dir or str(resolved).startswith(str(assets_dir) + os.sep):
            return _send_file(resolved, is_asset=True)
    return Response(content="Not found", status_code=404, media_type="text/plain")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, error: Exception) -> JSONResponse:
    default_log("error", "API handler failed", {"route": str(request.url.path), "message": str(error)})
    return JSONResponse(status_code=400, content={"ok": False, "error": str(error)})
