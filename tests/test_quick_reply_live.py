"""Opt-in real HTTP -> CLI -> WebSocket -> persisted transcript deployment gate.

Set SUBJECTIVE_DEVAGENTS_LIVE_URL and SUBJECTIVE_DEVAGENTS_LIVE_PROVIDERS to
explicitly select the authenticated providers required on this deployment. Selected
providers fail if unavailable; only an unset URL skips the test. Claude also resumes
with an image, exercising the variadic --add-dir argument that previously ate prompts.
Each run creates labelled smoke conversations and uses real provider turns.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
import uuid

import pytest
import websockets

BASE_URL = os.environ.get("SUBJECTIVE_DEVAGENTS_LIVE_URL", "").rstrip("/")
PROVIDERS = [p.strip() for p in os.environ.get(
    "SUBJECTIVE_DEVAGENTS_LIVE_PROVIDERS", "codex,claude,cursor,grok"
).split(",") if p.strip()]
TOKEN = os.environ.get("SUBJECTIVE_DEVAGENTS_API_TOKEN", "")
if BASE_URL and not PROVIDERS:
    raise ValueError("A live deployment gate must select at least one required provider")
TURN_TIMEOUT = float(os.environ.get("SUBJECTIVE_DEVAGENTS_LIVE_TIMEOUT", "120"))
# A valid 1x1 PNG; enough to exercise real image materialization and --add-dir.
IMAGE = {
    "filename": "smoke.png", "mime_type": "image/png",
    "content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=",
}
pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="set SUBJECTIVE_DEVAGENTS_LIVE_URL to run real CLI turns"
)


def _call(method: str, path: str, body: dict | None = None, timeout: float = 30):
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["X-DevAgents-Token"] = TOKEN
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _socket_url():
    parts = urllib.parse.urlsplit(BASE_URL)
    return urllib.parse.urlunsplit((
        "wss" if parts.scheme == "https" else "ws", parts.netloc,
        parts.path + "/ws", urllib.parse.urlencode({"token": TOKEN}) if TOKEN else "", "",
    ))


async def _turn(socket, target, text, expected, files=None):
    accepted = await asyncio.to_thread(_call, "POST", "/api/llm/send-message", {
        "target": target, "text": text, "files": files or [],
    })
    assert accepted.get("ok") and accepted.get("pending"), accepted
    reply = None
    busy = False
    while True:
        event = json.loads(await socket.recv())
        if event.get("target") != target:
            continue
        if event["type"] == "llm-busy":
            busy = busy or event.get("busy") is True
            if event.get("busy") is False:
                break
        # Cursor also emits a successful-send notice about refreshing its IDE tab.
        if event["type"] == "system" and re.match(r"^.+ error \([^)]+\):", event.get("text", "")):
            pytest.fail(f"Agent failed: {event.get('text')}")
        if event["type"] == "llm-message" and event.get("message", {}).get("role") == "assistant":
            reply = event["message"]
    assert busy, "WebSocket never delivered busy=true"
    assert reply and expected in reply.get("text", ""), f"Missing expected assistant reply: {reply}"
    saved = await asyncio.to_thread(
        _call, "GET", "/api/llm/conversation?" + urllib.parse.urlencode({"target": target})
    )
    conversation = saved["conversation"]
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant" and expected in messages[-1]["text"]
    assert any(m.get("role") == "user" and m.get("text") == text for m in messages)
    assert conversation.get("sessionId"), "Native CLI session was not persisted"
    return conversation


async def _exercise(provider):
    status = await asyncio.to_thread(_call, "GET", f"/api/llm/status?provider={provider}")
    state = status.get("status") or (status.get("result") or {}).get("status")
    assert state == "ready", f"Required provider {provider} is not ready: {status}"
    created = await asyncio.to_thread(_call, "POST", "/api/llm/create-conversation", {
        "provider": provider, "name": f"[smoke] real send {provider}",
        "workspace": os.environ.get("SUBJECTIVE_DEVAGENTS_LIVE_WORKSPACE", "/tmp"),
    })
    assert created.get("ok"), created
    target = created["conversation"]["target"]
    sentinel = "SMOKE_" + uuid.uuid4().hex[:12]
    async with websockets.connect(_socket_url(), open_timeout=15, max_size=8 * 1024 * 1024) as socket:
        # The server registers the subscriber before sending its initial config.
        initial = json.loads(await asyncio.wait_for(socket.recv(), timeout=15))
        assert initial["type"] == "config"
        first = await asyncio.wait_for(_turn(
            socket, target,
            f"Reply with exactly {sentinel}. Remember this marker. Do not use tools or edit files.",
            sentinel,
        ), timeout=TURN_TIMEOUT)
        if provider == "claude":
            # Do not repeat the marker: the second turn must actually resume context.
            second = await asyncio.wait_for(_turn(
                socket, target,
                "Ignore the attached image. Reply with exactly the marker from my previous message. "
                "Do not use tools or edit files.",
                sentinel, [IMAGE],
            ), timeout=TURN_TIMEOUT)
            assert second["sessionId"] == first["sessionId"], "Resume changed the native session"
            assert any(m.get("attachments") for m in second["messages"])


@pytest.mark.parametrize("provider", PROVIDERS)
def test_quick_reply_runs_a_real_turn(provider):
    asyncio.run(_exercise(provider))
