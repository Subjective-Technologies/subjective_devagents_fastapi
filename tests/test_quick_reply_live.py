"""Opt-in live end-to-end quick-reply test, one real turn per installed CLI.

Unlike test_quick_reply.py (which stubs the send), this drives a real agent turn through the
actual CLI — the only thing that proves clicking Yes truly works for a given provider end to
end. It is slow, costs a subscription/API turn, and needs each CLI authenticated, so it does
NOT run in the normal suite.

Enable it against a running service:

    SUBJECTIVE_DEVAGENTS_LIVE_URL=http://127.0.0.1:8787 \
    SUBJECTIVE_DEVAGENTS_LIVE_PROVIDERS=codex,claude,cursor \
    [SUBJECTIVE_DEVAGENTS_API_TOKEN=...] \
    python -m pytest tests/test_quick_reply_live.py -q

Each named provider must report ready first; a provider that is not ready is skipped, not
failed, so an unbilled grok or a logged-out CLI does not break the run.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pytest

BASE_URL = os.environ.get("SUBJECTIVE_DEVAGENTS_LIVE_URL", "").rstrip("/")
PROVIDERS = [p.strip() for p in os.environ.get("SUBJECTIVE_DEVAGENTS_LIVE_PROVIDERS", "codex,claude,cursor").split(",") if p.strip()]
TOKEN = os.environ.get("SUBJECTIVE_DEVAGENTS_API_TOKEN", "")

pytestmark = pytest.mark.skipif(not BASE_URL, reason="set SUBJECTIVE_DEVAGENTS_LIVE_URL to run the live quick-reply test")


def _call(method: str, path: str, body: dict | None = None, timeout: float = 120):
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["X-DevAgents-Token"] = TOKEN
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_quick_reply_runs_a_real_turn(provider):
    status = _call("GET", f"/api/llm/status?provider={provider}", timeout=60)
    state = status.get("status") or (status.get("result") or {}).get("status")
    if state != "ready":
        pytest.skip(f"{provider} is not ready ({state}); cannot run a live turn")

    created = _call("POST", "/api/llm/create-conversation", {"provider": provider, "name": f"live quick-reply {provider}"})
    target = created["conversation"]["target"]

    # Exactly what the Yes button posts.
    accepted = _call("POST", "/api/llm/send-message", {"target": target, "text": "Yes, proceed. Reply with the single word OK."})
    assert accepted["ok"] is True and accepted["pending"] is True

    # The turn runs in the background and lands on the transcript; poll the conversation
    # until an assistant message appears (or time out).
    deadline = time.time() + 120
    reply = None
    while time.time() < deadline:
        convo = _call("GET", f"/api/llm/conversation?target={urllib.parse.quote(target)}", timeout=60)
        messages = (convo.get("conversation") or {}).get("messages") or []
        reply = next((m for m in reversed(messages) if m.get("role") == "assistant" and m.get("text")), None)
        if reply:
            break
        time.sleep(3)

    assert reply, f"{provider} produced no assistant reply within the timeout"

