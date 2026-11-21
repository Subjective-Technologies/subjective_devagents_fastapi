"""DeepSeek sends were accepted and then failed authentication (2026-11-21 18:33:59 -0300).

Production chat llm:deepseek:local:97ce734f: `POST /api/llm/create-conversation` 200, then
`POST /api/llm/send-message` 200 at 21:33:41 UTC, `LLM send start` at 21:33:46 with
`newSession: true` — and 12.8 s later `status: failed`, `error: authentication_required`,
`authFailure: true`. DeepSeek has no CLI, so the identity is an `api_agent` built by
`legacy.proxy_identity` against the CLIProxy inference proxy.

The turn started, so the env key existed (a missing one returns from `ApiAgent._run`
instantly, not after 13 s): the proxy or its upstream rejected the call. Two things were
taken on trust where the identity is built — the presence of
`SUBJECTIVE_DEVAGENTS_LLM_PROXY_API_KEY`, and the idea that an operator-pinned model id
proves CLIProxy holds an upstream DeepSeek credential. Neither is checked, so the send is
admitted and only the turn finds out.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from subjective_devagents_fastapi.language_models import catalog, legacy

MODEL = "deepseek-chat"


class StubProxy:
    """A CLIProxyAPI stand-in: each POST pops the next scripted (status, payload)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[tuple[str, str | None, dict]] = []

    def __enter__(self):
        responses, requests = self.responses, self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                status, payload = responses.pop(0) if responses else (500, {"error": "no script left"})
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except BrokenPipeError:
                    pass

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def answer(content):
    return (200, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})


@pytest.fixture
def deepseek(client, monkeypatch):
    """A DeepSeek conversation on a configured proxy, with the catalog under test control."""
    from subjective_devagents_fastapi.app import core

    monkeypatch.setenv(legacy.PROXY_URL_ENV, "http://127.0.0.1:1/v1")
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV["deepseek"], MODEL)
    monkeypatch.setenv(legacy.PROXY_KEY_ENV, "fixture-proxy-key")
    monkeypatch.setattr(catalog, "proxy_models", lambda *a, **k: {"deepseek": [MODEL]})
    created = client.post("/api/llm/create-conversation", json={"provider": "deepseek", "name": "ds"}).json()
    assert created["ok"], created
    return core(), created["conversation"]["target"]


def test_a_missing_proxy_key_refuses_the_send_before_the_turn_starts(deepseek, monkeypatch, client):
    instance, target = deepseek
    monkeypatch.delenv(legacy.PROXY_KEY_ENV, raising=False)
    logged = []
    monkeypatch.setattr(instance.llm_service, "log",
                        lambda level, message, details=None: logged.append((level, message)))

    response = client.post("/api/llm/send-message", json={"target": target, "text": "one line please"})

    assert response.status_code == 401, response.text
    body = response.json()
    assert body == {"ok": False, "status": "authentication_required", "error": body["error"]}
    assert legacy.PROXY_KEY_ENV in body["error"]
    assert "LLM send start" not in [message for _level, message in logged], logged
    # The refusal is an auth failure, so the UI still gets its session chooser.
    assert instance.llm_service.is_auth_failure(body)


def test_a_proxy_without_an_upstream_deepseek_credential_refuses_the_send(deepseek, monkeypatch, client):
    """A pinned model id is routing metadata, not evidence of an upstream credential."""
    instance, target = deepseek
    monkeypatch.setattr(catalog, "proxy_models", lambda *a, **k: {"codex": ["gpt-x"], "deepseek": []})

    response = client.post("/api/llm/send-message", json={"target": target, "text": "one line please"})

    assert response.status_code == 401
    assert response.json()["status"] == "authentication_required"
    assert "deepseek" in response.json()["error"]
    conversation = client.get("/api/llm/conversation", params={"target": target}).json()["conversation"]
    assert conversation["messages"] == [], "a refused send must not record the prompt as sent"


def test_an_unreachable_catalog_is_not_treated_as_a_missing_credential(deepseek, monkeypatch, client):
    """Absence has to be proven; a catalog outage is reported by the turn, as before."""
    from subjective_devagents_fastapi.language_models.contracts import ModelError

    instance, target = deepseek

    def unavailable(*_args, **_kwargs):
        raise ModelError("model_catalog_unavailable", "Could not read the inference proxy model catalog.")

    monkeypatch.setattr(catalog, "proxy_models", unavailable)
    assert instance.llm_service.preflight_send(target) is None


def test_a_proxy_401_still_surfaces_authentication_required_with_authfailure(deepseek, monkeypatch, client):
    """The admitted-then-rejected case keeps its code and its llm-choose-session broadcast."""
    instance, target = deepseek
    events = []
    monkeypatch.setattr(instance, "_emit", events.append)
    with StubProxy([(401, {"error": {"message": "invalid api key"}})]) as proxy:
        monkeypatch.setenv(legacy.PROXY_URL_ENV, proxy.url)
        response = instance.answer_llm_message(target, "one line please", "tester", False)

    assert response["ok"] is False
    assert response["status"] == "failed" and response["error"] == "authentication_required"
    assert instance.llm_service.is_auth_failure(response) is True
    assert "llm-choose-session" in [event.get("type") for event in events]


def test_a_configured_proxy_completes_a_one_line_send(deepseek, monkeypatch, client):
    instance, target = deepseek
    with StubProxy([answer("One line, as requested.")]) as proxy:
        monkeypatch.setenv(legacy.PROXY_URL_ENV, proxy.url)
        response = instance.llm_service.send_message(target, "one line please")

    assert response["ok"] is True, response
    assert response["assistantMessage"]["text"].startswith("One line, as requested.")
    assert proxy.requests[0][0] == "/v1/chat/completions"
    assert proxy.requests[0][1] == "Bearer fixture-proxy-key", "the proxy key rides in Authorization"
    assert json.loads(json.dumps(proxy.requests[0][2]))["model"] == MODEL
