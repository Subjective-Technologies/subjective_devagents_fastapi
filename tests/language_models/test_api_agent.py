"""ApiAgent: the tool-executing replacement for the CLI drivers, over real HTTP and pipes."""
from dataclasses import replace
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from subjective_devagents_fastapi.language_models import ApiAgent, Identity, IdentityRegistry, ModelRequest
from subjective_devagents_fastapi.language_models.legacy import proxy_identity


@pytest.fixture
def identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Identity("agent-a", "claude", strategy="api_agent", model="test-model", workspace=str(workspace),
                    home=str(tmp_path / "home"), api_base_url="http://127.0.0.1:9", api_key_env="TEST_AGENT_KEY",
                    sandbox="workspace-write")


@pytest.fixture
def request_data(identity):
    return ModelRequest("turn", identity, "please help", timeout=5)


class ScriptedProxy:
    """A fake CLIProxyAPI: each call to /chat/completions pops the next scripted response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __enter__(self):
        responses, requests = self.responses, self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                status, delay, payload = responses.pop(0) if responses else (500, 0, {"error": "no script left"})
                time.sleep(delay)
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


def completion(content=None, tool_calls=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return (200, 0, {"choices": [{"message": message, "finish_reason": finish_reason}]})


def tool_call(call_id, name, **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_tool_loop_runs_bash_then_answers(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "Bash", command="echo hi")]),
        completion(content="The command printed hi."),
    ]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        events = []
        result = ApiAgent().run(request, lambda *event: events.append(event), lambda: False)
    assert result.status == "completed"
    assert result.text == "The command printed hi."
    assert ("activity", {"text": "Running: echo hi"}) in events
    # Second request carries the tool result back, per the documented wire format.
    second_messages = proxy.requests[1][2]["messages"]
    assert second_messages[-1]["role"] == "tool" and "hi" in second_messages[-1]["content"]
    assert proxy.requests[0][1] == "Bearer fixture-key"


def test_read_write_and_edit_round_trip_through_the_real_filesystem(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    target = request_data.identity.workspace + "/note.txt"
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "Write", file_path=target, content="hello world")]),
        completion(tool_calls=[tool_call("2", "Read", file_path=target)]),
        completion(tool_calls=[tool_call("3", "Edit", file_path=target, old_string="world", new_string="there")]),
        completion(content="Done."),
    ]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "completed" and result.text == "Done."
    assert open(target).read() == "hello there"


def test_write_and_edit_emit_file_diff_events(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    target = request_data.identity.workspace + "/note.txt"
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "Write", file_path=target, content="hello world\n")]),
        completion(tool_calls=[tool_call("2", "Edit", file_path=target, old_string="world", new_string="there")]),
        completion(content="Done."),
    ]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        events = []
        result = ApiAgent().run(request, lambda *event_args: events.append(event_args), lambda: False)
    assert result.status == "completed"
    diffs = [data for kind, data in events if kind == "file_diff"]
    assert len(diffs) == 2
    assert diffs[0]["path"] == target and diffs[0]["status"] == "created"
    assert "+hello world" in diffs[0]["diff"] and "-hello world" not in diffs[0]["diff"]
    assert diffs[1]["status"] == "modified"
    assert "-hello world" in diffs[1]["diff"] and "+hello there" in diffs[1]["diff"]


def test_write_without_content_change_emits_no_diff(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    target = request_data.identity.workspace + "/same.txt"
    open(target, "w").write("same content")
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "Write", file_path=target, content="same content")]),
        completion(content="Done."),
    ]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        events = []
        result = ApiAgent().run(request, lambda *event_args: events.append(event_args), lambda: False)
    assert result.status == "completed"
    assert not [data for kind, data in events if kind == "file_diff"]


def test_read_only_sandbox_refuses_mutating_tools_without_calling_them(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "Bash", command="rm -rf /")]),
        completion(content="Understood, I cannot do that here."),
    ]) as proxy:
        identity = replace(request_data.identity, api_base_url=proxy.url, sandbox="read-only")
        result = ApiAgent().run(replace(request_data, identity=identity), lambda *_: None, lambda: False)
    assert result.status == "completed"
    tool_message = proxy.requests[1][2]["messages"][-1]
    assert tool_message["role"] == "tool" and "not permitted" in tool_message["content"]
    first_tools = {t["function"]["name"] for t in proxy.requests[0][2]["tools"]}
    assert first_tools == {"Read", "Grep", "Glob", "TodoWrite"}


def test_allowed_tools_further_restricts_the_declared_set(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([completion(content="ok")]) as proxy:
        identity = replace(request_data.identity, api_base_url=proxy.url, allowed_tools=("Read", "Grep"))
        ApiAgent().run(replace(request_data, identity=identity), lambda *_: None, lambda: False)
    assert {t["function"]["name"] for t in proxy.requests[0][2]["tools"]} == {"Read", "Grep"}


def test_unknown_tool_name_is_reported_back_to_the_model_not_fatal(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "DeleteEverything")]),
        completion(content="ok"),
    ]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "completed"
    assert "not permitted" in proxy.requests[1][2]["messages"][-1]["content"]


@pytest.mark.parametrize("status,expected", [(401, "authentication_required"), (429, "rate_limited"), (500, "api_http_error")])
def test_http_failures_map_to_stable_codes_without_retry(request_data, monkeypatch, status, expected):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([(status, 0, {"error": "boom"})]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "failed" and result.error_code == expected
    assert len(proxy.requests) == 1


def test_missing_credential_fails_before_any_request(request_data, monkeypatch):
    monkeypatch.delenv("TEST_AGENT_KEY", raising=False)
    result = ApiAgent().run(request_data, lambda *_: None, lambda: False)
    assert result.status == "failed" and result.error_code == "authentication_required"


def test_attachments_are_sent_as_inline_image_content_parts(request_data, monkeypatch):
    import base64
    from subjective_devagents_fastapi.language_models.contracts import Attachment
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    png = b"\x89PNG-fixture"
    with ScriptedProxy([completion(content="I see the image.")]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url),
                          attachments=(Attachment("a.png", "image/png", png),))
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "completed" and result.text == "I see the image."
    user_message = proxy.requests[0][2]["messages"][-1]
    assert user_message["content"][0] == {"type": "text", "text": "please help"}
    image_part = user_message["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(png).decode()}"


def test_registry_reports_attachment_support_for_api_agent(identity):
    from subjective_devagents_fastapi.language_models.registry import image_support
    assert image_support(identity) is True


def test_context_size_check_ignores_image_bytes(request_data, monkeypatch):
    from subjective_devagents_fastapi.language_models.contracts import Attachment
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    big_image = b"\x89PNG" + b"x" * 400_000  # well over MAX_CONTEXT_BYTES once base64-inflated
    with ScriptedProxy([completion(content="ok")]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url),
                          attachments=(Attachment("big.png", "image/png", big_image),))
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "completed"


def test_cancellation_mid_tool_call_stops_the_loop(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([
        completion(tool_calls=[tool_call("1", "Bash", command="echo hi")]),
        completion(content="should not get here"),
    ]) as proxy:
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url))
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            return calls["n"] > 2  # let the first round complete, then stop before executing the tool

        result = ApiAgent().run(request, lambda *_: None, cancelled)
    assert result.status == "interrupted" and result.error_code == "cancelled_mid_tool"
    assert len(proxy.requests) == 1


def test_tool_loop_limit_is_enforced(request_data, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    from subjective_devagents_fastapi.language_models.strategies import MAX_TOOL_ROUNDS
    with ScriptedProxy([completion(tool_calls=[tool_call(str(i), "Bash", command="true")])
                        for i in range(MAX_TOOL_ROUNDS + 1)]) as proxy:
        # A local fake server is near-instant per round, but MAX_TOOL_ROUNDS rounds of
        # real async/HTTP overhead still need more than request_data's 5s fixture timeout.
        request = replace(request_data, identity=replace(request_data.identity, api_base_url=proxy.url), timeout=60)
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "failed" and result.error_code == "tool_loop_limit_exceeded"


def test_history_travels_on_every_request_with_no_native_session(request_data, monkeypatch):
    from subjective_devagents_fastapi.language_models.contracts import Message
    monkeypatch.setenv("TEST_AGENT_KEY", "fixture-key")
    with ScriptedProxy([completion(content="ok")]) as proxy:
        identity = replace(request_data.identity, api_base_url=proxy.url)
        request = replace(request_data, identity=identity, history=(Message("user", "before"), Message("assistant", "answer")))
        result = ApiAgent().run(request, lambda *_: None, lambda: False)
    assert result.status == "completed" and result.session_id is None
    roles = [m["role"] for m in proxy.requests[0][2]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


# --- registry -------------------------------------------------------------------------------


def test_registry_accepts_api_agent_with_ssh_and_os_user(identity):
    remote = replace(identity, ssh_host="devagents-worker", os_user="worker")
    assert IdentityRegistry([remote]).get(remote.id) == remote


def test_registry_rejects_api_agent_for_cursor(identity):
    with pytest.raises(ValueError, match="Unsupported api_agent provider"):
        IdentityRegistry([replace(identity, provider="cursor")])


def test_registry_rejects_relative_api_agent_workspace(identity):
    with pytest.raises(ValueError, match="absolute path"):
        IdentityRegistry([replace(identity, workspace="relative/path")])


def test_registry_accepts_plain_http_for_api_agent_unlike_api(identity):
    # The proxy this strategy targets runs TLS-off by design; "api" keeps its stricter rule.
    assert IdentityRegistry([identity]).get(identity.id).api_base_url == "http://127.0.0.1:9"


# --- legacy wiring ----------------------------------------------------------------------------


def test_legacy_proxy_identity_is_none_without_configuration(monkeypatch):
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL", raising=False)
    monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_LLM_MODEL_CLAUDE", raising=False)
    assert proxy_identity("claude", "alice", "/tmp/ws", "/home/alice") is None


def test_legacy_proxy_identity_is_none_for_cursor_even_when_configured(monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL", "http://127.0.0.1:8317/v1")
    assert proxy_identity("cursor", "alice", "/tmp/ws", "/home/alice") is None


def test_legacy_proxy_identity_is_built_when_configured(monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL", "http://127.0.0.1:8317/v1")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_LLM_MODEL_CLAUDE", "claude-sonnet-4-5")
    found = proxy_identity("claude", "alice", "/tmp/ws", "/home/alice")
    assert found is not None and found.strategy == "api_agent" and found.model == "claude-sonnet-4-5"
    assert found.api_base_url == "http://127.0.0.1:8317/v1" and found.api_key_env == "SUBJECTIVE_DEVAGENTS_LLM_PROXY_API_KEY"
