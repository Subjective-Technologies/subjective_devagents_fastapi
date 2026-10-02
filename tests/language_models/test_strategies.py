"""Exercise provider wire formats through real pipes, without paid model calls."""
from dataclasses import replace
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from subjective_devagents_fastapi.language_models import (
    Api, ClaudeCli, Cli, CodexCli, Identity, IdentityRegistry, Message, ModelRequest,
)
from subjective_devagents_fastapi.language_models.strategies import cli_prompt, isolated_environment
from subjective_devagents_fastapi.language_models.transports import ProcessResult


@pytest.fixture
def request_data(tmp_path):
    home = tmp_path / "account"
    home.mkdir()
    return ModelRequest("turn", Identity("claude-a", "claude", home=str(home),
                        config_dir=str(home / ".claude"), workspace=str(tmp_path)), "hello", timeout=2)


def executable(tmp_path, body):
    script = tmp_path / "fake-cli"
    script.write_text(f"#!{sys.executable}\n" + body)
    script.chmod(0o700)
    return str(script)


def test_claude_stream_and_resume_via_real_pipes(request_data, tmp_path):
    cli = executable(tmp_path, '''import json, os, sys
assert not sys.stdin.isatty()
assert sys.stdin.read() == "hello"
assert "--resume" in sys.argv and "session-a" in sys.argv
assert "OPENAI_API_KEY" not in os.environ
print(json.dumps({"type":"system", "session_id":"session-a"}), flush=True)
print(json.dumps({"type":"stream_event", "event":{"delta":{"type":"text_delta","text":"hi"}}}), flush=True)
print(json.dumps({"type":"result", "subtype":"success", "session_id":"session-a", "result":"hi"}), flush=True)
''')
    request = replace(request_data, identity=replace(request_data.identity, executable=cli), session_id="session-a")
    events = []
    result = Cli(ClaudeCli()).run(request, lambda *event: events.append(event), lambda: False)
    assert (result.status, result.text, result.session_id) == ("completed", "hi", "session-a")
    assert events[0] == ("text_delta", {"text": "hi"})
    assert events[-1] == ("process_exit", {"returncode": 0, "transport_status": "completed"})


@pytest.mark.parametrize("events,code", [
    ([{"type": "result", "is_error": True, "session_id": "s", "result": "secret provider error"}], "provider_error"),
    ([{"type": "result", "session_id": "s", "result": "denied", "permission_denials": [{"tool": "Bash"}]}], "approval_required"),
    ([{"type": "system", "session_id": "s"}], "incomplete_provider_output"),
    ([{"type": "result", "result": "no session"}], "incomplete_provider_output"),
    (["not-json"], "invalid_provider_output"),
])
def test_claude_never_treats_incomplete_or_failed_output_as_success(request_data, tmp_path, events, code):
    lines = [json.dumps(e) if isinstance(e, dict) else e for e in events]
    cli = executable(tmp_path, "import sys\nsys.stdin.read()\nprint(" + repr("\n".join(lines)) + ")\n")
    result = Cli(ClaudeCli()).run(replace(request_data, identity=replace(request_data.identity, executable=cli)),
                                  lambda *_: None, lambda: False)
    assert result.status == "failed"
    assert result.error_code == code
    assert "secret" not in repr(result)


def test_codex_requires_terminal_event_and_uses_native_session(request_data, tmp_path):
    cli = executable(tmp_path, '''import json,sys
assert sys.stdin.read() == "hello"
assert "resume" in sys.argv and "codex-session" in sys.argv
for event in [
 {"type":"thread.started", "thread_id":"codex-session"},
 {"type":"item.completed", "item":{"type":"agent_message", "text":"reply"}},
 {"type":"turn.completed", "usage":{}}]:
 print(json.dumps(event), flush=True)
''')
    identity = replace(request_data.identity, provider="codex", executable=cli)
    result = Cli(CodexCli()).run(replace(request_data, identity=identity, session_id="codex-session"),
                                 lambda *_: None, lambda: False)
    assert result.status == "completed" and result.session_id == "codex-session"


def test_handoff_prompt_keeps_roles_and_never_silently_truncates(request_data):
    history = (Message("user", "old task"), Message("assistant", "old reply"))
    prompt = cli_prompt(replace(request_data, history=history))
    assert '"role": "user", "content": "old task"' in prompt
    assert "Current user request:\nhello" in prompt
    assert cli_prompt(replace(request_data, history=history, session_id="same-account")) == "hello"
    with pytest.raises(Exception, match="transfer limit"):
        cli_prompt(replace(request_data, history=(Message("user", "x" * 600_000),)))


def test_credentials_are_not_inherited_or_global_environment_mutated(request_data, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "different-account")
    monkeypatch.setenv("CODEX_HOME", "/another/account")
    env = isolated_environment(request_data)
    assert "ANTHROPIC_API_KEY" not in env and "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["HOME"] == request_data.identity.home
    assert env["CLAUDE_CONFIG_DIR"] == request_data.identity.config_dir
    assert os.environ["CODEX_HOME"] == "/another/account"


def test_subagent_session_never_overwrites_parent():
    state = {"session_id": "parent"}
    ClaudeCli().consume({"type": "assistant", "session_id": "child", "parent_tool_use_id": "tool"}, state, lambda *_: None)
    assert state["session_id"] == "parent"


def test_provider_cannot_silently_fork_a_resumed_session(request_data, tmp_path):
    cli = executable(tmp_path, 'import sys,json\nsys.stdin.read()\nprint(json.dumps({"type":"result",'
                     '"session_id":"new-session","result":"wrong context"}))\n')
    request = replace(request_data, identity=replace(request_data.identity, executable=cli), session_id="old-session")
    result = Cli(ClaudeCli()).run(request, lambda *_: None, lambda: False)
    assert result.status == "failed" and result.error_code == "native_session_changed"


def test_local_os_user_switch_is_noninteractive_and_cwd_after_sudo(request_data):
    class Capture:
        def run(self, command, **kwargs):
            self.command = command
            return ProcessResult(-1, "", "", "unavailable")
    transport = Capture()
    request = replace(request_data, identity=replace(request_data.identity, os_user="different_account"))
    Cli(ClaudeCli(), transport).run(request, lambda *_: None, lambda: False)
    command = transport.command
    assert command.argv[:6] == ("sudo", "-n", "-u", "different_account", "--", "sh")
    assert command.cwd is None
    assert request.identity.workspace in command.argv
    assert "hello" not in command.argv and command.stdin == "hello"


def test_registry_rejects_shared_credentials_and_invalid_urls(request_data):
    with pytest.raises(ValueError, match="separate homes"):
        IdentityRegistry([request_data.identity, replace(request_data.identity, id="claude-b")])
    with pytest.raises(ValueError, match="HTTPS"):
        IdentityRegistry([Identity("api", "openai-compatible", strategy="api", model="test",
                                  api_base_url="http://external.example/v1", api_key_env="TEST_KEY")])


@pytest.fixture
def http_provider():
    requests = []
    settings = {"delay": 0, "status": 200, "finish_reason": "stop"}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers.get("Authorization"),
                             json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            time.sleep(settings["delay"])
            payload = json.dumps({"choices": [{"message": {"content": "http reply"},
                                                "finish_reason": settings["finish_reason"]}]}).encode()
            self.send_response(settings["status"])
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests, settings
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_api_real_http_preserves_history_and_posts_once(http_provider, request_data, monkeypatch):
    url, requests, _ = http_provider
    monkeypatch.setenv("TEST_LM_KEY", "fixture-key")
    identity = Identity("api", "compatible", strategy="api", model="test", api_base_url=url, api_key_env="TEST_LM_KEY")
    request = replace(request_data, identity=identity, history=(Message("user", "before"), Message("assistant", "answer")))
    result = Api().run(request, lambda *_: None, lambda: False)
    assert result.status == "completed" and result.text == "http reply"
    assert len(requests) == 1
    assert requests[0][:2] == ("/v1/chat/completions", "Bearer fixture-key")
    assert [m["content"] for m in requests[0][2]["messages"]] == ["before", "answer", "hello"]


@pytest.mark.parametrize("status,expected", [(401, "authentication_required"), (429, "rate_limited"), (500, "api_http_error")])
def test_api_does_not_retry_http_failures(http_provider, request_data, monkeypatch, status, expected):
    url, requests, settings = http_provider
    settings["status"] = status
    monkeypatch.setenv("TEST_LM_KEY", "fixture-key")
    identity = Identity("api", "compatible", strategy="api", model="test", api_base_url=url, api_key_env="TEST_LM_KEY")
    result = Api().run(replace(request_data, identity=identity), lambda *_: None, lambda: False)
    assert result.error_code == expected and len(requests) == 1


def test_api_cancel_reports_unknown_remote_outcome(http_provider, request_data, monkeypatch):
    url, requests, settings = http_provider
    settings["delay"] = 0.5
    monkeypatch.setenv("TEST_LM_KEY", "fixture-key")
    identity = Identity("api", "compatible", strategy="api", model="test", api_base_url=url, api_key_env="TEST_LM_KEY")
    result = Api().run(replace(request_data, identity=identity), lambda *_: None, lambda: bool(requests))
    assert result.status == "interrupted" and result.error_code == "api_outcome_unknown"
    assert len(requests) == 1


@pytest.mark.parametrize('provider,relative', [('cursor', '.config/cursor'), ('grok', '.grok')])
def test_provider_config_directory_is_an_enforced_binding(request_data, provider, relative):
    identity = replace(request_data.identity, provider=provider)
    with pytest.raises(ValueError, match='canonical config directory'):
        IdentityRegistry([identity])
    identity = replace(identity, config_dir=identity.home + '/' + relative)
    assert IdentityRegistry([identity]).get(identity.id) == identity


def test_absolute_executable_directory_provides_node_without_ambient_path(request_data, monkeypatch):
    monkeypatch.setenv('PATH', '/untrusted')
    identity = replace(request_data.identity, executable='/opt/node/bin/codex')
    env = isolated_environment(replace(request_data, identity=identity))
    assert env['PATH'].split(':')[0] == '/opt/node/bin'
    assert '/untrusted' not in env['PATH']
