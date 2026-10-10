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
from subjective_devagents_fastapi.language_models.strategies import CursorCli, GrokCli, cli_prompt, isolated_environment
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


@pytest.mark.parametrize("message,code", [
    ("You’ve hit your usage limit. Try again later.", "usage_limit"),
    ("authentication required", "authentication_required"),
])
def test_codex_provider_failures_keep_actionable_stable_codes(request_data, tmp_path, message, code):
    cli = executable(tmp_path, f'''import json,sys
sys.stdin.read()
print(json.dumps({{"type":"error", "message": {message!r}}}))
print(json.dumps({{"type":"turn.failed", "error": {{"message": {message!r}}}}}))
''')
    identity = replace(request_data.identity, provider="codex", executable=cli)
    result = Cli(CodexCli()).run(replace(request_data, identity=identity), lambda *_: None, lambda: False)
    assert result.status == "failed" and result.error_code == code


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


def test_api_cancel_is_a_clean_cancellation(http_provider, request_data, monkeypatch):
    url, requests, settings = http_provider
    settings["delay"] = 0.5
    monkeypatch.setenv("TEST_LM_KEY", "fixture-key")
    identity = Identity("api", "compatible", strategy="api", model="test", api_base_url=url, api_key_env="TEST_LM_KEY")
    result = Api().run(replace(request_data, identity=identity), lambda *_: None, lambda: bool(requests))
    assert result.status == "cancelled" and result.error_code == "cancelled_during_request"
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


# --- images -------------------------------------------------------------------------------

PNG = bytes.fromhex("89504e470d0a1a0a") + b"fixture-image-bytes"


def with_image(request_data, **identity):
    from subjective_devagents_fastapi.language_models.contracts import Attachment
    return replace(request_data, identity=replace(request_data.identity, **identity),
                   attachments=(Attachment("shot.png", "image/png", PNG),))


def test_codex_receives_images_as_files_that_exist_during_the_turn_and_are_removed_after(request_data, tmp_path):
    seen = tmp_path / "seen.json"
    cli = executable(tmp_path, f'''import json, os, sys
assert sys.stdin.read() == "hello"
paths = [sys.argv[i + 1] for i, arg in enumerate(sys.argv) if arg == "--image"]
assert sys.argv[-1] == "-" and len(paths) == 1
open({str(seen)!r}, "w").write(json.dumps({{"path": paths[0], "data": open(paths[0], "rb").read().hex(),
                                           "mode": oct(os.stat(paths[0]).st_mode & 0o777)}}))
for event in [{{"type":"thread.started","thread_id":"codex-session"}},
              {{"type":"item.completed","item":{{"type":"agent_message","text":"saw it"}}}},
              {{"type":"turn.completed"}}]:
    print(json.dumps(event), flush=True)
''')
    request = with_image(request_data, provider="codex", executable=cli)
    result = Cli(CodexCli()).run(request, lambda *_: None, lambda: False)
    assert result.status == "completed" and result.text == "saw it"
    recorded = json.loads(seen.read_text())
    assert bytes.fromhex(recorded["data"]) == PNG and recorded["mode"] == "0o600"
    assert not os.path.exists(recorded["path"]) and not os.path.exists(os.path.dirname(recorded["path"]))


def test_codex_resume_attaches_images_to_the_resumed_prompt():
    from subjective_devagents_fastapi.language_models.contracts import Attachment
    request = ModelRequest("turn", Identity("codex-a", "codex"), "hi", session_id="sess",
                           attachments=(Attachment("a.png", "image/png", PNG),))
    argv = CodexCli().argv(request, ("/tmp/a.png",))
    assert argv[argv.index("resume"):] == ("resume", "sess", "--image", "/tmp/a.png", "-")


def test_claude_receives_images_inline_as_stream_json_content_blocks(request_data, tmp_path):
    cli = executable(tmp_path, f'''import base64, json, sys
assert sys.argv[sys.argv.index("--input-format") + 1] == "stream-json"
line = json.loads(sys.stdin.read())
assert line["type"] == "user" and line["message"]["role"] == "user"
text, image = line["message"]["content"]
assert text == {{"type": "text", "text": "hello"}}
assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
assert base64.b64decode(image["source"]["data"]) == {PNG!r}
print(json.dumps({{"type":"system", "session_id":"claude-session"}}), flush=True)
print(json.dumps({{"type":"result", "subtype":"success", "session_id":"claude-session", "result":"saw it"}}), flush=True)
''')
    result = Cli(ClaudeCli()).run(with_image(request_data, executable=cli), lambda *_: None, lambda: False)
    assert result.status == "completed" and result.text == "saw it"


def test_text_only_claude_turns_keep_plain_stdin():
    request = ModelRequest("turn", Identity("claude-a", "claude"), "hello")
    assert "--input-format" not in ClaudeCli().argv(request)
    assert ClaudeCli().stdin(request, "hello") == "hello"


@pytest.mark.parametrize("strategy,identity", [
    (lambda: Cli(CodexCli()), {"provider": "codex", "ssh_host": "remote-box"}),
    (lambda: Cli(GrokCli()), {"provider": "grok"}),
    (lambda: Cli(CursorCli()), {"provider": "cursor"}),
    (lambda: Api(), {"provider": "codex", "strategy": "api", "api_base_url": "http://127.0.0.1:9", "api_key_env": "X"}),
])
def test_strategies_that_cannot_take_images_refuse_without_running(request_data, strategy, identity):
    class NeverRun:
        def run(self, *args, **kwargs):
            raise AssertionError("must not start a process")
    chosen = strategy()
    if isinstance(chosen, Cli):
        chosen.transport = NeverRun()
    result = chosen.run(with_image(request_data, **identity), lambda *_: None, lambda: False)
    assert result.status == "failed" and result.error_code == "unsupported_attachments"


# --- live step list -------------------------------------------------------------------------


def steps_from(driver, events):
    emitted = []
    state = {"session_id": None, "text": "", "completed": False, "error": None}
    for event in events:
        driver.consume(event, state, lambda kind, data: emitted.append((kind, data.get("text"))))
    return emitted


def test_codex_reports_commands_edits_reasoning_and_plan_as_steps():
    item = lambda kind, **fields: {"type": kind, "item": fields}
    emitted = steps_from(CodexCli(), [
        item("item.completed", type="reasoning", text="Need to inspect\nthe tests first"),
        item("item.started", type="command_execution", command="pytest -q", status="in_progress"),
        item("item.completed", type="command_execution", command="pytest -q", exit_code=1, status="failed"),
        item("item.completed", type="command_execution", command="ls", exit_code=0, status="completed"),
        item("item.completed", type="file_change", changes=[{"path": "app.py", "kind": "update"},
                                                           {"path": "new.py", "kind": "add"}]),
        item("item.updated", type="todo_list", items=[{"text": "fix", "completed": True},
                                                      {"text": "test", "completed": False}]),
        item("item.started", type="web_search", query="codex json events"),
        item("item.completed", type="agent_message", text="Done."),
    ])
    assert emitted == [
        ("activity", "Thinking: Need to inspect the tests first"),
        ("activity", "Running: pytest -q"),
        ("activity", "Command failed (exit 1): pytest -q"),
        ("activity", "Edited app.py; Created new.py"),
        ("activity", "Plan 1/2 — next: test"),
        ("activity", "Searching the web: codex json events"),
        ("message", "Done."),
    ]


def test_claude_reports_tool_calls_and_thoughts_but_not_subagent_noise():
    emitted = steps_from(ClaudeCli(), [
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "Check the config"},
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "git   status"}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/repo/app.py"}},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "/repo/app.py"}},
            {"type": "tool_use", "name": "TodoWrite", "input": {"todos": [
                {"content": "a", "status": "completed"},
                {"content": "b", "activeForm": "Running tests", "status": "in_progress"}]}},
            {"type": "tool_use", "name": "mcp__x__y", "input": {}},
        ]}},
        {"type": "assistant", "parent_tool_use_id": "sub", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "hidden"}}]}},
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "Le"}}},
    ])
    assert emitted == [
        ("activity", "Thinking: Check the config"),
        ("message", "Let me look."),
        ("activity", "Running: git status"),
        ("activity", "Reading /repo/app.py"),
        ("activity", "Editing /repo/app.py"),
        ("activity", "Plan 1/2 — Running tests"),
        ("activity", "Using mcp__x__y"),
        ("text_delta", "Le"),
    ]


def test_long_commands_are_cut_to_one_line():
    from subjective_devagents_fastapi.language_models.strategies import short
    assert short("a\n" + "b" * 300, 20) == "a " + "b" * 17 + "…"
