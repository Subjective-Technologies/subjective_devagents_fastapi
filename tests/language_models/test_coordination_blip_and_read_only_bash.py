"""A recovered coordination blip and a read-only shell are not a failed Claude request.

Production, subjective-devagents, 2026-11-22 (America/Sao_Paulo). Turn
0a09676e-371f-45ef-a77d-38459ebbc726 on channel conv_5995bbc1-f447-42e7-accc-727c6d01d8ad
(claude-opus-5-5, api_agent, proxy 127.0.0.1:8317) started 09:21:34 and finished
`status=completed`, error_code null, 2169 chars, 488076 ms. cli-proxy-api answered all 35
POST /v1/chat/completions with HTTP 200 and FastAPI logged no "LLM send failed" — the model
never failed. What the turn did contain:

* one `Coordination unavailable; retrying op=snapshot attempt=1` warning, with no status
  code and no reason because `_call_once` collapsed every failure into a bare
  `coordination_unavailable` raised `from None`;
* three TOOL_BLOCKED_COORDINATION Bash refusals (seq 147427 WRITE_LEASE_REQUIRED,
  147502 and 147628 LEASE_SCOPE_MISMATCH) for commands that only read.

The chat still showed a generic Claude request failure, which no server log contains.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from subjective_devagents_fastapi import channel_coordination as cc
from subjective_devagents_fastapi.channel_coordination import ChannelSession, shell_effect, unavailable
from subjective_devagents_fastapi.language_models import (
    Identity, IdentityRegistry, LanguageModel, ModelResult, ModelStrategy, TurnStore,
)
from subjective_devagents_fastapi.language_models.contracts import ModelError

CHANNEL = 'conv_5995bbc1-f447-42e7-accc-727c6d01d8ad'
BINDING = {'channel_id': CHANNEL, 'dispatch_id': 'dispatch', 'instance_id': 'ai_coder:run',
           'workflow_id': 'workflow'}


def journal_state():
    return {'enabled': True, 'workflow_id': 'workflow', 'workflow_version': 'published',
            'task_family': 'TASK', 'task_revision': '1', 'revision': 1,
            **{key: {} for key in ('facts', 'leases', 'decisions', 'capabilities', 'reviews',
                                   'gates', 'worktrees', 'discoveries')}}


class Blip:
    """A coordination endpoint that fails the first POST and then answers normally."""

    def __init__(self, status=503):
        self.status = status
        self.posts = 0

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length') or 0))
                outer.posts += 1
                if outer.posts == 1:
                    body, status = b'upstream temporarily unavailable', outer.status
                else:
                    body = json.dumps({'ok': True, 'data': {'state': journal_state()}}).encode()
                    status = 200
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.fixture(autouse=True)
def quick_retry(monkeypatch):
    monkeypatch.setattr(cc, 'CALL_BACKOFF', (0.01, 0.02))


def session(authorization='Bearer test-only', client=None):
    return ChannelSession(BINDING, 'ai_coder', authorization, client=client)


# --- 1. a blip that the retry recovers from --------------------------------------------


def test_a_recovered_snapshot_blip_logs_the_status_and_the_reason(monkeypatch, caplog):
    with Blip(503) as endpoint:
        monkeypatch.setenv('SUBJECTIVE_CHAT_COORDINATION_URL', endpoint.url)
        bound = session()
        with caplog.at_level(logging.INFO, logger=cc.__name__):
            state = bound.call('snapshot')['state']

    assert state['task_family'] == 'TASK', 'attempt 2 succeeded, so the caller sees no failure'
    assert endpoint.posts == 2
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert 'op=snapshot' in warnings[0] and 'attempt=1' in warnings[0]
    assert 'status=503' in warnings[0] and 'reason=http_status' in warnings[0]
    assert 'retrying' in warnings[0] and 'giving up' not in warnings[0]
    assert any('Coordination recovered op=snapshot attempts=2' in r.getMessage() for r in caplog.records)


def test_a_transport_error_is_named_without_quoting_the_request(monkeypatch, caplog):
    calls = []
    real = httpx.Client.post

    def flaky(self, url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ReadTimeout('timed out')
        return real(self, url, **kwargs)

    with Blip(503) as endpoint:
        endpoint.posts = 1          # the stub answers normally from here on
        monkeypatch.setenv('SUBJECTIVE_CHAT_COORDINATION_URL', endpoint.url)
        monkeypatch.setattr(httpx.Client, 'post', flaky)
        bound = session(authorization='Bearer super-secret-token')
        with caplog.at_level(logging.INFO, logger=cc.__name__):
            assert bound.call('snapshot')['state']['enabled'] is True

    warning = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert 'reason=transport_ReadTimeout' in warning and 'status=None' in warning
    for record in caplog.records:
        message = record.getMessage()
        assert 'super-secret-token' not in message
        assert 'Authorization' not in message and 'upstream temporarily unavailable' not in message


def test_a_turn_whose_snapshot_blipped_still_completes(tmp_path):
    """The turn result the chat renders: completed, no error_code."""
    attempts = []

    def flaky_journal(body):
        attempts.append(body['op'])
        if body['op'] == 'snapshot' and attempts.count('snapshot') == 1:
            raise unavailable('Channel coordination is temporarily unavailable',
                              status=503, reason='http_status')
        return {'state': journal_state()}

    identity = Identity('runtime', 'claude', model='claude-opus-5-5', strategy='api_agent',
                        workspace=str(tmp_path), home=str(tmp_path), sandbox='workspace-write',
                        api_base_url='http://127.0.0.1:8317/v1', api_key_env='TEST_KEY')
    store = TurnStore(str(tmp_path / 'language_models.sqlite3'))
    bound = session(client=flaky_journal)
    # The recorded warning is op=snapshot attempt=1 inside the turn: that is the dispatch's
    # own first snapshot, taken by start() when the worker picks the turn up.
    bound.defer_start(identity, 'published', 'coder_agent')

    class Working(ModelStrategy):
        def run(self, request, emit, cancelled):
            return ModelResult('completed', 'A' * 2169)

    model = LanguageModel(store, IdentityRegistry([identity]), {'api_agent:claude': Working()})
    try:
        conversation = model.create(identity.id)
        turn = model.submit(conversation['id'], 'investigate the failure', 'request-1', timeout=60,
                            coordination=bound)
        recorded = await_turn(store, turn['id'])
    finally:
        model.close(timeout=5)

    assert recorded['status'] == 'completed' and not recorded['error_code']
    assert recorded['text'].count('A') == 2169
    assert attempts.count('snapshot') >= 2, 'the retry has to have happened inside the turn'


def await_turn(store, identifier, timeout=10):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = store.turn(identifier)
        if turn['status'] not in {'queued', 'running'}:
            return turn
        time.sleep(0.01)
    pytest.fail('turn did not finish')


# --- 3. read-only Bash needs no write lease, and still fails closed --------------------


@pytest.fixture
def guarded(tmp_path):
    """A workspace-write dispatch holding no lease at all, as in the recorded turn."""
    workspace = tmp_path / 'workspace'
    (workspace / 'services' / 'subjective_devagents_fastapi').mkdir(parents=True)
    identity = Identity('runtime', 'claude', model='claude-opus-5-5', strategy='api_agent',
                        workspace=str(workspace), home=str(tmp_path), sandbox='workspace-write',
                        api_base_url='http://127.0.0.1:8317/v1', api_key_env='TEST_KEY')
    calls = []

    def journal(body):
        calls.append(body['op'])
        return {'state': journal_state()}

    bound = session(client=journal)
    bound.start(identity, 'published', 'coder_agent')
    calls.clear()
    return bound, calls, workspace


@pytest.mark.parametrize('command', [
    'git log --oneline -5',
    'git status --short',
    'cd services/subjective_devagents_fastapi && git log --oneline -5',
    'cd services/subjective_devagents_fastapi && grep -rn "coordination_unavailable" src | head -20',
    "sed -n '1,40p' src/subjective_devagents_fastapi/channel_coordination.py",
    'cat README.md | wc -l',
    'curl -s http://127.0.0.1:8317/v1/models',
    'curl -sS -X GET http://127.0.0.1:8317/health',
    'git log --format=%H | sort | uniq | wc -l',
    'awk "{print $1}" README.md',
])
def test_read_only_bash_runs_without_a_write_lease(guarded, command):
    bound, calls, _ = guarded
    assert shell_effect(command) == 'READ_ONLY', command
    bound.guard('Bash', {'command': command})
    assert calls == [], 'a read-only command must not even ask the journal for a lease'


@pytest.mark.parametrize('command', [
    'cd $HOME && git status',
    'cd "$(dirname /tmp/x)" && git log',
    "cd 'unterminated && git log",
    'cd ../../.. && git log --oneline',
])
def test_a_non_literal_or_escaping_cd_still_fails_closed(guarded, command):
    bound, _, _ = guarded
    with pytest.raises(ModelError) as raised:
        bound.guard('Bash', {'command': command})
    assert raised.value.code == 'LEASE_SCOPE_MISMATCH'
    assert str(raised.value), 'the refusal carries a one-line reason, not an empty message'


@pytest.mark.parametrize('command', [
    'curl -o notes.txt http://127.0.0.1:8317/v1/models',
    'curl -sO http://127.0.0.1:8317/v1/models',
    'curl -X POST -d @payload.json http://127.0.0.1:8317/v1/chat/completions',
    'sort -o sorted.txt README.md',
    'sed -i s/a/b/ README.md',
    'git commit -m wip',
])
def test_a_writing_command_is_not_mistaken_for_a_read(guarded, command):
    bound, _, _ = guarded
    assert shell_effect(command) in {'WRITE', 'UNKNOWN'}, command
    with pytest.raises(ModelError) as raised:
        bound.guard('Bash', {'command': command})
    assert raised.value.code in {'WRITE_LEASE_REQUIRED', 'LEASE_SCOPE_MISMATCH'}


def test_the_refusal_the_client_sees_is_the_code_and_its_reason(guarded):
    """Not "Request failed": the tool result carries the code, the reason and a JSON outcome."""
    from subjective_devagents_fastapi.language_models.strategies import (
        coordination_refusal, coordination_refused,
    )
    bound, _, _ = guarded
    arguments = {'command': 'cd $HOME && git status'}
    with pytest.raises(ModelError) as raised:
        bound.guard('Bash', arguments)

    text = coordination_refusal(raised.value, 'Bash', arguments)
    assert coordination_refused(text)
    assert text.startswith('Error: LEASE_SCOPE_MISMATCH: ')
    first, structured = text.split('\n', 1)
    assert len(first) > len('Error: LEASE_SCOPE_MISMATCH: '), 'the one-line reason is present'
    outcome = json.loads(structured)
    assert outcome['error_code'] == 'LEASE_SCOPE_MISMATCH' and outcome['executed'] is False
    assert outcome['retryable'] is True and outcome['recovery_action'] == 'ACQUIRE_LEASE'


def test_an_unset_callback_url_is_not_retried_for_fifteen_seconds(monkeypatch, caplog):
    """A missing environment variable is configuration, not a transient outage."""
    import time

    monkeypatch.setattr(cc, 'CALL_BACKOFF', (1.0, 2.0, 4.0, 8.0))
    monkeypatch.delenv('COORDINATION_CALLBACK_URL', raising=False)
    bound = session()
    bound.delegation = {**BINDING, 'owner_id': 'owner'}

    began = time.monotonic()
    with caplog.at_level(logging.INFO, logger=cc.__name__):
        with pytest.raises(ModelError) as raised:
            bound.call('snapshot')
    assert time.monotonic() - began < 1
    assert raised.value.code == 'coordination_unavailable'
    assert cc.transport_reason(raised.value) == 'callback_url_unset'
    assert any('callback URL is not configured' in r.getMessage() for r in caplog.records)


# --- a refusal only a person can act on must reach that person -------------------------
#
# 2026-11-23: a push was refused with PUBLICATION_SCOPE_CONFLICT naming the foreign commit
# and two concrete remedies. coordination_refusal returned that text only as a tool result
# into the model's own message list, so the operator saw the model's paraphrase of it:
# "@Name can push or add ce801ce to assignment. Needs input on architect prompt: (a) or (b)"
# - no SHA list, no remedies, and a literal placeholder.


def test_a_refusal_the_agent_cannot_retry_is_emitted_for_the_operator():
    from subjective_devagents_fastapi.language_models.strategies import coordination_refusal

    events = []
    detail = ("Pushing to origin/main would also publish 1 commit(s) that do not belong to this "
              "task: ce801ce01234 'unrelated work' (ai_other). Publish only this task: create a "
              "branch from origin/main, cherry-pick this task's commits and push that, or report "
              "PUBLICATION_SCOPE_CONFLICT to Manager, who can approve bundling by naming these "
              "SHAs in the assignment.")
    text = coordination_refusal(ModelError('PUBLICATION_SCOPE_CONFLICT', detail), 'Bash',
                                {'command': 'git push origin main'},
                                lambda kind, data: events.append((kind, data)))

    assert text.startswith('Error: PUBLICATION_SCOPE_CONFLICT: ')      # still reaches the model
    steps = [data['text'] for kind, data in events if kind == 'activity']
    assert len(steps) == 1, events
    assert 'ce801ce01234' in steps[0] and 'cherry-pick' in steps[0], 'the operator gets the specifics'


def test_a_refusal_the_agent_can_retry_is_not_announced_to_the_operator():
    """WRITE_LEASE_REQUIRED is the runtime's own business: it acquires and carries on."""
    from subjective_devagents_fastapi.language_models.strategies import coordination_refusal

    events = []
    coordination_refusal(ModelError('WRITE_LEASE_REQUIRED', 'Acquire a covering lease'), 'Write',
                         {'file_path': 'a.py'}, lambda kind, data: events.append((kind, data)))
    assert [kind for kind, _ in events] == []


def test_a_handoff_before_any_model_call_would_only_repeat_itself(tmp_path, monkeypatch):
    """The budget is internal pacing, not the provider limit: handing off from an oversized
    input produces a successor with the same oversized input, up to the chain cap."""
    import asyncio
    from dataclasses import replace as replace_fields
    from subjective_devagents_fastapi.language_models import ApiAgent, ModelRequest
    from subjective_devagents_fastapi.language_models.strategies import history_budget

    monkeypatch.setenv('SUBJECTIVE_AGENT_CONTEXT_TOKENS', '2000')      # everything is over budget
    monkeypatch.setenv('TEST_KEY', 'fixture')
    identity = Identity('runtime', 'claude', model='claude-opus-5-5', strategy='api_agent',
                        workspace=str(tmp_path), home=str(tmp_path), sandbox='workspace-write',
                        api_base_url='http://127.0.0.1:1/v1', api_key_env='TEST_KEY')
    bound = session(client=lambda body: {'state': journal_state()})
    bound.start(identity, 'published', 'coder_agent')

    agent, events = ApiAgent(), []
    calls = []

    async def answer(_identity, _token, messages, _tools, _timeout, _cancelled):
        calls.append(len(messages))
        return {'content': 'Answered despite the oversized input.'}, 'stop', {}

    monkeypatch.setattr(agent, '_call', answer)
    request = ModelRequest('turn', identity, 'x' * 40_000, timeout=30, coordination=bound)
    result = asyncio.run(agent._run(request, lambda kind, data: events.append((kind, data)),
                                    lambda: False, {}))

    assert result.status == 'completed', result
    assert calls, 'the model was called instead of handing off from a cold start'
    assert any(kind == 'context_budget_exceeded_at_start' for kind, _ in events)
    assert not any(data.get('handoff') for kind, data in events if kind == 'context_compaction')
    assert history_budget() > 0
