"""Acceptance scenarios A-E from the production chats, across Chat journal and DevAgents.

A  Manager assigns an agent, the runtime is busy / the dispatch never starts, nobody acts.
B  Coder hits tool_loop_limit_exceeded; a successor continues the same task from its checkpoint.
C  Verifier assigned tests/build needs a Node the host lacks: explicit blocker at once.
D  The coordination service is briefly unavailable: callbacks back off and recover.
E  Five retries produce one logical retry and one active execution.
Plus: a runtime restart reports service_restarted (a successor), not a user cancellation.
"""
import asyncio
import threading
import time

import httpx
import pytest

from test_collaboration_recovery import system  # noqa: F401 - shared cross-service fixture
from test_model_api import Waiting, engine  # noqa: F401
from test_channel_coordination import bound  # noqa: F401
from subjective_chat_api.coordination import ACTIVE_DISPATCH
from subjective_chat_api.execution_queue import ExecutionQueue
from subjective_devagents_fastapi import channel_coordination, collaboration
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import ModelResult, ModelStrategy
from subjective_devagents_fastapi.language_models.contracts import ModelError


def runtime_endpoint(service):
    """What /api/collaboration/execute returns over HTTP, including refusals."""
    def deliver(payload, _signed):
        try:
            return collaboration.execute(service, {'binding': {k: payload['binding'][k] for k in ('channel_id', 'dispatch_id')}})
        except ModelError as error:
            return {'ok': False, 'error': error.code, 'message': str(error)}
    return deliver


def tick(s):
    asyncio.run(ExecutionQueue(s.app, deliver=runtime_endpoint(s.service)).tick())


def dispatches(s, agent=None):
    return [x for x in s.journal.snapshot(s.channel)['dispatches'].values() if agent is None or x['agent_id'] == agent]


def test_scenario_a_busy_runtime_queues_visibly_then_starts_without_user(system):
    s = system
    ran = threading.Event()

    class Quick(ModelStrategy):
        def run(self, request, emit, cancelled):
            ran.set()
            return ModelResult('completed', 'TASK COMPLETE: verified')
    s.model.strategies['api_agent:codex'] = Quick()
    item = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'verify',
                                                 'execution': s.execution})['dispatch']
    full = {'busy': 16, 'workers': 16, 'available': 0}
    real_capacity = s.model.capacity
    s.model.capacity = lambda: full
    s.clock[0] = time.time()
    tick(s)                                   # the runtime answers "all workers busy"
    waiting = s.journal.snapshot(s.channel)['dispatches'][item['id']]
    assert waiting['status'] == 'DISPATCHED' and waiting['waiting_reason'] == 'RUNTIME_CAPACITY'
    # Reserved 200 s ago, past the start timeout, but the runtime keeps answering "busy":
    # still queued, visibly, not FAILED_TO_START.
    import json
    row = json.loads(s.database.query_one('SELECT payload FROM channel_coordination WHERE channel_id=?', (s.channel,))['payload'])
    row['dispatches'][item['id']]['dispatched_at'] -= 200
    s.database.execute('UPDATE channel_coordination SET payload=? WHERE channel_id=?', (json.dumps(row), s.channel))
    s.journal.mutate(s.channel, 'runtime:watchdog', {'op': 'reconcile'})
    assert s.journal.snapshot(s.channel)['dispatches'][item['id']]['status'] == 'DISPATCHED'
    s.model.capacity = real_capacity
    s.database.execute('UPDATE coordination_jobs SET next_attempt_at=0')
    tick(s)
    assert ran.wait(5), 'queued dispatch did not start once a worker was free'
    s.model.close()
    final = s.journal.snapshot(s.channel)['dispatches'][item['id']]
    assert final['status'] == 'COMPLETED' and final['delivered_at']


def test_scenario_b_tool_loop_limit_successor_continues_from_checkpoint(system):
    s = system
    seen, done = [], threading.Event()

    class Looping(ModelStrategy):
        def run(self, request, emit, cancelled):
            seen.append((request.coordination.binding['dispatch_id'], request.prompt, (s.repo / 'a.py').read_text()))
            if len(seen) == 1:
                (s.repo / 'a.py').write_text('half done\n')
                return ModelResult('failed', 'CONTINUATION_SUMMARY (automatic recovery ...):\n{"files_changed": ["a.py"], '
                                   '"next_action": "finish a.py"}', error_code='tool_loop_limit_exceeded')
            done.set()
            return ModelResult('completed', 'TASK COMPLETE: finished from checkpoint')
    s.model.strategies['api_agent:codex'] = Looping()
    s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'Implement TASK-1',
                                          'execution': s.execution})
    tick(s)
    deadline = time.time() + 5
    while time.time() < deadline and not any(x['status'] == 'DISPATCHED' and x['successor_of'] for x in dispatches(s)):
        time.sleep(0.05)
    tick(s)
    assert done.wait(5), 'successor did not continue'
    s.model.close()
    first, second = seen
    assert first[0] != second[0]
    assert 'CONTINUATION_SUMMARY' in second[1] and 'finish a.py' in second[1] and 'Implement TASK-1' in second[1]
    assert second[2] == 'half done\n'   # workspace preserved, not reset
    rows = dispatches(s)
    assert rows[0]['failure_code'] == 'TOOL_LOOP_LIMIT' or any(x['failure_code'] == 'TOOL_LOOP_LIMIT' for x in rows)
    assert [x['status'] for x in rows if x['status'] in ACTIVE_DISPATCH] == []


def test_scenario_c_missing_toolchain_blocks_verifier_immediately(system):
    s = system
    s.repository.add_member(s.channel, 'ai_integration')
    s.execution['agents']['ai_integration'] = {'user_id': 'ai_integration', 'company_id': 'company_test', 'provider': 'codex',
                                              'role_file': 'agents/roles/integration_agent.md', 'display_name': 'Integration',
                                              'mode': 'read'}
    s.execution['prompt'] = '@Integration Agent run tests, run tsc and run build on Node 41'
    item = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_integration', 'assignment': 'verify',
                                                 'execution': s.execution})['dispatch']
    tick(s)
    failed = s.journal.snapshot(s.channel)['dispatches'][item['id']]
    assert failed['status'] == 'FAILED' and failed['failure_code'] == 'BLOCKED_NO_COMPATIBLE_RUNTIME'
    assert 'Node 41 not installed' in failed['failure_reason']
    assert not [x for x in dispatches(s, 'ai_integration') if x['status'] in ACTIVE_DISPATCH]


def test_scenario_d_transient_callback_outage_is_retried(monkeypatch):
    monkeypatch.setattr(channel_coordination, 'CALL_BACKOFF', (0, 0, 0))
    calls = []

    def flaky(body):
        calls.append(body['op'])
        if len(calls) < 3:
            raise ModelError('coordination_unavailable', 'down')
        return {'state': {}}
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'i', 'workflow_id': 'w'},
                             'ai_coder', 'Bearer x', client=flaky)
    session.call('finish', status='completed')
    assert calls == ['finish', 'finish', 'finish']
    calls.clear()
    with pytest.raises(ModelError):
        session.call('touch')        # liveness pings rely on the grace window, not on blocking retries
    assert calls == ['touch']


def test_scenario_d_persistent_outage_is_explicit(monkeypatch):
    monkeypatch.setattr(channel_coordination, 'CALL_BACKOFF', (0, 0))

    def down(body):
        raise ModelError('coordination_unavailable', 'down')
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'i', 'workflow_id': 'w'},
                             'ai_coder', 'Bearer x', client=down)
    with pytest.raises(ModelError) as error:
        session.call('check', path='/x')
    assert error.value.code == 'coordination_unavailable'


def test_http_5xx_from_a_restarting_chat_container_is_transient(monkeypatch):
    monkeypatch.setattr(channel_coordination, 'CALL_BACKOFF', (0,))
    monkeypatch.setenv('SUBJECTIVE_CHAT_COORDINATION_URL', 'https://chat.invalid')
    responses = [httpx.Response(503, text='Container failed to become ready'),
                 httpx.Response(200, json={'ok': True, 'data': {'state': {'revision': 1}}})]
    real = httpx.Client

    class Fake:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw): return responses.pop(0)
    monkeypatch.setattr(channel_coordination.httpx, 'Client', Fake)
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'i', 'workflow_id': 'w'}, 'ai_coder', 'Bearer x')
    assert session.call('snapshot')['state']['revision'] == 1
    monkeypatch.setattr(channel_coordination.httpx, 'Client', real)


def test_scenario_e_five_retries_one_active_execution(system):
    s = system
    first = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'Implement TASK-1',
                                                  'execution': s.execution})['dispatch']
    s.clock[0] += 121
    for _ in range(5):
        s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'retry',
                                              'execution': s.execution})
    live = [x for x in dispatches(s) if x['status'] in ACTIVE_DISPATCH]
    assert len(live) == 1 and live[0]['successor_of'] == first['id']


def test_runtime_restart_is_reported_as_service_restarted(engine):
    waiting = Waiting()
    engine.strategies['cli:claude'] = waiting
    conversation = engine.create('claude-a')
    turn = engine.submit(conversation['id'], 'wait', 'one')
    assert waiting.started.wait(2)
    engine.close()
    row = engine.store.turn(turn['id'])
    assert row['status'] == 'cancelled' and row['error_code'] == 'service_restarted'


def test_capacity_reports_busy_workers(engine):
    waiting = Waiting()
    engine.strategies['cli:claude'] = waiting
    assert engine.capacity()['available'] == engine.max_workers
    conversation = engine.create('claude-a')
    engine.submit(conversation['id'], 'wait', 'one')
    assert waiting.started.wait(2)
    assert engine.capacity()['busy'] == 1
    engine.close()


def test_tool_loop_limit_returns_a_structured_continuation_summary(bound, monkeypatch):
    from dataclasses import replace
    import json
    from subjective_devagents_fastapi.language_models import ApiAgent, ModelRequest
    from subjective_devagents_fastapi.language_models import strategies
    from test_api_agent import ScriptedProxy, completion, tool_call
    session, journal, identity = bound
    monkeypatch.setenv('TEST_KEY', 'fixture')
    monkeypatch.setattr(strategies, 'MAX_TOOL_ROUNDS', 2)
    with ScriptedProxy([completion(content='Edited the parser; now testing.', tool_calls=[tool_call('1', 'Bash', command='ls')]),
                        completion(tool_calls=[tool_call('2', 'Bash', command='pytest -q')])]) as proxy:
        result = ApiAgent().run(ModelRequest('t', replace(identity, api_base_url=proxy.url), 'work', timeout=10,
                                             coordination=session), lambda *_: None, lambda: False)
    assert result.status == 'failed' and result.error_code == 'tool_loop_limit_exceeded'
    assert result.text.startswith('CONTINUATION_SUMMARY')
    body = json.loads(result.text.split('\n', 1)[1])
    assert body['tool_rounds'] == 2 and body['work_completed'] == 'Edited the parser; now testing.'
    assert body['tests_run'] and body['tests_run'][0].startswith('pytest -q')
    assert {'files_changed', 'current_failure', 'next_action', 'workspace_state'} <= set(body)
