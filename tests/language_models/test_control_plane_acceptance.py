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
    assert body['tool_rounds'] == 2 and body['work_completed'] == ['Edited the parser; now testing.']
    assert any(entry.startswith('pytest -q') for entry in body['tests_passed'] + body['tests_failed'])
    assert {'goal', 'phase', 'files_changed', 'tests_passed', 'tests_failed', 'current_failure', 'verified_facts',
            'decisions', 'repositories', 'next_action', 'workspace_state'} <= set(body)
    assert result.checkpoint and result.checkpoint['tool_rounds'] == 2   # structured, not only prose


def test_acceptance_done_latch_real_bug_end_to_end(system):
    """Coder -> Integration PASS -> Git push -> Release exact SHA live -> Manager DONE, browser unavailable.

    Through the real DevAgents runtime and the real Chat journal/outbox: Manager ends DONE with an
    optional confirmation, nothing runs after it, and a late duplicate PASS wakes nobody.
    """
    s = system
    sha = '3ae003d259c1194c4256ed17a827111319567075'
    roles = {'ai_manager': 'manager_agent', 'ai_integration': 'integration_agent', 'ai_git': 'git_agent',
             'ai_release': 'release_agent'}
    for agent, role in roles.items():
        s.repository.add_member(s.channel, agent)
        s.execution['agents'][agent] = {'user_id': agent, 'company_id': 'company_test', 'provider': 'codex',
                                        'role_file': f'agents/roles/{role}.md', 'display_name': agent,
                                        'mode': 'read' if role == 'integration_agent' else 'write'}
        s.export['files'].append({'path': f'agents/roles/{role}.md', 'content': 'Do your role.'})
    s.execution['manager_agent_id'] = 'ai_manager'
    answers = {
        'ai_manager': ['@ai_coder fix the modal', '@ai_integration verify ' + sha, '@ai_git push ' + sha,
                       '@ai_release verify production',
                       'Both fixes are live.\nDONE: production serves ' + sha[:7] + ' and the live bundle has the fix\n'
                       'OPTIONAL_USER_CONFIRMATION: Hard-refresh and confirm the interaction if you want.'],
        'ai_coder': ['IMPLEMENTATION_COMPLETE: ' + sha],
        'ai_integration': ['VERIFICATION_PASS: ' + sha + '\nACCEPTANCE_VERIFIED: AC1 — opened New task in a browser; '
                           'the modal renders above the page, not inside the right pane'],
        'ai_git': ['PUBLICATION_COMPLETE: refs/heads/main is now ' + sha],
        'ai_release': ['/__build.json reports ' + sha + '\nDEPLOYMENT_VERIFIED: ' + sha +
                       '\nOPTIONAL_USER_CONFIRMATION: authenticated browser session unavailable'],
    }
    turns = []

    class Scripted(ModelStrategy):
        def run(self, request, emit, cancelled):
            agent = request.coordination.actor
            turns.append(agent)
            # A real executed check: the backend attests it as [RT:...successful=1] on the result.
            emit('tool_call', {'tool_call_id': 'c1', 'name': 'Bash', 'detail': 'npx playwright test'})
            emit('tool_result', {'tool_call_id': 'c1', 'name': 'Bash', 'ok': True, 'output_chars': 10})
            return ModelResult('completed', answers[agent].pop(0) if answers[agent] else 'UNEXPECTED TURN')
    s.model.strategies['api_agent:codex'] = Scripted()
    s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_manager',
                                          'assignment': 'The modal is still inside the right pane on worktwins.com',
                                          'execution': s.execution}, human=True)

    def settle(rounds=40):
        for _ in range(rounds):
            tick(s)
            if not [x for x in dispatches(s) if x['status'] in ACTIVE_DISPATCH]:
                tick(s)
                if not [x for x in dispatches(s) if x['status'] in ACTIVE_DISPATCH]:
                    return
            time.sleep(0.05)
    settle()
    task = s.journal.snapshot(s.channel)['task']
    assert turns == ['ai_manager', 'ai_coder', 'ai_manager', 'ai_integration', 'ai_manager', 'ai_git', 'ai_manager',
                     'ai_release', 'ai_manager']
    assert task['status'] == 'DONE' and task['optional_user_confirmation'].startswith('Hard-refresh')
    assert task['facts']['published_sha'] == [sha] and task['facts']['deployment_verified']
    # No further automatic orchestration: more ticks, a watchdog sweep, nothing runs.
    s.journal.sweep()
    settle(3)
    assert len(turns) == 9 and 'UNEXPECTED TURN' not in str(turns)
    bodies = [m['body'] for m in s.database.query_all('SELECT body FROM messages WHERE conversation_id=?', (s.channel,))]
    assert not any('NEEDS INPUT' in body for body in bodies)


def test_server_answer_carries_changed_files_and_steps(system):
    """Server-runtime answers show the files the agent changed (with diffs) and its steps, like browser rounds."""
    s = system
    diff = '--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-original\n+changed\n'

    class Editing(ModelStrategy):
        def run(self, request, emit, cancelled):
            emit('activity', {'text': 'Running: git status'})
            emit('activity', {'text': 'Codex is still working… 30s'})
            emit('file_diff', {'path': str(s.repo / 'a.py'), 'status': 'modified', 'diff': diff})
            emit('activity', {'text': 'Edit: a.py'})
            emit('file_diff', {'path': str(s.repo / 'secret.env'), 'status': 'created',
                               'diff': '+TOKEN=Bearer abcdefghijklmnop'})
            return ModelResult('completed', 'Changed a.py.\nIMPLEMENTATION_COMPLETE: done')
    s.model.strategies['api_agent:codex'] = Editing()
    s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'Change a.py',
                                          'execution': s.execution}, human=True)
    tick(s)
    deadline = time.time() + 5
    while time.time() < deadline and any(x['status'] in ACTIVE_DISPATCH for x in dispatches(s)):
        time.sleep(0.05)
    tick(s)
    import json
    row = s.database.query_one("SELECT metadata_json FROM messages WHERE conversation_id=? AND sender_id='ai_coder'", (s.channel,))
    meta = json.loads(row['metadata_json'])
    assert [d['path'].rsplit('/', 1)[-1] for d in meta['diffs']] == ['a.py', 'secret.env']
    assert meta['diffs'][0]['diff'] == diff and meta['diffs'][0]['status'] == 'modified'
    assert 'Bearer' not in meta['diffs'][1]['diff'] and 'withheld' in meta['diffs'][1]['diff']
    assert meta['steps'] == ['Running: git status', 'Edit: a.py']


def test_context_handoff_successor_runs_from_the_checkpoint_not_the_transcript(system):
    """Real runtime + journal: a worker hands off at its budget; the successor gets a bounded brief."""
    s = system
    transcript = 'PRIOR CONTEXT (oldest first, each message verbatim):\n' + ('old channel chatter ' * 4000)
    s.execution['prompt'] = (transcript + '\nLATEST MESSAGE, from You — VERBATIM:\nAdd .xmi import to the dashboard'
                             '\n\nCONTEXT METADATA:\nhistory_complete: true\n')
    prompts, done = [], threading.Event()

    class Budgeted(ModelStrategy):
        def run(self, request, emit, cancelled):
            prompts.append(request.prompt)
            if len(prompts) == 1:
                return ModelResult('failed', 'CONTINUATION_SUMMARY (...)', error_code='context_handoff', checkpoint={
                    'schema': 2, 'reason': 'context_handoff', 'phase': 'TESTING', 'work_completed': ['added .xmi'],
                    'files_changed': ['src/files.ts'], 'tests_passed': ['npx vitest run (920 passed, 0 failed)'],
                    'tests_failed': [], 'repositories': [str(s.repo)], 'next_action': 'run the build'})
            done.set()
            return ModelResult('completed', 'IMPLEMENTATION_COMPLETE: built')
    s.model.strategies['api_agent:codex'] = Budgeted()
    s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'Add .xmi import',
                                          'execution': s.execution}, human=True)
    tick(s)
    deadline = time.time() + 5
    while time.time() < deadline and not any(x['status'] == 'DISPATCHED' and x['successor_of'] for x in dispatches(s)):
        time.sleep(0.05)
    tick(s)
    assert done.wait(5)
    s.model.close()
    first, second = prompts
    assert 'old channel chatter' in first and 'old channel chatter' not in second
    assert 'run the build' in second and 'TESTING' in second and 'Add .xmi import' in second
    assert len(second) < len(first) / 5
    kinds = [e['kind'] for e in s.journal.snapshot(s.channel)['events']]
    assert 'SUCCESSOR_AFTER_CONTEXT_LIMIT' in kinds and 'TASK_CHECKPOINT' in kinds


def test_scoping_limit_acceptance_xmi_editor(system, monkeypatch):
    """Production incident conv_8a72147b (2026-11-18, TAGGY-3521), end to end.

    The broad Coder assignment spends its scoping budget in the REAL ApiAgent loop without
    editing. Expected: dispatch BLOCKED, task NEEDS_RESCOPE, no further Coder model call,
    Manager woken at once; Manager narrows to one file; the new Coder dispatch implements.
    """
    import json as _json
    from subjective_devagents_fastapi.language_models import ApiAgent
    from subjective_devagents_fastapi.language_models import strategies
    from test_api_agent import tool_call
    s = system
    monkeypatch.setenv('FIXTURE_KEY', 'fixture')
    s.repository.add_member(s.channel, 'ai_manager')
    s.execution['agents']['ai_manager'] = {'user_id': 'ai_manager', 'company_id': 'company_test', 'provider': 'codex',
                                           'role_file': 'agents/roles/manager_agent.md', 'display_name': 'Manager',
                                           'mode': 'write'}
    s.export['files'].append({'path': 'agents/roles/manager_agent.md', 'content': 'Plan and dispatch.'})
    s.execution['manager_agent_id'] = 'ai_manager'
    broad = ('Make XMI code editable, add Apply and Save, support Undo, update diagram/explorer/properties, '
             'handle invalid XMI, add tests.')
    narrow = 'In src/ui/app.js only, implement Apply for edited XMI.'
    manager_answers = ['@ai_coder ' + broad, '@ai_coder ' + narrow, 'Apply is implemented; waiting for the next step.']
    turns, coder_model_calls = [], []

    searching = ApiAgent()

    async def search(_identity, _token, messages, _tools, _timeout, _cancelled):
        coder_model_calls.append(len(messages))
        n = len(coder_model_calls)
        return {'content': f'Reading the code view (step {n}).',
                'tool_calls': [tool_call(str(n), 'Read', file_path='a.py', offset=n)]}, 'tool_calls', {}
    monkeypatch.setattr(searching, '_call', search)

    class Runtime(ModelStrategy):
        def run(self, request, emit, cancelled):
            agent = request.coordination.actor
            turns.append((agent, request.prompt))
            if agent == 'ai_manager':
                return ModelResult('completed', manager_answers.pop(0) if manager_answers else 'UNEXPECTED TURN')
            if len([t for t in turns if t[0] == 'ai_coder']) == 1:
                return searching.run(request, emit, cancelled)      # the real loop hits SCOPING_LIMIT
            emit('tool_call', {'tool_call_id': 'c1', 'name': 'Bash', 'detail': 'node --test'})
            emit('tool_result', {'tool_call_id': 'c1', 'name': 'Bash', 'ok': True, 'output_chars': 10})
            return ModelResult('completed', 'IMPLEMENTATION_COMPLETE: Apply in src/ui/app.js; node --test passed')
    s.model.strategies['api_agent:codex'] = Runtime()
    s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_manager', 'assignment': 'TAGGY-3521',
                                          'execution': s.execution}, human=True)

    def settle(until, rounds=80):
        for _ in range(rounds):
            tick(s)
            if until():
                return
            time.sleep(0.05)

    coder_blocked = lambda: any(x['agent_id'] == 'ai_coder' and x['status'] == 'BLOCKED' for x in dispatches(s))
    settle(coder_blocked)
    state = s.journal.snapshot(s.channel)
    blocked = next(x for x in state['dispatches'].values() if x['agent_id'] == 'ai_coder')
    assert blocked['status'] == 'BLOCKED' and blocked['failure_code'] == 'CODER_SCOPING_LIMIT'
    assert state['task']['phase'] == 'NEEDS_RESCOPE'
    assert len(coder_model_calls) == strategies.CODER_SCOPING_ROUNDS
    # Manager is runnable in the same transaction that blocked the Coder.
    woken = [x for x in state['dispatches'].values() if x['agent_id'] == 'ai_manager' and x['status'] == 'DISPATCHED']
    assert woken and woken[0]['created_at'] == blocked['completed_at']

    implemented = lambda: s.journal.snapshot(s.channel)['task']['facts']['implementation_complete']
    settle(implemented)
    state = s.journal.snapshot(s.channel)
    coders = sorted((x for x in state['dispatches'].values() if x['agent_id'] == 'ai_coder'), key=lambda x: x['created_at'])
    assert [x['status'] for x in coders] == ['BLOCKED', 'COMPLETED']
    assert coders[1]['parent_dispatch_id'] == blocked['id'] and coders[1]['rescope_reason'] == 'SCOPING_LIMIT'
    # The blocked dispatch never ran again: still exactly its own model calls.
    assert len(coder_model_calls) == strategies.CODER_SCOPING_ROUNDS
    rescope_prompt = next(p for a, p in turns if a == 'ai_manager' and 'RESCOPE_REQUIRED' in p)
    assert '0 files changed' in rescope_prompt
    successor_prompt = [p for a, p in turns if a == 'ai_coder'][1]
    assert narrow in successor_prompt and 'narrowed successor of dispatch ' + blocked['id'] in successor_prompt
    assert '"files_inspected"' in successor_prompt and 'a.py' in successor_prompt   # handed over, not rediscovered
    assert len(successor_prompt) < 24_000    # a brief, not the old transcript
    events = [e['kind'] for e in state['events']]
    for kind in ('SCOPING_LIMIT_REACHED', 'DISPATCH_BLOCKED', 'MANAGER_RESCOPE_REQUESTED', 'RESCOPE_DISPATCH_CREATED'):
        assert kind in events, kind
    # Late model output for the blocked dispatch: audit only.
    session = ChannelSession(ChannelCoordination_binding(state, blocked), 'ai_coder', '')
    session.start_acknowledged = True
    session.finish('completed', None, result={'text': 'late'})
    after = s.journal.snapshot(s.channel)
    assert after['dispatches'][blocked['id']]['status'] == 'BLOCKED' and 'LATE_RESULT_IGNORED' in [e['kind'] for e in after['events']]
    assert 'UNEXPECTED TURN' not in str(turns)
    _json.dumps(after)  # journal stays serializable


def ChannelCoordination_binding(state, item):
    from subjective_chat_api.coordination import ChannelCoordination
    return ChannelCoordination.binding(state, item)
