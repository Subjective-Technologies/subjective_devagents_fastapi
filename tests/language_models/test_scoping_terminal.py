"""SCOPING_LIMIT is terminal in the runtime: no model request, tool or heartbeat after it."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import ApiAgent, strategies
from subjective_devagents_fastapi.language_models.contracts import ModelError
from subjective_devagents_fastapi.language_models.task_context import SCOPING_CHECKPOINT_LIMIT, SCOPING_FIELDS
from test_api_agent import identity, request_data, tool_call  # noqa: F401  (fixtures)

SENTINEL = 'RAW_OLD_TOOL_OUTPUT_7f3c'


class Control(SimpleNamespace):
    """The parts of ChannelSession ApiAgent uses, recording what it is asked to do."""

    def __init__(self, previous=None):
        super().__init__(agent_role='coder_agent', previous_checkpoint=previous, terminal=None, touches=0,
                         recorded=[])

    def context(self):
        return 'CHANNEL RUNTIME CONTEXT'

    def revoked(self):
        return bool(self.terminal)

    def checkpoint(self, checkpoint):
        self.recorded.append(checkpoint)

    def mark_terminal(self, reason):
        self.terminal = reason

    def activity(self, *args, **kwargs):
        assert not self.terminal, 'activity after SCOPING_LIMIT'
        self.touches += 1


def searching_agent(monkeypatch, calls, files=('src/ui/app.js', 'src/core/history.js', 'src/model/store.js')):
    async def respond(_identity, _token, messages, _tools, _timeout, _cancelled):
        calls.append(messages)
        n = len(calls)
        path = files[n % len(files)]
        name = ('Read', 'Grep', 'Glob')[n % 3]
        args = {'file_path': path, 'offset': n * 100} if name == 'Read' else {'path': path, 'pattern': f'q{n}'}
        return {'content': f'Looking at {path} for the Apply handler (step {n}).',
                'tool_calls': [tool_call(str(n), name, **args)]}, 'tool_calls', {}

    agent = ApiAgent()
    monkeypatch.setattr(agent, '_call', respond)
    monkeypatch.setattr(agent, '_dispatch', lambda *_a, **_k: SENTINEL + ' ' + 'x' * 5000)
    return agent


def test_scoping_limit_stops_every_model_call_and_returns_a_compact_checkpoint(request_data, monkeypatch):
    monkeypatch.setenv('TEST_AGENT_KEY', 'fixture-key')
    calls, control = [], Control()
    agent = searching_agent(monkeypatch, calls)
    result = agent.run(replace(request_data, timeout=600, coordination=control), lambda *_: None, lambda: False)

    assert result.status == 'blocked' and result.error_code == 'coder_scoping_limit'
    assert len(calls) == strategies.CODER_SCOPING_ROUNDS      # not one model request more
    assert control.terminal == 'coder_scoping_limit'
    checkpoint = result.checkpoint
    assert set(SCOPING_FIELDS) <= set(checkpoint) and checkpoint['task_phase'] == 'NEEDS_RESCOPE'
    assert checkpoint['files_inspected'][:3] == ['src/core/history.js', 'src/model/store.js', 'src/ui/app.js']
    assert checkpoint['files_changed'] == [] and checkpoint['checks_completed'] == []
    stats = checkpoint['scoping']
    assert stats['scoping_steps'] == strategies.CODER_SCOPING_ROUNDS
    assert stats['file_reads'] + stats['searches'] == strategies.CODER_SCOPING_ROUNDS
    assert stats['planning_turns'] == strategies.CODER_SCOPING_ROUNDS and stats['files_modified'] == 0
    # Small, and never the transcript or raw tool output.
    assert len(json.dumps(checkpoint)) <= SCOPING_CHECKPOINT_LIMIT and SENTINEL not in json.dumps(checkpoint)
    assert '12 scoping steps, 0 files changed, 0 checks completed' in result.text
    assert 'Returned to Manager for rescoping' in result.text


def test_narrowed_successor_gets_a_fresh_budget_and_no_old_tool_history(request_data, monkeypatch):
    monkeypatch.setenv('TEST_AGENT_KEY', 'fixture-key')
    first_calls, first = [], Control()
    blocked = searching_agent(monkeypatch, first_calls).run(
        replace(request_data, timeout=600, coordination=first), lambda *_: None, lambda: False)
    previous = dict(blocked.checkpoint, dispatch_id='blocked-dispatch')

    calls, successor = [], Control(previous=previous)

    async def implement(_identity, _token, messages, _tools, _timeout, _cancelled):
        calls.append(messages)
        if len(calls) == 1:
            return {'content': 'Adding Apply.', 'tool_calls': [tool_call('e', 'Edit', file_path='src/ui/app.js')]}, 'tool_calls', {}
        return {'content': 'IMPLEMENTATION_COMPLETE: Apply added.'}, 'stop', {}

    def dispatch(_call, _tools, _identity, _transport, _cancelled, _timeout, emit, *_args):
        emit('file_diff', {'path': 'src/ui/app.js', 'diff': '+apply'})
        return 'ok'

    agent = ApiAgent()
    monkeypatch.setattr(agent, '_call', implement)
    monkeypatch.setattr(agent, '_dispatch', dispatch)
    brief = 'TASK: In src/ui/app.js only, implement Apply.\nCheckpoint: ' + json.dumps(previous)
    result = agent.run(replace(request_data, prompt=brief, timeout=600, coordination=successor),
                       lambda *_: None, lambda: False)
    assert result.status == 'completed' and successor.terminal is None
    # The successor's context is its brief: the old dispatch's tool output never travels.
    assert all(SENTINEL not in json.dumps(messages) for messages in calls)
    assert 'src/core/history.js' in json.dumps(calls[0])   # already inspected, handed over


def test_successor_scoping_steps_count_only_its_own_rounds(request_data, monkeypatch):
    monkeypatch.setenv('TEST_AGENT_KEY', 'fixture-key')
    previous = {'schema': 'scoping/1', 'files_inspected': ['src/ui/app.js'], 'confirmed': ['Code view is read-only'],
                'scoping': {'scoping_steps': 12}}
    calls, control = [], Control(previous=previous)
    result = searching_agent(monkeypatch, calls, files=('src/ui/editor.js',)).run(
        replace(request_data, timeout=600, coordination=control), lambda *_: None, lambda: False)
    assert result.checkpoint['scoping']['scoping_steps'] == strategies.CODER_SCOPING_ROUNDS   # not 24
    assert result.checkpoint['files_inspected'][:2] == ['src/ui/app.js', 'src/ui/editor.js']
    assert result.checkpoint['confirmed'][0] == 'Code view is read-only'


def test_failed_check_is_implementation_progress(request_data, monkeypatch):
    """A reproduced failure (a check that ran and failed) ends scoping like a passing one."""
    monkeypatch.setenv('TEST_AGENT_KEY', 'fixture-key')
    calls = []

    async def respond(_identity, _token, messages, _tools, _timeout, _cancelled):
        calls.append(messages)
        if len(calls) == 1:
            return {'tool_calls': [tool_call('t', 'Bash', command='pytest -q tests/test_x.py')]}, 'tool_calls', {}
        if len(calls) <= strategies.CODER_SCOPING_ROUNDS + 3:
            return {'tool_calls': [tool_call(str(len(calls)), 'Grep', pattern=f'p{len(calls)}')]}, 'tool_calls', {}
        return {'content': 'Reproduced the failure.'}, 'stop', {}

    agent = ApiAgent()
    monkeypatch.setattr(agent, '_call', respond)
    monkeypatch.setattr(agent, '_dispatch', lambda call, *_a, **_k: (
        '1 failed\n[exit code 1]' if 'pytest' in call['function']['arguments'] else 'match'))
    monkeypatch.setattr(strategies, 'workspace_snapshot', lambda *_a: {})
    result = agent.run(replace(request_data, timeout=600, coordination=Control()), lambda *_: None, lambda: False)
    assert result.status == 'completed'


def session():
    journal = SimpleNamespace(ops=[])

    def call(op, **data):
        journal.ops.append(op)
        return {}
    item = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'ai_coder:1', 'workflow_id': 'w'},
                          'ai_coder', 'Bearer test-only')
    item.call = call
    item.identity = SimpleNamespace(sandbox='workspace-write', ssh_host=None, workspace='/tmp')
    return item, journal


def test_terminal_session_executes_nothing_but_still_reports_finish():
    item, journal = session()
    item.activity('Read', phase='tool')
    assert journal.ops == ['touch']
    item.mark_terminal('coder_scoping_limit')
    assert item.revoked() is True and item.can_execute() is False
    item.activity('Grep', phase='working')                      # "Preparing next step" never reaches the journal
    item.activity(usage={'id': 'u1', 'total_tokens': 10})       # a late usage receipt is kept for billing only
    item.checkpoint({'reason': 'late'})
    for call in (lambda: item.tool('acquire', {}), lambda: item.guard('Edit', {'file_path': 'a'})):
        with pytest.raises(ModelError) as refused:
            call()
        assert refused.value.code == 'DISPATCH_TERMINAL'
    assert journal.ops == ['touch']
    item.finish('blocked', 'coder_scoping_limit', result={'text': 'BLOCKED'})
    assert journal.ops == ['touch', 'finish']
    assert item.completion_payload('blocked', 'coder_scoping_limit')['data']['usage_receipts'][0]['id'] == 'u1'
