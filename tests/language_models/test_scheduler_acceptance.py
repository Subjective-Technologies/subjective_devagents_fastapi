"""Exercise two schedulers through the actual worker, fences, tools, and chat journal."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'subjective_api_chat' / 'src'))
pytest.importorskip('subjective_chat_api', reason='Companion Chat API checkout is needed')

from subjective_chat_api import scheduler_transport
from subjective_chat_api.coordination import ChannelCoordination
from subjective_chat_api.db import Database
from subjective_chat_api.errors import ApiError
from subjective_chat_api.execution_queue import ExecutionQueue
from subjective_chat_api.repository import ChatRepository
from subjective_chat_api.service_auth import approved_job
from subjective_devagents_fastapi import agent_workflow, collaboration
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import Identity, LanguageModel
from subjective_devagents_fastapi.language_models.contracts import ModelError
from subjective_devagents_fastapi.language_models.registry import IdentityRegistry
from subjective_devagents_fastapi.language_models.store import TurnStore
from subjective_devagents_fastapi.language_models.strategies import ApiAgent
from subjective_devagents_fastapi.schedulers.api import PlanRequest, plan
from test_collaboration_recovery import git


@pytest.mark.parametrize('live', [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        not os.environ.get('SCHEDULER_ACCEPTANCE_MODEL'), reason='Set SCHEDULER_ACCEPTANCE_MODEL for live model acceptance'
    )),
], ids=['fixture-model', 'live-model'])
def test_two_channels_use_same_executor_and_distinct_worktrees(tmp_path, monkeypatch, live):
    env = {
        'COORDINATION_RUNTIME_URL': 'https://fixture.invalid',
        'COORDINATION_COMPANY_ID': 'company_test',
        'COORDINATION_SCHEDULER_TOKEN': 'fixture-scheduler-token-32-characters',
        'COLLABORATION_DEFAULT_SCHEDULER': 'legacy_multi_agent',
        'COLLABORATION_DISABLED_SCHEDULERS': '',
        'SUBJECTIVE_DEVAGENTS_TASK_WORKTREES': '1',
        'FIXTURE_KEY': 'fixture',
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    base = tmp_path / 'repo'
    base.mkdir()
    git(base, 'init', '-q')
    (base / 'value.txt').write_text('before\n')
    git(base, 'add', 'value.txt')
    git(base, '-c', 'user.email=fixture@example.invalid', '-c', 'user.name=fixture', 'commit', '-qm', 'base')

    db = Database(str(tmp_path / 'chat.sqlite'))
    repo = ChatRepository(db)
    journal = ChannelCoordination(db)
    channels = {repo.create_conversation('group', 'owner')['id']: name
                for name in ('simple_v1', 'legacy_multi_agent')}
    for channel, name in channels.items():
        repo.add_member(channel, 'owner', role='owner')
        repo.add_member(channel, 'ai_coder')
        journal.mutate(channel, 'owner', {'op': 'enable', 'enabled': True}, manager=True, human=True)
        journal.mutate(channel, 'owner', {'op': 'set_scheduler', 'scheduler_preference': name}, manager=True, human=True)

    export = {'version': {'version': '20261114000000'}, 'files': [
        {'path': 'AGENTS.md', 'content': 'Complete assigned work.'},
        {'path': 'agents/roles/coder_agent.md', 'content': 'Implement and test.'},
    ]}
    monkeypatch.setattr(agent_workflow, 'fetch_export_as_service', lambda *_args, **_kwargs: export)
    monkeypatch.setattr(scheduler_transport, '_request', lambda _route, body:
                        json.loads(json.dumps(plan(PlanRequest(**body), env['COORDINATION_SCHEDULER_TOKEN']))))

    def callback(session, operation, **data):
        try:
            payload = {
                **data, 'op': operation, '_runtime': True, 'dispatch_id': session.binding['dispatch_id'],
                'instance_id': session.binding['instance_id'],
            }
            if session.lease_id:
                payload.setdefault('lease_id', session.lease_id)
            result = journal.mutate(session.binding['channel_id'], session.actor, json.loads(json.dumps(payload)))
        except ApiError as error:
            raise ModelError(error.code, error.message) from None
        session._state = result['state']
        return result

    monkeypatch.setattr(ChannelSession, 'call', callback)
    model_name = os.environ['SCHEDULER_ACCEPTANCE_MODEL'] if live else 'fixture-model'
    identity = Identity('fixture', 'claude', model=model_name, strategy='api_agent',
                        workspace=str(base), home=str(tmp_path), sandbox='workspace-write',
                        allowed_tools=('Read', 'Write', 'Bash'),
                        api_base_url='https://api.openai.com/v1' if live else 'https://fixture.invalid',
                        api_key_env='OPENAI_API_KEY' if live else 'FIXTURE_KEY')
    agent = ApiAgent()

    async def model_response(identity, token, messages, tools, timeout, cancelled):
        assert identity.model == 'fixture-model'
        past = [message for message in messages if message['role'] == 'tool']
        if not past:
            assert 'Write' in tools
            name, arguments = 'Write', {'file_path': 'value.txt', 'content': 'after\n'}
        elif len(past) == 1:
            assert 'Wrote' in past[-1]['content'], past[-1]
            name, arguments = 'Bash', {'command': "test \"$(cat value.txt)\" = after"}
        else:
            assert 'exit code' not in past[-1]['content'], past[-1]
            return {'content': 'IMPLEMENTATION_COMPLETE: changed value and ran check.'}, 'stop', {}
        return {'tool_calls': [{'id': f'call-{len(past)}', 'type': 'function',
                                 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, 'tool_calls', {}

    if not live:
        monkeypatch.setattr(agent, '_call', model_response)
    store = TurnStore(str(tmp_path / 'runtime.sqlite'))
    model = LanguageModel(store, IdentityRegistry([identity]), {'api_agent:claude': agent})
    service = SimpleNamespace(backend=SimpleNamespace(model=model, identity=lambda *_args, **_kwargs: identity),
                              _default_workspace=lambda _provider: str(base), app_home=str(tmp_path / 'app'))

    class Hub:
        async def broadcast_room(self, *args):
            pass

    app = SimpleNamespace(state=SimpleNamespace(database=db, repo=repo, hub=Hub()))
    monkeypatch.setattr(collaboration, 'fetch_assignment', lambda hint:
                        approved_job(app, hint['binding']['channel_id'], hint['binding']['dispatch_id'], ready=True))

    def deliver(payload, _unused):
        return collaboration.execute(service, {'binding': {
            key: payload['binding'][key] for key in ('channel_id', 'dispatch_id')
        }})

    queue = ExecutionQueue(app, deliver=deliver)
    workspaces = set()
    try:
        for channel in channels:
            assignment = (
                'Edit value.txt to contain exactly after plus a newline, using Write. '
                'Run one Bash check: test "$(cat value.txt)" = after. '
                'Then report IMPLEMENTATION_COMPLETE and the check result. '
                'Work in the current directory and make no Git changes.'
            )
            if channels[channel] == 'simple_v1':
                assignment = '@Coder, please ' + assignment[0].lower() + assignment[1:]
            request = {'op': 'start_execution', 'submission_id': 'fixture-message', 'assignment': assignment,
                       'addressed_agents': ['ai_coder'], 'execution': {'prompt': assignment, 'agents': {
                           'ai_coder': {'user_id': 'ai_coder', 'company_id': 'company_test', 'provider': 'claude',
                                        'role_file': 'agents/roles/coder_agent.md', 'mode': 'write', 'display_name': 'Coder'},
                       }}}
            state = journal._load(channel)
            chosen = scheduler_transport.plan(state, request)
            journal.mutate(channel, 'owner', {**request, '_scheduler_plan': chosen}, human=True)
        asyncio.run(queue.tick())
        for _ in range(1800 if live else 200):
            states = {channel: journal.snapshot(channel) for channel in channels}
            if all(state['dispatches'] and all(item['status'] in {'COMPLETED', 'FAILED', 'BLOCKED'}
                                              for item in state['dispatches'].values()) for state in states.values()):
                break
            time.sleep(0.025)
        asyncio.run(queue.tick())
        for channel, state in states.items():
            items = list(state['dispatches'].values())
            assert len(items) == 1, [{key: item.get(key) for key in ('status', 'failure_code', 'failure_reason')}
                                     for item in items]
            item = items[0]
            assert item['status'] == 'COMPLETED', item
            assert item['scheduler'] == {'id': channels[channel], 'version': '1.0.0'}
            workspace = Path(state['task']['workspace_identity']['path'])
            workspaces.add(workspace)
            assert (workspace / 'value.txt').read_text() == 'after\n'
            with store.connection() as conn:
                row = conn.execute('SELECT id FROM turns WHERE request_id=?', (item['id'],)).fetchone()
            evidence = model.turn_evidence(row['id'])
            assert set(evidence['toolsUsed']) >= {'Bash', 'Write'}
            assert evidence['successfulToolCalls'] == evidence['toolCallsCount']
            events = store.events(row['id'])
            observed_diff = next(event for event in events if event['kind'] == 'file_diff')
            job = db.query_one('SELECT delivery_attempts,published FROM coordination_jobs WHERE dispatch_id=?',
                               (item['id'],))
            assert job['published'] == 1
            print(json.dumps({'scheduler': channels[channel], 'model': model_name,
                              'executions': len(state['dispatches']),
                              'tool_calls': evidence['toolCallsCount'], 'delivery_retries': job['delivery_attempts'] - 1,
                              'failure_count': 0, 'first_change_seconds': round(observed_diff['created'] - item['dispatched_at'], 3),
                              'duration_seconds': round(item['completed_at'] - item['started_at'], 3),
                              'result': 'file changed and check passed', 'publication': 'local chat message only'}))
        assert len(workspaces) == 2
        assert (base / 'value.txt').read_text() == 'before\n'
    finally:
        model.close()
        db.close()
