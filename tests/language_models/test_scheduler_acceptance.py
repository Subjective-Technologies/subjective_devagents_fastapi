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
from subjective_chat_api.execution_queue import ExecutionQueue, authorize_execution
from subjective_chat_api.repository import ChatRepository
from subjective_chat_api.service_auth import approved_job
from subjective_devagents_fastapi import agent_workflow, collaboration
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import Identity, LanguageModel
from subjective_devagents_fastapi.language_models.contracts import ModelError, ModelResult
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
@pytest.mark.parametrize('scenario', ['edit', 'clarify', 'publish', 'provider_failure', 'blocked'])
def test_two_channels_use_same_executor_and_distinct_worktrees(tmp_path, monkeypatch, live, scenario):
    if live and scenario != 'edit':
        pytest.skip('The live provider smoke covers the edit scenario')
    env = {
        'COORDINATION_RUNTIME_URL': 'https://fixture.invalid',
        'COORDINATION_COMPANY_ID': 'company_test',
        'COLLABORATION_DEFAULT_SCHEDULER': 'legacy_multi_agent',
        'COLLABORATION_DISABLED_SCHEDULERS': '',
        'SUBJECTIVE_DEVAGENTS_TASK_WORKTREES': '1',
        'FIXTURE_KEY': 'fixture',
        'COLLABORATION_CAPABILITY_KEY': 'fixture-capability-key-at-least-32-characters',
        'SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN': 'fixture-workflow-token',
        'COLLABORATION_DIRECT_AGENT_GRANTS': json.dumps({'ai_coder':
            ['fs_read', 'fs_write', 'shell_exec', 'test_exec', 'git_read', 'git_write', 'network_access']}),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    base = tmp_path / 'repo'
    base.mkdir()
    git(base, 'init', '-q')
    (base / 'value.txt').write_text('before\n')
    git(base, 'add', 'value.txt')
    git(base, '-c', 'user.email=fixture@example.invalid', '-c', 'user.name=fixture', 'commit', '-qm', 'base')

    remote = tmp_path / 'remote.git'
    if scenario == 'publish':
        git(tmp_path, 'init', '-q', '--bare', str(remote))
        git(base, 'remote', 'add', 'origin', str(remote))
        git(base, 'push', '-q', 'origin', 'HEAD:refs/heads/main')

    db = Database(str(tmp_path / 'chat.sqlite'))
    repo = ChatRepository(db)
    journal = ChannelCoordination(db)
    channels = {repo.create_conversation('group', 'owner')['id']: name
                for name in ('simple_v1', 'legacy_multi_agent')}
    for channel, name in channels.items():
        repo.add_member(channel, 'owner', role='owner')
        repo.add_member(channel, 'ai_coder')
        repo.add_member(channel, 'ai_manager')
        journal.mutate(channel, 'owner', {'op': 'enable', 'enabled': True}, manager=True, human=True)
        journal.mutate(channel, 'owner', {'op': 'set_scheduler', 'scheduler_preference': name}, manager=True, human=True)

    export = {'version': {'version': '20261114000000'}, 'files': [
        {'path': 'AGENTS.md', 'content': '## Orchestration\nAsk Manager to delegate.\n## Security\nKeep credentials private.'},
        {'path': 'agents/roles/coder_agent.md', 'content': 'Implement and test.'},
    ]}
    monkeypatch.setattr(agent_workflow, 'fetch_export_as_service', lambda *_args, **_kwargs: export)
    monkeypatch.setattr(scheduler_transport, '_request', lambda _route, body, *, authorization, **_options:
                        json.loads(json.dumps(plan(PlanRequest(**body)))))

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
                        allowed_tools=('Read', 'Write', 'Bash', 'AskUser'),
                        api_base_url='https://api.openai.com/v1' if live else 'https://fixture.invalid',
                        api_key_env='OPENAI_API_KEY' if live else 'FIXTURE_KEY')
    agent = ApiAgent()

    async def model_response(identity, token, messages, tools, timeout, cancelled):
        assert identity.model == 'fixture-model'
        direct = 'The runtime never publishes' in messages[0]['content']
        if direct:
            assert 'Keep credentials private.' in messages[0]['content']
            assert 'Ask Manager to delegate.' not in messages[0]['content']
            assert 'Coder rule:' not in messages[0]['content']
            assert '## Your role' not in str(messages)
            assert 'Implement and test.' not in str(messages)
            assert 'rebases committed work' not in messages[0]['content']
            assert {'Read', 'Write', 'Bash'} <= tools.keys()
        past = [message for message in messages if message['role'] == 'tool']
        question_answer = direct and 'Answer: leave the edit as is' in messages[0]['content']
        if question_answer:
            if not past:
                return tool('Read', {'file_path': 'value.txt'}, 0)
            assert 'after' in past[-1]['content']
            return {'content': 'Kept the original edit.'}, 'stop', {}
        if scenario == 'provider_failure' and direct:
            return ModelResult('failed', 'Fixture provider stopped accepting requests.', error_code='provider_error')
        reads = 14 if direct else 0
        if len(past) < reads:
            name, arguments = 'Read', {'file_path': 'value.txt'}
        elif len(past) == reads:
            assert 'Write' in tools
            name, arguments = 'Write', {'file_path': 'value.txt', 'content': 'after\n'}
        elif len(past) == reads + 1:
            assert 'Wrote' in past[-1]['content'], past[-1]
            name, arguments = 'Bash', {'command': "test \"$(cat value.txt)\" = after"}
        elif direct and scenario == 'blocked':
            name, arguments = 'Coordination', {'operation': 'blocker', 'data': {'reason': 'Required fixture input is absent.'}}
        elif direct and scenario == 'clarify':
            name, arguments = 'AskUser', {'question': 'Should I keep the edited value?'}
        elif direct and scenario == 'publish' and len(past) < reads + 5:
            commands = [
                'git add value.txt',
                'git -c user.email=fixture@example.invalid -c user.name=fixture commit -m direct-edit',
                'git push origin HEAD:refs/heads/main',
            ]
            assert 'exit code' not in past[-1]['content'] and not past[-1]['content'].startswith('Error:'), past[-1]
            name, arguments = 'Bash', {'command': commands[len(past) - reads - 2]}
        else:
            assert 'exit code' not in past[-1]['content'], past[-1]
            return {'content': ('Changed value and ran check. @Manager please continue' if direct else
                                'IMPLEMENTATION_COMPLETE: changed value and ran check.')}, 'stop', {}
        return tool(name, arguments, len(past))

    def tool(name, arguments, index):
        return {'tool_calls': [{'id': f'call-{index}', 'type': 'function',
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

    def authorized(agent, role):
        directory = {'ok': True, 'user': {'user_id': agent, 'display_name': 'Manager' if agent == 'ai_manager' else 'Claude'},
                     'profile': {'workflow_company_id': 'company_test', 'role_file': role,
                                 'default_provider_id': 'claude', 'default_model_id': '' if scenario == 'clarify' else model_name}}
        class Reply:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                pass
            def read(self):
                return json.dumps(self.body).encode()
        def opener(request, **_kwargs):
            reply = Reply()
            reply.body = export if request.full_url.endswith('/workflow/export') else directory
            return reply
        return authorize_execution({'agents': {agent: {}}, 'prompt': 'placeholder'},
                                   'Bearer fixture', opener=opener, compatibility_manager=False)['agents'][agent]
    requests = {}
    try:
        for channel in channels:
            assignment = (
                'Edit value.txt to contain exactly after plus a newline, using Write. '
                'Run one Bash check: test "$(cat value.txt)" = after. '
                'Then report IMPLEMENTATION_COMPLETE and the check result. '
                'Work in the current directory and make no Git changes.'
            )
            if channels[channel] == 'simple_v1':
                assignment = 'Claude, please ' + assignment[0].lower() + assignment[1:]
                assignment = assignment.replace('IMPLEMENTATION_COMPLETE and the check result', 'the change and the check result')
            direct = channels[channel] == 'simple_v1'
            profile = authorized('ai_coder', None) if direct else {
                'user_id': 'ai_coder', 'company_id': 'company_test', 'provider': 'claude',
                'role_file': 'agents/roles/coder_agent.md', 'mode': 'write', 'display_name': 'Coder',
            }
            if direct:
                assert profile['role_file'] == '' and profile['mode'] == 'write'
                assert profile['model'] == ('' if scenario == 'clarify' else model_name)
            roster = {'ai_coder': profile}
            if direct:
                roster['ai_manager'] = authorized('ai_manager', None)
            request = {'op': 'start_execution', 'submission_id': 'fixture-message', 'assignment': assignment,
                       'addressed_agents': [] if direct else ['ai_coder'], 'execution': {'prompt': assignment, 'agents': roster}}
            if direct and scenario == 'publish':
                assignment = assignment.replace('make no Git changes', 'commit and push the change to main')
                request['assignment'] = request['execution']['prompt'] = assignment
            requests[channel] = request
            state = journal._load(channel)
            chosen = scheduler_transport.plan(state, request, authorization='Bearer fixture')
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
            direct = channels[channel] == 'simple_v1'
            expected = {'clarify': 'BLOCKED', 'blocked': 'BLOCKED', 'provider_failure': 'FAILED'}.get(scenario, 'COMPLETED') if direct else 'COMPLETED'
            assert item['status'] == expected, item
            if expected == 'FAILED':
                assert state['task']['scheduler_outcome'] == 'failed'
                continue
            assert item['scheduler'] == {'id': channels[channel], 'version': '1.0.0'}
            workspace = Path(state['task']['workspace_identity']['path'])
            workspaces.add(workspace)
            assert (workspace / 'value.txt').read_text() == 'after\n'
            with store.connection() as conn:
                row = conn.execute('SELECT id FROM turns WHERE request_id=?', (item['id'],)).fetchone()
            evidence = model.turn_evidence(row['id'])
            assert set(evidence['toolsUsed']) >= {'Bash', 'Write'}
            if expected == 'COMPLETED':
                assert evidence['successfulToolCalls'] == evidence['toolCallsCount']
            events = store.events(row['id'])
            observed_diff = next(event for event in events if event['kind'] == 'file_diff')
            job = db.query_one('SELECT delivery_attempts,published,result FROM coordination_jobs WHERE dispatch_id=?',
                               (item['id'],))
            assert job['published'] == 1
            if channels[channel] == 'simple_v1':
                text = json.loads(job['result'])['text']
                assert 'Tests: passed' in text
                assert 'Deployment: no verified deployment recorded' in text
                if scenario == 'publish':
                    assert git(remote, 'show', 'main:value.txt').strip() == 'after'
                    assert 'Push: origin/main at ' in text
                    assert 'Commit: no commit observed' not in text
                else:
                    assert 'Commit: no commit observed' in text
                    assert 'Push: no verified push observed' in text
                if scenario == 'edit' and not live:
                    assert evidence['toolCallsCount'] == 16
                    assert '@Manager please continue' in text
                if scenario == 'blocked':
                    assert state['task']['scheduler_outcome'] == 'blocked'
                    assert 'Required fixture input is absent.' in text
                if scenario == 'clarify':
                    assert state['task']['scheduler_outcome'] == 'needs_input'
                    question, = state['decisions'].values()
                    journal.mutate(channel, 'owner', {'op': 'set_scheduler', 'scheduler_preference': 'legacy_multi_agent'},
                                   manager=True, human=True)
                    request = {**requests[channel], 'assignment': 'Keep the edit.', 'submission_id': 'answer',
                               'addressed_agents': ['ai_manager']}
                    plan_result = scheduler_transport.plan(journal._load(channel), request, authorization='Bearer fixture')
                    assert plan_result['targets'] == ['ai_coder']
                    pending = journal.mutate(channel, 'owner', {**request, '_scheduler_plan': plan_result}, human=True)
                    assert pending['code'] == 'SCHEDULER_INPUT_PENDING'
                    journal.mutate(channel, 'owner', {'op': 'resolve_decision', 'id': question['id'],
                                   'status': 'RESOLVED', 'answer': 'leave the edit as is'}, manager=True, human=True)
                    continued = journal.mutate(channel, 'owner', {**request, '_scheduler_plan': plan_result}, human=True)['dispatch']
                    asyncio.run(queue.tick())
                    for _ in range(200):
                        current = journal.snapshot(channel)
                        if current['dispatches'][continued['id']]['status'] == 'COMPLETED':
                            break
                        time.sleep(0.025)
                    assert current['task']['scheduler_outcome'] == 'completed'
                    assert current['task']['workspace_identity']['path'] == str(workspace)
                    assert current['task']['scheduler_id'] == 'simple_v1'
                    assert current['task']['scheduler_runtime_model'] == {'provider': 'claude', 'model': model_name}
                    assert continued['task_revision'] == item['task_revision']
            print(json.dumps({'scheduler': channels[channel], 'model': model_name,
                              'executions': len(state['dispatches']),
                              'tool_calls': evidence['toolCallsCount'], 'delivery_retries': job['delivery_attempts'] - 1,
                              'failure_count': 0, 'first_change_seconds': round(observed_diff['created'] - item['dispatched_at'], 3),
                              'duration_seconds': round(item['completed_at'] - item['started_at'], 3),
                              'result': 'file changed and check passed', 'publication':
                              'verified push to fixture Git remote' if direct and scenario == 'publish' else 'chat reply only'}))
        assert len(workspaces) == (1 if scenario == 'provider_failure' else 2)
        assert (base / 'value.txt').read_text() == 'before\n'
    finally:
        model.close()
        db.close()
