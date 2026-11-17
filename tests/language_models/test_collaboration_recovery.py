"""Cross-service acceptance: worker death, durable delivery and real worktree takeover."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'subjective_api_chat' / 'src'))
# Both repositories are needed only for this explicitly cross-service integration suite.
pytest.importorskip("subjective_chat_api", reason="Cross-service acceptance requires the companion Chat API checkout")
from subjective_chat_api.coordination import ChannelCoordination, ACTIVE_DISPATCH
from subjective_chat_api.db import Database
from subjective_chat_api.repository import ChatRepository
from subjective_chat_api.execution_queue import ExecutionQueue
from subjective_devagents_fastapi import collaboration, agent_workflow
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import LanguageModel, Identity, ModelResult, ModelStrategy
from subjective_devagents_fastapi.language_models.registry import IdentityRegistry
from subjective_devagents_fastapi.language_models.store import TurnStore
from subjective_devagents_fastapi.language_models.contracts import ModelError
from subjective_devagents_fastapi.workspace_fence import mutation_fence


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.setenv('COORDINATION_RUNTIME_URL', 'http://fixture.invalid')
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', 'existing-service-token')
    monkeypatch.setenv('COORDINATION_COMPANY_ID', 'company_test')
    monkeypatch.setenv('SUBJECTIVE_CHAT_HOME', str(tmp_path / 'home'))
    repo = tmp_path / 'repo'; repo.mkdir()
    git(repo, 'init', '-q')
    (repo / 'a.py').write_text('original\n')
    git(repo, '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', 'add', '.')
    git(repo, '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', 'commit', '-qm', 'base')
    clock = [time.time()]
    database = Database(str(tmp_path / 'chat.db'))
    repository = ChatRepository(database)
    channel = repository.create_conversation('group', 'owner')['id']
    repository.add_member(channel, 'owner', role='owner'); repository.add_member(channel, 'ai_coder')
    journal = ChannelCoordination(database, clock=lambda: clock[0])
    journal.mutate(channel, 'owner', {'op': 'enable', 'enabled': True}, manager=True, human=True)
    original_call = ChannelSession.call
    def local_call(self, op, **data):
        from subjective_chat_api.errors import ApiError
        try:
            result = journal.mutate(channel, self.actor, {**data, 'op': op, '_runtime': True,
                'dispatch_id': self.binding['dispatch_id'], 'instance_id': self.binding['instance_id'],
                **({'lease_id': self.lease_id} if self.lease_id and 'lease_id' not in data else {})})
        except ApiError as error:
            raise ModelError(error.code, error.message) from None
        self._state = result['state']
        if result.get('code') in {'LEASE_CONFLICT', 'WORKFLOW_VERSION_CONFLICT', 'WORKFLOW_VERSION_MISMATCH'}:
            raise ModelError(result['code'], 'Channel coordination rejected operation')
        return result
    monkeypatch.setattr(ChannelSession, 'call', local_call)
    identity = Identity('runtime', 'codex', model='fixture', strategy='api_agent', workspace=str(repo), home=str(tmp_path),
                        sandbox='workspace-write', api_base_url='http://fixture.invalid', api_key_env='FIXTURE_KEY')
    seen = []; finished = threading.Event()
    class Worker(ModelStrategy):
        def run(self, request, emit, cancelled):
            seen.append((request.coordination.binding.copy(), (repo / 'a.py').read_text()))
            session = request.coordination
            assert session.lease_id  # runtime startup recovered task-owned lease automatically
            with mutation_fence(session):
                session.guard('Write', {'file_path': 'a.py'})
                (repo / 'a.py').write_text('coder B continued A\n')
            finished.set()
            return ModelResult('completed', 'TASK COMPLETE: recovered and tested')
    model = LanguageModel(TurnStore(tmp_path / 'runtime.db'), IdentityRegistry([identity]), {'api_agent:codex': Worker()})
    service = SimpleNamespace(backend=SimpleNamespace(model=model, identity=lambda *a, **kw: identity),
        _default_workspace=lambda provider: str(repo), app_home=str(tmp_path / 'app'))
    export = {'version': {'version': '20261114000000'}, 'files': [
        {'path': 'AGENTS.md', 'content': 'Complete assigned work.'},
        {'path': 'agents/roles/coder_agent.md', 'content': 'Implement and test.'}]}
    monkeypatch.setattr(agent_workflow, 'fetch_export_as_service', lambda company, **_kwargs: export)
    execution = {'prompt': 'Implement TASK-1', 'agents': {'ai_coder': {'user_id': 'ai_coder', 'company_id': 'company_test',
        'provider': 'codex', 'role_file': 'agents/roles/coder_agent.md', 'display_name': 'Coder'}}}
    class Hub:
        async def broadcast_room(self, *args): pass
    app = SimpleNamespace(state=SimpleNamespace(database=database, repo=repository, hub=Hub()))
    from subjective_chat_api.service_auth import approved_job
    monkeypatch.setattr(collaboration, 'fetch_assignment', lambda hint: approved_job(app, hint['binding']['channel_id'], hint['binding']['dispatch_id'], ready=True))
    yield SimpleNamespace(**locals())
    model.close(); database.close()


def test_manager_dispatch_coder_dies_retry_preserves_worktree(system):
    s = system
    first = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'Implement TASK-1',
                                              'execution': s.execution})['dispatch']
    binding = ChannelCoordination.binding(s.journal.snapshot(s.channel), first)
    coder_a = ChannelSession(binding, 'ai_coder', '')
    coder_a.start(s.identity, '20261114000000', 'coder_agent')
    coder_a.tool('acquire', {'repository': str(s.repo), 'paths': ['.']})
    (s.repo / 'a.py').write_text('coder A unfinished\n')
    # Preserve a real interrupted operation marker as well as uncommitted bytes.
    git_dir = Path(git(s.repo, 'rev-parse', '--absolute-git-dir'))
    (git_dir / 'CHERRY_PICK_HEAD').write_text(git(s.repo, 'rev-parse', 'HEAD'))
    s.clock[0] += 301
    retried = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'retry',
                                                'execution': s.execution})
    successor = retried['dispatch']
    assert successor['id'] != first['id']
    queue = ExecutionQueue(s.app, deliver=lambda payload, signed: collaboration.execute(s.service, {'binding': {key: payload['binding'][key] for key in ('channel_id','dispatch_id')}}))
    asyncio.run(queue.tick())
    assert s.finished.wait(5), 'successor was not scheduled promptly'
    s.model.close()
    assert s.seen[0][1] == 'coder A unfinished\n'
    assert (s.repo / 'a.py').read_text() == 'coder B continued A\n'
    assert (git_dir / 'CHERRY_PICK_HEAD').exists()
    state = s.journal.snapshot(s.channel)
    assert state['dispatches'][first['id']]['status'] == 'ABANDONED'
    assert state['dispatches'][first['id']]['superseded_by'] == successor['id']
    inherited = [lease for lease in state['leases'].values() if lease.get('recovered_from')]
    assert inherited and inherited[0]['inherited_workspace']['repositories'][0]['operation_in_progress'] == ['CHERRY_PICK_HEAD']
    assert not [item for item in state['dispatches'].values() if item['status'] in ACTIVE_DISPATCH]
    assert all('LEASE_OWNER_MISMATCH' not in str(event) for event in state['events'])


def test_service_admission_rejects_caller_supplied_recipe(system):
    s = system
    payload = {'binding': {'channel_id': s.channel, 'dispatch_id': 'd'}, 'execution': s.execution}
    with pytest.raises(ModelError, match='references only'):
        collaboration.execute(s.service, payload)


def test_assignment_lookup_uses_existing_identity_and_fixed_origin(monkeypatch):
    import httpx
    hint = {'binding': {'channel_id': 'channel_fixture', 'dispatch_id': 'dispatch_fixture'}}
    monkeypatch.setenv('COORDINATION_CALLBACK_URL', 'https://configured-chat.example')
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', 'existing-service-fixture')
    seen = []
    def reply(request):
        seen.append(request)
        assert str(request.url) == 'https://configured-chat.example/internal/collaboration/channel_fixture/dispatch/dispatch_fixture'
        assert request.headers['X-Workflow-Service-Token'] == 'existing-service-fixture'
        assert 'Authorization' not in request.headers
        return httpx.Response(200, json={'ok': True, 'data': {'binding': hint['binding'], 'owner_id': 'approved-owner', 'execution': {'prompt': 'approved'}}})
    client = httpx.Client
    def factory(**kwargs):
        assert kwargs['follow_redirects'] is False and kwargs['trust_env'] is False
        return client(transport=httpx.MockTransport(reply), **kwargs)
    monkeypatch.setattr(httpx, 'Client', factory)
    assert collaboration.fetch_assignment(hint)['owner_id'] == 'approved-owner'
    assert len(seen) == 1


def test_assignment_lookup_never_follows_redirects(monkeypatch):
    import httpx
    monkeypatch.setenv('COORDINATION_CALLBACK_URL', 'https://configured-chat.example')
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', 'existing-service-fixture')
    client = httpx.Client
    seen = []
    def redirect(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={'location': 'https://untrusted.example/collect'}, json={})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(redirect), **kwargs))
    with pytest.raises(ModelError):
        collaboration.fetch_assignment({'binding': {'channel_id': 'channel', 'dispatch_id': 'dispatch'}})
    assert len(seen) == 1 and 'untrusted' not in seen[0]


def test_read_only_runtime_does_not_recover_writer_leases(system):
    s = system
    first = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'work'})['dispatch']
    a = ChannelSession(ChannelCoordination.binding(s.journal.snapshot(s.channel), first), 'ai_coder', '')
    a.start(s.identity, '20261114000000'); a.tool('acquire', {'repository': str(s.repo), 'paths': ['.']})
    a.finish('failed')
    successor = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'retry'})['dispatch']
    b = ChannelSession(ChannelCoordination.binding(s.journal.snapshot(s.channel), successor), 'ai_coder', '')
    b.start(replace(s.identity, sandbox='read-only'), '20261114000000')
    assert b.lease_id is None
    with pytest.raises(ModelError, match='Read-only'):
        b.tool('recover_task_lease', {})


def test_never_started_dispatch_automatically_runs_one_replacement(system):
    s = system
    completed = threading.Event()
    count = []
    class Noop(ModelStrategy):
        def run(self, request, emit, cancelled):
            count.append(request.coordination.binding['dispatch_id'])
            completed.set()
            return ModelResult('completed', 'TASK COMPLETE: replacement ran')
    s.model.strategies['api_agent:codex'] = Noop()
    first = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'work',
                                               'execution': s.execution})['dispatch']
    s.clock[0] += 121
    replacement = s.journal.mutate(s.channel, 'watchdog', {'op': 'reconcile'})['successors'][0]
    asyncio.run(ExecutionQueue(s.app, deliver=lambda payload, signed: collaboration.execute(s.service, {'binding': {key: payload['binding'][key] for key in ('channel_id','dispatch_id')}})).tick())
    assert completed.wait(5)
    s.model.close()
    assert count == [replacement['dispatch_id']]
    assert s.journal.snapshot(s.channel)['dispatches'][first['id']]['status'] == 'ABANDONED'


def test_failed_result_callback_is_durable_and_replayed(system, monkeypatch):
    s = system
    class Noop(ModelStrategy):
        def run(self, request, emit, cancelled):
            return ModelResult('completed', 'TASK COMPLETE: durable result')
    s.model.strategies['api_agent:codex'] = Noop()
    first = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'work',
                                               'execution': s.execution})['dispatch']
    local_call = ChannelSession.call
    failed = []
    def temporary_failure(self, op, **data):
        if op == 'finish' and not failed:
            failed.append(True)
            raise ModelError('coordination_unavailable', 'temporary outage')
        return local_call(self, op, **data)
    monkeypatch.setattr(ChannelSession, 'call', temporary_failure)
    asyncio.run(ExecutionQueue(s.app, deliver=lambda payload, signed: collaboration.execute(s.service, {'binding': {key: payload['binding'][key] for key in ('channel_id','dispatch_id')}})).tick())
    s.model.close()
    with s.model.store.connection() as db:
        assert db.execute('SELECT count(*) FROM coordination_completions').fetchone()[0] == 1
    assert s.journal.snapshot(s.channel)['dispatches'][first['id']]['status'] == 'STARTED'
    collaboration.replay_completions(s.model.store)
    assert s.journal.snapshot(s.channel)['dispatches'][first['id']]['status'] == 'COMPLETED'
    with s.model.store.connection() as db:
        assert db.execute('SELECT count(*) FROM coordination_completions').fetchone()[0] == 0


def test_late_worker_cannot_adopt_different_assignment(system):
    s = system
    a = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'old'})['dispatch']
    ids = {'dispatch_id': a['id'], 'instance_id': a['instance_id']}
    s.journal.mutate(s.channel, 'ai_coder', {'op': 'start', **ids, 'runtime': 'r', 'workflow_version': 'wf', 'heartbeat_interval': 15})
    queued = s.journal.mutate(s.channel, 'owner', {'op': 'dispatch', 'agent_id': 'ai_coder', 'assignment': 'new instructions'})['dispatch']
    s.clock[0] += 301
    s.journal.mutate(s.channel, 'watchdog', {'op': 'reconcile'})
    with pytest.raises(Exception, match='claimed or revoked'):
        s.journal.mutate(s.channel, 'ai_coder', {'op': 'start', **ids, 'runtime': 'r', 'workflow_version': 'wf'})
    assert s.journal.snapshot(s.channel)['dispatches'][queued['id']]['status'] == 'DISPATCHED'


def test_turn_finish_and_completion_outbox_are_one_transaction(system):
    s = system
    conversation = s.model.store.create(s.identity)
    turn, _ = s.model.store.reserve(conversation['id'], s.identity, 'atomic', 'work', 30)
    assert s.model.store.claim(turn['id'])
    payload = {'binding': {'dispatch_id': 'atomic-dispatch'}, 'data': {'result': {'text': 'completed'}}}
    with s.model.store.connection() as db:
        db.execute("CREATE TRIGGER reject_callback BEFORE INSERT ON coordination_completions BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(Exception, match='fixture'):
        s.model.store.finish(turn['id'], ModelResult('completed', 'done'), completion=payload)
    assert s.model.store.turn(turn['id'])['status'] == 'running'
    with s.model.store.connection() as db:
        db.execute('DROP TRIGGER reject_callback')
    s.model.store.finish(turn['id'], ModelResult('completed', 'done'), completion=payload)
    assert s.model.store.turn(turn['id'])['status'] == 'completed'
    with s.model.store.connection() as db:
        assert db.execute('SELECT dispatch_id FROM coordination_completions').fetchone()[0] == 'atomic-dispatch'


def test_durable_images_are_owner_checked_and_survive_reload(system):
    import base64
    from subjective_devagents_fastapi.language_models.legacy import LegacyLanguageModel
    s = system
    s.service.backend.attachments = LegacyLanguageModel.attachments
    files = [{'filename': 'fixture.png', 'mime_type': 'image/png', 'content': base64.b64encode(b'fixture-image').decode()}]
    reference = collaboration.save_attachments(s.service, files, 'owner')
    assert collaboration.load_attachments(s.service, reference, 'owner')[0].data == b'fixture-image'
    with pytest.raises(ModelError, match='another requester'):
        collaboration.load_attachments(s.service, reference, 'another')
    with pytest.raises(ModelError):
        collaboration.load_attachments(s.service, '../runtime.db', 'owner')


def test_legacy_admission_http_payload_and_finish_after_login_expires(tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from subjective_chat_api.main import create_app
    from subjective_chat_api import service_auth
    monkeypatch.setenv('COORDINATION_COMPANY_ID', 'company_test')
    monkeypatch.setenv('COORDINATION_CALLBACK_URL', 'https://chat.fixture')
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', 'existing-service-token-fixture')
    monkeypatch.setenv('SUBJECTIVE_CHAT_ENVIRONMENT', 'development')
    monkeypatch.setenv('SUBJECTIVE_CHAT_ALLOW_DEV_AUTH', '1')
    monkeypatch.setenv('SUBJECTIVE_CHAT_DB_PATH', str(tmp_path/'admission.sqlite3'))
    monkeypatch.setattr(service_auth, 'verify_service_identity', lambda token: None if token == 'existing-service-token-fixture' else pytest.fail('wrong service identity'))
    with TestClient(create_app()) as chat:
        user = {'x-dev-user-id': 'owner'}
        channel = chat.post('/chat/conversations', headers=user, json={'type':'group','name':'Integration recovery','member_ids':['ai_integration']}).json()['data']['conversation']['id']
        route = f'/chat/conversations/{channel}/coordination'
        response = chat.post(route, headers=user, json={'op':'dispatch','agent_id':'ai_integration','assignment':'Integration check'}).json()['data']
        binding = ChannelCoordination.binding(response['state'], response['dispatch'])
        session = ChannelSession(binding, 'ai_integration', 'Bearer initial-browser-fixture')
        seen = []
        class Bridge:
            def __init__(self, **kwargs):
                assert kwargs['follow_redirects'] is False and kwargs['trust_env'] is False
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def post(self, url, *, headers, json):
                seen.append((url, dict(headers), dict(json)))
                if url.endswith('/admit'):
                    assert headers['Authorization'] == 'Bearer initial-browser-fixture'
                    headers = {**headers, **user}
                else:
                    assert 'Authorization' not in headers
                return chat.post(url.removeprefix('https://chat.fixture'), headers=headers, json=json)
        monkeypatch.setattr(httpx, 'Client', Bridge)
        session.authorize_service_callbacks('company_test')
        assert seen[0][2]['agent_id'] == 'ai_integration'
        assert session._authorization == '' and session.delegation['owner_id'] == 'owner'
        session.start(Identity('runtime', 'codex', model='fixture', strategy='api_agent', workspace=str(tmp_path),
            api_base_url='http://fixture.invalid', api_key_env='FIXTURE_KEY'), 'v', 'coder_agent')
        session.call('touch')
        session.finish('completed')
        state = chat.get(route, headers=user).json()['data']
        assert state['dispatches'][binding['dispatch_id']]['status'] == 'COMPLETED'


@pytest.mark.parametrize('base_id', ['runtime', 'x' * 80])
def test_server_dispatch_runs_read_only_role_with_valid_stable_identity(system, base_id):
    import re
    s = system
    base_identity = replace(s.identity, id=base_id)
    s.model.identities.register_legacy(base_identity)
    s.service.backend.identity = lambda *_args, **_kwargs: base_identity
    s.execution['agents']['ai_coder']['mode'] = 'read'
    seen = []
    class Observer(ModelStrategy):
        def run(self, request, emit, cancelled):
            seen.append(request.identity)
            assert request.identity.sandbox == 'read-only'
            assert request.coordination.lease_id is None
            with pytest.raises(ModelError): request.coordination.guard('Write', {'file_path': 'blocked.txt'})
            return ModelResult('completed', 'TASK COMPLETE: read-only verification')
    s.model.strategies['api_agent:codex'] = Observer()
    item = s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'Verify', 'execution':s.execution})['dispatch']
    asyncio.run(ExecutionQueue(s.app, deliver=lambda payload, _unused: collaboration.execute(s.service,
        {'binding':{key:payload['binding'][key] for key in ('channel_id','dispatch_id')}})).tick())
    s.model.close()
    assert len(seen) == 1 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', seen[0].id)
    assert s.model.identities.get(base_id).sandbox == 'workspace-write'
    assert s.journal.snapshot(s.channel)['dispatches'][item['id']]['status'] == 'COMPLETED'
    assert (s.repo/'a.py').read_text() == 'original\n'


def test_server_dispatch_hot_reloads_the_current_published_workflow(system, monkeypatch, caplog):
    """Production 2026-11-17: agents kept reporting [WF:20261114180055] after newer versions were
    published, because the channel pinned its first version. Every dispatch now loads what is
    published NOW, records it, and logs WORKFLOW_RELOADED."""
    s = system
    original = s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'prior stage'})['dispatch']
    actor = ChannelSession(ChannelCoordination.binding(s.journal.snapshot(s.channel), original), 'ai_coder', '')
    actor.start(s.identity, s.export['version']['version'], 'coder_agent')
    actor.finish('completed')
    assert s.journal.snapshot(s.channel)['workflow_version'] == '20261114000000'
    newer = {**s.export, 'version': {'version': '20261115000000'}}
    requested = []
    def published(company, **options):
        requested.append(options)
        return newer
    monkeypatch.setattr(agent_workflow, 'fetch_export_as_service', published)
    class Observer(ModelStrategy):
        def run(self, request, emit, cancelled):
            assert '[WF:20261115000000]' in request.prompt
            return ModelResult('completed', 'TASK COMPLETE: current workflow honored')
    s.model.strategies['api_agent:codex'] = Observer()
    item = s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'next stage','execution':s.execution})['dispatch']
    caplog.set_level('INFO', logger='subjective_devagents.workflow')
    asyncio.run(ExecutionQueue(s.app, deliver=lambda payload, _unused: collaboration.execute(s.service,
        {'binding':{key:payload['binding'][key] for key in ('channel_id','dispatch_id')}})).tick())
    s.model.close()
    assert len(requested) == 1 and requested[0].get('force') is True and 'version' not in requested[0]  # never pinned
    state = s.journal.snapshot(s.channel)
    dispatch = state['dispatches'][item['id']]
    assert dispatch['status'] == 'COMPLETED'
    assert dispatch['requested_workflow_version'] == '20261114000000'
    assert dispatch['loaded_workflow_version'] == '20261115000000'
    assert dispatch['workflow_source'] == 'published_workflow_database' and len(dispatch['workflow_digest']) == 64
    assert state['workflow_version'] == '20261115000000'
    assert any(e['kind'] == 'WORKFLOW_RELOADED' for e in state['events'])
    assert 'workflow.WORKFLOW_RELOADED' in caplog.text


def test_stale_published_workflow_is_refused_never_run(system, monkeypatch):
    """The runtime can only load a version older than the channel already ran: refuse, do not run."""
    s = system
    with s.database.lock:
        state = s.journal._load(s.channel)
        state['workflow_version'] = '20261117142005'
        s.database.execute('UPDATE channel_coordination SET payload=? WHERE channel_id=?', (json.dumps(state), s.channel))
    stale = {**s.export, 'version': {'version': '20261114180055'}}
    monkeypatch.setattr(agent_workflow, 'fetch_export_as_service', lambda company, **_options: stale)
    ran = []
    class Observer(ModelStrategy):
        def run(self, request, emit, cancelled):
            ran.append(request.prompt)
            return ModelResult('completed', 'should never run')
    s.model.strategies['api_agent:codex'] = Observer()
    item = s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'work','execution':s.execution})['dispatch']
    asyncio.run(ExecutionQueue(s.app, deliver=lambda payload, _unused: collaboration.execute(s.service,
        {'binding':{key:payload['binding'][key] for key in ('channel_id','dispatch_id')}})).tick())
    s.model.close()
    assert not ran
    row = s.database.query_one('SELECT result FROM coordination_jobs WHERE dispatch_id=?', (item['id'],))
    assert row is None or row['result'] is None or 'WORKFLOW_VERSION_MISMATCH' in row['result']
    dispatch = s.journal.snapshot(s.channel)['dispatches'][item['id']]
    assert dispatch['status'] != 'COMPLETED' and dispatch.get('loaded_workflow_version') is None


def test_worker_start_failure_has_durable_visible_result(system):
    s = system
    s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'old'})
    # An incompatible channel assignment must still report a failure; never fake start.
    with s.database.lock:
        state = s.journal._load(s.channel)
        state['workflow_version'] = '20991231000000'
        s.database.execute('UPDATE channel_coordination SET payload=? WHERE channel_id=?', (json.dumps(state), s.channel))
    item = s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'different', 'execution':s.execution})['dispatch']
    # Cancel the unrelated earlier reservation so this dispatch becomes ready.
    previous = item.get('queued_behind')
    if previous: s.journal.mutate(s.channel, 'owner', {'op':'cancel_pending','dispatch_id':previous}, human=True, manager=True)
    def over_http(payload, _unused):
        # As POST /api/collaboration/execute answers: a ModelError is a 409 carrying its code.
        try:
            return collaboration.execute(s.service, {'binding': {key: payload['binding'][key] for key in ('channel_id', 'dispatch_id')}})
        except ModelError as error:
            return {'ok': False, 'error': error.code, 'message': str(error)[:400]}
    queue = ExecutionQueue(s.app, deliver=over_http)
    asyncio.run(queue.tick())
    s.model.close()
    dispatch = s.journal.snapshot(s.channel)['dispatches'][item['id']]
    # Refused before any model call, with the explicit code; recovery decides the continuation.
    assert dispatch['status'] == 'FAILED' and dispatch['failure_code'] == 'WORKFLOW_VERSION_MISMATCH'
    assert dispatch['started_at'] is None
    with s.model.store.connection() as db: assert db.execute('SELECT count(*) FROM coordination_completions').fetchone()[0] == 0
    assert any(e['kind'] == 'WORKFLOW_VERSION_MISMATCH' for e in s.journal.snapshot(s.channel)['events'])


def test_conflicting_prestart_failure_cannot_finish_later_healthy_winner(system):
    s = system
    item = s.journal.mutate(s.channel, 'owner', {'op':'dispatch','agent_id':'ai_coder','assignment':'work','execution':s.execution})['dispatch']
    binding = ChannelCoordination.binding(s.journal.snapshot(s.channel), item)
    with s.database.lock:
        state = s.journal._load(s.channel)
        state['workflow_version'] = '20261115000000'
        s.database.execute('UPDATE channel_coordination SET payload=? WHERE channel_id=?', (json.dumps(state), s.channel))
    loser = ChannelSession(binding, 'ai_coder', '')
    loser.delegation = {**binding,'owner_id':'owner'}
    loser.completion_store = s.model.store
    loser.defer_start(s.identity, '20261114000000', 'coder_agent')       # a stale load
    with pytest.raises(ModelError) as conflict: loser.worker_started()
    assert conflict.value.code == 'WORKFLOW_VERSION_MISMATCH' and not loser.start_acknowledged
    winner = ChannelSession(binding, 'ai_coder', '')
    winner.start(s.identity, '20261115000000', 'coder_agent')
    winner.tool('acquire', {'repository':str(s.repo),'paths':['.']})
    with pytest.raises(ModelError) as rejected: loser.finish('failed', 'WORKFLOW_VERSION_MISMATCH')
    assert rejected.value.code == 'DISPATCH_ALREADY_STARTED'
    collaboration.replay_completions(s.model.store)
    state = s.journal.snapshot(s.channel)
    assert state['dispatches'][item['id']]['status'] in {'STARTED','RUNNING'}
    assert state['leases'][winner.lease_id]['status'] == 'active'
    with s.model.store.connection() as db:
        assert db.execute('SELECT count(*) FROM coordination_completions').fetchone()[0] == 0
