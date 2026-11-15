"""HTTP admission and worker lifecycle share one channel dispatch identity."""
import time
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models.contracts import ModelError, ModelResult, ModelStrategy


def test_channel_binding_claim_worker_context_and_no_omission_bypass(client, monkeypatch):
    from subjective_devagents_fastapi.app import core
    from subjective_devagents_fastapi.language_models import legacy
    monkeypatch.setenv(legacy.PROXY_URL_ENV,'http://unused/v1')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['codex'],'fixture-model')
    state={'enabled':True,'workflow_id':'workflow','workflow_version':None,'task_family':'TASK','task_revision':'1','revision':1,
           **{key:{} for key in ('facts','leases','decisions','capabilities','reviews','gates','worktrees','discoveries')}}
    calls=[];claim={'status':'pending'};requests=[]
    def call(self, op, **data):
        calls.append((op,data))
        if op=='start':
            if claim['status']!='pending': raise ModelError('DISPATCH_SUPPRESSED','Already claimed')
            claim['status']='running';state['workflow_version']=data['workflow_version']
        if op=='finish':claim['status']=data['status'];self.closed=True
        self._state=state
        return {'state':state}
    monkeypatch.setattr(ChannelSession,'call',call)
    class Recorder(ModelStrategy):
        def run(self, request, emit, cancelled):
            requests.append(request)
            return ModelResult('completed','[WF:UNKNOWN][agent:wrong_role]\nDone.')
    core().llm_service.backend.model.strategies['api_agent:codex']=Recorder()
    created=client.post('/api/llm/create-conversation',json={'provider':'codex','name':'channel fixture'}).json()
    target=created['conversation']['target']
    binding={'channel_id':'channel','workflow_id':'workflow','dispatch_id':'dispatch','instance_id':'ai_coder:instance','agent_id':'ai_coder'}
    response=client.post('/api/llm/send-message',json={'target':target,'text':'work','coordination':binding})
    assert response.status_code==200 and response.json()['pending']
    for _ in range(100):
        conversation=client.get('/api/llm/conversation',params={'target':target}).json()['conversation']
        answers=[m for m in conversation['messages'] if m['role']=='assistant']
        if answers:break
        time.sleep(.01)
    assert answers[0]['text']=='[WF:inline][agent:ai_coder]\nDone.'
    assert requests[0].coordination.binding['instance_id']=='ai_coder:instance'
    assert 'CHANNEL RUNTIME CONTEXT' not in requests[0].prompt # no stale snapshots persisted in user text
    assert any(op=='gate' and data['kind']=='WORKFLOW_VERSION_DIAGNOSTIC' for op,data in calls)
    assert claim['status']=='completed'
    assert client.post('/api/llm/send-message',json={'target':target,'text':'bypass'}).json()['error']=='channel_dispatch_required'
    duplicate=client.post('/api/llm/send-message',json={'target':target,'text':'duplicate','coordination':binding})
    assert duplicate.status_code==409 and duplicate.json()['error']=='DISPATCH_SUPPRESSED'


def test_busy_refused_before_dispatch_start_and_role_header_expected(client, monkeypatch):
    from subjective_devagents_fastapi.app import core
    from subjective_devagents_fastapi.language_models import legacy
    monkeypatch.setenv(legacy.PROXY_URL_ENV,'http://unused/v1')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['codex'],'fixture-model')
    state={'enabled':True,'workflow_id':'workflow','workflow_version':None,'task_family':'TASK','task_revision':'1','revision':1,
           **{key:{} for key in ('facts','leases','decisions','capabilities','reviews','gates','worktrees','discoveries')}}
    calls=[]
    def call(self, op, **data):
        calls.append((op,data))
        if op=='finish':self.closed=True
        self._state=state
        return {'state':state}
    monkeypatch.setattr(ChannelSession,'call',call)
    class Recorder(ModelStrategy):
        def run(self, request, emit, cancelled):
            return ModelResult('completed','[WF:7][agent:coder_agent]\nDone.')
    service=core().llm_service
    service.backend.model.strategies['api_agent:codex']=Recorder()
    target=client.post('/api/llm/create-conversation',json={'provider':'codex','name':'busy fixture'}).json()['conversation']['target']
    binding={'channel_id':'channel','workflow_id':'workflow','dispatch_id':'dispatch','instance_id':'ai_coder:instance','agent_id':'ai_coder'}
    service.busy.add(target)
    try:
        refused=client.post('/api/llm/send-message',json={'target':target,'text':'work','coordination':binding})
    finally:
        service.busy.discard(target)
    assert refused.status_code==409 and refused.json()['status']=='busy'
    assert 'already processing' in refused.json()['error']
    assert not any(op=='start' for op,_ in calls)  # the dispatch is still unspent
    service.agent_turns.setdefault(target,{})['header']=('7','coder_agent')
    accepted=client.post('/api/llm/send-message',json={'target':target,'text':'work','coordination':binding})
    assert accepted.status_code==200
    for _ in range(100):
        answers=[m for m in client.get('/api/llm/conversation',params={'target':target}).json()['conversation']['messages'] if m['role']=='assistant']
        if answers:break
        time.sleep(.01)
    assert answers[0]['text'].startswith('[WF:7][agent:coder_agent]\n')
    assert answers[0]['text'].endswith('\nDone.')
    assert '[RT:turn=' in answers[0]['text']
    assert [data['agent_role'] for op,data in calls if op=='start']==['coder_agent']
    assert not any(op=='gate' and data['kind']=='WORKFLOW_VERSION_DIAGNOSTIC' for op,data in calls)


def test_server_dispatch_endpoint_rejects_caller_recipe(client, monkeypatch):
    from subjective_devagents_fastapi.app import core
    before = dict(core().llm_service.backend.model._futures)
    response = client.post('/api/collaboration/execute', json={'binding': {}, 'execution': {}})
    assert response.status_code == 409
    assert response.json()['error'] == 'runtime_scope_mismatch'
    assert core().llm_service.backend.model._futures == before


def test_runtime_wake_rejects_large_body_even_with_forged_length(client):
    response = client.post('/api/collaboration/execute', content=b'x' * 2049,
        headers={'Content-Length': '2', 'Content-Type': 'application/json'})
    assert response.status_code == 413


def test_durable_attachment_endpoint_keeps_authenticated_owner(client, monkeypatch):
    import base64
    from subjective_devagents_fastapi import auth
    from subjective_devagents_fastapi.app import core
    from subjective_devagents_fastapi.collaboration import load_attachments
    monkeypatch.setattr(auth, 'authenticate_headers', lambda *_args, **_kwargs: {'sub': 'fixture-owner'})
    files = [{'filename': 'fixture.png', 'mime_type': 'image/png', 'content': base64.b64encode(b'fixture-image').decode()}]
    response = client.post('/api/collaboration/attachments', json={'files': files})
    assert response.status_code == 200
    reference = response.json()['reference']
    assert load_attachments(core().llm_service, reference, 'fixture-owner')[0].data == b'fixture-image'
    assert client.post('/api/collaboration/attachments', json={'files': []}).status_code == 422


def test_browser_admitted_workflow_turn_uses_channel_assigned_version(client, monkeypatch):
    from subjective_devagents_fastapi.app import core
    from subjective_devagents_fastapi import agent_workflow
    from subjective_devagents_fastapi.language_models import legacy
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://unused/v1')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['codex'], 'fixture-model')
    service = core().llm_service
    state = {'enabled':True,'workflow_id':'workflow','workflow_version':'20261114000000', 'task_family':'TASK','task_revision':'1','revision':1,
        **{key:{} for key in ('facts','leases','decisions','capabilities','reviews','gates','worktrees','discoveries')}}
    def call(self, op, **data):
        self._state = state
        if op == 'finish': self.closed = True
        return {'state':state}
    monkeypatch.setattr(ChannelSession, 'call', call)
    seen = []
    def prepare(turns, target, agent, authorization, text, app_home, **options):
        seen.append(options.get('version'))
        turns.setdefault(target, {})['header'] = ('20261114000000', 'coder_agent')
        return text
    monkeypatch.setattr(agent_workflow, 'prepare_turn', prepare)
    class Recorder(ModelStrategy):
        def run(self, request, emit, cancelled): return ModelResult('completed', 'Done.')
    service.backend.model.strategies['api_agent:codex'] = Recorder()
    target = client.post('/api/llm/create-conversation', json={'provider':'codex','name':'pinned workflow'}).json()['conversation']['target']
    binding = {'channel_id':'channel','workflow_id':'workflow','dispatch_id':'pinned','instance_id':'ai_coder:pinned','agent_id':'ai_coder'}
    accepted = client.post('/api/llm/send-message', json={'target':target,'text':'Continue', 'coordination':binding,
        'agent':{'company_id':'company_test','user_id':'ai_coder','role_file':'agents/roles/coder_agent.md'}})
    assert accepted.status_code == 200 and seen == ['20261114000000']


def test_api_cleanup_preserves_workers_durable_failure_receipt(client, monkeypatch):
    import json
    from subjective_devagents_fastapi.app import core
    from subjective_devagents_fastapi.language_models import legacy
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://unused/v1')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['codex'], 'fixture-model')
    service = core().llm_service
    state = {'enabled':True,'workflow_id':'workflow','workflow_version':None,'task_family':'TASK','task_revision':'1','revision':1,
        **{key:{} for key in ('facts','leases','decisions','capabilities','reviews','gates','worktrees','discoveries')}}
    finishes = []
    def call(self, op, **data):
        self._state = state
        if op == 'finish':
            finishes.append(data)
            raise ModelError('coordination_unavailable', 'temporary callback outage')
        return {'state':state}
    def admit(self, _company): self.delegation = {**self.binding, 'owner_id':'fixture-owner'}
    monkeypatch.setattr(ChannelSession, 'call', call)
    monkeypatch.setattr(ChannelSession, 'authorize_service_callbacks', admit)
    class Failure(ModelStrategy):
        def run(self, request, emit, cancelled): return ModelResult('failed', 'Provider refused execution', error_code='provider_unavailable')
    service.backend.model.strategies['api_agent:codex'] = Failure()
    target = client.post('/api/llm/create-conversation', json={'provider':'codex','name':'durable receipt'}).json()['conversation']['target']
    binding = {'channel_id':'channel','workflow_id':'workflow','dispatch_id':'durable','instance_id':'ai_coder:durable','agent_id':'ai_coder'}
    assert client.post('/api/llm/send-message', json={'target':target,'text':'work','coordination':binding}).status_code == 200
    for _ in range(100):
        with service.backend.model.store.connection() as db:
            row = db.execute('SELECT payload FROM coordination_completions WHERE dispatch_id=?', ('durable',)).fetchone()
        if row and target not in service.pending_sends: break
        time.sleep(.01)
    assert row is not None
    payload = json.loads(row['payload'])
    assert payload['data']['error_code'] == 'provider_unavailable'
    assert payload['data']['result']['text'] == 'Provider refused execution'
    assert len(finishes) == 1
