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
