"""The actual API tool loop must fence writes, not merely mention leases in prompts."""
from dataclasses import replace
import json
import time
import pytest
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import ApiAgent, Identity, ModelRequest
from subjective_devagents_fastapi.language_models.contracts import ModelError
from test_api_agent import ScriptedProxy, completion, tool_call


class Journal:
    def __init__(self):
        self.calls=[];self.revoked=False
        self.state={'enabled':True,'workflow_id':'workflow','workflow_version':'published','task_family':'TASK','task_revision':'1','revision':1,
                    **{key:{} for key in ('facts','leases','decisions','capabilities','reviews','gates','worktrees','discoveries')}}
    def __call__(self, body):
        self.calls.append(body)
        if body['op']=='check' and (self.revoked or body.get('lease_id')!='held'):
            raise ModelError('LEASE_RECOVERY_REQUIRED','Lease revoked')
        if body['op']=='heartbeat' and self.revoked:
            raise ModelError('LEASE_RECOVERY_REQUIRED','Lease revoked')
        if body['op']=='acquire': return {'lease':{'id':'held'},'state':self.state}
        if body['op']=='decision':
            item={'id':body['id'],'question':body['question'],'status':'PENDING'}
            created=item['id'] not in self.state['decisions'];self.state['decisions'][item['id']]=item
            return {'decision':item,'created':created,'state':self.state}
        return {'state':self.state}


@pytest.fixture
def bound(tmp_path):
    root=tmp_path/'workspace';root.mkdir()
    identity=Identity('runtime','codex',model='test',strategy='api_agent',workspace=str(root),home=str(tmp_path),sandbox='workspace-write',api_base_url='http://unused',api_key_env='TEST_KEY')
    journal=Journal()
    session=ChannelSession({'channel_id':'channel','dispatch_id':'dispatch','instance_id':'ai_coder:run','workflow_id':'workflow'},'ai_coder','Bearer test-only',client=journal)
    session.start(identity,'published','coder_agent')
    return session,journal,identity


def test_real_file_write_requires_lease_and_forwards_canonical_context(bound,monkeypatch):
    session,journal,identity=bound;monkeypatch.setenv('TEST_KEY','fixture')
    target=identity.workspace+'/note.txt'
    with ScriptedProxy([
        completion(tool_calls=[tool_call('1','Write',file_path=target,content='forbidden')]),
        completion(tool_calls=[tool_call('2','Coordination',operation='acquire',data={'repository':identity.workspace,'paths':['note.txt']})]),
        completion(tool_calls=[tool_call('3','Write',file_path=target,content='allowed')]),
        completion(content='done'),
    ]) as proxy:
        request=ModelRequest('turn',replace(identity,api_base_url=proxy.url),'work',timeout=5,coordination=session)
        result=ApiAgent().run(request,lambda *_:None,lambda:False)
    assert result.status=='completed'
    assert open(target).read()=='allowed'
    assert any('WRITE_LEASE_REQUIRED' in str(m.get('content')) for m in proxy.requests[1][2]['messages'] if m['role']=='tool')
    assert any('agent_instance_id' in str(m['content']) and 'ai_coder:run' in str(m['content']) for m in proxy.requests[0][2]['messages'])
    assert any(c['op']=='capability' and c['status']=='proven' and c['capability']=='fs_write' for c in journal.calls)
    assert 'Bearer test-only' not in repr(session)


def test_guard_bash_is_global_and_paths_canonicalized(bound):
    session,journal,identity=bound
    session.tool('acquire',{'repository':identity.workspace,'paths':['.']})
    session.guard('Bash',{'command':'rm note'})
    assert journal.calls[-1]['path']==identity.workspace
    session.guard('Write',{'file_path':'sub/../file'})
    assert journal.calls[-1]['path']==identity.workspace+'/file'
    with pytest.raises(ModelError):session.tool('acquire',{'repository':'/unrelated','paths':['.']})


def test_read_only_shell_does_not_require_a_write_lease(bound):
    from subjective_devagents_fastapi.channel_coordination import shell_effect
    session, journal, _identity = bound
    before = len(journal.calls)
    for command in ('pwd', 'ls', 'git status', 'cat README', 'npm test'):
        assert shell_effect(command) == 'READ_ONLY'
        session.guard('Bash', {'command': command})
    assert len(journal.calls) == before
    for command in ('rm file', 'git checkout main', "sed -i 's/a/b/' file"):
        assert shell_effect(command) == 'WRITE'
        with pytest.raises(ModelError, match='write lease') as error:
            session.guard('Bash', {'command': command})
        assert error.value.code == 'WRITE_LEASE_REQUIRED'


def test_status_chatter_is_not_a_decision(bound):
    from subjective_devagents_fastapi.channel_coordination import concrete_user_decision
    session, journal, _identity = bound
    for phrase in ('what is the status of this?', 'retry', 'continue', 'you there?'):
        assert session.ask({'question': phrase}) == phrase
        assert concrete_user_decision('NEEDS INPUT: ' + phrase) is False
    assert journal.state['decisions'] == {}
    assert concrete_user_decision('NEEDS INPUT: decision_id=cloudflare_scoped_token_creation May an agent create a scoped token?') is True


def test_revocation_or_network_loss_stops_execution(bound):
    session,journal,identity=bound;session.tool('acquire',{'repository':identity.workspace,'paths':['.']})
    journal.revoked=True;session.last_check=0
    assert session.revoked()
    with pytest.raises(ModelError):session.guard('Write',{'file_path':'file'})


def test_agent_cannot_change_actor_or_self_approve_security(bound):
    session,journal,_=bound
    with pytest.raises(ModelError):session.tool('resolve_decision',{'actor_id':'owner','answer':'APPROVE'})
    with pytest.raises(ModelError):session.tool('capability',{'status':'proven','target':'account','capability':'cloud_write','source':'I said so'})
    session.tool('fact',{'actor_id':'owner','instance_id':'other','id':'main','kind':'local_sha','value':'a'*40,'repository':'repo','source':'git'})
    assert journal.calls[-1]['actor_id']=='ai_coder' and journal.calls[-1]['instance_id']=='ai_coder:run'


def test_ask_user_stable_id_and_explicit_security_identity(bound):
    session,journal,_=bound
    with pytest.raises(ModelError):session.ask({'question':'Use global credential?','category':'SECURITY_AUTHORIZATION'})
    first=session.ask({'decision_id':'grant','category':'SECURITY_AUTHORIZATION','scope':'disposable','question':'May I use global credential?'})
    second=session.ask({'decision_id':'grant','category':'SECURITY_AUTHORIZATION','scope':'disposable','question':'May I use global credential?'})
    assert first==second=='[COORDINATION] Pending decision: grant'
    assert len(journal.state['decisions'])==1
    session.finish('completed');assert journal.calls[-1]['status']=='blocked'


def test_workflow_binding_mismatch_refuses_turn(bound):
    session,journal,identity=bound
    journal.state['workflow_id']='other'
    with pytest.raises(ModelError,match='identity changed'):session.start(identity,'published')


def test_published_snapshot_identity_is_synchronized_and_diagnostic_retained(tmp_path):
    from subjective_devagents_fastapi.agent_workflow import materialize
    export={'version':{'version':'20261109120000'},'files':[{'path':'WORKFLOW_VERSION','content':'old-version'},{'path':'AGENTS.md','content':'contract'}]}
    folder=materialize(export,'company',tmp_path)
    assert (folder/'WORKFLOW_VERSION').read_text().strip()=='20261109120000'
    manifest=json.loads((folder/'runtime_identity.json').read_text())
    assert manifest['legacy_file_version']=='old-version' and manifest['source']=='published_workflow_database'
    assert materialize(export,'company',tmp_path)==folder


def test_conflicting_model_workflow_header_is_corrected_with_diagnostic(bound):
    session,journal,_=bound
    reply=session.canonical_reply('[WF:UNKNOWN][agent:coder_agent]\nDone.')
    assert reply=='[WF:published][agent:coder_agent]\nDone.'
    assert any(c['op']=='gate' and c['kind']=='WORKFLOW_VERSION_DIAGNOSTIC' and c['metadata']['observed'].startswith('[WF:UNKNOWN]') for c in journal.calls)


def test_shared_state_survives_long_history_compaction(bound, monkeypatch):
    from subjective_devagents_fastapi.language_models.contracts import Message
    session,_,identity=bound;monkeypatch.setenv('TEST_KEY','fixture')
    history=tuple(Message(role, 'old history '*20000) for role in ['user','assistant']*3)
    with ScriptedProxy([completion(content='Done.')]) as proxy:
        request=ModelRequest('turn',replace(identity,api_base_url=proxy.url),'latest assignment',history=history,timeout=5,coordination=session)
        assert ApiAgent().run(request,lambda *_:None,lambda:False).status=='completed'
    messages=proxy.requests[0][2]['messages']
    assert 'CHANNEL RUNTIME CONTEXT' in messages[0]['content']
    assert 'ai_coder:run' in messages[0]['content']
    assert 'latest assignment' in messages[-1]['content']
    assert len(messages)<len(history)+2


def test_revocation_stops_actual_shell_before_later_write(bound,monkeypatch):
    import threading
    from pathlib import Path
    session,journal,identity=bound;monkeypatch.setenv('TEST_KEY','fixture')
    session.tool('acquire',{'repository':'/','paths':['.']})
    later=Path(identity.workspace)/'must-not-be-written'
    timer=threading.Timer(.3,lambda:setattr(journal,'revoked',True))
    with ScriptedProxy([completion(tool_calls=[tool_call('1','Bash',command=f'sleep 5; touch {later}')])]) as proxy:
        request=ModelRequest('turn',replace(identity,api_base_url=proxy.url),'work',timeout=6,coordination=session)
        timer.start()
        try:result=ApiAgent().run(request,lambda *_:None,lambda:False)
        finally:timer.cancel()
    assert result.status=='cancelled'
    assert not later.exists()
