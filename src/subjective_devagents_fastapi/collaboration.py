"""Authoritative, idempotent server dispatch admission; no browser or instance addressing."""
from __future__ import annotations
from dataclasses import replace
import json
import os
import threading
from pathlib import Path
import time


from .channel_coordination import ChannelSession
from .language_models.contracts import ModelError

_admission = threading.RLock()


def validate_hint(hint):
    import re
    if not isinstance(hint, dict) or set(hint) != {'binding'} or not isinstance(hint['binding'], dict) or set(hint['binding']) != {'channel_id', 'dispatch_id'}:
        raise ModelError('runtime_scope_mismatch', 'Wake requests contain dispatch references only')
    if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_:-]{1,256}', value) for value in hint['binding'].values()):
        raise ModelError('runtime_scope_mismatch', 'Invalid wake dispatch reference')


def fetch_assignment(hint):
    import httpx
    validate_hint(hint)
    base = os.environ.get('COORDINATION_CALLBACK_URL', '').rstrip('/')
    token = os.environ.get('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', '')
    if not base or not token:
        raise ModelError('coordination_unavailable', 'Existing workflow service authentication is unavailable')
    route = '/internal/collaboration/' + hint['binding']['channel_id'] + '/dispatch/' + hint['binding']['dispatch_id']
    try:
        with httpx.Client(timeout=12, follow_redirects=False, trust_env=False) as client:
            response = client.post(base + route, headers={'X-Workflow-Service-Token': token})
        payload = response.json()
        if response.status_code != 200 or not payload.get('ok'):
            raise ModelError('runtime_assignment_unavailable', 'Approved runtime assignment is unavailable')
        job = payload['data']
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ModelError('coordination_unavailable', 'Cannot retrieve approved runtime assignment') from None
    if any(job.get('binding', {}).get(key) != value for key, value in hint['binding'].items()):
        raise ModelError('runtime_scope_mismatch', 'Authoritative assignment differs from wake reference')
    return job


def execute(service, payload, _unused=None):
    # Bound unauthenticated wakeups; only the authoritative approved job supplies executable data.
    validate_hint(payload)
    if not _admission.acquire(blocking=False):
        raise ModelError('runtime_busy', 'Runtime admission is already processing a dispatch')
    try:
        job = fetch_assignment(payload)
        return _execute_approved(service, job)
    finally:
        _admission.release()


def _execute_approved(service, job):
    binding, execution = job['binding'], job['execution']
    claims = {**binding, 'owner_id': job['owner_id']}
    profile = execution['agents'][binding['agent_id']]
    if profile.get('company_id') != os.environ.get('COORDINATION_COMPANY_ID'):
        raise ModelError('runtime_scope_mismatch', 'Runtime company differs from installation')
    model = service.backend.model
    with _admission:
        with model.store.connection() as db:
            row = db.execute('SELECT conversation_id FROM collaboration_dispatches WHERE dispatch_id=?', (binding['dispatch_id'],)).fetchone()
        if row:
            conversation_id = row['conversation_id']
            conversation = model.store.conversation(conversation_id)
            if conversation['active_turn_id']:
                return {'ok': True, 'turn_id': conversation['active_turn_id'], 'reused': True}
            with model.store.connection() as db:
                previous = db.execute('SELECT id FROM turns WHERE conversation_id=? AND request_id=?',
                    (conversation_id, binding['dispatch_id'])).fetchone()
            if previous:
                return {'ok': True, 'turn_id': previous['id'], 'reused': True}
        else:
            conversation_id = None
        # Profiles can select a configured provider/model, never arbitrary workspaces/accounts/URLs.
        identity = service.backend.identity({'provider': profile['provider'], 'model': profile.get('model', ''),
            'workspace': service._default_workspace(profile['provider'])}, execution_required=True)
        if identity.strategy != 'api_agent' or identity.ssh_host:
            raise ModelError('coordinated_runtime_required', 'Collaboration requires a local runtime with per-tool fencing')
        if profile.get('mode') == 'read':
            identity = replace(identity, id=identity.id + ':readonly', sandbox='read-only')
            model.identities.register_legacy(identity)
        from . import agent_workflow
        export = agent_workflow.fetch_export_as_service(profile['company_id'])
        folder = agent_workflow.materialize(export, profile['company_id'], agent_workflow.workflow_root(service.app_home))
        definition = agent_workflow.compose_definition(export, profile, folder)
        prompt = agent_workflow.wrap_turn(definition, execution['prompt'])
        if conversation_id is None:
            conversation_id = model.create(identity.id)['id']
            with model.store.connection() as db:
                db.execute('INSERT INTO collaboration_dispatches VALUES(?,?)', (binding['dispatch_id'], conversation_id))
        channel = ChannelSession(binding, binding['agent_id'], '')
        # Callback identity is resolved from the approved job; the existing service token stays in process env.
        channel.delegation = {k: claims[k] for k in ('channel_id', 'dispatch_id', 'instance_id', 'workflow_id', 'agent_id', 'owner_id')}
        channel.fence_root = str(Path(model.store.path).parent / 'mutation-fences')
        channel.defer_start(identity, export['version']['version'], agent_workflow.agent_id_for(profile['role_file']))
        attachments = load_attachments(service, execution['attachment_ref'], claims['owner_id']) if execution.get('attachment_ref') else ()
        turn = model.submit(conversation_id, prompt, binding['dispatch_id'], timeout=1800, attachments=attachments, coordination=channel)
        return {'ok': True, 'turn_id': turn['id']}


def replay_completions(store):
    """Retry committed worker results after transient chat outages or a runtime restart."""
    with store.connection() as db:
        rows = db.execute('SELECT * FROM coordination_completions').fetchall()
    for row in rows:
        payload = json.loads(row['payload'])
        channel = ChannelSession(payload['binding'], payload['actor'], '')
        channel.delegation = payload['delegation']
        try:
            channel.call('finish', **payload['data'])
        except ModelError as error:
            if error.code not in {'INSTANCE_NOT_RUNNING', 'DISPATCH_NOT_RUNNING', 'STALE_ASSIGNMENT', 'DISPATCH_NOT_FOUND', 'not_a_member'}:
                continue
        with store.connection() as db:
            db.execute('DELETE FROM coordination_completions WHERE dispatch_id=?', (row['dispatch_id'],))


def save_attachments(service, files, owner_id):
    """Persist images on the runtime host; chat jobs keep only an owner-checked reference."""
    import uuid
    from .language_models.contracts import MAX_ATTACHMENTS
    if not isinstance(files, list) or not 0 < len(files) <= MAX_ATTACHMENTS:
        raise ModelError('invalid_attachment', 'A bounded list of images is required')
    service.backend.attachments(files)  # Apply the same size/type checks as ordinary sends.
    folder = Path(service.app_home) / 'collaboration-attachments'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    reference = str(uuid.uuid4())
    descriptor = os.open(folder / (reference + '.json'), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'owner_id': owner_id, 'files': files}, stream)
    return reference


def load_attachments(service, reference, owner_id):
    import re
    if not isinstance(reference, str) or not re.fullmatch(r'[a-f0-9-]{36}', reference):
        raise ModelError('invalid_attachment', 'Invalid durable attachment reference')
    try:
        record = json.loads((Path(service.app_home) / 'collaboration-attachments' / (reference + '.json')).read_text())
    except (OSError, ValueError):
        raise ModelError('invalid_attachment', 'Durable attachments are unavailable') from None
    if record.get('owner_id') != owner_id:
        raise ModelError('runtime_scope_mismatch', 'Attachments belong to another requester')
    return service.backend.attachments(record['files'])
