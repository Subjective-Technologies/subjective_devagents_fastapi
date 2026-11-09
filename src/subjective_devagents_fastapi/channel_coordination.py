"""Ephemeral binding from a DevAgents turn to the authoritative chat journal.

Bearer credentials stay in memory. No model-supplied URL/actor/instance is trusted.
Coordination loss fails closed before executing another tool.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time
import httpx
from .language_models.contracts import ModelError

TOOL_OPERATIONS = {'task', 'recover', 'reissue', 'adopt_worktree', 'blocker', 'snapshot', 'acquire', 'release', 'fact', 'decision', 'discovery',
                   'capability', 'route', 'privilege', 'review', 'gate', 'worktree'}
CONTROL_TOOL = {'type': 'function', 'function': {
    'name': 'Coordination',
    'description': ('Read/update authoritative channel state. Operations: snapshot; acquire '
        '(repository absolute, paths literal relative directories/files, mode read/write); release '
        '(lease_id); fact (id,kind,repository,value,source,ttl,expected_revision when replacing); '
        'decision (id,category,question,scope,blocking_tasks); discovery (id,result,credential_sources_checked,'
        'specialists_consulted); capability (capability,status,target,source); review '
        '(id,repository,sha,artifact_fact_id,test_revision,verdict); gate (id,kind,result,metadata); '
        'worktree (path,lease_id,base_sha,branch,state,dirty,pushed). '
        'Before Write/Edit acquire a covering write lease. Bash requires repository / and paths [.] '
        'because shell paths cannot be inferred safely. Release after reconciling worktrees. '
        'Never submit credentials. Security decisions require explicit IDs; agents cannot resolve them.'),
    'parameters': {'type': 'object', 'properties': {'operation': {'type': 'string'},
        'data': {'type': 'object'}}, 'required': ['operation', 'data']}}}


class ChannelSession:
    def __init__(self, binding, actor, authorization, *, client=None):
        if not isinstance(binding, dict):
            raise ModelError('invalid_coordination', 'A channel dispatch binding is required')
        self.binding = {k: binding.get(k) for k in ('channel_id', 'dispatch_id', 'instance_id', 'workflow_id')}
        if any(not isinstance(v, str) or not v or len(v) > 256 or '/' in v for v in self.binding.values()):
            raise ModelError('invalid_coordination', 'Invalid channel dispatch identity')
        self.actor, self._authorization, self._client = actor, authorization, client
        self.runtime = ''
        self.lease_id = None
        self.last_check = 0.0
        self.identity = None
        self.closed = False
        self.blocked = False
        self.privilege = None
        self._state = None
        self.tool_capabilities = set()

    def __repr__(self):
        return f'ChannelSession(channel={self.binding["channel_id"]!r}, instance={self.binding["instance_id"]!r})'

    def call(self, op, **data):
        body = {**data, 'op': op, 'actor_id': self.actor,
                'dispatch_id': self.binding['dispatch_id'], 'instance_id': self.binding['instance_id']}
        if 'lease_id' not in body and self.lease_id:
            body['lease_id'] = self.lease_id
        if self._client:
            result = self._client(body)
        else:
            base = os.environ.get('SUBJECTIVE_CHAT_COORDINATION_URL',
                'https://dashboard.subjectivetechnologies.com/api/chat-proxy').rstrip('/')
            try:
                with httpx.Client(timeout=12, follow_redirects=False, trust_env=False) as client:
                    response = client.post(base + '/chat/conversations/' + self.binding['channel_id'] + '/coordination',
                        headers={'Authorization': self._authorization}, json=body)
                payload = response.json()
                if response.status_code != 200 or not payload.get('ok'):
                    error = payload.get('error')
                    code = error.get('code', 'coordination_unavailable') if isinstance(error, dict) else 'coordination_unavailable'
                    raise ModelError(code, 'Channel coordination rejected operation')
                result = payload['data']
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise ModelError('coordination_unavailable', 'Channel coordination is unavailable') from None
        self._state = result.get('state', self._state)
        if result.get('code') in {'LEASE_CONFLICT', 'WORKFLOW_VERSION_CONFLICT'}:
            raise ModelError(result['code'], json.dumps({k: v for k, v in result.items() if k != 'state'}))
        return result

    def start(self, identity, version, agent_role=None):
        self.identity = identity
        self.runtime = identity.id
        self.version, self.agent_role = version, agent_role or self.actor
        snapshot = self.call('snapshot')['state']
        if not snapshot.get('enabled'):
            raise ModelError('coordination_disabled', 'Channel coordination is disabled')
        if snapshot['workflow_id'] != self.binding['workflow_id']:
            raise ModelError('workflow_identity_conflict', 'Channel workflow identity changed')
        self.call('start', workflow_version=version, runtime=identity.id, agent_role=agent_role or self.actor)
        legacy = Path(identity.workspace) / 'workflow/WORKFLOW_VERSION'
        if not identity.ssh_host and legacy.is_file():
            observed = legacy.read_text()[:80].strip()
            if observed != version:
                self.call('gate', id='legacy-workflow-version-' + self.binding['instance_id'],
                    kind='WORKFLOW_VERSION_DIAGNOSTIC', result='failed', metadata={
                        'source': str(legacy), 'observed': observed, 'authoritative': version})
        return self.context()

    def context(self):
        state = self.call('snapshot')['state']
        state['leases'] = {key: lease for key, lease in state['leases'].items() if lease['status'] in {'active', 'revoking', 'recovery_required'}}
        # The state is supplied separately on every turn; never rely on transcript replay.
        return '\nCHANNEL RUNTIME CONTEXT (authoritative journal; reported facts retain provenance):\n' + json.dumps({
            'workflow_id': state['workflow_id'], 'workflow_version': state['workflow_version'],
            'agent_instance_id': self.binding['instance_id'], 'task_family': state['task_family'],
            'task_revision': state['task_revision'], 'state_revision': state['revision'],
            **{key: state.get(key, [] if key == 'completed_gates' else {}) for key in ('facts', 'leases', 'decisions', 'capabilities', 'reviews', 'gates', 'worktrees', 'discoveries', 'blockers', 'completed_gates')}
        }, sort_keys=True)

    def tool(self, operation, data):
        if operation not in TOOL_OPERATIONS or not isinstance(data, dict):
            raise ModelError('coordination_operation_forbidden', 'Unknown or privileged coordination operation')
        data = dict(data)
        if operation == 'acquire':
            if data.get('mode', 'write') == 'write' and (self.identity.sandbox != 'workspace-write' or (self.identity.allowed_tools and not {'Bash', 'Write', 'Edit'}.intersection(self.identity.allowed_tools))):
                raise ModelError('BLOCKED_EXECUTION_CAPABILITY', 'Runtime has no mutating tools')
            # Canonicalize host-local paths, including symlinks. Remote executors cannot
            # pretend the DevAgents host's filesystem proves their remote target.
            if self.identity.ssh_host:
                raise ModelError('remote_scope_unverified', 'Remote path leases require a verified execution-host resolver')
            requested = str(Path(data.get('repository', self.identity.workspace)).resolve())
            root = str(Path(self.identity.workspace).resolve())
            registered = (self._state or {}).get('worktrees', {}).get(requested)
            owned_worktree = registered and registered['instance_id'] == self.binding['instance_id']
            if not owned_worktree and requested != '/' and not (requested == root or requested.startswith(root.rstrip('/') + '/')):
                raise ModelError('lease_outside_workspace', 'Lease repository is outside execution workspace')
            data['repository'] = requested
            resolved = [(Path(requested) / p).resolve() for p in data.get('paths', ['.'])]
            if any(not p.is_relative_to(requested) for p in resolved):
                raise ModelError('lease_outside_repository', 'Path resolves outside repository')
            data['paths'] = [str(p.relative_to(requested)) for p in resolved]
        if operation == 'worktree':
            if self.identity.ssh_host:
                raise ModelError('remote_scope_unverified', 'Worktree observations require a local execution-host resolver')
            location = str(Path(data['path']).resolve())
            repository = str(Path(data.get('repository', self.identity.workspace)).resolve())
            def git(root, *args):
                result = subprocess.run(['git', '-c', 'core.fsmonitor=false', '-C', root, *args],
                    capture_output=True, text=True, timeout=10, env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': self.identity.home or self.identity.workspace})
                if result.returncode:
                    raise ModelError('worktree_observation_failed', 'Cannot verify local Git worktree')
                return result.stdout.strip()
            if git(location, 'rev-parse', '--path-format=absolute', '--git-common-dir') != git(repository, 'rev-parse', '--path-format=absolute', '--git-common-dir'):
                raise ModelError('worktree_repository_mismatch', 'Worktree does not belong to reported repository')
            data.update(path=location, repository=repository, actual_sha=git(location, 'rev-parse', 'HEAD'),
                        dirty=bool(git(location, 'status', '--porcelain')), dirty_provenance='runtime_git_status')
        if operation == 'capability':
            data.update(agent_id=self.actor, runtime=self.runtime)
            # Agents may declare capabilities, but only observed tool execution proves them.
            if data.get('status') == 'proven':
                raise ModelError('capability_proof_required', 'Runtime tool observations, not declarations, prove capabilities')
        result = self.call(operation, **data)
        if operation == 'acquire':
            self.lease_id = result['lease']['id']
        if operation == 'privilege':
            self.privilege = {k: data[k] for k in ('privilege_level', 'scope', 'decision_id') if k in data}
        if operation != 'snapshot':
            result = {k: v for k, v in result.items() if k != 'state'} | {'state_revision': self._state['revision']}
        return json.dumps(result, sort_keys=True)

    def guard(self, name, arguments):
        # All shell commands are treated as possible writes. No regex pretending to be a sandbox.
        if name not in {'Bash', 'Write', 'Edit'}:
            return
        if self.privilege and self.privilege['privilege_level'] != 'scoped':
            self.call('privilege', **self.privilege)
        if not self.lease_id:
            raise ModelError('WRITE_LEASE_REQUIRED', 'Acquire a covering Coordination write lease before executing')
        if self.identity.ssh_host:
            raise ModelError('remote_scope_unverified', 'Remote execution scopes are not verified')
        target = '/' if name == 'Bash' else str((Path(self.identity.workspace) / arguments.get('file_path', '')).resolve())
        state = self.call('check', path=target)['state']
        for tree in state['worktrees'].values():
            foreign = tree['instance_id'] != self.binding['instance_id']
            affected = name == 'Bash' and (tree['state'] == 'abandoned' or tree.get('dirty'))
            affected = affected or Path(target).is_relative_to(tree['path'])
            if foreign and affected:
                raise ModelError('WORKTREE_OWNED', 'Explicit worktree adoption/reconciliation required')
        self.last_check = time.monotonic()

    def revoked(self):
        if not self.lease_id or time.monotonic() - self.last_check < 2:
            return False
        self.last_check = time.monotonic()
        try:
            self.call('heartbeat')
            return False
        except ModelError:
            return True

    def observed(self, name, ok):
        capability = {'Read': 'fs_read', 'Grep': 'fs_read', 'Glob': 'fs_read',
                      'Write': 'fs_write', 'Edit': 'fs_write', 'Bash': 'shell_exec'}.get(name)
        if ok and capability and (capability, ok) not in self.tool_capabilities:
            self.call('capability', agent_id=self.actor, runtime=self.runtime,
                target=self.identity.workspace, capability=capability,
                status='proven' if ok else 'failed', source='devagents_tool_result:' + name,
                credential_ref='runtime:' + self.identity.id, privilege_level='scoped')
            self.tool_capabilities.add((capability, ok))

    def ask(self, arguments):
        if any(l['instance_id'] == self.binding['instance_id'] and l['mode'] == 'write' and l['status'] in {'active', 'revoking'} for l in self.call('snapshot')['state']['leases'].values()):
            raise ModelError('RELEASE_WRITE_LEASE_BEFORE_DECISION', 'Reconcile worktrees and release writes before waiting for user')
        question = str(arguments.get('question', '')).strip()
        if not question:
            raise ModelError('invalid_question', 'Question is required')
        category = arguments.get('category', 'TECHNICAL_INTERNAL_BLOCKER')
        identifier = arguments.get('decision_id')
        if category in {'SECURITY_AUTHORIZATION', 'DESTRUCTIVE_ACTION_AUTHORIZATION'} and not identifier:
            raise ModelError('decision_id_required', 'Authorization questions require explicit stable IDs')
        identifier = identifier or 'question-' + hashlib.sha256(' '.join(question.lower().split()).encode()).hexdigest()[:24]
        result = self.call('decision', id=identifier, question=question, category=category,
            scope=arguments.get('scope'), discovery_id=arguments.get('discovery_id'),
            blocking_tasks=arguments.get('blocking_tasks', []))
        item = result['decision']
        self.blocked = item['status'] == 'PENDING'
        # UI renders the one canonical question. Repeated calls expose only reference ID.
        return f'[COORDINATION] Pending decision: {item["id"]}' if item['status'] == 'PENDING' else f'Decision {item["id"]}: {item.get("answer", item["status"])}'

    def canonical_reply(self, reply):
        from .agent_workflow import _HEADER, ensure_header
        observed = _HEADER.match(reply)
        expected = f"[WF:{self.version}][agent:{self.agent_role}]"
        if observed and observed.group(0).strip() != expected:
            self.call('gate', id='reply-header-' + self.binding['instance_id'],
                kind='WORKFLOW_VERSION_DIAGNOSTIC', result='failed', metadata={
                    'source': 'provider_reply', 'observed': observed.group(0).strip()[:256], 'authoritative': expected})
        if 'BLOCKED_EXECUTION_CAPABILITY' in reply:
            self.blocked = True
        return ensure_header(reply, self.version, self.agent_role)

    def finish(self, status, error_code=None):
        if self.closed:
            return
        self.closed = True
        self.call('finish', status='blocked' if self.blocked else 'completed' if status == 'completed' else 'blocked' if status == 'blocked' else 'failed', executor_stopped=status not in {'interrupted', 'timed_out'} and 'uncertain' not in str(error_code))
        self._authorization = ''
