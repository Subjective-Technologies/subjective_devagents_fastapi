"""Ephemeral binding from a DevAgents turn to the authoritative chat journal.

Bearer credentials stay in memory. No model-supplied URL/actor/instance is trusted.
Coordination loss fails closed before executing another tool.
"""
from __future__ import annotations
import hashlib
import logging
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
import time
import httpx
from .language_models.contracts import ModelError

log = logging.getLogger(__name__)

_READ_COMMANDS = {
    'pwd', 'ls', 'cat', 'head', 'tail', 'wc', 'whoami', 'date', 'true', 'false', 'echo', 'printf',
    'which', 'type', 'file', 'stat', 'test', 'rg', 'grep', 'find', 'du', 'df', 'uname', 'id',
    'nl',
}
_ASSIGNMENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
# env/npm flags that consume the next argv. Leaving them in the positional list
# would classify the directory (or the variable name) as the command.
_ENV_VALUED = {'-u', '--unset', '-C', '--chdir', '-S', '--split-string'}
_NPM_VALUED = {'--prefix', '--userconfig', '--cache', '--registry', '-C', '--package', '-p', '--call', '-c'}
_WRITE_COMMANDS = {
    'rm', 'mv', 'cp', 'mkdir', 'rmdir', 'touch', 'chmod', 'chown', 'tee', 'truncate', 'dd', 'install',
    'ln', 'unlink',
}
_GIT_READ = {'status', 'diff', 'log', 'show', 'rev-parse', 'branch', 'remote', 'ls-files', 'describe', 'blame', 'shortlog'}
_GIT_WRITE = {'checkout', 'switch', 'commit', 'push', 'reset', 'clean', 'rebase', 'merge', 'cherry-pick', 'am', 'apply', 'stash', 'add', 'rm', 'mv', 'restore'}
_STATUS_LINE = re.compile(
    r"^\s*(?:what(?:'s| is) the status(?: of this)?|status|continue|retry|try again|"
    r"is it done|what happened|verify this|you there)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _segments(command):
    parts = [piece.strip() for piece in re.split(r'\s*(?:&&|\|\||;|\|)\s*', command or '')]
    return [piece for piece in parts if piece] or ['']


def _positional(tokens):
    args = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == '--':
            args.extend(tokens[index + 1:])
            break
        if token in _NPM_VALUED:
            index += 2
            continue
        if token.startswith('--') and '=' in token:
            index += 1
            continue
        if token.startswith('-'):
            index += 1
            continue
        args.append(token)
        index += 1
    return args


def _env_command(tokens):
    """argv after env options and NAME=VALUE. Empty when env only prints the environment."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _ASSIGNMENT.match(token):
            index += 1
            continue
        if token == '--':
            return tokens[index + 1:]
        if token in _ENV_VALUED:
            index += 2
            continue
        if token.startswith('--') and '=' in token:
            index += 1
            continue
        if token.startswith('-'):
            index += 1
            continue
        return tokens[index:]
    return []


def _literal_directory(value):
    """A directory token the guard can resolve. Shell expansions are not literal."""
    if not value or value.startswith('-'):
        return None
    if any(ch in value for ch in '$`*?[]<>|&;(){}'):
        return None
    if value.startswith('~'):
        value = os.path.expanduser(value)
        if value.startswith('~'):
            return None
    return value


def _cd_destination(tokens):
    args = tokens[1:]
    if args[:1] == ['--']:
        args = args[1:]
    if len(args) != 1:
        return None
    return _literal_directory(args[0])


def _prefix_directory(tokens):
    """Return the --prefix directory, None when absent, or False when it is not literal."""
    for index, token in enumerate(tokens):
        if token == '--':
            return None
        if token == '--prefix' and index + 1 < len(tokens):
            return _literal_directory(tokens[index + 1]) or False
        if token.startswith('--prefix='):
            return _literal_directory(token.split('=', 1)[1]) or False
    return None


def bash_workdir(command, workspace):
    """Directory a Bash command runs in.

    A leading ``cd <dir>`` (including ``cd a && cd b``) or ``--prefix <dir>`` selects
    it. Otherwise the command runs at the workspace root. A non-literal cd or prefix
    fails closed: ``cd $HOME && rm`` must not inherit a workspace lease.
    """
    root = Path(workspace).resolve()
    current = root
    bits = re.split(r'\s*(&&|\|\||;|\|)\s*', command or '')
    index = 0
    while index < len(bits):
        part = bits[index].strip()
        separator = bits[index + 1] if index + 1 < len(bits) else None
        index += 2
        if not part:
            continue
        try:
            tokens = shlex.split(part, posix=True)
        except ValueError:
            raise ModelError('LEASE_SCOPE_MISMATCH', 'Shell working directory is not a literal path') from None
        while tokens and _ASSIGNMENT.match(tokens[0]):
            del tokens[0]
        if not tokens:
            continue
        if tokens[0] == 'cd':
            # `|` and `||` do not leave the following command in that directory.
            if separator not in {'&&', ';', None}:
                raise ModelError('LEASE_SCOPE_MISMATCH', 'Shell working directory is not a literal path')
            dest = _cd_destination(tokens)
            if not dest:
                raise ModelError('LEASE_SCOPE_MISMATCH', 'Shell working directory is not a literal path')
            path = Path(dest)
            current = (path if path.is_absolute() else current / path).resolve()
            continue
        prefix = _prefix_directory(tokens)
        if prefix is False:
            raise ModelError('LEASE_SCOPE_MISMATCH', 'Shell working directory is not a literal path')
        if prefix:
            path = Path(prefix)
            current = (path if path.is_absolute() else current / path).resolve()
        return str(current)
    return str(current)


def _segment_effect(part):
    try:
        tokens = shlex.split(part, posix=True)
    except ValueError:
        return 'UNKNOWN'
    # FOO=bar cmd runs cmd. The assignment itself writes nothing.
    while tokens and _ASSIGNMENT.match(tokens[0]):
        del tokens[0]
    if not tokens:
        return 'READ_ONLY'
    base = tokens[0].rsplit('/', 1)[-1]
    if base in {'pytest', 'py.test'} or (re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', base) and tokens[1:3] == ['-m', 'pytest']):
        return 'READ_ONLY'
    if base in {'sudo', 'command', 'time', 'nohup'} and len(tokens) > 1:
        return _segment_effect(' '.join(shlex.quote(token) for token in tokens[1:]))
    # `env rm` must be classified as rm. Bare env/printenv only print.
    if base == 'env':
        tail = _env_command(tokens[1:])
        if not tail:
            return 'READ_ONLY'
        return _segment_effect(' '.join(shlex.quote(token) for token in tail))
    if base == 'printenv':
        return 'READ_ONLY'
    if base == 'cd':
        return 'READ_ONLY'
    if base == 'git':
        sub = next((token for token in tokens[1:] if not token.startswith('-')), '')
        if sub in _GIT_WRITE:
            return 'WRITE'
        if sub in _GIT_READ:
            return 'READ_ONLY'
        return 'UNKNOWN'
    if base in {'npm', 'pnpm', 'yarn'}:
        args = _positional(tokens)
        # `npm run build` writes dist/. `npm test` / `npm ls` do not.
        if args[:2] == ['run', 'build']:
            return 'WRITE'
        if args[:1] == ['test'] or args[:2] == ['run', 'test'] or args[:1] in (['ls'], ['list']):
            return 'READ_ONLY'
        if 'test' in tokens[1:3]:
            return 'READ_ONLY'
        return 'UNKNOWN'
    if base == 'npx':
        if _positional(tokens)[:2] == ['vitest', 'run']:
            return 'READ_ONLY'
        return 'UNKNOWN'
    if base == 'node' and tokens[1:] and set(tokens[1:]) <= {'--version', '-v'}:
        return 'READ_ONLY'
    if base == 'sed':
        return 'WRITE' if '-i' in tokens or any(token.startswith('--in-place') for token in tokens) else 'READ_ONLY'
    if base in _WRITE_COMMANDS:
        return 'WRITE'
    if base in _READ_COMMANDS:
        return 'READ_ONLY'
    return 'UNKNOWN'


def shell_effect(command):
    """READ_ONLY, WRITE, or UNKNOWN. A read-only command does not need a write lease."""
    effects = [_segment_effect(part) for part in _segments(command)]
    if any(item == 'WRITE' for item in effects):
        return 'WRITE'
    if any(item == 'UNKNOWN' for item in effects):
        return 'UNKNOWN'
    return 'READ_ONLY'


def is_status_chatter(question):
    return bool(_STATUS_LINE.match(str(question or '').strip()))


def concrete_user_decision(text):
    """A user decision needs an explicit id and a question that is not status chatter."""
    raw = text or ''
    question = raw.split('NEEDS INPUT:', 1)[-1].strip() if 'NEEDS INPUT:' in raw else raw.strip()
    if is_status_chatter(question) or is_status_chatter(raw):
        return False
    if not re.search(r'(?i)\bdecision[_ -]?id\b\s*[:=]\s*[A-Za-z0-9_-]{3,}', raw):
        return False
    return bool(question)

def _seconds(name, default):
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return float(default)
    return value if value >= 0 else float(default)


# How long a running turn tolerates an unreachable journal before it stops. A Chat API
# redeploy restarts its container (cold boot plus Litestream restore takes up to ~2
# minutes); 30 seconds killed every running agent on every deploy. Keep it below the
# journal's heartbeat timeout (300 s) so a revived journal still sees this worker alive.
UNREACHABLE_GRACE = _seconds('COORDINATION_UNREACHABLE_GRACE', 240)
# Bounded exponential backoff for one coordination request (seconds between tries).
CALL_BACKOFF = tuple(float(x) for x in os.environ.get('COORDINATION_CALL_BACKOFF', '1,2,4,8').split(',') if x.strip())
# Liveness pings tolerate outages through UNREACHABLE_GRACE instead of blocking a tool loop.
_NO_RETRY = {'touch', 'heartbeat'}
# Bytes of journal state a worker sees (system prompt and snapshot tool), whatever the channel's age.
CONTEXT_STATE_LIMIT = int(os.environ.get('COORDINATION_CONTEXT_STATE_LIMIT', '8000'))
TOOL_OPERATIONS = {'task', 'recover_task_lease', 'recover', 'reissue', 'adopt_worktree', 'blocker', 'snapshot', 'acquire', 'release', 'fact', 'decision', 'discovery',
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
        'Before Write/Edit acquire a covering write lease. Read-only shell (pwd, ls, git status, cat, npm test, '
        'npm ls, node --version, npx vitest run) needs no write lease. A leading cd or VAR=value/env prefix '
        'does not itself write. npm run build and other mutating or unknown shell need a write lease covering '
        'the command directory (leading cd or --prefix), not the whole workspace and not the filesystem root. '
        'Release after reconciling worktrees. '
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
        self.revocation = None
        self._unreachable_since = None
        self.identity = None
        self.closed = False
        self.blocked = False
        self.privilege = None
        self._state = None
        self.tool_capabilities = set()
        self.delegation = None
        self.completion_store = None
        self.fence_root = None
        self.pending_start = None
        self.start_acknowledged = False
        self.heartbeat_interval = max(1.0, float(os.environ.get('COORDINATION_HEARTBEAT_INTERVAL', '15')))
        # Verify profile: a read-only runtime that may run tests, type checks and builds.
        self.verification = False
        self.runtime_profile = None
        # The latest instruction (tail of the dispatch prompt): Manager names commits it bundles here.
        self.assignment = ''
        self._last_touch = 0.0
        self._usage_pending = {}

    def __repr__(self):
        return f'ChannelSession(channel={self.binding["channel_id"]!r}, instance={self.binding["instance_id"]!r})'

    def call(self, op, **data):
        """One journal operation; transient outages are retried with bounded backoff."""
        delays = () if op in _NO_RETRY else CALL_BACKOFF
        for attempt in range(len(delays) + 1):
            try:
                return self._call_once(op, **data)
            except ModelError as error:
                if error.code != 'coordination_unavailable' or attempt >= len(delays):
                    raise
                log.warning('Coordination unavailable; retrying op=%s attempt=%d channel=%s',
                            op, attempt + 1, self.binding['channel_id'])
                time.sleep(delays[attempt])

    def _call_once(self, op, **data):
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
                route = '/chat/conversations/' + self.binding['channel_id'] + '/coordination'
                if self.delegation:
                    base = os.environ.get('COORDINATION_CALLBACK_URL', '').rstrip('/')
                    if not base:
                        raise ModelError('coordination_unavailable', 'Runtime callback URL is not configured')
                    route = '/internal/collaboration/' + self.binding['channel_id'] + '/coordination'
                with httpx.Client(timeout=12, follow_redirects=False, trust_env=False) as client:
                    response = client.post(base + route,
                        headers=self._headers(), json=body)
                if response.status_code >= 500 or response.status_code == 429:
                    # Container restarting, Worker warming up, issuer overloaded: transient.
                    raise ModelError('coordination_unavailable', 'Channel coordination is temporarily unavailable')
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

    def _headers(self):
        if self.delegation:
            token = os.environ.get('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', '')
            if not token:
                raise ModelError('coordination_unavailable', 'Existing workflow service identity is unavailable')
            return {'X-Workflow-Service-Token': token}
        return {'Authorization': self._authorization}

    def authorize_service_callbacks(self, company):
        """Bind this caller-approved dispatch to the installed runtime before its login expires."""
        if not company or company != os.environ.get('COORDINATION_COMPANY_ID'):
            return
        base = os.environ.get('COORDINATION_CALLBACK_URL', '').rstrip('/')
        token = os.environ.get('SUBJECTIVE_DEVAGENTS_WORKFLOW_SERVICE_TOKEN', '')
        if not base or not token:
            raise ModelError('coordination_unavailable', 'Existing runtime service authentication is unavailable')
        try:
            with httpx.Client(timeout=15, follow_redirects=False, trust_env=False) as client:
                response = client.post(base + '/internal/collaboration/' + self.binding['channel_id'] + '/admit',
                    headers={'Authorization': self._authorization, 'X-Workflow-Service-Token': token}, json={**self.binding, 'agent_id': self.actor})
            payload = response.json()
            if response.status_code != 200 or not payload.get('ok'):
                error = payload.get('error', {})
                raise ModelError(error.get('code', 'runtime_admission_denied') if isinstance(error, dict) else 'runtime_admission_denied',
                                 'Runtime callback admission was rejected')
            approved = payload['data']
            if (any(approved['binding'].get(key) != value for key, value in self.binding.items())
                    or approved['binding'].get('agent_id') != self.actor or not approved.get('owner_id')):
                raise ModelError('runtime_scope_mismatch', 'Runtime admission differs from requested dispatch')
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise ModelError('coordination_unavailable', 'Runtime callback admission is unavailable') from None
        self.delegation = {**self.binding, 'owner_id': approved['owner_id']}
        self._authorization = ''

    def defer_start(self, identity, version, agent_role=None):
        self.pending_start = (identity, version, agent_role)

    def worker_started(self):
        if self.pending_start:
            args, self.pending_start = self.pending_start, None
            try:
                self.start(*args)
            except ModelError as error:
                if not self.start_acknowledged and error.code in {
                    'DISPATCH_SUPPRESSED', 'EXECUTION_ACTIVE', 'INSTANCE_MISMATCH', 'INSTANCE_NOT_RUNNING',
                    'DISPATCH_NOT_RUNNING', 'STALE_ASSIGNMENT', 'workflow_identity_conflict', 'coordination_unavailable',
                }:
                    # A losing/uncertain admission owns no execution to finish. In
                    # particular it must never terminate another worker's live row.
                    self.closed = True
                    self.delegation = None
                    self._authorization = ''
                raise

    def activity(self, tool=None, *, phase=None, detail=None, call_id=None, usage=None):
        if isinstance(usage, dict) and isinstance(usage.get('id'), str):
            self._usage_pending[usage['id']] = usage
        phase = phase or ('working' if tool else None)
        now = time.monotonic()
        if tool or phase or usage or now - self._last_touch >= self.heartbeat_interval:
            try:
                from .activity_preview import safe_tool_detail
                payload = {'tool': tool} if tool else {}
                if phase in {'model', 'tool', 'working'}:
                    payload['progress_phase'] = phase
                if detail is not None:
                    payload['tool_detail'] = safe_tool_detail(detail, command=tool == 'Bash')
                if isinstance(call_id, str):
                    payload['tool_call_id'] = call_id[:128]
                pending = next(iter(self._usage_pending.values()), None)
                if pending:
                    payload['model_usage'] = pending
                self.call('touch', **payload)
                if pending and self._usage_pending.get(pending['id']) == pending:
                    self._usage_pending.pop(pending['id'], None)
            except ModelError as error:
                if error.code != 'coordination_unavailable':
                    raise
                self._unreachable_since = self._unreachable_since or now
                if now - self._unreachable_since >= UNREACHABLE_GRACE:
                    raise
                # Observation can continue briefly; guard() still fences every write.
                return
            self._unreachable_since = None
            self._last_touch = now

    def start(self, identity, version, agent_role=None):
        self.identity = identity
        self.runtime = identity.id
        self.version, self.agent_role = version, agent_role or self.actor
        snapshot = self.call('snapshot')['state']
        if not snapshot.get('enabled'):
            raise ModelError('coordination_disabled', 'Channel coordination is disabled')
        if snapshot['workflow_id'] != self.binding['workflow_id']:
            raise ModelError('workflow_identity_conflict', 'Channel workflow identity changed')
        started = self.call('start', workflow_version=version, runtime=identity.id, agent_role=agent_role or self.actor,
                            heartbeat_interval=self.heartbeat_interval,
                            **({'runtime_profile': self.runtime_profile} if self.runtime_profile else {}))
        self.start_acknowledged = True
        if started.get('dispatch'):
            # A late delivery can atomically adopt an unclaimed successor; bind all later calls to it.
            self.binding.update(dispatch_id=started['dispatch']['id'], instance_id=started['dispatch']['instance_id'])
            if self.delegation:
                self.delegation.update(dispatch_id=self.binding['dispatch_id'], instance_id=self.binding['instance_id'])
        self.activity()
        if identity.sandbox == 'workspace-write' and self.recoverable_leases(same_agent=True):
            recovered = json.loads(self.tool('recover_task_lease', {}))
            if recovered.get('leases'):
                self.lease_id = recovered['leases'][0]['id']
        legacy = Path(identity.workspace) / 'workflow/WORKFLOW_VERSION'
        if not identity.ssh_host and legacy.is_file():
            observed = legacy.read_text()[:80].strip()
            if observed != version:
                self.call('gate', id='legacy-workflow-version-' + self.binding['instance_id'],
                    kind='WORKFLOW_VERSION_DIAGNOSTIC', result='failed', metadata={
                        'source': str(legacy), 'observed': observed, 'authoritative': version})
        return self.context()

    def compact_state(self, state, limit=CONTEXT_STATE_LIMIT):
        """The journal as one worker needs it: its task revision, live leases, open questions.

        The full journal (every dispatch, 300 events, 200 metrics, capability history, old
        reviews) used to be sent as the system prompt and as every snapshot tool result;
        it grew with the channel's age and was most of a worker's starting context.
        """
        now = time.time()
        revision = state.get('task_revision')
        instance = self.binding['instance_id']
        leases = {key: {k: lease.get(k) for k in ('id', 'agent_id', 'instance_id', 'repository', 'absolute_paths',
                                                    'mode', 'status', 'expires_at', 'task_revision')}
                  for key, lease in (state.get('leases') or {}).items()
                  if lease.get('status') in {'active', 'revoking', 'recovery_required', 'expired', 'recoverable'}}
        facts = {key: {k: fact.get(k) for k in ('kind', 'repository', 'value', 'source', 'status')}
                 for key, fact in (state.get('facts') or {}).items()
                 if fact.get('task_revision') == revision and fact.get('expires_at', now + 1) > now}
        task = state.get('task') or {}
        view = {
            'workflow_id': state.get('workflow_id'), 'workflow_version': state.get('workflow_version'),
            'agent_instance_id': instance, 'task_family': state.get('task_family'), 'task_revision': revision,
            'state_revision': state.get('revision'),
            'task': {k: task.get(k) for k in ('status', 'phase', 'facts') if task.get(k) is not None},
            'leases': dict(list(leases.items())[-12:]),
            'facts': dict(list(facts.items())[-20:]),
            'decisions': {k: v for k, v in (state.get('decisions') or {}).items() if v.get('status') == 'PENDING'},
            'blockers': {k: v for k, v in (state.get('blockers') or {}).items() if v.get('status') == 'open'},
            'worktrees': {k: v for k, v in (state.get('worktrees') or {}).items()
                          if v.get('instance_id') == instance or v.get('state') in {'active', 'abandoned'}},
            'completed_gates': (state.get('completed_gates') or [])[-20:],
            'active_dispatches': [{k: d.get(k) for k in ('id', 'agent_id', 'status')}
                                  for d in (state.get('dispatches') or {}).values()
                                  if d.get('status') in {'CREATED', 'DISPATCHED', 'STARTED', 'RUNNING'}][-10:],
        }
        text = json.dumps(view, sort_keys=True, default=str)
        for key in ('facts', 'worktrees', 'leases', 'completed_gates', 'active_dispatches'):
            if len(text) <= limit:
                break
            view[key] = {} if isinstance(view[key], dict) else []
            view['omitted'] = view.get('omitted', []) + [key]
            text = json.dumps(view, sort_keys=True, default=str)
        return text[:limit]

    @property
    def previous_checkpoint(self):
        """The latest checkpoint this logical agent left for the current task revision, if any."""
        task = (self._state or {}).get('task') or {}
        checkpoint = (task.get('checkpoints') or {}).get(self.actor)
        return checkpoint if isinstance(checkpoint, dict) else None

    def checkpoint(self, checkpoint):
        """Persist a structured checkpoint in the journal (best effort, never blocks the work)."""
        try:
            self.call('checkpoint', checkpoint=checkpoint)
        except ModelError as error:
            log.warning('Checkpoint not recorded channel=%s dispatch=%s code=%s',
                        self.binding['channel_id'], self.binding['dispatch_id'], error.code)

    def context(self):
        state = self.call('snapshot')['state']
        # The state is supplied separately on every turn; never rely on transcript replay.
        return ('\nCHANNEL RUNTIME CONTEXT (authoritative journal for this task; reported facts retain provenance):\n'
                + self.compact_state(state))

    def recoverable_leases(self, same_agent=False):
        state = self._state or {}
        dispatch = state.get('dispatches', {}).get(self.binding['dispatch_id'], {})
        return [lease for lease in state.get('leases', {}).values()
                if lease.get('status') in {'recoverable', 'expired'}
                and (not same_agent or lease.get('agent_id') == self.actor)
                and (lease.get('task_id'), lease.get('task_revision')) == (dispatch.get('task_id'), dispatch.get('task_revision'))]

    def inspect_workspace(self, repository):
        """Observe on the execution host; preserve every byte of existing work in place."""
        if self.identity.ssh_host:
            raise ModelError('remote_scope_unverified', 'Recovery needs an execution-host observer')
        location = Path(repository).resolve()
        root = Path(self.identity.workspace).resolve()
        registered = (self._state or {}).get('worktrees', {}).get(str(location))
        if not location.is_relative_to(root) and not registered:
            raise ModelError('lease_outside_workspace', 'Cannot inspect an unrelated worktree')
        def git(*args):
            result = subprocess.run(['git', '-c', 'core.fsmonitor=false', '-C', str(location), *args],
                capture_output=True, text=True, timeout=10, env={'PATH': '/usr/local/bin:/usr/bin:/bin',
                'HOME': self.identity.home or self.identity.workspace, 'GIT_OPTIONAL_LOCKS': '0'})
            if result.returncode:
                raise ModelError('worktree_observation_failed', 'Cannot inspect repository before lease recovery')
            return result.stdout.strip()
        git_dir = Path(git('rev-parse', '--absolute-git-dir'))
        operations = [name for name in ('rebase-merge', 'rebase-apply', 'MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD')
                      if (git_dir / name).exists()]
        return {'repository': str(location), 'head': git('rev-parse', 'HEAD'),
                'dirty': bool(git('status', '--porcelain')), 'operation_in_progress': operations,
                'preserved': True, 'observed_at': time.time(), 'source': 'runtime_git_status'}

    def tool(self, operation, data):
        if operation not in TOOL_OPERATIONS or not isinstance(data, dict):
            raise ModelError('coordination_operation_forbidden', 'Unknown or privileged coordination operation')
        data = dict(data)
        if operation == 'recover_task_lease' and self.identity.sandbox != 'workspace-write':
            raise ModelError('BLOCKED_EXECUTION_CAPABILITY', 'Read-only runtime cannot recover write leases')
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
        if operation in {'acquire', 'recover_task_lease'}:
            self.call('snapshot')
            from .workspace_fence import mutation_fence
            recoverable = self.recoverable_leases()
            if recoverable:
                with mutation_fence(self):
                    repositories = sorted({lease.get('physical_repository', lease['repository']) for lease in recoverable})
                    observations = [self.inspect_workspace(repository) for repository in repositories]
                    data['workspace_observation'] = {'repositories': observations, 'preserved': True}
                    result = self.call(operation, **data)
            else:
                result = self.call(operation, **data)
        else:
            result = self.call(operation, **data)
        if operation == 'acquire':
            self.lease_id = result['lease']['id']
        if operation == 'release' and data.get('lease_id', self.lease_id) == self.lease_id:
            # The agent gave its own lease back. Heartbeating it afterwards fails, and that
            # failure must not read as a revocation that stops the turn.
            self.lease_id = None
        if operation == 'privilege':
            self.privilege = {k: data[k] for k in ('privilege_level', 'scope', 'decision_id') if k in data}
        if operation != 'snapshot':
            result = {k: v for k, v in result.items() if k != 'state'} | {'state_revision': self._state['revision']}
        else:
            return self.compact_state(result['state'])
        return json.dumps(result, sort_keys=True)

    def guard(self, name, arguments):
        if name not in {'Bash', 'Write', 'Edit'}:
            return
        if name == 'Bash':
            self.check_publication(str(arguments.get('command', '')))
        if self.identity.sandbox != 'workspace-write':
            command = str(arguments.get('command', ''))
            if name == 'Bash' and self.verification:
                from .runtime_profiles import is_verification_command
                if is_verification_command(command):
                    return
            if name != 'Bash' or shell_effect(command) != 'READ_ONLY' or re.search(r'[<>$`()\n]', command):
                raise ModelError('BLOCKED_EXECUTION_CAPABILITY', 'Read-only runtime permits observation and test commands only'
                                 + ('' if self.verification else '; this dispatch has no verify profile'))
            return
        if name == 'Bash' and shell_effect(str(arguments.get('command', ''))) == 'READ_ONLY':
            return
        if self.privilege and self.privilege['privilege_level'] != 'scoped':
            self.call('privilege', **self.privilege)
        if not self.lease_id:
            raise ModelError('WRITE_LEASE_REQUIRED', 'Acquire a covering Coordination write lease before executing')
        if self.identity.ssh_host:
            raise ModelError('remote_scope_unverified', 'Remote execution scopes are not verified')
        # Bash runs in a leading `cd <dir>` or `--prefix <dir>`, otherwise the workspace root.
        # A lease on that directory is enough. Do not demand a lease on the whole workspace.
        if name == 'Bash':
            target = bash_workdir(str(arguments.get('command', '')), self.identity.workspace)
        else:
            target = str((Path(self.identity.workspace) / arguments.get('file_path', '')).resolve())
        checked = self.call('check', path=target)
        self.lease_id = checked.get('lease', {}).get('id', self.lease_id)
        state = checked['state']
        for tree in state['worktrees'].values():
            foreign = tree['instance_id'] != self.binding['instance_id']
            affected = name == 'Bash' and (tree['state'] == 'abandoned' or tree.get('dirty'))
            affected = affected or Path(target).is_relative_to(tree['path'])
            if foreign and affected:
                raise ModelError('WORKTREE_OWNED', 'Explicit worktree adoption/reconciliation required')
        self.last_check = time.monotonic()

    def _command_dir(self, command):
        try:
            return bash_workdir(command, self.identity.workspace)
        except ModelError:
            return None

    def check_publication(self, command):
        """Refuse a push that would publish commits of another task (PUBLICATION_SCOPE_CONFLICT)."""
        from . import publication_scope as scope
        if scope.COMMIT_MAKERS.search(command or ''):
            # Remember HEAD so the commits this command creates are attributed to this task.
            folder = self._command_dir(command)
            if folder:
                self._heads_before = {**getattr(self, '_heads_before', {}),
                                      folder: (scope.git(folder, 'rev-parse', 'HEAD'), time.time())}
        if not scope.PUSH.search(command or '') or not self.identity:
            return
        folder = self._command_dir(command)
        if not folder:
            raise ModelError('PUBLICATION_SCOPE_UNVERIFIED', 'git push needs a literal working directory')
        state = None
        for repo, remote, src, dst in scope.push_targets(command, folder):
            pending = scope.unpublished(repo, remote, src, dst)
            if not pending or not pending['commits']:
                continue
            state = state or self.call('snapshot')['state']
            foreign = scope.foreign_commits(pending['commits'], state, self.assignment)
            if not foreign:
                continue
            try:
                self.call('publication_scope', repository=str(repo), target=f'{remote}/{dst}',
                          candidate=pending['candidate'], commits=foreign)
            except ModelError:
                pass    # the refusal below does not depend on recording it
            log.warning('Publication scope conflict channel=%s dispatch=%s target=%s/%s foreign=%d',
                        self.binding['channel_id'], self.binding['dispatch_id'], remote, dst, len(foreign))
            raise ModelError('PUBLICATION_SCOPE_CONFLICT', scope.conflict_message(repo, foreign, remote, dst))

    def after_bash(self, command):
        """Attribute commits a successful command created to this dispatch's task revision."""
        from . import publication_scope as scope
        if not scope.COMMIT_MAKERS.search(command or ''):
            return
        folder = self._command_dir(command)
        if not folder:
            return
        before, started = getattr(self, '_heads_before', {}).pop(folder, (None, None))
        head = scope.git(folder, 'rev-parse', 'HEAD')
        if not head or head == before or started is None:
            return
        # Only commits this command wrote: a pull/merge also brings in other people's commits,
        # which keep their older committer time and are never attributed to this task.
        listing = scope.git(folder, 'log', '--no-merges', '--max-count=20', '--format=%H %ct',
                            f'{before}..{head}' if before else '-1')
        created = [line.split()[0] for line in (listing or '').splitlines()
                   if len(line.split()) == 2 and int(line.split()[1]) >= int(started) - 5]
        repository = scope.git(folder, 'rev-parse', '--show-toplevel') or folder
        for sha in created[:20]:
            try:
                self.call('commit', sha=sha, repository=repository)
            except ModelError as error:
                log.warning('Commit ownership not recorded sha=%s code=%s', sha[:12], error.code)
                return

    def revoked(self):
        if self.closed or time.monotonic() - self.last_check < 2:
            return False
        self.last_check = time.monotonic()
        try:
            pending = next(iter(self._usage_pending.values()), None)
            self.call('heartbeat' if self.lease_id else 'touch', **({'model_usage': pending} if pending else {}))
            if pending and self._usage_pending.get(pending['id']) == pending:
                self._usage_pending.pop(pending['id'], None)
            self._unreachable_since = None
            return False
        except ModelError as error:
            if error.code == 'coordination_unavailable':
                # One failed request is not a revocation. Writes stay fenced by guard();
                # only a journal that stays unreachable stops the turn.
                self._unreachable_since = self._unreachable_since or self.last_check
                if self.last_check - self._unreachable_since < UNREACHABLE_GRACE:
                    return False
            self.revocation = error.code
            log.warning('Coordination lease lost channel=%s lease=%s code=%s',
                        self.binding['channel_id'], self.lease_id, error.code)
            return True

    def observed(self, name, ok):
        capability = {'Read': 'fs_read', 'Grep': 'fs_read', 'Glob': 'fs_read',
                      'Write': 'fs_write', 'Edit': 'fs_write', 'Bash': 'shell_exec'}.get(name)
        if ok and capability and (capability, ok) not in self.tool_capabilities:
            try:
                self.call('capability', agent_id=self.actor, runtime=self.runtime,
                    target=self.identity.workspace, capability=capability,
                    status='proven' if ok else 'failed', source='devagents_tool_result:' + name,
                    credential_ref='runtime:' + self.identity.id, privilege_level='scoped')
            except ModelError as error:
                now = time.monotonic()
                if error.code != 'coordination_unavailable':
                    raise
                self._unreachable_since = self._unreachable_since or now
                if now - self._unreachable_since >= UNREACHABLE_GRACE:
                    raise
                return  # Leave proof uncached so the next successful observation retries.
            self.tool_capabilities.add((capability, ok))

    def ask(self, arguments):
        if any(l['instance_id'] == self.binding['instance_id'] and l['mode'] == 'write' and l['status'] in {'active', 'revoking'} for l in self.call('snapshot')['state']['leases'].values()):
            raise ModelError('RELEASE_WRITE_LEASE_BEFORE_DECISION', 'Reconcile worktrees and release writes before waiting for user')
        question = str(arguments.get('question', '')).strip()
        if not question:
            raise ModelError('invalid_question', 'Question is required')
        if is_status_chatter(question):
            return question
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

    def completion_payload(self, status, error_code=None, result=None):
        outcome = ('blocked' if self.blocked or status == 'blocked' else
                   'completed' if status == 'completed' else 'failed')
        data = {'status': outcome, 'error_code': error_code,
                'executor_stopped': (status not in {'interrupted', 'timed_out'}
                                     or error_code in {'model_request_timed_out', 'service_restarted', 'tool_timed_out'})
                                    and 'uncertain' not in str(error_code),
                'failed_before_start': not self.start_acknowledged}
        if self._usage_pending:
            data['usage_receipts'] = list(self._usage_pending.values())[-600:]
        if result is not None:
            data['result'] = result
        return {'binding': self.binding, 'actor': self.actor, 'delegation': self.delegation, 'data': data}

    def finish(self, status, error_code=None, result=None):
        if self.closed:
            return
        payload = self.completion_payload(status, error_code, result)
        if self.delegation and self.completion_store:
            with self.completion_store.connection() as db:
                db.execute('INSERT OR REPLACE INTO coordination_completions VALUES(?,?)',
                           (self.binding['dispatch_id'], json.dumps(payload)))
        self.call('finish', **payload['data'])
        if self.delegation and self.completion_store:
            with self.completion_store.connection() as db:
                db.execute('DELETE FROM coordination_completions WHERE dispatch_id=? AND payload=?',
                           (self.binding['dispatch_id'], json.dumps(payload)))
        self.closed = True
        self._authorization = ''
