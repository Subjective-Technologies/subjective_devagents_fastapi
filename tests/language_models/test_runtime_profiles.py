"""Runtime capability routing: the assigned action, not the role name, selects the runtime.

Production 2026-11-15: Coder ran with Node 22, Integration got /usr/bin/node v18 and a
read-only shell that refused `npm run build`/`tsc`, although Manager assigned tests, tsc
and build, and the project's CI pins Node 22.
"""
from dataclasses import replace
import json
import os
import stat
import time

import pytest

from subjective_devagents_fastapi import runtime_profiles as rp
from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import ApiAgent, Identity, ModelRequest
from subjective_devagents_fastapi.language_models.contracts import ModelError
from test_api_agent import ScriptedProxy, completion, tool_call
from test_channel_coordination import Journal


def fake_node(root, version):
    folder = root / version / 'bin'
    folder.mkdir(parents=True)
    binary = folder / 'node'
    binary.write_text(f'#!/bin/sh\necho fake-{version}\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return folder


@pytest.fixture
def toolchains(tmp_path, monkeypatch):
    nodes = tmp_path / 'nvm' / 'versions' / 'node'
    bins = {v: fake_node(nodes, v) for v in ('v18.20.4', 'v22.23.4', 'v24.20.0')}
    monkeypatch.setenv('SUBJECTIVE_RUNTIME_NODE_DIRS', str(nodes))
    monkeypatch.setenv('NVM_DIR', str(tmp_path / 'no-nvm'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    rp._PINS.clear()
    rp.node_toolchains(refresh=True)
    yield bins
    rp._PINS.clear()
    rp._TOOLCHAINS.clear()


def test_semver_constraints_map_to_installed_majors(toolchains):
    pick = lambda spec: rp.select_node(rp._major_constraints(spec)).major
    assert pick('22') == 22 and pick('v22.1.0') == 22 and pick('^22.0.0') == 22 and pick('22.x') == 22
    assert pick('>=18 <23') == 22 and pick('>=20') == 24 and pick('18 || 22') == 22
    assert rp._major_constraints('lts/*') is None
    assert rp.select_node(rp._major_constraints('26')) is None


def test_project_pin_precedence_nvmrc_engines_ci(tmp_path, toolchains):
    repo = tmp_path / 'repo'
    (repo / '.git').mkdir(parents=True)
    (repo / '.github' / 'workflows').mkdir(parents=True)
    (repo / '.github' / 'workflows' / 'ci.yml').write_text("steps:\n  - uses: actions/setup-node@v4\n    with:\n      node-version: '22'\n")
    web = repo / 'web' / 'src'
    web.mkdir(parents=True)
    alternatives, source = rp.project_node_pin(web, repo)
    assert rp.select_node(alternatives).major == 22 and source.endswith('ci.yml#node-version')
    (repo / 'web' / 'package.json').write_text(json.dumps({'engines': {'node': '>=24'}}))
    rp._PINS.clear()
    alternatives, source = rp.project_node_pin(web, repo)
    assert rp.select_node(alternatives).major == 24 and source.endswith('package.json#engines.node')
    (repo / 'web' / '.nvmrc').write_text('18\n')
    rp._PINS.clear()
    alternatives, source = rp.project_node_pin(web, repo)
    assert rp.select_node(alternatives).major == 18 and source.endswith('.nvmrc')


def test_pinned_node_is_first_on_the_agent_path(tmp_path, toolchains, monkeypatch):
    """Coder and verifier both get the project's Node, not /usr/bin/node."""
    workspace = tmp_path / 'workspace'
    (workspace / 'dashboard').mkdir(parents=True)
    (workspace / 'dashboard' / '.nvmrc').write_text('22')
    identity = Identity('runtime', 'codex', model='m', strategy='api_agent', workspace=str(workspace),
                        home=str(tmp_path), sandbox='workspace-write', api_base_url='http://unused', api_key_env='TEST_KEY')
    monkeypatch.setenv('TEST_KEY', 'fixture')
    with ScriptedProxy([completion(tool_calls=[tool_call('1', 'Bash', command='cd dashboard && node --version')]),
                        completion(tool_calls=[tool_call('2', 'Bash', command='node --version')]),
                        completion(content='done')]) as proxy:
        result = ApiAgent().run(ModelRequest('t', replace(identity, api_base_url=proxy.url), 'check', timeout=10),
                                lambda *_: None, lambda: False)
    assert result.status == 'completed'
    outputs = [m['content'] for m in proxy.requests[-1][2]['messages'] if m['role'] == 'tool']
    assert outputs[0].strip() == 'fake-v22.23.4'
    # No pin at the workspace root: the host default stays as it was.
    assert 'fake-' not in outputs[1]


def test_requirements_come_from_the_latest_instruction(tmp_path, toolchains):
    req = rp.requirements_for('History mentions on Node 20 long ago.' + 'x' * 7000 + '@Integration Agent: run tests, run tsc and build on Node 22')
    assert req.test_exec and req.build_exec and req.node_majors == {22}
    assert rp.requirements_for('run Node 22 tests').node_majors == {22}
    # Quoted history is not an instruction.
    assert rp.requirements_for('Integration got Node 18 and Node 20 is EOL').node_majors == set()
    workspace = tmp_path / 'ws'
    (workspace / 'clients' / 'web').mkdir(parents=True)
    (workspace / 'clients' / 'web' / '.nvmrc').write_text('24')
    req = rp.requirements_for('verify clients/web please', workspace)
    assert req.node_majors == {24} and req.node_sources[24].endswith('.nvmrc')


def test_capability_mismatch_selects_the_compatible_profile(toolchains):
    """Runtime A: Node 18, read-only. Runtime B: Node 22 with build. The task needs B."""
    a = rp.RuntimeProfile('A', 'read-only', ('v18.20.4',))
    b = rp.RuntimeProfile('B', 'read-only', ('v22.23.4',), test_exec=True, build_exec=True)
    need = rp.Requirements(test_exec=True, build_exec=True, node_majors={22})
    assert rp.select_profile(need, [a, b]).runtime_id == 'B'


def test_no_compatible_runtime_is_an_immediate_explicit_blocker(toolchains):
    need = rp.Requirements(test_exec=True, node_majors={26})
    with pytest.raises(ModelError) as error:
        rp.select_profile(need, rp.available_profiles('id', 'read-only', verify_allowed=True))
    assert error.value.code == 'BLOCKED_NO_COMPATIBLE_RUNTIME' and 'Node 26 not installed' in str(error.value)


def test_least_privileged_profile_wins(toolchains):
    profiles = rp.available_profiles('id', 'read-only', verify_allowed=True)
    assert rp.select_profile(rp.Requirements(), profiles).runtime_id == 'id:read'
    assert rp.select_profile(rp.Requirements(build_exec=True), profiles).runtime_id == 'id:verify'


@pytest.mark.parametrize('command', [
    'npm test', 'npm run build', 'npx tsc --noEmit', 'tsc -p .', 'cd web && npm run build && npm test',
    'npm run typecheck 2>&1 | tail -n 40', 'pnpm build', 'yarn test', 'python -m pytest -q', 'go test ./...',
    'make test', '. ~/.nvm/nvm.sh && nvm use 22 && npm test', 'CI=1 npx vitest run', 'npm run test:unit',
])
def test_verification_commands_are_recognized(command):
    assert rp.is_verification_command(command)


@pytest.mark.parametrize('command', [
    'npm install', 'npm ci', 'npm run deploy', 'rm -rf dist', 'npm test > out.txt', 'npm test; rm x',
    'npm test || rm x', 'npm test & sleep 1', 'npx tsc $(echo)', 'npm run build | tee log', 'cd web',
    'git commit -am x', 'npx wrangler deploy', 'npm test | sed -i s/a/b/ f',
])
def test_non_verification_commands_are_rejected(command):
    assert not rp.is_verification_command(command)


def verify_session(tmp_path, verification):
    root = tmp_path / 'workspace'
    root.mkdir(exist_ok=True)
    identity = Identity('runtime', 'codex', model='t', strategy='api_agent', workspace=str(root), home=str(tmp_path),
                        sandbox='read-only', api_base_url='http://unused', api_key_env='TEST_KEY')
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'ai_integration:1', 'workflow_id': 'workflow'},
                             'ai_integration', 'Bearer test-only', client=Journal())
    session.verification = verification
    session.start(identity, 'published', 'integration_agent')
    return session


def test_verify_profile_runs_build_and_tsc_but_never_edits(tmp_path):
    session = verify_session(tmp_path, True)
    for command in ('npm run build', 'npx tsc --noEmit', 'cd web && npm run build && npm test 2>&1 | tail -n 20'):
        session.guard('Bash', {'command': command})
    for command in ('rm -rf src', 'echo x > src/a.ts', 'npm install left-pad', 'git commit -am x'):
        with pytest.raises(ModelError):
            session.guard('Bash', {'command': command})
    with pytest.raises(ModelError):
        session.guard('Write', {'file_path': 'src/a.ts'})


def test_plain_read_profile_still_blocks_build(tmp_path):
    session = verify_session(tmp_path, False)
    with pytest.raises(ModelError) as error:
        session.guard('Bash', {'command': 'npm run build'})
    assert 'no verify profile' in str(error.value)


def test_runtime_profile_is_reported_at_start(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    journal = Journal()
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'i', 'workflow_id': 'workflow'},
                             'ai_integration', 'Bearer x', client=journal)
    session.runtime_profile = {'runtime_id': 'r:verify', 'sandbox': 'read-only'}
    identity = Identity('runtime', 'codex', model='t', strategy='api_agent', workspace=str(root), home=str(tmp_path),
                        sandbox='read-only', api_base_url='http://unused', api_key_env='K')
    session.start(identity, 'published', 'integration_agent')
    assert next(c for c in journal.calls if c['op'] == 'start')['runtime_profile']['runtime_id'] == 'r:verify'


def test_select_runtime_gives_integration_the_verify_profile(tmp_path, toolchains):
    from subjective_devagents_fastapi.collaboration import select_runtime
    identity = Identity('runtime', 'claude', model='t', strategy='api_agent', workspace=str(tmp_path), home=str(tmp_path),
                        sandbox='workspace-write', api_base_url='http://unused', api_key_env='K')
    profile, requirements = select_runtime({'mode': 'read', 'role_file': 'agents/roles/integration_agent.md'}, identity,
                                           'Manager: @Integration Agent run tests, run tsc, run build on Node 22')
    assert profile.runtime_id.endswith(':verify') and profile.build_exec and requirements.node_majors == {22}
    writer, _ = select_runtime({'mode': 'write', 'role_file': 'agents/roles/coder_agent.md'}, identity, 'Implement it')
    assert writer.sandbox == 'workspace-write'
    with pytest.raises(ModelError) as error:
        select_runtime({'mode': 'read', 'role_file': 'agents/roles/integration_agent.md'}, identity, 'run tests on Node 26')
    assert error.value.code == 'BLOCKED_NO_COMPATIBLE_RUNTIME'
