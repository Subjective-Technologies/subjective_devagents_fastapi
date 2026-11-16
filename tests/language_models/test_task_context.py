"""Bounded, task-scoped worker context (production 2026-11-16: 13 workers died of context_too_large).

The scenario: a three-repo task in a monorepo workspace with many unrelated dirty repositories.
Manager's plan is concrete; the Coder reads files, edits, runs dashboard and pipelines suites, and
reaches its context budget while the UML build remains. It must hand off with a checkpoint that
holds exactly the task's state and repositories, before any request nears the budget.
"""
import json
import subprocess
from dataclasses import replace

import pytest

from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import ApiAgent, Identity, ModelRequest
from subjective_devagents_fastapi.language_models import task_context as tc
from test_api_agent import ScriptedProxy, completion, tool_call
from test_channel_coordination import Journal


def git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


def make_repo(path, files=()):
    path.mkdir(parents=True)
    git(path, 'init', '-q', '-b', 'main')
    for name, body in files:
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(body)
    (path / 'README').write_text('x\n')
    git(path, '-c', 'user.email=t@x.invalid', '-c', 'user.name=t', 'add', '.')
    git(path, '-c', 'user.email=t@x.invalid', '-c', 'user.name=t', 'commit', '-qm', 'base')
    return path


@pytest.fixture
def monorepo(tmp_path):
    root = make_repo(tmp_path / 'subjective')
    dashboard = make_repo(root / 'clients/web/subjective_web_dashboard',
                          [(f'src/file{n}.ts', ('export const value = %d;\n' % n) * 400) for n in range(12)])
    pipelines = make_repo(root / 'clients/web/subjective_web_pipelines', [('src/import.ts', 'export {};\n')])
    uml = make_repo(root / 'clients/tools/subjective_uml_web', [('src/platform/workflow.js', 'export {};\n')])
    for n in range(50):   # unrelated dirty repositories, like the production superproject
        other = make_repo(root / f'unrelated/repo{n:02d}')
        (other / 'README').write_text('dirty\n')
    (root / 'workflow.md').write_text('dirty superproject file\n')
    return root, dashboard, pipelines, uml


class RecordingJournal(Journal):
    def __call__(self, body):
        if body['op'] == 'checkpoint':
            self.calls.append(body)
            return {'state': self.state}
        return super().__call__(body)


def coder_session(root):
    identity = Identity('runtime', 'codex', model='t', strategy='api_agent', workspace=str(root), home=str(root.parent),
                        sandbox='workspace-write', api_base_url='http://unused', api_key_env='TEST_KEY')
    journal = RecordingJournal()
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'ai_coder:1', 'workflow_id': 'workflow'},
                             'ai_coder', 'Bearer test-only', client=journal)
    session.start(identity, 'published', 'coder_agent')
    session.lease_id = 'held'
    return identity, session, journal


def test_production_scenario_hands_off_with_a_task_scoped_checkpoint(monorepo, monkeypatch):
    root, dashboard, pipelines, uml = monorepo
    monkeypatch.setenv('TEST_KEY', 'fixture')
    monkeypatch.setenv('SUBJECTIVE_AGENT_CONTEXT_TOKENS', '14000')
    identity, session, journal = coder_session(root)
    script = [completion(content='Plan is concrete; implementing.', tool_calls=[
        tool_call('r0', 'Read', file_path=str(dashboard / 'src/file0.ts'))])]
    script += [completion(tool_calls=[tool_call(f'r{n}', 'Read', file_path=str(dashboard / f'src/file{n}.ts'))]) for n in range(1, 11)]
    script += [
        completion(content='Added .xmi to TEXT_EXTENSIONS.', tool_calls=[tool_call('e1', 'Edit', file_path=str(pipelines / 'src/import.ts'),
                                                                                   old_string='export {};', new_string='export const xmi = true;')]),
        completion(tool_calls=[tool_call('t1', 'Bash', command=f'cd {dashboard} && echo " Test Files  127 passed (127)" && '
                                                                'echo "      Tests  920 passed (920)" # npx vitest run')]),
        completion(content='Dashboard 920/920 pass.', tool_calls=[tool_call('t2', 'Bash', command=f'cd {pipelines} && echo "398 passed" # npm test')]),
        completion(content='Pipelines 398/398 pass. Now run the UML build.', tool_calls=[
            tool_call('b1', 'Bash', command=f'cd {uml} && echo "error: Cannot find module workflow-bridge" && exit 2 # npm run build')]),
    ] + [completion(content=f'Reasoning about the UML build failure, attempt {n}: ' + 'analysis ' * 450,
                    tool_calls=[tool_call(f'x{n}', 'Read', file_path=str(uml / 'src/platform/workflow.js'))]) for n in range(30)]
    with ScriptedProxy(script) as proxy:
        result = ApiAgent().run(ModelRequest('t', replace(identity, api_base_url=proxy.url), 'Implement the plan',
                                             timeout=30, coordination=session), lambda *_: None, lambda: False)
    assert result.status == 'failed' and result.error_code == 'context_handoff'
    checkpoint = result.checkpoint
    assert checkpoint['phase'] == 'TESTING'
    assert any('920 passed' in entry for entry in checkpoint['tests_passed'])
    assert any('398 passed' in entry for entry in checkpoint['tests_passed'])
    assert checkpoint['current_failure']['exit_code'] == 2 and 'workflow-bridge' in checkpoint['current_failure']['excerpt']
    assert str(pipelines / 'src/import.ts') in checkpoint['files_changed']
    # Task-scoped workspace: exactly the three repositories, never the superproject or the 50 others.
    assert set(checkpoint['repositories']) == {str(dashboard), str(pipelines), str(uml)}
    assert set(checkpoint['workspace_state']) == set(checkpoint['repositories'])
    rendered = tc.render_checkpoint(checkpoint)
    assert 'unrelated/repo' not in rendered and 'workflow.md' not in rendered
    assert checkpoint['next_action'].startswith('Fix and re-run') and 'npm run build' in checkpoint['next_action']
    # Checkpoints were persisted during the run, before the handoff.
    assert [c for c in journal.calls if c['op'] == 'checkpoint']
    # No request ever came close to the budget: compaction then handoff kept it bounded.
    budget_chars = 14000 * tc.CHARS_PER_TOKEN
    sizes = [sum(len(tc.text_of(m.get('content'))) for m in request[2]['messages']) for request in proxy.requests]
    assert max(sizes) <= 0.9 * budget_chars


def test_provider_context_rejection_is_not_resent(monorepo, monkeypatch):
    root, dashboard, _, _ = monorepo
    monkeypatch.setenv('TEST_KEY', 'fixture')
    identity, session, _ = coder_session(root)
    with ScriptedProxy([completion(tool_calls=[tool_call('r', 'Read', file_path=str(dashboard / 'src/file0.ts'))]),
                        (400, 0, {'error': {'message': 'prompt is too long: 212345 tokens > 200000 maximum'}}),
                        completion(content='must never be requested')]) as proxy:
        result = ApiAgent().run(ModelRequest('t', replace(identity, api_base_url=proxy.url), 'work', timeout=30,
                                             coordination=session), lambda *_: None, lambda: False)
    assert result.error_code == 'context_too_large'
    assert len(proxy.requests) == 2                    # the rejected payload was not retried
    assert result.checkpoint and 'CONTINUATION_SUMMARY' in result.text


def test_old_tool_outputs_become_digests_and_pairing_is_kept():
    log = '\n'.join(f'ok {n} test case' for n in range(5000)) + '\n Test Files  127 passed (127)\n      Tests  920 passed (920)\n'
    messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]
    for n in range(10):
        messages.append({'role': 'assistant', 'content': None,
                         'tool_calls': [tool_call(f'c{n}', 'Bash', command='cd web && npx vitest run')]})
        messages.append({'role': 'tool', 'tool_call_id': f'c{n}', 'content': log})
    before = tc.estimate_tokens(messages)
    compacted, dropped = tc.compact_tool_results(messages, 2, keep_recent=3)
    assert compacted == 7 and dropped > 0 and tc.estimate_tokens(messages) < before / 2
    digest = messages[3]['content']
    assert '920 passed, 0 failed' in digest and 'exit 0' in digest and len(digest) < 600
    assert messages[-1]['content'] == log                          # the newest stay whole
    assert [m['tool_call_id'] for m in messages if m['role'] == 'tool'] == [f'c{n}' for n in range(10)]


def test_failure_fingerprint_ignores_timestamps_and_counts_repeats():
    a = tc.fingerprint('npm run build', 2, 'Error at 12:01:33 (1532ms): Cannot find module x 0x7ffe12')
    b = tc.fingerprint('npm run build', 2, 'Error at 12:09:58 (1610ms): Cannot find module x 0x7ffe99')
    c = tc.fingerprint('npm run build', 2, 'Error: type mismatch in workflow.js')
    assert a == b != c
    messages = [{'role': 'user', 'content': 'u'}]
    for n in range(3):
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': [tool_call(f'b{n}', 'Bash', command='npm run build')]})
        messages.append({'role': 'tool', 'tool_call_id': f'b{n}',
                         'content': f'Error at 12:0{n}:00: Cannot find module x\n[exit code 2]'})
    checkpoint = tc.build_checkpoint(messages, 0, set(), '/nonexistent', reason='t', repositories=False)
    assert len(checkpoint['failures']) == 1 and checkpoint['failures'][0]['count'] == 3


def test_successor_resumes_the_phase_and_carries_established_results():
    previous = {'phase': 'TESTING', 'tests_passed': ['npx vitest run (920 passed, 0 failed)'],
                'files_changed': ['/ws/a.ts'], 'work_completed': ['implemented the bridge'], 'tool_rounds': 25}
    messages = [{'role': 'user', 'content': 'continue'},
                {'role': 'assistant', 'content': 'Checking git status.', 'tool_calls': [tool_call('g', 'Bash', command='git status --short')]},
                {'role': 'tool', 'tool_call_id': 'g', 'content': ' M a.ts'}]
    checkpoint = tc.build_checkpoint(messages, 0, set(), '/nonexistent', reason='t', previous=previous, rounds=1,
                                     repositories=False)
    assert checkpoint['phase'] == 'TESTING'                       # looking around is not a return to scoping
    assert checkpoint['tests_passed'] == previous['tests_passed'] and '/ws/a.ts' in checkpoint['files_changed']
    assert checkpoint['tool_rounds'] == 26 and 'implemented the bridge' in checkpoint['work_completed']


def test_journal_view_is_bounded_whatever_the_channel_age():
    identity = Identity('runtime', 'codex', model='t', strategy='api_agent', workspace='/tmp', home='/tmp',
                        sandbox='workspace-write', api_base_url='http://unused', api_key_env='K')
    journal = Journal()
    journal.state.update({
        'facts': {f'f{n}': {'kind': 'local_sha', 'value': 'v' * 300, 'task_revision': '1', 'expires_at': 9e12} for n in range(400)},
        'capabilities': {f'c{n}': {'agent_id': 'a', 'capability': 'x' * 200} for n in range(500)},
        'dispatches': {f'd{n}': {'agent_id': 'ai_x', 'status': 'COMPLETED', 'activity_rows': ['r' * 500] * 20} for n in range(300)},
        'events': [{'kind': 'X' * 50}] * 300, 'reviews': {f'r{n}': {'x': 'y' * 400} for n in range(200)}})
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'i', 'workflow_id': 'workflow'},
                             'ai_coder', 'Bearer x', client=journal)
    session.start(identity, 'published', 'coder_agent')
    full = len(json.dumps(journal.state))
    assert full > 500_000
    context = session.context()
    assert len(context) < 8_500
    assert len(session.tool('snapshot', {})) <= 8_000


def test_long_read_loops_stay_bounded_without_handoff(monorepo, monkeypatch):
    """Interpreted outputs are digested, so 40 large reads finish inside the budget."""
    root, dashboard, _, _ = monorepo
    monkeypatch.setenv('TEST_KEY', 'fixture')
    monkeypatch.setenv('SUBJECTIVE_AGENT_CONTEXT_TOKENS', '20000')
    identity, session, _ = coder_session(root)
    script = [completion(tool_calls=[tool_call(f'r{n}', 'Read', file_path=str(dashboard / f'src/file{n % 12}.ts'))])
              for n in range(40)] + [completion(content='IMPLEMENTATION_COMPLETE: read everything')]
    with ScriptedProxy(script) as proxy:
        result = ApiAgent().run(ModelRequest('t', replace(identity, api_base_url=proxy.url), 'Read all', timeout=30,
                                             coordination=session), lambda *_: None, lambda: False)
    assert result.status == 'completed'
    sizes = [sum(len(tc.text_of(m.get('content'))) for m in request[2]['messages']) for request in proxy.requests]
    assert max(sizes) <= 0.9 * 20000 * tc.CHARS_PER_TOKEN
    assert sizes[-1] < sizes[15] * 1.5          # flat, not linear in the number of reads
