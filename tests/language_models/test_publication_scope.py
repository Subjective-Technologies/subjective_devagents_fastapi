"""A push publishes only the current task's commits.

remote main = A, task X commit = B (created by this dispatch), unrelated unpublished
commit C (another task / another person) on top: pushing C must stop with
PUBLICATION_SCOPE_CONFLICT unless Manager explicitly named C, and a clean cherry-picked
branch of B alone publishes fine.
"""
import subprocess

import pytest

from subjective_devagents_fastapi.channel_coordination import ChannelSession
from subjective_devagents_fastapi.language_models import Identity
from subjective_devagents_fastapi.language_models.contracts import ModelError
from subjective_devagents_fastapi import publication_scope as scope
from test_channel_coordination import Journal


class OwnershipJournal(Journal):
    def __call__(self, body):
        if body['op'] == 'commit':
            self.calls.append(body)
            owners = self.state.setdefault('commit_owners', {})
            owners.setdefault(body['sha'], {'task_revision': self.state['task_revision'], 'dispatch_id': body['dispatch_id']})
            return {'owner': owners[body['sha']], 'state': self.state}
        return super().__call__(body)


def run(cwd, *args):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / 'remote.git'
    run(tmp_path, 'git', 'init', '-q', '--bare', '-b', 'main', str(remote))
    work = tmp_path / 'workspace'
    run(tmp_path, 'git', 'clone', '-q', str(remote), str(work))
    for key, value in (('user.email', 'agent@example.invalid'), ('user.name', 'Agent'), ('commit.gpgsign', 'false')):
        run(work, 'git', 'config', key, value)
    (work / 'a.txt').write_text('a\n')
    run(work, 'git', 'add', '.')
    run(work, 'git', 'commit', '-qm', 'A: base')
    run(work, 'git', 'push', '-q', 'origin', 'main')
    return work


def session_for(work, journal, assignment='Push the verified fix.'):
    identity = Identity('runtime', 'codex', model='t', strategy='api_agent', workspace=str(work), home=str(work.parent),
                        sandbox='workspace-write', api_base_url='http://unused', api_key_env='K')
    session = ChannelSession({'channel_id': 'c', 'dispatch_id': 'd', 'instance_id': 'ai_git:1', 'workflow_id': 'workflow'},
                             'ai_git', 'Bearer test-only', client=journal)
    session.start(identity, 'published', 'git_agent')
    session.assignment = assignment
    return session


def commit_as_task(session, work, name):
    command = f'git commit -qam "{name}"'
    (work / 'a.txt').write_text((work / 'a.txt').read_text() + name + '\n')
    session.check_publication(command)          # guard() runs this before the command
    run(work, 'git', 'commit', '-qam', name)
    session.after_bash(command)                 # and this after it succeeded
    return run(work, 'git', 'rev-parse', 'HEAD')


def test_unrelated_ancestry_stops_the_push(repo):
    journal = OwnershipJournal()
    session = session_for(repo, journal)
    task_commit = commit_as_task(session, repo, 'B: task X fix')
    assert journal.state['commit_owners'][task_commit]['task_revision'] == '1'
    (repo / 'other.txt').write_text('c\n')
    run(repo, 'git', 'add', 'other.txt')
    run(repo, 'git', 'commit', '-qm', 'C: TAGGY-3515 unrelated menu change')   # not made by this task
    unrelated = run(repo, 'git', 'rev-parse', 'HEAD')
    with pytest.raises(ModelError) as error:
        session.guard('Bash', {'command': 'git push origin main'})
    assert error.value.code == 'PUBLICATION_SCOPE_CONFLICT'
    assert unrelated[:12] in str(error.value) and task_commit[:12] not in str(error.value)
    recorded = next(c for c in journal.calls if c['op'] == 'publication_scope')
    assert [c['sha'] for c in recorded['commits']] == [unrelated] and recorded['target'] == 'origin/main'
    assert run(repo, 'git', 'ls-remote', 'origin', 'main').split()[0] != unrelated   # nothing was pushed


def test_clean_cherry_picked_branch_publishes(repo):
    journal = OwnershipJournal()
    session = session_for(repo, journal)
    task_commit = commit_as_task(session, repo, 'B: task X fix')
    run(repo, 'git', 'commit', '-q', '--allow-empty', '-m', 'C: someone else')
    # The clean publication branch: origin/main + this task's commit only.
    command = 'git checkout -q -b publish/x origin/main && git cherry-pick ' + task_commit
    session.check_publication(command)
    run(repo, 'git', 'checkout', '-q', '-b', 'publish/x', 'origin/main')
    run(repo, 'git', 'cherry-pick', task_commit)
    session.after_bash(command)
    session.lease_id = 'held'
    session.guard('Bash', {'command': 'git push origin publish/x:main'})   # no conflict: reaches the lease check


def test_manager_named_bundle_is_allowed(repo):
    journal = OwnershipJournal()
    session = session_for(repo, journal)
    commit_as_task(session, repo, 'B: task X fix')
    run(repo, 'git', 'commit', '-q', '--allow-empty', '-m', 'C: TAGGY-3515')
    unrelated = run(repo, 'git', 'rev-parse', 'HEAD')
    session.assignment = f'@Git Agent push main. Bundling {unrelated[:7]} (TAGGY-3515) is approved.'
    session.lease_id = 'held'
    session.guard('Bash', {'command': 'git push origin HEAD:main'})


def test_commit_owned_by_an_older_revision_is_foreign():
    state = {'task_revision': '3', 'commit_owners': {'b' * 40: {'task_revision': '2'}}, 'task': {'facts': {}}}
    foreign = scope.foreign_commits([('b' * 40, 'old fix')], state, '')
    assert foreign and foreign[0]['owner'] == 'task revision 2'
    state['task']['facts']['implementation_sha'] = 'b' * 40
    assert scope.foreign_commits([('b' * 40, 'old fix')], state, '') == []


def test_push_all_is_refused(repo):
    with pytest.raises(ModelError) as error:
        scope.push_targets('git push --all origin', str(repo))
    assert error.value.code == 'PUBLICATION_SCOPE_CONFLICT'


def test_pulled_commits_are_not_attributed_to_the_task(repo):
    """A merge/pull brings other people's commits into HEAD; they keep their own (older) committer time."""
    journal = OwnershipJournal()
    session = session_for(repo, journal)
    run(repo, 'git', 'checkout', '-q', '-b', 'other')
    subprocess.run(['git', 'commit', '-q', '--allow-empty', '-m', 'C: someone else'], cwd=repo, check=True,
                   env={**__import__('os').environ, 'GIT_COMMITTER_DATE': '2020-01-01T00:00:00Z',
                        'GIT_AUTHOR_DATE': '2020-01-01T00:00:00Z'})
    foreign = run(repo, 'git', 'rev-parse', 'HEAD')
    run(repo, 'git', 'checkout', '-q', 'main')
    command = 'git merge -q --ff-only other'
    session.check_publication(command)
    run(repo, 'git', 'merge', '-q', '--ff-only', 'other')
    session.after_bash(command)
    assert foreign not in journal.state.get('commit_owners', {})
    with pytest.raises(ModelError):
        session.guard('Bash', {'command': 'git push origin main'})
