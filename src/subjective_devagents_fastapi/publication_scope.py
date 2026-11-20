"""Publication scope: a push publishes the current task's commits, not whatever is in ancestry.

Production 2026-11-16: a Git agent pushed the verified fix and, because it sat on top of
another person's unpublished TAGGY-3515 commit, published that one too (Manager had cited
a different SHA for it). The runtime now owns this check instead of the prompt:

* commits created by a coordinated dispatch are recorded in the chat journal
  (``commit`` op -> ``commit_owners[sha] = task revision``);
* before ``git push`` the range ``<remote>/<branch>..<candidate>`` (merges excluded) is
  computed; every commit must belong to the current task revision, be one of its canonical
  SHAs, or be named explicitly in the assignment (Manager's decision to bundle it);
* anything else stops the push with ``PUBLICATION_SCOPE_CONFLICT`` and the list of
  foreign commits, so the agent publishes a clean cherry-picked branch or asks Manager.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from .language_models.contracts import ModelError

PUSH = re.compile(r'\bgit\b((?:\s+-C\s+\S+|\s+-c\s+\S+)*)\s+push\b([^;&|\n]*)')
COMMIT_MAKERS = re.compile(r'\bgit\b(?:\s+-C\s+\S+|\s+-c\s+\S+)*\s+(?:commit|cherry-pick|merge|rebase|am|revert|pull)\b')
_VALUED = {'--repo', '--receive-pack', '--exec', '-o', '--push-option', '--signed'}
MAX_RANGE = 200


def git(cwd, *args, timeout=15):
    try:
        done = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _directory(prefix, workdir):
    folder = Path(workdir)
    try:
        tokens = shlex.split(prefix or '')
    except ValueError:
        return folder
    for index, token in enumerate(tokens):
        if token == '-C' and index + 1 < len(tokens):
            path = Path(tokens[index + 1])
            folder = path if path.is_absolute() else folder / path
    return folder


def push_targets(command, workdir, *, strict=False):
    """[(repo_dir, remote, src, dst)] for each ``git push`` in a shell command."""
    targets = []
    for match in PUSH.finditer(command or ''):
        folder = _directory(match.group(1), workdir)
        try:
            tokens = shlex.split(match.group(2))
        except ValueError:
            raise ModelError('PUBLICATION_SCOPE_UNVERIFIED', 'git push arguments are not literal') from None
        if strict and (len(tokens) != 2 or not re.fullmatch(r'[A-Za-z0-9_.-]+', tokens[0])
                       or not re.fullmatch(r'[A-Za-z0-9_./-]+:refs/heads/[A-Za-z0-9_./-]+', tokens[1])
                       or match.group(1) and re.search(r'\s-c\s', match.group(1))):
            raise ModelError('PUBLICATION_SCOPE_UNVERIFIED',
                             'Push a single explicit source to a named remote branch: git push origin HEAD:refs/heads/branch')
        positional, skip = [], False
        for token in tokens:
            if skip:
                skip = False
                continue
            if token in _VALUED:
                skip = True
                continue
            if token.startswith('-'):
                if token in {'--all', '--mirror', '--tags'}:
                    raise ModelError('PUBLICATION_SCOPE_CONFLICT',
                                     f'git push {token} publishes refs beyond this task; push the task branch explicitly')
                continue
            positional.append(token)
        remote = positional[0] if positional else (git(folder, 'config', f'branch.{git(folder, "branch", "--show-current") or ""}.remote') or 'origin')
        refspecs = positional[1:] or ['HEAD']
        for spec in refspecs:
            spec = spec.lstrip('+')
            src, _, dst = spec.partition(':')
            if not src:
                continue    # deleting a remote ref publishes no commits
            if not dst:
                dst = (git(folder, 'branch', '--show-current') or '') if src == 'HEAD' else src
            dst = re.sub(r'^refs/heads/', '', dst)
            if dst:
                targets.append((folder, remote, src, dst))
    return targets


def unpublished(folder, remote, src, dst, *, runner=None):
    """Commits the push would publish. The direct path uses the fenced Git runner."""
    query = runner or git
    candidate = query(folder, 'rev-parse', '--verify', '-q', src + '^{commit}')
    if not candidate:
        return None
    if runner:
        # Check the actual destination in the same identity that will push.
        base = remote_head(folder, remote, dst, runner)
        if base == '':
            output = runner(folder, 'ls-remote', remote, 'HEAD') or ''
            sha, _, ref = output.partition('\t')
            base = sha if ref == 'HEAD' and re.fullmatch(r'[0-9a-f]{40,64}', sha) else None
        if not base or query(folder, 'merge-base', '--is-ancestor', base, candidate) is None:
            return None
    else:
        base = (git(folder, 'rev-parse', '--verify', '-q', f'refs/remotes/{remote}/{dst}')
                or git(folder, 'rev-parse', '--verify', '-q', f'refs/remotes/{remote}/HEAD'))
        if not base:
            return None
    flags = () if runner else ('--no-merges',)
    limit = MAX_RANGE + 1 if runner else MAX_RANGE
    listing = query(folder, 'log', *flags, '--reverse', f'--max-count={limit}', '--format=%H%x09%s',
                    f'{base}..{candidate}')
    if listing is None:
        return None
    rows = [line.split('\t', 1) for line in listing.splitlines() if line.strip()]
    if len(rows) > MAX_RANGE:
        return None
    return {'candidate': candidate, 'base': base,
            'commits': [(row[0], row[1] if len(row) > 1 else '') for row in rows]}


def remote_head(folder, remote, dst, runner):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', remote) or remote.startswith('-'):
        return None
    ref = 'refs/heads/' + dst
    fetch_urls = (runner(folder, 'remote', 'get-url', '--all', remote) or '').splitlines()
    push_urls = (runner(folder, 'remote', 'get-url', '--push', '--all', remote) or '').splitlines()
    if len(fetch_urls) != 1 or fetch_urls != push_urls or not runner(folder, 'check-ref-format', '--normalize', ref):
        return None
    response = runner(folder, 'ls-remote', '--refs', remote, ref)
    if response is None or not response:
        return response
    sha, _, observed_ref = response.partition('\t')
    return sha if observed_ref == ref and re.fullmatch(r'[0-9a-f]{40,64}', sha) else None


def foreign_commits(commits, state, assignment):
    """Commits this task revision does not own and Manager did not name."""
    revision = str(state.get('task_revision'))
    owners = state.get('commit_owners') or {}
    facts = ((state.get('task') or {}).get('facts') or {})
    canonical = [sha for sha in [facts.get('implementation_sha'), facts.get('verified_sha'), facts.get('commit_sha')]
                 + list(facts.get('published_sha') or []) if sha]
    text = assignment or ''
    foreign = []
    for sha, subject in commits:
        owner = owners.get(sha)
        if owner and str(owner.get('task_revision')) == revision:
            continue
        if any(sha.startswith(known) or known.startswith(sha) for known in canonical):
            continue
        if re.search(r'(?<![0-9a-f])' + sha[:7] + r'[0-9a-f]{0,33}(?![0-9a-f])', text):
            continue    # Manager explicitly listed it in this assignment: an intentional bundle
        foreign.append({'sha': sha, 'subject': subject[:200],
                        'owner': f"task revision {owner['task_revision']}" if owner else 'not created by this task'})
    return foreign


def conflict_message(target, foreign, remote, dst):
    listed = '; '.join(f"{c['sha'][:12]} {c['subject']!r} ({c['owner']})" for c in foreign[:10])
    more = f' and {len(foreign) - 10} more' if len(foreign) > 10 else ''
    return (f'Pushing to {remote}/{dst} would also publish {len(foreign)} commit(s) that do not belong to this task: '
            f'{listed}{more}. Publish only this task: create a branch from {remote}/{dst}, cherry-pick this task\'s '
            'commits and push that, or report PUBLICATION_SCOPE_CONFLICT to Manager, who can approve bundling '
            'by naming these SHAs in the assignment.')
