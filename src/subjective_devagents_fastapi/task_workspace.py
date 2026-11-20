"""One folder per task, so concurrent tasks never share a checkout (TAGGY-3532).

The shared workspace (``/subjective``) is a parent repository holding dozens of nested
repositories. A task folder is a git worktree of the parent repository (small: only its
own files) under ``<base>/.worktrees/<slug>``. A nested repository appears inside it, as
that repository's own worktree on branch ``task/<slug>``, the first time a tool of the
task touches a path inside it. Only repositories the task uses are copied.

Effects:
- A task's changed-file list is computed inside its own folder, so another task's edits
  cannot appear in it.
- The mutation fence is keyed by workspace, so tasks in different folders stop waiting
  on each other (WORKSPACE_BUSY).

Disable with SUBJECTIVE_DEVAGENTS_TASK_WORKTREES=0 (dispatches then share the base again).
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

from .language_models.contracts import ModelError

FOLDER = '.worktrees'
# Gitignored dependency folders reused from the base checkout instead of reinstalled.
# Node resolves modules relative to the importing file, so code still comes from the task
# folder. Python virtualenvs are NOT shared: an editable install would import base code.
SHARED_DEPENDENCIES = ('node_modules',)
_TAGGY = re.compile(r'\bTAGGY-(\d{1,7})\b')


def enabled() -> bool:
    return os.environ.get('SUBJECTIVE_DEVAGENTS_TASK_WORKTREES', '1').strip().lower() not in {'0', 'false', 'off', 'no'}


def slug(binding: dict, assignment: str = '') -> str:
    """Readable and unique per channel task revision: taggy-3532-1a2b3c or task-1a2b3c4d."""
    material = '\x1f'.join(str(binding.get(k) or '') for k in ('channel_id', 'task_id', 'task_revision'))
    digest = hashlib.sha256(material.encode()).hexdigest()
    found = _TAGGY.search(assignment or '')
    return f'taggy-{found.group(1)}-{digest[:6]}' if found else f'task-{digest[:8]}'


def branch_for(folder: Path) -> str:
    return 'task/' + Path(folder).name


def _git(cwd, *args, timeout=120):
    result = subprocess.run(['git', '-c', 'core.fsmonitor=false', '-C', str(cwd), *args], capture_output=True,
                            text=True, timeout=timeout,
                            env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/tmp'),
                                 'GIT_TERMINAL_PROMPT': '0'})
    return result.returncode, result.stdout.strip(), result.stderr.strip()


@contextmanager
def _repo_lock(repository: Path):
    """Serialize `git worktree add` against one repository across concurrent tasks."""
    code, common, _ = _git(repository, 'rev-parse', '--path-format=absolute', '--git-common-dir', timeout=10)
    lock = Path(common if code == 0 and common else repository) / 'devagents-task-worktree.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _add_worktree(repository: Path, location: Path, branch: str):
    if location.exists() and any(location.iterdir()):
        raise ModelError('task_workspace_failed', f'{location} exists and is not empty; cannot place a worktree there')
    code, _, _ = _git(repository, 'rev-parse', '--verify', '--quiet', f'refs/heads/{branch}', timeout=10)
    args = ('worktree', 'add', str(location), branch) if code == 0 else \
        ('worktree', 'add', '-b', branch, str(location), 'HEAD')
    with _repo_lock(repository):
        code, _, error = _git(repository, *args, timeout=600)
    if code:
        raise ModelError('task_workspace_failed', f'git worktree add failed in {repository}: {error[-300:]}')


def create(base: str | Path, name: str) -> Path:
    """The task folder (created once): a worktree of the base's own repository."""
    base = Path(base).resolve()
    folder = base / FOLDER / name
    if (folder / '.git').exists():
        return folder
    (base / FOLDER).mkdir(exist_ok=True)
    _add_worktree(base, folder, branch_for(folder))
    return folder


def task_root(workspace: str | Path) -> tuple[Path, Path] | None:
    """(base, folder) when `workspace` is a task folder, otherwise None."""
    folder = Path(workspace).resolve()
    if folder.parent.name != FOLDER:
        return None
    return folder.parent.parent, folder


def _nested_repositories(base: Path, relative: Path) -> list[Path]:
    """Nested repositories of the base along `relative`, outermost first."""
    found, probe = [], base
    for part in relative.parts:
        probe = probe / part
        if part == FOLDER and probe.parent == base:
            return []
        if (probe / '.git').exists():
            found.append(probe.relative_to(base))
        elif not probe.is_dir():
            break
    return found


def materialize(workspace: str | Path, target: str | Path) -> list[str]:
    """Create, inside the task folder, the worktrees a path in it needs. Returns new ones."""
    roots = task_root(workspace)
    if roots is None:
        return []
    base, folder = roots
    try:
        relative = Path(os.path.normpath(Path(folder, target))).relative_to(folder)
    except ValueError:
        return []
    created = []
    for nested in _nested_repositories(base, relative):
        location = folder / nested
        if (location / '.git').exists():
            continue
        _add_worktree(base / nested, location, branch_for(folder))
        for name in SHARED_DEPENDENCIES:
            source, link = base / nested / name, location / name
            if source.is_dir() and not link.exists() and _git(location, 'check-ignore', '-q', name + '/', timeout=10)[0] == 0:
                link.symlink_to(source)
        created.append(str(location))
    return created


_PATH_CHARS = r"[^\s'\"`;|&<>()$]*"


def tool_targets(workspace: str, name: str, arguments: dict) -> list[str]:
    """Paths inside the task folder a tool call names (absolute or workspace-relative)."""
    folder = str(Path(workspace).resolve())
    targets = []
    for key in ('file_path', 'path'):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            targets.append(str(Path(folder, value.strip())))
    if name == 'Bash':
        command = str(arguments.get('command') or '')
        targets += re.findall(re.escape(folder) + r'/' + _PATH_CHARS, command)
        try:
            from .channel_coordination import bash_workdir
            targets.append(str(bash_workdir(command, folder)))
        except ModelError:
            pass
    return [t for t in targets if t == folder or t.startswith(folder + '/')]


def retire(folder: str | Path, *, force: bool = False) -> dict:
    """Remove a finished task folder. Refuses while any worktree has uncommitted changes
    or commits on its task branch that the repository's main line does not contain."""
    roots = task_root(folder)
    if roots is None:
        raise ModelError('task_workspace_failed', 'Not a task folder')
    base, folder = roots
    branch = branch_for(folder)
    trees = sorted((p.parent for p in folder.rglob('.git') if 'node_modules' not in p.parts),
                   key=lambda p: -len(p.parts))
    problems = []
    for tree in trees:
        # Nested worktrees are checked on their own, so the parent ignores its gitlinks.
        code, status, _ = _git(tree, 'status', '--porcelain', '--ignore-submodules=all', timeout=60)
        if code or status:
            problems.append(f'{tree}: uncommitted changes')
            continue
        code, upstream, _ = _git(tree, 'rev-parse', '--abbrev-ref', '--symbolic-full-name', 'origin/HEAD', timeout=10)
        mainline = upstream if code == 0 and upstream else 'origin/main'
        code, ahead, _ = _git(tree, 'rev-list', '--count', f'{mainline}..{branch}', timeout=30)
        if code or ahead not in {'', '0'}:
            problems.append(f'{tree}: {ahead or "?"} commit(s) on {branch} not in {mainline}')
    if problems and not force:
        return {'removed': False, 'problems': problems}
    for tree in trees:
        for name in SHARED_DEPENDENCIES:
            link = tree / name
            if link.is_symlink():
                link.unlink()
        # Run from the repository that owns the worktree, never from inside the tree itself.
        owner = base / tree.relative_to(folder) if tree != folder else base
        _git(owner, 'worktree', 'remove', *(['--force'] if force else []), str(tree), timeout=120)
    return {'removed': not folder.exists(), 'problems': problems}
