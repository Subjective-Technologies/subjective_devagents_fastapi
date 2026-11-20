"""TAGGY-3532: each task works in its own folder, so tasks never share a checkout."""
import subprocess
from pathlib import Path

import pytest

from subjective_devagents_fastapi import task_workspace
from subjective_devagents_fastapi.workspace_fence import fence_key


def git(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def init(path):
    path.mkdir(parents=True, exist_ok=True)
    git(path, 'init', '-q', '-b', 'main')
    git(path, 'config', 'user.email', 't@example.com')
    git(path, 'config', 'user.name', 't')


@pytest.fixture
def base(tmp_path):
    root = tmp_path / 'subjective'
    init(root)
    (root / 'README.md').write_text('parent\n')
    nested = root / 'services' / 'svc'
    init(nested)
    (nested / 'app.py').write_text('print(1)\n')
    (nested / '.gitignore').write_text('node_modules/\n')
    git(nested, 'add', '.')
    git(nested, 'commit', '-qm', 'svc')
    (nested / 'node_modules').mkdir()
    (root / '.gitignore').write_text('.worktrees/\nservices/svc/\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'parent')
    return root


def test_slug_is_readable_and_unique_per_task():
    one = task_workspace.slug({'channel_id': 'a', 'task_id': '1', 'task_revision': '1'}, '[taggy-task:9]\nTAGGY-3532 x')
    two = task_workspace.slug({'channel_id': 'b', 'task_id': '1', 'task_revision': '1'}, 'TAGGY-3532 x')
    assert one.startswith('taggy-3532-') and two.startswith('taggy-3532-') and one != two
    assert task_workspace.slug({'channel_id': 'a'}).startswith('task-')


def test_folder_and_nested_worktree_are_created_lazily(base):
    folder = task_workspace.create(base, 'taggy-1-abc')
    assert folder == base / '.worktrees' / 'taggy-1-abc'
    assert (folder / 'README.md').read_text() == 'parent\n'
    assert git(folder, 'branch', '--show-current') == 'task/taggy-1-abc'
    assert not (folder / 'services' / 'svc' / 'app.py').exists()
    assert task_workspace.create(base, 'taggy-1-abc') == folder  # idempotent

    created = task_workspace.materialize(folder, 'services/svc/app.py')
    tree = folder / 'services' / 'svc'
    assert created == [str(tree)]
    assert (tree / 'app.py').read_text() == 'print(1)\n'
    assert git(tree, 'branch', '--show-current') == 'task/taggy-1-abc'
    assert (tree / 'node_modules').is_symlink()
    assert task_workspace.materialize(folder, str(tree / 'app.py')) == []

    # Edits in the task folder never touch the shared checkout.
    (tree / 'app.py').write_text('print(2)\n')
    assert (base / 'services' / 'svc' / 'app.py').read_text() == 'print(1)\n'
    assert git(base / 'services' / 'svc', 'status', '--porcelain') == ''


def test_materialize_ignores_paths_outside_a_task_folder(base):
    assert task_workspace.materialize(base, 'services/svc/app.py') == []
    folder = task_workspace.create(base, 'taggy-2-abc')
    assert task_workspace.materialize(folder, '/etc/passwd') == []


def test_tool_targets(base):
    folder = str(task_workspace.create(base, 'taggy-3-abc'))
    assert task_workspace.tool_targets(folder, 'Read', {'file_path': 'services/svc/app.py'}) == \
        [f'{folder}/services/svc/app.py']
    targets = task_workspace.tool_targets(folder, 'Bash', {'command': f'cd {folder}/services/svc && ls'})
    assert f'{folder}/services/svc' in targets
    assert task_workspace.tool_targets(folder, 'Read', {'file_path': '/etc/passwd'}) == []


def test_fence_is_per_task_folder(base):
    one = task_workspace.create(base, 'taggy-4-a')
    two = task_workspace.create(base, 'taggy-4-b')
    assert fence_key(str(base)) == 'local-mutations'
    assert fence_key(str(one)) != fence_key(str(two))
    assert fence_key(str(one)).startswith('local-mutations-')


def test_retire_refuses_unmerged_work_then_removes(base):
    folder = task_workspace.create(base, 'taggy-5-abc')
    task_workspace.materialize(folder, 'services/svc/app.py')
    tree = folder / 'services' / 'svc'
    (tree / 'app.py').write_text('print(3)\n')
    result = task_workspace.retire(folder)
    assert not result['removed'] and any('uncommitted' in p for p in result['problems'])
    assert (tree / 'app.py').exists()
    git(tree, 'checkout', '--', 'app.py')
    result = task_workspace.retire(folder, force=True)
    assert result['removed'] and not folder.exists()
    assert (base / 'services' / 'svc' / 'node_modules').is_dir()
    assert (base / 'services' / 'svc' / 'app.py').read_text() == 'print(1)\n'
