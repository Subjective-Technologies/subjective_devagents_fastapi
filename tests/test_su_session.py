"""Switching to another OS user must not need the server account to enter their home.

"Continue as <user>" failed with `[Errno 13] Permission denied: '/home/worktwins'` on a
normal, private (drwx------) home. The cause is subtle: subprocess applies `cwd=` in the
forked child *before* exec, so the chdir runs with the server's uid, not the target's.
`su -` is a login shell and chdirs into the target's home itself, after the uid switch —
which is the only point at which that directory is readable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi.su_session_manager import SuSessionManager  # noqa: E402


@pytest.fixture
def unreadable_home(tmp_path):
    """A home the current process cannot enter, like any other user's real home."""
    home = tmp_path / "someone"
    home.mkdir()
    os.chmod(home, 0o000)
    yield home
    os.chmod(home, 0o700)


def test_su_is_not_spawned_from_the_target_users_home(monkeypatch, unreadable_home):
    """Drives the real spawn directly: ensure_session() would go on to authenticate a pty
    nothing is answering, and the cwd is chosen before any of that."""
    from subjective_devagents_fastapi.su_session_manager import _Session

    manager = SuSessionManager()
    spawned: dict = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            spawned["argv"] = argv
            spawned["kwargs"] = kwargs
            # Reproduce what the real fork does with cwd=, so a regression fails here
            # rather than only in production.
            if kwargs.get("cwd") is not None:
                os.listdir(kwargs["cwd"])
            self.pid = 1234

        def poll(self):
            return 1

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    def no_prompt(*_args, **_kwargs):
        raise TimeoutError  # skip the interactive password wait

    monkeypatch.setattr(manager, "_wait_for_pattern", no_prompt)
    monkeypatch.setattr(manager, "_poll_auth_result", lambda *_a, **_k: {"success": False})
    monkeypatch.setattr(manager, "_reader_thread", lambda *_a, **_k: None)

    session = _Session(unreadable_home.name)
    try:
        manager._run_connect_flow(session)
    except Exception:
        pass  # the fake pty never authenticates; only the spawn arguments matter here

    assert spawned, "su was never spawned"
    cwd = spawned["kwargs"].get("cwd")
    assert spawned["argv"] == ["su", "-", unreadable_home.name], spawned["argv"]
    assert cwd != str(unreadable_home), (
        "su was spawned with cwd set to the target user's home; the chdir runs as the "
        "server account and fails on any private home"
    )
    # The real invariant: whatever cwd is, the server account must be able to enter it.
    assert cwd is None or os.access(cwd, os.X_OK), f"server account cannot enter cwd={cwd!r}"
