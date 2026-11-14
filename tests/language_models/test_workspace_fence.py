"""Orphaned subprocess groups cannot hold global mutation fencing forever."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from subjective_devagents_fastapi.workspace_fence import mutation_fence


@pytest.mark.parametrize('leader_exits', [False, True])
def test_orphan_from_other_channel_is_stopped_without_old_agent(tmp_path, leader_exits):
    ready = tmp_path / 'ready'
    script = '''import pathlib, sys
from types import SimpleNamespace
from subjective_devagents_fastapi.workspace_fence import mutation_fence
from subjective_devagents_fastapi.language_models.transports import LocalProcess, Command
root = pathlib.Path(sys.argv[1])
session = SimpleNamespace(identity=SimpleNamespace(ssh_host='', workspace=str(root), home=str(root)),
    binding={'channel_id':'old-channel', 'dispatch_id':'old-dispatch', 'instance_id':'old-instance'},
    fence_root=str(root/'fences'), call=lambda *a, **k: {'state':{'dispatches':{}}})
with mutation_fence(session):
    command = 'sleep 30 & echo ready > "$1"' if sys.argv[2] == 'True' else 'echo ready > "$1"; sleep 30'
    LocalProcess().run(Command(('sh','-c',command,'fixture',str(root/'ready')), cwd=str(root),
        env={'PATH':'/usr/bin:/bin','HOME':str(root)}), timeout=35, cancelled=lambda:False, on_stdout=lambda text:None)
'''
    parent = subprocess.Popen([sys.executable, '-c', script, str(tmp_path), str(leader_exits)], start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not ready.exists():
            assert parent.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.01)
        owner = json.loads((tmp_path / 'fences/local-mutations.json').read_text())
        parent.kill(); parent.wait(timeout=5)
        successor = SimpleNamespace(identity=SimpleNamespace(ssh_host='', workspace=str(tmp_path), home=str(tmp_path)),
            binding={'channel_id':'new-channel', 'dispatch_id':'new-dispatch', 'instance_id':'new-instance'},
            fence_root=str(tmp_path/'fences'), call=lambda *a, **k: {'state':{'dispatches':{}}})
        began = time.monotonic()
        with mutation_fence(successor):
            assert time.monotonic()-began < 3
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=5)
        if 'owner' in locals():
            try: os.killpg(owner['pgid'], signal.SIGKILL)
            except ProcessLookupError: pass
