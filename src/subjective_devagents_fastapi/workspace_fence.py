"""Local mutation fences survive worker death through inherited subprocess FDs.

Readers never take this lock. Takeover checks the journal before stopping an orphan
process group and never resets files. Local mutating tools sharing a runtime journal
share this fence; path leases still enforce the narrower repository scope.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import sys
import time

from .language_models.contracts import ModelError

_local = threading.local()


def inherited_fds():
    return tuple(getattr(_local, 'fds', ()))


def fence_key(workspace):
    """One fence per checkout. The shared base keeps its historical name; each task folder
    (TAGGY-3532) gets its own, so tasks in different folders never wait on each other."""
    from .task_workspace import task_root
    roots = task_root(workspace) if workspace else None
    if roots is None:
        return 'local-mutations'
    import hashlib
    return 'local-mutations-' + hashlib.sha256(str(roots[1]).encode()).hexdigest()[:16]


def _start(pid):
    try:
        # comm can contain spaces and parentheses; fields after its final ')' are stable.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def child_command(argv):
    holder = getattr(_local, 'holder', None)
    if not holder:
        return argv
    path, owner = holder
    # Registration happens in the child before the command can run, even if the API
    # dies immediately after Popen. No task code runs without a registered fence.
    script = """import json, os, pathlib, sys, tempfile
path, owner = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2])
start = pathlib.Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
owner.update(pgid=os.getpgrp(), process_start=start)
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.holder-')
with os.fdopen(fd, 'w') as stream: json.dump(owner, stream)
os.replace(temporary, path)
os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
"""
    return (sys.executable, '-c', script, str(path), json.dumps(owner), *argv)


@contextmanager
def mutation_fence(session):
    if session.identity.ssh_host:
        raise ModelError('remote_scope_unverified', 'Remote mutation cannot use a local fence')
    holder = getattr(_local, 'holder', None)
    if holder and holder[1].get('dispatch_id') == session.binding['dispatch_id'] \
            and holder[1].get('instance_id') == session.binding['instance_id']:
        # Already fenced by this execution (the guard acquiring a lease inside a fenced tool
        # call). A second flock on a new descriptor would wait for ourselves.
        yield
        return
    folder = Path(os.environ.get('SUBJECTIVE_CHAT_HOME', str(Path(session.identity.home or session.identity.workspace) / '.subjective_chat'))) / 'mutation-fences'
    if session.fence_root:
        folder = Path(session.fence_root)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = fence_key(session.identity.workspace)
    metadata = folder / (key + '.json')
    descriptor = os.open(folder / (key + '.lock'), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                try:
                    owner = json.loads(metadata.read_text())
                except (OSError, ValueError):
                    owner = {}
                state = session.call('snapshot')['state']
                old = state.get('dispatches', {}).get(owner.get('dispatch_id'))
                parent_gone = owner.get('worker_start') and _start(owner.get('worker_pid')) != owner['worker_start']
                dead = parent_gone or old and old['status'] in {'ABANDONED', 'FAILED', 'CANCELLED', 'SUPERSEDED', 'COMPLETED', 'BLOCKED'}
                pid = owner.get('pgid')
                # Never signal an unrelated/reused PID, a healthy dispatch, or the API process.
                if dead and isinstance(pid, int) and pid > 1 and owner.get('process_start') and _start(pid) == owner['process_start']:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if dead and pid and _start(pid) is None:
                    # Group leader may have exited while descendants retained this exact fence FD.
                    inode = os.fstat(descriptor)
                    for candidate in Path('/proc').glob('[0-9]*/fd/*'):
                        try:
                            stat = candidate.stat()
                            child = int(candidate.parent.parent.name)
                            if (stat.st_ino, stat.st_dev) != (inode.st_ino, inode.st_dev) or child == os.getpid():
                                continue
                            fields = Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()
                            group = int(fields[2])
                            if group == pid and group > 1 and group != os.getpgrp() and int(fields[19]) >= int(owner['process_start']):
                                os.killpg(group, signal.SIGKILL)
                        except (OSError, ValueError, IndexError):
                            continue
                if time.monotonic() >= deadline:
                    raise ModelError('WORKSPACE_BUSY', 'Previous mutation has not stopped; workspace remains fenced')
                time.sleep(.05)
        owner = {'dispatch_id': session.binding['dispatch_id'], 'instance_id': session.binding['instance_id'],
                 'channel_id': session.binding['channel_id'], 'worker_pid': os.getpid(), 'worker_start': _start(os.getpid())}
        metadata.write_text(json.dumps(owner))
        previous_fds, previous_holder = inherited_fds(), getattr(_local, 'holder', None)
        _local.fds, _local.holder = (*previous_fds, descriptor), (metadata, owner)
        try:
            yield
        finally:
            _local.fds, _local.holder = previous_fds, previous_holder
    finally:
        # close, not LOCK_UN: an orphan child must retain the fence until it has stopped.
        os.close(descriptor)
