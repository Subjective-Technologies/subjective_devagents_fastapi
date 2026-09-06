#!/usr/bin/env python3
"""Real pty spawn used when node-pty's native module cannot load (e.g. the web
server running under stock Node while node-pty was rebuilt for Electron).

Node talks to this process over stdin/stdout; this process talks to the child
(`su - user`, …) over a pty slave. Binary, unbuffered.
"""
from __future__ import annotations

import fcntl
import os
import pty
import select
import struct
import sys
import termios


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    packed = struct.pack("HHHH", max(rows, 1), max(cols, 1), 0, 0)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


def _waitstatus_to_exitcode(status: int) -> int:
    if hasattr(os, "waitstatus_to_exitcode"):
        return os.waitstatus_to_exitcode(status)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: pty_fallback.py <command> [args...]\n")
        return 2
    command = sys.argv[1:]
    cols = int(os.environ.get("SUBJECTIVE_PTY_COLS") or "100")
    rows = int(os.environ.get("SUBJECTIVE_PTY_ROWS") or "30")
    cwd = os.environ.get("SUBJECTIVE_PTY_CWD") or os.getcwd()

    pid, master = pty.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
        except OSError:
            pass
        os.execvp(command[0], command)

    _set_winsize(master, rows, cols)
    for fd in (0, 1, master):
        try:
            os.set_blocking(fd, False)
        except OSError:
            pass

    child_status = 0
    child_exited = False
    try:
        while True:
            readable, _, _ = select.select([0, master], [], [], 0.25)
            if 0 in readable:
                try:
                    data = os.read(0, 8192)
                except BlockingIOError:
                    data = b""
                if not data:
                    try:
                        os.kill(pid, 15)
                    except OSError:
                        pass
                    break
                os.write(master, data)
            if master in readable:
                try:
                    data = os.read(master, 8192)
                except OSError:
                    data = b""
                if not data:
                    break
                os.write(1, data)
            if not child_exited:
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    child_exited = True
                    child_status = _waitstatus_to_exitcode(status)
            if child_exited:
                ready, _, _ = select.select([master], [], [], 0.05)
                if master in ready:
                    try:
                        leftover = os.read(master, 8192)
                    except OSError:
                        leftover = b""
                    if leftover:
                        os.write(1, leftover)
                        continue
                break
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if not child_exited:
            try:
                waited, status = os.waitpid(pid, 0)
                if waited == pid:
                    child_status = _waitstatus_to_exitcode(status)
            except ChildProcessError:
                pass
    return child_status


if __name__ == "__main__":
    sys.exit(main())
