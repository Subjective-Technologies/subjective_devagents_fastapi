"""Bounded, non-interactive process execution. No terminal emulation or replay."""
from __future__ import annotations

from abc import ABC, abstractmethod
import codecs
from dataclasses import dataclass, field
import os
import re
import selectors
import shlex
import signal
import subprocess
import time
from typing import Callable


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    stdin: str = ""
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    status: str = "completed"


class ProcessTransport(ABC):
    @abstractmethod
    def run(self, command: Command, *, timeout: float,
            cancelled: Callable[[], bool],
            on_stdout: Callable[[str], None]) -> ProcessResult:
        """Execute once; callbacks receive decoded stdout chunks, not lines."""


class LocalProcess(ProcessTransport):
    """POSIX pipes with concurrent input/output and process-group termination."""

    def __init__(self, max_output_bytes: int = 8 * 1024 * 1024):
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        # Kill the group even if its leader exited: children can retain pipe FDs.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def run(self, command: Command, *, timeout: float,
            cancelled: Callable[[], bool],
            on_stdout: Callable[[str], None]) -> ProcessResult:
        if timeout <= 0 or not command.argv:
            raise ValueError("positive timeout and nonempty argv required")
        if os.name != "posix":
            return ProcessResult(-1, "", "", "unavailable")
        if cancelled():
            return ProcessResult(-1, "", "", "cancelled")
        deadline = time.monotonic() + timeout
        try:
            process = subprocess.Popen(
                command.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=command.cwd, env=dict(command.env),
                start_new_session=True, bufsize=0,
            )
        except (OSError, ValueError):
            # Exception messages can contain credential-bearing paths or args.
            return ProcessResult(-1, "", "", "unavailable")

        outputs: dict[str, list[str]] = {"stdout": [], "stderr": []}
        decoders = {name: codecs.getincrementaldecoder("utf-8")("replace")
                    for name in outputs}
        status = "completed"
        total = 0
        stopped = False
        try:
            pending = memoryview(command.stdin.encode("utf-8"))
            with selectors.DefaultSelector() as selector:
                for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                if pending:
                    os.set_blocking(process.stdin.fileno(), False)
                    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
                else:
                    process.stdin.close()

                while selector.get_map() or process.poll() is None:
                    if cancelled():
                        status = "cancelled"
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timed_out"
                        break
                    for key, _ in selector.select(min(0.05, remaining)):
                        stream, name = key.fileobj, key.data
                        if name == "stdin":
                            try:
                                written = os.write(stream.fileno(), pending[:65536])
                                pending = pending[written:]
                            except BrokenPipeError:
                                pending = pending[:0]
                            except BlockingIOError:
                                continue
                            if not pending:
                                selector.unregister(stream)
                                stream.close()
                            continue
                        try:
                            chunk = os.read(stream.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                        allowed = self.max_output_bytes - total
                        limited = len(chunk) > allowed
                        chunk = chunk[:allowed]
                        total += len(chunk)
                        decoded = decoders[name].decode(chunk, final=not chunk or limited)
                        if decoded:
                            outputs[name].append(decoded)
                            if name == "stdout":
                                on_stdout(decoded)
                        if limited:
                            status = "output_limit"
                            break
                    if status != "completed":
                        break
            if status != "completed":
                self._stop(process)
                stopped = True
            elif process.returncode != 0:
                status = "failed"
            return ProcessResult(process.returncode, "".join(outputs["stdout"]),
                                 "".join(outputs["stderr"]), status)
        finally:
            # Also covers cancellation/callback exceptions; no orphan CLI tree.
            if not stopped:
                self._stop(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


class SshTunnel(ProcessTransport):
    """SSH pipe transport, not a PTY or port-forwarding server.

    Killing local ssh cannot prove remote execution stopped. Interrupted or failed
    remote executions return remote_outcome_unknown and must never be replayed
    automatically. Remote credentials must already exist in the isolated home;
    only non-secret environment configuration is permitted in remote argv.
    """

    REMOTE_ENV = frozenset({"PATH", "HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME",
                            "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER", "LOGNAME"})

    def __init__(self, host: str, os_user: str | None = None,
                 ssh_executable: str = "ssh"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@\[\]-]*", host):
            raise ValueError("invalid SSH host")
        if os_user is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*[$]?", os_user):
            raise ValueError("invalid OS user")
        self.host = host
        self.os_user = os_user
        self.ssh_executable = ssh_executable
        self.local = LocalProcess()

    def run(self, command: Command, *, timeout: float,
            cancelled: Callable[[], bool],
            on_stdout: Callable[[str], None]) -> ProcessResult:
        if not command.argv or command.env.keys() - self.REMOTE_ENV:
            raise ValueError("SSH requires argv and only non-secret environment configuration")
        if cancelled():
            return ProcessResult(-1, "", "", "cancelled")
        remote = ["env", "-i", *(f"{key}={value}" for key, value in command.env.items()),
                  *command.argv]
        if command.cwd is not None:
            # Change directory after sudo so private workspaces need not be
            # accessible to the SSH login user. Values remain positional args.
            remote = ["sh", "-c", 'cd -- "$1" && shift && exec "$@"',
                      "devagents", command.cwd, *remote]
        if self.os_user:
            remote = ["sudo", "-n", "-u", self.os_user, "--", *remote]
        remote_command = "exec " + shlex.join(remote)
        local_env = {key: value for key, value in os.environ.items()
                     if key in {"PATH", "HOME", "SSH_AUTH_SOCK", "LANG", "LC_ALL"}}
        result = self.local.run(Command(
            (self.ssh_executable, "-T", "-o", "BatchMode=yes", "-o",
             "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", "--",
             self.host, remote_command), command.stdin, env=local_env,
        ), timeout=timeout, cancelled=cancelled, on_stdout=on_stdout)
        if result.status == "cancelled" and result.returncode == -1:
            # LocalProcess observed cancellation before Popen (including a race
            # after this wrapper's preflight check), so nothing ran remotely.
            return result
        if result.status not in {"completed", "unavailable"} or result.returncode != 0:
            if result.status != "unavailable":
                return ProcessResult(result.returncode, result.stdout, result.stderr,
                                     "remote_outcome_unknown")
        return result
