"""Run a shell command on the machine hosting this service and return its output.

This exists so the phone can do the small things that do not deserve a whole agent turn —
`git status`, `pytest -q`, tail a log — without a terminal. The agent CLIs already execute
commands themselves (inside their own sandbox); this is the direct path for the operator.

It is, plainly, remote code execution. Three interlocks keep it from becoming one by
accident:

1. **Off unless asked.** `SUBJECTIVE_DEVAGENTS_EXEC_ENABLED=1`.
2. **Never without authentication.** Enabling it while no API token is configured is
   refused, not warned about. The tool is meant to be private; this makes the deployment
   prove that before it will run anything.
3. **Bounded.** Every call has a timeout and an output cap, so one command cannot hang the
   worker pool or return a gigabyte.

Commands run through a shell on purpose — pipes and redirection are most of the value from
a phone — so there is no argument-injection boundary to defend here: whoever holds the token
already has a shell. That is the whole feature. Guard the token accordingly.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from . import auth

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600
# Per stream. A phone cannot render more than this, and the JSON response has to stay sane.
MAX_OUTPUT_BYTES = 1_000_000


class ExecDisabled(RuntimeError):
    """Raised when the endpoint is switched off, or unsafe to switch on."""


def exec_enabled() -> bool:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_EXEC_ENABLED") or "").strip() == "1"


def ensure_available() -> None:
    """Fail closed, and say which interlock stopped it."""
    if not exec_enabled():
        raise ExecDisabled(
            "Command execution is disabled. Set SUBJECTIVE_DEVAGENTS_EXEC_ENABLED=1 to enable it."
        )
    if not auth.auth_enabled():
        raise ExecDisabled(
            "Refusing to enable command execution on an unauthenticated service: set "
            "SUBJECTIVE_DEVAGENTS_API_TOKEN first, or anyone who finds this host gets a shell."
        )


def default_workdir() -> str:
    return os.environ.get("SUBJECTIVE_DEVAGENTS_EXEC_CWD") or str(Path.home())


def _clamp_timeout(requested) -> int:
    try:
        seconds = int(requested)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return max(1, min(seconds, MAX_TIMEOUT_SECONDS))


def _truncate(text: str) -> tuple[str, bool]:
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text, False
    return encoded[:MAX_OUTPUT_BYTES].decode("utf-8", "ignore"), True


def run_command(command: str, cwd: str = "", timeout: int | None = None) -> dict:
    """Run one shell command, returning what happened rather than raising.

    A non-zero exit is a normal outcome and reported as `ok: true` with the exit code —
    `git diff --quiet` failing is an answer, not an error. `ok` is false only when the
    command could not be run or did not finish.
    """
    ensure_available()

    clean = str(command or "").strip()
    if not clean:
        return {"ok": False, "error": "Command is empty."}

    workdir = str(cwd or default_workdir())
    if not Path(workdir).is_dir():
        return {"ok": False, "error": f"Working directory does not exist: {workdir}"}

    seconds = _clamp_timeout(timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS)
    started = time.time()
    try:
        completed = subprocess.run(
            clean,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=seconds,
            # Detach from this process's controlling terminal so a command that tries to read
            # a password prompt fails fast instead of hanging until the timeout.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as expired:
        stdout, stdout_truncated = _truncate(expired.stdout.decode("utf-8", "replace") if isinstance(expired.stdout, bytes) else (expired.stdout or ""))
        stderr, stderr_truncated = _truncate(expired.stderr.decode("utf-8", "replace") if isinstance(expired.stderr, bytes) else (expired.stderr or ""))
        return {
            "ok": False,
            "status": "timeout",
            "error": f"Command exceeded {seconds}s and was killed.",
            "command": clean,
            "cwd": workdir,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_truncated or stderr_truncated,
            "duration_ms": int((time.time() - started) * 1000),
        }
    except OSError as error:
        return {"ok": False, "status": "spawn_error", "error": str(error), "command": clean, "cwd": workdir}

    stdout, stdout_truncated = _truncate(completed.stdout or "")
    stderr, stderr_truncated = _truncate(completed.stderr or "")
    return {
        "ok": True,
        "status": "completed",
        "command": clean,
        "cwd": workdir,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": stdout_truncated or stderr_truncated,
        "duration_ms": int((time.time() - started) * 1000),
    }


def summarize_for_log(command: str) -> str:
    """A short, quoted form for the audit line — every invocation is logged."""
    clean = " ".join(str(command or "").split())
    return shlex.quote(clean[:200] + ("…" if len(clean) > 200 else ""))
