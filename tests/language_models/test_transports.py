import os
from pathlib import Path
import shlex
import sys
import time

import pytest

from subjective_devagents_fastapi.language_models.transports import (
    Command, LocalProcess, ProcessResult, SshTunnel,
)


def run_script(script, *, stdin="", timeout=5, cancelled=lambda: False,
               on_stdout=lambda chunk: None, env=None, transport=None):
    return (transport or LocalProcess()).run(
        Command((sys.executable, "-c", script), stdin=stdin, env=env or {}),
        timeout=timeout, cancelled=cancelled, on_stdout=on_stdout,
    )


def test_large_stdin_and_stderr_do_not_deadlock_and_no_tty():
    result = run_script(
        "import sys, os; sys.stderr.write('e'*200000); sys.stderr.flush(); "
        "s=sys.stdin.read(); print(len(s), os.isatty(0), os.isatty(1))",
        stdin="x" * 300000,
    )
    assert result.status == "completed"
    assert result.stdout == "300000 False False\n"
    assert result.stderr == "e" * 200000


def test_streaming_callback_arrives_before_exit(tmp_path):
    release = tmp_path / "release"
    chunks = []

    def receive(chunk):
        chunks.append(chunk)
        release.touch()

    result = run_script(
        f"import pathlib, time; print('ready', flush=True); p=pathlib.Path({str(release)!r}); "
        "\nwhile not p.exists(): time.sleep(.01)\nprint('done')",
        on_stdout=receive,
    )
    assert result.status == "completed"
    assert "".join(chunks) == "ready\ndone\n"


def test_environment_is_exact_and_parent_untouched(monkeypatch):
    monkeypatch.setenv("PARENT_SECRET", "not-for-child")
    result = run_script(
        "import os; print(os.getenv('PARENT_SECRET')); print(os.getenv('CHILD_ONLY'))",
        env={"CHILD_ONLY": "child"},
    )
    assert result.stdout == "None\nchild\n"
    assert os.environ["PARENT_SECRET"] == "not-for-child"


@pytest.mark.parametrize("cause", ["timed_out", "cancelled", "callback"])
def test_cleanup_stops_child_even_when_parent_exits(tmp_path, cause):
    marker = tmp_path / "child-survived"
    pidfile = tmp_path / "child-pid"
    child = (
        "import signal,time,pathlib,os; signal.signal(signal.SIGINT, signal.SIG_IGN); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        f"time.sleep(1); pathlib.Path({str(marker)!r}).touch(); time.sleep(30)"
    )
    script = (
        "import subprocess,sys,time,pathlib; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"p=pathlib.Path({str(pidfile)!r}); "
        "\nwhile not p.exists(): time.sleep(.01)\nprint('ready',flush=True)"
    )

    def receive(chunk):
        if cause == "callback":
            raise RuntimeError("consumer failed")

    if cause == "callback":
        with pytest.raises(RuntimeError, match="consumer failed"):
            run_script(script, on_stdout=receive)
    else:
        result = run_script(script, timeout=.5,
                            cancelled=lambda: cause == "cancelled" and pidfile.exists())
        assert result.status == cause
    # Linux zombies have stopped executing, though their adoptive parent may reap later.
    child_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        stat = Path(f"/proc/{child_pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(.01)
    else:
        pytest.fail("child remains running")
    assert not marker.exists()


def test_output_limit_bounds_combined_streams():
    result = run_script("import sys; sys.stderr.write('e'*10000); print('x'*10000)",
                        transport=LocalProcess(max_output_bytes=1000))
    assert result.status == "output_limit"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 1000


def test_missing_executable_returns_safe_unavailable():
    result = LocalProcess().run(Command(("/missing/secret-path",)), timeout=1,
                                cancelled=lambda: False, on_stdout=lambda text: None)
    assert result == ProcessResult(-1, "", "", "unavailable")


def test_nonzero_exit_is_failed():
    assert run_script("raise SystemExit(7)").status == "failed"


def test_deadline_enforced_while_child_never_reads_large_stdin():
    start = time.monotonic()
    result = run_script("import time; time.sleep(30)", stdin="x" * 1000000, timeout=.15)
    assert result.status == "timed_out"
    assert time.monotonic() - start < 2


def test_closed_child_stdin_does_not_abort_output_collection():
    result = run_script("import os; os.close(0); print('result')", stdin="x" * 1000000)
    assert result.status == "completed"
    assert result.stdout == "result\n"


@pytest.mark.parametrize("host", ["-oProxyCommand=evil", "host\ncommand", "host;evil", ""])
def test_ssh_rejects_unsafe_hosts(host):
    with pytest.raises(ValueError):
        SshTunnel(host)


def test_ssh_quoting_prompt_stdin_and_no_pty(monkeypatch):
    captured = []

    def execute(self, command, **kwargs):
        captured.append(command)
        return ProcessResult(0, "ok", "")

    monkeypatch.setattr(LocalProcess, "run", execute)
    tunnel = SshTunnel("host.example", os_user="worker")
    result = tunnel.run(Command(("/opt/cli name", "arg; touch nope"),
                                stdin="private prompt", cwd="/work dir/'quoted'",
                                env={"HOME": "/home/worker space", "XDG_CONFIG_HOME": "/config"}),
                        timeout=5, cancelled=lambda: False, on_stdout=lambda text: None)
    command = captured[0]
    assert result.status == "completed"
    assert "-T" in command.argv
    assert "BatchMode=yes" in command.argv
    assert "StrictHostKeyChecking=yes" in command.argv
    assert "private prompt" not in " ".join(command.argv)
    assert command.stdin == "private prompt"
    assert shlex.split(command.argv[-1]) == [
        "exec", "sudo", "-n", "-u", "worker", "--", "sh", "-c",
        'cd -- "$1" && shift && exec "$@"', "devagents", "/work dir/'quoted'",
        "env", "-i", "HOME=/home/worker space", "XDG_CONFIG_HOME=/config",
        "/opt/cli name", "arg; touch nope",
    ]


@pytest.mark.parametrize("status,code", [("timed_out", -9), ("cancelled", -2),
                                         ("failed", 255), ("output_limit", -9)])
def test_ssh_ambiguous_execution_never_claims_remote_stopped(monkeypatch, status, code):
    monkeypatch.setattr(LocalProcess, "run",
                        lambda *args, **kwargs: ProcessResult(code, "", "", status))
    result = SshTunnel("host").run(Command(("cli",)), timeout=1,
                                  cancelled=lambda: False, on_stdout=lambda text: None)
    assert result.status == "remote_outcome_unknown"


def test_ssh_rejects_secret_environment():
    with pytest.raises(ValueError, match="non-secret"):
        SshTunnel("host").run(Command(("cli",), env={"API_KEY": "secret"}), timeout=1,
                              cancelled=lambda: False, on_stdout=lambda text: None)


def test_ssh_pre_cancelled_never_starts_transport(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("SSH launched despite cancellation")

    monkeypatch.setattr(LocalProcess, "run", unexpected)
    result = SshTunnel("host").run(Command(("cli",)), timeout=1,
                                  cancelled=lambda: True, on_stdout=lambda text: None)
    assert result.status == "cancelled"


def test_ssh_local_preflight_cancellation_is_known(monkeypatch):
    monkeypatch.setattr(LocalProcess, "run",
                        lambda *args, **kwargs: ProcessResult(-1, "", "", "cancelled"))
    result = SshTunnel("host").run(Command(("cli",)), timeout=1,
                                  cancelled=lambda: False, on_stdout=lambda text: None)
    assert result.status == "cancelled"
