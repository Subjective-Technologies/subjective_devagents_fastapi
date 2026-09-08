"""Tests for the command-execution endpoint.

This is remote code execution by design — the operator running `git status` from a phone —
so the tests are mostly about the interlocks that keep it from being RCE by accident: it
must stay off by default, and it must refuse to turn on while the service is unauthenticated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

TOKEN = "exec-test-token"


def build_client(monkeypatch, tmp_path, *, token=TOKEN, exec_enabled=True, exec_cwd=None):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SUBJECTIVE_CHAT_HISTORY_REFRESH_MS", "0")
    monkeypatch.delenv("SUBJECTIVE_CHAT_IRC", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_CLIENT_ROOT", str(tmp_path / "client"))
    (tmp_path / "client" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "client" / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "client" / "web" / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")

    if token is None:
        monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", token)
    if exec_enabled:
        monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_EXEC_ENABLED", "1")
    else:
        monkeypatch.delenv("SUBJECTIVE_DEVAGENTS_EXEC_ENABLED", raising=False)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_EXEC_CWD", str(exec_cwd or tmp_path))

    from subjective_devagents_fastapi.app import app

    return TestClient(app)


def hdr(token=TOKEN):
    return {"X-DevAgents-Token": token} if token else {}


# --- the interlocks ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path, exec_enabled=False) as client:
        status = client.get("/api/exec/status", headers=hdr()).json()
        assert status["enabled"] is False
        run = client.post("/api/exec", json={"command": "echo hi"}, headers=hdr())
        assert run.status_code == 403
        assert "disabled" in run.json()["error"].lower()


def test_refuses_to_enable_without_a_token(monkeypatch, tmp_path):
    """The core safety property: exec must not run on an unauthenticated service even when
    switched on, or anyone who finds the host gets a shell."""
    with build_client(monkeypatch, tmp_path, token=None, exec_enabled=True) as client:
        # With no token the whole API is open, so the request reaches the handler...
        run = client.post("/api/exec", json={"command": "echo hi"})
        assert run.status_code == 403
        assert "unauthenticated" in run.json()["error"].lower()


def test_requires_the_api_token_like_every_other_endpoint(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path) as client:
        assert client.post("/api/exec", json={"command": "echo hi"}).status_code == 401
        assert client.get("/api/exec/status").status_code == 401


# --- running commands -------------------------------------------------------------------------


def test_runs_a_command_and_returns_its_output(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "echo hello-from-shell"}, headers=hdr()).json()
        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "hello-from-shell"


def test_shell_features_work(monkeypatch, tmp_path):
    """Pipes and redirection are most of the value from a phone, so a shell is intended."""
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "printf 'a\\nb\\nc\\n' | wc -l"}, headers=hdr()).json()
        assert result["stdout"].strip() == "3"


def test_a_nonzero_exit_is_reported_not_treated_as_an_error(monkeypatch, tmp_path):
    """`git diff --quiet` failing is an answer. ok stays true; the exit code carries it."""
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "exit 3"}, headers=hdr()).json()
        assert result["ok"] is True
        assert result["exit_code"] == 3


def test_stderr_is_captured_separately(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "echo out; echo err 1>&2"}, headers=hdr()).json()
        assert result["stdout"].strip() == "out"
        assert result["stderr"].strip() == "err"


def test_runs_in_the_configured_working_directory(monkeypatch, tmp_path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    with build_client(monkeypatch, tmp_path, exec_cwd=tmp_path) as client:
        result = client.post("/api/exec", json={"command": "ls"}, headers=hdr()).json()
        assert "marker.txt" in result["stdout"]


def test_a_per_request_cwd_overrides_the_default(monkeypatch, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "only-here.txt").write_text("x", encoding="utf-8")
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "ls", "cwd": str(sub)}, headers=hdr()).json()
        assert "only-here.txt" in result["stdout"]


def test_a_missing_cwd_is_rejected(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "ls", "cwd": "/no/such/dir"}, headers=hdr()).json()
        assert result["ok"] is False
        assert "does not exist" in result["error"]


def test_an_empty_command_is_rejected(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path) as client:
        assert client.post("/api/exec", json={"command": "   "}, headers=hdr()).json()["ok"] is False


def test_a_timeout_is_enforced(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "sleep 5", "timeout": 1}, headers=hdr()).json()
        assert result["ok"] is False
        assert result["status"] == "timeout"


def test_a_command_that_reads_stdin_does_not_hang(monkeypatch, tmp_path):
    """stdin is /dev/null, so a command waiting on input ends immediately instead of
    blocking until the timeout."""
    with build_client(monkeypatch, tmp_path) as client:
        result = client.post("/api/exec", json={"command": "cat", "timeout": 5}, headers=hdr()).json()
        assert result["ok"] is True
        assert result["exit_code"] == 0


def test_large_output_is_truncated(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import shell_exec

    with build_client(monkeypatch, tmp_path) as client:
        result = client.post(
            "/api/exec",
            json={"command": "yes abcdefghij | head -n 500000"},
            headers=hdr(),
        ).json()
        assert result["truncated"] is True
        assert len(result["stdout"].encode("utf-8")) <= shell_exec.MAX_OUTPUT_BYTES


def test_the_timeout_is_clamped_to_a_ceiling(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import shell_exec

    assert shell_exec._clamp_timeout(10_000) == shell_exec.MAX_TIMEOUT_SECONDS
    assert shell_exec._clamp_timeout(0) == 1
    assert shell_exec._clamp_timeout("nonsense") == shell_exec.DEFAULT_TIMEOUT_SECONDS
