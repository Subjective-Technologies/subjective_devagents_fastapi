"""Fixtures for the HTTP-surface suite.

This suite drives the app through its own HTTP layer, which is the half the rest of
``tests/`` does not touch: the existing files call ``LlmService`` and
``ssh_connections`` directly, so body parsing, field coercion, status codes, error
envelopes and the ``require_api_token`` gate are unexercised by them.

Every fixture here builds a whole app against a throwaway ``SUBJECTIVE_CHAT_HOME``,
a throwaway ``~/.ssh`` and a throwaway agent registry, so a test can create chats,
save connections and upload keys without touching the operator's real state — and
so a test that reads "the keys this server can use" gets the same answer on every
machine.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT / "src"))

REPO_ROOT = SERVICE_ROOT.parents[1]

TOKEN = "api-suite-token-4f21c0d8"

# A machine the suite may actually reach. `subjective-agents` is the alias the browser
# suite uses, so a green live run here and a green browser run mean the same host.
LIVE_SSH_HOST = os.environ.get("SUBJECTIVE_DEVAGENTS_TEST_SSH_HOST", "subjective-agents")
# Reserved for documentation (RFC 5737) and therefore guaranteed never to answer.
UNREACHABLE_HOST = "203.0.113.1"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: needs a reachable remote machine; skipped with the host named when it is not.",
    )


# --- the app under test -----------------------------------------------------------------


def _isolate(monkeypatch, tmp_path: Path, **overrides) -> Path:
    """Point every path and interlock the app reads at this test's own directory."""
    home = tmp_path / "chat-home"
    client_root = tmp_path / "client"
    (client_root / "web").mkdir(parents=True, exist_ok=True)
    (client_root / "src").mkdir(parents=True, exist_ok=True)
    (client_root / "web" / "index.html").write_text(
        '<!doctype html><html lang="en"><head><title>DevAgents</title>\n'
        '<meta name="devagents-api" content="https://devagents-api.example.invalid">\n'
        "</head><body>DEVAGENTS_DEPLOY_VERSION</body></html>",
        encoding="utf-8",
    )

    ssh_dir = tmp_path / "server-ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)

    environment = {
        "SUBJECTIVE_CHAT_HOME": str(home),
        "SUBJECTIVE_DEVAGENTS_CLIENT_ROOT": str(client_root),
        "SUBJECTIVE_CHAT_HISTORY_REFRESH_MS": "0",
        "SUBJECTIVE_DEVAGENTS_D1_SYNC": "0",
        # The server's own ~/.ssh is read by GET /api/ssh/connections. Pointing it at an
        # empty directory keeps the operator's real keys out of every assertion.
        "SUBJECTIVE_DEVAGENTS_SSH_DIR": str(ssh_dir),
        # POST /api/agents/role rewrites a role file in place. Never the real one.
        "SUBJECTIVE_AGENTS_DIR": str(_fixture_agents_dir(tmp_path)),
        "SUBJECTIVE_DEVAGENTS_AUTH_MODE": "development",
    }
    environment.update({key: value for key, value in overrides.items() if value is not None})

    for key, value in environment.items():
        monkeypatch.setenv(key, str(value))
    for key in ("SUBJECTIVE_CHAT_IRC",):
        monkeypatch.delenv(key, raising=False)
    for key, value in overrides.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
    return home


def _fixture_agents_dir(tmp_path: Path) -> Path:
    """A registry with one real-looking agent, so role read/write has something to act on."""
    agents = tmp_path / "workflow-agents"
    roles = agents / "roles"
    roles.mkdir(parents=True, exist_ok=True)
    (agents / "registry.md").write_text(
        textwrap.dedent(
            """\
            # Agent Registry

            | Agent ID | Role file | Primary capability |
            |---|---|---|
            | `pytest_agent` | `roles/pytest_agent.md` | automated test verification |
            """
        ),
        encoding="utf-8",
    )
    (roles / "pytest_agent.md").write_text(
        "# pytest_agent\n\n## Mission\n\nOwn automated verification.\n",
        encoding="utf-8",
    )
    return agents


def build_client(monkeypatch, tmp_path: Path, *, raise_server_exceptions: bool = True, **overrides) -> TestClient:
    """A client over the real app.

    `raise_server_exceptions=False` is what a browser sees: Starlette's error
    middleware re-raises into the test client after running the app's catch-all
    handler, so a route that leans on that handler looks like a crash here and like
    a 400 envelope in production. Turning it off is the only way to assert the
    envelope the browser actually receives.
    """
    home = _isolate(monkeypatch, tmp_path, **overrides)
    from subjective_devagents_fastapi.app import app

    client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
    client.chat_home = home
    client.tmp_path = tmp_path
    return client


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """Development mode: no credential needed, so a test can assert on the handler."""
    with build_client(monkeypatch, tmp_path) as test_client:
        yield test_client


@pytest.fixture()
def browser_client(monkeypatch, tmp_path):
    """As `client`, but showing the envelope a browser gets when a handler raises."""
    with build_client(monkeypatch, tmp_path, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture()
def token_client(monkeypatch, tmp_path):
    """Token mode with a token set — the automation credential is accepted."""
    with build_client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
    ) as test_client:
        test_client.headers.update({"X-DevAgents-Token": TOKEN})
        yield test_client


@pytest.fixture()
def bare_token_client(monkeypatch, tmp_path):
    """Token mode, no default credential on the client: for testing the gate itself."""
    with build_client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
    ) as test_client:
        yield test_client


@pytest.fixture()
def owner_client(monkeypatch, tmp_path):
    """Owner mode with a static token configured — which must not open the gate."""
    with build_client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="owner",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
    ) as test_client:
        yield test_client


@pytest.fixture()
def test_endpoint_client(monkeypatch, tmp_path):
    """Development mode with the assistant-injection endpoint on.

    TEST_ENDPOINTS_ENABLED is read at import time, so the module-level flag is patched
    rather than only the environment.
    """
    import subjective_devagents_fastapi.app as app_module

    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_ENABLE_TEST_ENDPOINTS", "1")
    monkeypatch.setattr(app_module, "TEST_ENDPOINTS_ENABLED", True)
    with build_client(monkeypatch, tmp_path) as test_client:
        yield test_client


@pytest.fixture()
def ssh_terminal_client(monkeypatch, tmp_path):
    """The SSH terminal interlock on, with a token so enabling it is permitted."""
    with build_client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
        SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED="1",
    ) as test_client:
        test_client.headers.update({"X-DevAgents-Token": TOKEN})
        yield test_client


# --- cleanup ------------------------------------------------------------------------------


@pytest.fixture()
def cleanup():
    """Undo registry for anything a test creates on a machine or in a store.

    Runs on failure as well as success, and keeps going after one entry raises: a
    leaked SSH chat holds a session open on a real machine, and the first failing
    cleanup must not strand the rest.
    """
    actions: list = []

    def register(action):
        actions.append(action)
        return action

    yield register

    errors = []
    for action in reversed(actions):
        try:
            action()
        except Exception as error:  # noqa: BLE001 - report every failure, not the first
            errors.append(error)
    if errors:
        raise AssertionError(f"Cleanup failed: {errors}")


@pytest.fixture()
def chats(client, cleanup):
    """Create chats through the API and delete them again, whatever the test does."""

    def create(**body) -> dict:
        response = client.post("/api/llm/create-conversation", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload.get("ok") is True, payload
        conversation = payload["conversation"]
        cleanup(lambda target=conversation["target"]: client.post(
            "/api/llm/delete-conversation", json={"target": target}
        ))
        return conversation

    return create


@pytest.fixture()
def connections(client, cleanup):
    """Save SSH connection records and remove them again."""

    def create(**body) -> dict:
        body.setdefault("host", "saved.example.invalid")
        response = client.post("/api/ssh/connections", json=body)
        assert response.status_code == 200, response.text
        connection = response.json()["connection"]
        cleanup(lambda identifier=connection["id"]: client.post(
            "/api/ssh/connections/delete", json={"id": identifier}
        ))
        return connection

    return create


# --- the live machine -----------------------------------------------------------------------


_REACHABILITY: dict[str, str] = {}


def ssh_unreachable_reason(host: str = LIVE_SSH_HOST) -> str:
    """"" when the host answers a batch-mode SSH command, else why it did not.

    Probed once per session and cached: every live test would otherwise pay the
    connect cost again just to decide whether to skip.
    """
    if host in _REACHABILITY:
        return _REACHABILITY[host]
    if shutil.which("ssh") is None:
        _REACHABILITY[host] = "no ssh client on this machine"
        return _REACHABILITY[host]
    try:
        completed = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                host, "echo devagents-api-suite-probe",
            ],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError) as error:
        _REACHABILITY[host] = f"{host} could not be reached: {error}"
        return _REACHABILITY[host]
    if completed.returncode != 0 or "devagents-api-suite-probe" not in completed.stdout:
        detail = (completed.stderr or completed.stdout or "no output").strip().splitlines()
        _REACHABILITY[host] = f"{host} refused a batch-mode SSH login: {detail[-1] if detail else '?'}"
        return _REACHABILITY[host]
    _REACHABILITY[host] = ""
    return ""


def remote_has(command: str, host: str = LIVE_SSH_HOST) -> bool:
    """Whether a CLI exists on the live machine — the skip condition for the PTY cases."""
    key = f"{host}::{command}"
    if key not in _REACHABILITY:
        try:
            completed = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                 f"command -v {command}"],
                capture_output=True, text=True, timeout=25,
            )
            _REACHABILITY[key] = "" if completed.returncode == 0 and completed.stdout.strip() else "missing"
        except (OSError, subprocess.SubprocessError):
            _REACHABILITY[key] = "missing"
    return _REACHABILITY[key] == ""


@pytest.fixture(scope="session")
def live_host() -> str:
    reason = ssh_unreachable_reason()
    if reason:
        pytest.skip(reason)
    return LIVE_SSH_HOST


@pytest.fixture()
def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
