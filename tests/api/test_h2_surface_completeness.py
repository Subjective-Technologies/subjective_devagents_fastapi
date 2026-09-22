"""The rest of the surface, so I1 covers every operation in the spec.

These routes are not in the assignment's untested list — most have module-level tests
already — but none of them was exercised at the HTTP layer, which is the layer this
suite is for. Each gets a success and a refusal, with the interlocks asserted where a
route has any.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from .conftest import TOKEN, build_client
from .helpers import envelope, refusal

EXEC_ENV = "SUBJECTIVE_DEVAGENTS_EXEC_ENABLED"


# --- POST /api/save-config --------------------------------------------------------------


def test_save_config_round_trips_through_the_public_config(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    applied: list[tuple] = []
    monkeypatch.setattr(
        state.core, "apply_connection_and_reconnect",
        lambda connection, save_bookmark: applied.append((connection, save_bookmark)),
    )

    body = envelope(client.post("/api/save-config", json={
        "active": {"host": "irc.example.invalid", "port": 6697, "nick": "tester"},
        "saveBookmark": True,
    }), ok=True)

    assert applied == [({"host": "irc.example.invalid", "port": 6697, "nick": "tester"}, True)]
    assert {"host", "port", "channel", "nick", "bookmarks"} <= set(body["config"])


def test_save_config_treats_a_bare_body_as_the_connection(client, monkeypatch):
    """The browser posts either `{active: {...}}` or the fields themselves."""
    from subjective_devagents_fastapi.app import state

    applied: list[tuple] = []
    monkeypatch.setattr(
        state.core, "apply_connection_and_reconnect",
        lambda connection, save_bookmark: applied.append((connection, save_bookmark)),
    )
    envelope(client.post("/api/save-config", json={"host": "bare.example.invalid"}), ok=True)
    assert applied[0][0] == {"host": "bare.example.invalid"}


@pytest.mark.parametrize("flag,expected", [(True, True), (False, False), (None, True)])
def test_save_config_only_skips_the_bookmark_when_explicitly_told_to(client, monkeypatch, flag, expected):
    """`is not False` — so an absent flag saves, and only an explicit false does not."""
    from subjective_devagents_fastapi.app import state

    applied: list[tuple] = []
    monkeypatch.setattr(
        state.core, "apply_connection_and_reconnect",
        lambda connection, save_bookmark: applied.append(save_bookmark),
    )
    payload = {"host": "b.example.invalid"}
    if flag is not None:
        payload["saveBookmark"] = flag
    envelope(client.post("/api/save-config", json=payload), ok=True)
    assert applied == [expected]


def test_save_config_with_an_empty_body_is_harmless(client, monkeypatch):
    from subjective_devagents_fastapi.app import state

    monkeypatch.setattr(state.core, "apply_connection_and_reconnect", lambda connection, save_bookmark: None)
    assert envelope(client.post("/api/save-config", json={}), ok=True)["config"]


# --- POST /api/agents/model ----------------------------------------------------------------


def test_setting_an_agent_model_writes_the_runtime_section(client):
    """Never the registry table: a model is runtime metadata, not agent identity."""
    body = envelope(client.post("/api/agents/model", json={"id": "pytest_agent", "model": "claude"}), ok=True)
    assert body["model"] == "claude"

    role = Path(client.tmp_path) / "workflow-agents" / "roles" / "pytest_agent.md"
    text = role.read_text(encoding="utf-8")
    assert "## Runtime" in text
    assert "claude" in text

    registry = (Path(client.tmp_path) / "workflow-agents" / "registry.md").read_text(encoding="utf-8")
    assert "claude" not in registry, "the model leaked into the registry table"

    assert envelope(client.get("/api/agents/role", params={"agent": "pytest_agent"}), ok=True)["model"] == "claude"


def test_changing_an_agent_model_replaces_rather_than_appends(client):
    envelope(client.post("/api/agents/model", json={"id": "pytest_agent", "model": "claude"}), ok=True)
    envelope(client.post("/api/agents/model", json={"id": "pytest_agent", "model": "codex"}), ok=True)

    text = (Path(client.tmp_path) / "workflow-agents" / "roles" / "pytest_agent.md").read_text(encoding="utf-8")
    assert text.count("## Runtime") == 1, "a second Runtime section was appended"
    assert "codex" in text and "claude" not in text


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"id": "", "model": "claude"}, id="no-id"),
        pytest.param({"id": "../escape", "model": "claude"}, id="traversal"),
        pytest.param({"id": "never_registered", "model": "claude"}, id="unknown-agent"),
        pytest.param({}, id="nothing"),
    ],
)
def test_setting_a_model_on_a_bad_agent_is_refused(client, body):
    refusal(client.post("/api/agents/model", json=body))


def test_the_agent_listing_and_a_role_read_agree(client):
    listed = envelope(client.get("/api/agents"), ok=True)
    ids = [agent["id"] for agent in listed["agents"]]
    assert "pytest_agent" in ids

    role = envelope(client.get("/api/agents/role", params={"agent": "pytest_agent"}), ok=True)
    assert role["id"] == "pytest_agent"


# --- POST /api/exec and GET /api/exec/status --------------------------------------------------


def test_exec_status_reports_both_interlocks(client):
    """The UI reads this to decide whether to show a terminal input at all."""
    body = envelope(client.get("/api/exec/status"), ok=True)
    assert body["enabled"] is False
    assert EXEC_ENV in body["reason"]


def test_exec_is_refused_with_403_when_disabled(client):
    """403, not 401: the caller authenticated fine, the capability is switched off."""
    error = refusal(client.post("/api/exec", json={"command": "echo must not run"}), status=403)
    assert EXEC_ENV in error, error


def test_exec_is_refused_on_an_unauthenticated_service(monkeypatch, tmp_path):
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="development",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=None,
        **{EXEC_ENV: "1"},
    ) as client:
        error = refusal(client.post("/api/exec", json={"command": "echo must not run"}), status=403)
        assert "SUBJECTIVE_DEVAGENTS_API_TOKEN" in error, error
        assert envelope(client.get("/api/exec/status"), ok=True)["enabled"] is False


def test_exec_runs_a_command_when_both_interlocks_are_satisfied(monkeypatch, tmp_path):
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
        SUBJECTIVE_DEVAGENTS_EXEC_CWD=str(tmp_path),
        **{EXEC_ENV: "1"},
    ) as client:
        client.headers.update({"X-DevAgents-Token": TOKEN})

        status = envelope(client.get("/api/exec/status"), ok=True)
        assert status["enabled"] is True
        assert status["cwd"] == str(tmp_path)

        body = envelope(client.post("/api/exec", json={"command": "echo devagents-exec-marker"}), ok=True)
        assert "devagents-exec-marker" in body["stdout"]
        assert body["exit_code"] == 0

        # `ok` means the command ran, not that it succeeded: a non-zero exit is data,
        # and the terminal UI needs to tell "it failed" apart from "it never started".
        failing = envelope(client.post("/api/exec", json={"command": "exit 7"}), ok=True)
        assert failing["status"] == "completed"
        assert failing["exit_code"] == 7


def test_exec_is_gated_by_the_token_like_every_other_route(monkeypatch, tmp_path):
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
        **{EXEC_ENV: "1"},
    ) as client:
        assert client.post("/api/exec", json={"command": "echo nope"}).status_code == 401


# --- POST /api/file/open and /api/file/reveal ---------------------------------------------------


@pytest.mark.parametrize("path", ["/api/file/open", "/api/file/reveal"])
def test_the_desktop_only_file_routes_refuse_and_say_why(client, path):
    """They exist so the web UI gets an explanation rather than a 404."""
    error = refusal(client.post(path))
    assert "desktop app" in error, error


@pytest.mark.parametrize("path", ["/api/file/open", "/api/file/reveal"])
def test_the_desktop_only_file_routes_are_still_gated(bare_token_client, path):
    assert bare_token_client.post(path).status_code == 401


# --- POST /api/log -------------------------------------------------------------------------------


def test_a_browser_log_line_reaches_the_service_log(client, capsys):
    assert envelope(client.post("/api/log", json={
        "level": "warn", "message": "renderer failed to bind", "details": {"where": "composer"},
    }), ok=True) == {"ok": True}

    written = capsys.readouterr().out
    assert "renderer failed to bind" in written
    assert "[warn]" in written
    assert "composer" in written


def test_a_browser_log_line_with_nothing_in_it_is_still_accepted(client, capsys):
    assert envelope(client.post("/api/log", json={}), ok=True) == {"ok": True}
    assert "Browser log" in capsys.readouterr().out


def test_the_log_route_is_gated(bare_token_client):
    """Otherwise anyone who finds the host can write into this service's journal."""
    assert bare_token_client.post("/api/log", json={"message": "injected"}).status_code == 401


# --- POST /api/llm/import-chats ----------------------------------------------------------------------


def make_zip(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_importing_something_that_is_not_a_zip_is_refused(client):
    error = refusal(client.post(
        "/api/llm/import-chats",
        files={"file": ("chats.zip", b"this is not a zip file", "application/zip")},
    ))
    assert error.strip()


def test_importing_an_empty_zip_imports_nothing(client):
    response = client.post(
        "/api/llm/import-chats",
        files={"file": ("chats.zip", make_zip({}), "application/zip")},
    )
    body = envelope(response)
    assert not body.get("imported"), body


def test_importing_without_a_file_is_a_422(client):
    """The field is declared `File(...)`, so FastAPI refuses before the handler runs."""
    assert client.post("/api/llm/import-chats").status_code == 422


def test_the_import_route_is_gated(bare_token_client):
    assert bare_token_client.post(
        "/api/llm/import-chats",
        files={"file": ("chats.zip", make_zip({}), "application/zip")},
    ).status_code == 401


# --- GET /ssh and GET /ssh/assets/{name} ---------------------------------------------------------------


def test_the_ssh_console_is_absent_when_the_terminal_is_disabled(client):
    """It cannot turn up on a deployment that has the feature switched off."""
    response = client.get("/ssh")
    assert response.status_code == 404, response.text
    assert "SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED" in response.json()["error"]


def test_the_ssh_console_is_served_when_the_terminal_is_enabled(ssh_terminal_client):
    response = ssh_terminal_client.get("/ssh")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    # A console for opening shells is never worth serving from a cache.
    assert response.headers["Cache-Control"] == "no-store"
    assert "<" in response.text and len(response.text) > 200


@pytest.mark.parametrize("name", ["xterm.js", "xterm.css", "xterm.LICENSE"])
def test_the_vendored_console_assets_are_served_by_exact_name(ssh_terminal_client, name):
    """Vendored rather than loaded from a CDN: the page must work with no internet."""
    response = ssh_terminal_client.get(f"/ssh/assets/{name}")
    assert response.status_code == 200, f"{name}: {response.text[:200]}"
    assert response.content, f"{name} was served empty"
    assert response.headers["Cache-Control"] == "public, max-age=3600"


@pytest.mark.parametrize(
    "name",
    ["xterm.map", "../app.py", "..%2Fapp.py", "ssh_terminal.html", "", "xterm.js.map"],
)
def test_no_other_file_can_be_read_through_the_asset_route(ssh_terminal_client, name):
    """A path parameter here would be a file-read primitive; the allowlist is the fence."""
    response = ssh_terminal_client.get(f"/ssh/assets/{name}")
    assert response.status_code == 404, f"{name} was served: {response.text[:200]}"


def test_the_console_assets_are_absent_when_the_terminal_is_disabled(client):
    assert client.get("/ssh/assets/xterm.js").status_code == 404


def test_the_ssh_console_is_gated(bare_token_client):
    """`/ssh` is not under `/api/`, so this pins which side of the gate it is on."""
    response = bare_token_client.get("/ssh")
    # The terminal is off on this fixture, so 404 is the terminal interlock answering
    # first; what matters is that it never serves the console to an anonymous caller.
    assert response.status_code in (401, 404), response.text
    assert "<html" not in response.text.lower()
