"""K. The agents' read-only workflow routes.

`/api/workflow/files` and `/api/workflow/file` serve the company's published workflow
to agents holding the separate read-only token (X-Workflow-Token). The WorkTwins export
is replaced with a fixed one, so these cases assert on the handlers, not the network.
"""
from __future__ import annotations

import pytest

from .conftest import build_client

WORKFLOW_TOKEN = "w" * 40
HEADERS = {"X-Workflow-Token": WORKFLOW_TOKEN}
EXPORT = {
    "version": {"version": "20261009190101"},
    "files": [
        {"path": "AGENTS.md", "category": "contract", "revision": 1, "content": "# Contract"},
        {"path": "agents/roles/coder_agent.md", "category": "role", "revision": 3, "content": "# coder_agent"},
    ],
}


@pytest.fixture()
def workflow_client(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import agent_workflow

    monkeypatch.setattr(agent_workflow, "fetch_export_as_service", lambda company: EXPORT)
    with build_client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_WORKFLOW_READ_TOKEN=WORKFLOW_TOKEN,
        SUBJECTIVE_DEVAGENTS_WORKFLOW_COMPANY="company_test",
    ) as test_client:
        yield test_client


@pytest.fixture()
def unavailable_client(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import agent_workflow

    def unavailable(company):
        raise agent_workflow.WorkflowUnavailable("WorkTwins is unreachable")

    monkeypatch.setattr(agent_workflow, "fetch_export_as_service", unavailable)
    with build_client(
        monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_WORKFLOW_READ_TOKEN=WORKFLOW_TOKEN
    ) as test_client:
        yield test_client


def test_k_the_published_files_are_listed(workflow_client):
    response = workflow_client.get("/api/workflow/files", headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["company_id"] == "company_test"
    assert body["version"] == "20261009190101"
    assert [item["path"] for item in body["files"]] == ["AGENTS.md", "agents/roles/coder_agent.md"]
    assert all("content" not in item for item in body["files"])


def test_k_one_file_is_read_as_text(workflow_client):
    response = workflow_client.get(
        "/api/workflow/file", params={"path": "workflow/agents/roles/coder_agent.md"}, headers=HEADERS
    )
    assert response.status_code == 200
    assert response.text == "# coder_agent"
    assert response.headers["X-Workflow-Version"] == "20261009190101"
    assert response.headers["X-Workflow-Path"] == "agents/roles/coder_agent.md"


def test_k_an_unpublished_path_is_404(workflow_client):
    response = workflow_client.get("/api/workflow/file", params={"path": "nope.md"}, headers=HEADERS)
    assert response.status_code == 404
    assert response.json()["ok"] is False


@pytest.mark.parametrize("path", ["/api/workflow/files", "/api/workflow/file"])
def test_k_without_the_read_token_both_routes_refuse(workflow_client, path):
    assert workflow_client.get(path).status_code == 401
    assert workflow_client.get(path, headers={"X-Workflow-Token": "x" * 40}).status_code == 401


@pytest.mark.parametrize("path", ["/api/workflow/files", "/api/workflow/file"])
def test_k_an_unreachable_worktwins_is_503(unavailable_client, path):
    response = unavailable_client.get(path, params={"path": "AGENTS.md"}, headers=HEADERS)
    assert response.status_code == 503
    assert "unreachable" in response.json()["error"]
