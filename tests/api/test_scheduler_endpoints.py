"""HTTP contract for the installation-authenticated scheduler endpoints."""

import pytest

from .conftest import TOKEN


@pytest.fixture
def scheduler_headers(monkeypatch):
    token = "scheduler-api-fixture-token-32-characters"
    monkeypatch.setenv("COORDINATION_SCHEDULER_TOKEN", token)
    monkeypatch.setenv("COLLABORATION_DEFAULT_SCHEDULER", "legacy_multi_agent")
    monkeypatch.delenv("COLLABORATION_DISABLED_SCHEDULERS", raising=False)
    return {"X-Scheduler-Token": token}


def test_scheduler_catalog_requires_service_token_and_reports_default(owner_client, scheduler_headers):
    path = "/api/collaboration/schedulers/catalog"
    assert owner_client.post(path).status_code == 401
    assert owner_client.post(path, headers={"X-DevAgents-Token": TOKEN}).status_code == 401

    response = owner_client.post(path, params={"repository_available": True}, headers=scheduler_headers)
    assert response.status_code == 200
    rows = response.json()["schedulers"]
    assert {row["id"] for row in rows} == {"legacy_multi_agent", "single_owner", "simple_v1"}
    assert [row["id"] for row in rows if row["is_default"]] == ["legacy_multi_agent"]


def test_scheduler_plan_honors_preference_and_refuses_missing_pin(owner_client, scheduler_headers):
    path = "/api/collaboration/schedulers/plan"
    task = {"agents": ["ai_worker"], "preference": "simple_v1"}
    assert owner_client.post(path, json=task).status_code == 401

    response = owner_client.post(path, json=task, headers=scheduler_headers)
    assert response.status_code == 200
    plan = response.json()
    assert plan["scheduler"] == {"id": "simple_v1", "version": "1.0.0"}
    assert plan["decision"]["action"] == "DISPATCH_AGENT"
    assert plan["decision"]["agents"] == ["ai_worker"]

    task["pinned"] = {"id": "legacy_multi_agent", "version": "0.0.1"}
    refused = owner_client.post(path, json=task, headers=scheduler_headers)
    assert refused.status_code == 409
    assert refused.json()["error"] == "SCHEDULER_UNAVAILABLE"


def test_scheduler_plan_rejects_oversized_body(owner_client, scheduler_headers):
    response = owner_client.post(
        "/api/collaboration/schedulers/plan",
        headers={**scheduler_headers, "Content-Type": "application/json"},
        content=b"x" * 24001,
    )
    assert response.status_code == 413

def test_catalog_and_plan_use_existing_company_login(owner_client, monkeypatch):
    from subjective_devagents_fastapi import agent_workflow

    monkeypatch.delenv("COORDINATION_SCHEDULER_TOKEN", raising=False)
    monkeypatch.setenv("COORDINATION_COMPANY_ID", "fixture-company")
    monkeypatch.delenv("COLLABORATION_DISABLED_SCHEDULERS", raising=False)
    calls = []

    def fetch_export(company, authorization, *, opener):
        calls.append(company)
        if authorization != "Bearer member-fixture":
            raise agent_workflow.WorkflowUnavailable("Caller lacks company workflow access")
        return {"version": {"version": "20261114000000"}, "files": []}

    monkeypatch.setattr(agent_workflow, "fetch_export", fetch_export)
    path = "/api/collaboration/schedulers"
    rejected = owner_client.post(path + "/catalog", headers={"Authorization": "Bearer other-company-fixture"})
    assert rejected.status_code == 401
    headers = {"Authorization": "Bearer member-fixture"}
    catalog = owner_client.post(path + "/catalog", headers=headers).json()["schedulers"]
    assert any(row["id"] == "simple_v1" and row["display_name"] == "Direct LLM" for row in catalog)
    response = owner_client.post(path + "/plan", headers=headers,
                                 json={"agents": ["ai_worker"], "preference": "simple_v1"})
    assert response.status_code == 200
    assert response.json()["decision"]["agents"] == ["ai_worker"]
    assert calls == ["fixture-company"] * 3
