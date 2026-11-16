"""GET /api/architecture/xmi: the company's UML model from this server's filesystem."""
from __future__ import annotations

import os

import pytest

from .conftest import build_client


def _client(monkeypatch, tmp_path, **overrides):
    return build_client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_ARCHITECTURE_ROOT=str(tmp_path / "fs"),
        **overrides,
    )


def test_l1_reads_the_company_file(monkeypatch, tmp_path):
    target = tmp_path / "fs" / "cardnexia" / "cardnexia" / "architecture" / "cardnexia.xmi"
    target.parent.mkdir(parents=True)
    target.write_text("<XMI>model</XMI>", encoding="utf-8")
    os.utime(target, (1_800_000_000, 1_800_000_000))
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        response = client.get("/api/architecture/xmi")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True and body["found"] is True
    assert body["path"] == str(target)
    assert body["xmi_text"] == "<XMI>model</XMI>"
    assert body["size"] == len("<XMI>model</XMI>")
    assert body["modified_at"] == "2027-01-15T08:00:00Z"
    assert body["company"] == "CardNexia"


def test_l1_a_missing_file_is_found_false(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="Acme Corp") as client:
        body = client.get("/api/architecture/xmi").json()
    assert body == {
        "ok": True,
        "found": False,
        "path": str(tmp_path / "fs" / "acme-corp" / "acme-corp" / "architecture" / "acme-corp.xmi"),
    }


def test_l1_the_caller_cannot_choose_the_file(monkeypatch, tmp_path):
    secret = tmp_path / "fs" / "etc" / "passwd"
    secret.parent.mkdir(parents=True)
    secret.write_text("root:x:0:0", encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="acme") as client:
        body = client.get("/api/architecture/xmi?path=/etc/passwd&company=etc").json()
    assert body["found"] is False
    assert body["path"].endswith("/acme/acme/architecture/acme.xmi")


def test_l1_no_company_configured_is_503(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import auth

    monkeypatch.setattr(auth, "COMPANY_NAME_FILE", tmp_path / "no-company-file")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME=None) as client:
        response = client.get("/api/architecture/xmi")
    assert response.status_code == 503
    assert response.json() == {"ok": False, "error": "company_not_configured"}


def test_l1_too_large_is_refused(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import architecture_file

    target = tmp_path / "fs" / "acme" / "acme" / "architecture" / "acme.xmi"
    target.parent.mkdir(parents=True)
    target.write_text("x" * 64, encoding="utf-8")
    monkeypatch.setattr(architecture_file, "MAX_BYTES", 16)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="acme") as client:
        body = client.get("/api/architecture/xmi").json()
    assert body["ok"] is False and body["error"] == "file_too_large"


def test_l1_is_behind_the_gate(monkeypatch, tmp_path):
    with _client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="owner",
        SUBJECTIVE_DEVAGENTS_COMPANY_NAME="acme",
    ) as client:
        response = client.get("/api/architecture/xmi")
    assert response.status_code == 401


@pytest.mark.parametrize("name,slug", [("CardNexia", "cardnexia"), ("Gordon Brothers", "gordon-brothers"), ("../..", "")])
def test_l1_company_slug(name, slug):
    from subjective_devagents_fastapi.architecture_file import company_slug

    assert company_slug(name) == slug
