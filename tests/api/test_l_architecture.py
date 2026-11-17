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
        body = client.get("/api/architecture/xmi?path=/etc/passwd").json()
        other = client.get("/api/architecture/xmi?path=/etc/passwd&company=etc")
    assert body["found"] is False
    assert body["path"].endswith("/acme/acme/architecture/acme.xmi")
    assert other.status_code == 404 and other.json()["error"] == "company_mismatch"


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


MODEL = '<?xml version="1.0" encoding="UTF-8"?><XMI xmi.version="1.2"><XMI.content/></XMI>'


def _clone(tmp_path, slug="cardnexia"):
    folder = tmp_path / "fs" / slug / slug
    folder.mkdir(parents=True)
    return folder


def test_l2_the_load_carries_a_sha256_and_honours_the_company_name(monkeypatch, tmp_path):
    import hashlib

    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text(MODEL, encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        same = client.get("/api/architecture/xmi?company=cardnexia").json()
        other = client.get("/api/architecture/xmi?company=othercorp")
    assert same["found"] is True and same["sha256"] == hashlib.sha256(MODEL.encode()).hexdigest()
    assert other.status_code == 404
    assert other.json() == {"ok": False, "error": "company_mismatch", "company": "CardNexia"}


def test_l2_save_creates_the_model_and_a_reload_sees_it(monkeypatch, tmp_path):
    clone = _clone(tmp_path)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        saved = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": "", "company": "CardNexia"})
        loaded = client.get("/api/architecture/xmi").json()
    assert saved.status_code == 200, saved.text
    body = saved.json()
    target = clone / "architecture" / "cardnexia.xmi"
    assert body["ok"] is True and body["saved"] is True and body["path"] == str(target)
    assert body["previous_sha256"] == "" and body["size"] == len(MODEL)
    assert target.read_text(encoding="utf-8") == MODEL
    assert oct(target.stat().st_mode & 0o777) == oct(0o664)
    assert loaded["xmi_text"] == MODEL and loaded["sha256"] == body["sha256"]
    assert not [p for p in target.parent.iterdir() if p.name != "cardnexia.xmi"], "temporary file left behind"


def test_l2_save_on_the_loaded_version_replaces_it_and_keeps_the_mode(monkeypatch, tmp_path):
    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text("<XMI>old</XMI>", encoding="utf-8")
    target.chmod(0o640)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        base = client.get("/api/architecture/xmi").json()["sha256"]
        saved = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": base})
    assert saved.status_code == 200 and saved.json()["previous_sha256"] == base
    assert target.read_text(encoding="utf-8") == MODEL
    assert oct(target.stat().st_mode & 0o777) == oct(0o640)


def test_l2_a_stale_save_is_a_conflict_and_changes_nothing(monkeypatch, tmp_path):
    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text("<XMI>agents wrote this</XMI>", encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        stale = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": "0" * 64})
        blind = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": ""})
        current = client.get("/api/architecture/xmi").json()["sha256"]
    for response in (stale, blind):
        assert response.status_code == 409
        assert response.json()["error"] == "conflict" and response.json()["sha256"] == current
    assert target.read_text(encoding="utf-8") == "<XMI>agents wrote this</XMI>"


def test_l2_without_a_base_the_save_overwrites(monkeypatch, tmp_path):
    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text("<XMI>old</XMI>", encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        saved = client.put("/api/architecture/xmi", json={"xmi_text": MODEL})
    assert saved.status_code == 200 and target.read_text(encoding="utf-8") == MODEL


@pytest.mark.parametrize("text,error", [
    ("", "xmi_text_required"),
    ("   ", "xmi_text_required"),
    ("<XMI><open></XMI>", "invalid_xmi"),
    ("not xml at all", "invalid_xmi"),
    ('<!DOCTYPE x [<!ENTITY a "aaaa">]><XMI>&a;</XMI>', "invalid_xmi"),
    ('<!DOCTYPE x [<!ENTITY s SYSTEM "file:///etc/passwd">]><XMI>&s;</XMI>', "invalid_xmi"),
])
def test_l2_only_well_formed_xml_without_entities_is_saved(monkeypatch, tmp_path, text, error):
    clone = _clone(tmp_path)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        response = client.put("/api/architecture/xmi", json={"xmi_text": text})
    assert response.status_code == 400 and response.json()["error"] == error
    assert not (clone / "architecture").exists()


def test_l2_save_refuses_bad_requests(monkeypatch, tmp_path):
    _clone(tmp_path)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        other = client.put("/api/architecture/xmi?company=othercorp", json={"xmi_text": MODEL})
        other_body = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "company": "OtherCorp"})
        bad_base = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": "abc"})
        not_json = client.put("/api/architecture/xmi", content=b"<XMI/>", headers={"Content-Type": "application/json"})
        not_object = client.put("/api/architecture/xmi", json=[MODEL])
    assert other.status_code == 404 and other.json()["error"] == "company_mismatch"
    assert other_body.status_code == 404 and other_body.json()["error"] == "company_mismatch"
    assert bad_base.status_code == 400 and bad_base.json()["error"] == "invalid_base_sha256"
    assert not_json.status_code == 400 and not_json.json()["error"] == "invalid_json"
    assert not_object.status_code == 400 and not_object.json()["error"] == "invalid_json"


def test_l2_save_needs_the_cloned_company_folder(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        response = client.put("/api/architecture/xmi", json={"xmi_text": MODEL})
    assert response.status_code == 404 and response.json()["error"] == "company_folder_missing"
    assert not (tmp_path / "fs" / "cardnexia").exists()


def test_l2_save_too_large_is_refused(monkeypatch, tmp_path):
    from subjective_devagents_fastapi import architecture_file

    _clone(tmp_path)
    monkeypatch.setattr(architecture_file, "MAX_BYTES", 16)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        response = client.put("/api/architecture/xmi", json={"xmi_text": MODEL})
    assert response.status_code == 413 and response.json()["error"] == "file_too_large"


def test_l2_save_reports_a_read_only_file(monkeypatch, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text("<XMI>old</XMI>", encoding="utf-8")
    target.parent.chmod(0o555)
    try:
        with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
            response = client.put("/api/architecture/xmi", json={"xmi_text": MODEL})
    finally:
        target.parent.chmod(0o755)
    assert response.status_code == 403 and response.json()["error"] == "permission_denied"
    assert target.read_text(encoding="utf-8") == "<XMI>old</XMI>"


def test_l2_save_is_behind_the_gate(monkeypatch, tmp_path):
    clone = _clone(tmp_path)
    with _client(
        monkeypatch,
        tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="owner",
        SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia",
    ) as client:
        response = client.put("/api/architecture/xmi", json={"xmi_text": MODEL})
    assert response.status_code == 401
    assert not (clone / "architecture").exists()
