"""GET /api/architecture/xmi: the company's UML model from this server's filesystem."""
from __future__ import annotations

import os
import re

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
    files = list((clone / "architecture").iterdir())
    assert len(files) == 1, "temporary file left behind"
    target = files[0]
    assert re.fullmatch(r"\d{4}(_\d{2}){5}-cardnexia\.xmi", target.name)
    assert body["ok"] is True and body["saved"] is True and body["path"] == str(target)
    assert body["previous_sha256"] == "" and body["previous_path"] == "" and body["size"] == len(MODEL)
    assert target.read_text(encoding="utf-8") == MODEL
    assert oct(target.stat().st_mode & 0o777) == oct(0o664)
    assert loaded["xmi_text"] == MODEL and loaded["sha256"] == body["sha256"] and loaded["path"] == str(target)


def test_l2_save_on_the_loaded_version_adds_a_version_and_keeps_the_mode(monkeypatch, tmp_path):
    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text("<XMI>old</XMI>", encoding="utf-8")
    target.chmod(0o640)
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        base = client.get("/api/architecture/xmi").json()["sha256"]
        saved = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": base})
    body = saved.json()
    assert saved.status_code == 200 and body["previous_sha256"] == base and body["previous_path"] == str(target)
    assert target.read_text(encoding="utf-8") == "<XMI>old</XMI>", "the loaded file is never changed"
    new = target.parent / os.path.basename(body["path"])
    assert new != target and new.read_text(encoding="utf-8") == MODEL
    assert oct(new.stat().st_mode & 0o777) == oct(0o640)


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


def test_l2_without_a_base_the_save_is_not_checked(monkeypatch, tmp_path):
    target = _clone(tmp_path) / "architecture" / "cardnexia.xmi"
    target.parent.mkdir()
    target.write_text("<XMI>old</XMI>", encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        saved = client.put("/api/architecture/xmi", json={"xmi_text": MODEL})
        loaded = client.get("/api/architecture/xmi").json()
    assert saved.status_code == 200 and loaded["xmi_text"] == MODEL
    assert target.read_text(encoding="utf-8") == "<XMI>old</XMI>"


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


def test_l3_the_latest_dated_version_opens(monkeypatch, tmp_path):
    folder = _clone(tmp_path) / "architecture"
    folder.mkdir()
    (folder / "cardnexia.xmi").write_text("<XMI>plain</XMI>", encoding="utf-8")
    (folder / "2026_11_01_09_00_00-cardnexia.xmi").write_text("<XMI>old</XMI>", encoding="utf-8")
    (folder / "2026_11_20_18_30_05-cardnexia.xmi").write_text("<XMI>newest</XMI>", encoding="utf-8")
    (folder / "2026_11_05_10_00_00-cardnexia.xmi").write_text("<XMI>middle</XMI>", encoding="utf-8")
    # Not this company's versions, or not a version name at all: ignored.
    (folder / "2099_01_01_00_00_00-othercorp.xmi").write_text("<XMI>other</XMI>", encoding="utf-8")
    (folder / "2099_01_01_00_00_00-cardnexia.xmi.bak").write_text("<XMI>bak</XMI>", encoding="utf-8")
    (folder / "2099_01_01-cardnexia.xmi").write_text("<XMI>short</XMI>", encoding="utf-8")
    (folder / "2099_12_31_00_00_00-cardnexia.xmi").mkdir()
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        body = client.get("/api/architecture/xmi").json()
    assert body["found"] is True and body["xmi_text"] == "<XMI>newest</XMI>"
    assert body["path"] == str(folder / "2026_11_20_18_30_05-cardnexia.xmi")


def test_l3_without_versions_the_plain_file_opens(monkeypatch, tmp_path):
    folder = _clone(tmp_path) / "architecture"
    folder.mkdir()
    (folder / "cardnexia.xmi").write_text("<XMI>plain</XMI>", encoding="utf-8")
    (folder / "notes-cardnexia.xmi").write_text("<XMI>no</XMI>", encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        body = client.get("/api/architecture/xmi").json()
    assert body["xmi_text"] == "<XMI>plain</XMI>" and body["path"] == str(folder / "cardnexia.xmi")


def test_l3_a_save_writes_a_new_latest_version(monkeypatch, tmp_path):
    folder = _clone(tmp_path) / "architecture"
    folder.mkdir()
    (folder / "cardnexia.xmi").write_text("<XMI>plain</XMI>", encoding="utf-8")
    # Dated in the future: the save must still sort after it, or the editor would not open it.
    latest = folder / "2099_11_20_18_30_05-cardnexia.xmi"
    latest.write_text("<XMI>newest</XMI>", encoding="utf-8")
    with _client(monkeypatch, tmp_path, SUBJECTIVE_DEVAGENTS_COMPANY_NAME="CardNexia") as client:
        loaded = client.get("/api/architecture/xmi").json()
        saved = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": loaded["sha256"]})
        reloaded = client.get("/api/architecture/xmi").json()
        stale = client.put("/api/architecture/xmi", json={"xmi_text": MODEL, "base_sha256": loaded["sha256"]})
    body = saved.json()
    assert saved.status_code == 200 and body["previous_path"] == str(latest)
    assert body["path"] == str(folder / "2099_11_20_18_30_06-cardnexia.xmi")
    assert reloaded["path"] == body["path"] and reloaded["xmi_text"] == MODEL
    assert latest.read_text(encoding="utf-8") == "<XMI>newest</XMI>"
    assert (folder / "cardnexia.xmi").read_text(encoding="utf-8") == "<XMI>plain</XMI>"
    assert stale.status_code == 409 and stale.json()["sha256"] == body["sha256"]


@pytest.mark.parametrize("names,now,expected", [
    ([], 1_800_000_000, "2027_01_15_08_00_00-acme.xmi"),
    (["2020_01_01_00_00_00-acme.xmi"], 1_800_000_000, "2027_01_15_08_00_00-acme.xmi"),
    (["2027_01_15_08_00_00-acme.xmi"], 1_800_000_000, "2027_01_15_08_00_01-acme.xmi"),
    (["2099_12_31_23_59_59-acme.xmi", "2099_12_31_23_59_59-other.xmi"], 1_800_000_000, "2100_01_01_00_00_00-acme.xmi"),
])
def test_l3_next_version_name(names, now, expected):
    from subjective_devagents_fastapi.architecture_file import next_version_name

    assert next_version_name(names, "acme", now) == expected


@pytest.mark.parametrize("names,expected", [
    ([], ""),
    (["acme.xmi"], ""),
    (["2026_01_02_03_04_05-acme.xmi", "2025_12_31_23_59_59-acme.xmi"], "2026_01_02_03_04_05-acme.xmi"),
    (["2026_01_02_03_04_05-acme-corp.xmi"], ""),
    (["x2026_01_02_03_04_05-acme.xmi"], ""),
])
def test_l3_latest_version_name(names, expected):
    from subjective_devagents_fastapi.architecture_file import latest_version_name

    assert latest_version_name(names, "acme") == expected
