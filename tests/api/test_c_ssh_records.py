"""C. SSH connection records, keys and policy — over HTTP.

`tests/test_ssh_connections.py` covers the module directly and covers it well. What
it cannot cover is the route: the status code a bad payload earns, whether an update
to a missing record silently creates one, and whether a stored password can be talked
out of the process through `GET /api/ssh/connections`. Those are the cases here.

`POST /api/ssh/connections/update` had neither unit nor browser coverage before this
file, so editing a saved connection was untested end to end.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from .conftest import UNREACHABLE_HOST
from .helpers import envelope, has_no_secret, no_secret_fields, refusal, tree_has_no_secret

PASSWORD = "c0rrect-horse-battery-staple-ssh"
PASSPHRASE = "passphrase-that-must-never-come-back"

# A real, syntactically valid OpenSSH private key header. The body is filler: nothing
# here ever asks ssh to use it, and a real key in a test file is a real key in a repo.
PRIVATE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    + "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt\n" * 3
    + "-----END OPENSSH PRIVATE KEY-----\n"
)
PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyBodyForTesting test@example.invalid\n"


# --- C1: every authentication method ----------------------------------------------------


@pytest.mark.parametrize(
    "auth_method,extra,has_password",
    [
        pytest.param("agent", {}, False, id="agent"),
        pytest.param("private_key_path", {"private_key_path": "/home/someone/.ssh/id_ed25519"}, False, id="key"),
        pytest.param("password", {"password": PASSWORD}, True, id="password"),
    ],
)
def test_c1_each_auth_method_round_trips(client, connections, auth_method, extra, has_password):
    created = connections(
        name=f"c1-{auth_method}", host="c1.example.invalid", username="someone",
        auth_method=auth_method, **extra,
    )
    assert created["auth_method"] == auth_method
    assert created["hasPassword"] is has_password, created
    no_secret_fields(created)

    listed = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]
    stored = next(item for item in listed if item["id"] == created["id"])
    assert stored["auth_method"] == auth_method
    # `hasPassword` has to reflect reality, not the method that was asked for.
    assert stored["hasPassword"] is has_password
    no_secret_fields(stored)


def test_c1_the_key_method_demands_a_key(client):
    error = refusal(
        client.post("/api/ssh/connections", json={
            "host": "c1.example.invalid", "auth_method": "private_key_path",
        }),
        status=400,
    )
    assert "key" in error.lower(), error


def test_c1_an_unknown_auth_method_is_refused(client):
    error = refusal(
        client.post("/api/ssh/connections", json={"host": "c1.example.invalid", "auth_method": "telepathy"}),
        status=400,
    )
    assert "agent" in error and "password" in error, error


# --- C2: host-key policy ----------------------------------------------------------------


@pytest.mark.parametrize("policy", ["accept-new", "strict", "off"])
def test_c2_each_host_key_policy_round_trips(client, connections, policy):
    created = connections(name=f"c2-{policy}", host="c2.example.invalid", host_key_policy=policy)
    assert created["host_key_policy"] == policy

    listed = envelope(client.get("/api/ssh/connections"), ok=True)
    stored = next(item for item in listed["connections"] if item["id"] == created["id"])
    assert stored["host_key_policy"] == policy
    assert policy in listed["hostKeyPolicies"]


def test_c2_an_unknown_policy_is_refused(client):
    error = refusal(
        client.post("/api/ssh/connections", json={"host": "c2.example.invalid", "host_key_policy": "maybe"}),
        status=400,
    )
    assert "accept-new" in error, error


def test_c2_a_policy_overrides_the_boolean_it_replaced(client, connections):
    """Otherwise the old `strict_host_key_checking` keeps overriding the new choice."""
    created = connections(
        name="c2-both", host="c2.example.invalid",
        host_key_policy="off", strict_host_key_checking=True,
    )
    assert created["host_key_policy"] == "off"
    assert "strict_host_key_checking" not in created, created


# --- C3: what a read may return ----------------------------------------------------------


def test_c3_the_listing_never_returns_a_secret(client, connections):
    connections(
        name="c3", host="c3.example.invalid", username="someone",
        auth_method="password", password=PASSWORD, key_passphrase=PASSPHRASE,
    )
    response = client.get("/api/ssh/connections")
    body = envelope(response, ok=True)

    has_no_secret(response.text, PASSWORD, PASSPHRASE)
    for connection in body["connections"]:
        no_secret_fields(connection)
        assert "hasPassword" in connection and "hasKeyPassphrase" in connection

    stored = body["connections"][0]
    assert stored["hasPassword"] is True
    assert stored["hasKeyPassphrase"] is True


def test_c3_the_secret_is_on_disk_but_only_readable_by_this_account(client, connections):
    """It is a plaintext secret on the server's disk; the file mode is the protection."""
    connections(name="c3-mode", host="c3.example.invalid", auth_method="password", password=PASSWORD)
    store = Path(client.chat_home) / "ssh-connections.json"
    assert store.is_file()
    mode = stat.S_IMODE(store.stat().st_mode)
    assert mode == 0o600, f"the connection store is {oct(mode)}, not 0600"
    assert PASSWORD in store.read_text(encoding="utf-8"), "fixture assumption: it is stored"


def test_c3_the_listing_reports_the_vocabulary_the_form_needs(client):
    body = envelope(client.get("/api/ssh/connections"), ok=True)
    assert body["authMethods"] == ["agent", "private_key_path", "password"]
    assert body["hostKeyPolicies"] == ["accept-new", "strict", "off"]
    assert isinstance(body["keys"], list) and isinstance(body["configHosts"], list)


# --- C4: update ---------------------------------------------------------------------------


def test_c4_a_rename_persists(client, connections):
    created = connections(name="before", host="c4.example.invalid")
    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "name": "after"}),
        ok=True,
    )["connection"]
    assert updated["name"] == "after"
    assert updated["id"] == created["id"]

    listed = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]
    assert [item["name"] for item in listed if item["id"] == created["id"]] == ["after"]


def test_c4_a_host_change_persists(client, connections):
    created = connections(name="c4-host", host="old.example.invalid")
    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "host": "new.example.invalid"}),
        ok=True,
    )["connection"]
    assert updated["host"] == "new.example.invalid"

    listed = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]
    assert next(item for item in listed if item["id"] == created["id"])["host"] == "new.example.invalid"


def test_c4_a_port_change_persists_as_a_number(client, connections):
    created = connections(name="c4-port", host="c4.example.invalid", port=22)
    # The browser sends a form field, which is a string.
    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "port": "2222"}),
        ok=True,
    )["connection"]
    assert updated["port"] == 2222, "a string port was not coerced"


@pytest.mark.parametrize("port", ["0", "65536", "-1", "not-a-port", "22.5"])
def test_c4_an_impossible_port_is_refused(client, connections, port):
    created = connections(name="c4-badport", host="c4.example.invalid")
    error = refusal(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "port": port}),
        status=400,
    )
    assert "port" in error.lower(), error


def test_c4_switching_auth_mode_persists(client, connections):
    created = connections(
        name="c4-auth", host="c4.example.invalid", auth_method="password", password=PASSWORD,
    )
    assert created["hasPassword"] is True

    updated = envelope(
        client.post("/api/ssh/connections/update", json={
            "id": created["id"], "auth_method": "private_key_path",
            "private_key_path": "/home/someone/.ssh/id_ed25519",
        }),
        ok=True,
    )["connection"]
    assert updated["auth_method"] == "private_key_path"
    assert updated["private_key_path"] == "/home/someone/.ssh/id_ed25519"
    no_secret_fields(updated)


def test_c4_an_update_keeps_the_id_and_moves_the_timestamp(client, connections):
    created = connections(name="c4-stamp", host="c4.example.invalid")
    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "name": "moved"}),
        ok=True,
    )["connection"]
    assert updated["id"] == created["id"]
    assert updated["createdAt"] == created["createdAt"]
    assert updated["updatedAt"] >= created["updatedAt"]


# --- C5: updating something that is not there ---------------------------------------------


def test_c5_updating_a_missing_connection_is_a_404_and_creates_nothing(client):
    before = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]

    error = refusal(
        client.post("/api/ssh/connections/update", json={
            "id": "no-such-connection-id", "host": "c5.example.invalid", "name": "ghost",
        }),
        status=404,
    )
    assert "no longer exists" in error.lower(), error

    after = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]
    assert after == before, "a failed update created a connection"


def test_c5_updating_with_no_id_is_refused(client, connections):
    connections(name="c5-guard", host="c5.example.invalid")
    before = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]
    refusal(client.post("/api/ssh/connections/update", json={"host": "c5.example.invalid"}), status=404)
    assert envelope(client.get("/api/ssh/connections"), ok=True)["connections"] == before


# --- C6: an omitted secret is not a cleared secret ------------------------------------------


def test_c6_an_omitted_password_survives_an_update(client, connections):
    """Editing the port must not silently unset the password stored months ago."""
    created = connections(
        name="c6", host="c6.example.invalid", auth_method="password", password=PASSWORD,
    )
    assert created["hasPassword"] is True

    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "port": 2200}),
        ok=True,
    )["connection"]
    assert updated["hasPassword"] is True, "an unrelated edit blanked the stored password"

    store = json.loads((Path(client.chat_home) / "ssh-connections.json").read_text(encoding="utf-8"))
    stored = next(item for item in store["connections"] if item["id"] == created["id"])
    assert stored["password"] == PASSWORD, "the secret itself was lost, not just its flag"


def test_c6_an_omitted_passphrase_survives_an_update(client, connections):
    created = connections(
        name="c6-pass", host="c6.example.invalid",
        auth_method="private_key_path", private_key_path="/home/someone/.ssh/id_ed25519",
        key_passphrase=PASSPHRASE,
    )
    assert created["hasKeyPassphrase"] is True
    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "name": "renamed"}),
        ok=True,
    )["connection"]
    assert updated["hasKeyPassphrase"] is True


def test_c6_an_explicitly_empty_password_does_clear_it(client, connections):
    """The other half of the rule: absent means keep, empty means clear."""
    created = connections(
        name="c6-clear", host="c6.example.invalid", auth_method="password", password=PASSWORD,
    )
    updated = envelope(
        client.post("/api/ssh/connections/update", json={"id": created["id"], "password": ""}),
        ok=True,
    )["connection"]
    assert updated["hasPassword"] is False

    store = (Path(client.chat_home) / "ssh-connections.json").read_text(encoding="utf-8")
    has_no_secret(store, PASSWORD)


# --- C7–C9: uploading a key -----------------------------------------------------------------


def test_c7_an_uploaded_private_key_is_stored_private_and_named_not_returned(client, cleanup):
    response = client.post("/api/ssh/keys", json={"name": "id_ed25519", "content": PRIVATE_KEY})
    body = envelope(response, ok=True)
    key = body["key"]
    cleanup(lambda: client.post("/api/ssh/keys/delete", json={"path": key["path"]}))

    assert key["name"] == "id_ed25519"
    assert key["source"] == "uploaded"
    assert "content" not in key, "the key body came back to the browser"
    has_no_secret(response.text, PRIVATE_KEY.strip())

    path = Path(key["path"])
    assert path.is_file()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"the key is {oct(mode)}; ssh refuses a key others can read"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700, "the key directory is not private"

    listed = envelope(client.get("/api/ssh/connections"), ok=True)["keys"]
    entry = next(item for item in listed if item["path"] == key["path"])
    assert "content" not in entry
    has_no_secret(entry, PRIVATE_KEY.strip())


def test_c7_a_browser_filename_cannot_escape_the_upload_directory(client, cleanup):
    """The name is untrusted input that becomes part of a path."""
    body = envelope(
        client.post("/api/ssh/keys", json={"name": "../../../etc/evil", "content": PRIVATE_KEY}),
        ok=True,
    )
    key = body["key"]
    cleanup(lambda: client.post("/api/ssh/keys/delete", json={"path": key["path"]}))

    uploads = (Path(client.chat_home) / "ssh-keys").resolve()
    assert Path(key["path"]).resolve().parent == uploads, key["path"]
    assert "/" not in key["name"] and ".." not in key["name"], key["name"]


def test_c8_a_public_key_is_refused_with_the_right_file_named(client):
    response = client.post("/api/ssh/keys", json={"name": "id_ed25519.pub", "content": PUBLIC_KEY})
    error = refusal(response, status=400)
    assert ".pub" in error, f"the error did not name the .pub mistake: {error}"
    assert "id_ed25519" in error, f"the error did not name the file to choose instead: {error}"

    uploads = Path(client.chat_home) / "ssh-keys"
    assert not uploads.exists() or not list(uploads.iterdir()), "a refused upload was stored"


@pytest.mark.parametrize(
    "name,content,expect",
    [
        pytest.param("huge", "-----BEGIN OPENSSH PRIVATE KEY-----\n" + "A" * (64 * 1024), "too large", id="oversized"),
        pytest.param("empty", "", "private key", id="empty"),
        pytest.param("blank", "   \n\t\n  ", "private key", id="whitespace"),
        pytest.param("prose", "This is my key, I promise.", "private key", id="not-a-key"),
        pytest.param("certificate", "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n", "private key", id="certificate"),
    ],
)
def test_c9_a_bad_key_upload_is_refused_and_stores_nothing(client, name, content, expect):
    error = refusal(client.post("/api/ssh/keys", json={"name": name, "content": content}), status=400)
    assert expect in error.lower(), error

    uploads = Path(client.chat_home) / "ssh-keys"
    assert not uploads.exists() or not list(uploads.iterdir()), "a refused upload was stored"


# --- C10/C11: deleting a key ----------------------------------------------------------------


def test_c10_an_uploaded_key_can_be_deleted(client):
    key = envelope(
        client.post("/api/ssh/keys", json={"name": "throwaway", "content": PRIVATE_KEY}), ok=True
    )["key"]
    assert Path(key["path"]).is_file()

    envelope(client.post("/api/ssh/keys/delete", json={"path": key["path"]}), ok=True)
    assert not Path(key["path"]).exists()

    listed = envelope(client.get("/api/ssh/connections"), ok=True)["keys"]
    assert all(item["path"] != key["path"] for item in listed)


def test_c10_deleting_an_already_deleted_key_is_a_404(client):
    key = envelope(
        client.post("/api/ssh/keys", json={"name": "twice", "content": PRIVATE_KEY}), ok=True
    )["key"]
    envelope(client.post("/api/ssh/keys/delete", json={"path": key["path"]}), ok=True)
    refusal(client.post("/api/ssh/keys/delete", json={"path": key["path"]}), status=404)


def test_c11_a_key_in_the_servers_own_ssh_directory_is_never_deletable(client):
    """The server's ~/.ssh holds the box's own keys. This endpoint does not reach them."""
    server_ssh = Path(os.environ["SUBJECTIVE_DEVAGENTS_SSH_DIR"])
    victim = server_ssh / "id_ed25519"
    victim.write_text(PRIVATE_KEY, encoding="utf-8")
    victim.chmod(0o600)

    error = refusal(client.post("/api/ssh/keys/delete", json={"path": str(victim)}), status=404)
    assert "cannot be deleted" in error.lower(), error
    assert victim.is_file(), "a key in the server's own ~/.ssh was deleted"


@pytest.mark.parametrize("path", ["/etc/passwd", "~/.ssh/id_rsa", "", "../../../etc/shadow"])
def test_c11_no_path_outside_the_upload_directory_can_be_deleted(client, path):
    refusal(client.post("/api/ssh/keys/delete", json={"path": path}), status=404)


def test_c11_a_traversal_out_of_the_upload_directory_is_refused(client):
    """However the caller spells it: resolved, then checked against the directory."""
    server_ssh = Path(os.environ["SUBJECTIVE_DEVAGENTS_SSH_DIR"])
    victim = server_ssh / "id_rsa"
    victim.write_text(PRIVATE_KEY, encoding="utf-8")

    uploads = Path(client.chat_home) / "ssh-keys"
    uploads.mkdir(parents=True, exist_ok=True)
    spelled = str(uploads / ".." / ".." / server_ssh.name / "id_rsa")

    refusal(client.post("/api/ssh/keys/delete", json={"path": spelled}), status=404)
    assert victim.is_file()


# --- C12/C13: testing a connection ------------------------------------------------------------


def test_c13_testing_is_refused_when_the_ssh_terminal_is_disabled(client):
    """The interlock, not a stack trace — and it has to say which interlock."""
    error = refusal(
        client.post("/api/ssh/connections/test", json={"host": "c13.example.invalid"}),
        status=403,
    )
    assert "SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED" in error, error


def test_c13_testing_is_refused_on_an_unauthenticated_service(monkeypatch, tmp_path):
    """Enabling remote shells on an open service is refused rather than warned about."""
    from .conftest import build_client

    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="development",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=None,
        SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED="1",
    ) as client:
        error = refusal(
            client.post("/api/ssh/connections/test", json={"host": "c13.example.invalid"}),
            status=403,
        )
        assert "SUBJECTIVE_DEVAGENTS_API_TOKEN" in error, error


def test_c12_an_unroutable_host_reports_a_reason_rather_than_raising(ssh_terminal_client):
    """203.0.113.1 is reserved and never answers. A timeout is an answer, not a 500.

    Takes about as long as the datasource's connect timeout; that wait is the test.
    """
    response = ssh_terminal_client.post("/api/ssh/connections/test", json={
        "host": UNREACHABLE_HOST, "username": "nobody", "auth_method": "agent",
        "host_key_policy": "off",
    })
    body = envelope(response, ok=False)
    assert body["status"], "no status was reported for a failed probe"
    assert str(body["error"]).strip(), "a failed probe gave no reason"


def test_c12_testing_a_connection_with_no_host_is_refused(ssh_terminal_client):
    error = refusal(ssh_terminal_client.post("/api/ssh/connections/test", json={}), status=400)
    assert "host" in error.lower(), error


def test_c12_testing_a_saved_connection_that_is_gone_is_refused(ssh_terminal_client):
    error = refusal(
        ssh_terminal_client.post("/api/ssh/connections/test", json={"id": "vanished"}),
        status=400,
    )
    assert "no longer exists" in error.lower(), error


def test_c12_a_probe_never_echoes_the_password_it_was_given(ssh_terminal_client):
    """The response is the one place a supplied secret could come straight back."""
    response = ssh_terminal_client.post("/api/ssh/connections/test", json={
        "host": UNREACHABLE_HOST, "username": "nobody",
        "auth_method": "password", "password": PASSWORD, "host_key_policy": "off",
    })
    assert response.status_code in (200, 400), response.text
    has_no_secret(response.text, PASSWORD)


# --- C14: delete --------------------------------------------------------------------------------


def test_c14_a_deleted_connection_is_gone_from_the_catalog(client):
    created = envelope(
        client.post("/api/ssh/connections", json={"name": "c14", "host": "c14.example.invalid"}),
        ok=True,
    )["connection"]

    envelope(client.post("/api/ssh/connections/delete", json={"id": created["id"]}), ok=True)

    listed = envelope(client.get("/api/ssh/connections"), ok=True)["connections"]
    assert all(item["id"] != created["id"] for item in listed)


def test_c14_deleting_twice_is_a_404(client):
    created = envelope(
        client.post("/api/ssh/connections", json={"name": "c14-twice", "host": "c14.example.invalid"}),
        ok=True,
    )["connection"]
    envelope(client.post("/api/ssh/connections/delete", json={"id": created["id"]}), ok=True)
    error = refusal(client.post("/api/ssh/connections/delete", json={"id": created["id"]}), status=404)
    assert "no such connection" in error.lower(), error


def test_c14_deleting_an_unknown_connection_is_a_404(client):
    refusal(client.post("/api/ssh/connections/delete", json={"id": "never-existed"}), status=404)


# --- the whole-store sweep -------------------------------------------------------------------------


def test_no_secret_reaches_any_response_or_artifact(client, connections, cleanup):
    """Group-wide: create everything secret-bearing, then read everything back."""
    created = connections(
        name="sweep", host="sweep.example.invalid", username="someone",
        auth_method="password", password=PASSWORD, key_passphrase=PASSPHRASE,
    )
    key = envelope(
        client.post("/api/ssh/keys", json={"name": "sweep", "content": PRIVATE_KEY}), ok=True
    )["key"]
    cleanup(lambda: client.post("/api/ssh/keys/delete", json={"path": key["path"]}))

    for response in (
        client.get("/api/ssh/connections"),
        client.get("/api/ssh/sessions"),
        client.post("/api/ssh/connections/update", json={"id": created["id"], "name": "swept"}),
    ):
        has_no_secret(response.text, PASSWORD, PASSPHRASE, PRIVATE_KEY.strip())

    # The one place they are allowed to be is the 0600 store and the 0600 key file.
    allowed = {
        str(Path(client.chat_home) / "ssh-connections.json"),
        key["path"],
    }
    for path in sorted(Path(client.chat_home).rglob("*")):
        if not path.is_file() or str(path) in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert PASSWORD not in text and PASSPHRASE not in text, f"{path} holds a secret"
        assert PRIVATE_KEY.strip() not in text, f"{path} holds an uploaded private key"
