"""Saved SSH connections: the store, what it refuses, and what it never hands back.

The store holds plaintext passwords, so most of what matters here is negative: the
file's mode, and every read path stripping the secrets before anything can return
them to a browser or a log.
"""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import ssh_connections  # noqa: E402


@pytest.fixture()
def home(tmp_path):
    return str(tmp_path / "app-home")


# --- creating and editing ------------------------------------------------------------


def test_a_connection_round_trips(home):
    created = ssh_connections.create(home, {
        "name": "Agents box", "host": "example.test", "username": "deploy", "port": 2222,
    })
    assert created["id"]
    listed = ssh_connections.list_connections(home)
    assert [item["name"] for item in listed] == ["Agents box"]
    assert listed[0]["port"] == 2222


def test_the_host_stands_in_for_a_missing_name(home):
    created = ssh_connections.create(home, {"host": "example.test"})
    assert created["name"] == "example.test"


def test_a_connection_without_a_host_is_refused(home):
    with pytest.raises(ValueError) as caught:
        ssh_connections.create(home, {"name": "nowhere"})
    assert "host is required" in str(caught.value)


def test_a_key_method_without_a_key_is_refused(home):
    """Otherwise ssh falls back to other keys and the operator's choice is silently ignored."""
    with pytest.raises(ValueError) as caught:
        ssh_connections.create(home, {"host": "h", "auth_method": "private_key_path"})
    assert "Choose a key" in str(caught.value)


def test_an_unknown_auth_method_is_refused(home):
    with pytest.raises(ValueError):
        ssh_connections.create(home, {"host": "h", "auth_method": "telepathy"})


def test_a_nonsense_port_is_refused(home):
    with pytest.raises(ValueError):
        ssh_connections.create(home, {"host": "h", "port": 99999})
    with pytest.raises(ValueError):
        ssh_connections.create(home, {"host": "h", "port": "not-a-number"})


def test_editing_keeps_the_fields_that_were_not_sent(home):
    created = ssh_connections.create(home, {
        "host": "h", "username": "deploy", "auth_method": "password", "password": "hunter2",
    })
    updated = ssh_connections.update(home, created["id"], {"host": "h", "username": "other"})
    assert updated["username"] == "other"
    # The password was not resent, so it must still be there.
    assert updated["hasPassword"] is True
    assert ssh_connections.get(home, created["id"])["password"] == "hunter2"


def test_an_empty_secret_clears_it(home):
    """A form sends "" for a field the operator emptied; that has to mean delete."""
    created = ssh_connections.create(home, {
        "host": "h", "auth_method": "password", "password": "hunter2",
    })
    updated = ssh_connections.update(home, created["id"], {"host": "h", "password": ""})
    assert updated["hasPassword"] is False
    assert "password" not in ssh_connections.get(home, created["id"])


def test_editing_something_that_is_gone_says_so(home):
    with pytest.raises(LookupError):
        ssh_connections.update(home, "nope", {"host": "h"})


def test_deleting(home):
    created = ssh_connections.create(home, {"host": "h"})
    assert ssh_connections.delete(home, created["id"]) is True
    assert ssh_connections.list_connections(home) == []
    assert ssh_connections.delete(home, created["id"]) is False


# --- secrets ---------------------------------------------------------------------------


def test_the_store_is_private_on_disk(home):
    """It holds plaintext passwords; anything wider than 0600 is a leak."""
    ssh_connections.create(home, {"host": "h", "auth_method": "password", "password": "hunter2"})
    mode = stat.S_IMODE(ssh_connections.store_path(home).stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_a_listed_connection_never_carries_its_secrets(home):
    ssh_connections.create(home, {
        "host": "h", "auth_method": "password", "password": "hunter2", "key_passphrase": "letmein",
    })
    listed = ssh_connections.list_connections(home)
    assert "hunter2" not in str(listed)
    assert "letmein" not in str(listed)
    # ...but the form still needs to know one is stored.
    assert listed[0]["hasPassword"] is True
    assert listed[0]["hasKeyPassphrase"] is True


def test_the_secret_is_still_there_for_opening_a_session(home):
    """`get` is the one path that returns them, and it is not reachable from the API."""
    created = ssh_connections.create(home, {
        "host": "h", "auth_method": "password", "password": "hunter2",
    })
    assert ssh_connections.get(home, created["id"])["password"] == "hunter2"


def test_a_corrupt_store_reads_as_empty_rather_than_crashing(home):
    path = ssh_connections.store_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert ssh_connections.list_connections(home) == []


# --- what ~/.ssh already knows -----------------------------------------------------------


@pytest.fixture()
def fake_ssh_dir(tmp_path, monkeypatch):
    directory = tmp_path / "dot-ssh"
    directory.mkdir()
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_DIR", str(directory))
    return directory


def test_keys_are_listed_by_path_and_never_read(fake_ssh_dir):
    (fake_ssh_dir / "id_ed25519").write_text("PRIVATE KEY BODY", encoding="utf-8")
    (fake_ssh_dir / "id_ed25519").chmod(0o600)
    (fake_ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA", encoding="utf-8")
    (fake_ssh_dir / "known_hosts").write_text("", encoding="utf-8")
    (fake_ssh_dir / "config").write_text("", encoding="utf-8")

    keys = ssh_connections.list_keys()
    assert [key["name"] for key in keys] == ["id_ed25519"]
    assert keys[0]["hasPublicKey"] is True
    assert "PRIVATE KEY BODY" not in str(keys)


def test_a_key_others_can_read_is_flagged(fake_ssh_dir):
    """ssh refuses such a key; saying so here beats a failure the operator never sees."""
    loose = fake_ssh_dir / "id_loose"
    loose.write_text("x", encoding="utf-8")
    loose.chmod(0o644)
    assert [key["tooOpen"] for key in ssh_connections.list_keys()] == [True]


def test_config_aliases_are_offered(fake_ssh_dir):
    (fake_ssh_dir / "config").write_text(
        "Host agents-box\n  HostName 10.0.0.1\n  User deploy\n\n"
        "Host web1 web2\n  User root\n",
        encoding="utf-8",
    )
    assert [host["host"] for host in ssh_connections.list_config_hosts()] == [
        "agents-box", "web1", "web2",
    ]


def test_wildcard_config_entries_are_not_offered(fake_ssh_dir):
    """`Host *` is a block of defaults, not a machine anyone can connect to."""
    (fake_ssh_dir / "config").write_text(
        "Host *\n  ServerAliveInterval 30\n\nHost real\n  HostName 10.0.0.2\n", encoding="utf-8"
    )
    assert [host["host"] for host in ssh_connections.list_config_hosts()] == ["real"]


def test_a_missing_ssh_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_DIR", str(tmp_path / "nothing-here"))
    assert ssh_connections.list_keys() == []
    assert ssh_connections.list_config_hosts() == []


# --- resolving what a conversation stored --------------------------------------------------


def test_a_saved_reference_resolves_to_its_settings(home):
    created = ssh_connections.create(home, {
        "name": "Box", "host": "example.test", "username": "deploy", "port": 2222,
        "auth_method": "password", "password": "hunter2",
    })
    settings = ssh_connections.connection_settings(home, f"saved:{created['id']}")
    assert settings["host"] == "example.test"
    assert settings["username"] == "deploy"
    assert settings["port"] == 2222
    # The datasource needs the secret; this is the path that is allowed to carry it.
    assert settings["password"] == "hunter2"
    # ...but not the label, which is ours and means nothing to ssh.
    assert "name" not in settings


def test_a_longhand_target_still_works(home):
    """SSH chats predate this store, and their workspace strings must keep resolving."""
    settings = ssh_connections.connection_settings(home, "deploy@server:/srv/app")
    assert settings["host"] == "server"
    assert settings["username"] == "deploy"


def test_a_deleted_connection_says_what_to_do(home):
    with pytest.raises(ValueError) as caught:
        ssh_connections.connection_settings(home, "saved:gone")
    assert "no longer exists" in str(caught.value)


def test_an_empty_reference_says_what_to_do(home):
    with pytest.raises(ValueError) as caught:
        ssh_connections.connection_settings(home, "")
    assert "no host" in str(caught.value)


def test_a_reference_describes_itself_for_the_ui(home):
    created = ssh_connections.create(home, {"name": "Box", "host": "h", "username": "deploy"})
    assert ssh_connections.describe_reference(home, f"saved:{created['id']}") == "Box (deploy@h)"
    assert ssh_connections.describe_reference(home, "deploy@h") == "deploy@h"
    assert ssh_connections.describe_reference(home, "saved:gone") == "(deleted connection)"
