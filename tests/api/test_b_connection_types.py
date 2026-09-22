"""B. The five kinds of connection a chat can have, at the API.

A chat carries three fields that between them say where its turn runs: `osUser` (a
local account), `workspace` (a directory, or an SSH target written longhand), and
`sshConnection` (a saved machine). This file creates one chat of each shape over
HTTP and asserts what comes back out — including B8, which does not pass today.
"""
from __future__ import annotations

import pytest

from .helpers import envelope, refusal

CONFIG_ALIAS = "b5-config-alias"


def find_in_listing(client, target: str) -> dict:
    listing = envelope(client.get("/api/llm/conversations"), ok=True)
    matches = [item for item in listing["conversations"] if item["target"] == target]
    assert matches, f"{target} is not in the listing of {listing['total']} conversations"
    return matches[0]


def hydrate(client, target: str) -> dict:
    return envelope(client.get("/api/llm/conversation", params={"target": target}), ok=True)["conversation"]


# --- B1/B2: a local chat --------------------------------------------------------------------


def test_b1_a_local_chat_with_no_os_user_runs_as_the_service_account(client, chats):
    conversation = chats(provider="claude", name="b1-local")

    assert conversation["origin"] == "local"
    # `osUser` is set only when it is *not* the service account's own: send_message reads
    # a set osUser as "route through that user's session".
    assert conversation["osUser"] in (None, ""), conversation["osUser"]
    assert conversation["sourceUser"], "the chat records whose it is even when it is ours"
    assert conversation["provider"] == "claude"

    listed = find_in_listing(client, conversation["target"])
    assert listed["origin"] == "local"
    assert listed["osUser"] in (None, "")


def test_b2_a_local_chat_for_another_account_records_that_account(client, chats, monkeypatch):
    """The chat is routed through the su session manager, which `osUser` is the flag for."""
    from subjective_devagents_fastapi import llm_service

    monkeypatch.setattr(llm_service, "is_valid_username", lambda username: username == "someone-else")
    conversation = chats(provider="claude", name="b2-other-user", osUser="someone-else")

    assert conversation["osUser"] == "someone-else"
    assert conversation["sourceUser"] == "someone-else"
    assert "someone-else" in conversation["description"]

    assert find_in_listing(client, conversation["target"])["osUser"] == "someone-else"
    assert hydrate(client, conversation["target"])["osUser"] == "someone-else"


def test_b2_an_impossible_os_user_is_refused(client, monkeypatch):
    from subjective_devagents_fastapi import llm_service

    monkeypatch.setattr(llm_service, "is_valid_username", lambda username: False)
    error = refusal(client.post("/api/llm/create-conversation", json={
        "provider": "claude", "name": "b2-bad", "osUser": "root; rm -rf /",
    }))
    assert "username" in error.lower(), error
    assert envelope(client.get("/api/llm/conversations"), ok=True)["total"] == 0


# --- B3/B4/B5: an SSH-provider chat -----------------------------------------------------------


def test_b3_a_saved_connection_reference_resolves_to_its_key_and_options(client, connections, chats):
    """`saved:<id>` is stored, not expanded, so the credentials can change later."""
    saved = connections(
        name="b3", host="b3.example.invalid", username="deploy", port=2222,
        auth_method="private_key_path", private_key_path="/home/deploy/.ssh/id_ed25519",
        host_key_policy="accept-new", working_directory="/srv/app",
    )
    conversation = chats(provider="ssh", name="b3-chat", workspace=f"saved:{saved['id']}")
    assert conversation["workspace"] == f"saved:{saved['id']}"

    from subjective_devagents_fastapi import ssh_connections

    settings = ssh_connections.connection_settings(str(client.chat_home), conversation["workspace"])
    assert settings["host"] == "b3.example.invalid"
    assert settings["username"] == "deploy"
    assert settings["port"] == 2222
    assert settings["private_key_path"] == "/home/deploy/.ssh/id_ed25519"
    assert settings["host_key_policy"] == "accept-new"
    assert settings["working_directory"] == "/srv/app"
    # The name is a label for the picker, not a connection field.
    assert "name" not in settings


def test_b3_a_saved_reference_that_is_deleted_is_reported_not_guessed(client, connections, chats):
    saved = connections(name="b3-doomed", host="b3.example.invalid")
    conversation = chats(provider="ssh", name="b3-orphan", workspace=f"saved:{saved['id']}")
    envelope(client.post("/api/ssh/connections/delete", json={"id": saved["id"]}), ok=True)

    from subjective_devagents_fastapi import ssh_connections

    with pytest.raises(ValueError, match="no longer exists"):
        ssh_connections.connection_settings(str(client.chat_home), conversation["workspace"])
    # The chat itself survives so the operator can repoint it.
    assert find_in_listing(client, conversation["target"])


@pytest.mark.parametrize(
    "workspace,expected",
    [
        pytest.param("deploy@b4.example.invalid", {"host": "b4.example.invalid", "username": "deploy"}, id="user-at-host"),
        pytest.param(
            "deploy@b4.example.invalid:/srv/app",
            {"host": "b4.example.invalid", "username": "deploy", "working_directory": "/srv/app"},
            id="with-path",
        ),
        pytest.param(
            "deploy@b4.example.invalid:2222/srv/app",
            {"host": "b4.example.invalid", "username": "deploy", "port": 2222, "working_directory": "/srv/app"},
            id="with-port",
        ),
        pytest.param(
            "ssh://deploy@b4.example.invalid/srv/app",
            {"host": "b4.example.invalid", "username": "deploy", "working_directory": "/srv/app"},
            id="ssh-url",
        ),
    ],
)
def test_b4_a_bare_target_needs_no_saved_entry(client, chats, workspace, expected):
    conversation = chats(provider="ssh", name="b4-bare", workspace=workspace)
    assert conversation["workspace"] == workspace

    from subjective_devagents_fastapi import ssh_connections

    settings = ssh_connections.connection_settings(str(client.chat_home), workspace)
    for key, value in expected.items():
        assert settings[key] == value, (workspace, key, settings)

    # Nothing was saved: a longhand target creates no connection record.
    assert envelope(client.get("/api/ssh/connections"), ok=True)["connections"] == []


def test_b5_a_config_alias_is_resolved_from_the_servers_own_config(client, chats, monkeypatch):
    """An alias is already a saved connection. Listing it is right; copying it is not."""
    from subjective_devagents_fastapi import ssh_connections

    monkeypatch.setattr(
        ssh_connections, "list_config_hosts",
        lambda: [{"host": CONFIG_ALIAS, "source": "config"}],
    )
    listed = envelope(client.get("/api/ssh/connections"), ok=True)
    assert [item["host"] for item in listed["configHosts"]] == [CONFIG_ALIAS]

    conversation = chats(provider="ssh", name="b5-alias", workspace=CONFIG_ALIAS)
    settings = ssh_connections.connection_settings(str(client.chat_home), CONFIG_ALIAS)
    assert settings["host"] == CONFIG_ALIAS
    # ssh resolves the account, port and key from its own config; nothing is copied here.
    assert "username" not in settings or not settings.get("username")
    assert "private_key_path" not in settings

    # And the alias is never copied into the connection store.
    assert envelope(client.get("/api/ssh/connections"), ok=True)["connections"] == []
    assert conversation["workspace"] == CONFIG_ALIAS


def test_b5_an_unparseable_workspace_is_refused_rather_than_guessed(client, chats):
    """Connecting somewhere unintended is worse than reporting an incomplete chat."""
    from subjective_devagents_fastapi import ssh_connections

    conversation = chats(provider="ssh", name="b5-nonsense", workspace="/a/local/path")
    with pytest.raises(ValueError, match="no host"):
        ssh_connections.connection_settings(str(client.chat_home), conversation["workspace"])


# --- B6/B7: an agent running over SSH -----------------------------------------------------------


def test_b6_an_agent_over_ssh_keeps_the_machine_and_the_directory_apart(client, connections, chats):
    """`workspace` stays the remote directory; the machine is named by `sshConnection`."""
    saved = connections(name="b6", host="b6.example.invalid", username="deploy")
    conversation = chats(
        provider="claude", name="b6-agent-over-ssh",
        sshConnection=saved["id"], workspace="/srv/checkout",
    )

    assert conversation["provider"] == "claude"
    assert conversation["sshConnection"] == saved["id"]
    assert conversation["workspace"] == "/srv/checkout", "the remote directory was overwritten"

    hydrated = hydrate(client, conversation["target"])
    assert hydrated["sshConnection"] == saved["id"]
    assert hydrated["workspace"] == "/srv/checkout"


def test_b6_a_chat_cannot_be_both_another_user_and_another_machine(client, connections, monkeypatch):
    """`send_message` checks `osUser` first, so a chat carrying both never reaches SSH."""
    from subjective_devagents_fastapi import llm_service

    monkeypatch.setattr(llm_service, "is_valid_username", lambda username: True)
    saved = connections(name="b6-both", host="b6.example.invalid")

    error = refusal(client.post("/api/llm/create-conversation", json={
        "provider": "claude", "name": "b6-both", "sshConnection": saved["id"], "osUser": "someone-else",
    }))
    assert "not both" in error.lower(), error
    assert envelope(client.get("/api/llm/conversations"), ok=True)["total"] == 0


def test_b7_a_chat_naming_a_deleted_connection_survives_and_is_reportable(client, connections, chats):
    """The UI has to be able to mark it broken, which means the chat must still list."""
    saved = connections(name="b7", host="b7.example.invalid")
    conversation = chats(provider="claude", name="b7-orphan", sshConnection=saved["id"], workspace="/srv/app")
    envelope(client.post("/api/ssh/connections/delete", json={"id": saved["id"]}), ok=True)

    listed = find_in_listing(client, conversation["target"])
    assert listed["label"] == "b7-orphan", "the chat vanished when its connection was deleted"

    hydrated = hydrate(client, conversation["target"])
    assert hydrated["sshConnection"] == saved["id"], (
        "the hydrate lost the reference, so the UI cannot say which connection is missing"
    )

    from subjective_devagents_fastapi import ssh_connections

    assert ssh_connections.describe_reference(str(client.chat_home), f"saved:{saved['id']}") == (
        "(deleted connection)"
    )


# --- B8: the known defect --------------------------------------------------------------------------


def test_known_defect_b8_the_conversation_listing_returns_the_ssh_connection(client, connections, chats):
    """KNOWN DEFECT — expected red until the fix lands.

    `create-conversation` stores `sshConnection` and `GET /api/llm/conversation` returns
    it, but the list projection in `llm_service.list_conversations` whitelists the fields
    it copies and omits this one, while keeping `osUser` and `workspace`. The sidebar is
    built from the listing, so every agent-over-SSH chat renders as running on this
    machine.

    The fix is one line: add `"sshConnection": conversation["sshConnection"]` to that
    projection, beside `osUser`.
    """
    saved = connections(name="b8", host="b8.example.invalid", username="deploy")
    conversation = chats(
        provider="claude", name="b8-agent-over-ssh",
        sshConnection=saved["id"], workspace="/srv/checkout",
    )

    # The two fields that are there prove the whitelist itself works.
    listed = find_in_listing(client, conversation["target"])
    assert listed["workspace"] == "/srv/checkout"
    assert "osUser" in listed

    # And the hydrate proves the value exists to be projected.
    assert hydrate(client, conversation["target"])["sshConnection"] == saved["id"]

    assert listed.get("sshConnection") == saved["id"], (
        "GET /api/llm/conversations dropped sshConnection: the chat runs on "
        f"{saved['host']} but the listing cannot say so"
    )


def test_b8_a_chat_with_no_ssh_connection_lists_without_one(client, chats):
    """The other side of B8: a local chat must not gain a spurious connection either."""
    conversation = chats(provider="claude", name="b8-local")
    listed = find_in_listing(client, conversation["target"])
    assert not listed.get("sshConnection"), listed


# --- creation refusals ---------------------------------------------------------------------------------


def test_creating_a_chat_for_an_unknown_provider_falls_back_rather_than_inventing_one(client, chats):
    """An unwritable provider key is not an error: it becomes the default agent."""
    from subjective_devagents_fastapi.llm_service import DEFAULT_PROVIDER

    conversation = chats(provider="definitely-not-an-agent", name="unknown-provider")
    assert conversation["provider"] == DEFAULT_PROVIDER, (
        "an unknown provider produced a chat no datasource can answer"
    )


def test_every_writable_provider_can_be_created_and_listed(client, chats):
    from subjective_devagents_fastapi.llm_service import WRITABLE_PROVIDERS

    for key, provider in WRITABLE_PROVIDERS.items():
        conversation = chats(provider=key, name=f"all-{key}")
        assert conversation["provider"] == key
        listed = find_in_listing(client, conversation["target"])
        assert listed["provider"] == key
        assert listed["label"] == f"all-{key}"
        assert listed["icon"], f"{key} has no icon for the sidebar"
        assert listed["writable"] is True
