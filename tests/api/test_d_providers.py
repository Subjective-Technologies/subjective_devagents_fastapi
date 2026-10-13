"""D. Connecting locally to each LLM.

One case per datasource, against the real plugin. These call the installed
`subjective_*_datasource` packages rather than a stub, so a case skips — naming the
CLI — when that agent is not installed on this machine, and is a real reachability
check when it is.
"""
from __future__ import annotations

import shutil

import pytest

from .helpers import envelope, refusal

# The provider key, the CLI that answers it, and the plugin package behind it.
PROVIDERS = {
    "codex": ("codex", "subjective_codex_datasource"),
    "claude": ("claude", "subjective_claude_cli_datasource"),
    "cursor": ("cursor-agent", "subjective_cursor_datasource"),
    "grok": ("grok", "subjective_grok_cli_datasource"),
    "ssh": ("ssh", "subjective_ssh_datasource"),
}
# Agents with no CLI or plugin: they run only through the LLM proxy (ApiAgent).
PROXY_ONLY = {"kimi"}
# `localchats` is a reader, not a writable agent: it imports history rather than
# answering a turn, so it has a datasource but no entry in the new-chat picker.
READ_ONLY_DATASOURCES = {"localchats"}


def cli_missing(provider: str) -> str:
    command = PROVIDERS[provider][0]
    return "" if shutil.which(command) else f"the {command} CLI is not installed on this machine"


# --- D1: the picker's own list -------------------------------------------------------------


def test_d1_the_provider_list_carries_a_key_and_a_label_for_each_agent(client):
    """The picker used to hard-code this in two files, which drifted from the backend."""
    body = envelope(client.get("/api/llm/providers"), ok=True)
    keys = [provider["key"] for provider in body["providers"]]
    assert set(keys) == set(PROVIDERS) | PROXY_ONLY, keys
    assert body["default"] in keys

    for provider in body["providers"]:
        assert provider["label"], f"{provider['key']} has no label for the picker"
        assert isinstance(provider["available"], bool)


def test_d1_availability_is_a_hint_and_not_a_gate(client, chats):
    """A provider whose module is missing still runs through its exported bundle."""
    body = envelope(client.get("/api/llm/providers"), ok=True)
    for provider in body["providers"]:
        conversation = chats(provider=provider["key"], name=f"d1-{provider['key']}")
        assert conversation["provider"] == provider["key"], (
            f"{provider['key']} is listed as creatable but produced a "
            f"{conversation['provider']} chat"
        )


def test_d1_the_list_matches_what_the_backend_will_actually_create(client):
    from subjective_devagents_fastapi.llm_service import DEFAULT_PROVIDER, WRITABLE_PROVIDERS

    body = envelope(client.get("/api/llm/providers"), ok=True)
    assert {provider["key"] for provider in body["providers"]} == set(WRITABLE_PROVIDERS)
    assert body["default"] == DEFAULT_PROVIDER
    for provider in body["providers"]:
        assert provider["label"] == WRITABLE_PROVIDERS[provider["key"]]["label"]


# --- D2: the installed plugin set -------------------------------------------------------------


def test_d2_the_datasource_report_says_where_each_plugin_resolved_from(client):
    """The first thing to check when a provider misbehaves: module, or exported bundle."""
    body = envelope(client.get("/api/datasources"), ok=True)
    report = body["datasources"]

    assert set(report) == set(PROVIDERS) | READ_ONLY_DATASOURCES | PROXY_ONLY, sorted(report)
    for key, entry in report.items():
        assert entry["package"], f"{key} reports no package name"
        assert isinstance(entry["found"], bool)
        assert isinstance(entry["loaded"], bool)
        if entry["loaded"]:
            assert entry["path"], f"{key} loaded but reports no path"
        else:
            assert entry.get("error"), f"{key} failed to load without saying why"

    assert body["pluginRoots"], "no plugin roots were reported"
    assert isinstance(body["usingModules"], bool)


def test_d2_the_two_endpoints_agree_about_availability(client):
    """`available` in the picker is exactly `loaded` in the datasource report."""
    report = envelope(client.get("/api/datasources"), ok=True)["datasources"]
    for provider in envelope(client.get("/api/llm/providers"), ok=True)["providers"]:
        assert provider["available"] == bool(report[provider["key"]]["loaded"]), provider["key"]


# --- D3–D7: a chat per provider, then its status ---------------------------------------------------


@pytest.mark.parametrize("provider", ["codex", "claude", "cursor", "grok"])
def test_d3_to_d7_a_chat_can_be_created_for_each_agent(client, chats, provider):
    conversation = chats(provider=provider, name=f"d-{provider}")
    assert conversation["provider"] == provider
    assert conversation["writable"] is True
    assert conversation["origin"] == "local"
    assert conversation["workspace"], "a chat with no workspace has nowhere to run"


@pytest.mark.parametrize("provider", ["codex", "claude", "cursor", "grok"])
def test_d3_to_d7_status_reports_reachability_or_a_reason(client, chats, provider):
    """Whatever the answer, it has to be a usable one: a status, and an error if not ready."""
    if reason := cli_missing(provider):
        pytest.skip(reason)
    chats(provider=provider, name=f"d-status-{provider}")

    body = envelope(client.get("/api/llm/status", params={"provider": provider}))
    # The route sets `provider` to the key it was asked for, then splats the datasource's
    # own result over it — so a plugin that names itself (`claude_cli`, `grok_cli`) wins.
    assert body["provider"].startswith(provider), body["provider"]
    assert body["status"], f"{provider} reported no status at all"
    if body["status"] != "ready":
        assert str(body.get("error") or "").strip(), (
            f"{provider} is not ready and did not say why: {body}"
        )


@pytest.mark.parametrize("provider", ["codex", "claude", "grok"])
def test_known_defect_status_reports_ok_false_for_a_ready_datasource(client, provider):
    """KNOWN DEFECT — expected red until the fix lands. Found by this suite.

    `api_llm_status` computes `ok` as `bool(result.get("success"))`, but the datasource's
    own status result carries `status: "ready"` and no `success` key at all. So a
    perfectly reachable agent is reported as `{"ok": false, "status": "ready"}`, and any
    caller that branches on `ok` — which is what every other endpoint here teaches it to
    do — concludes the agent is unusable.

    The fix is for the route to derive `ok` from the status it actually got, e.g.
    `bool(result.get("success") or result.get("status") == "ready")`.
    """
    if reason := cli_missing(provider):
        pytest.skip(reason)
    body = envelope(client.get("/api/llm/status", params={"provider": provider}))
    if body["status"] != "ready":
        pytest.skip(f"{provider} is not ready on this machine ({body['status']}), so the flag is right")
    assert body["ok"] is True, (
        f"{provider} reported status 'ready' and ok=False; the UI reads ok"
    )


def test_d_status_for_an_unknown_provider_falls_back_to_the_default(client):
    from subjective_devagents_fastapi.llm_service import DEFAULT_PROVIDER

    body = envelope(client.get("/api/llm/status", params={"provider": "not-an-agent"}))
    assert body["provider"] == DEFAULT_PROVIDER


def test_d_ssh_is_a_provider_but_not_a_local_cli_chat(client, chats):
    """The other end is a shell on another machine, so a workspace is what it needs."""
    conversation = chats(provider="ssh", name="d-ssh", workspace="deploy@d.example.invalid")
    assert conversation["provider"] == "ssh"
    assert conversation["workspace"] == "deploy@d.example.invalid"


# --- D8: an unknown provider ------------------------------------------------------------------------


def test_d8_an_unknown_provider_creates_no_second_chat(client, chats):
    """It falls back to the default agent rather than erroring — but exactly one chat."""
    from subjective_devagents_fastapi.llm_service import DEFAULT_PROVIDER

    conversation = chats(provider="totally-made-up", name="d8")
    assert conversation["provider"] == DEFAULT_PROVIDER

    listing = envelope(client.get("/api/llm/conversations"), ok=True)
    assert listing["total"] == 1, listing


@pytest.mark.parametrize("provider", [None, "", 0, [], {}])
def test_d8_a_missing_provider_field_still_produces_a_usable_chat(client, chats, provider):
    from subjective_devagents_fastapi.llm_service import DEFAULT_PROVIDER

    conversation = chats(provider=provider, name="d8-empty")
    assert conversation["provider"] == DEFAULT_PROVIDER
    assert conversation["writable"] is True


# --- D9: each provider in the listing ------------------------------------------------------------------


def test_d9_each_providers_chat_lists_with_its_own_provider_icon_and_label(client, chats):
    created = {}
    for provider in PROVIDERS:
        conversation = chats(provider=provider, name=f"d9-{provider}")
        created[provider] = conversation["target"]

    listing = envelope(client.get("/api/llm/conversations"), ok=True)
    by_target = {item["target"]: item for item in listing["conversations"]}
    assert listing["total"] == len(PROVIDERS)

    icons = {}
    for provider, target in created.items():
        listed = by_target[target]
        assert listed["provider"] == provider
        assert listed["label"] == f"d9-{provider}"
        assert listed["icon"], f"{provider} has no icon"
        icons[provider] = listed["icon"]

    assert len(set(icons.values())) == len(icons), (
        f"two agents share an icon, so the sidebar cannot tell them apart: {icons}"
    )


def test_d9_a_search_finds_a_chat_by_its_provider_name(client, chats):
    chats(provider="grok", name="d9-searchable-grok")
    chats(provider="claude", name="d9-searchable-claude")

    found = envelope(client.get("/api/llm/conversations", params={"q": "grok"}), ok=True)
    assert [item["label"] for item in found["conversations"]] == ["d9-searchable-grok"]


@pytest.mark.parametrize("limit,expected", [(1, 1), (0, 3), (500, 3), (100000, 3)])
def test_d9_the_listing_limit_is_capped_and_coerced(client, chats, limit, expected):
    for index in range(3):
        chats(provider="claude", name=f"d9-limit-{index}")
    body = envelope(client.get("/api/llm/conversations", params={"limit": limit}), ok=True)
    assert len(body["conversations"]) == expected
    assert body["total"] == 3, "the total must count every match, not the page"


def test_d9_a_nonsense_limit_is_rejected_by_the_query_type_not_the_handler(client, chats):
    """FastAPI coerces `limit` before the handler runs, so this never reaches the store.

    The handler's own `int(limit or 200)` fallback is therefore unreachable from HTTP —
    worth knowing, because it looks like the defence and is not.
    """
    chats(provider="claude", name="d9-nonsense-limit")
    response = client.get("/api/llm/conversations", params={"limit": "not-a-number"})
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["query", "limit"]

    # And the listing still works for a sane caller afterwards.
    assert envelope(client.get("/api/llm/conversations"), ok=True)["total"] == 1


# --- D10: sessions -------------------------------------------------------------------------------------------


def test_d10_activating_a_conversation_then_listing_sessions(client, chats):
    """`GET /api/llm/sessions` had no test of any kind before this one."""
    conversation = chats(provider="claude", name="d10-activate")

    activated = envelope(
        client.post("/api/llm/activate-conversation", json={"target": conversation["target"]}), ok=True
    )
    assert "busy" in activated, "activation must report authoritative busy state"
    assert activated["busy"] is False

    body = envelope(client.get("/api/llm/sessions"), ok=True)
    assert body["currentUser"], "the session listing does not say who this service runs as"
    assert isinstance(body["users"], list) and body["users"], body

    me = next(user for user in body["users"] if user["current"])
    assert me["username"] == body["currentUser"]
    assert me["home"], "a login with no home directory has nowhere to keep its credentials"
    assert isinstance(me["providers"], dict)


def test_d10_sessions_can_be_narrowed_to_one_provider(client):
    """Narrowing flattens each user to that provider's readiness, best candidate first."""
    everything = envelope(client.get("/api/llm/sessions"), ok=True)
    narrowed = envelope(client.get("/api/llm/sessions", params={"provider": "claude"}), ok=True)

    assert narrowed["provider"] == "claude"
    assert narrowed["currentUser"] == everything["currentUser"]
    assert len(narrowed["users"]) == len(everything["users"])

    for user in narrowed["users"]:
        assert set(user) == {"username", "current", "ready", "unknown"}, sorted(user)
        expected = next(item for item in everything["users"] if item["username"] == user["username"])
        assert user["ready"] is bool((expected["providers"].get("claude") or {}).get("ready"))

    # A ready login has to sort above one that is not, or the picker offers the wrong account.
    order = [(not user["ready"], not user["unknown"], user["username"]) for user in narrowed["users"]]
    assert order == sorted(order), order


def test_d10_the_unnarrowed_listing_keeps_the_full_provider_map(client):
    body = envelope(client.get("/api/llm/sessions"), ok=True)
    for user in body["users"]:
        assert isinstance(user["providers"], dict)
        for provider, state in user["providers"].items():
            assert set(state) == {"ready", "unknown"}, (provider, state)


def test_d10_sessions_never_return_a_credential(client):
    """It reports which logins exist, not what is in them."""
    text = client.get("/api/llm/sessions").text.lower()
    for forbidden in ("password", "token", "secret", "api_key", "apikey", "refresh_token"):
        assert forbidden not in text, f"the session listing mentions {forbidden}"


def test_d10_an_unknown_provider_reports_nobody_ready_rather_than_everybody(client):
    body = envelope(client.get("/api/llm/sessions", params={"provider": "not-an-agent"}), ok=True)
    assert body["provider"] == "not-an-agent"
    assert body["users"], "narrowing dropped the users entirely"
    assert all(user["ready"] is False for user in body["users"]), body["users"]
    assert all(user["unknown"] is False for user in body["users"]), body["users"]


def test_d10_activating_an_unknown_conversation_is_refused(client):
    error = refusal(client.post("/api/llm/activate-conversation", json={"target": "llm:claude:local:nope"}))
    assert "not found" in error.lower(), error


def test_d10_activating_with_no_target_is_refused(client):
    refusal(client.post("/api/llm/activate-conversation", json={}))
