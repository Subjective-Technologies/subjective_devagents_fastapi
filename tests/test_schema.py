"""Tests for the `devagents` D1 schema.

These assert the guarantees the schema is supposed to give us, not that SQLite works:
that a global conversation id survives switching provider, that a provider's own session
id cannot be claimed by two conversations, that token accounting adds up and that
subscription usage never contributes to spend, that pending questions and approvals are
findable, and that deleting a conversation leaves nothing behind.
"""
from __future__ import annotations

import sqlite3

import pytest

from conftest import (
    NOW,
    SCHEMA_PATH,
    SEED_PATH,
    insert,
    make_conversation,
    make_session,
    make_turn,
    new_id,
)


# --- catalog ---------------------------------------------------------------------------


def test_seed_loads_providers_and_models(seeded):
    providers = {row["id"]: row for row in seeded.execute("SELECT * FROM providers")}
    assert {"codex", "claude", "cursor", "anthropic", "openai"} <= set(providers)
    assert providers["cursor"]["access_modes"] == "subscription"
    assert providers["openai"]["access_modes"] == "api"

    models = {row["id"]: row for row in seeded.execute("SELECT * FROM models")}
    assert models["claude-opus-5"]["price_input_per_mtok"] == 15.0
    # Every model must hang off a provider that exists — a broken seed would fail the FK.
    assert all(row["provider_id"] in providers for row in models.values())


def test_seed_is_idempotent(seeded):
    before = seeded.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
    seeded.executescript(SEED_PATH.read_text(encoding="utf-8"))
    after = seeded.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
    assert before == after


def test_account_records_where_a_credential_lives_never_the_secret(seeded):
    columns = {row["name"] for row in seeded.execute("PRAGMA table_info(provider_accounts)")}
    # A column that could tempt someone to paste a key into the database.
    assert not {"api_key", "token", "secret", "password"} & columns
    assert "credential_ref" in columns


def test_auth_mode_is_constrained(seeded):
    with pytest.raises(sqlite3.IntegrityError):
        insert(seeded, "provider_accounts", provider_id="claude", label="bad", auth_mode="carrier_pigeon")


# --- the global id vs the provider's own id ----------------------------------------------


def test_one_global_conversation_spans_many_provider_sessions(seeded):
    """The point of the two-level id: 'Continue as…' and cross-provider handoff start a new
    provider-side session, but the conversation the user sees keeps its identity."""
    conversation = make_conversation(seeded, title="Refactor the parser", os_user="goldenthinker")

    first = make_session(
        seeded, conversation, seq=1, provider_id="claude",
        provider_session_id="9f1c-claude-session", os_user="goldenthinker", start_reason="new",
    )
    # Same human, different OS account: no session can be carried across, because each
    # CLI's session store lives in its own $HOME.
    second = make_session(
        seeded, conversation, seq=2, provider_id="claude",
        provider_session_id="2b7a-claude-session", os_user="worktwins",
        start_reason="os_user_switch", parent_session_id=first,
    )
    # And now a different vendor entirely, carrying the transcript as prompt text.
    third = make_session(
        seeded, conversation, seq=3, provider_id="codex",
        provider_session_id="55e0-codex-session", start_reason="provider_switch", parent_session_id=second,
    )

    rows = seeded.execute(
        "SELECT provider_id, provider_session_id, start_reason FROM conversation_sessions "
        "WHERE conversation_id = ? ORDER BY seq",
        (conversation,),
    ).fetchall()
    assert [row["provider_id"] for row in rows] == ["claude", "claude", "codex"]
    assert [row["start_reason"] for row in rows] == ["new", "os_user_switch", "provider_switch"]
    # Three provider-side ids, one global id.
    assert len({row["provider_session_id"] for row in rows}) == 3
    assert third is not None


def test_provider_session_id_is_unique_per_provider(seeded):
    """A rescanned transcript must map back to exactly one conversation."""
    first = make_conversation(seeded)
    second = make_conversation(seeded)
    make_session(seeded, first, provider_id="claude", provider_session_id="shared-uuid")

    with pytest.raises(sqlite3.IntegrityError):
        make_session(seeded, second, provider_id="claude", provider_session_id="shared-uuid")

    # The same string from a different vendor is a different session, and must be allowed.
    make_session(seeded, second, provider_id="codex", provider_session_id="shared-uuid")
    assert seeded.execute(
        "SELECT COUNT(*) FROM conversation_sessions WHERE provider_session_id = 'shared-uuid'"
    ).fetchone()[0] == 2


def test_many_sessions_may_still_be_awaiting_a_provider_id(seeded):
    """A brand-new session has no provider id until the first reply lands, so several may
    sit NULL at once. SQL treats NULLs as distinct in a unique index, so this holds with or
    without the WHERE clause on that index — the clause is there to keep the index small,
    and this test pins the behaviour callers depend on either way."""
    conversation = make_conversation(seeded)
    make_session(seeded, conversation, seq=1, provider_id="claude", provider_session_id=None)
    make_session(seeded, conversation, seq=2, provider_id="claude", provider_session_id=None)
    assert seeded.execute(
        "SELECT COUNT(*) FROM conversation_sessions WHERE provider_session_id IS NULL"
    ).fetchone()[0] == 2


def test_session_sequence_is_unique_within_a_conversation(seeded):
    conversation = make_conversation(seeded)
    make_session(seeded, conversation, seq=1)
    with pytest.raises(sqlite3.IntegrityError):
        make_session(seeded, conversation, seq=1)


def test_conversation_target_is_unique(seeded):
    make_conversation(seeded, target="llm:claude:local:abc")
    with pytest.raises(sqlite3.IntegrityError):
        make_conversation(seeded, target="llm:claude:local:abc")


# --- the chat text ------------------------------------------------------------------------


def test_user_and_assistant_text_round_trips(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)

    insert(seeded, "messages", conversation_id=conversation, session_id=session, turn_id=turn,
           seq=1, role="user", content="Why does the parser drop the last line?", sent_at=NOW)
    insert(seeded, "messages", conversation_id=conversation, session_id=session, turn_id=turn,
           seq=2, role="assistant", content="Because the buffer is popped before the flush.",
           token_count=11, sent_at=NOW)

    rows = seeded.execute(
        "SELECT role, content FROM messages WHERE turn_id = ? ORDER BY seq", (turn,)
    ).fetchall()
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[0]["content"].startswith("Why does the parser")


def test_raw_content_is_kept_beside_the_cleaned_text(seeded):
    """Envelope stripping is a rule that changes; keeping the original means revising the
    rule never requires re-importing the archive."""
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)
    insert(seeded, "messages", conversation_id=conversation, session_id=session, turn_id=turn,
           seq=1, role="user", content="real question",
           content_raw="<user_query>real question</user_query><git_status>…</git_status>", sent_at=NOW)
    row = seeded.execute("SELECT content, content_raw FROM messages").fetchone()
    assert row["content"] == "real question"
    assert "<user_query>" in row["content_raw"]


def test_role_is_constrained(seeded):
    conversation = make_conversation(seeded)
    with pytest.raises(sqlite3.IntegrityError):
        insert(seeded, "messages", conversation_id=conversation, seq=1, role="wizard",
               content="x", sent_at=NOW)


def test_provider_message_id_is_unique_within_a_session(seeded):
    """Re-importing a transcript must update rows, not duplicate them."""
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    insert(seeded, "messages", conversation_id=conversation, session_id=session, seq=1,
           role="assistant", content="hi", provider_message_id="msg_01", sent_at=NOW)
    with pytest.raises(sqlite3.IntegrityError):
        insert(seeded, "messages", conversation_id=conversation, session_id=session, seq=2,
               role="assistant", content="hi again", provider_message_id="msg_01", sent_at=NOW)


# --- chat elements -------------------------------------------------------------------------


def test_thoughts_are_ordered_and_may_be_redacted(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)

    insert(seeded, "thoughts", conversation_id=conversation, turn_id=turn, seq=1,
           content="The traceback points at the flush, not the parse.", token_count=120)
    insert(seeded, "thoughts", conversation_id=conversation, turn_id=turn, seq=2,
           content="Check whether the buffer is popped first.", token_count=80)
    # Providers may hand back an opaque block that must be replayed verbatim.
    insert(seeded, "thoughts", conversation_id=conversation, turn_id=turn, seq=3,
           content=None, is_redacted=1, signature="opaque-blob", token_count=40)

    rows = seeded.execute("SELECT * FROM thoughts WHERE turn_id = ? ORDER BY seq", (turn,)).fetchall()
    assert len(rows) == 3
    assert rows[2]["content"] is None and rows[2]["is_redacted"] == 1
    assert rows[2]["signature"] == "opaque-blob"
    assert sum(row["token_count"] for row in rows) == 240


def test_document_access_is_queryable_by_path(seeded):
    """'What has any agent read or written under this path?' is the audit question, and it
    must not require scanning tool-call arguments."""
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)
    tool_call = insert(seeded, "tool_calls", conversation_id=conversation, turn_id=turn, seq=1,
                       tool_name="read_file", tool_kind="file")

    insert(seeded, "document_accesses", conversation_id=conversation, turn_id=turn, seq=2,
           tool_call_id=tool_call, access_kind="read", uri="/srv/app/parser.py",
           byte_size=8210, line_start=1, line_end=120, content_sha256="abc123", accessed_at=NOW)
    insert(seeded, "document_accesses", conversation_id=conversation, turn_id=turn, seq=3,
           access_kind="write", uri="/srv/app/parser.py", byte_size=8300, accessed_at=NOW)
    insert(seeded, "document_accesses", conversation_id=conversation, turn_id=turn, seq=4,
           access_kind="read", uri="/etc/hosts", outside_workspace=1, accessed_at=NOW)

    touched = seeded.execute(
        "SELECT access_kind FROM document_accesses WHERE uri = ? ORDER BY seq", ("/srv/app/parser.py",)
    ).fetchall()
    assert [row["access_kind"] for row in touched] == ["read", "write"]

    outside = seeded.execute(
        "SELECT uri FROM document_accesses WHERE outside_workspace = 1"
    ).fetchall()
    assert [row["uri"] for row in outside] == ["/etc/hosts"]


def test_authorization_request_carries_its_decision(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session, status="awaiting_authorization")
    request = insert(seeded, "authorization_requests", conversation_id=conversation, turn_id=turn,
                     seq=1, request_kind="shell_command", title="Run database migration",
                     requested_action="alembic upgrade head", risk_level="destructive",
                     requested_at=NOW)

    pending = seeded.execute("SELECT decision, decided_at FROM authorization_requests WHERE id = ?", (request,)).fetchone()
    assert pending["decision"] == "pending"
    assert pending["decided_at"] is None

    seeded.execute(
        "UPDATE authorization_requests SET decision = 'allowed', decided_by = 'user', "
        "decided_by_user_id = 'u1', scope = 'session', decided_at = ? WHERE id = ?",
        (NOW, request),
    )
    seeded.execute("UPDATE turns SET status = 'running' WHERE id = ?", (turn,))
    decided = seeded.execute("SELECT * FROM authorization_requests WHERE id = ?", (request,)).fetchone()
    assert decided["decision"] == "allowed"
    assert decided["decided_by"] == "user"
    assert decided["scope"] == "session"


def test_authorization_decision_is_constrained(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)
    with pytest.raises(sqlite3.IntegrityError):
        insert(seeded, "authorization_requests", conversation_id=conversation, turn_id=turn, seq=1,
               request_kind="tool_use", title="x", decision="maybe", requested_at=NOW)


def test_user_prompt_holds_options_and_answer(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session, status="awaiting_input")
    prompt = insert(seeded, "user_prompts", conversation_id=conversation, turn_id=turn, seq=1,
                    prompt_kind="choice", header="Auth method", question="Which auth should the API use?",
                    options_json='[{"value":"oauth","label":"OAuth"},{"value":"apikey","label":"API key"}]',
                    asked_at=NOW)

    assert seeded.execute("SELECT status FROM user_prompts WHERE id = ?", (prompt,)).fetchone()["status"] == "pending"

    seeded.execute(
        "UPDATE user_prompts SET status = 'answered', answer_values_json = ?, answer_text = ?, "
        "answered_by_user_id = ?, answered_at = ? WHERE id = ?",
        ('["oauth"]', "OAuth, with PKCE", "u1", NOW, prompt),
    )
    answered = seeded.execute("SELECT * FROM user_prompts WHERE id = ?", (prompt,)).fetchone()
    assert answered["status"] == "answered"
    assert answered["answer_values_json"] == '["oauth"]'
    assert answered["answered_at"] == NOW


def test_attachments_record_direction_and_storage(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)
    message = insert(seeded, "messages", conversation_id=conversation, session_id=session,
                     turn_id=turn, seq=1, role="user", content="see screenshot", sent_at=NOW)
    insert(seeded, "attachments", conversation_id=conversation, turn_id=turn, message_id=message,
           direction="inbound", filename="shot.png", mime_type="image/png", byte_size=44100,
           storage_kind="r2", storage_ref="devagents/att/shot.png")

    row = seeded.execute("SELECT * FROM attachments").fetchone()
    assert row["direction"] == "inbound"
    assert row["storage_ref"].startswith("devagents/")


# --- accounting -----------------------------------------------------------------------------


def test_turn_tokens_roll_up_from_the_model_calls(seeded):
    """An agentic turn makes many model calls, so per-call usage is the source of truth and
    the turn totals are a rollup of it."""
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation, provider_id="anthropic", model_id="claude-opus-5")
    turn = make_turn(seeded, conversation, session)

    calls = [
        {"input_tokens": 1200, "output_tokens": 300, "reasoning_tokens": 150, "cache_read_tokens": 900},
        {"input_tokens": 1800, "output_tokens": 450, "reasoning_tokens": 200, "cache_read_tokens": 1500},
        {"input_tokens": 2100, "output_tokens": 900, "reasoning_tokens": 0, "cache_read_tokens": 1500},
    ]
    for index, call in enumerate(calls, start=1):
        total = call["input_tokens"] + call["output_tokens"] + call["reasoning_tokens"]
        insert(seeded, "usage_records", conversation_id=conversation, session_id=session, turn_id=turn,
               provider_id="anthropic", model_id="claude-opus-5", seq=index, billing_mode="api",
               total_tokens=total, cost_usd=0.0, recorded_at=NOW, **call)

    rolled = seeded.execute(
        "SELECT SUM(input_tokens) AS inp, SUM(output_tokens) AS out, SUM(reasoning_tokens) AS rea, "
        "SUM(total_tokens) AS tot, COUNT(*) AS calls FROM usage_records WHERE turn_id = ?",
        (turn,),
    ).fetchone()
    assert rolled["calls"] == 3
    assert rolled["inp"] == 5100
    assert rolled["out"] == 1650
    assert rolled["rea"] == 350
    assert rolled["tot"] == 7100

    seeded.execute(
        "UPDATE turns SET input_tokens = ?, output_tokens = ?, reasoning_tokens = ?, "
        "total_tokens = ?, model_call_count = ? WHERE id = ?",
        (rolled["inp"], rolled["out"], rolled["rea"], rolled["tot"], rolled["calls"], turn),
    )
    turn_row = seeded.execute("SELECT * FROM turns WHERE id = ?", (turn,)).fetchone()
    assert turn_row["total_tokens"] == 7100
    assert turn_row["model_call_count"] == 3


def test_cost_matches_the_seeded_prices(seeded):
    """Cost is computed from the model's price columns and snapshotted, so the arithmetic
    the writer performs has to agree with the catalog."""
    price = seeded.execute(
        "SELECT price_input_per_mtok, price_output_per_mtok, price_cache_read_per_mtok "
        "FROM models WHERE id = 'claude-opus-5'"
    ).fetchone()

    input_tokens, output_tokens, cache_read = 10_000, 2_000, 50_000
    expected = (
        input_tokens / 1_000_000 * price["price_input_per_mtok"]
        + output_tokens / 1_000_000 * price["price_output_per_mtok"]
        + cache_read / 1_000_000 * price["price_cache_read_per_mtok"]
    )
    assert expected == pytest.approx(0.375)  # 0.15 + 0.15 + 0.075

    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation, provider_id="anthropic", model_id="claude-opus-5")
    turn = make_turn(seeded, conversation, session)
    insert(seeded, "usage_records", conversation_id=conversation, session_id=session, turn_id=turn,
           provider_id="anthropic", model_id="claude-opus-5", billing_mode="api",
           input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read,
           total_tokens=input_tokens + output_tokens, cost_usd=round(expected, 6),
           price_snapshot_json='{"input":15.0,"output":75.0,"cache_read":1.5}', recorded_at=NOW)

    stored = seeded.execute("SELECT cost_usd, price_snapshot_json FROM usage_records").fetchone()
    assert stored["cost_usd"] == pytest.approx(0.375)
    assert "15.0" in stored["price_snapshot_json"]


def test_subscription_usage_counts_tokens_but_not_money(seeded):
    """The unifying rule: a seat still reports tokens worth tracking against the plan, but
    bills nothing per token, so only api rows may move spend."""
    conversation = make_conversation(seeded)
    subscription_session = make_session(seeded, conversation, seq=1, provider_id="claude",
                                        model_id="claude-cli-default")
    api_session = make_session(seeded, conversation, seq=2, provider_id="anthropic",
                               model_id="claude-opus-5")
    subscription_turn = make_turn(seeded, conversation, subscription_session, seq=1)
    api_turn = make_turn(seeded, conversation, api_session, seq=2)

    insert(seeded, "usage_records", conversation_id=conversation, session_id=subscription_session,
           turn_id=subscription_turn, provider_id="claude", model_id="claude-cli-default",
           billing_mode="subscription", input_tokens=30_000, output_tokens=5_000,
           total_tokens=35_000, cost_usd=0.0, recorded_at=NOW)
    insert(seeded, "usage_records", conversation_id=conversation, session_id=api_session,
           turn_id=api_turn, provider_id="anthropic", model_id="claude-opus-5",
           billing_mode="api", input_tokens=10_000, output_tokens=2_000,
           total_tokens=12_000, cost_usd=0.30, recorded_at=NOW)

    costs = seeded.execute("SELECT * FROM v_conversation_costs WHERE conversation_id = ?", (conversation,)).fetchone()
    assert costs["total_tokens"] == 47_000
    assert costs["subscription_tokens"] == 35_000
    assert costs["api_tokens"] == 12_000
    # Both modes are visible in one number of tokens, but only the API call cost money.
    assert costs["cost_usd"] == pytest.approx(0.30)


def test_billing_mode_is_constrained(seeded):
    conversation = make_conversation(seeded)
    with pytest.raises(sqlite3.IntegrityError):
        insert(seeded, "usage_records", conversation_id=conversation, provider_id="claude",
               billing_mode="invoice_later", recorded_at=NOW)


def test_subscription_window_tracks_plan_allowance(seeded):
    """Seat plans meter by rolling window, which cannot be derived from token counts."""
    account = insert(seeded, "provider_accounts", provider_id="claude", label="Claude Max",
                     auth_mode="subscription", os_user="goldenthinker", plan_name="max_20x")
    window = insert(seeded, "account_usage_windows", account_id=account, window_kind="rolling_5h",
                    window_start="2026-09-06T09:00:00Z", window_end="2026-09-06T14:00:00Z",
                    used_tokens=820_000, limit_tokens=1_000_000, utilization_pct=82.0)
    row = seeded.execute("SELECT * FROM account_usage_windows WHERE id = ?", (window,)).fetchone()
    assert row["used_tokens"] / row["limit_tokens"] == pytest.approx(0.82)
    assert row["exhausted_at"] is None


# --- views -------------------------------------------------------------------------------------


def test_turn_timeline_replays_every_element_in_order(seeded):
    """The renderer walks one ordered stream: thinking, then tool use, then the file it
    read, then the approval it asked for, then the question, then the reply."""
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)

    insert(seeded, "messages", conversation_id=conversation, session_id=session, turn_id=turn,
           seq=1, role="user", content="Migrate the database", sent_at=NOW)
    insert(seeded, "thoughts", conversation_id=conversation, turn_id=turn, seq=2,
           content="Need to inspect the current revision first.")
    tool_call = insert(seeded, "tool_calls", conversation_id=conversation, turn_id=turn, seq=3,
                       tool_name="bash", tool_kind="shell", result_text="head: 4f2a")
    insert(seeded, "document_accesses", conversation_id=conversation, turn_id=turn, seq=4,
           tool_call_id=tool_call, access_kind="read", uri="/srv/app/alembic.ini", accessed_at=NOW)
    insert(seeded, "authorization_requests", conversation_id=conversation, turn_id=turn, seq=5,
           request_kind="shell_command", title="Run alembic upgrade head",
           risk_level="destructive", requested_at=NOW)
    insert(seeded, "user_prompts", conversation_id=conversation, turn_id=turn, seq=6,
           question="Take a backup first?", asked_at=NOW)
    insert(seeded, "messages", conversation_id=conversation, session_id=session, turn_id=turn,
           seq=7, role="assistant", content="Waiting on your approval.", sent_at=NOW)

    timeline = seeded.execute(
        "SELECT element_type, label FROM v_turn_timeline WHERE turn_id = ? ORDER BY seq", (turn,)
    ).fetchall()
    assert [row["element_type"] for row in timeline] == [
        "message", "thought", "tool_call", "document", "authorization", "user_prompt", "message",
    ]
    assert timeline[0]["label"] == "user"
    assert timeline[2]["label"] == "bash"
    assert timeline[3]["label"] == "read"


def test_pending_human_actions_spans_conversations(seeded):
    """The approval inbox: one query for everything blocked on a human anywhere."""
    first = make_conversation(seeded, title="Chat A")
    second = make_conversation(seeded, title="Chat B")
    first_turn = make_turn(seeded, first, make_session(seeded, first))
    second_turn = make_turn(seeded, second, make_session(seeded, second))

    insert(seeded, "authorization_requests", conversation_id=first, turn_id=first_turn, seq=1,
           request_kind="file_delete", title="Delete build artifacts", risk_level="high", requested_at=NOW)
    insert(seeded, "user_prompts", conversation_id=second, turn_id=second_turn, seq=1,
           question="Which region should this deploy to?", asked_at=NOW)
    # Already handled — must not appear in the inbox.
    insert(seeded, "authorization_requests", conversation_id=second, turn_id=second_turn, seq=2,
           request_kind="tool_use", title="Read the changelog", decision="auto_allowed",
           decided_by="policy", requested_at=NOW, decided_at=NOW)

    pending = seeded.execute("SELECT * FROM v_pending_human_actions ORDER BY element_type").fetchall()
    assert [row["element_type"] for row in pending] == ["authorization", "user_prompt"]
    assert pending[0]["severity"] == "high"
    assert pending[1]["prompt"].startswith("Which region")


# --- integrity ---------------------------------------------------------------------------------


def test_deleting_a_conversation_leaves_nothing_behind(seeded):
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation)
    turn = make_turn(seeded, conversation, session)
    message = insert(seeded, "messages", conversation_id=conversation, session_id=session,
                     turn_id=turn, seq=1, role="user", content="hi", sent_at=NOW)
    insert(seeded, "thoughts", conversation_id=conversation, turn_id=turn, seq=2, content="thinking")
    tool_call = insert(seeded, "tool_calls", conversation_id=conversation, turn_id=turn, seq=3, tool_name="bash")
    insert(seeded, "document_accesses", conversation_id=conversation, turn_id=turn, seq=4,
           tool_call_id=tool_call, access_kind="read", uri="/x", accessed_at=NOW)
    insert(seeded, "authorization_requests", conversation_id=conversation, turn_id=turn, seq=5,
           request_kind="tool_use", title="x", requested_at=NOW)
    insert(seeded, "user_prompts", conversation_id=conversation, turn_id=turn, seq=6, question="x?", asked_at=NOW)
    insert(seeded, "attachments", conversation_id=conversation, turn_id=turn, message_id=message,
           direction="inbound", filename="a.png")
    insert(seeded, "usage_records", conversation_id=conversation, session_id=session, turn_id=turn,
           provider_id="claude", billing_mode="subscription", recorded_at=NOW)

    seeded.execute("DELETE FROM conversations WHERE id = ?", (conversation,))

    for table in ("conversation_sessions", "turns", "messages", "thoughts", "tool_calls",
                  "document_accesses", "authorization_requests", "user_prompts",
                  "attachments", "usage_records"):
        remaining = seeded.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert remaining == 0, f"{table} still holds rows after the conversation was deleted"


def test_rows_not_attached_to_a_turn_are_still_cleaned_up(seeded):
    """Imported history has no turn structure — its messages hang off the conversation
    alone. Those cannot be left behind by relying on the turn cascade to reach them."""
    conversation = make_conversation(seeded, origin="imported")
    session = make_session(seeded, conversation)
    insert(seeded, "messages", conversation_id=conversation, session_id=session, turn_id=None,
           seq=1, role="user", content="imported, no turn", sent_at=NOW)
    insert(seeded, "attachments", conversation_id=conversation, turn_id=None, message_id=None,
           direction="inbound", filename="orphan.png")
    insert(seeded, "usage_records", conversation_id=conversation, session_id=session, turn_id=None,
           provider_id="claude", billing_mode="unknown", total_tokens=10, recorded_at=NOW)

    seeded.execute("DELETE FROM conversations WHERE id = ?", (conversation,))

    for table in ("messages", "attachments", "usage_records"):
        remaining = seeded.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert remaining == 0, f"{table} kept a turn-less row after the conversation was deleted"


def test_import_source_survives_its_conversation_being_deleted(seeded):
    """The importer's fingerprint ledger must outlive a deleted chat, or the next scan
    re-imports a conversation the user removed on purpose."""
    conversation = make_conversation(seeded)
    insert(seeded, "import_sources", conversation_id=conversation, provider_id="claude",
           source_uri="/home/goldenthinker/.claude/projects/a/b.jsonl",
           fingerprint="1730000000000:8210", import_status="imported")

    seeded.execute("DELETE FROM conversations WHERE id = ?", (conversation,))
    row = seeded.execute("SELECT conversation_id, source_uri FROM import_sources").fetchone()
    assert row is not None
    assert row["conversation_id"] is None


def test_import_source_uri_is_unique(seeded):
    insert(seeded, "import_sources", source_uri="/home/a/chat.jsonl")
    with pytest.raises(sqlite3.IntegrityError):
        insert(seeded, "import_sources", source_uri="/home/a/chat.jsonl")


def test_a_session_cannot_reference_a_missing_conversation(seeded):
    with pytest.raises(sqlite3.IntegrityError):
        make_session(seeded, new_id())


def test_a_provider_in_use_cannot_be_deleted(seeded):
    """RESTRICT on the session's provider: losing the provider row would orphan the only
    record of which vendor answered."""
    conversation = make_conversation(seeded)
    make_session(seeded, conversation, provider_id="claude")
    with pytest.raises(sqlite3.IntegrityError):
        seeded.execute("DELETE FROM providers WHERE id = 'claude'")


def test_deleting_an_account_keeps_the_conversation_history(seeded):
    """Revoking a key or seat must not erase what it was used for."""
    account = insert(seeded, "provider_accounts", provider_id="anthropic", label="key",
                     auth_mode="api", credential_ref="env:ANTHROPIC_API_KEY")
    conversation = make_conversation(seeded)
    session = make_session(seeded, conversation, provider_id="anthropic", account_id=account)
    turn = make_turn(seeded, conversation, session)
    insert(seeded, "usage_records", conversation_id=conversation, session_id=session, turn_id=turn,
           account_id=account, provider_id="anthropic", billing_mode="api",
           total_tokens=500, cost_usd=0.01, recorded_at=NOW)

    seeded.execute("DELETE FROM provider_accounts WHERE id = ?", (account,))

    assert seeded.execute("SELECT COUNT(*) FROM conversation_sessions").fetchone()[0] == 1
    usage = seeded.execute("SELECT account_id, cost_usd FROM usage_records").fetchone()
    assert usage["account_id"] is None
    assert usage["cost_usd"] == pytest.approx(0.01)


def test_no_view_exceeds_d1s_compound_select_limit(db):
    """D1 builds SQLite with SQLITE_MAX_COMPOUND_SELECT = 5, so a sixth UNION branch is
    rejected at deploy time with 'too many terms in compound SELECT'. Plain SQLite accepts
    500 and would never catch it, so count the branches here instead — a view referencing
    another view costs one term, which is how the timeline stays within budget."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in sql.split(";"):
        if "CREATE VIEW" not in statement.upper():
            continue
        name = statement.split("IF NOT EXISTS")[1].split("AS")[0].strip()
        terms = statement.upper().count("UNION") + 1
        assert terms <= 5, f"view {name} has {terms} compound terms; D1 rejects more than 5"


def test_every_table_and_view_is_present(db):
    names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {
        "providers", "models", "provider_accounts", "account_usage_windows",
        "conversations", "conversation_sessions", "turns", "messages",
        "thoughts", "tool_calls", "document_accesses", "authorization_requests",
        "user_prompts", "attachments", "usage_records", "import_sources",
        "v_turn_timeline", "v_conversation_costs", "v_pending_human_actions",
    } <= names


def test_schema_is_reentrant(db):
    """Every statement is IF NOT EXISTS, so applying the file twice must be a no-op — the
    migration path for an already-provisioned database."""
    db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0] == 16
