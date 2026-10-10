"""Durable turn ownership and cross-account conversation contracts."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Barrier

import pytest

from subjective_devagents_fastapi.language_models.contracts import Identity, Message, ModelError, ModelResult
from subjective_devagents_fastapi.language_models.store import TurnStore


@pytest.fixture
def identity():
    return Identity("claude-a", "claude", home="/profiles/a", config_dir="/profiles/a/config")


@pytest.fixture
def store(tmp_path):
    return TurnStore(tmp_path / "turns.sqlite")


def reserve(store, identity, conversation_id, request_id="request-1"):
    return store.reserve(conversation_id, identity, request_id, "Explain this code", 30)[0]


def test_concurrent_reservations_across_instances_have_exactly_one_owner(store, identity):
    conversation = store.create(identity)
    stores = [TurnStore(store.path) for _ in range(8)]
    barrier = Barrier(len(stores))

    def attempt(index):
        barrier.wait(timeout=5)
        try:
            return reserve(stores[index], identity, conversation["id"], f"request-{index}")
        except ModelError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        results = list(pool.map(attempt, range(len(stores))))
    winners = [result for result in results if isinstance(result, dict)]
    assert len(winners) == 1
    assert results.count("busy") == 7
    assert store.conversation(conversation["id"])["active_turn_id"] == winners[0]["id"]


def test_idempotency_survives_completion_and_reopening(store, identity):
    conversation = store.create(identity)
    turn = reserve(store, identity, conversation["id"])
    duplicate, created = store.reserve(conversation["id"], identity, "request-1", "Explain this code", 30)
    assert not created
    assert duplicate["id"] == turn["id"]
    store.finish(turn["id"], ModelResult("completed", "Answer", "native-1"))
    reopened = TurnStore(store.path)
    duplicate, created = reopened.reserve(conversation["id"], identity, "request-1", "Explain this code", 30)
    assert not created
    assert duplicate["status"] == "completed"
    assert duplicate["text"] == "Answer"
    assert len(reopened.events(turn["id"])) == 2


@pytest.mark.parametrize("prompt,timeout", [("Different prompt", 30), ("Explain this code", 31)])
def test_request_id_cannot_name_different_payload(store, identity, prompt, timeout):
    conversation = store.create(identity)
    reserve(store, identity, conversation["id"])
    with pytest.raises(ModelError) as error:
        store.reserve(conversation["id"], identity, "request-1", prompt, timeout)
    assert error.value.code == "idempotency_conflict"


def test_concurrent_claims_start_execution_once(store, identity):
    turn = reserve(store, identity, store.create(identity)["id"])
    stores = [TurnStore(store.path) for _ in range(6)]
    barrier = Barrier(len(stores))

    def claim(other):
        barrier.wait(timeout=5)
        return other.claim(turn["id"])

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        assert list(pool.map(claim, stores)).count(True) == 1
    assert [event["data"]["status"] for event in store.events(turn["id"])] == ["queued", "running"]


def test_same_cli_account_cannot_race_credentials_across_conversations(store, identity):
    first = reserve(store, identity, store.create(identity)["id"])
    second_conversation = store.create(identity)
    with pytest.raises(ModelError) as error:
        reserve(TurnStore(store.path), identity, second_conversation["id"])
    assert error.value.code == "identity_busy"
    store.finish(first["id"], ModelResult("interrupted", error_code="remote_outcome_unknown"))
    with pytest.raises(ModelError) as error:
        reserve(store, identity, second_conversation["id"])
    assert error.value.code == "reconciliation_required"
    store.reconcile(first["id"])
    assert reserve(store, identity, second_conversation["id"])["status"] == "queued"


def test_an_interrupted_http_turn_does_not_lock_the_providers_other_conversations(store, identity):
    http = replace(identity, id="legacy-api-proxy", strategy="api_agent", api_base_url="http://127.0.0.1:8317/v1",
                   api_key_env="KEY", model="m")
    first = reserve(store, http, store.create(http)["id"])
    store.finish(first["id"], ModelResult("interrupted", error_code="worker_exception"))
    other = store.create(http)
    assert reserve(store, http, other["id"])["status"] == "queued"


def test_different_cli_accounts_can_run_concurrently(store, identity):
    other = replace(identity, id="claude-b", home="/profiles/b", config_dir="/profiles/b/config")
    first = reserve(store, identity, store.create(identity)["id"])
    second = reserve(store, other, store.create(other)["id"])
    assert store.claim(first["id"]) and store.claim(second["id"])


@pytest.mark.parametrize("running", [False, True])
def test_handoff_waits_for_terminal_turn_then_preserves_history(store, identity, running):
    conversation = store.create(identity)
    turn = reserve(store, identity, conversation["id"])
    target = Identity("codex-b", "codex", home="/profiles/b")
    if running:
        store.claim(turn["id"])
    with pytest.raises(ModelError) as error:
        store.handoff(conversation["id"], target)
    assert error.value.code == "busy"
    assert store.conversation(conversation["id"])["handoffs"] == []
    store.finish(turn["id"], ModelResult("completed", "Answer", "claude-session"))
    switched = store.handoff(conversation["id"], target)
    assert switched["identity"] == target.snapshot()
    assert switched["session_id"] is None
    assert switched["messages"] == [{"role": "user", "content": "Explain this code"}, {"role": "assistant", "content": "Answer"}]
    handoff = switched["handoffs"][0]
    assert json.loads(handoff["previous_identity"]) == identity.snapshot()
    assert json.loads(handoff["next_identity"]) == target.snapshot()
    assert handoff["previous_session_id"] == "claude-session"
    next_turn = reserve(store, target, conversation["id"], "request-2")
    assert next_turn["session_id"] is None
    assert store.turn(turn["id"])["identity"] == identity.snapshot()


@pytest.mark.parametrize("field,value", [("home", "/other"), ("model", "new-model"), ("provider", "codex"), ("ssh_host", "other-host"), ("config_dir", "/new-config")])
def test_identity_snapshot_drift_requires_explicit_handoff(store, identity, field, value):
    conversation = store.create(identity)
    with pytest.raises(ModelError) as error:
        reserve(store, replace(identity, **{field: value}), conversation["id"])
    assert error.value.code == "identity_changed"
    assert store.conversation(conversation["id"])["active_turn_id"] is None


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out", "interrupted"])
def test_unsuccessful_turn_excluded_from_history_and_clears_resume(store, identity, status):
    conversation = store.create(identity)
    first = reserve(store, identity, conversation["id"])
    store.finish(first["id"], ModelResult("completed", "Known answer", "native-1"))
    second = reserve(store, identity, conversation["id"], "request-2")
    assert second["session_id"] == "native-1"
    store.finish(second["id"], ModelResult(status, "Partial unsafe answer", "native-2", "provider_error"))
    reopened = TurnStore(store.path)
    result = reopened.conversation(conversation["id"])
    assert result["session_id"] is None
    assert result["active_turn_id"] is None
    assert len(result["messages"]) == 2
    assert result["messages"][-1]["content"] == "Known answer"
    assert reopened.turn(second["id"])["error_code"] == "provider_error"


def test_cancel_and_event_cursor_survive_restart_without_releasing_active_turn(store, identity):
    conversation = store.create(identity)
    turn = reserve(store, identity, conversation["id"])
    store.claim(turn["id"])
    store.event(turn["id"], "progress", {"step": "working"})
    store.cancel(turn["id"])
    reopened = TurnStore(store.path)
    assert reopened.turn(turn["id"])["cancel_requested"] == 1
    assert reopened.conversation(conversation["id"])["active_turn_id"] == turn["id"]
    events = reopened.events(turn["id"])
    assert [event["kind"] for event in events] == ["state", "state", "progress", "cancel_requested"]
    assert reopened.events(turn["id"], events[1]["sequence"]) == events[2:]
    reopened.finish(turn["id"], ModelResult("cancelled"))
    finished = reopened.events(turn["id"])
    reopened.finish(turn["id"], ModelResult("completed", "Late answer", "late-session"))
    reopened.cancel(turn["id"])
    assert reopened.events(turn["id"]) == finished
    assert reopened.turn(turn["id"])["status"] == "cancelled"


def test_import_keeps_pairs_ordered_without_native_session(store, identity):
    history = (Message("user", "First"), Message("assistant", "One"), Message("user", "Second"), Message("assistant", "Two"))
    conversation = store.create(identity, history)
    assert conversation["messages"] == [{"role": message.role, "content": message.content} for message in history]
    assert conversation["session_id"] is None
    assert conversation["active_turn_id"] is None


@pytest.mark.parametrize("tail", [(Message("user", "unfinished"),), (Message("assistant", "wrong order"), Message("user", "wrong order"))])
def test_invalid_import_rolls_back_conversation_and_preceding_pairs(store, identity, tail):
    with pytest.raises(ModelError) as error:
        store.create(identity, (Message("user", "Valid"), Message("assistant", "Pair")) + tail)
    assert error.value.code == "invalid_history"
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM conversations").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM turns").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["conversation", "turn", "events", "cancel", "finish", "handoff", "reserve"])
def test_unknown_ids_raise_public_not_found(store, identity, operation):
    calls = {
        "conversation": lambda: store.conversation("missing"),
        "turn": lambda: store.turn("missing"),
        "events": lambda: store.events("missing"),
        "cancel": lambda: store.cancel("missing"),
        "finish": lambda: store.finish("missing", ModelResult("completed")),
        "handoff": lambda: store.handoff("missing", identity),
        "reserve": lambda: reserve(store, identity, "missing"),
    }
    with pytest.raises(ModelError) as error:
        calls[operation]()
    assert error.value.code == "not_found"
