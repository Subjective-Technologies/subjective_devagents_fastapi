"""G. Prompts and selecting options — the round trip.

`tests/test_prompt_options.py` already covers the annotator as a function. What was
missing is the round trip through HTTP: an option offered by `GET /api/llm/conversation`,
selected by the caller, and delivered verbatim to the backend. That is the part that
breaks when the field is renamed or the reply text is re-derived on the client.

G15–G19 are the security-critical half: the remote password prompt, and proving the
secret never reaches a transcript, a history file or a log.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from .helpers import envelope, has_no_secret, refusal

# The annotator attaches its result as `promptOptions`, not `options`.
FIELD = "promptOptions"


def inject(client, target: str, text: str) -> dict:
    """Put words in the assistant's mouth without running a CLI turn."""
    return envelope(
        client.post("/api/llm/test/inject-assistant-message", json={"target": target, "text": text}),
        ok=True,
    )


def hydrate(client, target: str) -> dict:
    return envelope(client.get("/api/llm/conversation", params={"target": target}), ok=True)["conversation"]


def options_of(message: dict | None) -> dict:
    return (message or {}).get(FIELD) or {}


def last_assistant(conversation: dict) -> dict | None:
    for message in reversed(conversation.get("messages") or []):
        if message.get("role") == "assistant":
            return message
    return None


@pytest.fixture()
def chat(test_endpoint_client, cleanup):
    """A chat whose assistant replies this file can dictate."""
    response = test_endpoint_client.post(
        "/api/llm/create-conversation", json={"provider": "claude", "name": "g-prompts"}
    )
    conversation = envelope(response, ok=True)["conversation"]
    cleanup(lambda: test_endpoint_client.post(
        "/api/llm/delete-conversation", json={"target": conversation["target"]}
    ))
    return conversation["target"]


# --- G1–G7: what is and is not a question -------------------------------------------------


def test_g1_a_statement_offers_no_options(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "I have finished refactoring the parser and the tests pass.")
    assert options_of(last_assistant(hydrate(test_endpoint_client, chat))) == {}


@pytest.mark.parametrize(
    "text",
    [
        "I updated the config. Nothing else to report.",
        "Done.",
        "The build succeeded in 42 seconds.",
        "Here is the diff you asked for.",
    ],
)
def test_g1_ordinary_prose_never_grows_buttons(test_endpoint_client, chat, text):
    inject(test_endpoint_client, chat, text)
    assert options_of(last_assistant(hydrate(test_endpoint_client, chat))) == {}


def test_g2_a_plain_question_offers_yes_no_and_a_way_out(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "I can also update the changelog. Shall I continue?")
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))

    assert offered["asks_question"] is True
    assert offered["kind"] == "yes_no"
    assert [option["label"] for option in offered["options"]] == ["Yes", "No"]
    assert offered["allows_other"] is True, "the user must be able to answer in their own words"
    assert offered["other_hint"]
    for option in offered["options"]:
        assert option["description"], f"{option['label']} has no gloss under it"
        assert option["reply"], f"{option['label']} sends nothing"


def test_g3_keep_or_revert_is_detected_with_descriptions(test_endpoint_client, chat):
    inject(
        test_endpoint_client, chat,
        "I rewrote the retry loop. Do you want to keep the changes, or should I revert them?",
    )
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))

    assert offered["kind"] == "keep_revert"
    assert [option["id"] for option in offered["options"]] == ["keep", "revert"]
    assert all(option["description"] for option in offered["options"])


def test_g4_a_permission_request_offers_allow_and_deny(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "This edit needs permission to write outside the repo. Allow?")
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))

    assert offered["kind"] == "allow_deny"
    assert [option["id"] for option in offered["options"]] == ["allow", "deny"]


def test_g5_a_run_request_offers_run_and_do_not_run(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "Shall I run the migration against staging?")
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))

    assert offered["kind"] == "run"
    assert [option["id"] for option in offered["options"]] == ["run", "do-not-run"]


def test_g6_enumerated_choices_each_carry_their_own_description_and_reply(test_endpoint_client, chat):
    inject(
        test_endpoint_client, chat,
        "There are three ways to fix this.\n"
        "A. Cache the parsed config at startup\n"
        "B. Invalidate on file change with a watcher\n"
        "C. Reparse on every read and accept the cost\n"
        "Which would you like — A, B, or C?",
    )
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))

    assert offered["kind"] == "enumerated"
    assert [option["reply"] for option in offered["options"]] == ["A", "B", "C"]
    assert [option["label"] for option in offered["options"]] == ["Option A", "Option B", "Option C"]

    descriptions = [option["description"] for option in offered["options"]]
    assert all(descriptions), f"an enumerated option lost its line: {descriptions}"
    assert descriptions[0].startswith("Cache the parsed config")
    assert descriptions[1].startswith("Invalidate on file change")
    assert descriptions[2].startswith("Reparse on every read")
    assert len(set(descriptions)) == 3, "two options share a description"


def test_g7_a_question_inside_a_code_block_is_ignored(test_endpoint_client, chat):
    inject(
        test_endpoint_client, chat,
        "Here is the prompt the installer prints:\n\n"
        "```\n"
        "Do you want to continue?\n"
        "```\n\n"
        "That is all it does.",
    )
    assert options_of(last_assistant(hydrate(test_endpoint_client, chat))) == {}


def test_g7_a_question_in_inline_code_is_ignored(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "The script asks `Shall I continue?` and then exits.")
    assert options_of(last_assistant(hydrate(test_endpoint_client, chat))) == {}


# --- G8/G9/G10: where options may appear ---------------------------------------------------------


def test_g8_only_the_last_assistant_message_is_annotated(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "Shall I start with the parser?")
    inject(test_endpoint_client, chat, "Shall I do the serializer as well?")
    inject(test_endpoint_client, chat, "Both are done. The suite is green.")

    conversation = hydrate(test_endpoint_client, chat)
    assistants = [message for message in conversation["messages"] if message["role"] == "assistant"]
    assert len(assistants) == 3

    # The last one is a statement, so nothing carries options at all.
    assert all(FIELD not in message for message in assistants), (
        "an earlier assistant message kept its buttons after the agent moved on"
    )


def test_g8_the_earlier_question_loses_its_options_when_a_newer_one_arrives(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "Shall I also update the changelog?")
    first = hydrate(test_endpoint_client, chat)
    assert options_of(last_assistant(first))["kind"] == "yes_no"

    inject(test_endpoint_client, chat, "Shall I run the migration?")
    conversation = hydrate(test_endpoint_client, chat)
    assistants = [message for message in conversation["messages"] if message["role"] == "assistant"]

    assert FIELD not in assistants[0], "the superseded question kept its buttons"
    assert options_of(assistants[-1])["kind"] == "run"


def test_g9_the_hydrate_endpoint_is_what_carries_the_options(test_endpoint_client, chat):
    """The UI reads them from here; the annotation is a view, never stored."""
    inject(test_endpoint_client, chat, "Shall I proceed?")
    assert options_of(last_assistant(hydrate(test_endpoint_client, chat)))["kind"] == "yes_no"

    # The stored transcript itself is clean: the options are recomputed per request.
    state_file = next(Path(test_endpoint_client.chat_home).rglob("*.json"), None)
    if state_file is not None:
        assert FIELD not in state_file.read_text(encoding="utf-8"), (
            "the annotation was persisted into the transcript"
        )


def test_g10_a_non_question_carries_no_options_field_at_all(test_endpoint_client, chat):
    """Absent, not an empty object: the UI tests for the key."""
    inject(test_endpoint_client, chat, "All done.")
    message = last_assistant(hydrate(test_endpoint_client, chat))
    assert FIELD not in message, message


def test_g10_a_user_message_is_never_annotated(test_endpoint_client, chat):
    inject(test_endpoint_client, chat, "Shall I proceed?")
    conversation = hydrate(test_endpoint_client, chat)
    for message in conversation["messages"]:
        if message["role"] != "assistant":
            assert FIELD not in message, message


# --- G11–G14: selecting an option --------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected_replies",
    [
        pytest.param("Shall I continue?", ["Yes, proceed.", "No, do not proceed."], id="yes-no"),
        pytest.param(
            "Do you want to keep the changes or revert them?",
            ["Keep the code and continue.", "Revert those changes."],
            id="keep-revert",
        ),
        pytest.param(
            "This needs permission to write outside the repo. Allow?",
            ["Allow it and continue.", "Do not allow it."],
            id="allow-deny",
        ),
        pytest.param(
            "Shall I run the migration?",
            ["Run it and report the result.", "Do not run it."],
            id="run",
        ),
    ],
)
def test_g11_selecting_each_option_sends_its_reply_verbatim(
    test_endpoint_client, chat, monkeypatch, prompt, expected_replies
):
    """Not only Yes: every option of every kind, and the exact text it sends.

    `send_llm_message` is spied because the reply's *journey* is what is under test, not
    what an agent would say back.
    """
    from subjective_devagents_fastapi.app import state

    sent: list[str] = []
    monkeypatch.setattr(
        state.core, "send_llm_message",
        lambda target, text, files: sent.append(text) or {"ok": True},
    )

    inject(test_endpoint_client, chat, prompt)
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))
    assert [option["reply"] for option in offered["options"]] == expected_replies

    for option in offered["options"]:
        envelope(test_endpoint_client.post("/api/llm/send-message", json={
            "target": chat, "text": option["reply"],
        }), ok=True)

    _wait_for(lambda: len(sent) == len(expected_replies))
    assert sent == expected_replies, (
        "an option's reply was altered between being offered and being delivered"
    )


def test_g11_selecting_an_enumerated_option_sends_its_bare_label(test_endpoint_client, chat, monkeypatch):
    from subjective_devagents_fastapi.app import state

    sent: list[str] = []
    monkeypatch.setattr(
        state.core, "send_llm_message",
        lambda target, text, files: sent.append(text) or {"ok": True},
    )

    inject(
        test_endpoint_client, chat,
        "A. Cache it\nB. Watch the file\nC. Reparse each time\nWhich do you prefer, A, B, or C?",
    )
    offered = options_of(last_assistant(hydrate(test_endpoint_client, chat)))
    for option in offered["options"]:
        envelope(test_endpoint_client.post("/api/llm/send-message", json={
            "target": chat, "text": option["reply"],
        }), ok=True)

    _wait_for(lambda: len(sent) == 3)
    assert sent == ["A", "B", "C"]


def test_g12_selecting_an_option_on_an_unknown_conversation_is_refused(test_endpoint_client):
    error = refusal(test_endpoint_client.post("/api/llm/send-message", json={
        "target": "llm:claude:local:nope", "text": "Yes, proceed.",
    }))
    assert "not found" in error.lower(), error


@pytest.mark.parametrize("reply", ["", "   ", "\n\t ", None])
def test_g13_an_empty_selected_reply_is_rejected_before_the_backend(
    test_endpoint_client, chat, monkeypatch, reply
):
    from subjective_devagents_fastapi.app import state

    sent: list[str] = []
    monkeypatch.setattr(
        state.core, "send_llm_message",
        lambda target, text, files: sent.append(text) or {"ok": True},
    )

    body = envelope(test_endpoint_client.post("/api/llm/send-message", json={
        "target": chat, "text": reply,
    }), ok=False)
    assert body["status"] == "invalid_request"
    assert sent == [], "an empty reply reached the agent"


def test_g14_the_reply_endpoint_is_gated_when_a_token_is_set(bare_token_client):
    from .conftest import TOKEN

    assert bare_token_client.post("/api/llm/send-message", json={
        "target": "anything", "text": "Yes, proceed.",
    }).status_code == 401

    # And with the credential it gets past the gate to a real (not-found) answer.
    allowed = bare_token_client.post(
        "/api/llm/send-message",
        json={"target": "llm:claude:local:nope", "text": "Yes, proceed."},
        headers={"X-DevAgents-Token": TOKEN},
    )
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "not_found"


def test_g14_the_injection_endpoint_is_off_unless_explicitly_enabled(client, chats):
    """It lets a caller put words in an assistant's mouth. Production must never have it."""
    conversation = chats(provider="claude", name="g14-injection")
    error = refusal(client.post("/api/llm/test/inject-assistant-message", json={
        "target": conversation["target"], "text": "Shall I proceed?",
    }))
    assert "disabled" in error.lower(), error


# --- G15/G19: detecting a password prompt --------------------------------------------------------------


@pytest.mark.parametrize(
    "tail,who",
    [
        pytest.param("[sudo] password for deploy:", "deploy", id="sudo"),
        pytest.param("deploy@server's password:", "deploy@server", id="ssh-password"),
        pytest.param("Enter passphrase for key '/home/deploy/.ssh/id_ed25519':", "id_ed25519", id="passphrase"),
        pytest.param("Password:", "", id="bare-password"),
        pytest.param("Passphrase:", "", id="bare-passphrase"),
        pytest.param("Verification code:", "", id="2fa"),
    ],
)
def test_g15_a_blocked_prompt_is_detected_on_the_last_line(tail, who):
    from subjective_devagents_fastapi.ssh_chat import secret_prompt

    detected = secret_prompt(f"Running the deploy script\nsome output\n{tail}")
    assert detected, f"{tail!r} was not recognised as a prompt"
    if who:
        assert who in detected, (who, detected)


@pytest.mark.parametrize(
    "output",
    [
        pytest.param("Password: hunter2 was rejected\n$ ", id="returned-to-prompt"),
        pytest.param("The log line said Password: and then carried on\ndone\n", id="mid-output"),
        pytest.param("grep -n 'Password:' auth.log\nauth.log:14:Password:\n$ ", id="grep-result"),
        pytest.param("Password:\nAuthentication succeeded\n$ ", id="already-answered"),
        pytest.param("", id="nothing-at-all"),
    ],
)
def test_g19_output_that_merely_mentions_a_password_does_not_trigger_one(output):
    """Only the last non-empty line counts: something has to actually be waiting."""
    from subjective_devagents_fastapi.ssh_chat import secret_prompt

    assert secret_prompt(output) == "", f"{output!r} was mistaken for a live prompt"


# --- G16/G17/G18: delivering the secret -----------------------------------------------------------------


SECRET = "a-real-looking-remote-password-9f2b"


def test_g16_the_secret_reaches_the_session_that_asked_for_it(client, chats, monkeypatch):
    """`POST /api/llm/ssh-secret` had no test of any kind. This is the delivery half."""
    from subjective_devagents_fastapi import ssh_chat
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="ssh", name="g16", workspace="deploy@g16.example.invalid")
    delivered: list[tuple] = []

    def fake_send_secret(target, secret, timeout=None):
        delivered.append((target, secret))
        return {"success": True, "status": "completed", "response": "Deploying...\n", "awaitingSecret": ""}

    monkeypatch.setattr(ssh_chat.sessions, "send_secret", fake_send_secret)

    body = envelope(client.post("/api/llm/ssh-secret", json={
        "target": conversation["target"], "secret": SECRET,
    }), ok=True)

    assert delivered == [(conversation["target"], SECRET)], (
        "the secret did not reach the session, or reached the wrong one"
    )
    assert body["assistantMessage"]["text"] == "Deploying...\n"
    assert body["userMessage"]["secretInput"] is True
    assert body["userMessage"]["text"] == "••••••••", body["userMessage"]


def test_g17_the_secret_never_appears_in_the_response_transcript_or_any_file(
    client, chats, monkeypatch, capsys
):
    """The whole reason this is a separate route from send-message."""
    from subjective_devagents_fastapi import ssh_chat
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="ssh", name="g17", workspace="deploy@g17.example.invalid")
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: {
            "success": True, "status": "completed", "response": "Authenticated.\n",
        },
    )

    response = client.post("/api/llm/ssh-secret", json={
        "target": conversation["target"], "secret": SECRET,
    })
    has_no_secret(response.text, SECRET)

    # The transcript the UI will render next.
    hydrated = hydrate(client, conversation["target"])
    has_no_secret(hydrated, SECRET)
    user_messages = [m for m in hydrated["messages"] if m["role"] == "user"]
    assert user_messages and user_messages[-1]["text"] == "••••••••"

    # The listing, and every file under the app home.
    has_no_secret(client.get("/api/llm/conversations").text, SECRET)
    for path in sorted(Path(client.chat_home).rglob("*")):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8", errors="replace"), (
                f"{path} holds the secret"
            )

    # And the log line, which records the length and not the value.
    logged = capsys.readouterr().out
    assert SECRET not in logged, "the secret was written to a log line"
    assert "SSH secret sent" in logged
    assert f'"chars": {len(SECRET)}' in logged, "the log did not record the length instead"


def test_g17_the_secret_does_not_reach_the_broadcast_event(client, chats, monkeypatch):
    """Every connected tab receives this event; none of them may receive the secret."""
    from subjective_devagents_fastapi import ssh_chat
    from subjective_devagents_fastapi.app import state

    conversation = chats(provider="ssh", name="g17-ws", workspace="deploy@g17.example.invalid")
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: {
            "success": True, "status": "completed", "response": "Authenticated.\n",
        },
    )

    published: list[dict] = []
    monkeypatch.setattr(state.hub, "publish", lambda event: published.append(event))

    envelope(client.post("/api/llm/ssh-secret", json={
        "target": conversation["target"], "secret": SECRET,
    }), ok=True)

    assert published, "no event was broadcast at all"
    has_no_secret(published, SECRET)
    assert published[0]["type"] == "llm-message"


@pytest.mark.parametrize(
    "body,expected_status",
    [
        pytest.param({"target": "llm:ssh:local:nope", "secret": SECRET}, "not_found", id="unknown-target"),
        pytest.param({"secret": SECRET}, "not_found", id="no-target"),
    ],
)
def test_g18_a_secret_for_an_unknown_target_is_refused(client, body, expected_status):
    response = client.post("/api/llm/ssh-secret", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["status"] == expected_status, response.text


def test_g18_a_secret_for_a_chat_that_is_not_an_ssh_chat_is_refused(client, chats):
    """A CLI agent has no terminal waiting on a password; the secret must go nowhere."""
    conversation = chats(provider="claude", name="g18-not-ssh")
    response = client.post("/api/llm/ssh-secret", json={
        "target": conversation["target"], "secret": SECRET,
    })
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "read_only"
    assert "not an SSH chat" in body["error"]
    has_no_secret(response.text, SECRET)


def test_g18_an_empty_secret_is_refused_before_the_session(client, chats, monkeypatch):
    from subjective_devagents_fastapi import ssh_chat

    conversation = chats(provider="ssh", name="g18-empty", workspace="deploy@g18.example.invalid")
    reached: list = []
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: reached.append(secret) or {"success": True, "response": ""},
    )

    response = client.post("/api/llm/ssh-secret", json={"target": conversation["target"], "secret": ""})
    assert response.status_code == 400
    assert response.json()["status"] == "invalid_request"
    assert reached == [], "an empty secret was still written to the terminal"


def test_g18_a_secret_for_a_session_that_is_not_waiting_is_refused(client, chats, monkeypatch):
    """`send_secret` never opens a session: a secret is only ever an answer."""
    from subjective_devagents_fastapi import ssh_chat

    conversation = chats(provider="ssh", name="g18-no-session", workspace="deploy@g18.example.invalid")
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: {
            "success": False, "status": "no_session", "response": "",
            "error": "That chat has no live shell any more. Send the command again.",
        },
    )

    response = client.post("/api/llm/ssh-secret", json={
        "target": conversation["target"], "secret": SECRET,
    })
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "no_session"
    assert "no live shell" in body["error"]
    has_no_secret(response.text, SECRET)


def test_g18_a_failed_delivery_records_nothing_in_the_transcript(client, chats, monkeypatch):
    from subjective_devagents_fastapi import ssh_chat

    conversation = chats(provider="ssh", name="g18-failed", workspace="deploy@g18.example.invalid")
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: {
            "success": False, "status": "closed", "response": "",
            "error": "The remote shell has ended.",
        },
    )

    client.post("/api/llm/ssh-secret", json={"target": conversation["target"], "secret": SECRET})

    hydrated = hydrate(client, conversation["target"])
    assert hydrated["messages"] == [], "a failed secret left a masked message behind"


def test_g18_a_wrong_password_is_reported_as_still_awaiting_one(client, chats, monkeypatch):
    """A wrong password is usually followed by the same prompt again."""
    from subjective_devagents_fastapi import ssh_chat

    conversation = chats(provider="ssh", name="g18-retry", workspace="deploy@g18.example.invalid")
    monkeypatch.setattr(
        ssh_chat.sessions, "send_secret",
        lambda target, secret, timeout=None: {
            "success": True, "status": "completed",
            "response": "Sorry, try again.\n[sudo] password for deploy:",
            "awaitingSecret": "[sudo] password for deploy:",
        },
    )

    body = envelope(client.post("/api/llm/ssh-secret", json={
        "target": conversation["target"], "secret": SECRET,
    }), ok=True)
    assert body["assistantMessage"]["awaitingSecret"] == "[sudo] password for deploy:"
    has_no_secret(body, SECRET)


# --- helpers ----------------------------------------------------------------------------------------------


def _wait_for(condition, timeout: float = 5.0) -> None:
    """A deterministic wait on a condition, not a sleep on a guess.

    `send-message` hands the turn to a background task and returns `pending`, so the
    spy fills in a moment later.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition was not met within {timeout}s")
