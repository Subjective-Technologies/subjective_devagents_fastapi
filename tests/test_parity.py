"""Differential test: the Python port must agree with the Node backend it replaces.

parity_dump.js runs the original clients/desktop/subjective_devagents helpers over the
fixed inputs in parity_cases.json; this test runs the ported functions over the same
inputs and asserts identical output. Skipped when node or the client checkout is absent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SERVICE_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import llm_service, system_users  # noqa: E402

CASES = json.loads((TESTS_DIR / "parity_cases.json").read_text(encoding="utf-8"))
CLIENT_ROOT = (SERVICE_ROOT / ".." / ".." / "clients" / "desktop" / "subjective_devagents").resolve()


@pytest.fixture(scope="module")
def node_output() -> dict:
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    if not (CLIENT_ROOT / "src" / "llmService.js").exists():
        pytest.skip(f"DevAgents client checkout not found at {CLIENT_ROOT}")
    completed = subprocess.run(
        ["node", str(TESTS_DIR / "parity_dump.js")], capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        pytest.skip(f"parity_dump.js failed: {completed.stderr.strip()[:400]}")
    return json.loads(completed.stdout)


def test_provider_label(node_output):
    assert [llm_service.provider_label(value) for value in CASES["providers"]] == node_output["providerLabel"]


def test_provider_icon(node_output):
    assert [llm_service.provider_icon(value) for value in CASES["providers"]] == node_output["providerIcon"]


def test_normalize_role(node_output):
    assert [llm_service.normalize_role(value) for value in CASES["roles"]] == node_output["normalizeRole"]


def test_epoch_ms(node_output):
    ours = [llm_service.epoch_ms(value) for value in CASES["times"]]
    assert [float(value) for value in ours] == [float(value) for value in node_output["epochMs"]]


def test_export_filename_time(node_output):
    ours = [llm_service.export_filename_time(value) for value in CASES["exportPaths"]]
    assert [float(value) for value in ours] == [float(value) for value in node_output["exportFilenameTime"]]


def test_clean_visible_message(node_output):
    ours = [llm_service.clean_visible_message(value, "user") for value in CASES["messages"]]
    assert ours == node_output["cleanVisibleMessage"]


def test_is_generic_provider_label(node_output):
    ours = [llm_service.is_generic_provider_label(label, provider) for label, provider in CASES["labelPairs"]]
    assert ours == node_output["isGenericProviderLabel"]


def test_prefer_conversation_label(node_output):
    ours = [
        llm_service.prefer_conversation_label(existing, incoming, provider)
        for existing, incoming, provider in CASES["labelTriples"]
    ]
    assert ours == node_output["preferConversationLabel"]


def test_irc_channel_name(node_output):
    ours = [llm_service.irc_channel_name(label, identity, provider) for label, identity, provider in CASES["channelInputs"]]
    assert ours == node_output["ircChannelName"]


def test_composer_id_from_source(node_output):
    assert [llm_service.composer_id_from_source(value) for value in CASES["sourceFiles"]] == node_output["composerIdFromSource"]


def test_source_user_from_path(node_output):
    assert [llm_service.source_user_from_path(value) for value in CASES["sourceFiles"]] == node_output["sourceUserFromPath"]


def test_session_identity(node_output):
    assert [llm_service.session_identity(value) for value in CASES["identityItems"]] == node_output["sessionIdentity"]


def test_format_transcript_for_handoff(node_output):
    ours = [llm_service.format_transcript_for_handoff(value) for value in CASES["transcripts"]]
    assert ours == node_output["formatTranscriptForHandoff"]


def test_tail_conversation_for_client(node_output):
    ours = [llm_service.tail_conversation_for_client(value) for value in CASES["tailConversations"]]
    assert ours == node_output["tailConversationForClient"]


def test_tail_conversation_truncates_beyond_limit():
    messages = [{"role": "user", "text": str(index), "time": index} for index in range(200)]
    tailed = llm_service.tail_conversation_for_client({"target": "t", "messages": messages})
    assert len(tailed["messages"]) == llm_service.HYDRATE_MESSAGE_LIMIT
    assert tailed["truncated"] is True
    assert tailed["hiddenBefore"] == 200 - llm_service.HYDRATE_MESSAGE_LIMIT
    assert tailed["messages"][-1]["text"] == "199"


def test_is_valid_username(node_output):
    assert [system_users.is_valid_username(value) for value in CASES["usernames"]] == node_output["isValidUsername"]


def test_parse_passwd_text(node_output):
    ours = [
        {"username": entry.username, "uid": entry.uid, "gecos": entry.gecos, "home": entry.home, "shell": entry.shell}
        for entry in system_users.parse_passwd_text(CASES["passwdText"])
    ]
    assert ours == node_output["parsePasswdText"]


def test_filter_human_users(node_output):
    entries = system_users.parse_passwd_text(CASES["passwdText"])
    ours = [
        {"username": entry.username, "uid": entry.uid, "gecos": entry.gecos, "home": entry.home, "shell": entry.shell}
        for entry in system_users.filter_human_users(entries, exclude_username="bob")
    ]
    assert ours == node_output["filterHumanUsers"]
