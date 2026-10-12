"""The prompt-building metadata a client sends with a turn is logged as short scalars only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subjective_devagents_fastapi.app import _context_meta  # noqa: E402


def test_only_short_scalars_are_kept():
    meta = _context_meta({
        "agent_id": "ai_manager",
        "history_messages_included": 31,
        "latest_message_truncated": False,
        "resumed_after_seq": None,
        "prompt": {"nested": "content"},
        "body": "x" * 500,
    })
    assert meta["agent_id"] == "ai_manager"
    assert meta["history_messages_included"] == 31
    assert meta["latest_message_truncated"] is False
    assert meta["resumed_after_seq"] is None
    assert "prompt" not in meta
    assert len(meta["body"]) == 120
    assert _context_meta("not a dict") == {}
