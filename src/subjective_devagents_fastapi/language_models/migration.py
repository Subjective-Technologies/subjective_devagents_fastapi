"""Strict, read-only conversion of a legacy conversation export into import input.

Usage: python -m subjective_devagents_fastapi.language_models.migration export.json identity-id
The resulting JSON goes to stdout. No credentials or native session are transferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import ModelError


def import_payload(export: dict, identity_id: str) -> dict:
    if not isinstance(export, dict):
        raise ModelError("invalid_history", "Export must be a conversation object.")
    conversation = export.get("conversation", export)
    if not isinstance(conversation, dict) or not isinstance(conversation.get("messages"), list):
        raise ModelError("invalid_history", "Export must contain a conversation with messages.")
    messages = conversation["messages"]
    if conversation.get("messagesTruncated") or conversation.get("truncated"):
        raise ModelError("invalid_history", "Export is truncated; obtain the full conversation first.")
    if len(messages) % 2:
        raise ModelError("invalid_history", "Export contains an incomplete turn; reconcile it before importing.")
    history = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ModelError("invalid_history", "Each exported message must be an object.")
        expected = "user" if index % 2 == 0 else "assistant"
        if (message.get("role") != expected or not isinstance(message.get("text"), str)
                or message.get("attachments") or message.get("files") or message.get("failed")
                or message.get("sendFailed")):
            raise ModelError("unsupported_history", "Only complete text-only user/assistant turns can be imported.")
        history.append({"role": expected, "content": message["text"]})
    return {"identity_id": identity_id, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("identity_id")
    args = parser.parse_args()
    try:
        body = import_payload(json.loads(args.export.read_text()), args.identity_id)
    except (ModelError, ValueError, OSError):
        parser.exit(2, "Cannot convert export: expected complete, untruncated, text-only conversation.\n")
    print(json.dumps(body, ensure_ascii=False))


if __name__ == "__main__":
    main()
