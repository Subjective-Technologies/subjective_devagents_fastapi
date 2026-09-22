"""Assertions this suite makes over and over.

The point of the suite is the envelope — the status code, `ok`, the error text, and
the fields that must *never* come back — so those checks live here rather than being
spelled out differently in each file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

UNAUTHORIZED = {"ok": False, "error": "Unauthorized."}


def envelope(response, *, ok: bool | None = None, status: int = 200) -> dict:
    """The decoded body, having checked the status code and `ok` flag."""
    assert response.status_code == status, (
        f"{response.request.method} {response.request.url.path} answered "
        f"{response.status_code}, expected {status}: {response.text[:400]}"
    )
    body = response.json()
    assert isinstance(body, dict), f"expected an object envelope, got {type(body).__name__}"
    if ok is not None:
        assert body.get("ok") is ok, f"expected ok={ok}, got {body!r}"
    return body


def refusal(response, *, status: int | Iterable[int] = 200) -> str:
    """A refusal: `ok` false and a non-empty error a person could act on."""
    allowed = {status} if isinstance(status, int) else set(status)
    assert response.status_code in allowed, (
        f"expected one of {sorted(allowed)}, got {response.status_code}: {response.text[:400]}"
    )
    body = response.json()
    assert body.get("ok") is False, f"expected a refusal, got {body!r}"
    error = str(body.get("error") or "")
    assert error.strip(), f"refusal carried no error text: {body!r}"
    return error


def has_no_secret(haystack: Any, *secrets: str) -> None:
    """Assert none of `secrets` appears anywhere in `haystack`.

    Serialised first, so a secret nested in a conversation's messages is caught as
    surely as one at the top level.
    """
    text = haystack if isinstance(haystack, str) else json.dumps(haystack, default=str)
    for secret in secrets:
        if not secret:
            continue
        assert secret not in text, (
            f"a secret of {len(secret)} characters appeared in {text[:300]!r}"
        )


def tree_has_no_secret(root: Path, *secrets: str) -> None:
    """The same check over every file under a directory — logs and stores included."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for secret in secrets:
            if secret and secret in text:
                raise AssertionError(f"{path} contains a secret of {len(secret)} characters")


def no_secret_fields(payload: dict) -> None:
    """A connection as the API may return it: flags about secrets, never the secrets."""
    for field in ("password", "key_passphrase", "private_key_content"):
        assert field not in payload, f"{field} was returned to the browser: {payload!r}"


def messages_of(conversation: dict) -> list[dict]:
    return list(conversation.get("messages") or [])


def last_assistant(conversation: dict) -> dict | None:
    for message in reversed(messages_of(conversation)):
        if message.get("role") == "assistant":
            return message
    return None
