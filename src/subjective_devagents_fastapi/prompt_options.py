"""Extract the choices an assistant is offering the user, as structured options.

When an agent's reply ends by asking the user something — "Shall I proceed?", "Keep or
revert?", "Which — A, B, or C?" — the UI should show real buttons with descriptions rather
than make the reader retype an answer. This module is the authoritative source of those
options: the backend computes them and hands them to the UI, so every client renders the
same choices instead of each re-guessing from the prose.

This is "between-turns" elicitation. The agent finishes its turn and asks; the user picks
an option (or writes their own via the always-present "Other"); the selected reply is sent
as the next turn's prompt. It does not pause a running turn — that would be a different,
interactive protocol.

Each option is {id, label, description, reply}:
  - `reply` is the exact text sent back to the agent when the option is chosen.
  - `description` is the human-readable gloss shown under the label.
Every set also carries `allows_other`: the UI shows a free-text box so the user can answer
something the detected options do not cover.

Ported from the client's detectPromptActions so behaviour is identical where it overlaps,
then enriched with descriptions the prose heuristic never carried.
"""
from __future__ import annotations

import re
from typing import Any

# Strip fenced and inline code so a code sample never trips the question/label detection.
_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

_ASKS_QUESTION_RE = re.compile(
    r"\?\s*$"
    r"|\b(?:choose|confirm|permission required|waiting for approval|would you like|"
    r"do you want|shall i|should i|may i)\b",
    re.IGNORECASE,
)
_KEEP_ACCEPT_RE = re.compile(
    r"\b(?:keep|accept)\b[^\n]{0,50}\b(?:code|changes?|implementation|edits?)\b"
    r"|\b(?:code|changes?|implementation|edits?)\b[^\n]{0,50}\b(?:keep|accept)\b",
    re.IGNORECASE,
)
_ALLOW_RE = re.compile(r"\b(?:allow|approve|approval|authorize|authorization|permission|permit)\b", re.IGNORECASE)
_RUN_RE = re.compile(r"\b(?:run|execute|launch|start)\b", re.IGNORECASE)
_WHICH_RE = re.compile(r"\b(?:which|choose|pick|select|option)\b", re.IGNORECASE)


def prose_only(value: Any) -> str:
    text = _FENCE_RE.sub("", str(value or ""))
    text = _INLINE_CODE_RE.sub("", text)
    return text.strip()


def _choice_from_label(prose: str, label: str, index: int) -> dict:
    """Build an option for a lettered/numbered choice, using the line that defined it
    ("A. Do the thing") as the description."""
    line = re.search(rf"^[ \t]*(?:[-*]\s*)?{re.escape(label)}[.)]\s+(.+)$", prose, re.IGNORECASE | re.MULTILINE)
    summary = re.sub(r"[:.,;]?\s*$", "", line.group(1)).strip()[:120] if line else ""
    return {
        "id": f"choice-{label.lower()}",
        "label": f"Option {label}",
        "description": summary,
        "reply": label,
    }


def _consecutive_lettered_options(prose: str) -> list[str]:
    found = [m.group(1) for m in re.finditer(r"^[ \t]*(?:[-*]\s*)?([A-Z])[.)]\s+\S", prose, re.MULTILINE)]
    if len(found) < 2 or found[0] != "A":
        return []
    labels: list[str] = []
    for index, letter in enumerate(found):
        if letter != chr(65 + index):
            break
        labels.append(letter)
    return labels if len(labels) >= 2 else []


def _detect_enumerated(prose: str, tail: str) -> list[dict]:
    normalized_tail = re.sub(r",?\s+or\s+", ", ", tail, flags=re.IGNORECASE)
    named = re.search(r"\b([A-Z](?:\s*[,/]\s*[A-Z]){1,7})\s*\?\s*$", normalized_tail, re.IGNORECASE) \
        or re.search(r"\b([1-9](?:\s*[,/]\s*[1-9]){1,8})\s*\?\s*$", normalized_tail)
    if named:
        seen: list[str] = []
        for label in named.group(1).upper().split(","):
            label = label.strip().strip("/").strip()
            if label and label not in seen:
                seen.append(label)
        if len(seen) >= 2:
            return [_choice_from_label(prose, label, index) for index, label in enumerate(seen)]
    lettered = _consecutive_lettered_options(prose)
    if len(lettered) >= 2 and _WHICH_RE.search(tail):
        return [_choice_from_label(prose, label, index) for index, label in enumerate(lettered)]
    return []


def extract_prompt_options(text: Any) -> dict:
    """Return the structured choices an assistant reply is offering, or {asks_question: False}.

    The shape the UI consumes:
      {
        "asks_question": bool,
        "kind": "enumerated" | "keep_revert" | "allow_deny" | "run" | "yes_no",
        "options": [{"id", "label", "description", "reply"}],
        "allows_other": bool,
        "other_hint": str,
      }
    """
    empty = {"asks_question": False, "options": [], "allows_other": False}
    prose = prose_only(text)
    if not prose:
        return empty
    tail = prose[-900:]
    if not _ASKS_QUESTION_RE.search(tail):
        return empty

    enumerated = _detect_enumerated(prose, tail)
    if enumerated:
        return _result("enumerated", enumerated)

    if _KEEP_ACCEPT_RE.search(tail):
        return _result("keep_revert", [
            {"id": "keep", "label": "Keep code", "description": "Keep the changes and continue.", "reply": "Keep the code and continue."},
            {"id": "revert", "label": "Revert", "description": "Undo those changes.", "reply": "Revert those changes."},
        ])
    if _ALLOW_RE.search(tail):
        return _result("allow_deny", [
            {"id": "allow", "label": "Allow", "description": "Grant the requested permission and continue.", "reply": "Allow it and continue."},
            {"id": "deny", "label": "Deny", "description": "Refuse the requested permission.", "reply": "Do not allow it."},
        ])
    if _RUN_RE.search(tail):
        return _result("run", [
            {"id": "run", "label": "Run", "description": "Run it and report the result.", "reply": "Run it and report the result."},
            {"id": "do-not-run", "label": "Don't run", "description": "Do not run it.", "reply": "Do not run it."},
        ])
    return _result("yes_no", [
        {"id": "yes", "label": "Yes", "description": "Proceed.", "reply": "Yes, proceed."},
        {"id": "no", "label": "No", "description": "Do not proceed.", "reply": "No, do not proceed."},
    ])


def annotate_message(message: dict | None) -> dict | None:
    """Return the message with `promptOptions` attached when an assistant reply asks the
    user something. A shallow copy, so this never mutates or persists the stored transcript
    — the options are a client-facing view, recomputed each time the message is served."""
    if not message or message.get("role") != "assistant":
        return message
    options = extract_prompt_options(message.get("text"))
    if not options.get("asks_question"):
        return message
    return {**message, "promptOptions": options}


def _result(kind: str, options: list[dict]) -> dict:
    return {
        "asks_question": True,
        "kind": kind,
        "options": options,
        # Always offer a way out of the detected set: the heuristic can only guess, and the
        # user may want to answer something none of these cover.
        "allows_other": True,
        "other_hint": "Answer in your own words",
    }
