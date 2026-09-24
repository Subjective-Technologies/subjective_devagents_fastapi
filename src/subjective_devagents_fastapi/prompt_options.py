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


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug[:40] or fallback


def _option_item(line: str) -> tuple[str, str] | None:
    """A list item as (marker, text) — "- Brief" -> ("", "Brief"), "2) Ubuntu" -> ("2",
    "Ubuntu") — or None when the line is not a list item at all. The marker matters over
    SSH: a shell `select` menu is answered with the number it printed, not with the word."""
    match = re.match(r"^[ \t]*(?:([0-9]{1,2}|[A-Za-z])[.)]|[-*•‣▪])[ \t]+(\S.*)$", line)
    if not match:
        return None
    return (match.group(1) or "", match.group(2).strip())


def _question_text(line: str) -> str | None:
    """The question a line asks, with any list/step numbering stripped ("2. Which style?"),
    or None when the line does not end in a question mark."""
    head, mark, rest = line.strip().rpartition("?")
    if not mark:
        return None
    # "Which style? Optional: add context." — agents often put a short aside after the
    # question on the same line. Anything longer than an aside means the "?" was somewhere
    # in the middle of a paragraph, and the list below it is not its menu.
    if len(rest.strip()) > 80:
        return None
    # Only the closing sentence is the question: "I can do that in three ways. Which one?"
    sentence = re.split(r"(?<=[.!])\s+", f"{head}?")[-1]
    without_marker = re.sub(r"^(?:[-*•‣▪]|\*{0,2}\d{1,2}[.)]\*{0,2})[ \t]+", "", sentence)
    return re.sub(r"^\*+|\*+$", "", without_marker).strip() or None


def _menu_options(items: list[tuple[str, str]], answer_with_marker: bool = False) -> list[dict]:
    """Turn the collected list items into options. The item text is both the label and the
    reply: an agent that wrote "Detailed" as a choice understands "Detailed" as the answer,
    which no mapping table can improve on.

    `answer_with_marker` is the SSH case. There the other end is a program, not a model:
    `select` and "1) Debian 2) Ubuntu #?" read the number they printed and nothing else, so
    the reply is the marker while the label still shows the operator what it means.
    """
    options: list[dict] = []
    seen: set[str] = set()
    for index, (marker, item) in enumerate(items):
        # "Brief — short answers" / "Brief: short answers": the head is the choice, the
        # rest explains it, which is exactly the label/description split the UI draws.
        split = re.match(r"^(.{1,60}?)\s*(?:[—–:-]\s+|\s+[—–]\s*)(\S.*)$", item)
        label = (split.group(1) if split else item).strip().rstrip(".").strip()
        description = split.group(2).strip() if split else ""
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        if answer_with_marker and marker:
            reply = marker
        elif split:
            reply = label[:60]
        else:
            reply = item.strip()[:200]
        options.append({
            "id": _slug(label, f"opt-{index + 1}"),
            "label": label[:60],
            "description": description[:120],
            "reply": reply,
        })
    return options


def _detect_question_blocks(prose: str, answer_with_marker: bool = False) -> list[dict]:
    """Questions whose choices are spelled out as a list underneath them.

    This is what agents actually write when asked for a menu:

        Which response style?
        - Brief
        - Detailed
        - Step by step

    The lettered form ("A. ...", answered with "A") is handled by _detect_enumerated, which
    reads options written *before* a closing question. This one reads options written
    *after* the question they belong to, keeps their own words as the reply, and collects
    every such block so one message can ask several questions at once.

    The list has to sit directly under the question, or an unrelated list of findings that
    happens to follow a question would become a menu. Items must be short and none of them
    may be a question itself, for the same reason.
    """
    lines = prose.splitlines()
    blocks: list[dict] = []
    index = 0
    while index < len(lines):
        question = _question_text(lines[index])
        if not question:
            index += 1
            continue
        cursor = index + 1
        # One blank line between the question and its list is normal formatting; two mean
        # the list has stopped belonging to the question.
        if cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        items: list[tuple[str, str]] = []
        while cursor < len(lines):
            line = lines[cursor]
            if not line.strip():
                # A blank line inside the list is only a gap if more items follow it.
                following = cursor + 1
                if following < len(lines) and _option_item(lines[following]) and items:
                    cursor += 1
                    continue
                break
            item = _option_item(line)
            if item is None or len(item[1]) > 80 or item[1].endswith("?"):
                break
            items.append(item)
            cursor += 1
        options = _menu_options(items, answer_with_marker) if len(items) >= 2 else []
        if len(options) >= 2 and len(options) <= 8:
            blocks.append({"question": question, "options": options})
            index = cursor
            continue
        index += 1
    return blocks


def is_shell_provider(provider: Any) -> bool:
    """An SSH chat is a shell, not a model. Its replies must be what a program reads."""
    return str(provider or "").strip().lower() == "ssh"


def extract_prompt_options(text: Any, provider: Any = None) -> dict:
    """Return the structured choices an assistant reply is offering, or {asks_question: False}.

    The shape the UI consumes:
      {
        "asks_question": bool,
        "kind": "menu" | "enumerated" | "keep_revert" | "allow_deny" | "run" | "yes_no",
        "groups": [{"id", "question", "kind", "options", "allows_other"}],
        "options": [{"id", "label", "description", "reply"}],
        "allows_other": bool,
        "other_hint": str,
        "free_text": {"enabled", "label", "placeholder"},
      }

    `groups` is the real shape: a message may ask several questions, each with its own
    choices. `kind`/`options` mirror the first group so a client written against the
    single-question shape keeps working unchanged.
    """
    empty = {"asks_question": False, "groups": [], "options": [], "allows_other": False}
    prose = prose_only(text)
    if not prose:
        return empty
    shell = is_shell_provider(provider)

    # A message that lays out questions with their choices underneath is asking, whatever
    # its closing sentence looks like. The tail test below cannot see that: agents often
    # end such a message with a statement ("3 questions ready."), and demanding a trailing
    # "?" there is what left multi-question messages with no buttons at all.
    blocks = _detect_question_blocks(prose, answer_with_marker=shell)
    if blocks:
        return _result_groups([
            {"question": block["question"], "kind": "menu", "options": block["options"]}
            for block in blocks
        ])

    tail = prose[-900:]
    if not _ASKS_QUESTION_RE.search(tail):
        return empty

    if shell:
        # Everything below sends an English sentence, which a shell would only echo back as
        # a command-not-found. A prompt waiting on a shell is waiting for y or n.
        return _result("yes_no", [
            {"id": "yes", "label": "Yes", "description": "Answer the prompt with y.", "reply": "y"},
            {"id": "no", "label": "No", "description": "Answer the prompt with n.", "reply": "n"},
        ])

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


def annotate_message(message: dict | None, provider: Any = None) -> dict | None:
    """Return the message with `promptOptions` attached when an assistant reply asks the
    user something. A shallow copy, so this never mutates or persists the stored transcript
    — the options are a client-facing view, recomputed each time the message is served.

    `provider` decides what an answer has to look like: a model reads words, a shell over
    SSH reads the number or letter it printed."""
    if not message or message.get("role") != "assistant":
        return message
    options = extract_prompt_options(message.get("text"), provider)
    if not options.get("asks_question"):
        return message
    return {**message, "promptOptions": options}


OTHER_HINT = "Answer in your own words"
FREE_TEXT_LABEL = "Add anything else (optional)"


def _result(kind: str, options: list[dict]) -> dict:
    return _result_groups([{"question": "", "kind": kind, "options": options}])


def _result_groups(groups: list[dict]) -> dict:
    """One envelope for both shapes. The first group is repeated at the top level because
    that is where a client written before multi-question support looks for it."""
    numbered = [
        {
            "id": group.get("id") or f"q{index + 1}",
            "question": group.get("question") or "",
            "kind": group.get("kind") or "menu",
            "options": group["options"],
            # Always offer a way out of the detected set: the heuristic can only guess, and
            # the user may want to answer something none of these cover.
            "allows_other": True,
        }
        for index, group in enumerate(groups)
    ]
    return {
        "asks_question": True,
        "kind": numbered[0]["kind"],
        "groups": numbered,
        "options": numbered[0]["options"],
        "allows_other": True,
        "other_hint": OTHER_HINT,
        # The free-text box sits beside the buttons rather than replacing them: picking an
        # option and adding a sentence of context is one answer, not two turns.
        "free_text": {
            "enabled": True,
            "label": FREE_TEXT_LABEL,
            "placeholder": OTHER_HINT,
        },
    }
