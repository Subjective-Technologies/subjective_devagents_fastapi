"""Bounded, task-scoped model context for long coordinated work.

Production 2026-11-16: a three-repo task ran through 13 worker instances and every one
ended with ``context_too_large``. Each worker started at 43-54k tokens (a full channel
transcript, stage results, and an unbounded journal dump in the system prompt), re-read the
same files, replayed every raw tool output on every round (+1-3k tokens per round), and
handed its successor a summary whose commands read "[arguments hidden]" and whose
``workspace_state`` was the monorepo's 88-line ``git status``.

This module keeps a worker's context inside a budget and makes its checkpoint the
successor's primary context:

* ``estimate_tokens`` / ``ContextBudget``: thresholds (checkpoint 60%, compact 75%,
  hand off 85% of ``SUBJECTIVE_AGENT_CONTEXT_TOKENS``), configurable;
* ``compact_tool_results``: an interpreted tool output older than the last few rounds is
  replaced by a digest (command, exit, test counts, failure excerpt) instead of replayed;
* ``build_checkpoint``: actionable state only (phase, work completed, files, tests passed and
  failed, failure fingerprints, repositories, next action, task-scoped workspace state);
* ``task_repositories`` / ``repository_state``: only the repositories the task touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def _number(name, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


# Characters per token for code, logs and English mixed: deliberately conservative.
CHARS_PER_TOKEN = 3.2
CHECKPOINT_SCHEMA = 2


class ContextBudget:
    """Where a worker stands against its configured context budget (never the provider maximum)."""

    def __init__(self):
        self.max_tokens = _number('SUBJECTIVE_AGENT_CONTEXT_TOKENS', 90_000, int)
        self.checkpoint_at = _number('SUBJECTIVE_AGENT_CONTEXT_CHECKPOINT', 0.60)
        self.compact_at = _number('SUBJECTIVE_AGENT_CONTEXT_COMPACT', 0.75)
        self.handoff_at = _number('SUBJECTIVE_AGENT_CONTEXT_HANDOFF', 0.85)
        self.keep_recent_tools = _number('SUBJECTIVE_AGENT_KEEP_RECENT_TOOL_RESULTS', 6, int)
        self.checkpoint_every = _number('SUBJECTIVE_AGENT_CHECKPOINT_EVERY_TOOLS', 8, int)

    def level(self, tokens):
        share = tokens / max(1, self.max_tokens)
        if share >= self.handoff_at:
            return 'handoff'
        if share >= self.compact_at:
            return 'compact'
        if share >= self.checkpoint_at:
            return 'checkpoint'
        return 'ok'


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(str(part.get('text') or '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
    return ''


def estimate_tokens(messages):
    total = 0
    for message in messages:
        total += len(text_of(message.get('content')))
        for call in message.get('tool_calls') or []:
            total += len(str((call.get('function') or {}).get('arguments') or ''))
    return int(total / CHARS_PER_TOKEN) + 4 * len(messages)


def breakdown(messages, first_of_turn):
    """Token estimate per source, for the context-budget log line."""
    parts = {'system_prompt_tokens': 0, 'workflow_prompt_tokens': 0, 'chat_history_tokens': 0,
             'task_prompt_tokens': 0, 'tool_history_tokens': 0, 'assistant_tokens': 0}
    for index, message in enumerate(messages):
        tokens = estimate_tokens([message])
        role = message.get('role')
        if index == 0 and role == 'system':
            parts['system_prompt_tokens'] += tokens
        elif role == 'tool':
            parts['tool_history_tokens'] += tokens
        elif role == 'assistant':
            parts['assistant_tokens'] += tokens
        elif index < first_of_turn - 1:
            key = 'workflow_prompt_tokens' if '[[subjective-agent-definition]]' in text_of(message.get('content')) else 'chat_history_tokens'
            parts[key] += tokens
        else:
            parts['task_prompt_tokens'] += tokens
    return parts


# ---- tool output digests ------------------------------------------------------------------

_EXIT = re.compile(r'\[exit code (-?\d+)\]')
_COUNTS = [
    re.compile(r'Tests?\s+(\d+)\s+passed(?:\s*\|\s*(\d+)\s+failed)?(?:\s*\((\d+)\))?', re.I),        # vitest
    re.compile(r'(\d+)\s+passed(?:,\s*(\d+)\s+failed)?', re.I),                                        # pytest / jest-ish
    re.compile(r'#\s*pass\s+(\d+)[\s\S]{0,80}?#\s*fail\s+(\d+)', re.I),                                # node:test
    re.compile(r'Tests:\s+(?:(\d+)\s+failed,\s*)?(\d+)\s+passed', re.I),                               # jest
]
_ERROR_LINE = re.compile(r'(?i)(error|failed|failure|exception|traceback|assert|cannot|not found|denied|✗|×|FAIL\b)')
_CHECK = re.compile(r"(?i)\b(?:pytest|vitest|jest|node\s+--test|npm (?:run )?(?:test|build|lint|typecheck)|npx (?:vitest|tsc|jest|vite build)|"
                    r"pnpm (?:test|build)|yarn (?:test|build)|tsc\b|go (?:test|build|vet)|cargo (?:test|build|check)|make (?:test|build|check))")
_BUILD = re.compile(r'(?i)\b(?:build|tsc|typecheck|lint|vite build)\b')
_COMMIT = re.compile(r'\bgit\b[^;&|\n]*\b(?:commit|push)\b')
_SECRETISH = re.compile(r'(?i)(Bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|'
                        r'(?:token|secret|password|api[_-]?key)\s*[=:]\s*\S+|-----BEGIN [A-Z ]*PRIVATE KEY)')


def clean_command(command, limit=240):
    """A command as the next worker needs to see it: whole, unless it may carry a credential."""
    text = re.sub(r'\s+', ' ', str(command or '')).strip()
    if _SECRETISH.search(text):
        return '[command withheld: may contain a credential]'
    return text[:limit] + ('…' if len(text) > limit else '')


def test_counts(output):
    """(passed, failed) from common test-runner summaries, or None."""
    tail = output[-4000:]
    for pattern in _COUNTS:
        found = list(pattern.finditer(tail))
        if not found:
            continue
        groups = [g for g in found[-1].groups()]
        if pattern is _COUNTS[3]:
            return int(groups[1] or 0), int(groups[0] or 0)
        if pattern is _COUNTS[0] and groups[2]:
            return int(groups[0]), int(groups[1] or 0)
        return int(groups[0] or 0), int(groups[1] or 0)
    return None


def error_excerpt(output, limit=600):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    hits = [line for line in lines if _ERROR_LINE.search(line)]
    chosen = (hits[-6:] if hits else lines[-6:])
    return '\n'.join(line[:200] for line in chosen)[-limit:]


def fingerprint(command, exit_code, excerpt):
    """Same command failing the same way -> same fingerprint, whatever the timestamps/paths/numbers."""
    normal = re.sub(r'\b\d+(\.\d+)?(ms|s)?\b', 'N', excerpt or '')
    normal = re.sub(r'0x[0-9a-f]+|[0-9a-f]{12,}', 'H', normal)
    normal = re.sub(r'\s+', ' ', normal)[:400]
    return hashlib.sha256(f'{clean_command(command, 400)}|{exit_code}|{normal}'.encode()).hexdigest()[:16]


def digest_tool_output(name, arguments, output):
    """What an already-interpreted tool result is worth keeping in context."""
    output = output or ''
    if name == 'Bash':
        command = clean_command(arguments.get('command'))
        exit_match = _EXIT.search(output[-200:])
        exit_code = int(exit_match.group(1)) if exit_match else (1 if output.startswith('Error:') else 0)
        counts = test_counts(output)
        parts = [f'$ {command}', f'exit {exit_code}']
        if counts:
            parts.append(f'{counts[0]} passed, {counts[1]} failed')
        body = error_excerpt(output, 400) if exit_code else '\n'.join(
            line[:160] for line in [l for l in output.splitlines() if l.strip()][-3:])
        return (f'[Earlier tool output summarized; re-run if you need it in full: {" · ".join(parts)}'
                + (f'\n{body}' if body else '') + ']')
    if name == 'Read':
        return (f"[Earlier Read of {arguments.get('file_path')} summarized: {output.count(chr(10)) + 1} lines. "
                'Re-read the lines you need.]')
    if name in {'Grep', 'Glob'}:
        lines = [line for line in output.splitlines() if line.strip()]
        return f"[Earlier {name} {arguments.get('pattern') or arguments.get('path') or ''} summarized: {len(lines)} result lines.]"
    if name == 'Coordination':
        return '[Earlier Coordination result summarized; the current journal state is in the system prompt.]'
    return f'[Earlier {name} output summarized: {len(output)} chars.]'


def _calls_by_id(messages):
    calls = {}
    for message in messages:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            try:
                arguments = json.loads(function.get('arguments') or '{}')
            except ValueError:
                arguments = {}
            calls[call.get('id') or ''] = (function.get('name') or '', arguments if isinstance(arguments, dict) else {})
    return calls


def compact_tool_results(messages, first_of_turn, keep_recent=6):
    """Replace this turn's interpreted tool outputs (all but the newest `keep_recent`) with digests.

    Returns (results compacted, estimated tokens dropped). The tool-call/result pairing that the
    API requires is preserved; only the content of old results shrinks.
    """
    calls = _calls_by_id(messages)
    indexes = [i for i in range(first_of_turn, len(messages)) if messages[i].get('role') == 'tool']
    compacted, dropped = 0, 0
    for index in indexes[:-keep_recent] if keep_recent else indexes:
        text = text_of(messages[index].get('content'))
        if text.startswith('[Earlier ') or len(text) < 400:
            continue
        name, arguments = calls.get(messages[index].get('tool_call_id') or '', ('', {}))
        digest = digest_tool_output(name, arguments, text)
        messages[index] = {**messages[index], 'content': digest}
        compacted += 1
        dropped += int((len(text) - len(digest)) / CHARS_PER_TOKEN)
    return compacted, dropped


# ---- task-scoped workspace -----------------------------------------------------------------

def _repo_root(path, workspace):
    """Nearest enclosing Git repository (a submodule's own root, never its superproject)."""
    try:
        current = Path(path)
        if not current.is_absolute():
            current = Path(workspace or '/') / current
        current = current if current.is_dir() else current.parent
        stop = Path('/')
        while current != stop:
            if (current / '.git').exists():
                return str(current)
            current = current.parent
    except OSError:
        return None
    return None


_CD = re.compile(r'(?:^|&&|;)\s*cd\s+("[^"]+"|\'[^\']+\'|[^\s;&|]+)')
_GIT_C = re.compile(r'\bgit\s+-C\s+("[^"]+"|\'[^\']+\'|\S+)')


def command_directories(command, workspace):
    found = []
    for match in list(_CD.finditer(command or '')) + list(_GIT_C.finditer(command or '')):
        path = match.group(1).strip('"\'')
        if path.startswith('$') or path.startswith('~'):
            continue
        found.append(path if os.path.isabs(path) else os.path.join(workspace or '/', path))
    return found


def task_repositories(paths, workspace, limit=8):
    """Repositories the task touched, nearest first. The workspace root counts only when a
    touched path belongs to it directly (not via a submodule)."""
    roots = []
    for path in paths:
        root = _repo_root(path, workspace)
        if root and root not in roots:
            roots.append(root)
    return roots[:limit]


def _git(repo, *args, timeout=8):
    try:
        done = subprocess.run(['git', '-c', 'core.fsmonitor=false', '-C', repo, *args], capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def repository_state(repo, max_lines=20):
    status = _git(repo, 'status', '--short')
    lines = (status or '').splitlines()
    return {'branch': _git(repo, 'branch', '--show-current'), 'head': _git(repo, 'rev-parse', '--short', 'HEAD'),
            'origin_main': _git(repo, 'rev-parse', '--short', '--verify', '-q', 'origin/main'),
            'status': lines[:max_lines] + ([f'... {len(lines) - max_lines} more'] if len(lines) > max_lines else [])}


# ---- checkpoints ---------------------------------------------------------------------------

def _phase(edits, checks_after_edit, commits, role):
    if role and re.search(r'integration|pytest|selenium|verif|review|test', role):
        return 'VERIFYING'
    if commits.get('push'):
        return 'PUBLISHING'
    if checks_after_edit:
        return 'TESTING'
    if edits:
        return 'IMPLEMENTING'
    return 'SCOPING'


def build_checkpoint(messages, first_of_turn, edited, workspace, *, reason, role=None, goal=None,
                     previous=None, rounds=0, repositories=True):
    """Actionable state of this worker, for its successor. No prose history, no raw tool output."""
    calls = _calls_by_id(messages)
    files, notes, commands, passed, failed, failures = set(edited or ()), [], [], {}, {}, {}
    touched_dirs = []
    edits, checks_after_edit, commits = 0, False, {}
    for item in messages[first_of_turn:]:
        if item.get('role') == 'assistant':
            note = text_of(item.get('content')).strip()
            if note:
                notes.append(note)
            for call in item.get('tool_calls') or []:
                name, arguments = calls.get(call.get('id') or '', ('', {}))
                if name in {'Write', 'Edit'} and arguments.get('file_path'):
                    files.add(str(arguments['file_path']))
                    edits += 1
                    checks_after_edit = False
                if name in {'Read', 'Grep', 'Glob'} and (arguments.get('file_path') or arguments.get('path')):
                    touched_dirs.append(str(arguments.get('file_path') or arguments.get('path')))
        elif item.get('role') == 'tool':
            name, arguments = calls.get(item.get('tool_call_id') or '', ('', {}))
            if name != 'Bash':
                continue
            raw = str(arguments.get('command') or '')
            output = text_of(item.get('content'))
            command = clean_command(raw)
            touched_dirs.extend(command_directories(raw, workspace))
            summarized = output.startswith('[Earlier tool output summarized')
            exit_match = _EXIT.search(output[-200:])
            exit_code = int(exit_match.group(1)) if exit_match else (1 if output.startswith('Error:') else 0)
            if summarized:
                exit_code = int(re.search(r'exit (-?\d+)', output).group(1)) if re.search(r'exit (-?\d+)', output) else 0
            commands.append(f"{command} -> {'ok' if exit_code == 0 else f'exit {exit_code}'}")
            if _COMMIT.search(raw) and exit_code == 0:
                commits['push' if ' push' in raw else 'commit'] = True
            if _CHECK.search(raw):
                counts = test_counts(output)
                label = command + (f' ({counts[0]} passed, {counts[1]} failed)' if counts else '')
                if exit_code == 0 and (not counts or counts[1] == 0):
                    passed[command] = label
                    failed.pop(command, None)
                    if edits:
                        checks_after_edit = True
                else:
                    failed[command] = label
                    passed.pop(command, None)
            if exit_code != 0:
                excerpt = error_excerpt(output)
                key = fingerprint(raw, exit_code, excerpt)
                entry = failures.setdefault(key, {'fingerprint': key, 'command': command, 'exit_code': exit_code,
                                                  'excerpt': excerpt, 'count': 0})
                entry['count'] += 1
            elif _CHECK.search(raw):
                # A check that passes now resolves the earlier failures of the same command.
                for key in [k for k, v in failures.items() if v['command'] == command]:
                    failures.pop(key)
    # Carry the predecessor's established state forward (bounded), so nothing is re-discovered.
    previous = previous or {}
    for label in previous.get('tests_passed') or []:
        command = label.split(' (')[0]
        if command not in passed and command not in failed:
            passed[command] = label
    for entry in previous.get('failures') or []:
        if isinstance(entry, dict) and entry.get('fingerprint') and entry['fingerprint'] not in failures \
                and entry.get('command') not in passed:
            failures[entry['fingerprint']] = dict(entry)
        elif isinstance(entry, dict) and entry.get('fingerprint') in failures:
            failures[entry['fingerprint']]['count'] += int(entry.get('count') or 0)
    files |= {f for f in previous.get('files_changed') or [] if isinstance(f, str)}
    work = [line for line in (previous.get('work_completed') or []) if isinstance(line, str)]
    for note in notes[-4:]:
        line = note.splitlines()[0][:300] if note else ''
        if line and line not in work:
            work.append(line)
    work = work[-12:]
    repos = task_repositories(sorted(files) + touched_dirs[-40:] + list(previous.get('repositories') or []), workspace) \
        if repositories else []
    current_failure = None
    open_failures = sorted(failures.values(), key=lambda f: -f['count'])
    if failed:
        last = list(failed.values())[-1]
        current_failure = next((f for f in open_failures if f['command'] in last), {'command': last})
    phase = _phase(edits or len(files), checks_after_edit or bool(passed), commits, role)
    if previous.get('phase') in {'TESTING', 'VERIFYING', 'PUBLISHING'} and phase in {'SCOPING', 'IMPLEMENTING'} and not edits:
        phase = previous['phase']   # a successor that only looked around has not gone back to scoping
    last_note = notes[-1].strip() if notes else ''
    next_action = (f"Fix and re-run: {current_failure.get('command')}" if current_failure
                   else (last_note.splitlines()[-1][:300] if last_note else previous.get('next_action')
                         or 'Continue from the last completed step.'))
    checkpoint = {
        'schema': CHECKPOINT_SCHEMA, 'reason': reason, 'goal': (goal or previous.get('goal') or '')[:1500],
        'phase': phase, 'work_completed': work, 'files_changed': sorted(files)[:60],
        'tests_passed': list(passed.values())[-15:], 'tests_failed': list(failed.values())[-10:],
        'current_failure': current_failure, 'failures': open_failures[:8],
        'verified_facts': (previous.get('verified_facts') or [])[-20:],
        'decisions': (previous.get('decisions') or [])[-20:],
        'repositories': repos, 'recent_commands': commands[-10:], 'next_action': next_action,
        'tool_rounds': rounds + int(previous.get('tool_rounds') or 0),
    }
    if repositories:
        checkpoint['workspace_state'] = {repo: repository_state(repo) for repo in repos}
    return checkpoint


def progress_key(checkpoint):
    """Changes only when the task moved: files, passing checks, completed work, or open failures."""
    if not checkpoint:
        return None
    material = json.dumps([sorted(checkpoint.get('files_changed') or []), sorted(checkpoint.get('tests_passed') or []),
                           checkpoint.get('work_completed') or [],
                           sorted(f.get('fingerprint', '') for f in checkpoint.get('failures') or [])], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def render_checkpoint(checkpoint, limit=12000):
    text = ('CONTINUATION_SUMMARY (automatic recovery of the same logical task; the workspace was preserved, '
            'continue from it and do not restart from scratch):\n' + json.dumps(checkpoint, indent=1))
    if len(text) <= limit:
        return text
    trimmed = dict(checkpoint, recent_commands=checkpoint.get('recent_commands', [])[-4:],
                   workspace_state={k: {**v, 'status': v.get('status', [])[:8]} for k, v in (checkpoint.get('workspace_state') or {}).items()})
    return ('CONTINUATION_SUMMARY (automatic recovery of the same logical task; the workspace was preserved, '
            'continue from it and do not restart from scratch):\n' + json.dumps(trimmed, indent=1))[:limit]
