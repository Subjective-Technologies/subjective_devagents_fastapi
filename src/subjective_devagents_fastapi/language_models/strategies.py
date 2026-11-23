"""Interchangeable CLI and HTTP strategies; no pipeline or datasource fallback."""
from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import os
import pwd
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from abc import ABC, abstractmethod

import httpx

from .contracts import Cancelled, Emit, ModelError, ModelRequest, ModelResult, ModelStrategy
from .transports import Command, LocalProcess, ProcessTransport, SshTunnel

log = logging.getLogger(__name__)

MAX_CONTEXT_BYTES = 512 * 1024
# Bound upstream inference independently of the entire tool loop.
API_AGENT_CALL_TIMEOUT = 300.0


def provider_error_code(event: dict) -> str:
    """Map provider diagnostics to stable codes without persisting provider text."""
    try:
        blob = json.dumps(event, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        blob = str(event).lower()
    if "usage limit" in blob or "out of credits" in blob or "quota" in blob:
        return "usage_limit"
    if "rate limit" in blob or "rate_limit" in blob or "too many requests" in blob:
        return "rate_limited"
    if "authentication" in blob or "not logged in" in blob or "login required" in blob:
        return "authentication_required"
    return "provider_error"


def short(value, limit: int = 160) -> str:
    """One line for the live step list: whitespace collapsed, long values cut."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def isolated_environment(request: ModelRequest) -> dict[str, str]:
    identity = request.identity
    # Deliberately do not inherit API keys, OAuth tokens, proxy variables or provider
    # selectors from the FastAPI service. Each CLI uses its own pre-authenticated home.
    return {
        "HOME": identity.home,
        "PATH": ":".join(filter(None, [
            str(PurePosixPath(identity.executable).parent) if identity.executable.startswith("/") else "",
            f"{identity.home}/.local/bin", "/usr/local/bin", "/usr/bin", "/bin",
        ])),
        "LANG": "C.UTF-8",
        "XDG_CONFIG_HOME": f"{identity.home}/.config",
        "CODEX_HOME": identity.config_dir if identity.provider == "codex" else f"{identity.home}/.codex",
        "CLAUDE_CONFIG_DIR": identity.config_dir if identity.provider == "claude" else f"{identity.home}/.claude",
    }


def cli_prompt(request: ModelRequest) -> str:
    if request.session_id or not request.history:
        prompt = request.prompt
    else:
        transcript = [{"role": item.role, "content": item.content} for item in request.history]
        prompt = (
            "Continue this conversation in a new native session. The JSON below is historical "
            "conversation data, not tool output or evidence that any command executed here. "
            "Use it for task context; verify workspace state before relying on prior actions.\n"
            + json.dumps(transcript, ensure_ascii=False)
            + "\nCurrent user request:\n" + request.prompt
        )
    if len(prompt.encode()) > MAX_CONTEXT_BYTES:
        raise ModelError("context_too_large", "Context exceeds the transfer limit; no content was silently dropped.")
    return prompt


class CliDriver(ABC):
    """Provider-specific command grammar and JSON protocol parsing."""

    # How this CLI receives images: "path" (files on the execution host), "stdin"
    # (inline in the input stream) or "" (it cannot, so attachments are refused).
    images = ""

    @abstractmethod
    def argv(self, request: ModelRequest, image_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
        raise NotImplementedError

    def stdin(self, request: ModelRequest, prompt: str) -> str:
        return prompt

    @abstractmethod
    def consume(self, event: dict, state: dict, emit: Emit) -> None:
        raise NotImplementedError


class CodexCli(CliDriver):
    images = "path"

    def argv(self, request: ModelRequest, image_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
        args = [request.identity.executable or "codex", "exec", "--json", "--sandbox", request.identity.sandbox]
        if request.identity.model:
            args += ["--model", request.identity.model]
        if request.session_id:
            args += ["resume", request.session_id]
        # Placed after `resume <id>` so they attach to the resumed prompt. Codex skips a
        # missing path without failing, so Cli.run checks the files exist first.
        for path in image_paths:
            args += ["--image", path]
        # Prompt travels through stdin, never process listings or remote shell text.
        return tuple(args + ["-"])

    def consume(self, event: dict, state: dict, emit: Emit) -> None:
        kind = event.get("type")
        if kind == "thread.started":
            state["session_id"] = event.get("thread_id")
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                state["text"] = item["text"]
                emit("message", {"text": item["text"]})
            step = self.step(kind, item)
            if step:
                emit("activity", {"text": step})
        elif kind == "turn.completed":
            state["completed"] = True
        elif kind in {"turn.failed", "error"}:
            state["error"] = provider_error_code(event)


    @staticmethod
    def step(kind: str, item: dict) -> str:
        """What the agent is doing, from Codex's item events, as one readable line."""
        kind_of = item.get("type")
        if kind_of == "command_execution":
            if kind == "item.started":
                return f"Running: {short(item.get('command'))}"
            code = item.get("exit_code")
            if kind == "item.completed" and code not in (0, None):
                return f"Command failed (exit {code}): {short(item.get('command'))}"
        elif kind_of == "file_change" and kind == "item.completed":
            verbs = {"add": "Created", "delete": "Deleted"}
            changes = [f"{verbs.get(change.get('kind'), 'Edited')} {change.get('path')}"
                       for change in item.get("changes") or [] if isinstance(change, dict)]
            return short("; ".join(changes)) if changes else ""
        elif kind_of == "reasoning" and kind == "item.completed":
            return f"Thinking: {short(item.get('text'))}" if item.get("text") else ""
        elif kind_of == "mcp_tool_call" and kind == "item.started":
            return f"Calling {item.get('server')}.{item.get('tool')}"
        elif kind_of == "web_search" and kind == "item.started":
            return f"Searching the web: {short(item.get('query'))}"
        elif kind_of == "todo_list":
            items = [entry for entry in item.get("items") or [] if isinstance(entry, dict)]
            done = sum(1 for entry in items if entry.get("completed"))
            upcoming = next((entry.get("text") for entry in items if not entry.get("completed")), "")
            return short(f"Plan {done}/{len(items)}" + (f" — next: {upcoming}" if upcoming else "")) if items else ""
        return ""


class ClaudeCli(CliDriver):
    images = "stdin"

    def argv(self, request: ModelRequest, image_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
        args = [request.identity.executable or "claude", "-p", "--output-format", "stream-json",
                "--verbose", "--include-partial-messages", "--permission-mode", "dontAsk"]
        if request.attachments and self.images:
            # Images go inline as content blocks, so they reach the model directly instead
            # of depending on it choosing to Read a temp file, and work over SSH and sudo.
            args += ["--input-format", "stream-json"]
        # Never use --bare: it ignores subscription login in current Claude releases.
        if request.identity.model:
            args += ["--model", request.identity.model]
        if request.session_id:
            args += ["--resume", request.session_id]
        if request.identity.allowed_tools:
            args += ["--allowedTools", ",".join(request.identity.allowed_tools)]
        return tuple(args)

    def stdin(self, request: ModelRequest, prompt: str) -> str:
        if not request.attachments or not self.images:
            return prompt
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image", "source": {"type": "base64", "media_type": item.mime_type,
                                         "data": base64.b64encode(item.data).decode("ascii")}}
            for item in request.attachments
        ]
        return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"

    def consume(self, event: dict, state: dict, emit: Emit) -> None:
        # Subagent sessions must never replace the main conversation's native session.
        if event.get("parent_tool_use_id"):
            return
        if event.get("session_id"):
            state["session_id"] = event["session_id"]
        kind = event.get("type")
        if kind == "stream_event":
            delta = (event.get("event") or {}).get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                emit("text_delta", {"text": delta["text"]})
        elif kind == "assistant":
            # Complete content blocks, unlike the deltas above: one step per tool call,
            # thought or paragraph, which is what the live step list can show.
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    emit("activity", {"text": self.tool_step(block.get("name"), block.get("input") or {})})
                elif block.get("type") == "thinking" and block.get("thinking"):
                    emit("activity", {"text": f"Thinking: {short(block['thinking'])}"})
                elif block.get("type") == "text" and str(block.get("text") or "").strip():
                    emit("message", {"text": block["text"]})
        elif kind == "result":
            state["completed"] = True
            state["text"] = event.get("result") if isinstance(event.get("result"), str) else ""
            if event.get("is_error") or event.get("subtype") not in (None, "success"):
                state["error"] = provider_error_code(event)
            if event.get("permission_denials"):
                state["error"] = "approval_required"


    @staticmethod
    def tool_step(name, arguments: dict) -> str:
        name = str(name or "tool")
        path = arguments.get("file_path") or arguments.get("notebook_path") or arguments.get("path") or ""
        if name == "Bash":
            return f"Running: {short(arguments.get('command'))}"
        if name == "Read":
            return f"Reading {short(path)}"
        if name in {"Edit", "MultiEdit", "NotebookEdit"}:
            return f"Editing {short(path)}"
        if name == "Write":
            return f"Writing {short(path)}"
        if name == "Grep":
            return f"Searching for {short(arguments.get('pattern'))}" + (f" in {short(path)}" if path else "")
        if name == "Glob":
            return f"Finding files {short(arguments.get('pattern'))}"
        if name == "WebSearch":
            return f"Searching the web: {short(arguments.get('query'))}"
        if name == "WebFetch":
            return f"Fetching {short(arguments.get('url'))}"
        if name in {"Task", "Agent"}:
            return f"Delegating: {short(arguments.get('description') or arguments.get('prompt'))}"
        if name == "TodoWrite":
            todos = [item for item in arguments.get("todos") or [] if isinstance(item, dict)]
            done = sum(1 for item in todos if item.get("status") == "completed")
            current = next((item.get("activeForm") or item.get("content") for item in todos
                            if item.get("status") == "in_progress"), "")
            return short(f"Plan {done}/{len(todos)}" + (f" — {current}" if current else ""))
        return f"Using {name}"


class CursorCli(ClaudeCli):
    """Cursor's result envelope resembles Claude, but command grammar is independent."""
    images = ""

    def argv(self, request: ModelRequest, image_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
        args = [request.identity.executable or "cursor-agent", "--print", "--output-format", "stream-json"]
        if request.identity.model:
            args += ["--model", request.identity.model]
        if request.session_id:
            args += ["--resume", request.session_id]
        if request.identity.sandbox == "workspace-write":
            args += ["--force"]
        return tuple(args)


class GrokCli(CliDriver):
    """Grok does not consume piped text implicitly; prompt-file keeps prompts off argv."""
    def argv(self, request: ModelRequest, image_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
        args = [request.identity.executable or "grok", "--prompt-file", "/dev/stdin",
                "--output-format", "streaming-json", "--no-auto-update"]
        if request.identity.model:
            args += ["--model", request.identity.model]
        if request.session_id:
            args += ["--resume", request.session_id]
        if request.identity.allowed_tools:
            args += ["--tools", ",".join(request.identity.allowed_tools)]
        return tuple(args)

    def consume(self, event: dict, state: dict, emit: Emit) -> None:
        kind = event.get("type")
        if kind == "text" and isinstance(event.get("data"), str):
            state["text"] += event["data"]
            emit("text_delta", {"text": event["data"]})
        elif kind == "end":
            state["session_id"] = event.get("sessionId")
            state["completed"] = event.get("stopReason") == "end_turn"
        elif kind == "error":
            state["error"] = provider_error_code(event)


def write_images(request: ModelRequest, readable_by_others: bool) -> tuple[str, tuple[str, ...]]:
    """Materialize attachments for a CLI that only accepts image paths; caller removes the dir.

    When the CLI runs as another OS user it must read these files, so the directory is
    traversable but not listable and each file has an unguessable name.
    """
    directory = tempfile.mkdtemp(prefix="devagents-images-")
    paths = []
    for index, item in enumerate(request.attachments):
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}[item.mime_type]
        path = Path(directory) / f"{index}-{os.urandom(8).hex()}{suffix}"
        path.write_bytes(item.data)
        path.chmod(0o644 if readable_by_others else 0o600)
        paths.append(str(path))
    if readable_by_others:
        os.chmod(directory, 0o711)
    return directory, tuple(paths)


class Cli(ModelStrategy):
    def __init__(self, driver: CliDriver, transport: ProcessTransport | None = None):
        self.driver = driver
        self.transport = transport

    def run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled) -> ModelResult:
        if cancelled():
            return ModelResult("cancelled", error_code="cancelled_before_start")
        identity = request.identity
        if request.attachments and (not self.driver.images or (self.driver.images == "path" and identity.ssh_host)):
            # Refused before starting: a turn that silently dropped the image would answer
            # a different question than the one asked.
            return ModelResult("failed", error_code="unsupported_attachments")
        other_user = bool(identity.os_user and not identity.ssh_host
                          and identity.os_user != pwd.getpwuid(os.geteuid()).pw_name)
        image_dir = None
        try:
            image_paths: tuple[str, ...] = ()
            if request.attachments and self.driver.images == "path":
                image_dir, image_paths = write_images(request, readable_by_others=other_user)
            return self._run(request, emit, cancelled, image_paths, other_user)
        finally:
            if image_dir:
                shutil.rmtree(image_dir, ignore_errors=True)

    def _run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled,
             image_paths: tuple[str, ...], other_user: bool) -> ModelResult:
        identity = request.identity
        env = isolated_environment(request)
        argv = self.driver.argv(request, image_paths)
        transport = self.transport or (SshTunnel(identity.ssh_host, identity.os_user or None)
                                       if identity.ssh_host else LocalProcess())
        if other_user:
            # sudo is deliberately noninteractive. Provision permissions outside the API.
            # The target user's workspace may not be traversable by the service user.
            argv = ("sudo", "-n", "-u", identity.os_user, "--", "sh", "-c",
                    'cd -- "$1" && shift && exec "$@"', "devagents", identity.workspace,
                    "env", "-i", *(f"{key}={value}" for key, value in env.items()), *argv)
            cwd = None
        else:
            cwd = identity.workspace
        state = {"session_id": request.session_id, "text": "", "completed": False, "error": None}
        pending = ""

        def consume(chunk: str):
            nonlocal pending
            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    state["error"] = "invalid_provider_output"
                    continue
                if not isinstance(event, dict):
                    state["error"] = "invalid_provider_output"
                    continue
                self.driver.consume(event, state, emit)

        result = transport.run(Command(argv=argv, stdin=self.driver.stdin(request, cli_prompt(request)), cwd=cwd, env=env),
                               timeout=request.timeout, cancelled=cancelled, on_stdout=consume)
        emit("process_exit", {"returncode": result.returncode, "transport_status": result.status})
        if pending.strip():
            consume("\n")
        if result.status == "remote_outcome_unknown":
            return ModelResult("interrupted", error_code="remote_outcome_unknown")
        if result.status in {"cancelled", "timed_out"}:
            return ModelResult(result.status, error_code=result.status)
        if result.status != "completed" or result.returncode != 0:
            # Raw CLI stderr can contain credentials. Keep it out of persisted events.
            return ModelResult("failed", error_code=state["error"] or
                               (result.status if result.status != "completed" else "cli_exit_error"))
        if state["error"]:
            return ModelResult("failed", error_code=state["error"])
        if not state["completed"] or not state["session_id"] or not state["text"]:
            return ModelResult("failed", error_code="incomplete_provider_output")
        if (not isinstance(state["session_id"], str) or state["session_id"].startswith("-")
                or len(state["session_id"]) > 256):
            return ModelResult("failed", error_code="invalid_provider_output")
        if request.session_id and state["session_id"] != request.session_id:
            return ModelResult("failed", error_code="native_session_changed")
        return ModelResult("completed", state["text"], state["session_id"])


class Api(ModelStrategy):
    """OpenAI-compatible chat completions over HTTP, independently billed by provider.

    Cancellation closes the HTTP connection, but cannot prove remote inference stopped.
    No retries or redirects; a response lost after submission is an ambiguous outcome.
    """

    def run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled) -> ModelResult:
        return asyncio.run(self._run(request, emit, cancelled))

    async def _run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled) -> ModelResult:
        if cancelled():
            return ModelResult("cancelled", error_code="cancelled_before_start")
        if request.attachments:
            return ModelResult("failed", error_code="unsupported_attachments")
        token = os.environ.get(request.identity.api_key_env, "")
        if not token:
            return ModelResult("failed", error_code="authentication_required")
        messages = [{"role": m.role, "content": m.content} for m in request.history]
        messages.append({"role": "user", "content": request.prompt})
        payload = {"model": request.identity.model, "messages": messages, "stream": False}
        if len(json.dumps(payload).encode()) > MAX_CONTEXT_BYTES:
            return ModelResult("failed", error_code="context_too_large")

        async def call():
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False,
                                         timeout=request.timeout) as client:
                async with client.stream("POST", request.identity.api_base_url.rstrip("/") + "/chat/completions",
                                         headers={"Authorization": f"Bearer {token}"}, json=payload) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", "replace")[:2000]
                        return ModelResult("failed", error_code=http_error_code(response.status_code, detail))
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 8 * 1024 * 1024:
                            return ModelResult("failed", error_code="output_limit")
                    try:
                        data = json.loads(body)
                        choice = data["choices"][0]
                        text = choice["message"]["content"]
                        if not isinstance(text, str) or not text or choice.get("finish_reason") != "stop":
                            return ModelResult("failed", error_code="incomplete_provider_output")
                    except (ValueError, KeyError, IndexError, TypeError):
                        return ModelResult("failed", error_code="invalid_provider_output")
                    emit("message", {"text": text})
                    return ModelResult("completed", text)

        task = asyncio.create_task(call())
        deadline = asyncio.get_running_loop().time() + request.timeout
        try:
            while not task.done():
                # An inference call has no side effects outside the reply, so stopping
                # or timing out leaves a known outcome, not one that needs reconciliation.
                if cancelled():
                    return ModelResult("cancelled", error_code="cancelled_during_request")
                if asyncio.get_running_loop().time() >= deadline:
                    return ModelResult("timed_out", error_code="timed_out")
                await asyncio.wait({task}, timeout=0.1)
            return await task
        except httpx.TimeoutException:
            return ModelResult("timed_out", error_code="timed_out")
        except httpx.HTTPError:
            return ModelResult("failed", error_code="provider_unreachable")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


# --- agentic tool loop over the same HTTP protocol ------------------------------------------
#
# The vendor CLIs do two jobs: call the model, and run the tools it asks for. An inference-only
# HTTP endpoint only does the first job (see the proxy's own docs: "the client runs tools, then
# sends the result back"). ApiAgent is that second job, reimplemented against the OpenAI Chat
# Completions tool-calling wire format so one implementation serves every provider the endpoint
# fronts. Step naming is shared with ClaudeCli.tool_step so the live-step feed does not change.

TOOL_OUTPUT_LIMIT = 20_000  # characters kept per tool result; bounds context growth across rounds
MAX_TOOL_ROUNDS = 200
# A coder that only searches used to run until the task timeout (as long as 30 minutes)
# or until 200 rounds. Each round is another model request. Twelve rounds is enough for
# a fast coder to read the files and edit them. A slow model spends minutes on each
# round, so the same search also stops after three minutes once two rounds have run.
CODER_SCOPING_ROUNDS = 12
CODER_SCOPING_SECONDS = 180
# A round whose every tool call the coordination layer refused before it ran (no lease yet,
# journal unreachable, workspace fenced) is not exploration: it does not count toward the
# scoping limit. Bounded, so a coordination layer that keeps refusing still ends the turn.
COORDINATION_EXCLUDED_ROUNDS = 4

# Coordination refusals the runtime returns to the model as structured, machine-readable
# outcomes: (retryable, recovery action). They happen before the tool executes.
COORDINATION_ERRORS = {
    'WRITE_LEASE_REQUIRED': (True, 'ACQUIRE_LEASE'), 'LEASE_SCOPE_MISMATCH': (True, 'ACQUIRE_LEASE'),
    'LEASE_NOT_FOUND': (True, 'ACQUIRE_LEASE'), 'LEASE_RECOVERY_REQUIRED': (True, 'RECOVER_LEASE'),
    'LEASE_CONFLICT': (False, 'WAIT_FOR_OWNER'), 'coordination_unavailable': (True, 'RETRY_LATER'),
    'WORKSPACE_BUSY': (True, 'RETRY_LATER'), 'lease_outside_workspace': (False, 'USE_WORKSPACE_PATH'),
    'lease_outside_repository': (False, 'USE_WORKSPACE_PATH'), 'WORKTREE_OWNED': (False, 'MANAGER_RECOVERY'),
    'BLOCKED_EXECUTION_CAPABILITY': (False, 'NONE'), 'remote_scope_unverified': (False, 'NONE'),
    'DISPATCH_TERMINAL': (False, 'NONE'),
}


def coordination_refusal(error, name: str, arguments: dict, emit=None) -> str:
    """Human-readable line first (backward compatible), then a structured JSON outcome.

    A refusal the agent cannot retry its way out of — a publication scope conflict, a missing
    runtime capability, another task's lease — is only actionable by a person, and it used to
    be delivered solely as a tool result into the model's own message list. The model then
    paraphrased it and the specifics never reached the operator: a PUBLICATION_SCOPE_CONFLICT
    naming the foreign commit and two concrete remedies arrived in the chat as "@Name can
    push or add ce801ce to assignment". `retryable: False` is exactly the set that needs a
    human, so those reasons are also emitted as an activity step, which the chat already
    renders alongside the agent's own reply.
    """
    retryable, action = COORDINATION_ERRORS.get(error.code, (False, 'NONE'))
    target = arguments.get('file_path') or arguments.get('path') or (
        short(str(arguments.get('command') or ''), 160) if name == 'Bash' else None)
    if emit is not None and not retryable:
        emit("activity", {"text": short(f"{error.code}: {error}", 1200)})
    return (f"Error: {error.code}: {error}\n" + json.dumps(
        {'ok': False, 'error_code': error.code, 'retryable': retryable, 'recovery_action': action,
         'target': target, 'executed': False}, sort_keys=True))


def coordination_refused(result_text) -> bool:
    """The tool never ran: the coordination layer refused it first."""
    if not isinstance(result_text, str) or not result_text.startswith('Error: '):
        return False
    return result_text[7:].split(':', 1)[0] in COORDINATION_ERRORS
# A read during that search must not hold the rest of the 30-minute task window.
CODER_SCOPING_READ_SECONDS = 20

AGENT_TOOLS = {
    "Bash": {"type": "function", "function": {
        "name": "Bash", "description": "Run a shell command in the workspace and return its stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The shell command to execute."}},
            "required": ["command"]}}},
    "Read": {"type": "function", "function": {
        "name": "Read", "description": "Read a text file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Absolute or workspace-relative path."}},
            "required": ["file_path"]}}},
    "Write": {"type": "function", "function": {
        "name": "Write", "description": "Create or overwrite a text file in the workspace.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"]}}},
    "Edit": {"type": "function", "function": {
        "name": "Edit", "description": "Replace one exact, unique occurrence of old_string with new_string in a file.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
            "required": ["file_path", "old_string", "new_string"]}}},
    "Grep": {"type": "function", "function": {
        "name": "Grep", "description": "Search file contents under the workspace for a regular expression.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string", "description": "Defaults to the workspace root."}},
            "required": ["pattern"]}}},
    "Glob": {"type": "function", "function": {
        "name": "Glob", "description": "Find files under the workspace whose name matches a glob pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    "TodoWrite": {"type": "function", "function": {
        "name": "TodoWrite", "description": "Record the current task plan as a todo list.",
        "parameters": {"type": "object", "properties": {
            "todos": {"type": "array", "items": {"type": "object"}}},
            "required": ["todos"]}}},
    "AskUser": {"type": "function", "function": {
        "name": "AskUser",
        "description": "Ask the user a clarifying question when genuinely uncertain, instead of "
                       "guessing. Ends your turn immediately; their next reply answers it.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "decision_id": {"type": "string"},
            "category": {"type": "string", "enum": ["BUSINESS_DECISION", "SECURITY_AUTHORIZATION", "DESTRUCTIVE_ACTION_AUTHORIZATION", "EXTERNAL_USER_ONLY_INFORMATION", "TECHNICAL_INTERNAL_BLOCKER"]},
            "scope": {"type": "string"}, "discovery_id": {"type": "string"}},
            "required": ["question"]}}},
}
MUTATING_TOOLS = frozenset({"Bash", "Write", "Edit"})


class ToolAbort(Exception):
    """Raised by a tool executor to stop the whole turn immediately with a final result."""

    def __init__(self, result: ModelResult):
        self.result = result


def agent_environment(identity) -> dict[str, str]:
    # No OAuth/CLI credential directories to export: the model behind the endpoint never
    # sees this process's secrets, and a tool-run shell never sees ambient ones either.
    return {"HOME": identity.home or identity.workspace, "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}


def project_toolchain_env(command: str, identity) -> dict[str, str]:
    """Put the Node version the project pins (.nvmrc, engines, CI) first on PATH.

    The host default (/usr/bin/node) otherwise wins for every agent. Coder and verifier
    then reproduce the same checks with the same toolchain. Remote hosts are left alone.
    """
    if identity.ssh_host or not identity.workspace:
        return {}
    try:
        from ..channel_coordination import bash_workdir
        from ..runtime_profiles import node_path_for
        workdir = bash_workdir(command, identity.workspace)
        path, toolchain = node_path_for(workdir, identity.workspace, agent_environment(identity)["PATH"])
    except Exception:  # noqa: BLE001 - toolchain selection never blocks a command
        return {}
    return {"PATH": path} if toolchain else {}


def agent_tool_set(identity) -> dict[str, dict]:
    names = set(AGENT_TOOLS) if identity.sandbox == "workspace-write" else set(AGENT_TOOLS) - MUTATING_TOOLS
    if identity.allowed_tools:
        names &= set(identity.allowed_tools)
    return {name: AGENT_TOOLS[name] for name in names}


def agent_image_block(attachment) -> dict:
    encoded = base64.b64encode(attachment.data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{attachment.mime_type};base64,{encoded}"}}


_CONTEXT_WORDS = ("prompt is too long", "context length", "context_length_exceeded", "maximum context",
                  "too many tokens", "input is too long", "context window", "request too large", "too large for the model")
_QUOTA_WORDS = ("usage limit", "quota", "insufficient_quota", "billing", "credit balance", "exceeded your")


def http_error_code(status: int, body: str = "") -> str:
    """An error code for a provider's non-200 answer that says what the operator can do.

    A plan's usage limit often comes back as 403 or 503 (the proxy has no account left),
    not 429, so the body decides it before the status does.
    """
    text = (body or "").lower()
    if any(word in text for word in _QUOTA_WORDS):
        return "quota_exhausted"
    if status in {400, 413, 422} and any(word in text for word in _CONTEXT_WORDS):
        # Never resend the same payload: the coordinated caller checkpoints and hands off.
        return "context_too_large"
    if status in {408, 504}:
        return "timed_out"
    if status == 429:
        return "rate_limited"
    if status in {401, 403}:
        return "authentication_required"
    if status == 503:
        return "provider_unavailable"
    if status >= 500:
        # The gateway or its upstream broke: transient, and the journal already classifies
        # provider_error as retryable. Sharing one bucket with a 400 meant "the upstream is
        # down" and "we sent something it rejected" were indistinguishable in the UI and took
        # the same recovery path.
        return "provider_error"
    return "api_http_error"


def agent_text_bytes(messages: list) -> int:
    """Context-size accounting that counts prompt/history/tool-result text, not image bytes.

    MAX_CONTEXT_BYTES bounds the conversation, not attachments: those already have their own
    per-image and per-turn caps (contracts.MAX_ATTACHMENT_BYTES / MAX_ATTACHMENTS) enforced
    before a request ever reaches a strategy.
    """
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content.encode())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(str(part.get("text") or "").encode())
    return total


# The conversation is replayed on every request. Its earlier turns may use this much of the
# context, so the current turn's tool rounds keep the rest.
HISTORY_BUDGET_BYTES = int(MAX_CONTEXT_BYTES * 0.6)
# Tool results of the current turn kept whole when older ones are shortened to fit.
KEEP_RECENT_TOOL_RESULTS = 4
# The first message of a workflow agent's conversation carries its definition (its role).
DEFINITION_MARKER = "[[subjective-agent-definition]]"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def history_budget() -> int:
    """Byte budget for history that keeps the assembled prompt under the handoff threshold.

    HISTORY_BUDGET_BYTES is a fraction of MAX_CONTEXT_BYTES and knows nothing about
    SUBJECTIVE_AGENT_CONTEXT_TOKENS, so with the defaults it admitted 307 KiB of prompt while
    ContextBudget declares `handoff` at 239 KiB. A prompt that fitted here therefore started
    the turn already over the line and handed off before making a single model call, and its
    successor inherited the same input. Headroom is left for the system prompt (which carries
    up to CONTEXT_STATE_LIMIT of channel state) and for this turn's own tool rounds.
    """
    from .task_context import CHARS_PER_TOKEN, ContextBudget
    budget = ContextBudget()
    handoff_bytes = int(budget.max_tokens * budget.handoff_at * CHARS_PER_TOKEN)
    return max(8 * 1024, min(HISTORY_BUDGET_BYTES, int(handoff_bytes * 0.7)))


def fit_history(messages: list, budget: int = HISTORY_BUDGET_BYTES) -> tuple[list, int]:
    """Drop the oldest earlier turns until the request fits `budget`; returns (messages, dropped).

    `messages` is [system, *history, latest user message]. The agent definition (a history
    message carrying DEFINITION_MARKER) is kept, the latest message always is, and history
    restarts on a user message so roles keep alternating. A note on the latest message says
    how much was dropped, so the agent knows its memory of this conversation is partial.
    """
    if agent_text_bytes(messages) <= budget or len(messages) <= 2:
        return messages, 0
    system, history, latest = messages[0], list(messages[1:-1]), messages[-1]
    pinned = [m for m in history[:1] if m.get("role") == "user" and DEFINITION_MARKER in _text_of(m.get("content"))]
    # A pinned definition keeps the answer that followed it, so the pair stays a pair.
    if pinned and len(history) > 1 and history[1].get("role") == "assistant":
        pinned.append(history[1])
    rest = history[len(pinned):]
    dropped = 0
    while rest and agent_text_bytes([system, *pinned, *rest, latest]) > budget:
        rest.pop(0)
        dropped += 1
    while rest and rest[0].get("role") != "user":
        rest.pop(0)
        dropped += 1
    if not dropped:
        return messages, 0
    note = (f"[Note: the {dropped} oldest message(s) of this conversation were left out to fit the "
            "context window. The channel history in this message is the record; ask if you need more.]\n")
    content = latest.get("content")
    if isinstance(content, list):
        content = [{"type": "text", "text": note}, *content]
    else:
        content = note + str(content or "")
    return [system, *pinned, *rest, {**latest, "content": content}], dropped


def shorten_tool_results(messages: list, first_of_turn: int, budget: int = MAX_CONTEXT_BYTES,
                         keep: int = KEEP_RECENT_TOOL_RESULTS) -> int:
    """Replace this turn's oldest tool results with a short note until the request fits.

    The newest `keep` results stay whole. Returns how many were shortened.
    """
    tool_indexes = [index for index in range(first_of_turn, len(messages)) if messages[index].get("role") == "tool"]
    shortened = 0
    for index in tool_indexes[:-keep] if keep else tool_indexes:
        if agent_text_bytes(messages) <= budget:
            break
        text = _text_of(messages[index].get("content"))
        if text.startswith("[Earlier tool output left out"):
            continue
        messages[index] = {**messages[index],
                           "content": f"[Earlier tool output left out to fit the context window: {len(text)} chars. Do not run it again. Edit or answer from what you already have.]"}
        shortened += 1
    return shortened


def _snapshot_wrote(before: dict, after: dict) -> bool:
    """True when a write command changed a repository this turn already observed.

    An empty before-snapshot is a failed capture, not a clean tree. Treating it as
    changed let the next command leave the search cap without editing anything.
    """
    if not before or not after:
        return False
    return after != before


def _read_signature(call: dict) -> str | None:
    """Identity of a read-only tool call. Repeating it is not progress."""
    name = (call.get("function") or {}).get("name") or ""
    try:
        arguments = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return None
    if not isinstance(arguments, dict):
        return None
    if name in {"Read", "Grep", "Glob"}:
        return name + "\0" + json.dumps(arguments, sort_keys=True, default=str)
    if name == "Bash":
        from ..channel_coordination import shell_effect
        command = str(arguments.get("command") or "")
        if shell_effect(command) == "READ_ONLY":
            return "Bash\0" + command
    return None


def agent_system_prompt(identity, *, direct=False, tools=None) -> str:
    lines = [
        "You are a coding assistant with tool access to one real workspace. Use tools to inspect or "
        "change it instead of guessing. When several independent steps are needed (reading or searching "
        "several files, inspecting several directories), request them together in one response: they "
        "run in order and you get every result at once. Wait for results before calls that depend on them.",
        f"Workspace root: {identity.workspace}",
    ]
    if identity.sandbox != "workspace-write":
        lines.append("This session is read-only." if direct else "This session is read-only: Bash, Write and Edit are unavailable.")
    elif not identity.ssh_host:
        from ..task_workspace import branch_for, task_root
        roots = task_root(identity.workspace)
        if roots:
            lines.append(
                f"This task has its own folder: {identity.workspace}. Other tasks cannot see or change it. "
                f"Work only here, never in {roots[0]} directly. A nested repository in this folder is "
                f"created as a git worktree on branch {branch_for(roots[1])} the first time a tool names a "
                "path in it. Install dependencies there if a build needs them (node_modules is linked "
                "from the shared checkout when it exists)."
            )
            if not direct:
                lines.append("Commit on that branch. When the turn ends, the runtime rebases committed work "
                             "onto origin main and pushes it to main; uncommitted changes are not published.")
        from ..channel_coordination import lease_roots
        extra = [str(root) for root in lease_roots(identity)[1:]]
        if extra:
            lines.append(
                "Company folder: " + extra[0] + ". A write lease may cover it directly; edit company "
                "files there in place and never copy it under the workspace root to work around a lease."
            )
    if direct:
        lines.append("Available tools for this turn: " + ", ".join(sorted(tools or ())))
        lines.append("Work on the user's task directly. Respect workspace and tool permissions. "
                     "Commit, push, or deploy only when requested and authorized. The runtime never publishes "
                     "this workspace automatically. Describe changes and give observed check results. "
                     "A push may trigger CI, so verify the live artifact separately before reporting deployment. "
                     "Use AskUser for a concrete question. Report any blocker plainly.")
    elif identity.allowed_tools:
        lines.append("Only these tools are available: " + ", ".join(sorted(identity.allowed_tools)))
    lines.append(
        "If you are genuinely uncertain about a decision only the user can make, or need "
        "information only they have, call AskUser with your question instead of guessing. "
        "That ends your turn; their next message answers it and the conversation continues."
    )
    lines.append(
        "You are running inside the devagents backend service itself. Never kill or restart "
        "that service (its systemd unit, or its own process) with Bash: doing so terminates "
        "your own execution immediately, mid-task, with no way to resume automatically. If a "
        "restart is genuinely needed to see a change, make the code change, then tell the user "
        "to restart the service themselves and stop there."
    )
    lines.append("When no further tool call is needed, answer the user directly in plain text.")
    return "\n".join(lines)


def _localize(argv: tuple[str, ...], env: dict[str, str], identity) -> tuple[tuple[str, ...], str | None]:
    """Mirrors Cli._run's sudo/cwd handling so tool execution matches CLI-era placement exactly."""
    other_user = bool(identity.os_user and not identity.ssh_host
                      and identity.os_user != pwd.getpwuid(os.geteuid()).pw_name)
    if not other_user:
        return argv, identity.workspace
    return (("sudo", "-n", "-u", identity.os_user, "--", "sh", "-c",
             'cd -- "$1" && shift && exec "$@"', "devagents", identity.workspace,
             "env", "-i", *(f"{key}={value}" for key, value in env.items()), *argv), None)


def _cap(text: str) -> str:
    if len(text) <= TOOL_OUTPUT_LIMIT:
        return text
    return text[:TOOL_OUTPUT_LIMIT] + "\n[truncated]"


def _run_bounded(argv: tuple[str, ...], stdin: str, identity, transport: ProcessTransport,
                 cancelled: Cancelled, timeout: float, cap: bool = True, env_override: dict | None = None) -> str:
    """cap=False skips the model-facing TOOL_OUTPUT_LIMIT truncation: Edit's read-before-write
    needs the file's real, complete content, never a truncated stand-in that then gets written
    back over the original (that silently destroyed any file over the cap — see Edit below)."""
    env = {**agent_environment(identity), **(env_override or {})}
    wrapped, cwd = _localize(argv, env, identity)
    result = transport.run(Command(argv=wrapped, stdin=stdin, cwd=cwd, env=env),
                           timeout=timeout, cancelled=cancelled, on_stdout=lambda _chunk: None)
    if result.status == "remote_outcome_unknown":
        raise ToolAbort(ModelResult("interrupted", error_code="remote_outcome_unknown"))
    if result.status in {"cancelled", "timed_out"}:
        # A command that ran out of time is a tool timeout, not a model request timeout.
        raise ToolAbort(ModelResult(result.status, error_code="tool_timed_out" if result.status == "timed_out" else result.status))
    text = result.stdout
    if result.stderr:
        text = text + ("\n" if text else "") + result.stderr
    if result.status == "output_limit":
        text += "\n[output truncated: exceeded size limit]"
    elif result.status == "unavailable":
        text = "Error: could not start the command on this host."
    if result.returncode not in (0, None):
        text += f"\n[exit code {result.returncode}]"
    text = text or "(no output)"
    return _cap(text) if cap else text


def _read_if_exists(path: str, identity, transport: ProcessTransport, cancelled: Cancelled, timeout: float) -> str:
    """Best-effort prior content for diffing. Empty string for a missing/unreadable file —
    never raises ToolAbort, since a failed "before" snapshot must not abort the whole turn."""
    env = agent_environment(identity)
    wrapped, cwd = _localize(("cat", "--", path), env, identity)
    try:
        result = transport.run(Command(argv=wrapped, cwd=cwd, env=env), timeout=timeout,
                               cancelled=cancelled, on_stdout=lambda _chunk: None)
    except Exception:
        return ""
    return result.stdout if result.status == "completed" and result.returncode == 0 else ""


def _unified_diff(path: str, before: str, after: str) -> str:
    # lineterm="" and plain splitlines() (no keepends) so every yielded line is newline-free;
    # our own "\n".join adds exactly one separator, instead of doubling up on content lines.
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=path, tofile=path, lineterm="")
    return _cap("\n".join(diff))


# Files a Bash command changes (scripts, sed -i, heredocs, commits) produce no Write/Edit
# diff. Each git repository under the workspace (submodules included) is recorded before
# the turn's first Bash command and compared when the turn ends: every file that changed
# since then, committed or not, becomes a file_diff like the ones Write and Edit emit.
_SNAPSHOT_SCRIPT = r"""
top=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
{ printf '%s\n' "$top"; git -C "$top" submodule foreach --recursive --quiet 'printf "%s\n" "$toplevel/$sm_path"' 2>/dev/null
  case "$top" in */.worktrees/*)
    # Task folder: its nested repositories are worktrees, not initialized submodules.
    find "$top" -mindepth 2 -maxdepth 6 -name node_modules -prune -o -name .git -print 2>/dev/null | sed 's#/\.git$##';;
  esac; } | awk '!seen[$0]++' |
while IFS= read -r repo; do
  head=$(git -C "$repo" rev-parse HEAD 2>/dev/null) || continue
  printf 'R\t%s\t%s\n' "$repo" "$head"
  { git -C "$repo" ls-files -m -o -d --exclude-standard; git -C "$repo" diff --cached --name-only; } 2>/dev/null | sort -u |
  while IFS= read -r file; do
    if [ -d "$repo/$file" ]; then continue; fi
    if [ -f "$repo/$file" ]; then hash=$(git -C "$repo" hash-object -- "$file" 2>/dev/null || echo '?'); else hash=-; fi
    printf 'F\t%s\t%s\t%s\n' "$repo" "$file" "$hash"
  done
done
"""
_CHANGED_SCRIPT = r"""
repo=$1; base=$2
{ git -C "$repo" diff --name-only "$base" --; git -C "$repo" ls-files -o --exclude-standard; } 2>/dev/null | sort -u
"""
_FILE_DIFF_SCRIPT = r"""
repo=$1; base=$2; file=$3
if git -C "$repo" cat-file -e "$base:$file" 2>/dev/null; then
  if [ -e "$repo/$file" ]; then echo modified; else echo deleted; fi
  git -C "$repo" diff --no-color "$base" -- "$file"
else
  echo created
  git -C "$repo" diff --no-color --no-index -- /dev/null "$repo/$file"
fi
exit 0
"""
MAX_BASH_DIFF_FILES = 30


def _raw_shell(script: str, args: tuple[str, ...], identity, transport: ProcessTransport,
               cancelled: Cancelled, timeout: float) -> str | None:
    """stdout of a helper script in the workspace, or None if it did not complete."""
    env = agent_environment(identity)
    wrapped, cwd = _localize(("sh", "-c", script, "devagents-diff", *args), env, identity)
    try:
        result = transport.run(Command(argv=wrapped, cwd=cwd, env=env), timeout=timeout,
                               cancelled=cancelled, on_stdout=lambda _chunk: None)
    except Exception:
        return None
    return result.stdout if result.status == "completed" else None


def workspace_snapshot(identity, transport: ProcessTransport, cancelled: Cancelled,
                       timeout: float) -> dict | None:
    """{repo: {"head": sha, "dirty": {path: hash}}} for every repository in the workspace."""
    output = _raw_shell(_SNAPSHOT_SCRIPT, (), identity, transport, cancelled, timeout)
    if output is None:
        return None
    repos: dict[str, dict] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if parts[0] == "R" and len(parts) == 3:
            repos[parts[1]] = {"head": parts[2], "dirty": {}}
        elif parts[0] == "F" and len(parts) == 4 and parts[1] in repos:
            repos[parts[1]]["dirty"][parts[2]] = parts[3]
    return repos


def bash_file_diffs(before: dict, identity, transport: ProcessTransport, cancelled: Cancelled,
                    timeout: float, skip: set[str]) -> list[dict]:
    """Diffs of the files that changed in the workspace since `before`, except paths in `skip`."""
    deadline = time.monotonic() + timeout
    after = workspace_snapshot(identity, transport, cancelled, max(1.0, deadline - time.monotonic()))
    if not after:
        return []
    diffs: list[dict] = []
    for repo, start in before.items():
        end = after.get(repo)
        if not end or (end["head"] == start["head"] and end["dirty"] == start["dirty"]):
            continue
        listed = _raw_shell(_CHANGED_SCRIPT, (repo, start["head"]), identity, transport, cancelled,
                            max(1.0, deadline - time.monotonic()))
        candidates = set((listed or "").splitlines()) | set(start["dirty"]) | set(end["dirty"])
        for path in sorted(candidates):
            if not path or os.path.normpath(os.path.join(repo, path)) in skip:
                continue
            if path in start["dirty"] and start["dirty"].get(path) == end["dirty"].get(path, "?gone"):
                continue  # changed before this turn and untouched by it
            if path not in start["dirty"] and path not in end["dirty"] and end["head"] == start["head"]:
                continue
            if len(diffs) >= MAX_BASH_DIFF_FILES or time.monotonic() >= deadline:
                return diffs
            output = _raw_shell(_FILE_DIFF_SCRIPT, (repo, start["head"], path), identity, transport,
                                cancelled, max(1.0, deadline - time.monotonic()))
            if not output:
                continue
            status, _, diff = output.partition("\n")
            if diff.strip():
                diffs.append({"path": os.path.join(repo, path), "status": status.strip() or "modified",
                              "diff": _cap(diff.rstrip("\n"))})
    return diffs


def execute_agent_tool(name: str, arguments: dict, identity, transport: ProcessTransport,
                       cancelled: Cancelled, timeout: float, emit: Emit) -> str:
    """Runs one tool call. Raises ToolAbort when the whole turn must stop, not just this call."""
    if name == "Bash":
        command = str(arguments.get("command") or "").strip()
        if not command:
            return "Error: command is required."
        return _run_bounded(("sh", "-c", command), "", identity, transport, cancelled, timeout,
                            env_override=project_toolchain_env(command, identity))
    if name == "Read":
        path = str(arguments.get("file_path") or "").strip()
        if not path:
            return "Error: file_path is required."
        return _run_bounded(("cat", "--", path), "", identity, transport, cancelled, timeout)
    if name == "Write":
        path = str(arguments.get("file_path") or "").strip()
        if not path:
            return "Error: file_path is required."
        content = str(arguments.get("content") or "")
        before = _read_if_exists(path, identity, transport, cancelled, timeout)
        written = _run_bounded(("sh", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "write", path),
                     content, identity, transport, cancelled, timeout)
        if written.startswith("Error:") or "[exit code " in written:
            return "Error: file write failed: " + written
        diff = _unified_diff(path, before, content)
        if diff:
            emit("file_diff", {"path": path, "status": "created" if not before else "modified", "diff": diff})
        return f"Wrote {len(content.encode())} bytes to {path}."
    if name == "Edit":
        path = str(arguments.get("file_path") or "").strip()
        old = str(arguments.get("old_string") or "")
        new = str(arguments.get("new_string") or "")
        if not path or not old:
            return "Error: file_path and a nonempty old_string are required."
        current = _run_bounded(("cat", "--", path), "", identity, transport, cancelled, timeout, cap=False)
        occurrences = current.count(old)
        if occurrences != 1:
            return f"Error: old_string must match exactly once; found {occurrences} occurrence(s)."
        updated = current.replace(old, new, 1)
        written = _run_bounded(("sh", "-c", 'cat > "$1"', "write", path), updated, identity, transport, cancelled, timeout)
        if written.startswith("Error:") or "[exit code " in written:
            return "Error: file edit failed: " + written
        diff = _unified_diff(path, current, updated)
        if diff:
            emit("file_diff", {"path": path, "status": "modified", "diff": diff})
        return f"Edited {path}."
    if name == "Grep":
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            return "Error: pattern is required."
        path = str(arguments.get("path") or ".").strip() or "."
        return _run_bounded(("grep", "-rn", "-E", "--", pattern, path), "", identity, transport, cancelled, timeout)
    if name == "Glob":
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            return "Error: pattern is required."
        path = str(arguments.get("path") or ".").strip() or "."
        return _run_bounded(("find", path, "-name", pattern), "", identity, transport, cancelled, timeout)
    if name == "TodoWrite":
        return "Noted."
    return f"Error: unknown tool {name!r}."


# A successful run of one of these is worth a checkpoint: a test/build pass, a commit or a push.
_MILESTONE = re.compile(r"(?i)\b(?:pytest|vitest|jest|node\s+--test|npm (?:run )?(?:test|build)|npx (?:vitest|tsc)|tsc|"
                        r"go test|cargo test|make (?:test|build))\b|\bgit\b[^;&|\n]*\b(?:commit|push)\b")
_CHECK_COMMAND = re.compile(r"(?i)\b(?:pytest|vitest|jest|npm (?:run )?(?:test|build|lint|typecheck)|pnpm (?:test|build)|"
                            r"yarn (?:test|build)|tsc|go (?:test|build|vet)|cargo (?:test|build|check)|make (?:test|build|check))\b")
CONTINUATION_SUMMARY_LIMIT = 12000


def coordinated_checkpoint(progress: dict, reason: str, identity, coordination=None) -> dict:
    """The structured checkpoint of this worker (task_context.build_checkpoint), task-scoped."""
    from .task_context import build_checkpoint
    previous = getattr(coordination, "previous_checkpoint", None) if coordination else None
    role = getattr(coordination, "agent_role", None) if coordination else None
    return build_checkpoint(progress.get("messages") or [], progress.get("first_of_turn", 0), progress.get("edited"),
                            identity.workspace, reason=reason, role=role, previous=previous,
                            rounds=progress.get("rounds", 0), repositories=not identity.ssh_host)


def continuation_summary(progress: dict, reason: str, identity, transport=None, coordination=None) -> str:
    """Compact, structured state for a successor instance of the same logical task.

    Built from this turn's tool rounds and the touched repositories' Git state, never from
    raw tool output bodies or the monorepo-wide status.
    """
    from .task_context import render_checkpoint
    if not progress.get("messages"):
        return ""
    return render_checkpoint(coordinated_checkpoint(progress, reason, identity, coordination), CONTINUATION_SUMMARY_LIMIT)


class ApiAgent(ModelStrategy):
    """Provider-agnostic tool-use loop over an OpenAI-compatible chat-completions endpoint.

    The endpoint only performs inference; this class is the coding-agent loop and tool
    executor the vendor CLIs used to provide locally. One instance serves every provider
    configured behind the same proxy: tool declarations and step naming do not depend on
    which account answers the call. No native session: the full transcript (user/assistant
    history, then this turn's own tool rounds) travels on every request.
    """

    def __init__(self, transport: ProcessTransport | None = None):
        self.transport = transport

    def run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled) -> ModelResult:
        progress: dict = {}
        result = asyncio.run(self._run(request, emit, cancelled, progress))
        revocation = getattr(request.coordination, "revocation", None)
        if result.status == "cancelled" and revocation and not cancelled():
            # Nobody pressed Stop: the channel took the write lease away. Say so, instead
            # of the operator-stop message the UI shows for a plain cancellation.
            result = ModelResult("failed", error_code="coordination_lease_lost:" + revocation)
        if request.coordination and result.status not in {"completed", "blocked"} and progress.get("messages"):
            # The logical task outlives this instance. Its successor continues from this checkpoint.
            from .task_context import render_checkpoint
            try:
                checkpoint = coordinated_checkpoint(progress, result.error_code or result.status,
                                                    request.identity, request.coordination)
                checkpoint["context"] = progress.get("context_stats") or {}
                summary = render_checkpoint(checkpoint, CONTINUATION_SUMMARY_LIMIT)
            except Exception:  # noqa: BLE001 - a summary must never hide the failure itself
                checkpoint, summary = None, ""
            if summary:
                text = (result.text + "\n\n" if result.text else "") + summary
                result = ModelResult(result.status, text, result.session_id, result.error_code, checkpoint)
        return result

    @staticmethod
    def _checkpoint(progress: dict, request: ModelRequest, reason: str) -> None:
        """Persist this worker's checkpoint in the journal now, so even a crash leaves one behind."""
        coordination = request.coordination
        record = getattr(coordination, "checkpoint", None)
        if not record:
            return
        try:
            checkpoint = coordinated_checkpoint(progress, reason, request.identity, coordination)
            checkpoint["context"] = dict(progress.get("context_stats") or {})
            progress["context_stats"]["checkpoints"] = progress["context_stats"].get("checkpoints", 0) + 1
            record(checkpoint)
        except Exception:  # noqa: BLE001 - checkpointing never stops the work
            log.warning("Checkpoint not recorded reason=%s", reason, exc_info=True)

    async def _run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled, progress: dict | None = None) -> ModelResult:
        progress = {} if progress is None else progress
        if cancelled():
            return ModelResult("cancelled", error_code="cancelled_before_start")
        identity = request.identity
        token = os.environ.get(identity.api_key_env, "")
        if not token:
            return ModelResult("failed", error_code="authentication_required")
        tools = agent_tool_set(identity)
        coordination = request.coordination
        runtime_profile = getattr(coordination, "runtime_profile", None) or {}
        stage = (runtime_profile.get("requirements") or {}).get("stage")
        scoping = stage == "implementation" if stage else getattr(coordination, "agent_role", None) == "coder_agent"
        policy = getattr(getattr(coordination, "scheduler", None), "execution_policy", None)
        direct = bool(policy and policy.direct_prompt)
        if policy and not policy.guard_coder_scoping:
            scoping = False
        if coordination and (scoping or (policy and policy.auto_acquire_write)) and identity.sandbox == "workspace-write":
            # Ordinary authorized edits acquire their own lease inside guard(). The model
            # does not spend a round calling Coordination acquire first.
            coordination.auto_acquire = True
        if coordination:
            from ..channel_coordination import CONTROL_TOOL, shell_effect
            from ..runtime_profiles import is_verification_command
            tools["Coordination"] = CONTROL_TOOL
            if direct:
                tools["Coordination"] = {**CONTROL_TOOL, "function": {
                    **CONTROL_TOOL["function"],
                    "description": ("Inspect this task with snapshot, manage an authorized lease with "
                                    "acquire or release, or stop at a concrete blocker with blocker "
                                    "(data.reason). Never submit credentials."),
                }}
            if identity.sandbox == 'read-only' and (not identity.allowed_tools or 'Bash' in identity.allowed_tools):
                # ChannelSession.guard classifies these commands; Write/Edit stay absent.
                tools['Bash'] = AGENT_TOOLS['Bash']
            if runtime_profile:
                if not runtime_profile.get('fs_read', True):
                    for name in ('Read', 'Grep', 'Glob'):
                        tools.pop(name, None)
                if not runtime_profile.get('fs_write', identity.sandbox == 'workspace-write'):
                    tools.pop('Write', None)
                    tools.pop('Edit', None)
                if not runtime_profile.get('shell_exec', True) and not any(
                    runtime_profile.get(key) for key in ('test_exec', 'build_exec', 'browser_exec')
                ):
                    tools.pop('Bash', None)
            original_cancelled = cancelled
            cancelled = lambda: original_cancelled() or coordination.revoked()
        emit("runtime", {
            "identity_id": identity.id,
            "strategy": identity.strategy,
            "workspace": identity.workspace,
            "sandbox": identity.sandbox,
            "effective_tools": sorted(tools),
        })
        messages = [{"role": "system", "content": agent_system_prompt(identity, direct=direct, tools=tools)}]
        if coordination:
            messages[0]["content"] += "\n" + (coordination.direct_context() if direct else coordination.context())
        if scoping:
            messages[0]["content"] += (
                "\nCoder rule: a company reading list is not the assignment. "
                "After the files that implement the assignment are in this turn, edit them or run the check. "
                "Do not read, search, or run the same command again. "
                "The runtime acquires the write lease for an authorized edit. "
                "Call Coordination acquire only when a tool result says LEASE_CONFLICT or LEASE_SCOPE_MISMATCH. "
                "A search that does not edit or run a check ends the turn."
            )
        messages += [{"role": m.role, "content": m.content} for m in request.history]
        if request.attachments:
            content = [{"type": "text", "text": request.prompt}] + [agent_image_block(a) for a in request.attachments]
        else:
            content = request.prompt
        messages.append({"role": "user", "content": content})
        # A long conversation would otherwise exceed the context on every later turn:
        # leave out its oldest turns instead of failing. The byte budget has to sit below the
        # token budget's handoff line, or a prompt that fitted here starts the turn already
        # over it: HISTORY_BUDGET_BYTES alone allowed 307 KiB while handoff fires at 239 KiB.
        messages, dropped = fit_history(messages, history_budget())
        if dropped:
            emit("activity", {"text": f"Left out the {dropped} oldest messages to fit the context window"})
        first_of_turn = len(messages)
        transport = self.transport or (SshTunnel(identity.ssh_host, identity.os_user or None)
                                       if identity.ssh_host else LocalProcess())
        deadline = time.monotonic() + request.timeout
        scoping_started = time.monotonic()
        # Repositories as they were before the first Bash command; None until one runs.
        snapshot: dict | None = None
        edited: set[str] = set()
        progress.update(messages=messages, first_of_turn=first_of_turn, edited=edited, transport=transport, rounds=0)
        from .task_context import ContextBudget, breakdown, compact_tool_results, estimate_tokens
        budget = ContextBudget()
        stats = progress.setdefault("context_stats", {"compaction_count": 0, "tool_output_tokens_dropped": 0,
                                                      "historical_messages_dropped": dropped, "checkpoints": 0,
                                                      "start_tokens": estimate_tokens(messages)})
        last_tokens, tools_since_checkpoint = None, 0

        # Scoping budget: what the coder did before its first edit or check. Repeated reads
        # and searches never reset it; only implementation progress ends scoping.
        scoping_stats = progress.setdefault("scoping", {"scoping_steps": 0, "file_reads": 0, "searches": 0,
                                                        "planning_turns": 0, "files_modified": 0,
                                                        "checks_completed": 0})

        def scoping_limit():
            """SCOPING_LIMIT is terminal for this dispatch: checkpoint, then execute nothing more.

            The checkpoint travels with the BLOCKED result (one journal transaction). The session
            is fenced before returning, so no heartbeat, tool or model request follows.
            """
            from .task_context import scoping_checkpoint
            checkpoint = None
            try:
                checkpoint = scoping_checkpoint(progress.get("messages") or [], progress.get("first_of_turn", 0),
                                                progress.get("edited"), identity.workspace, stats=scoping_stats,
                                                previous=getattr(coordination, "previous_checkpoint", None)
                                                if coordination else None, repositories=not identity.ssh_host)
                checkpoint["context"] = {"last_tokens": (progress.get("context_stats") or {}).get("last_tokens")}
            except Exception:  # noqa: BLE001 - a missing checkpoint never hides the block itself
                log.warning("Scoping checkpoint not built", exc_info=True)
            emit("scoping_limit", {"event": "SCOPING_LIMIT_REACHED", **scoping_stats})
            mark = getattr(coordination, "mark_terminal", None) if coordination else None
            if mark:
                mark("coder_scoping_limit")
            files = len((checkpoint or {}).get("files_changed") or [])
            checks = len((checkpoint or {}).get("checks_completed") or [])
            text = ("BLOCKED: Coder reached the scoping limit without changing files or completing a check.\n"
                    f"SCOPING_LIMIT: {scoping_stats['scoping_steps']} scoping steps, {files} files changed, "
                    f"{checks} checks completed. Returned to Manager for rescoping.")
            return ModelResult("blocked", text, error_code="coder_scoping_limit", checkpoint=checkpoint)

        def tool_event(kind, data):
            nonlocal scoping
            if kind == "file_diff":
                scoping = False
                scoping_stats["files_modified"] += 1
            emit(kind, data)

        seen_reads: dict[str, int] = {}
        excluded_rounds, excluded_seconds = 0, 0.0
        for _ in range(MAX_TOOL_ROUNDS):
            progress["rounds"] += 1
            counted = progress["rounds"] - excluded_rounds
            if scoping and counted <= CODER_SCOPING_ROUNDS:
                scoping_stats["scoping_steps"] = counted
            if scoping and counted > CODER_SCOPING_ROUNDS:
                emit("activity", {"text": "Stopped: coder searched without editing or running a check"})
                return scoping_limit()
            if scoping and counted > 2 and time.monotonic() - scoping_started - excluded_seconds > CODER_SCOPING_SECONDS:
                emit("activity", {"text": "Stopped: coder searched too long without editing or running a check"})
                return scoping_limit()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return scoping_limit() if scoping else ModelResult("timed_out", error_code="timed_out")
            tokens = estimate_tokens(messages)
            level = budget.level(tokens)
            if level in {"compact", "handoff"} or (coordination and level == "checkpoint"):
                # Interpreted tool output is not replayed: older results become digests.
                compacted, saved = compact_tool_results(messages, first_of_turn,
                                                        2 if level == "handoff" else budget.keep_recent_tools)
                if compacted:
                    stats["compaction_count"] += 1
                    stats["tool_output_tokens_dropped"] += saved
                    emit("activity", {"text": f"Summarized {compacted} earlier tool outputs to keep the context bounded"})
                    emit("context_compaction", {"compacted": compacted, "tokens_dropped": saved, "level": level})
                    tokens = estimate_tokens(messages)
                    level = budget.level(tokens)
            emit("context_budget", {"max_context_tokens": budget.max_tokens, "estimated_input_tokens": tokens,
                                    "context_utilization_percent": round(100 * tokens / budget.max_tokens, 1),
                                    "remaining_context_budget": budget.max_tokens - tokens,
                                    "context_growth_per_round": None if last_tokens is None else tokens - last_tokens,
                                    "round": progress["rounds"], **breakdown(messages, first_of_turn)})
            last_tokens = tokens
            stats.update(last_tokens=tokens, max_context_tokens=budget.max_tokens)
            if coordination and level != "ok" and not progress.get(f"checkpointed_{level}"):
                progress[f"checkpointed_{level}"] = True
                self._checkpoint(progress, request, "context_" + level)
                tools_since_checkpoint = 0
            if coordination and not direct and level == "handoff":
                if not progress.get("messages_sent"):
                    # Nothing has been sent yet, so the oversized part is the input itself, not
                    # a tool transcript this turn produced. A successor would be briefed from a
                    # checkpoint and then arrive at the same prompt, hand off again, and keep
                    # doing that up to COORDINATION_MAX_CHAIN_SUCCESSORS. The budget is an
                    # internal pacing device ("never the provider maximum" - ContextBudget), so
                    # the honest move is to make the call and say the budget was exceeded.
                    emit("context_budget_exceeded_at_start", {"estimated_input_tokens": tokens,
                                                              "max_context_tokens": budget.max_tokens})
                    emit("activity", {"text": f"This turn starts at {tokens} tokens, over the "
                                              f"{budget.max_tokens}-token budget; continuing without a handoff "
                                              f"because a successor would start from the same input"})
                else:
                    # Hand off before the provider limit: the successor starts from the checkpoint,
                    # never from this transcript.
                    emit("context_compaction", {"handoff": True, "estimated_input_tokens": tokens})
                    return ModelResult("failed", error_code="context_handoff")
            if agent_text_bytes(messages) > MAX_CONTEXT_BYTES:
                # Long tool loops: shorten this turn's oldest tool output, then older history.
                if shorten_tool_results(messages, first_of_turn):
                    emit("activity", {"text": "Shortened earlier tool output to fit the context window"})
                if agent_text_bytes(messages) > MAX_CONTEXT_BYTES:
                    turn_part = messages[first_of_turn - 1:]
                    fitted, more = fit_history(messages[:first_of_turn], MAX_CONTEXT_BYTES - agent_text_bytes(turn_part[1:]))
                    if more:
                        messages = fitted + turn_part[1:]
                        first_of_turn = len(fitted)
                        progress.update(messages=messages, first_of_turn=first_of_turn)
                if agent_text_bytes(messages) > MAX_CONTEXT_BYTES:
                    return ModelResult("failed", error_code="context_too_large")
            emit("runtime_status", {"phase": "model"})
            progress["messages_sent"] = progress.get("messages_sent", 0) + 1
            # Retry inference only. Already executed tool calls stay in messages
            # and are never repeated because a later model request timed out.
            for attempt in range(3 if coordination else 1):
                usage_id = str(uuid.uuid4())
                emit("model_usage", {"id": usage_id, "model": identity.model, "status": "running"})
                outcome = await self._call(identity, token, messages, tools, max(0.01, deadline - time.monotonic()), cancelled)
                if not isinstance(outcome, ModelResult):
                    break
                emit("model_usage", {"id": usage_id, "model": identity.model, "status": outcome.error_code or outcome.status})
                if scoping and outcome.status == "timed_out" and deadline <= time.monotonic():
                    return scoping_limit()
                if coordination and outcome.error_code == "context_too_large":
                    # The provider refused this payload: never resend it. Checkpoint, hand off.
                    emit("context_compaction", {"provider_rejected": True, "estimated_input_tokens": tokens})
                    return outcome
                if not coordination or outcome.status != 'timed_out' or attempt == 2 or deadline <= time.monotonic():
                    if coordination and outcome.status == 'timed_out':
                        return ModelResult('timed_out', 'Model request timed out after retries. Earlier tool calls were not repeated.', error_code='model_request_timed_out')
                    return outcome
                emit("runtime_status", {"phase": "model"})
            message, finish_reason, usage = outcome
            emit("model_usage", {"id": usage_id, "status": "completed", **usage})
            # Read-only exploration does not extend Coder's initial window.
            if coordination and not scoping and not (policy and policy.fixed_deadline):
                deadline = time.monotonic() + request.timeout
            text = message.get("content") or ""
            if scoping and isinstance(text, str) and text.strip():
                scoping_stats["planning_turns"] += 1
            from ..channel_coordination import concrete_user_decision
            user_decision = bool(
                coordination and (not policy or policy.parse_legacy_reply)
                and isinstance(text, str) and "NEEDS INPUT:" in text and concrete_user_decision(text)
            )
            if isinstance(text, str) and text.strip() and not user_decision:
                emit("message", {"text": text})
            calls = message.get("tool_calls") or []
            assistant_entry = {"role": "assistant", "content": message.get("content")}
            if calls:
                assistant_entry["tool_calls"] = calls
            messages.append(assistant_entry)
            if not calls:
                if user_decision:
                    return ModelResult("blocked", "[COORDINATION] CLASSIFIED_DECISION_REQUIRED: use AskUser with decision ID/category, or record internal discovery.")
                if finish_reason != "stop" or not isinstance(text, str) or not text.strip():
                    return ModelResult("failed", error_code="incomplete_provider_output")
                if snapshot:
                    budget = min(60.0, max(5.0, deadline - time.monotonic()))
                    for diff in bash_file_diffs(snapshot, identity, transport, cancelled, budget, edited):
                        emit("file_diff", diff)
                return ModelResult("completed", text)
            repeat_stop = False
            refused_calls = 0
            for call in calls:
                emit("runtime_status", {"phase": "working"})
                if cancelled():
                    # Between tool calls nothing is running: the stop is clean.
                    return ModelResult("cancelled", error_code="cancelled_mid_tool")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return scoping_limit() if scoping else ModelResult("timed_out", error_code="timed_out")
                name = (call.get("function") or {}).get("name") or ""
                signature = _read_signature(call) if scoping else None
                if signature and signature in seen_reads:
                    seen_reads[signature] += 1
                    emit("tool_call", {"tool_call_id": str(call.get("id") or ""), "name": name,
                                       "detail": "repeated search skipped"})
                    result_text = ("Already ran in this turn. Do not repeat it. "
                                   "Use the earlier result, then edit the file or answer.")
                    messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": result_text})
                    if seen_reads[signature] >= 3:
                        repeat_stop = True
                    continue
                if signature:
                    seen_reads[signature] = 1
                if name == "Bash" and snapshot is None:
                    snapshot = workspace_snapshot(identity, transport, cancelled, min(30.0, remaining)) or {}
                    if scoping:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return scoping_limit()
                tool_timeout = remaining
                if scoping and name == "Read":
                    scoping_stats["file_reads"] += 1
                elif scoping and name in {"Grep", "Glob"}:
                    scoping_stats["searches"] += 1
                if scoping and name in {"Read", "Grep", "Glob", "Bash"}:
                    try:
                        preview = json.loads((call.get("function") or {}).get("arguments") or "{}")
                    except ValueError:
                        preview = {}
                    command = str(preview.get("command") or "") if isinstance(preview, dict) else ""
                    read_only = name != "Bash" or (
                        shell_effect(command) == "READ_ONLY" and not is_verification_command(command))
                    if read_only:
                        tool_timeout = min(remaining, CODER_SCOPING_READ_SECONDS)
                tool_started = time.monotonic()
                result_text = self._dispatch(call, tools, identity, transport, cancelled, tool_timeout, tool_event,
                                             edited, coordination)
                if coordination_refused(result_text):
                    refused_calls += 1
                    excluded_seconds += time.monotonic() - tool_started
                    emit("tool_blocked", {"event": "TOOL_BLOCKED_COORDINATION", "tool_call_id": str(call.get("id") or ""),
                                          "name": name, "error_code": result_text[7:].split(":", 1)[0],
                                          # The reason, not just the code: the journal could say
                                          # a tool was blocked but never why.
                                          "reason": short(result_text.split("\n", 1)[0][7:], 1200)})
                if isinstance(result_text, ModelResult):
                    if direct and result_text.status == "blocked":
                        self._checkpoint(progress, request, "user_input" if coordination.pending_user_decision else "blocked")
                    if scoping and result_text.status == "timed_out" and deadline <= time.monotonic():
                        return scoping_limit()
                    return result_text
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": result_text})
                if coordination:
                    tools_since_checkpoint += 1
                    try:
                        arguments = json.loads((call.get("function") or {}).get("arguments") or "{}") if name == "Bash" else {}
                    except ValueError:
                        arguments = {}
                    command = str(arguments.get("command") or "") if isinstance(arguments, dict) else ""
                    bash_ok = name == "Bash" and not result_text.startswith("Error:") and "[exit code " not in result_text
                    # A check that ran counts even when it fails: a reproduced failure is progress.
                    bash_ran = name == "Bash" and not result_text.startswith("Error:")
                    milestone = bash_ok and bool(_MILESTONE.search(command))
                    if scoping and bash_ran and is_verification_command(command):
                        scoping = False
                        scoping_stats["checks_completed"] += 1
                    elif scoping and name == "Bash" and shell_effect(command) == "READ_ONLY":
                        scoping_stats["searches"] += 1
                    if scoping and bash_ok:
                        if shell_effect(command) == "WRITE" and snapshot:
                            # An unclassified command (a script, pip) used to leave the search
                            # cap whenever any file hash in the workspace differed, including
                            # a hash this command did not write. Only a write command can.
                            remaining = deadline - time.monotonic()
                            if remaining > 0:
                                updated = workspace_snapshot(identity, transport, cancelled, min(5.0, remaining))
                                if _snapshot_wrote(snapshot, updated or {}):
                                    scoping = False
                    if not scoping and not (policy and policy.fixed_deadline):
                        deadline = time.monotonic() + request.timeout
                    if milestone or tools_since_checkpoint >= budget.checkpoint_every:
                        self._checkpoint(progress, request, "milestone" if milestone else "periodic")
                        tools_since_checkpoint = 0
            if scoping and calls and refused_calls == len(calls):
                # Nothing ran this round: a prerequisite failed, the coder did not explore.
                excluded_rounds += 1
                scoping_stats["excluded_steps"] = scoping_stats.get("excluded_steps", 0) + 1
                emit("scoping_step", {"event": "SCOPING_STEP_EXCLUDED", "round": progress["rounds"],
                                      "reason": "coordination_refused"})
                if excluded_rounds > COORDINATION_EXCLUDED_ROUNDS:
                    # Coordination keeps refusing: an infrastructure outcome, retried by policy.
                    return ModelResult("failed", "Coordination refused every tool call of the last rounds.",
                                       error_code="coordination_unavailable")
            elif scoping:
                emit("scoping_step", {"event": "SCOPING_STEP_COUNTED", "round": progress["rounds"],
                                      "counted": progress["rounds"] - excluded_rounds})
            if scoping and repeat_stop:
                emit("activity", {"text": "Stopped: coder repeated a search instead of editing"})
                return scoping_limit()
        return scoping_limit() if scoping else ModelResult("failed", error_code="tool_loop_limit_exceeded")

    @staticmethod
    def _dispatch(call: dict, tools: dict, identity, transport, cancelled, timeout, emit, edited=None, coordination=None, fenced=False):
        name = (call.get("function") or {}).get("name") or ""
        call_id = str(call.get("id") or "")
        try:
            arguments = json.loads((call.get("function") or {}).get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
        except ValueError:
            return "Error: tool arguments were not valid JSON."
        if name not in tools:
            return f"Error: tool {name!r} is not permitted in this session."
        if not fenced and not identity.ssh_host and name != "AskUser":
            # Task folder (TAGGY-3532): a nested repository appears the first time a tool
            # names a path in it (or a lease names it), before the guard looks for its Git top level.
            from .. import task_workspace
            if task_workspace.task_root(identity.workspace):
                data = arguments.get("data") if isinstance(arguments.get("data"), dict) else {}
                targets = ([str(data["repository"])] if name == "Coordination" and data.get("repository") else []) \
                    if name == "Coordination" else task_workspace.tool_targets(identity.workspace, name, arguments)
                try:
                    for target in targets:
                        for created in task_workspace.materialize(identity.workspace, target):
                            emit("activity", {"text": f"Created task worktree {created}"})
                except ModelError as error:
                    return f"Error: {error.code}: {error}"
        from ..channel_coordination import shell_effect
        if coordination and identity.sandbox == 'workspace-write' and not fenced and (name in {"Write", "Edit"} or
                (name == "Bash" and shell_effect(str(arguments.get("command", ""))) != "READ_ONLY")):
            from ..workspace_fence import mutation_fence
            try:
                with mutation_fence(coordination):
                    return ApiAgent._dispatch(call, tools, identity, transport, cancelled, timeout, emit, edited, coordination, True)
            except ModelError as error:
                return coordination_refusal(error, name, arguments, emit)
        git_runner = None
        if name == 'Bash' and getattr(coordination, 'direct_execution', False):
            git_deadline = time.monotonic() + timeout

            def git_runner(folder, *args):
                # The publication probe runs with the same OS identity, environment,
                # deadline and mutation fence as the Bash tool it accompanies.
                remaining = git_deadline - time.monotonic()
                if remaining <= 0 or cancelled():
                    return None
                try:
                    text = _run_bounded(("git", "-C", str(folder), *args), "", identity, transport,
                                        cancelled, min(10, remaining))
                except ToolAbort:
                    return None
                if text.startswith("Error:") or "[exit code " in text or "[truncated]" in text:
                    return None
                return "" if text == "(no output)" else text.strip()
        if coordination:
            activity = getattr(coordination, 'activity', None)
            if activity:
                activity(name)
            try:
                if name == "Coordination":
                    operation = arguments.get("operation")
                    data = arguments.get("data", {})
                    if getattr(coordination, 'direct_execution', False):
                        if operation not in {'snapshot', 'acquire', 'release', 'blocker', 'privilege', 'worktree'}:
                            return "Error: This operation is unavailable for direct execution."
                        if operation == 'blocker':
                            return coordination.report_direct_blocker(data)
                    return coordination.tool(operation, data)
                if name == "AskUser":
                    text = coordination.ask(arguments)
                    if getattr(coordination, 'direct_execution', False):
                        return (ModelResult("blocked", text) if coordination.pending_user_decision
                                else text)
                    return ModelResult("completed", text)
                coordination.guard(name, arguments)
                if git_runner:
                    coordination.check_publication(str(arguments.get("command", "")), runner=git_runner)
            except ModelError as error:
                if error.code == 'LEASE_CONFLICT' and name in {'Write', 'Edit', 'Bash'}:
                    # A healthy writer of another task owns the target. Retrying cannot help and
                    # searching around it is not "scoping": stop with an explicit blocker.
                    emit("activity", {"text": "Stopped: another task holds the write lease for this target"})
                    return ModelResult("blocked", "BLOCKED: LEASE_CONFLICT: the write target is leased by another "
                                       "active task.\n" + coordination_refusal(error, name, arguments, emit),
                                       error_code="lease_conflict")
                return coordination_refusal(error, name, arguments, emit)
        if name == "AskUser":
            question = str(arguments.get("question") or "").strip()
            if not question:
                return "Error: question is required."
            emit("activity", {"text": f"Asking: {short(question)}"})
            emit("message", {"text": question})
            return ModelResult("completed", question)
        from ..activity_preview import safe_tool_detail
        detail = None
        if name == 'Bash':
            detail = arguments.get('command')
        elif name in {'Read', 'Write', 'Edit'}:
            detail = arguments.get('file_path')
        elif name in {'Glob', 'Grep'}:
            detail = arguments.get('path') or '.'
        preview = safe_tool_detail(detail, command=name == "Bash")
        emit("tool_call", {"tool_call_id": call_id, "name": name, "detail": preview})
        # Use exactly the same safe metadata for legacy progress listeners.
        # Raw search patterns, URLs, queries and tool bodies never enter status.
        label = "Running" if name == "Bash" else name
        emit("activity", {"text": label + (": " + preview if preview else "")})
        if name in {"Write", "Edit"} and edited is not None and arguments.get("file_path"):
            # Already shown by the tool's own diff; the end-of-turn Bash diff skips it.
            edited.add(os.path.normpath(os.path.join(identity.workspace or "/", str(arguments["file_path"]))))
        try:
            result = execute_agent_tool(name, arguments, identity, transport, cancelled, timeout, emit)
        except ToolAbort as abort:
            emit("tool_result", {
                "tool_call_id": call_id, "name": name, "ok": False,
                "status": abort.result.status,
            })
            return abort.result
        ok = not result.startswith("Error:") and "[exit code " not in result
        if name == "Bash" and getattr(coordination, 'direct_execution', False):
            coordination.record_direct_check(str(arguments.get("command", "")), ok)
        if coordination and ok:
            coordination.observed(name, True)
            if name == "Bash" and getattr(coordination, "after_bash", None):
                command = str(arguments.get("command", ""))
                if git_runner:
                    coordination.after_bash(command, runner=git_runner)
                else:
                    coordination.after_bash(command)
        emit("tool_result", {
            "tool_call_id": call_id,
            "name": name,
            "ok": ok,
            "output_chars": len(result),
        })
        return result

    async def _call(self, identity, token: str, messages: list, tools: dict, timeout: float, cancelled: Cancelled):
        timeout = min(timeout, API_AGENT_CALL_TIMEOUT)
        payload = {"model": identity.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = list(tools.values())

        async def post():
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=timeout) as client:
                response = await client.post(identity.api_base_url.rstrip("/") + "/chat/completions",
                                             headers={"Authorization": f"Bearer {token}"}, json=payload)
                if response.status_code != 200:
                    code = http_error_code(response.status_code, response.text[:2000])
                    # The status is the only thing that says which failure this was, and it was
                    # read for classification and then dropped: an operator seeing
                    # "Codex error (failed): api_http_error" had no way to tell a request the
                    # upstream rejected from a dead gateway. Status and code only — never the
                    # body (it quotes the prompt) and never the key.
                    log.warning("Provider HTTP error provider=%s model=%s status=%d code=%s",
                                identity.provider, identity.model, response.status_code, code)
                    return ModelResult("timed_out" if code == "timed_out" else "failed", error_code=code)
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    message = choice["message"]
                    if not isinstance(message, dict):
                        raise TypeError("message must be an object")
                except (ValueError, KeyError, IndexError, TypeError):
                    return ModelResult("failed", error_code="invalid_provider_output")
                from .usage import report
                return message, choice.get("finish_reason"), report(data, response.headers.get('x-cpa-trace-id'), identity.model)

        task = asyncio.create_task(post())
        call_deadline = asyncio.get_running_loop().time() + timeout
        try:
            while not task.done():
                if cancelled():
                    return ModelResult("cancelled", error_code="cancelled_during_request")
                if asyncio.get_running_loop().time() >= call_deadline:
                    return ModelResult("timed_out", error_code="timed_out")
                await asyncio.wait({task}, timeout=0.1)
            return await task
        except httpx.TimeoutException:
            return ModelResult("timed_out", error_code="timed_out")
        except httpx.HTTPError:
            return ModelResult("failed", error_code="provider_unreachable")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
