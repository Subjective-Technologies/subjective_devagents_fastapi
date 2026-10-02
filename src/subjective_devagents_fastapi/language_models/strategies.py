"""Interchangeable CLI and HTTP strategies; no pipeline or datasource fallback."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import pwd
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from abc import ABC, abstractmethod

import httpx

from .contracts import Cancelled, Emit, ModelError, ModelRequest, ModelResult, ModelStrategy
from .transports import Command, LocalProcess, ProcessTransport, SshTunnel

MAX_CONTEXT_BYTES = 512 * 1024


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
                        code = ("authentication_required" if response.status_code in {401, 403}
                                else "rate_limited" if response.status_code == 429 else "api_http_error")
                        return ModelResult("failed", error_code=code)
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
                if cancelled() or asyncio.get_running_loop().time() >= deadline:
                    return ModelResult("interrupted", error_code="api_outcome_unknown")
                await asyncio.wait({task}, timeout=0.1)
            return await task
        except httpx.HTTPError:
            return ModelResult("interrupted", error_code="api_outcome_unknown")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
