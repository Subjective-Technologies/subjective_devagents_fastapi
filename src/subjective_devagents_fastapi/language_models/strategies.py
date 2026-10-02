"""Interchangeable CLI and HTTP strategies; no pipeline or datasource fallback."""
from __future__ import annotations

import asyncio
import json
import os
import pwd
from pathlib import PurePosixPath
from abc import ABC, abstractmethod

import httpx

from .contracts import Cancelled, Emit, ModelError, ModelRequest, ModelResult, ModelStrategy
from .transports import Command, LocalProcess, ProcessTransport, SshTunnel

MAX_CONTEXT_BYTES = 512 * 1024


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

    @abstractmethod
    def argv(self, request: ModelRequest) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def consume(self, event: dict, state: dict, emit: Emit) -> None:
        raise NotImplementedError


class CodexCli(CliDriver):
    def argv(self, request: ModelRequest) -> tuple[str, ...]:
        args = [request.identity.executable or "codex", "exec", "--json", "--sandbox", request.identity.sandbox]
        if request.identity.model:
            args += ["--model", request.identity.model]
        if request.session_id:
            args += ["resume", request.session_id]
        # Prompt travels through stdin, never process listings or remote shell text.
        return tuple(args + ["-"])

    def consume(self, event: dict, state: dict, emit: Emit) -> None:
        kind = event.get("type")
        if kind == "thread.started":
            state["session_id"] = event.get("thread_id")
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                state["text"] = item["text"]
                emit("message", {"text": item["text"]})
        elif kind == "turn.completed":
            state["completed"] = True
        elif kind in {"turn.failed", "error"}:
            state["error"] = "provider_error"


class ClaudeCli(CliDriver):
    def argv(self, request: ModelRequest) -> tuple[str, ...]:
        args = [request.identity.executable or "claude", "-p", "--output-format", "stream-json",
                "--verbose", "--include-partial-messages", "--permission-mode", "dontAsk"]
        # Never use --bare: it ignores subscription login in current Claude releases.
        if request.identity.model:
            args += ["--model", request.identity.model]
        if request.session_id:
            args += ["--resume", request.session_id]
        if request.identity.allowed_tools:
            args += ["--allowedTools", ",".join(request.identity.allowed_tools)]
        return tuple(args)

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
        elif kind == "result":
            state["completed"] = True
            state["text"] = event.get("result") if isinstance(event.get("result"), str) else ""
            if event.get("is_error") or event.get("subtype") not in (None, "success"):
                state["error"] = "provider_error"
            if event.get("permission_denials"):
                state["error"] = "approval_required"


class CursorCli(ClaudeCli):
    """Cursor's result envelope resembles Claude, but command grammar is independent."""
    def argv(self, request: ModelRequest) -> tuple[str, ...]:
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
    def argv(self, request: ModelRequest) -> tuple[str, ...]:
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
            state["error"] = "provider_error"


class Cli(ModelStrategy):
    def __init__(self, driver: CliDriver, transport: ProcessTransport | None = None):
        self.driver = driver
        self.transport = transport

    def run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled) -> ModelResult:
        if cancelled():
            return ModelResult("cancelled", error_code="cancelled_before_start")
        identity = request.identity
        env = isolated_environment(request)
        argv = self.driver.argv(request)
        transport = self.transport or (SshTunnel(identity.ssh_host, identity.os_user or None)
                                       if identity.ssh_host else LocalProcess())
        if identity.os_user and not identity.ssh_host and identity.os_user != pwd.getpwuid(os.geteuid()).pw_name:
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

        result = transport.run(Command(argv=argv, stdin=cli_prompt(request), cwd=cwd, env=env),
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
