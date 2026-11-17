"""LanguageModel facade: identity selection and durable, supervised turn lifecycle."""
from __future__ import annotations
from dataclasses import replace

import logging
import re
import math
import os
import threading
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor

from .contracts import MAX_ATTACHMENTS, Attachment, Identity, Message, ModelError, ModelRequest, ModelResult, ModelStrategy
from .registry import IdentityRegistry, image_support
from .store import TERMINAL, TurnStore
from .strategies import Api, ApiAgent, ClaudeCli, Cli, CodexCli, CursorCli, GrokCli, MAX_CONTEXT_BYTES

log = logging.getLogger(__name__)


def _workers() -> int:
    try:
        value = int(os.environ.get("SUBJECTIVE_LANGUAGE_MODEL_WORKERS", "16"))
    except ValueError:
        value = 16
    return min(max(value, 1), 128)



def work_trail(trail):
    """Changed files and steps for the chat message, within the chat service's size bounds."""
    diffs, total = [], 0
    for diff in trail['diffs']:
        if total + len(diff['diff']) > 300_000:
            break
        total += len(diff['diff'])
        diffs.append(diff)
    return {**({'diffs': diffs} if diffs else {}), **({'steps': trail['steps']} if trail['steps'] else {})}


class LanguageModel:

    def turn_evidence(self, identifier: str) -> dict:
        """Tool calls this turn actually executed, from its recorded events (not from model text)."""
        try:
            events = self.store.events(identifier)
        except Exception:  # noqa: BLE001 - no evidence is reported as none
            return {"turnId": identifier}
        runtime = next((item["data"] for item in events if item["kind"] == "runtime"), {})
        calls = [item["data"] for item in events if item["kind"] == "tool_call"]
        results = [item["data"] for item in events if item["kind"] == "tool_result"]
        return {"turnId": identifier, "strategy": runtime.get("strategy") or "",
                "toolCallsCount": len(calls),
                "successfulToolCalls": sum(1 for item in results if item.get("ok") is True),
                "toolsUsed": sorted({str(item.get("name") or "") for item in calls if item.get("name")})}

    def __init__(self, store: TurnStore, identities: IdentityRegistry,
                 strategies: dict[str, ModelStrategy] | None = None, max_workers: int | None = None):
        self.store = store
        self.identities = identities
        # Present while an operator drains this runtime for a restart (scripts/drain_restart.sh):
        # new coordinated work stays queued in the chat journal instead of being cut off later.
        # A freshly started service is never draining.
        self.drain_flag = Path(store.path).parent / "DRAINING"
        try:
            self.drain_flag.unlink(missing_ok=True)
        except OSError:
            pass
        self.strategies = strategies if strategies is not None else {
            "cli:codex": Cli(CodexCli()), "cli:claude": Cli(ClaudeCli()),
            "cli:cursor": Cli(CursorCli()), "cli:grok": Cli(GrokCli()), "api": Api(),
            # One ApiAgent instance serves every provider the proxy fronts: tool declarations
            # and step naming are provider-agnostic. Cursor has no equivalent backend there,
            # so it stays on cli: above.
            "api_agent:codex": ApiAgent(), "api_agent:claude": ApiAgent(), "api_agent:grok": ApiAgent(),
            # Kimi has no CLI at all: the proxy is its only way in.
            "api_agent:kimi": ApiAgent(), "api_agent:antigravity": ApiAgent(),
        }
        # Turns are I/O bound (model HTTP, tool subprocesses). Four shared workers let
        # four long turns starve every later dispatch until its start watchdog fired.
        self.max_workers = max_workers or _workers()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="language-model")
        self._lock = threading.RLock()
        self._futures: dict[str, Future] = {}
        # Image bytes never enter SQLite; an accepted turn's images live here until it starts.
        self._attachments: dict[str, tuple[Attachment, ...]] = {}
        self._coordination: dict[str, object] = {}
        self._stopping = threading.Event()

    def create(self, identity_id: str, history: tuple[Message, ...] = ()) -> dict:
        if sum(len(item.content.encode()) for item in history) > MAX_CONTEXT_BYTES:
            raise ModelError("context_too_large", "Imported conversation exceeds the transfer limit.")
        return self.store.create(self.identities.get(identity_id), history)

    def handoff(self, conversation_id: str, identity_id: str) -> dict:
        return self.store.handoff(conversation_id, self.identities.get(identity_id))

    def submit(self, conversation_id: str, prompt: str, request_id: str, timeout: float = 300,
               attachments: tuple[Attachment, ...] = (), coordination=None) -> dict:
        if not prompt.strip() or len(prompt.encode()) > MAX_CONTEXT_BYTES:
            raise ModelError("invalid_request", "Prompt must be nonempty and at most 512 KiB.")
        if not request_id or len(request_id) > 128 or not math.isfinite(timeout) or not 1 <= timeout <= 1800:
            raise ModelError("invalid_request", "Request id and a timeout between 1 and 1800 seconds are required.")
        if len(attachments) > MAX_ATTACHMENTS:
            raise ModelError("invalid_request", f"At most {MAX_ATTACHMENTS} images can be attached to one turn.")
        with self._lock:
            if self._stopping.is_set():
                raise ModelError("unavailable", "Language model service is shutting down.")
            conversation = self.store.conversation(conversation_id)
            identity = self.identities.get(conversation["identity"]["id"])
            if attachments and not image_support(identity):
                raise ModelError("unsupported_attachments", "This identity cannot receive images.")
            turn, created = self.store.reserve(conversation_id, identity, request_id, prompt, timeout)
            if created:
                if coordination is not None:
                    self._coordination[turn["id"]] = coordination
                if attachments:
                    self._attachments[turn["id"]] = tuple(attachments)
                    self.store.event(turn["id"], "attachments", {"images": [item.metadata() for item in attachments]})
                try:
                    future = self._executor.submit(self._execute, turn["id"])
                    self._futures[turn["id"]] = future
                    future.add_done_callback(lambda done, key=turn["id"]: self._forget(key))
                except RuntimeError:
                    self._attachments.pop(turn["id"], None)
                    self._coordination.pop(turn["id"], None)
                    self.store.finish(turn["id"], ModelResult("failed", error_code="dispatch_failed"))
                    raise ModelError("unavailable", "Could not dispatch the accepted turn.")
            return turn

    def capacity(self) -> dict:
        """Running/queued turns versus workers. Admission refuses rather than hiding a queue."""
        with self._lock:
            busy = sum(1 for future in self._futures.values() if not future.done())
        draining = self.drain_flag.exists()
        return {"busy": busy, "workers": self.max_workers, "draining": draining,
                "available": 0 if draining else max(0, self.max_workers - busy)}

    def _forget(self, identifier: str):
        with self._lock:
            self._futures.pop(identifier, None)

    def _execute(self, identifier: str):
        with self._lock:
            attachments = self._attachments.pop(identifier, ())
            coordination = self._coordination.pop(identifier, None)
        if not self.store.claim(identifier):
            return
        turn = self.store.turn(identifier)
        log.info("LanguageModel turn started turn_id=%s identity=%s", identifier, turn["identity"]["id"])
        # What the turn did, posted with a coordinated answer: changed files and steps.
        trail = {'diffs': [], 'steps': []}

        def remember(kind, data):
            if not isinstance(data, dict):
                return
            if kind == 'file_diff' and isinstance(data.get('path'), str) and isinstance(data.get('diff'), str):
                trail['diffs'] = (trail['diffs'] + [{'path': data['path'], 'status': data.get('status') or 'modified',
                                                     'diff': data['diff'][:100_000]}])[-50:]
            elif kind == 'activity' and isinstance(data.get('text'), str):
                step = data['text'].strip()[:500]
                if step and not re.search(r' is still working… \d+s$', step) and (not trail['steps'] or trail['steps'][-1] != step):
                    trail['steps'] = (trail['steps'] + [step])[-200:]
        try:
            if coordination:
                coordination.completion_store = self.store
                coordination.worker_started()
            identity = Identity(**turn["identity"])
            history = tuple(Message(**item) for item in self.store.conversation(turn["conversation_id"])["messages"])
            request = ModelRequest(identifier, identity, turn["prompt"], history, turn["session_id"], turn["timeout"],
                                   attachments, coordination)
            strategy = self.strategies.get(f"{identity.strategy}:{identity.provider}") or self.strategies.get(identity.strategy)
            if strategy is None:
                raise ModelError("unsupported_provider", "No execution strategy is registered.")
            from .usage import ActivityScope
            activity_scope = ActivityScope()
            def emit(kind, data):
                remember(kind, data)
                if coordination:
                    if kind == 'model_usage':
                        activity_scope.observe(data)
                        coordination.activity(usage=data)
                    elif kind == 'tool_call':
                        coordination.activity(data.get('name'), phase='tool', detail=data.get('detail'), call_id=activity_scope.tool_id(data.get('tool_call_id')))
                    elif kind == 'tool_result':
                        coordination.activity(phase='working')
                    elif kind == 'runtime_status':
                        coordination.activity(phase=data.get('phase'))
                    elif kind in {'message', 'activity'}:
                        coordination.activity()
                self.store.event(identifier, kind, data)
            result = strategy.run(request, emit,
                                  lambda: self._stopping.is_set() or bool(self.store.turn(identifier)["cancel_requested"]))
        except ModelError as error:
            result = ModelResult("failed", error_code=error.code)
        except Exception as error:
            # The strategy may have started work. Preserve uncertainty; do not retry.
            log.error("LanguageModel worker exception turn_id=%s exception_type=%s", identifier, type(error).__name__)
            result = ModelResult("interrupted", error_code="worker_exception")
        if self._stopping.is_set() and result.status == "cancelled":
            # Shutdown, not an operator stop: the logical task continues in a successor.
            result = ModelResult("cancelled", result.text, result.session_id, "service_restarted")
        if result.status not in TERMINAL:
            # An unlisted status used to raise out of finish() and leave the row running.
            # The sender then polled until the process was restarted.
            log.error("LanguageModel non-terminal status turn_id=%s status=%s", identifier, result.status)
            result = ModelResult("interrupted", result.text, result.session_id, result.error_code or result.status)
        if coordination and result.status == "completed":
            try:
                result = replace(result, text=coordination.canonical_reply(result.text))
                # Backend-owned [RT:] line from this turn's recorded tool events. The Chat API
                # accepts functional acceptance (ACCEPTANCE_VERIFIED) only with executed checks,
                # and a model cannot forge this line: lookalikes are stripped first.
                if coordination.delegation:   # server runtime; the browser path attests in llm_service
                    from ..agent_workflow import ensure_runtime_attestation
                    result = replace(result, text=ensure_runtime_attestation(result.text, self.turn_evidence(identifier)))
            except ModelError as error:
                result = ModelResult("interrupted", error_code=error.code)
        try:
            answer = {'text': result.text, 'error_code': result.error_code, **work_trail(trail),
                      **({'checkpoint': result.checkpoint} if result.checkpoint else {})}
            completion = (coordination.completion_payload(result.status, result.error_code, answer)
                          if coordination and coordination.delegation else None)
            self.store.finish(identifier, result, completion=completion)
        except Exception:
            log.exception("LanguageModel could not record the turn turn_id=%s", identifier)
            result = ModelResult("interrupted", error_code="worker_exception")
            answer = {'text': '', 'error_code': result.error_code}
            self.store.finish(identifier, result)
        if coordination:
            try:
                coordination.finish(result.status, result.error_code, result=answer)
            except ModelError as error:
                self.store.event(identifier, "coordination_reconciliation_required", {"code": error.code})
        log.info("LanguageModel turn finished turn_id=%s status=%s error_code=%s",
                 identifier, result.status, result.error_code)

    def cancel(self, turn_id: str) -> dict:
        return self.store.cancel(turn_id)

    def reconcile(self, turn_id: str, confirmed_stopped: bool) -> dict:
        if not confirmed_stopped:
            raise ModelError("confirmation_required", "Confirm that the worker and any remote job have stopped.")
        with self._lock:
            future = self._futures.get(turn_id)
            if future and not future.done():
                raise ModelError("busy", "This worker is still executing the turn; cancel it first.")
            return self.store.reconcile(turn_id)

    def close(self):
        with self._lock:
            self._stopping.set()
        self._executor.shutdown(wait=True)
