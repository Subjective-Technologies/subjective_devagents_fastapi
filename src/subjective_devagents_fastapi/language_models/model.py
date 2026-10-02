"""LanguageModel facade: identity selection and durable, supervised turn lifecycle."""
from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from .contracts import MAX_ATTACHMENTS, Attachment, Identity, Message, ModelError, ModelRequest, ModelResult, ModelStrategy
from .registry import IdentityRegistry, image_support
from .store import TurnStore
from .strategies import Api, ApiAgent, ClaudeCli, Cli, CodexCli, CursorCli, GrokCli, MAX_CONTEXT_BYTES

log = logging.getLogger(__name__)


class LanguageModel:
    def __init__(self, store: TurnStore, identities: IdentityRegistry,
                 strategies: dict[str, ModelStrategy] | None = None, max_workers: int = 4):
        self.store = store
        self.identities = identities
        self.strategies = strategies if strategies is not None else {
            "cli:codex": Cli(CodexCli()), "cli:claude": Cli(ClaudeCli()),
            "cli:cursor": Cli(CursorCli()), "cli:grok": Cli(GrokCli()), "api": Api(),
            # One ApiAgent instance serves every provider the proxy fronts: tool declarations
            # and step naming are provider-agnostic. Cursor has no equivalent backend there,
            # so it stays on cli: above.
            "api_agent:codex": ApiAgent(), "api_agent:claude": ApiAgent(), "api_agent:grok": ApiAgent(),
        }
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="language-model")
        self._lock = threading.RLock()
        self._futures: dict[str, Future] = {}
        # Image bytes never enter SQLite; an accepted turn's images live here until it starts.
        self._attachments: dict[str, tuple[Attachment, ...]] = {}
        self._stopping = threading.Event()

    def create(self, identity_id: str, history: tuple[Message, ...] = ()) -> dict:
        if sum(len(item.content.encode()) for item in history) > MAX_CONTEXT_BYTES:
            raise ModelError("context_too_large", "Imported conversation exceeds the transfer limit.")
        return self.store.create(self.identities.get(identity_id), history)

    def handoff(self, conversation_id: str, identity_id: str) -> dict:
        return self.store.handoff(conversation_id, self.identities.get(identity_id))

    def submit(self, conversation_id: str, prompt: str, request_id: str, timeout: float = 300,
               attachments: tuple[Attachment, ...] = ()) -> dict:
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
                if attachments:
                    self._attachments[turn["id"]] = tuple(attachments)
                    self.store.event(turn["id"], "attachments", {"images": [item.metadata() for item in attachments]})
                try:
                    future = self._executor.submit(self._execute, turn["id"])
                    self._futures[turn["id"]] = future
                    future.add_done_callback(lambda done, key=turn["id"]: self._forget(key))
                except RuntimeError:
                    self._attachments.pop(turn["id"], None)
                    self.store.finish(turn["id"], ModelResult("failed", error_code="dispatch_failed"))
                    raise ModelError("unavailable", "Could not dispatch the accepted turn.")
            return turn

    def _forget(self, identifier: str):
        with self._lock:
            self._futures.pop(identifier, None)

    def _execute(self, identifier: str):
        with self._lock:
            attachments = self._attachments.pop(identifier, ())
        if not self.store.claim(identifier):
            return
        turn = self.store.turn(identifier)
        log.info("LanguageModel turn started turn_id=%s identity=%s", identifier, turn["identity"]["id"])
        try:
            identity = Identity(**turn["identity"])
            history = tuple(Message(**item) for item in self.store.conversation(turn["conversation_id"])["messages"])
            request = ModelRequest(identifier, identity, turn["prompt"], history, turn["session_id"], turn["timeout"],
                                   attachments)
            strategy = self.strategies.get(f"{identity.strategy}:{identity.provider}") or self.strategies.get(identity.strategy)
            if strategy is None:
                raise ModelError("unsupported_provider", "No execution strategy is registered.")
            result = strategy.run(request, lambda kind, data: self.store.event(identifier, kind, data),
                                  lambda: self._stopping.is_set() or bool(self.store.turn(identifier)["cancel_requested"]))
        except ModelError as error:
            result = ModelResult("failed", error_code=error.code)
        except Exception as error:
            # The strategy may have started work. Preserve uncertainty; do not retry.
            log.error("LanguageModel worker exception turn_id=%s exception_type=%s", identifier, type(error).__name__)
            result = ModelResult("interrupted", error_code="worker_exception")
        self.store.finish(identifier, result)
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
