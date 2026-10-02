"""Opt-in HTTP surface protected by the application's existing owner auth gate."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

import base64
import binascii

from .contracts import Attachment, MAX_ATTACHMENTS, Message, ModelError
from .model import LanguageModel
from .registry import IdentityRegistry
from .store import TurnStore

router = APIRouter(prefix="/api/language-model", tags=["LanguageModel"])


class Payload(BaseModel):
    # Attachments/options are not silently ignored: unsupported features fail validation.
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(Payload):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=512 * 1024)


class CreateConversation(Payload):
    identity_id: str = Field(min_length=1, max_length=80)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=1000)


class Handoff(Payload):
    identity_id: str = Field(min_length=1, max_length=80)


class Image(Payload):
    filename: str = Field(default="image", max_length=200)
    mime_type: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=14 * 1024 * 1024, description="Base64 image bytes")

    def attachment(self) -> Attachment:
        try:
            data = base64.b64decode(self.content, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ModelError("invalid_attachment", "An attached image is not valid base64.") from error
        return Attachment(self.filename, self.mime_type.lower(), data)


class Send(Payload):
    prompt: str = Field(min_length=1, max_length=512 * 1024)
    request_id: str = Field(min_length=1, max_length=128)
    timeout: float = Field(default=300, ge=1, le=1800, allow_inf_nan=False)
    attachments: list[Image] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)


class Reconcile(Payload):
    confirmed_stopped: bool = False


def configured_model() -> LanguageModel | None:
    config = os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_CONFIG", "")
    if not config:
        return None
    registry = IdentityRegistry.load(config)
    home = Path(os.environ.get("SUBJECTIVE_CHAT_HOME", "~/.subjective_chat")).expanduser()
    database = Path(os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_DB", str(home / "language_models.sqlite3"))).expanduser()
    return LanguageModel(TurnStore(database), registry)


def model(request: Request) -> LanguageModel:
    instance = getattr(request.app.state, "language_model", None)
    if instance is None:
        raise HTTPException(503, detail={"code": "disabled", "message": "LanguageModel is not configured."})
    return instance


def invoke(function, *args):
    try:
        return function(*args)
    except ModelError as error:
        status = (404 if error.code in {"not_found", "unknown_identity"}
                  else 409 if error.code in {"busy", "identity_busy", "identity_changed", "idempotency_conflict",
                                            "reconciliation_required", "invalid_state"}
                  else 503 if error.code in {"unavailable", "overloaded"} else 422)
        raise HTTPException(status, detail={"code": error.code, "message": str(error)}) from error


@router.get("/identities")
def identities(request: Request):
    return {"identities": model(request).identities.public()}


@router.post("/conversations", status_code=201)
def create(body: CreateConversation, request: Request):
    return invoke(model(request).create, body.identity_id, tuple(Message(**item.model_dump()) for item in body.history))


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: str, request: Request):
    return invoke(model(request).store.conversation, conversation_id)


@router.post("/conversations/{conversation_id}/handoff")
def handoff(conversation_id: str, body: Handoff, request: Request):
    return invoke(model(request).handoff, conversation_id, body.identity_id)


@router.post("/conversations/{conversation_id}/turns", status_code=202)
def send(conversation_id: str, body: Send, request: Request):
    return invoke(lambda: model(request).submit(conversation_id, body.prompt, body.request_id, body.timeout,
                                                tuple(item.attachment() for item in body.attachments)))


@router.get("/turns/{turn_id}")
def turn(turn_id: str, request: Request):
    return invoke(model(request).store.turn, turn_id)


@router.get("/turns/{turn_id}/events")
def events(turn_id: str, request: Request, after: int = Query(default=0, ge=0)):
    return {"events": invoke(model(request).store.events, turn_id, after)}


@router.post("/turns/{turn_id}/cancel")
def cancel(turn_id: str, request: Request):
    return invoke(model(request).cancel, turn_id)


@router.post("/turns/{turn_id}/reconcile")
def reconcile(turn_id: str, body: Reconcile, request: Request):
    return invoke(model(request).reconcile, turn_id, body.confirmed_stopped)
