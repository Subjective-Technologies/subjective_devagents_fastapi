"""Provider-neutral contracts. This package has no datasource dependencies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable


class ModelError(Exception):
    """A public, credential-free failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Identity:
    """Operator-defined account binding; never accepted as arbitrary HTTP input.

    Change the id when changing the account, runtime, model, or credential location.
    home/config_dir are paths on the execution host, not necessarily this server.
    """

    id: str
    provider: str
    model: str = ""
    workspace: str = "/tmp"
    home: str = ""
    config_dir: str = ""
    os_user: str = ""
    ssh_host: str = ""
    executable: str = ""
    strategy: str = "cli"
    api_base_url: str = ""
    api_key_env: str = ""
    sandbox: str = "read-only"
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))

    def snapshot(self) -> dict:
        return {**asdict(self), "allowed_tools": list(self.allowed_tools)}


@dataclass(frozen=True)
class Message:
    role: str
    content: str


IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENTS = 10


@dataclass(frozen=True)
class Attachment:
    """An image sent with one turn. Held in memory only; the turn store keeps metadata."""

    name: str
    mime_type: str
    data: bytes = field(repr=False)

    def __post_init__(self):
        if self.mime_type not in IMAGE_TYPES:
            raise ModelError("unsupported_attachments", "Only PNG, JPEG, GIF and WebP images can be attached.")
        if not self.data or len(self.data) > MAX_ATTACHMENT_BYTES:
            raise ModelError("attachment_too_large", "Each image must be nonempty and at most 10 MiB.")

    def metadata(self) -> dict:
        return {"name": self.name, "mime_type": self.mime_type, "bytes": len(self.data)}


@dataclass(frozen=True)
class ModelRequest:
    turn_id: str
    identity: Identity
    prompt: str
    history: tuple[Message, ...] = ()
    session_id: str | None = None
    timeout: float = 300.0
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True)
class ModelResult:
    status: str
    text: str = ""
    session_id: str | None = None
    error_code: str | None = None


Emit = Callable[[str, dict], None]
Cancelled = Callable[[], bool]


class ModelStrategy(ABC):
    """An execution strategy must run at most once and report its actual outcome."""

    @abstractmethod
    def run(self, request: ModelRequest, emit: Emit, cancelled: Cancelled) -> ModelResult:
        raise NotImplementedError
