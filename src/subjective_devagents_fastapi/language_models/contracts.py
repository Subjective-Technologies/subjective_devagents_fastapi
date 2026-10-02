"""Provider-neutral contracts. This package has no datasource dependencies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ModelRequest:
    turn_id: str
    identity: Identity
    prompt: str
    history: tuple[Message, ...] = ()
    session_id: str | None = None
    timeout: float = 300.0


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
