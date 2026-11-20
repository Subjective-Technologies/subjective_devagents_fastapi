"""Versioned scheduler inputs and actions. No transport or execution dependencies."""
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class SchedulerPin:
    id: str
    version: str


@dataclass(frozen=True)
class SchedulerMetadata:
    id: str
    version: str
    display_name: str
    description: str
    enabled: bool = True
    is_default: bool = False

    @property
    def pin(self) -> SchedulerPin:
        return SchedulerPin(self.id, self.version)


@dataclass(frozen=True)
class TaskRequest:
    agents: tuple[str, ...]
    addressed: tuple[str, ...] = ()
    orchestrators: tuple[str, ...] = ()
    dedicated_orchestrators: tuple[str, ...] = ()
    implementation_owners: tuple[str, ...] = ()
    repository_available: bool = False
    active_agent: str | None = None


class Action(str, Enum):
    DISPATCH_AGENT = 'DISPATCH_AGENT'
    START_PIPELINE = 'START_PIPELINE'
    COMPLETE_TASK = 'COMPLETE_TASK'
    FAIL_TASK = 'FAIL_TASK'
    RESUME_EXECUTION = 'RESUME_EXECUTION'
    BLOCK_TASK = 'BLOCK_TASK'
    REQUEST_USER_INPUT = 'REQUEST_USER_INPUT'
    CANCEL_EXECUTION = 'CANCEL_EXECUTION'
    APPLY_COMPATIBILITY_RESULT = 'APPLY_COMPATIBILITY_RESULT'
    ROUTE_LEGACY_RESULT = 'ROUTE_LEGACY_RESULT'
    REQUEST_RESCOPE = 'REQUEST_RESCOPE'
    BLOCK_IMPLEMENTATION_SCOPE = 'BLOCK_IMPLEMENTATION_SCOPE'


@dataclass(frozen=True)
class SchedulerDecision:
    action: Action
    agents: tuple[str, ...] = ()
    decision_id: str | None = None
    manager: str | None = None
    mentioned_agents: tuple[str, ...] = ()


@dataclass(frozen=True)
class Assignment:
    agent_id: str
    manager: str | None
    roster: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ExecutionOutcome:
    status: str
    decision_id: str | None = None
    error_code: str | None = None
    assignment: Assignment | None = None
    reply: str = ''


@dataclass(frozen=True)
class ExecutionPolicy:
    guard_coder_scoping: bool = True
    auto_acquire_write: bool = False
    fixed_deadline: bool = False
    publish_task_workspace: bool = True
    parse_legacy_reply: bool = True
    prompt_suffix: str = ''


class SchedulerError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
