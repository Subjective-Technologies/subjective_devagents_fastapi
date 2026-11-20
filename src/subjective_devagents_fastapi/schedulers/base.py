from typing import Protocol

from .models import ExecutionOutcome, ExecutionPolicy, SchedulerDecision, SchedulerMetadata, TaskRequest


class Scheduler(Protocol):
    metadata: SchedulerMetadata
    execution_policy: ExecutionPolicy

    def compatible(self, request: TaskRequest) -> bool: ...

    def start(self, request: TaskRequest) -> SchedulerDecision: ...

    def on_outcome(self, result: ExecutionOutcome) -> SchedulerDecision: ...


class CompatibilityScheduler:
    """Bridge existing journal transitions during rollout."""

    execution_policy = ExecutionPolicy()

    def on_outcome(self, result: ExecutionOutcome) -> SchedulerDecision:
        from .models import Action

        return SchedulerDecision(Action.APPLY_COMPATIBILITY_RESULT)
