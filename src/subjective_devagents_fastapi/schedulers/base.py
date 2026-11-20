from typing import Protocol

from .models import Action, ExecutionOutcome, ExecutionPolicy, SchedulerDecision, SchedulerMetadata, TaskRequest


class Scheduler(Protocol):
    metadata: SchedulerMetadata
    execution_policy: ExecutionPolicy

    def compatible(self, request: TaskRequest) -> bool: ...

    def start(self, request: TaskRequest) -> SchedulerDecision: ...

    def on_outcome(self, result: ExecutionOutcome) -> SchedulerDecision: ...


class CompatibilityScheduler:
    """Bridge existing journal transitions during rollout."""

    execution_policy = ExecutionPolicy()
    scoping_action: Action

    def on_outcome(self, result: ExecutionOutcome) -> SchedulerDecision:
        action = self.scoping_action if (
            result.status != 'completed' and result.error_code in {'coder_scoping_limit', 'CODER_SCOPING_LIMIT', 'SCOPING_LIMIT'}
        ) else Action.APPLY_COMPATIBILITY_RESULT
        return SchedulerDecision(action)
