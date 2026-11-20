from .base import CompatibilityScheduler
from .models import Action, SchedulerDecision, SchedulerMetadata, TaskRequest


class LegacyMultiAgent(CompatibilityScheduler):
    metadata = SchedulerMetadata(
        'legacy_multi_agent', '1.0.0', 'Legacy multi-agent',
        'Manager-led workflow with specialist handoffs.',
    )

    def compatible(self, request: TaskRequest) -> bool:
        return bool(request.agents)

    def start(self, request: TaskRequest) -> SchedulerDecision:
        managers = request.dedicated_orchestrators or request.orchestrators
        targets = request.addressed or managers[:1] or request.agents[:1]
        return SchedulerDecision(Action.DISPATCH_AGENT, targets, manager=managers[0] if managers else None)
