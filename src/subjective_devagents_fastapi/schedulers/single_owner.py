from .base import CompatibilityScheduler
from .models import Action, SchedulerDecision, SchedulerMetadata, TaskRequest


class SingleOwner(CompatibilityScheduler):
    metadata = SchedulerMetadata(
        'single_owner', '1.0.0', 'Single owner',
        'One implementation owner with controller-led verification and delivery.',
    )

    def compatible(self, request: TaskRequest) -> bool:
        return request.repository_available and bool(request.implementation_owners)

    def start(self, request: TaskRequest) -> SchedulerDecision:
        requested = request.addressed[:1]
        owner = requested if requested and requested[0] in request.implementation_owners else request.implementation_owners[:1]
        return SchedulerDecision(Action.START_PIPELINE, owner)
