from .base import Scheduler
from .models import SchedulerError, SchedulerPin, TaskRequest
from .registry import SchedulerRegistry


def select(
    registry: SchedulerRegistry,
    request: TaskRequest,
    *,
    pinned: SchedulerPin | None = None,
    preference: str | None = None,
) -> Scheduler:
    if pinned is not None:
        return registry.lookup(pinned, request)
    candidates = registry.candidates(request)
    if preference is not None:
        for metadata in candidates:
            if metadata.id == preference:
                return registry.lookup(metadata.pin)
        raise SchedulerError('SCHEDULER_UNAVAILABLE', f'Scheduler {preference} is unavailable for this task')
    if not candidates:
        raise SchedulerError('SCHEDULER_UNAVAILABLE', 'No compatible execution scheduler is enabled')
    selected = candidates[0] if len(candidates) == 1 else registry.default(candidates)
    return registry.lookup(selected.pin)
