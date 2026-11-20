"""Collaboration schedulers hosted by DevAgents."""
import os

from .legacy import LegacyMultiAgent
from .registry import SchedulerRegistry
from .simple_v1 import SimpleV1
from .single_owner import SingleOwner


def configured_registry() -> SchedulerRegistry:
    registry = SchedulerRegistry()
    mode = os.environ.get('COLLABORATION_EXECUTION_MODE', '').strip().lower()
    original = 'single_owner' if mode in {'single_owner', 'single_owner_pipeline'} else 'legacy_multi_agent'
    default = os.environ.get('COLLABORATION_DEFAULT_SCHEDULER', original).strip()
    disabled = {value.strip() for value in os.environ.get('COLLABORATION_DISABLED_SCHEDULERS', '').split(',')}
    for scheduler in (LegacyMultiAgent(), SingleOwner(), SimpleV1()):
        identifier = scheduler.metadata.id
        registry.register(scheduler, enabled=identifier not in disabled, default=identifier == default)
    return registry
