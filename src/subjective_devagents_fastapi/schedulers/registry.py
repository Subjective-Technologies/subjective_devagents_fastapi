"""Registry of explicitly installed scheduler implementations."""
from dataclasses import replace
import re

from .base import Scheduler
from .models import SchedulerError, SchedulerMetadata, SchedulerPin, TaskRequest


class SchedulerRegistry:
    def __init__(self):
        self._entries: dict[SchedulerPin, tuple[Scheduler, SchedulerMetadata]] = {}

    def register(self, scheduler: Scheduler, *, enabled: bool = True, default: bool = False) -> None:
        metadata = replace(scheduler.metadata, enabled=enabled, is_default=default)
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', metadata.id) or not re.fullmatch(r'\d+\.\d+\.\d+', metadata.version):
            raise ValueError('Scheduler ID or version is invalid')
        if metadata.pin in self._entries:
            raise ValueError(f'Duplicate scheduler registration: {metadata.id}@{metadata.version}')
        self._entries[metadata.pin] = scheduler, metadata

    @staticmethod
    def _version(metadata: SchedulerMetadata) -> tuple[int, ...]:
        return tuple(int(part) for part in metadata.version.split('.'))

    def list(self, *, enabled_only: bool = False) -> tuple[SchedulerMetadata, ...]:
        return tuple(sorted(
            (metadata for _, metadata in self._entries.values() if metadata.enabled or not enabled_only),
            key=lambda metadata: (metadata.id, self._version(metadata)),
        ))

    def candidates(self, request: TaskRequest | None = None) -> tuple[SchedulerMetadata, ...]:
        current: dict[str, SchedulerMetadata] = {}
        for metadata in self.list(enabled_only=True):
            if request is None or self._entries[metadata.pin][0].compatible(request):
                current[metadata.id] = metadata
        return tuple(current.values())

    def lookup(self, pin: SchedulerPin, request: TaskRequest | None = None) -> Scheduler:
        entry = self._entries.get(pin)
        if not entry or not entry[1].enabled or (request is not None and not entry[0].compatible(request)):
            raise SchedulerError('SCHEDULER_UNAVAILABLE', f'Scheduler {pin.id}@{pin.version} is unavailable for this task')
        return entry[0]

    @staticmethod
    def default(candidates: tuple[SchedulerMetadata, ...]) -> SchedulerMetadata:
        defaults = [metadata for metadata in candidates if metadata.is_default]
        if len(defaults) != 1:
            raise SchedulerError('SCHEDULER_CONFIGURATION', 'Multiple schedulers require exactly one compatible default')
        return defaults[0]
