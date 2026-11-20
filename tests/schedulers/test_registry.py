from dataclasses import replace

import pytest

from subjective_devagents_fastapi.schedulers.api import PlanRequest, plan
from subjective_devagents_fastapi.schedulers.legacy import LegacyMultiAgent
from subjective_devagents_fastapi.schedulers.models import Action, ExecutionOutcome, SchedulerError, SchedulerPin, TaskRequest
from subjective_devagents_fastapi.schedulers.registry import SchedulerRegistry
from subjective_devagents_fastapi.schedulers.selector import select
from subjective_devagents_fastapi.schedulers.simple_v1 import SimpleV1
from subjective_devagents_fastapi.schedulers.single_owner import SingleOwner


REQUEST = TaskRequest(('ai_one',))


def test_lone_enabled_scheduler_needs_no_default():
    registry = SchedulerRegistry()
    registry.register(SimpleV1())
    registry.register(LegacyMultiAgent(), enabled=False)
    assert select(registry, REQUEST).metadata.id == 'simple_v1'
    assert [entry.id for entry in registry.list(enabled_only=True)] == ['simple_v1']


@pytest.mark.parametrize('reverse', [False, True])
def test_default_is_independent_of_registration_order(reverse):
    registry = SchedulerRegistry()
    entries = [(SimpleV1(), False), (LegacyMultiAgent(), True)]
    for scheduler, default in reversed(entries) if reverse else entries:
        registry.register(scheduler, default=default)
    assert select(registry, REQUEST).metadata.id == 'legacy_multi_agent'
    assert select(registry, REQUEST, preference='simple_v1').metadata.id == 'simple_v1'


def test_duplicate_and_conflicting_defaults_are_rejected():
    registry = SchedulerRegistry()
    registry.register(SimpleV1(), default=True)
    with pytest.raises(ValueError, match='Duplicate'):
        registry.register(SimpleV1())
    registry.register(LegacyMultiAgent(), default=True)
    with pytest.raises(SchedulerError) as failure:
        select(registry, REQUEST)
    assert failure.value.code == 'SCHEDULER_CONFIGURATION'


@pytest.mark.parametrize('preference', [None, 'simple_v1'])
def test_no_scheduler_and_disabled_preference_are_explicit(preference):
    registry = SchedulerRegistry()
    registry.register(SimpleV1(), enabled=False)
    if preference:
        registry.register(LegacyMultiAgent(), default=True)
    with pytest.raises(SchedulerError) as failure:
        select(registry, REQUEST, preference=preference)
    assert failure.value.code == 'SCHEDULER_UNAVAILABLE'


def test_pin_wins_and_must_match_exact_version():
    registry = SchedulerRegistry()
    old = SimpleV1()
    new = SimpleV1()
    new.metadata = replace(old.metadata, version='1.1.0')
    registry.register(old)
    registry.register(new)
    registry.register(LegacyMultiAgent(), default=True)
    assert select(registry, REQUEST, preference='simple_v1') is new
    assert select(registry, REQUEST, preference='legacy_multi_agent', pinned=old.metadata.pin) is old
    with pytest.raises(SchedulerError, match='simple_v1@0.0.1'):
        select(registry, REQUEST, pinned=SchedulerPin('simple_v1', '0.0.1'))


def test_single_owner_requires_compatible_host_and_agent():
    registry = SchedulerRegistry()
    registry.register(SimpleV1())
    registry.register(SingleOwner(), default=True)
    assert select(registry, REQUEST).metadata.id == 'simple_v1'
    request = TaskRequest(('ai_one', 'ai_two'), addressed=('ai_one',),
                          implementation_owners=('ai_two',), repository_available=True)
    selected = select(registry, request)
    assert selected.start(request).agents == ('ai_two',)
    assert selected.start(request).action is Action.START_PIPELINE


def test_simple_consumes_typed_outcome_and_keeps_original_agent():
    scheduler = SimpleV1()
    request = TaskRequest(('ai_one', 'ai_two'), addressed=('ai_two',), active_agent='ai_one')
    assert scheduler.start(request).agents == ('ai_one',)
    assert scheduler.on_outcome(ExecutionOutcome('failed')).action is Action.FAIL_TASK
    assert scheduler.on_outcome(ExecutionOutcome('cancelled')).action is Action.CANCEL_EXECUTION
    decision = scheduler.on_outcome(ExecutionOutcome('blocked', 'question-1'))
    assert decision.action is Action.REQUEST_USER_INPUT
    assert decision.decision_id == 'question-1'


def test_plan_routes_manager_by_authorized_skill_and_preserves_explicit_targets(monkeypatch):
    token = 'test-scheduler-service-token-not-a-secret'
    monkeypatch.setenv('COORDINATION_SCHEDULER_TOKEN', token)
    monkeypatch.setenv('COLLABORATION_DEFAULT_SCHEDULER', 'legacy_multi_agent')
    monkeypatch.delenv('COLLABORATION_DISABLED_SCHEDULERS', raising=False)
    request = PlanRequest(agents=['ai_worker', 'ai_planner'], orchestrators=['ai_planner'])
    result = plan(request, token)
    assert result['decision']['agents'] == ('ai_planner',)
    request.addressed = ['ai_worker']
    result = plan(request, token)
    assert result['decision']['agents'] == ('ai_worker',)
    assert result['decision']['manager'] == 'ai_planner'


def test_legacy_strategy_returns_manager_handoff_candidates():
    from subjective_devagents_fastapi.schedulers.models import Assignment

    scheduler = LegacyMultiAgent()
    roster = (('ai_planner', ('Manager',)), ('ai_worker', ('Coder',)), ('ai_other', ('Other',)))
    manager = scheduler.on_outcome(ExecutionOutcome(
        'completed', assignment=Assignment('ai_planner', 'ai_planner', roster),
        reply='@Coder fix the test. @OtherAgent is outside the roster.',
    ))
    worker = scheduler.on_outcome(ExecutionOutcome(
        'completed', assignment=Assignment('ai_worker', 'ai_planner', roster), reply='@Other please continue',
    ))
    assert manager.action is Action.ROUTE_LEGACY_RESULT
    assert manager.agents == ('ai_worker',)
    assert worker.agents == ('ai_planner',)
    assert worker.mentioned_agents == ('ai_other',)


@pytest.mark.parametrize(('message', 'needs_write'), [
    ('@Claude, please edit value.txt', True),
    ('@Claude, can you fix the tests?', True),
    ('Hey @Claude, please edit value.txt', True),
    ('@Claude — could you explain the failure?', False),
])
def test_direct_assignment_preserves_write_intent_in_addressed_messages(message, needs_write):
    from subjective_devagents_fastapi.collaboration import addressed_assignment
    from subjective_devagents_fastapi.runtime_profiles import requirements_for

    instruction = addressed_assignment(message, {'display_name': 'Claude'}, 'ai_claude')
    assert ('fs_write' in requirements_for(instruction).runtime_capabilities()) is needs_write
