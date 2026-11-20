from .models import Action, ExecutionOutcome, ExecutionPolicy, SchedulerDecision, SchedulerError, SchedulerMetadata, TaskRequest


class SimpleV1:
    metadata = SchedulerMetadata(
        'simple_v1', '1.0.0', 'Simple',
        'One agent works directly on the task with its existing permissions.',
    )
    execution_policy = ExecutionPolicy(
        guard_coder_scoping=False, auto_acquire_write=True, fixed_deadline=True,
        publish_task_workspace=False, parse_legacy_reply=False,
    )

    def compatible(self, request: TaskRequest) -> bool:
        return bool(request.agents)

    def start(self, request: TaskRequest) -> SchedulerDecision:
        if request.active_agent and request.active_agent not in request.agents:
            raise SchedulerError('SCHEDULER_AGENT_UNAVAILABLE', 'The original task agent is unavailable')
        agents = ((request.active_agent,) if request.active_agent else
                  request.addressed[:1] or request.agents[:1])
        return SchedulerDecision(Action.DISPATCH_AGENT, agents)

    def on_outcome(self, result: ExecutionOutcome) -> SchedulerDecision:
        if result.decision_id:
            return SchedulerDecision(Action.REQUEST_USER_INPUT, decision_id=result.decision_id)
        if result.error_code == 'service_restarted':
            return SchedulerDecision(Action.RESUME_EXECUTION)
        action = {
            'completed': Action.COMPLETE_TASK, 'blocked': Action.BLOCK_TASK,
            'cancelled': Action.CANCEL_EXECUTION,
        }.get(result.status, Action.FAIL_TASK)
        return SchedulerDecision(action)
