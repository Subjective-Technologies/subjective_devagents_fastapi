from .models import Action, ExecutionOutcome, ExecutionPolicy, SchedulerDecision, SchedulerError, SchedulerMetadata, TaskRequest


class SimpleV1:
    metadata = SchedulerMetadata(
        'simple_v1', '1.0.0', 'Direct LLM',
        'One LLM executes the request directly with authorized computer tools. No handoffs or specialized role required.',
    )
    execution_policy = ExecutionPolicy(
        guard_coder_scoping=False, auto_acquire_write=True, fixed_deadline=True,
        publish_task_workspace=False, parse_legacy_reply=False, direct_prompt=True,
    )

    def compatible(self, request: TaskRequest) -> bool:
        return bool(request.agents)

    def start(self, request: TaskRequest) -> SchedulerDecision:
        eligible = request.agents
        if request.write_required:
            allowed = set(request.write_capable_agents)
            eligible = tuple(agent for agent in request.agents if agent in allowed)
        if request.active_agent and request.active_agent not in eligible:
            message = ('The original task agent lacks authorized workspace write access'
                       if request.write_required else 'The original task agent is unavailable')
            raise SchedulerError('SCHEDULER_AGENT_UNAVAILABLE', message)
        if request.active_agent:
            return SchedulerDecision(Action.DISPATCH_AGENT, (request.active_agent,))
        addressed = tuple(agent for agent in request.addressed if agent in eligible)
        if len(request.addressed) == 1 and len(addressed) == 1:
            return SchedulerDecision(Action.DISPATCH_AGENT, addressed)
        if request.write_required and len(request.addressed) == 1 and not addressed:
            raise SchedulerError('SCHEDULER_AGENT_UNAVAILABLE',
                                 'The addressed AI user lacks authorized workspace write access')
        if not request.addressed:
            if len(eligible) == 1:
                return SchedulerDecision(Action.DISPATCH_AGENT, eligible)
            if request.default_agent in eligible:
                return SchedulerDecision(Action.DISPATCH_AGENT, (request.default_agent,))
        if request.write_required and not eligible:
            raise SchedulerError('SCHEDULER_AGENT_UNAVAILABLE',
                                 'No available AI user has authorized workspace write access')
        raise SchedulerError('SCHEDULER_AGENT_SELECTION_REQUIRED', 'Address one available AI user to start this task')

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
