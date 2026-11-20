import re

from .base import CompatibilityScheduler
from .models import Action, ExecutionOutcome, ExecutionPolicy, SchedulerDecision, SchedulerMetadata, TaskRequest


class LegacyMultiAgent(CompatibilityScheduler):
    scoping_action = Action.REQUEST_RESCOPE
    execution_policy = ExecutionPolicy(prompt_suffix=(
        '\n\nLegacy handoff rules:\n'
        '- The coordinating Manager names the next agent using their exact @name.\n'
        '- Specialists report one authorized outcome and mention Manager. Use IMPLEMENTATION_COMPLETE, '
        'VERIFICATION_PASS, VERIFICATION_FAIL, PUBLICATION_COMPLETE <sha>, DEPLOYMENT_VERIFIED <sha>, or BLOCKED.\n'
        '- Manager closes with DONE: <evidence> or NEEDS INPUT: <question>. '
        'Use OPTIONAL_USER_CONFIRMATION when all system-observable criteria have been verified.'
    ))
    metadata = SchedulerMetadata(
        'legacy_multi_agent', '1.0.0', 'Legacy multi-agent',
        'Manager-led workflow with specialist handoffs.',
    )

    def compatible(self, request: TaskRequest) -> bool:
        return bool(request.published_role_request().agents)

    def start(self, request: TaskRequest) -> SchedulerDecision:
        request = request.published_role_request()
        managers = request.dedicated_orchestrators or request.orchestrators
        targets = request.addressed or managers[:1] or request.agents[:1]
        return SchedulerDecision(Action.DISPATCH_AGENT, targets, manager=managers[0] if managers else None)


    def on_outcome(self, result: ExecutionOutcome) -> SchedulerDecision:
        assignment = result.assignment
        if result.status != 'completed' or not assignment:
            return super().on_outcome(result)
        mentioned = tuple(agent for agent, names in assignment.roster
                          if agent != assignment.agent_id and any(
                              re.search(r'@' + re.escape(name.lstrip('@')) + r'(?![\w-])', result.reply, re.I)
                              for name in names if name
                          ))
        if assignment.agent_id == assignment.manager:
            targets = mentioned
        else:
            targets = (assignment.manager,) if assignment.manager else ()
        return SchedulerDecision(Action.ROUTE_LEGACY_RESULT, targets, mentioned_agents=mentioned)
