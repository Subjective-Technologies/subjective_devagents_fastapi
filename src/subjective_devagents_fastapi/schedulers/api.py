"""Planning for Chat using existing company authorization."""
from dataclasses import asdict
import hmac
import os
from typing import Annotated
from urllib.request import HTTPRedirectHandler, build_opener

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from . import configured_registry
from .models import SchedulerError, SchedulerPin, TaskRequest
from .selector import select


router = APIRouter(prefix='/api/collaboration/schedulers')
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class PlanRequest(BaseModel):
    agents: list[Identifier] = Field(max_length=40)
    addressed: list[Identifier] = Field(default_factory=list, max_length=40)
    orchestrators: list[Identifier] = Field(default_factory=list, max_length=40)
    dedicated_orchestrators: list[Identifier] = Field(default_factory=list, max_length=40)
    implementation_owners: list[Identifier] = Field(default_factory=list, max_length=40)
    repository_available: bool = False
    active_agent: Identifier | None = None
    unbound_agents: list[Identifier] = Field(default_factory=list, max_length=40)
    pinned: SchedulerPin | None = None
    preference: Identifier | None = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


_workflow_opener = build_opener(_NoRedirect())


def authorized(token: str, authorization: str = '') -> bool:
    # Installed service tokens keep working during rollout and background recovery.
    expected = os.environ.get('COORDINATION_SCHEDULER_TOKEN', '')
    if len(expected) >= 32 and hmac.compare_digest(expected.encode(), token.encode()):
        return True
    company = os.environ.get('COORDINATION_COMPANY_ID', '')
    if not company or len(authorization) > 16384 or not authorization.lower().startswith('bearer '):
        return False
    from ..agent_workflow import WorkflowUnavailable, fetch_export

    try:
        # WorkTwins verifies the existing login and access to this runtime's company.
        fetch_export(company, authorization, opener=_workflow_opener.open)
    except WorkflowUnavailable:
        return False
    return True


@router.post('/catalog')
def catalog(repository_available: bool = False):
    request = TaskRequest(('available',), implementation_owners=('available',),
                          repository_available=repository_available)
    entries = configured_registry().candidates(request)
    return {'schedulers': [asdict(entry) for entry in entries]}


@router.post('/plan')
def plan(body: PlanRequest):
    agents = tuple(dict.fromkeys(body.agents))
    request = TaskRequest(
        agents, tuple(agent for agent in dict.fromkeys(body.addressed) if agent in agents),
        orchestrators=tuple(a for a in body.orchestrators if a in agents),
        dedicated_orchestrators=tuple(a for a in body.dedicated_orchestrators if a in agents),
        implementation_owners=tuple(sorted(set(body.implementation_owners).intersection(agents))),
        repository_available=body.repository_available, active_agent=body.active_agent,
        default_agent=os.environ.get("COLLABORATION_DEFAULT_DIRECT_AGENT"),
        unbound_agents=tuple(agent for agent in body.unbound_agents if agent in agents),
    )
    try:
        scheduler = select(configured_registry(), request, pinned=body.pinned, preference=body.preference)
        decision = scheduler.start(request)
    except SchedulerError as error:
        return JSONResponse(status_code=409, content={'error': error.code, 'message': str(error)})
    return {'scheduler': asdict(scheduler.metadata.pin), 'decision': asdict(decision),
            'structured_outcomes': not scheduler.execution_policy.parse_legacy_reply}
