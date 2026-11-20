"""Installation-authenticated planning for the Chat journal."""
from dataclasses import asdict
import hmac
import os
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
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
    pinned: SchedulerPin | None = None
    preference: Identifier | None = None


def authorized(token: str) -> bool:
    expected = os.environ.get('COORDINATION_SCHEDULER_TOKEN', '')
    return len(expected) >= 32 and hmac.compare_digest(expected.encode(), token.encode())


@router.post('/catalog')
def catalog(repository_available: bool = False, x_scheduler_token: str = Header(default='')):
    if not authorized(x_scheduler_token):
        raise HTTPException(401, 'Scheduler service authentication is required')
    request = TaskRequest(('available',), implementation_owners=('available',),
                          repository_available=repository_available)
    entries = configured_registry().candidates(request)
    return {'schedulers': [asdict(entry) for entry in entries]}


@router.post('/plan')
def plan(body: PlanRequest, x_scheduler_token: str = Header(default='')):
    if not authorized(x_scheduler_token):
        raise HTTPException(401, 'Scheduler service authentication is required')
    agents = tuple(dict.fromkeys(body.agents))
    request = TaskRequest(
        agents, tuple(agent for agent in dict.fromkeys(body.addressed) if agent in agents),
        orchestrators=tuple(a for a in body.orchestrators if a in agents),
        dedicated_orchestrators=tuple(a for a in body.dedicated_orchestrators if a in agents),
        implementation_owners=tuple(sorted(set(body.implementation_owners).intersection(agents))),
        repository_available=body.repository_available, active_agent=body.active_agent,
    )
    try:
        scheduler = select(configured_registry(), request, pinned=body.pinned, preference=body.preference)
        decision = scheduler.start(request)
    except SchedulerError as error:
        return JSONResponse(status_code=409, content={'error': error.code, 'message': str(error)})
    return {'scheduler': asdict(scheduler.metadata.pin), 'decision': asdict(decision),
            'structured_outcomes': not scheduler.execution_policy.parse_legacy_reply}
