"""A restart must not lose the turn or the prompt (production incident 2026-11-21 15:44:22 -0300).

Root ran `systemctl restart subjective-devagents-fastapi.service` while a Claude turn was
17 minutes in (conversation llm:claude:local:63535c4f, dispatch 86760d1a, lease 23c0a4cd,
`ms: 1018860`). What the journal recorded:

* `Coordination unavailable; retrying op=checkpoint` 1-4, then the lease declared lost,
* `[error] LLM send failed` with `coordination_lease_lost:coordination_unavailable`, and
  the plain-text prompt rolled back,
* `Coordination unavailable; retrying op=finish` still running after uvicorn's
  `Shutting down`,
* 30 s later `State 'stop-sigterm' timed out`, SIGKILL of pid 3541347 and three children,
  `Failed with result 'timeout'`,
* after the restart, `LLM conversation hydrate` for that target reported
  `messageCount: 0` where it had been 1.

So: close() was an unbounded wait, the interruption was classified as a failed task
instead of `service_restarted` (which `simple_v1.on_outcome` resumes), and the prompt was
thrown away. `TimeoutStopSec=30` stays the deadline the code has to meet.
"""
from __future__ import annotations

import threading
import time

import pytest

from subjective_devagents_fastapi.channel_coordination import UNREACHABLE_GRACE, ChannelSession
from subjective_devagents_fastapi.language_models import (
    Identity, IdentityRegistry, LanguageModel, ModelResult, ModelStrategy, TurnStore,
)
from subjective_devagents_fastapi.language_models.contracts import ModelError
from subjective_devagents_fastapi.language_models.model import _close_timeout

# What systemd allows SIGTERM in the live unit (/etc/systemd/system/...service).
STOP_TIMEOUT_SECONDS = 30.0


class Unavailable:
    """A journal that answers until `failing` is set, like a redeploying Chat API."""

    def __init__(self):
        self.calls: list[str] = []
        self.failing = False
        self.state = {'enabled': True, 'workflow_id': 'workflow', 'workflow_version': 'published',
                      'task_family': 'TASK', 'task_revision': '1', 'revision': 1,
                      **{key: {} for key in ('facts', 'leases', 'decisions', 'capabilities',
                                             'reviews', 'gates', 'worktrees', 'discoveries')}}

    def __call__(self, body):
        self.calls.append(body['op'])
        if self.failing:
            raise ModelError('coordination_unavailable', 'Channel coordination is unavailable')
        return {'state': self.state}


@pytest.fixture
def runtime(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    identity = Identity('runtime', 'claude', model='test', strategy='api_agent', workspace=str(workspace),
                        home=str(tmp_path), sandbox='workspace-write', api_base_url='http://unused',
                        api_key_env='TEST_KEY')
    store = TurnStore(str(tmp_path / 'turns.sqlite3'))
    model = LanguageModel(store, IdentityRegistry([identity]), strategies={}, max_workers=4)
    yield model, store, identity
    model.close(timeout=2)


def session(identity, journal, *, delegated=True):
    bound = ChannelSession({'channel_id': 'channel', 'dispatch_id': 'dispatch',
                            'instance_id': 'ai_coder:run', 'workflow_id': 'workflow'},
                           'ai_coder', 'Bearer test-only', client=journal)
    bound.start(identity, 'published', 'coder_agent')
    if delegated:
        # A delegated dispatch owns a durable receipt; that is what makes a local
        # completion replayable after the restart (collaboration.replay_completions).
        bound.delegation = {**bound.binding, 'owner_id': 'owner'}
    return bound


def test_the_close_deadline_is_inside_the_units_stop_timeout():
    assert _close_timeout() < STOP_TIMEOUT_SECONDS, 'close() must finish before systemd SIGKILLs the unit'


def test_close_returns_even_when_a_worker_ignores_the_stop_signal(runtime, monkeypatch):
    """`Executor.shutdown(wait=True)` never came back; the wait has to be bounded."""
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODEL_CLOSE_TIMEOUT', '2')
    model, store, identity = runtime
    started, release = threading.Event(), threading.Event()

    class Deaf(ModelStrategy):
        def run(self, request, emit, cancelled):
            started.set()
            release.wait(timeout=60)      # a provider HTTP call that does not look at cancelled()
            return ModelResult('completed', 'late')

    model.strategies['api_agent:claude'] = Deaf()
    conversation = model.create(identity.id)
    model.submit(conversation['id'], 'seventeen minutes of work', 'request-1', timeout=120)
    assert started.wait(5)

    began = time.monotonic()
    alive = model.close()
    elapsed = time.monotonic() - began
    assert elapsed < STOP_TIMEOUT_SECONDS / 2, f'close() took {elapsed:.1f}s'
    assert alive, 'the blocked worker is reported, not silently waited for'
    assert all(thread.daemon for thread in alive), 'a non-daemon worker keeps the process alive past SIGTERM'
    release.set()


def test_an_accepted_turn_that_never_started_is_recorded_not_left_queued(runtime, monkeypatch):
    """The sender polls for a terminal status; a row nothing finishes keeps that poll alive."""
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODEL_CLOSE_TIMEOUT', '2')
    model, store, identity = runtime
    started, release = threading.Event(), threading.Event()

    class Busy(ModelStrategy):
        def run(self, request, emit, cancelled):
            started.set()
            release.wait(timeout=60)
            return ModelResult('completed', 'late')

    model.strategies['api_agent:claude'] = Busy()
    model.max_workers = 1
    model._executor.max_workers = 1
    first, second = model.create(identity.id), model.create(identity.id)
    model.submit(first['id'], 'the turn in flight', 'request-1', timeout=120)
    assert started.wait(5)
    queued = model.submit(second['id'], 'the turn still waiting', 'request-2', timeout=120)
    assert store.turn(queued['id'])['status'] == 'queued'

    model.close()
    assert store.turn(queued['id'])['error_code'] == 'service_restarted'
    assert store.turn(queued['id'])['status'] not in {'queued', 'running'}
    release.set()


def test_a_lost_lease_during_shutdown_is_service_restarted_not_a_failed_task(runtime):
    """The journal's last word was coordination_unavailable; the task still has to resume."""
    model, store, identity = runtime
    journal = Unavailable()
    bound = session(identity, journal)
    stopping, finished = threading.Event(), threading.Event()

    class Working(ModelStrategy):
        def run(self, request, emit, cancelled):
            journal.failing = True        # the Chat API container goes away mid-turn
            stopping.wait(timeout=10)
            while not cancelled():        # the real loop polls between provider rounds
                time.sleep(0.01)
            # ApiAgent reports the revocation it observed; close() was already under way.
            return ModelResult('failed', error_code='coordination_lease_lost:' + (bound.revocation or 'unknown'))

    model.strategies['api_agent:claude'] = Working()
    conversation = model.create(identity.id)
    turn = model.submit(conversation['id'], 'seventeen minutes of work', 'request-1', timeout=120,
                        coordination=bound)
    threading.Thread(target=lambda: (finished.set(), model.close())).start()
    stopping.set()
    assert finished.wait(5)

    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline and store.turn(turn['id'])['status'] in {'queued', 'running'}:
        time.sleep(0.01)
    recorded = store.turn(turn['id'])
    assert recorded['error_code'] == 'service_restarted', recorded
    assert recorded['status'] == 'cancelled'

    from subjective_devagents_fastapi.schedulers.simple_v1 import SimpleV1
    from subjective_devagents_fastapi.schedulers.models import Action, ExecutionOutcome
    outcome = ExecutionOutcome('failed', None, recorded['error_code'])
    assert SimpleV1().on_outcome(outcome).action == Action.RESUME_EXECUTION


def test_finish_during_shutdown_makes_one_attempt_and_leaves_a_durable_receipt(runtime):
    """CALL_BACKOFF (1,2,4,8) on `finish` is what the SIGKILL interrupted."""
    model, store, identity = runtime
    journal = Unavailable()
    bound = session(identity, journal)
    bound.completion_store = store
    bound.stopping = lambda: True
    journal.failing = True
    journal.calls.clear()

    payload = {'binding': bound.binding, 'actor': bound.actor, 'delegation': bound.delegation,
               'data': {'status': 'failed', 'error_code': 'service_restarted'}}
    with store.connection() as db:
        db.execute('INSERT OR REPLACE INTO coordination_completions VALUES(?,?)',
                   (bound.binding['dispatch_id'], __import__('json').dumps(payload)))

    began = time.monotonic()
    bound.finish('cancelled', 'service_restarted', result={'text': '', 'error_code': 'service_restarted'})
    assert time.monotonic() - began < 1, 'finish retried instead of persisting the completion'
    assert journal.calls.count('finish') == 1
    with store.connection() as db:
        rows = db.execute('SELECT dispatch_id FROM coordination_completions').fetchall()
    assert rows, 'the completion has to survive for the next process to replay'
    assert bound.closed


def test_a_journal_outage_on_a_healthy_process_still_revokes_after_the_grace(runtime, monkeypatch):
    """UNREACHABLE_GRACE is untouched: only shutdown is the exception."""
    model, store, identity = runtime
    journal = Unavailable()
    bound = session(identity, journal, delegated=False)
    journal.failing = True

    clock = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    bound.last_check = 0.0
    assert bound.revoked() is False, 'one failed request is not a revocation'
    clock[0] += UNREACHABLE_GRACE + 1
    assert bound.revoked() is True
    assert bound.revocation == 'coordination_unavailable'


def test_the_same_outage_while_stopping_is_reported_as_a_restart(runtime, monkeypatch):
    model, store, identity = runtime
    journal = Unavailable()
    bound = session(identity, journal, delegated=False)
    journal.failing = True
    bound.stopping = lambda: True

    clock = [1000.0]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    bound.last_check = 0.0
    assert bound.revoked() is True, 'a stopping process cannot reach the journal again'
    assert bound.revocation == 'service_restarted'


def test_the_prompt_survives_the_restart_and_hydrate_still_sees_it(tmp_path):
    """`messageCount: 1` must not become `0`: the rollback is what emptied the conversation."""
    from subjective_devagents_fastapi.llm_service import LlmService

    service = LlmService(root=str(tmp_path / 'root'), app_home=str(tmp_path / 'home'))
    chat = service.create_conversation('claude', 'interrupted by a restart')
    # Exactly what LegacyLanguageModel.outcome() reports for the interrupted turn above.
    service.backend.send = lambda conversation, source, request, register, clear, progress: {
        'success': False, 'status': 'cancelled', 'error': 'service_restarted',
        'response': '', 'session_id': '', 'turn_id': 'turn-1',
    }

    response = service.send_message(chat['target'], 'seventeen minutes of work')
    assert response['ok'] is False
    assert response['continued'] is True and response['status'] == 'service_restarted'
    assert response['error'] != 'service_restarted', 'the UI shows text, not the runtime code'

    stored = service.find_conversation(chat['target'], include_messages=True)
    assert [m['text'] for m in stored['messages']] == ['seventeen minutes of work']
    user_message = stored['messages'][0]
    assert not user_message.get('stopped') and not user_message.get('sendFailed')
    # The successor's prompt history still contains it (history() drops stopped/failed lines).
    assert [m.content for m in service.backend.history(stored)] == ['seventeen minutes of work']
