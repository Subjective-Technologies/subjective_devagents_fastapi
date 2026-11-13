"""Owner policy and transport tests with real RSA signature verification."""
import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from subjective_devagents_fastapi import auth, firebase_auth
from subjective_devagents_fastapi import app as app_module

TEST_OWNER_EMAIL = 'owner@example.com'
TEST_OWNER_UID = 'uid-owner-test'
TEST_FIREBASE_PROJECT = 'firebase-test-project'
TEST_COMPANY = 'acme'
FIREBASE_USER = 'firebase-user'


def identity(token, user_id=FIREBASE_USER, company=TEST_COMPANY):
    return {
        'Authorization': 'Bearer ' + token,
        'X-User-Id': user_id,
        'X-Company-Name': company,
    }


def auth_frame(token, user_id=FIREBASE_USER, company=TEST_COMPANY):
    return {'type': 'auth', 'token': token, 'user_id': user_id, 'company_name': company}


@pytest.fixture
def signer(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_OWNER_EMAIL', TEST_OWNER_EMAIL)
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_OWNER_UID', TEST_OWNER_UID)
    monkeypatch.setenv('FIREBASE_PROJECT_ID', TEST_FIREBASE_PROJECT)
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_COMPANY_NAME', TEST_COMPANY)
    monkeypatch.setattr(firebase_auth, '_signing_key_for', lambda token: private.public_key())
    def issue(**overrides):
        now = int(time.time())
        claims = dict(sub='firebase-user', email='user@example.com', email_verified=True,
                      aud=firebase_auth.firebase_project_id(), iss=firebase_auth.issuer(),
                      iat=now, auth_time=now, exp=now+3600)
        claims.update(overrides)
        return jwt.encode(claims, private, algorithm='RS256', headers={'kid': 'test-key'})
    return issue


@pytest.fixture
def owner_client(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_OWNER_EMAIL', TEST_OWNER_EMAIL)
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_OWNER_UID', TEST_OWNER_UID)
    monkeypatch.setenv('FIREBASE_PROJECT_ID', TEST_FIREBASE_PROJECT)
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_COMPANY_NAME', TEST_COMPANY)
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_AUTH_MODE', 'owner')
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_API_TOKEN', 'legacy-secret-must-not-bypass')
    monkeypatch.setenv('SUBJECTIVE_CHAT_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_CLIENT_ROOT', str(tmp_path / 'client'))
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_D1_SYNC', '0')
    monkeypatch.setenv('SUBJECTIVE_CHAT_HISTORY_REFRESH_MS', '0')
    monkeypatch.delenv('SUBJECTIVE_CHAT_IRC', raising=False)
    with TestClient(app_module.app) as client:
        yield client


def test_default_and_invalid_modes_require_owner(monkeypatch):
    monkeypatch.delenv('SUBJECTIVE_DEVAGENTS_AUTH_MODE', raising=False)
    monkeypatch.delenv('SUBJECTIVE_DEVAGENTS_API_TOKEN', raising=False)
    assert auth.auth_enabled()
    assert not auth.token_is_valid('')
    monkeypatch.setenv('SUBJECTIVE_DEVAGENTS_AUTH_MODE', 'typo')
    assert auth.auth_enabled()
    assert not auth.token_is_valid('')


@pytest.mark.parametrize('changes', [
    {},
    {'sub': 'another-uid', 'email': 'other@example.com', 'email_verified': True},
    {'sub': 'another-uid', 'email': 'Other@Example.com', 'email_verified': False},
])
def test_any_firebase_account_is_accepted(owner_client, signer, changes):
    token = signer(**changes)
    assert owner_client.get('/api/auth/session', headers=identity(token, changes.get('sub', FIREBASE_USER))).status_code == 200


class Relative:
    """A claim time expressed as an offset, resolved when the test runs.

    A parametrize list is built once, at import. `int(time.time()) + 120` written
    there freezes "two minutes from now" at collection, so the future auth_time
    case stopped being in the future as soon as a full run took longer than two
    minutes to reach it — and the test then failed, having quietly stopped
    testing what it claims some time before that.
    """

    def __init__(self, seconds: int):
        self.seconds = seconds

    def resolve(self) -> int:
        return int(time.time()) + self.seconds


@pytest.mark.parametrize('changes', [
    {'exp': Relative(-120)},
    {'aud': 'wrong-project'},
    {'iss': 'https://securetoken.google.com/wrong-project'},
    {'auth_time': Relative(+120)},
])
def test_invalid_identity_and_claims_rejected(owner_client, signer, changes):
    changes = {key: value.resolve() if isinstance(value, Relative) else value
               for key, value in changes.items()}
    token = signer(**changes)
    response = owner_client.get('/api/config', headers=identity(token, changes.get('sub', FIREBASE_USER)))
    assert response.status_code == 401
    assert response.json()['error'] == 'invalid firebase token'


def test_static_query_and_unsigned_tokens_cannot_bypass_owner(owner_client, signer):
    tokens = ['legacy-secret-must-not-bypass', 'bad.token.value', jwt.encode({'sub': 'firebase-user'}, '', algorithm='none')]
    for token in tokens:
        assert owner_client.get('/api/config', headers={'Authorization': 'Bearer '+token}).status_code == 401
    assert owner_client.get('/api/config', params={'token': signer()}).status_code == 401
    assert owner_client.post('/api/exec', json={'command': 'must not run'}).status_code == 401
    assert owner_client.get('/docs').status_code == 401


def test_cors_preflight_and_denial(owner_client):
    origin = 'https://devagents.subjectivetechnologies.com'
    response = owner_client.options('/api/llm/send-message', headers={
        'Origin': origin, 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'authorization,content-type',
    })
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] in ('*', origin)
    denied = owner_client.get('/api/config', headers={'Origin': origin})
    assert denied.status_code == 401
    assert denied.headers['access-control-allow-origin'] in ('*', origin)


def test_public_bootstrap_contains_no_credentials(owner_client):
    response = owner_client.get('/api/auth/config')
    body = response.json()
    assert body['mode'] == 'owner'
    assert body['required'] is True
    assert body['ownerEmail'] == TEST_OWNER_EMAIL
    assert body['companyName'] == TEST_COMPANY
    assert body['firebase']['apiKey'] == ''
    assert 'legacy-secret' not in response.text
    assert owner_client.get('/healthz').json() == {'ok': True}


def test_socket_authenticates_first_frame_before_data(owner_client, signer):
    with owner_client.websocket_connect('/ws') as socket:
        socket.send_json(auth_frame(signer()))
        assert socket.receive_json()['type'] == 'config'


@pytest.mark.parametrize('message', [{}, {'type':'auth','token':'bad'}, {'type':'auth','token':None}])
def test_socket_denies_before_broadcast_registration(owner_client, message):
    before = len(app_module.state.hub.queues)
    with pytest.raises(WebSocketDisconnect):
        with owner_client.websocket_connect('/ws') as socket:
            socket.send_json(message)
            socket.receive_json()
    assert len(app_module.state.hub.queues) == before


def test_socket_closes_at_token_expiry(owner_client, signer):
    token = signer(exp=time.time()+0.5)
    with owner_client.websocket_connect('/ws') as socket:
        socket.send_json(auth_frame(token))
        assert socket.receive_json()['type'] == 'config'
        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                socket.receive_json()
        assert closed.value.code == 4401


def test_tampered_signature_is_rejected(owner_client, signer):
    token = signer()
    header, payload, signature = token.split('.')
    signature = ('A' if signature[0] != 'A' else 'B') + signature[1:]
    response = owner_client.get('/api/auth/session', headers=identity(f'{header}.{payload}.{signature}'))
    assert response.status_code == 401
    assert response.json()['error'] == 'invalid firebase token'


def test_firebase_token_alone_is_refused(owner_client, signer):
    response = owner_client.get('/api/config', headers={'Authorization': 'Bearer ' + signer()})
    assert response.status_code == 401
    assert response.json()['error'] == 'user id, firebase token, and company name are required'


def test_user_id_must_be_the_token_subject(owner_client, signer):
    response = owner_client.get('/api/config', headers=identity(signer(), 'not-the-subject'))
    assert response.status_code == 401
    assert response.json()['error'] == 'invalid firebase token'


def test_company_must_match_this_server(owner_client, signer):
    response = owner_client.get('/api/config', headers=identity(signer(), company='other'))
    assert response.status_code == 403
    assert response.json()['error'] == 'company name does not match this server'
