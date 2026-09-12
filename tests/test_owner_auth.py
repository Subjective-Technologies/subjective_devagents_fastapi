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


@pytest.fixture
def signer(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(firebase_auth, '_signing_key_for', lambda token: private.public_key())
    def issue(**overrides):
        now = int(time.time())
        claims = dict(sub=auth.OWNER_UID, email='changed@example.com', email_verified=False,
                      aud=firebase_auth.FIREBASE_PROJECT_ID, iss=firebase_auth.ISSUER,
                      iat=now, auth_time=now, exp=now+3600)
        claims.update(overrides)
        return jwt.encode(claims, private, algorithm='RS256', headers={'kid': 'test-key'})
    return issue


@pytest.fixture
def owner_client(tmp_path, monkeypatch):
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
    {},  # UID alone is sufficient, regardless of changed email.
    {'sub': 'another-uid', 'email': auth.OWNER_EMAIL, 'email_verified': True},
    {'sub': 'another-uid', 'email': auth.OWNER_EMAIL.upper(), 'email_verified': True},
])
def test_owner_uid_or_verified_email(owner_client, signer, changes):
    assert owner_client.get('/api/auth/session', headers={'Authorization': 'Bearer '+signer(**changes)}).status_code == 200


@pytest.mark.parametrize('changes', [
    {'sub': 'other', 'email': 'other@example.com', 'email_verified': True},
    {'sub': 'other', 'email': auth.OWNER_EMAIL, 'email_verified': False},
    {'exp': int(time.time()) - 120},
    {'aud': 'wrong-project'},
    {'iss': 'https://securetoken.google.com/wrong-project'},
    {'auth_time': int(time.time()) + 120},
])
def test_invalid_identity_and_claims_rejected(owner_client, signer, changes):
    assert owner_client.get('/api/config', headers={'Authorization': 'Bearer '+signer(**changes)}).status_code == 401


def test_static_query_and_unsigned_tokens_cannot_bypass_owner(owner_client, signer):
    tokens = ['legacy-secret-must-not-bypass', 'bad.token.value', jwt.encode({'sub': auth.OWNER_UID}, '', algorithm='none')]
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
    assert response.json() == {'mode': 'owner', 'required': True}
    assert 'legacy-secret' not in response.text
    assert owner_client.get('/healthz').json() == {'ok': True}


def test_socket_authenticates_first_frame_before_data(owner_client, signer):
    with owner_client.websocket_connect('/ws') as socket:
        socket.send_json({'type': 'auth', 'token': signer()})
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
        socket.send_json({'type': 'auth', 'token': token})
        assert socket.receive_json()['type'] == 'config'
        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                socket.receive_json()
        assert closed.value.code == 4401


def test_tampered_signature_is_rejected(owner_client, signer):
    token = signer()
    header, payload, signature = token.split('.')
    signature = ('A' if signature[0] != 'A' else 'B') + signature[1:]
    response = owner_client.get('/api/auth/session', headers={'Authorization': f'Bearer {header}.{payload}.{signature}'})
    assert response.status_code == 401
