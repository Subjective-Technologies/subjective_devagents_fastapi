"""Browser-facing model discovery and conversation binding endpoints."""
import pytest

from subjective_devagents_fastapi.language_models import catalog, legacy
from subjective_devagents_fastapi.language_models.contracts import ModelError


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://proxy/v1')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['codex'], 'gpt-test-v1')
    monkeypatch.setattr(catalog, 'proxy_models', lambda: {'codex': ['gpt-test-v1', 'gpt-test-v2'], 'claude': ['claude-test']})


def test_catalog_exposes_provider_models_and_default(client, models):
    response = client.get('/api/llm/providers')
    assert response.status_code == 200
    codex = next(row for row in response.json()['providers'] if row['key'] == 'codex')
    assert codex['models'] == ['gpt-test-v1', 'gpt-test-v2']
    assert codex['defaultModel'] == 'gpt-test-v1'


def test_create_switch_and_reload_binding(client, models):
    created = client.post('/api/llm/create-conversation', json={'provider': 'codex', 'model': 'gpt-test-v1'}).json()
    assert created['ok']
    target = created['conversation']['target']
    changed = client.post('/api/llm/set-model', json={'target': target, 'model': 'gpt-test-v2'})
    assert changed.status_code == 200
    assert changed.json()['conversation']['model'] == 'gpt-test-v2'
    read = client.get('/api/llm/conversation', params={'target': target}).json()
    assert read['conversation']['model'] == 'gpt-test-v2'
    rejected = client.post('/api/llm/set-model', json={'target': target, 'model': 'claude-test'})
    assert rejected.status_code == 422
    assert rejected.json()['detail']['code'] == 'model_unavailable'
    assert client.get('/api/llm/conversation', params={'target': target}).json()['conversation']['model'] == 'gpt-test-v2'


def test_model_switch_requires_auth(bare_token_client):
    assert bare_token_client.post('/api/llm/set-model', json={'target': 'anything', 'model': 'anything'}).status_code == 401


def test_catalog_failure_keeps_provider_discovery(client, monkeypatch):
    def unavailable():
        raise ModelError('model_catalog_unavailable', 'Proxy unavailable')
    monkeypatch.setattr(catalog, 'proxy_models', unavailable)
    body = client.get('/api/llm/providers').json()
    assert body['ok'] and body['providers']
    assert body['modelError'] == 'model_catalog_unavailable'
    assert all(not provider['models'] for provider in body['providers'])
