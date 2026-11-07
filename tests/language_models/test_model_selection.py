"""Model discovery and binding checks without provider quota consumption."""
import json
from unittest.mock import Mock

import httpx
import pytest

from subjective_devagents_fastapi.language_models import catalog, legacy
from subjective_devagents_fastapi.language_models.contracts import ModelError, ModelResult, ModelStrategy
from subjective_devagents_fastapi.llm_service import LlmService


@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.delenv('SUBJECTIVE_LANGUAGE_MODELS_CONFIG', raising=False)
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://127.0.0.1:8317/v1')
    monkeypatch.setenv(legacy.PROXY_KEY_ENV, 'test-only-key')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['codex'], 'gpt-test-v1')
    monkeypatch.setattr(catalog, 'proxy_models', lambda: {'codex': ['gpt-test-v1', 'gpt-test-v2'], 'claude': ['claude-test']})


class Recorder(ModelStrategy):
    def __init__(self):
        self.requests = []

    def run(self, request, emit, cancelled):
        self.requests.append(request)
        return ModelResult('completed', 'answer')


def test_discovery_groups_by_family_and_owner_and_excludes_media(monkeypatch):
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://proxy/v1')
    monkeypatch.setenv(legacy.PROXY_KEY_ENV, 'test-only-key')
    response = httpx.Response(200, request=httpx.Request('GET', 'http://proxy/v1/models'), json={'data': [
        {'id': 'gpt-test', 'owned_by': 'openai'}, {'id': 'claude-test', 'owned_by': 'anthropic'},
        {'id': 'gpt-image-2', 'owned_by': 'openai'}, {'id': 'claude-other', 'owned_by': 'antigravity'},
        {'id': 'gemini-test', 'owned_by': 'antigravity'},
        {'id': 'gpt-test', 'owned_by': 'openai'}, {'id': 'grok-test', 'owned_by': 'xai'}]})
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.return_value = response
    monkeypatch.setattr(catalog.httpx, 'Client', lambda **kwargs: client)
    assert catalog.proxy_models() == {'codex': ['gpt-test'], 'claude': ['claude-other', 'claude-test'], 'grok': ['grok-test'], 'kimi': [], 'antigravity': ['claude-other', 'gemini-test']}
    catalog.validate_model('claude', 'claude-other')
    with pytest.raises(ModelError, match='not available'):
        catalog.validate_model('claude', 'gemini-test')


def test_wrong_provider_and_unknown_model_rejected(proxy):
    for model in ['claude-test', 'gpt-missing', '../bad']:
        with pytest.raises(ModelError):
            catalog.validate_model('codex', model)


def test_model_handoff_preserves_history_and_survives_reload(proxy, tmp_path, monkeypatch):
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODELS_DB', str(tmp_path / 'turns.sqlite3'))
    home = str(tmp_path / 'home')
    service = LlmService(root=str(tmp_path), app_home=home)
    recorder = Recorder()
    service.backend.model.strategies = {'api_agent:codex': recorder}
    conversation = service.create_conversation('codex', model='gpt-test-v1')
    def send(prompt):
        result = service.backend.send(conversation, {}, {'prompt': prompt, 'timeout': 5}, lambda *a: None, lambda *a: None, None)
        assert result['success']
        service.backend.projected(conversation['target'], result['turn_id'])
        return result
    try:
        first = send('first question')
        service.set_model(conversation['target'], 'gpt-test-v2')
        second = send('second question')
        assert first['conversation_id'] == second['conversation_id']
        assert [r.identity.model for r in recorder.requests] == ['gpt-test-v1', 'gpt-test-v2']
        assert [m.content for m in recorder.requests[1].history] == ['first question', 'answer']
        assert recorder.requests[1].session_id is None
        saved = json.loads(open(service.state_path).read())
        assert saved['conversations'][0]['model'] == 'gpt-test-v2'
        reloaded = LlmService(root=str(tmp_path), app_home=home)
        try:
            assert reloaded.find_conversation(conversation['target'])['model'] == 'gpt-test-v2'
        finally:
            reloaded.backend.close()
    finally:
        service.backend.close()


def test_active_turn_blocks_handoff_without_changing_model(proxy, tmp_path, monkeypatch):
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODELS_DB', str(tmp_path / 'turns.sqlite3'))
    service = LlmService(root=str(tmp_path), app_home=str(tmp_path / 'home'))
    try:
        conversation = service.create_conversation('codex', model='gpt-test-v1')
        identity = service.backend.identity(conversation)
        bound = service.backend.model.create(identity.id)
        with service.backend.model.store.connection() as db:
            db.execute('INSERT INTO legacy_bindings(target,conversation_id) VALUES (?,?)', (conversation['target'], bound['id']))
        service.backend.model.store.reserve(bound['id'], identity, 'request', 'question', 5)
        with pytest.raises(ModelError, match='active turn'):
            service.set_model(conversation['target'], 'gpt-test-v2')
        assert conversation['model'] == 'gpt-test-v1'
    finally:
        service.backend.close()
