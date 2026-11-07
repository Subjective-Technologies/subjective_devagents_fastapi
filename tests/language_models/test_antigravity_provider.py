"""Antigravity discovery must produce executable bindings, not only a menu label."""
import pwd
import os

import pytest

from subjective_devagents_fastapi.language_models import catalog, legacy
from subjective_devagents_fastapi.language_models.contracts import ModelError
from subjective_devagents_fastapi.language_models.registry import IdentityRegistry
from subjective_devagents_fastapi.llm_service import writable_provider
from subjective_devagents_fastapi.d1_sync import map_provider


def test_antigravity_binding_uses_discovered_default_and_selected_model(monkeypatch):
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://proxy/v1')
    monkeypatch.delenv(legacy.PROXY_MODEL_ENV['antigravity'], raising=False)
    monkeypatch.setattr(catalog, 'proxy_models', lambda: {'antigravity': ['claude-proxy', 'gemini-proxy']})
    account = pwd.getpwuid(os.geteuid())
    assert legacy.proxy_configured('antigravity')
    assert writable_provider('antigravity')
    identity = legacy.proxy_identity('antigravity', account.pw_name, '/tmp', account.pw_dir)
    assert identity.model == 'claude-proxy'
    assert identity.provider == 'antigravity' and identity.strategy == 'api_agent'
    IdentityRegistry([identity])
    chosen = legacy.proxy_identity('antigravity', account.pw_name, '/tmp', account.pw_dir, 'gemini-proxy')
    assert chosen.model == 'gemini-proxy' and chosen.id != identity.id
    assert map_provider('antigravity') == 'antigravity'


def test_antigravity_default_override_and_unavailable_catalog(monkeypatch):
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://proxy/v1')
    monkeypatch.setenv(legacy.PROXY_MODEL_ENV['antigravity'], 'operator-default')
    assert legacy.proxy_default_model('antigravity', {}) == 'operator-default'
    monkeypatch.delenv(legacy.PROXY_MODEL_ENV['antigravity'])
    def unavailable():
        raise ModelError('model_catalog_unavailable', 'Unavailable')
    monkeypatch.setattr(catalog, 'proxy_models', unavailable)
    assert not legacy.proxy_configured('antigravity')
    assert legacy.proxy_identity('antigravity', 'unused', '/tmp', '/tmp') is None


def test_antigravity_turn_runs_registered_strategy(monkeypatch, tmp_path):
    from subjective_devagents_fastapi.language_models.contracts import ModelResult, ModelStrategy
    from subjective_devagents_fastapi.llm_service import LlmService
    monkeypatch.delenv('SUBJECTIVE_LANGUAGE_MODELS_CONFIG', raising=False)
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://proxy/v1')
    monkeypatch.setenv('SUBJECTIVE_LANGUAGE_MODELS_DB', str(tmp_path / 'turns.sqlite3'))
    monkeypatch.setattr(catalog, 'proxy_models', lambda: {'antigravity': ['gemini-proxy']})
    requests = []
    class Recorder(ModelStrategy):
        def run(self, request, emit, cancelled):
            requests.append(request)
            return ModelResult('completed', 'fixture answer')
    service = LlmService(root=str(tmp_path), app_home=str(tmp_path / 'home'))
    service.backend.model.strategies['api_agent:antigravity'] = Recorder()
    try:
        conversation = service.create_conversation('antigravity', model='gemini-proxy')
        result = service.backend.send(conversation, {}, {'prompt': 'test prompt', 'timeout': 5}, lambda *a: None, lambda *a: None, None)
        assert result['success']
        assert requests[0].identity.provider == 'antigravity'
        assert requests[0].identity.model == 'gemini-proxy'
    finally:
        service.backend.close()
