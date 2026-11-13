"""Discovery cache must reduce gateway work without mixing credential scopes."""
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from subjective_devagents_fastapi.language_models import catalog, legacy
from subjective_devagents_fastapi.language_models.contracts import ModelError


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    monkeypatch.setenv(legacy.PROXY_URL_ENV, 'http://test-proxy/v1')
    monkeypatch.setenv(legacy.PROXY_KEY_ENV, 'fixture-key')
    monkeypatch.setattr(catalog, '_cached_models', None)
    monkeypatch.setattr(catalog, '_cache_key', None)
    monkeypatch.setattr(catalog, '_cache_expires', 0)


def test_parallel_and_repeated_reads_use_one_gateway_request(monkeypatch):
    calls = []
    def load(*args):
        calls.append(1)
        time.sleep(0.02)
        return {'claude': ['discovered-model']}
    monkeypatch.setattr(catalog, '_load_proxy_models', load)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: catalog.proxy_models(), range(8)))
    assert len(calls) == 1
    rows[0]['claude'].append('client-mutation')
    assert catalog.proxy_models()['claude'] == ['discovered-model']


def test_expiry_refresh_and_credential_change_fetch_again(monkeypatch):
    calls = []
    monkeypatch.setattr(catalog, '_load_proxy_models', lambda *args: calls.append(1) or {'claude': [str(len(calls))]})
    assert catalog.proxy_models()['claude'] == ['1']
    assert catalog.proxy_models(force=True)['claude'] == ['2']
    monkeypatch.setattr(catalog, '_cache_expires', 0)
    assert catalog.proxy_models()['claude'] == ['3']
    monkeypatch.setenv(legacy.PROXY_KEY_ENV, 'other-fixture-key')
    assert catalog.proxy_models()['claude'] == ['4']


def test_failed_discovery_is_not_cached(monkeypatch):
    calls = []
    def load(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ModelError('model_catalog_unavailable', 'Unavailable')
        return {'claude': ['recovered']}
    monkeypatch.setattr(catalog, '_load_proxy_models', load)
    with pytest.raises(ModelError):
        catalog.proxy_models()
    assert catalog.proxy_models()['claude'] == ['recovered']
