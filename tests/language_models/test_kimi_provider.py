"""Every provider the picker offers through the proxy has a strategy that runs its turns."""

from subjective_devagents_fastapi.language_models.legacy import PROXY_MODEL_ENV
from subjective_devagents_fastapi.language_models.model import LanguageModel
from subjective_devagents_fastapi.language_models.registry import IdentityRegistry
from subjective_devagents_fastapi.language_models.store import TurnStore
from subjective_devagents_fastapi.language_models.strategies import ApiAgent


def test_every_proxied_provider_has_an_api_agent_strategy(tmp_path):
    model = LanguageModel(TurnStore(str(tmp_path / "turns.sqlite3")), IdentityRegistry([]))
    for provider in PROXY_MODEL_ENV:
        assert isinstance(model.strategies.get(f"api_agent:{provider}"), ApiAgent), provider
    assert "api_agent:kimi" in model.strategies
