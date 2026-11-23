"""Credential-free text-model discovery from the operator-configured proxy."""
import os
import hashlib
import threading
import time
import re

import httpx

from .contracts import ModelError

# Native provider owners and gateway routes are distinct from model families.
OWNERS = {"codex": {"openai", "codex"}, "claude": {"anthropic", "claude"},
          "grok": {"xai", "grok"}, "kimi": {"moonshot", "kimi"},
          "deepseek": {"deepseek"}, "antigravity": {"antigravity"}}
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
MEDIA = re.compile(r"image|video|audio|realtime|transcrib|tts|embedding", re.I)


CACHE_SECONDS = 60
_cache_lock = threading.Lock()
_cache_key = None
_cached_models = None
_cache_expires = 0.0


def proxy_models(force: bool = False) -> dict[str, list[str]]:
    """A short, credential-scoped cache serializes concurrent gateway discovery."""
    from .legacy import PROXY_URL_ENV, PROXY_KEY_ENV, PROXY_MODEL_ENV
    global _cache_key, _cached_models, _cache_expires
    url, key = os.environ.get(PROXY_URL_ENV, "").strip(), os.environ.get(PROXY_KEY_ENV, "")
    if not url or not key:
        raise ModelError("model_catalog_unavailable", "Model discovery requires a configured inference proxy.")
    configured_models = tuple(sorted((provider, os.environ.get(env_name, "").strip())
                                     for provider, env_name in PROXY_MODEL_ENV.items()))
    cache_key = (url, hashlib.sha256(key.encode()).digest(), configured_models)
    with _cache_lock:
        if not force and _cache_key == cache_key and _cached_models is not None and time.monotonic() < _cache_expires:
            return {provider: list(models) for provider, models in _cached_models.items()}
        models = _load_proxy_models(url, key)
        _cache_key, _cached_models = cache_key, models
        _cache_expires = time.monotonic() + CACHE_SECONDS
        return {provider: list(ids) for provider, ids in models.items()}


def _load_proxy_models(url: str, key: str) -> dict[str, list[str]]:
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
            response = client.get(url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
            rows = response.json()["data"]
        if not isinstance(rows, list):
            raise ValueError("Invalid model list")
        grouped = {provider: set() for provider in OWNERS}
        proxy_ids = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("id")
            if not isinstance(name, str) or not MODEL_ID.fullmatch(name) or MEDIA.search(name):
                continue
            proxy_ids.add(name)
            owner = str(row.get("owned_by") or "").strip().casefold()
            owner_key = re.sub(r"[^a-z0-9]", "", owner)
            for provider, owners in OWNERS.items():
                native = owner in owners
                # CLIProxy's OpenAI-compatible API-key routes can expose DeepSeek
                # models with the generic `openai` owner. Keep them selectable under
                # DeepSeek by their explicit model namespace; do not reassign them to
                # another provider or accept unrelated OpenAI models as DeepSeek.
                if name.casefold().startswith("deepseek-"):
                    native = provider == "deepseek"
                if provider == "deepseek" and "deepseek" in owner_key:
                    native = True
                # Antigravity exposes both Gemini and Claude. Include only its Claude
                # family here, retaining the exact gateway ID for inference routing.
                routed_claude = (provider == "claude" and owner == "antigravity"
                                 and name.startswith("claude-"))
                if native or routed_claude:
                    grouped[provider].add(name)
        # An operator-pinned provider model is authoritative routing metadata. Some
        # OpenAI-compatible API-key entries use a generic owner and an arbitrary alias;
        # retain that exact model under the provider explicitly configured for it.
        from .legacy import PROXY_MODEL_ENV
        for provider, env_name in PROXY_MODEL_ENV.items():
            configured = os.environ.get(env_name, "").strip()
            if configured and configured in proxy_ids:
                grouped[provider].add(configured)
        return {provider: sorted(models) for provider, models in grouped.items()}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ModelError("model_catalog_unavailable", "Could not read the inference proxy model catalog.") from None


def validate_model(provider: str, model: str) -> None:
    if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
        raise ModelError("invalid_model", "Invalid model ID.")
    available = proxy_models().get(provider, [])
    if model not in available:
        # Name the pinned model and what the proxy actually serves. "Selected model is not
        # available for this provider" could not tell a stale pin (the chat keeps a model id
        # the proxy has stopped serving) from an upstream that went away entirely — the
        # grouping below keys on owned_by, so a changed owner empties a provider silently.
        # Model ids are not secrets; the key and the proxy URL stay out of this.
        raise ModelError("model_unavailable",
                         f"The inference proxy does not serve {provider} model {model}. "
                         + (f"It currently serves: {', '.join(available[:8])}." if available
                            else f"It currently lists no {provider} model at all, "
                                 f"so that provider's upstream is unavailable."))
