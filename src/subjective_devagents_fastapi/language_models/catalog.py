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
          "grok": {"xai", "grok"}, "kimi": {"moonshot", "kimi"}}
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
MEDIA = re.compile(r"image|video|audio|realtime|transcrib|tts|embedding", re.I)


CACHE_SECONDS = 60
_cache_lock = threading.Lock()
_cache_key = None
_cached_models = None
_cache_expires = 0.0


def proxy_models(force: bool = False) -> dict[str, list[str]]:
    """A short, credential-scoped cache serializes concurrent gateway discovery."""
    from .legacy import PROXY_URL_ENV, PROXY_KEY_ENV
    global _cache_key, _cached_models, _cache_expires
    url, key = os.environ.get(PROXY_URL_ENV, "").strip(), os.environ.get(PROXY_KEY_ENV, "")
    if not url or not key:
        raise ModelError("model_catalog_unavailable", "Model discovery requires a configured inference proxy.")
    cache_key = (url, hashlib.sha256(key.encode()).digest())
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
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("id")
            if not isinstance(name, str) or not MODEL_ID.fullmatch(name) or MEDIA.search(name):
                continue
            for provider, owners in OWNERS.items():
                native = row.get("owned_by") in owners
                # Antigravity exposes both Gemini and Claude. Include only its Claude
                # family here, retaining the exact gateway ID for inference routing.
                routed_claude = (provider == "claude" and row.get("owned_by") == "antigravity"
                                 and name.startswith("claude-"))
                if native or routed_claude:
                    grouped[provider].add(name)
        return {provider: sorted(models) for provider, models in grouped.items()}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ModelError("model_catalog_unavailable", "Could not read the inference proxy model catalog.") from None


def validate_model(provider: str, model: str) -> None:
    if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
        raise ModelError("invalid_model", "Invalid model ID.")
    if model not in proxy_models().get(provider, []):
        raise ModelError("model_unavailable", "Selected model is not available for this provider.")
