"""Credential-free text-model discovery from the operator-configured proxy."""
import os
import re

import httpx

from .contracts import ModelError

# CLIProxy's provider ownership metadata, not guesses based on model names.
OWNERS = {"codex": {"openai", "codex"}, "claude": {"anthropic", "claude"},
          "grok": {"xai", "grok"}, "kimi": {"moonshot", "kimi"}}
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
MEDIA = re.compile(r"image|video|audio|realtime|transcrib|tts|embedding", re.I)


def proxy_models() -> dict[str, list[str]]:
    from .legacy import PROXY_URL_ENV, PROXY_KEY_ENV
    url, key = os.environ.get(PROXY_URL_ENV, "").strip(), os.environ.get(PROXY_KEY_ENV, "")
    if not url or not key:
        raise ModelError("model_catalog_unavailable", "Model discovery requires a configured inference proxy.")
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
                if row.get("owned_by") in owners:
                    grouped[provider].add(name)
        return {provider: sorted(models) for provider, models in grouped.items()}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ModelError("model_catalog_unavailable", "Could not read the inference proxy model catalog.") from None


def validate_model(provider: str, model: str) -> None:
    if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
        raise ModelError("invalid_model", "Invalid model ID.")
    if model not in proxy_models().get(provider, []):
        raise ModelError("model_unavailable", "Selected model is not available for this provider.")
