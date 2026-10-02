"""J. The LanguageModel surface the cutover added under /api/language-model.

Execution behaviour (handoff, resume, cancellation, no replay) is covered against a fake
provider in tests/language_models/. This file drives the same routes through the whole
app — auth gate, lifespan wiring, error mapping — and only along paths that never start
a provider process, so no subscription is consumed.
"""
from __future__ import annotations

import pytest


def _refused(response, status: int, code: str) -> None:
    assert response.status_code == status, f"{response.request.url.path}: {response.status_code} {response.text[:300]}"
    assert response.json()["detail"]["code"] == code


def test_identities_are_listed_without_credential_material(client):
    response = client.get("/api/language-model/identities")
    assert response.status_code == 200
    identities = response.json()["identities"]
    assert isinstance(identities, list)
    for identity in identities:
        assert {"id", "provider"} <= set(identity)
        assert not {"token", "api_key", "password"} & {key.lower() for key in identity}


def test_creating_a_conversation_for_an_unknown_identity_is_refused(client):
    _refused(client.post("/api/language-model/conversations", json={"identity_id": "no-such-identity"}),
             404, "unknown_identity")


def test_unsupported_fields_fail_validation_instead_of_being_ignored(client):
    response = client.post("/api/language-model/conversations",
                           json={"identity_id": "any", "attachments": [{"name": "a.png"}]})
    assert response.status_code == 422


# Literal paths, not f-strings: test_i1 reads string constants to decide coverage.
@pytest.mark.parametrize("method,path,body", [
    ("get", "/api/language-model/conversations/00000000-0000-0000-0000-000000000000", None),
    ("post", "/api/language-model/conversations/00000000-0000-0000-0000-000000000000/handoff", {"identity_id": "any"}),
    ("post", "/api/language-model/conversations/00000000-0000-0000-0000-000000000000/turns", {"prompt": "hi", "request_id": "r1"}),
    ("get", "/api/language-model/turns/00000000-0000-0000-0000-000000000000", None),
    ("get", "/api/language-model/turns/00000000-0000-0000-0000-000000000000/events", None),
    ("post", "/api/language-model/turns/00000000-0000-0000-0000-000000000000/cancel", None),
    ("post", "/api/language-model/turns/00000000-0000-0000-0000-000000000000/reconcile", {"confirmed_stopped": False}),
])
def test_unknown_conversations_and_turns_answer_404_not_500(client, method, path, body):
    response = client.request(method.upper(), path, json=body)
    assert response.status_code in {404, 422}, f"{path}: {response.status_code} {response.text[:300]}"
    if response.status_code == 404:
        assert response.json()["detail"]["code"] in {"not_found", "unknown_identity"}
