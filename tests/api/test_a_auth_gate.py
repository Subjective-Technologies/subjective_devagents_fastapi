"""A. Authentication and the gate.

`require_api_token` is the only thing between the public internet and a service that
reads every conversation on the machine, lists its OS accounts and starts agent turns.
These cases pin the properties that have to hold whatever the handlers do: which paths
are protected, which credentials are accepted in which mode, and that a rejection is
still a well-formed, CORS-carrying 401.
"""
from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from subjective_devagents_fastapi import auth, firebase_auth

from .conftest import TOKEN, build_client
from .helpers import UNAUTHORIZED

# One path per protected prefix, plus the endpoints that actually expose data or act.
PROTECTED_PATHS = [
    "/api/version",
    "/api/config",
    "/api/llm/conversations",
    "/api/llm/conversation?target=x",
    "/api/system/users",
    "/api/datasources",
    "/api/ssh/connections",
    "/api/ssh/sessions",
    "/api/agents",
    "/docs",
    "/redoc",
    "/openapi.json",
]


@pytest.fixture()
def signer(monkeypatch):
    """Issue real RS256 tokens the app verifies for real, against a key we hold."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(firebase_auth, "_signing_key_for", lambda token: private.public_key())

    def issue(**overrides) -> str:
        now = int(time.time())
        claims = {
            "sub": auth.OWNER_UID,
            "email": auth.OWNER_EMAIL,
            "email_verified": True,
            "aud": firebase_auth.FIREBASE_PROJECT_ID,
            "iss": firebase_auth.ISSUER,
            "iat": now,
            "auth_time": now,
            "exp": now + 3600,
        }
        claims.update(overrides)
        return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test-key"})

    return issue


# --- A1/A2: which paths are gated ------------------------------------------------------


@pytest.mark.parametrize("path", PROTECTED_PATHS)
def test_a1_protected_paths_refuse_an_anonymous_caller(bare_token_client, path):
    response = bare_token_client.get(path)
    assert response.status_code == 401, f"{path} answered without a credential"
    assert response.json() == UNAUTHORIZED
    assert response.headers.get("WWW-Authenticate") == "Bearer", (
        f"{path} refused without telling the caller how to authenticate"
    )


def test_a1_every_protected_prefix_is_covered_by_a_case():
    """The paths above are a sample; this keeps the sample honest if a prefix is added."""
    for prefix in auth.PROTECTED_PREFIXES:
        assert any(path.startswith(prefix) for path in PROTECTED_PATHS), (
            f"{prefix} is protected but no case in this file exercises it"
        )


@pytest.mark.parametrize("path", list(auth.PUBLIC_PATHS))
def test_a2_public_paths_answer_without_a_credential(bare_token_client, path):
    response = bare_token_client.get(path)
    assert response.status_code == 200, f"{path} is public but refused an anonymous caller"


def test_a2_the_auth_config_endpoint_never_leaks_the_token(bare_token_client):
    body = bare_token_client.get("/api/auth/config").json()
    assert body == {"mode": "token", "required": True}
    assert TOKEN not in bare_token_client.get("/api/auth/config").text


# --- A3/A4: accepted header credentials ------------------------------------------------


def test_a3_the_token_header_is_accepted(bare_token_client):
    response = bare_token_client.get("/api/version", headers={"X-DevAgents-Token": TOKEN})
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_a4_a_bearer_authorization_header_is_accepted(bare_token_client):
    response = bare_token_client.get("/api/version", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_a4_bearer_is_matched_case_insensitively(bare_token_client):
    response = bare_token_client.get("/api/version", headers={"Authorization": f"bearer {TOKEN}"})
    assert response.status_code == 200


# --- A5/A6: a token in the URL ---------------------------------------------------------


def test_a5_a_query_token_is_rejected_in_owner_mode(owner_client, signer):
    """An owner token in a URL lands in every access log between here and the browser."""
    assert owner_client.get("/api/config", params={"token": signer()}).status_code == 401
    assert owner_client.get("/api/config", params={"token": TOKEN}).status_code == 401


def test_a6_a_query_token_is_rejected_over_http_in_token_mode(bare_token_client):
    """`token_from_request` is called with headers only from the HTTP middleware.

    The query fallback exists for legacy socket clients; it must not become an HTTP
    credential just because the mode permits it for the socket.
    """
    assert bare_token_client.get("/api/config", params={"token": TOKEN}).status_code == 401


def test_a6_the_http_middleware_passes_no_query_params(monkeypatch, tmp_path):
    """The same property at the source, so a refactor cannot reintroduce it."""
    seen = {}
    real = auth.token_from_request

    def spy(headers, query_params=None):
        seen["query_params"] = query_params
        return real(headers, query_params)

    monkeypatch.setattr(auth, "token_from_request", spy)
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="token",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
    ) as client:
        client.get("/api/config", params={"token": TOKEN})
    assert seen.get("query_params") is None, "the HTTP gate read the query string"


# --- A7: bad credentials ---------------------------------------------------------------


@pytest.mark.parametrize(
    "credential",
    [
        pytest.param("wrong-token-entirely", id="wrong"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("\t\n", id="whitespace-control"),
        # Right shape, wrong value: a comparison that only checked length would pass.
        pytest.param("x" * len(TOKEN), id="right-length-wrong-value"),
        pytest.param(TOKEN[:-1], id="truncated"),
        pytest.param(TOKEN + "x", id="extended"),
        pytest.param(TOKEN.upper(), id="wrong-case"),
        pytest.param(f" {TOKEN} ", id="padded"),
    ],
)
def test_a7_wrong_credentials_are_refused(bare_token_client, credential):
    header = bare_token_client.get("/api/config", headers={"X-DevAgents-Token": credential})
    bearer = bare_token_client.get("/api/config", headers={"Authorization": f"Bearer {credential}"})
    # A padded token is stripped before comparison and is the one legitimate exception.
    expected = 200 if credential.strip() == TOKEN else 401
    assert header.status_code == expected, f"{credential!r} via header: {header.status_code}"
    assert bearer.status_code == expected, f"{credential!r} via bearer: {bearer.status_code}"


def test_a7_an_absurdly_long_credential_is_refused_without_verification(bare_token_client):
    """16KB is the cap; beyond it the token is rejected before any signature work."""
    response = bare_token_client.get(
        "/api/config", headers={"X-DevAgents-Token": "x" * 16385}
    )
    assert response.status_code == 401


# --- A8: a non-owner identity ----------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"sub": "someone-else", "email": "someone@example.com"}, id="another-account"),
        pytest.param(
            {"sub": "someone-else", "email": auth.OWNER_EMAIL, "email_verified": False},
            id="owner-email-unverified",
        ),
        pytest.param({"exp": int(time.time()) - 60}, id="expired"),
        pytest.param({"aud": "some-other-project"}, id="wrong-audience"),
        pytest.param({"iss": "https://securetoken.google.com/other"}, id="wrong-issuer"),
    ],
)
def test_a8_a_non_owner_firebase_token_is_refused(owner_client, signer, changes):
    response = owner_client.get("/api/config", headers={"Authorization": f"Bearer {signer(**changes)}"})
    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED


def test_a8_the_owner_is_accepted(owner_client, signer):
    response = owner_client.get("/api/config", headers={"Authorization": f"Bearer {signer()}"})
    assert response.status_code == 200


def test_a8_a_static_token_cannot_bypass_owner_mode(owner_client):
    """`SUBJECTIVE_DEVAGENTS_API_TOKEN` is set on this fixture and must not be enough."""
    assert owner_client.get("/api/config", headers={"X-DevAgents-Token": TOKEN}).status_code == 401


# --- A9: a mistyped mode ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ownr", "OWNER ", "", "open", "none", "disabled", "developement"])
def test_a9_a_mistyped_auth_mode_falls_back_to_owner(monkeypatch, tmp_path, mode):
    """A typo in the deployment's environment must close the gate, never open it."""
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE=mode,
        SUBJECTIVE_DEVAGENTS_API_TOKEN=None,
    ) as client:
        assert client.get("/api/config").status_code == 401, f"{mode!r} opened the gate"
        assert client.get("/api/auth/config").json()["required"] is True
        # `OWNER ` normalises to owner by case and strip; the rest fall back to it.
        assert client.get("/api/auth/config").json()["mode"] == "owner"


def test_a9_development_mode_is_the_only_open_one(monkeypatch, tmp_path):
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="development",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=None,
    ) as client:
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/auth/config").json() == {"mode": "development", "required": False}


def test_a9_development_mode_with_a_token_still_requires_it(monkeypatch, tmp_path):
    """Configuring a token is an explicit decision and outranks the loose mode."""
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_AUTH_MODE="development",
        SUBJECTIVE_DEVAGENTS_API_TOKEN=TOKEN,
    ) as client:
        assert client.get("/api/config").status_code == 401
        assert client.get("/api/config", headers={"X-DevAgents-Token": TOKEN}).status_code == 200


# --- A10: CORS -------------------------------------------------------------------------


ORIGIN = "https://dashboard.subjectivetechnologies.com"


def test_a10_preflight_succeeds_without_a_credential(bare_token_client):
    """A browser never sends credentials on a preflight, so gating it breaks every call."""
    response = bare_token_client.options(
        "/api/llm/send-message",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200, "the preflight was gated"
    assert response.headers.get("access-control-allow-origin")
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_a10_a_401_still_carries_cors_headers(bare_token_client):
    """Without these the browser reports a CORS error and the real 401 never surfaces."""
    response = bare_token_client.get("/api/config", headers={"Origin": ORIGIN})
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin"), (
        "a rejected cross-origin call had no CORS headers, so the browser cannot read the 401"
    )
