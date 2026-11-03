"""Fail-closed owner authentication for HTTP and WebSocket data access.

The default and production mode is ``owner``: only a verified Firebase ID token
for the configured owner is accepted. A static secret cannot bypass this mode.
``token`` explicitly permits the existing automation credential alongside owner
Firebase tokens. ``development`` is an explicit local/test opt-out when no static
secret is configured. No credentials are included in public HTML or auth config.
"""
from __future__ import annotations

import hmac
import os

from . import firebase_auth

PROTECTED_PREFIXES = ("/api/", "/docs", "/redoc", "/openapi.json")
PUBLIC_PATHS = ("/healthz", "/api/auth/config")
TOKEN_HEADER = "X-DevAgents-Token"


def owner_email() -> str:
    """Verified owner mailbox for this installation. Empty until the operator sets it."""
    return os.environ.get("SUBJECTIVE_DEVAGENTS_OWNER_EMAIL", "").strip().lower()


def owner_uid() -> str:
    """Firebase uid of this installation's owner. Empty until the operator sets it."""
    return os.environ.get("SUBJECTIVE_DEVAGENTS_OWNER_UID", "").strip()


def auth_mode() -> str:
    mode = os.environ.get("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "owner").strip().lower()
    # A typo must not silently disable the gate.
    return mode if mode in {"owner", "token", "development"} else "owner"


def configured_token() -> str:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_API_TOKEN") or "").strip()


def auth_enabled() -> bool:
    return auth_mode() != "development" or bool(configured_token())


def public_client_config() -> dict:
    """Browser sign-in settings for this installation. Nothing here is a server secret."""
    origins = [
        item.strip()
        for item in os.environ.get("SUBJECTIVE_DEVAGENTS_PARENT_ORIGINS", "").split(",")
        if item.strip()
    ]
    return {
        "mode": auth_mode(),
        "required": auth_enabled(),
        "ownerEmail": owner_email(),
        "ownerUid": owner_uid(),
        "dashboardUrl": os.environ.get("SUBJECTIVE_DEVAGENTS_DASHBOARD_URL", "").strip(),
        "parentOrigins": origins,
        "firebase": {
            "apiKey": os.environ.get("FIREBASE_API_KEY", "").strip(),
            "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "").strip(),
            "projectId": firebase_auth.firebase_project_id(),
            "appId": os.environ.get("FIREBASE_APP_ID", "").strip(),
        },
    }


def authenticated_claims(supplied: str) -> dict | None:
    mode = auth_mode()
    expected = configured_token()
    if mode == "development" and not expected:
        return {"credential": "development"}
    if not isinstance(supplied, str) or not supplied or len(supplied) > 16384:
        return None
    if mode != "owner" and expected and hmac.compare_digest(supplied.encode(), expected.encode()):
        return {"credential": "static"}
    try:
        claims = firebase_auth.verify_firebase_id_token(supplied)
    except firebase_auth.FirebaseTokenError:
        return None
    expected_uid = owner_uid()
    expected_email = owner_email()
    if not expected_uid and not expected_email:
        return None
    uid_ok = bool(expected_uid) and claims.get("sub") == expected_uid
    email_ok = (bool(expected_email)
                and str(claims.get("email") or "").lower() == expected_email
                and claims.get("email_verified") is True)
    if not (uid_ok or email_ok):
        return None
    return claims


def token_is_valid(supplied: str) -> bool:
    return authenticated_claims(supplied) is not None


# Agents read their company's published prompts here with a read-only token
# (X-Workflow-Token), not the owner's login. Nothing else accepts that token.
WORKFLOW_READ_PREFIX = "/api/workflow/"
WORKFLOW_TOKEN_HEADER = "X-Workflow-Token"


def workflow_read_token() -> str:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_WORKFLOW_READ_TOKEN") or "").strip()


def workflow_token_is_valid(headers) -> bool:
    expected = workflow_read_token()
    supplied = str(headers.get(WORKFLOW_TOKEN_HEADER) or headers.get(WORKFLOW_TOKEN_HEADER.lower()) or "").strip()
    if len(expected) < 32 or not supplied:
        return False
    return hmac.compare_digest(supplied.encode(), expected.encode())


def is_workflow_read_path(path: str) -> bool:
    return path.startswith(WORKFLOW_READ_PREFIX)


def path_requires_token(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return False
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def token_from_request(headers, query_params=None) -> str:
    supplied = headers.get(TOKEN_HEADER) or headers.get(TOKEN_HEADER.lower()) or ""
    if not supplied:
        authorization = headers.get("authorization") or headers.get("Authorization") or ""
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
    # Legacy socket clients may use a query token only in explicitly selected token
    # or development mode. Owner tokens never need to enter URLs/access logs.
    if not supplied and query_params is not None and auth_mode() != "owner":
        supplied = query_params.get("token") or ""
    return str(supplied).strip()
