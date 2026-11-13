"""Fail-closed authentication for HTTP and WebSocket data access.

Firebase callers authenticate the way CLIProxy does: a user id, a Firebase ID
token, and the company name. The token must verify, its subject must be that
user id, and the company must be this server's company. The default production
mode is ``owner``: any identity that passes that check is accepted. The virtual
machine is the account boundary, so a new VM does not need an owner allowlist.
A static secret cannot bypass this mode. ``token`` explicitly permits the
existing automation credential alongside that Firebase identity. ``development``
is an explicit local/test opt-out when no static secret is configured. No
credentials are included in public HTML or auth config.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from . import firebase_auth

PROTECTED_PREFIXES = ("/api/", "/docs", "/redoc", "/openapi.json")
PUBLIC_PATHS = ("/healthz", "/api/auth/config")
TOKEN_HEADER = "X-DevAgents-Token"
USER_ID_HEADER = "X-User-Id"
COMPANY_HEADER = "X-Company-Name"
FIREBASE_TOKEN_HEADER = "X-Firebase-Token"
COMPANY_NAME_FILE = Path("/company_name")


def owner_email() -> str:
    """Optional address published to the browser as a sign-in hint. Not an allowlist."""
    return os.environ.get("SUBJECTIVE_DEVAGENTS_OWNER_EMAIL", "").strip().lower()


def owner_uid() -> str:
    """Optional uid published to the browser as a sign-in hint. Not an allowlist."""
    return os.environ.get("SUBJECTIVE_DEVAGENTS_OWNER_UID", "").strip()


def auth_mode() -> str:
    mode = os.environ.get("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "owner").strip().lower()
    # A typo must not silently disable the gate.
    return mode if mode in {"owner", "token", "development"} else "owner"


def configured_token() -> str:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_API_TOKEN") or "").strip()


def auth_enabled() -> bool:
    return auth_mode() != "development" or bool(configured_token())


def expected_company_name() -> str:
    """This server's company. Env wins; otherwise the same file CLIProxy reads."""
    name = os.environ.get("SUBJECTIVE_DEVAGENTS_COMPANY_NAME", "").strip()
    if name:
        return name
    try:
        return COMPANY_NAME_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def public_client_config() -> dict:
    """Browser sign-in settings for this installation. Nothing here is a server secret.

    ``companyName`` is the value the sign-in form must send back. It identifies
    which server this is. It does not authorize anyone by itself.
    """
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
        "companyName": expected_company_name(),
        "dashboardUrl": os.environ.get("SUBJECTIVE_DEVAGENTS_DASHBOARD_URL", "").strip(),
        "parentOrigins": origins,
        "firebase": {
            "apiKey": os.environ.get("FIREBASE_API_KEY", "").strip(),
            "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "").strip(),
            "projectId": firebase_auth.firebase_project_id(),
            "appId": os.environ.get("FIREBASE_APP_ID", "").strip(),
        },
    }


class AuthFailure(Exception):
    """A presented credential was rejected. ``incomplete`` means a socket may still send a first frame."""

    def __init__(self, status: int, message: str, *, incomplete: bool = False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.incomplete = incomplete


def _header(headers, name: str) -> str:
    if headers is None:
        return ""
    return str(headers.get(name) or headers.get(name.lower()) or "").strip()


def _frame_text(frame: dict | None, key: str) -> str:
    if not isinstance(frame, dict):
        return ""
    value = frame.get(key)
    return value.strip() if isinstance(value, str) else ""


def _firebase_material(headers, frame: dict | None) -> tuple[str, str, str]:
    """User id, company, and Firebase token. A first-frame value fills a missing header."""
    user_id = _frame_text(frame, "user_id") or _header(headers, USER_ID_HEADER)
    company = _frame_text(frame, "company_name") or _header(headers, COMPANY_HEADER)
    token = _frame_text(frame, "token") or _header(headers, FIREBASE_TOKEN_HEADER)
    if not token:
        authorization = _header(headers, "Authorization")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
    return user_id, company, token


def verify_firebase_identity(user_id: str, company: str, token: str) -> dict:
    """The CLIProxy check: all three present, company matches, token subject is the user id."""
    user_id = user_id.strip()
    company = company.strip()
    token = token.strip()
    if not user_id or not company or not token:
        raise AuthFailure(401, "user id, firebase token, and company name are required", incomplete=True)
    if len(user_id) > 128 or len(company) > 128 or len(token) > 16384:
        raise AuthFailure(401, "invalid firebase token")
    expected = expected_company_name()
    if not expected:
        raise AuthFailure(403, "company is not configured")
    if company.casefold() != expected.casefold():
        raise AuthFailure(403, "company name does not match this server")
    try:
        claims = firebase_auth.verify_firebase_id_token(token)
    except firebase_auth.FirebaseTokenError:
        raise AuthFailure(401, "invalid firebase token") from None
    if claims.get("sub") != user_id:
        raise AuthFailure(401, "invalid firebase token")
    return claims


def authenticate_headers(headers, query_params=None, frame: dict | None = None) -> dict:
    """Claims for this call, or ``AuthFailure``.

    ``token`` mode still accepts the static automation secret alone. A Firebase
    caller always needs the user id, the token, and the company name. The token
    is never read from the query string.
    """
    mode = auth_mode()
    expected = configured_token()
    if mode == "development" and not expected:
        return {"credential": "development"}

    static = token_from_request(headers, query_params) or _frame_text(frame, "token")
    if mode != "owner" and expected and static and hmac.compare_digest(static.encode(), expected.encode()):
        return {"credential": "static"}

    user_id, company, token = _firebase_material(headers, frame)
    # A wrong automation secret, with no Firebase identity beside it, stays a plain refusal.
    if mode != "owner" and static and not user_id and not company:
        raise AuthFailure(401, "Unauthorized.")
    if not user_id and not company and not token:
        raise AuthFailure(401, "Unauthorized.", incomplete=True)
    return verify_firebase_identity(user_id, company, token)


def token_is_valid(supplied: str) -> bool:
    """True when this string alone opens the gate.

    An open development install does. A configured automation secret does, outside
    owner mode. A Firebase token does not: it also needs the user id and company.
    """
    mode = auth_mode()
    expected = configured_token()
    if mode == "development" and not expected:
        return True
    if not isinstance(supplied, str) or not supplied or len(supplied) > 16384:
        return False
    return bool(mode != "owner" and expected and hmac.compare_digest(supplied.encode(), expected.encode()))


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
