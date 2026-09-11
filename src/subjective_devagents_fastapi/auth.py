"""Auth gate for the API: a shared secret, or a Firebase sign-in as the one owner account.

This service can read every conversation on the machine, list its OS accounts, and start an
agent CLI turn — which is arbitrary code execution by another name. It is reachable from the
public internet through a Cloudflare tunnel, and internet scanners find it: the logs carry
probes for /mcp, /jsonrpc, /sse and /.well-known/security.txt. Before this module existed
every one of those callers got a 200 from the real API.

Two credentials are accepted, either is enough:

1. A constant-time comparison against SUBJECTIVE_DEVAGENTS_API_TOKEN. Deliberately not a
   cookie: the page that calls this API is served from a different origin, and cookie auth
   across origins needs SameSite=None plus credentialed CORS, which in turn forbids the
   wildcard origin the deployment relies on. A header carries no ambient authority — a
   scanner that does not know the token cannot acquire one by being a browser.
2. A Firebase ID token (see firebase_auth.py) whose uid/email match OWNER_UID/OWNER_EMAIL —
   the same allowlist check as clients/web/subjective_web_dashboard's isOwnerToken(). Signing
   in as anyone else verifies fine as a real token but is still refused: a valid signature
   only proves Firebase issued it, not that it belongs to this deployment's operator.

Neither credential is ever stamped into the served HTML. That page is public, so anything
baked into it is public too; the browser is expected to hold the static token in
localStorage (seeded once by the operator, see webBridge.js) or send a Firebase ID token as
an Authorization: Bearer header.

When SUBJECTIVE_DEVAGENTS_API_TOKEN is unset the gate stays fully open regardless of what a
caller sends (including a bogus Firebase token) — an existing deployment must not lock itself
out on restart — but it says so loudly at startup, every time.
"""
from __future__ import annotations

import hmac
import os

from . import firebase_auth

# Paths that must never answer an unauthenticated caller. /docs, /openapi.json and /redoc are
# included on purpose: publishing the full API surface to an anonymous scanner is exactly the
# reconnaissance step to deny.
PROTECTED_PREFIXES = ("/api/", "/docs", "/redoc", "/openapi.json")

# Reachable without a token, so an uptime probe does not need the secret. Returns no data
# about conversations, users, or the machine.
PUBLIC_PATHS = ("/healthz",)

TOKEN_HEADER = "X-DevAgents-Token"

# The one operator this deployment is for. Signing in with any other Google/Firebase account
# verifies fine (it is a real Firebase ID token) but is not this person, so it is still
# refused — a valid signature only proves Firebase issued it, not that it should open this
# door. Matches clients/web/subjective_web_dashboard's isOwnerToken()/OWNER_EMAIL/OWNER_UID.
OWNER_EMAIL = "subjectivetechnologies@gmail.com"
OWNER_UID = "WO55hP1UlbRziVhokDMJyQoqVyl2"


def _is_owner_firebase_token(supplied: str) -> bool:
    """A Firebase ID token for OWNER_UID/OWNER_EMAIL is accepted alongside the static secret.

    Any other well-formed string (including a wrong static-token guess) is simply not a JWT
    and fails fast in verify_firebase_id_token's own shape check — this never widens what an
    anonymous scanner can do, it only adds a second way for the one real operator to get in.
    """
    try:
        claims = firebase_auth.verify_firebase_id_token(supplied)
    except firebase_auth.FirebaseTokenError:
        return False
    return (
        str(claims.get("sub") or "") == OWNER_UID
        and str(claims.get("email") or "").lower() == OWNER_EMAIL.lower()
    )


def configured_token() -> str:
    return (os.environ.get("SUBJECTIVE_DEVAGENTS_API_TOKEN") or "").strip()


def auth_enabled() -> bool:
    return bool(configured_token())


def path_requires_token(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return False
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def token_from_request(headers, query_params=None) -> str:
    """Read the token from the header, an Authorization: Bearer, or a query parameter.

    The query parameter exists for the WebSocket: a browser cannot set headers on a
    WebSocket handshake, so there is nowhere else to put it.
    """
    supplied = headers.get(TOKEN_HEADER) or headers.get(TOKEN_HEADER.lower()) or ""
    if not supplied:
        authorization = headers.get("authorization") or headers.get("Authorization") or ""
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
    if not supplied and query_params is not None:
        supplied = query_params.get("token") or ""
    return str(supplied).strip()


def token_is_valid(supplied: str) -> bool:
    expected = configured_token()
    if not expected:
        return True  # gate disabled
    # compare_digest keeps the comparison constant-time; a plain == leaks the token's prefix
    # to anyone willing to measure enough requests.
    if hmac.compare_digest(supplied, expected):
        return True
    return _is_owner_firebase_token(supplied)
