"""Shared-secret gate for the API.

This service can read every conversation on the machine, list its OS accounts, and start an
agent CLI turn — which is arbitrary code execution by another name. It is reachable from the
public internet through a Cloudflare tunnel, and internet scanners find it: the logs carry
probes for /mcp, /jsonrpc, /sse and /.well-known/security.txt. Before this module existed
every one of those callers got a 200 from the real API.

The check is a constant-time comparison against SUBJECTIVE_DEVAGENTS_API_TOKEN. Deliberately
not a cookie: the page that calls this API is served from a different origin, and cookie auth
across origins needs SameSite=None plus credentialed CORS, which in turn forbids the wildcard
origin the deployment relies on. A header carries no ambient authority — a scanner that does
not know the token cannot acquire one by being a browser.

The token must NOT be stamped into the served HTML. That page is public, so anything baked
into it is public too; the browser is expected to hold the token in localStorage, seeded once
by the operator (see webBridge.js).

When the variable is unset the gate stays open, so an existing deployment does not lock
itself out on restart — but it says so loudly at startup, every time.
"""
from __future__ import annotations

import hmac
import os

# Paths that must never answer an unauthenticated caller. /docs, /openapi.json and /redoc are
# included on purpose: publishing the full API surface to an anonymous scanner is exactly the
# reconnaissance step to deny.
PROTECTED_PREFIXES = ("/api/", "/docs", "/redoc", "/openapi.json")

# Reachable without a token, so an uptime probe does not need the secret. Returns no data
# about conversations, users, or the machine.
PUBLIC_PATHS = ("/healthz",)

TOKEN_HEADER = "X-DevAgents-Token"


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
    return hmac.compare_digest(supplied, expected)
