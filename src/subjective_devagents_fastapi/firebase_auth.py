"""Firebase ID token verification.

Mirrors ``subjective_user_service/auth/firebase.py`` and the same module in
``subjective_pipeline_fastapi``/``subjective_uml2code``/``subjective_semantizer_fastapi``:
RS256 only, audience and issuer pinned to the Firebase project, JWKS cached for an hour
with on-demand refresh on an unknown ``kid``. No service-account credentials are needed —
this only verifies tokens Firebase already signed, it never mints them.
"""

from __future__ import annotations

import os
import time

import jwt
from jwt import PyJWKClient, PyJWKClientError

# Google's public Firebase token signing keys. Not a customer service account.
JWKS_URL = (
    "https://www.googleapis.com/service_accounts/v1/jwk/"
    "securetoken@system.gserviceaccount.com"
)
_LEEWAY_SECONDS = 30


def firebase_project_id() -> str:
    """This installation's Firebase project. Empty until the operator sets it."""
    return os.environ.get("FIREBASE_PROJECT_ID", "").strip()


def issuer() -> str:
    project = firebase_project_id()
    return f"https://securetoken.google.com/{project}" if project else ""

_jwk_client = PyJWKClient(JWKS_URL, cache_keys=True, max_cached_keys=16, lifespan=3600, timeout=5)


class FirebaseTokenError(Exception):
    """Raised when a present token fails verification."""


def _signing_key_for(token: str):
    try:
        return _jwk_client.get_signing_key_from_jwt(token).key
    except PyJWKClientError as exc:
        # PyJWKClient already refreshes once for an unknown kid. Do not mutate
        # private cache state or repeat unbounded network retries.
        raise FirebaseTokenError("signing_key_unavailable") from exc


def verify_firebase_id_token(token: str) -> dict:
    """Verify a Firebase ID token and return its decoded payload."""
    if not token or len(token) > 16384 or token.count(".") != 2:
        raise FirebaseTokenError("bad_token")
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str) or not 0 < len(header["kid"]) <= 256:
            raise FirebaseTokenError("bad_header")
        project = firebase_project_id()
        expected_issuer = issuer()
        if not project or not expected_issuer:
            raise FirebaseTokenError("project_unconfigured")
        payload = jwt.decode(
            token,
            _signing_key_for(token),
            algorithms=["RS256"],
            audience=project,
            issuer=expected_issuer,
            options={"require": ["exp", "iat", "iss", "aud", "sub", "auth_time"]},
            leeway=_LEEWAY_SECONDS,
        )
    except FirebaseTokenError:
        raise
    except jwt.PyJWTError as exc:
        raise FirebaseTokenError(str(exc) or "bad_token") from exc
    if not isinstance(payload["sub"], str) or not 0 < len(payload["sub"]) <= 128:
        raise FirebaseTokenError("bad_subject")
    auth_time = payload["auth_time"]
    if isinstance(auth_time, bool) or not isinstance(auth_time, (int, float)) or not 0 <= auth_time <= time.time() + _LEEWAY_SECONDS:
        raise FirebaseTokenError("bad_auth_time")
    return payload
