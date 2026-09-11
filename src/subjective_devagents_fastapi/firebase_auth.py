"""Firebase ID token verification.

Mirrors ``subjective_user_service/auth/firebase.py`` and the same module in
``subjective_pipeline_fastapi``/``subjective_uml2code``/``subjective_semantizer_fastapi``:
RS256 only, audience and issuer pinned to the Firebase project, JWKS cached for an hour
with on-demand refresh on an unknown ``kid``. No service-account credentials are needed —
this only verifies tokens Firebase already signed, it never mints them.
"""

from __future__ import annotations

import os
import threading

import jwt
from jwt import PyJWKClient, PyJWKClientError

# Matches the marketing site's config/firebaseConfig.js and subjective_user_service.
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "subjectivetechnologies-login")
JWKS_URL = (
    "https://www.googleapis.com/service_accounts/v1/jwk/"
    "securetoken@system.gserviceaccount.com"
)
ISSUER = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"
_LEEWAY_SECONDS = 30

_jwk_client = PyJWKClient(JWKS_URL, cache_keys=True, max_cached_keys=16, lifespan=3600)
_jwk_lock = threading.Lock()


class FirebaseTokenError(Exception):
    """Raised when a present token fails verification."""


def _signing_key_for(token: str):
    try:
        return _jwk_client.get_signing_key_from_jwt(token).key
    except PyJWKClientError as exc:
        # Force a JWKS refresh in case the kid is brand new.
        with _jwk_lock:
            _jwk_client.jwk_set_cache.last_updated = 0  # type: ignore[attr-defined]
            try:
                return _jwk_client.get_signing_key_from_jwt(token).key
            except PyJWKClientError as exc2:
                raise FirebaseTokenError("unknown_kid") from exc2


def verify_firebase_id_token(token: str) -> dict:
    """Verify a Firebase ID token and return its decoded payload."""
    if not token or token.count(".") != 2:
        raise FirebaseTokenError("bad_token")
    try:
        payload = jwt.decode(
            token,
            _signing_key_for(token),
            algorithms=["RS256"],
            audience=FIREBASE_PROJECT_ID,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            leeway=_LEEWAY_SECONDS,
        )
    except FirebaseTokenError:
        raise
    except jwt.PyJWTError as exc:
        raise FirebaseTokenError(str(exc) or "bad_token") from exc
    return payload
