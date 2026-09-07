"""HTTPS client for the `devagents` Cloudflare D1 database.

D1 has no SQL wire protocol. This process runs on Hetzner, so the Worker binding
in wrangler.toml is unused — every statement is POST /query with bound params.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Protocol

CF_API = "https://api.cloudflare.com/client/v4"
DEFAULT_ACCOUNT_ID = "5d3bde66e7070ff31dd195de1c8d04cf"
DEFAULT_DATABASE_ID = "2348859b-1168-4ba9-81ba-044442898904"
DEFAULT_DATABASE_NAME = "devagents"
USER_EMAIL = "subjectivetechnologies@gmail.com"


class SqlExecutor(Protocol):
    def query(self, sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]: ...
    def execute(self, sql: str, params: Iterable[Any] | None = None) -> None: ...


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_cloudflare_env() -> dict[str, str]:
    """Process env wins; then /subjective/.env (or the first ancestor .env)."""
    merged: dict[str, str] = {}
    here = Path(__file__).resolve()
    for parent in [Path("/subjective/.env"), *list(here.parents)]:
        candidate = parent if parent.name == ".env" else parent / ".env"
        if candidate.is_file():
            merged.update(_parse_env_file(candidate))
            break
    merged.update({key: value for key, value in os.environ.items() if value})
    return merged


def d1_sync_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or load_cloudflare_env()
    flag = str(env.get("SUBJECTIVE_DEVAGENTS_D1_SYNC") or "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(d1_auth_headers(env))


def d1_auth_headers(env: dict[str, str]) -> dict[str, str]:
    # Global API key is what can actually query D1 on this account. Scoped
    # deploy tokens often 401/403 against /d1/database.
    api_key = env.get("CLOUDFLARE_API_KEY_1") or env.get("CLOUDFLARE_API_KEY") or ""
    if api_key:
        return {"X-Auth-Email": env.get("CLOUDFLARE_EMAIL") or USER_EMAIL, "X-Auth-Key": api_key}
    for key in (
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_API_TOKEN_5",
        "CLOUDFLARE_API_TOKEN_2",
        "CLOUDFLARE_API_TOKEN_4",
        "CLOUDFLARE_API_TOKEN_3",
    ):
        token = env.get(key) or ""
        if token:
            return {"Authorization": f"Bearer {token}"}
    return {}


class D1Error(RuntimeError):
    pass


class D1Client:
    def __init__(self, env: dict[str, str] | None = None):
        self.env = env or load_cloudflare_env()
        self.account_id = self.env.get("CLOUDFLARE_ACCOUNT_ID") or DEFAULT_ACCOUNT_ID
        self.database_id = (
            self.env.get("SUBJECTIVE_DEVAGENTS_D1_DATABASE_ID")
            or self.env.get("DEVAGENTS_D1_DATABASE_ID")
            or DEFAULT_DATABASE_ID
        )
        self.database_name = self.env.get("SUBJECTIVE_DEVAGENTS_D1_DATABASE_NAME") or DEFAULT_DATABASE_NAME
        self._headers = {"Content-Type": "application/json", **d1_auth_headers(self.env)}
        if "Authorization" not in self._headers and "X-Auth-Key" not in self._headers:
            raise D1Error("No Cloudflare token/key available for D1.")

    def _url(self) -> str:
        return f"{CF_API}/accounts/{self.account_id}/d1/database/{self.database_id}/query"

    def _post(self, sql: str, params: Iterable[Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"sql": sql}
        if params is not None:
            body["params"] = [param for param in params]
        request = urllib.request.Request(
            self._url(),
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers=self._headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise D1Error(f"D1 HTTP {error.code}: {detail}") from error
        if not payload.get("success"):
            raise D1Error(f"D1 error: {payload.get('errors')}")
        return payload

    def query(self, sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
        payload = self._post(sql, params)
        rows: list[dict[str, Any]] = []
        for block in payload.get("result") or []:
            rows.extend(block.get("results") or [])
        return rows

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> None:
        self._post(sql, params)


class SqliteExecutor:
    """Same surface as D1Client, for tests against schema/devagents.sql."""

    def __init__(self, connection):
        self.connection = connection
        self.connection.row_factory = __import__("sqlite3").Row

    def query(self, sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self.connection.execute(sql, tuple(params or ()))
        return [dict(row) for row in cursor.fetchall()]

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> None:
        self.connection.execute(sql, tuple(params or ()))
        self.connection.commit()
