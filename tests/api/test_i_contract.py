"""I. Contract — properties that hold across the whole surface.

I1 is the one that keeps the rest of the suite honest: it reads the app's own OpenAPI
schema, collects every path this suite actually calls, and fails naming any operation
nothing exercises. A route added without a test fails this file rather than quietly
joining the 18 that had no coverage at all.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from .conftest import build_client
from .helpers import envelope

SUITE_DIR = Path(__file__).parent

# Paths exercised by the WebSocket cases rather than by an HTTP call, and the catch-all
# static route, which the spec lists as an operation but which serves files.
NOT_HTTP_OPERATIONS = {
    ("GET", "/{path}"),
}


def spec_operations(client) -> set[tuple[str, str]]:
    schema = client.app.openapi()
    return {
        (method.upper(), path)
        for path, item in schema["paths"].items()
        for method in item
        if method in {"get", "post", "put", "patch", "delete"}
    }


def paths_called_by_this_suite() -> set[str]:
    """Every literal request path in this directory, read out of the source.

    Static analysis rather than a runtime recorder: a recorder only sees the tests that
    ran, so a skipped live test would silently stop counting as coverage of its route.
    """
    called: set[str] = set()
    literal = re.compile(r"""["'](/(?:api|ssh|healthz|docs|redoc|openapi|ws)[^"'{}]*)["']""")
    for source_file in sorted(SUITE_DIR.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                match = literal.fullmatch(f'"{node.value}"')
                if match:
                    called.add(match.group(1).split("?")[0])
    return called


# --- I1: every operation is exercised ---------------------------------------------------


def test_i1_every_operation_in_the_spec_is_exercised_by_this_suite(client):
    """The mechanical guard the assignment asks for.

    A new route with no test in `tests/api/` fails here by name.
    """
    called = paths_called_by_this_suite()
    # "/" and the static catch-all are reached by path, not by a literal in a call.
    called.update({"/", "/{path}"})

    missing = sorted(
        f"{method} {path}"
        for method, path in spec_operations(client)
        if (method, path) not in NOT_HTTP_OPERATIONS
        and path not in called
        and not _matches_templated(path, called)
    )
    assert not missing, (
        "these operations are in openapi.json and no test in tests/api/ calls them:\n  "
        + "\n  ".join(missing)
    )


def _matches_templated(path: str, called: set[str]) -> bool:
    """`/ssh/assets/{name}` is satisfied by a call to `/ssh/assets/anything`."""
    if "{" not in path:
        return False
    pattern = re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", re.escape(path).replace(r"\{", "{").replace(r"\}", "}")) + "$")
    return any(pattern.match(candidate) for candidate in called)


def test_i1_the_committed_spec_still_matches_the_app(client):
    """The artifact the assignment was written against is the app's own schema.

    If they have drifted, the per-handler coverage numbers in the assignment describe a
    different service and the gap analysis needs redoing.
    """
    committed = SUITE_DIR.parents[3] / "artifacts" / "devagents-api-docs" / "openapi.json"
    if not committed.is_file():
        pytest.skip(f"no committed spec at {committed}")

    import json

    recorded = {
        (method.upper(), path)
        for path, item in json.loads(committed.read_text(encoding="utf-8"))["paths"].items()
        for method in item
        if method in {"get", "post", "put", "patch", "delete"}
    }
    live = spec_operations(client)
    assert live == recorded, (
        f"added since the spec was captured: {sorted(live - recorded)}; "
        f"gone: {sorted(recorded - live)}"
    )


def test_i1_the_docs_endpoints_are_behind_the_gate(bare_token_client):
    """There is no authenticated URL for them: the middleware reads headers only."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert bare_token_client.get(path).status_code == 401, path
        assert bare_token_client.get(path, params={"token": "anything"}).status_code == 401, path


# --- I2: version headers ------------------------------------------------------------------


VERSION_PROBES = [
    "/healthz",
    "/api/auth/config",
    "/api/version",
    "/api/config",
    "/api/llm/conversations",
    "/api/ssh/connections",
    "/api/datasources",
    "/api/agents",
    "/api/d1/sync",
    "/api/exec/status",
    "/api/system/users",
    "/api/ssh/sessions",
]


@pytest.mark.parametrize("path", VERSION_PROBES)
def test_i2_every_response_carries_both_version_headers(client, path):
    response = client.get(path)
    assert response.headers.get("X-DevAgents-Version"), f"{path} has no X-DevAgents-Version"
    assert response.headers.get("X-DevAgents-Backend-Version"), f"{path} has no X-DevAgents-Backend-Version"


def test_i2_the_headers_are_on_refusals_too(bare_token_client):
    """A 401 is the response most likely to be read by a person debugging a deployment."""
    response = bare_token_client.get("/api/config")
    assert response.status_code == 401
    assert response.headers.get("X-DevAgents-Version")
    assert response.headers.get("X-DevAgents-Backend-Version")


def test_i2_the_headers_are_on_a_404_too(client):
    response = client.get("/definitely-not-a-route")
    assert response.status_code == 404
    assert response.headers.get("X-DevAgents-Version")
    assert response.headers.get("X-DevAgents-Backend-Version")


def test_i2_the_ui_and_backend_versions_are_reported_separately(client):
    """The UI can live in another repository; its SHA cannot identify this API."""
    body = envelope(client.get("/api/version"), ok=True)
    assert body["version"] == client.get("/healthz").headers["X-DevAgents-Version"]
    assert body["backendVersion"] == client.get("/healthz").headers["X-DevAgents-Backend-Version"]
    assert body["version"] != body["backendVersion"], "the fixture UI root has no Git checkout"


# --- I3: no-store on the API ------------------------------------------------------------------


@pytest.mark.parametrize("path", [p for p in VERSION_PROBES if p.startswith("/api/")])
def test_i3_every_api_response_is_uncacheable(client, path):
    """A cached GET /api/llm/conversations shows a chat under its pre-rename name."""
    assert client.get(path).headers.get("Cache-Control") == "no-store", path


def test_i3_a_refused_api_call_is_uncacheable_too(bare_token_client):
    response = bare_token_client.get("/api/llm/conversations")
    assert response.status_code == 401
    assert response.headers.get("Cache-Control") == "no-store"


def test_i3_a_post_is_uncacheable_too(client):
    assert client.post("/api/d1/sync").headers.get("Cache-Control") == "no-store"


def test_i3_a_rename_is_immediately_visible_through_the_list(client, chats):
    """The behaviour the header exists for, asserted end to end."""
    conversation = chats(provider="claude", name="before-the-rename")
    envelope(client.post("/api/llm/rename-conversation", json={
        "target": conversation["target"], "name": "after-the-rename",
    }), ok=True)

    listing = client.get("/api/llm/conversations")
    assert listing.headers.get("Cache-Control") == "no-store"
    names = [item["label"] for item in envelope(listing, ok=True)["conversations"]]
    assert "after-the-rename" in names
    assert "before-the-rename" not in names


def test_i3_non_api_responses_are_not_forced_uncacheable(client):
    """Only `/api/` is stamped; the static shell may be cached normally."""
    assert client.get("/healthz").headers.get("Cache-Control") is None


# --- I4: the 404 envelope ------------------------------------------------------------------------


def test_i4_the_404_envelope_carries_docs_and_web(monkeypatch, tmp_path):
    """On the API host the UI is not served, and a 404 has to point at both."""
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_SERVE_UI="0",
        SUBJECTIVE_DEVAGENTS_WEB_URL="https://devagents.subjectivetechnologies.com/",
    ) as client:
        response = client.get("/no-such-page")
        assert response.status_code == 404
        body = response.json()
        assert body["ok"] is False
        assert body["docs"] == "/docs"
        assert body["web"] == "https://devagents.subjectivetechnologies.com/"


def test_i4_the_api_index_carries_the_same_pointers(monkeypatch, tmp_path):
    with build_client(
        monkeypatch, tmp_path,
        SUBJECTIVE_DEVAGENTS_SERVE_UI="0",
        SUBJECTIVE_DEVAGENTS_WEB_URL="https://devagents.subjectivetechnologies.com/",
    ) as client:
        body = envelope(client.get("/"), ok=True)
        assert body["docs"] == "/docs"
        assert body["openapi"] == "/openapi.json"
        assert body["web"] == "https://devagents.subjectivetechnologies.com/"
        assert body["service"] == "subjective_devagents_fastapi"


def test_i4_an_unknown_api_path_is_a_404_not_a_200(client):
    """The catch-all serves the UI shell; it must not answer for a mistyped API path."""
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404, response.text


def test_i4_a_404_is_still_gated_under_the_api_prefix(bare_token_client):
    """A scanner must not learn which `/api/` paths exist by comparing 401 against 404."""
    assert bare_token_client.get("/api/does-not-exist").status_code == 401
    assert bare_token_client.get("/api/llm/conversations").status_code == 401


# --- the envelope shape across the surface ------------------------------------------------------------


GET_ENDPOINTS = [
    "/api/version",
    "/api/auth/session",
    "/api/llm/conversations",
    "/api/llm/providers",
    "/api/llm/sessions",
    "/api/datasources",
    "/api/system/users",
    "/api/switchuser/status",
    "/api/ssh/connections",
    "/api/agents",
    "/api/d1/sync",
    "/api/exec/status",
    "/api/context-menu?target=",
]


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_every_get_answers_a_json_object_with_an_ok_flag(client, path):
    """The UI branches on `ok` everywhere; an endpoint without it is a client crash."""
    response = client.get(path)
    assert response.status_code == 200, f"{path}: {response.text[:200]}"
    body = response.json()
    assert isinstance(body, dict), f"{path} answered {type(body).__name__}, not an object"
    assert "ok" in body, f"{path} answered without an `ok` flag: {sorted(body)[:8]}"


def _raising_config_client(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path, raise_server_exceptions=False)
    from subjective_devagents_fastapi.app import state

    def explode():
        raise RuntimeError("the config store is unreadable")

    client.__enter__()
    monkeypatch.setattr(state.core, "public_config", explode)
    return client


def test_a_handler_that_raises_becomes_a_400_envelope_not_a_500(monkeypatch, tmp_path):
    """The app's catch-all: the browser gets `{ok: false, error}` rather than a 500 page."""
    client = _raising_config_client(monkeypatch, tmp_path)
    try:
        response = client.get("/api/config")
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["ok"] is False
        assert "unreadable" in body["error"]
    finally:
        client.__exit__(None, None, None)


def test_known_defect_an_error_envelope_loses_its_version_and_no_store_headers(monkeypatch, tmp_path):
    """KNOWN DEFECT — expected red until the fix lands. Found by this suite.

    `@app.exception_handler(Exception)` runs inside Starlette's `ServerErrorMiddleware`,
    which is installed *outside* both of this app's own middlewares. So when a handler
    raises, `add_version_and_no_store` never sees the response: the 400 envelope goes out
    with no `X-DevAgents-Version`, no `X-DevAgents-Backend-Version` and no
    `Cache-Control: no-store`.

    Two consequences. A proxy is free to cache a transient `/api/` failure — the exact
    thing I3's `no-store` exists to prevent, on the responses most worth not caching. And
    a person reading an error in production cannot tell which build produced it, which is
    what the version headers are for.

    The fix is for the catch-all to stamp the three headers itself, or for the error
    handling to sit inside the version middleware rather than outside it.
    """
    client = _raising_config_client(monkeypatch, tmp_path)
    try:
        response = client.get("/api/config")
        assert response.status_code == 400
        assert response.headers.get("Cache-Control") == "no-store", (
            "an /api/ error envelope is cacheable"
        )
        assert response.headers.get("X-DevAgents-Version"), "an error envelope has no build stamp"
        assert response.headers.get("X-DevAgents-Backend-Version")
    finally:
        client.__exit__(None, None, None)
