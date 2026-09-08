"""Tests for loading the existing datasource submodules and calling them in-process.

The datasource fleet spans two generations of the plugin API and the pinned submodules are
not all on the same side of that line, so these pin the adapter's tolerance for both, and
pin that a missing or broken plugin degrades to the exported bundle instead of taking the
service down.

Nothing here launches a CLI or talks to a model: the real packages are only imported, and
behaviour is exercised against stand-in classes.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from subjective_devagents_fastapi import datasources  # noqa: E402


@pytest.fixture(autouse=True)
def clear_cache():
    datasources._class_cache.clear()
    yield
    datasources._class_cache.clear()


# --- the two plugin API generations -------------------------------------------------------


class V2DataSource:
    """Current shape: takes connection=, reads self._connection."""

    def __init__(self, connection=None):
        self._connection = connection or {}

    def run(self, request):
        return {"success": True, "status": "ready", "generation": "v2", "seen": self._connection}


class V1DataSource:
    """Older shape, still pinned for some providers: takes params=, reads self.params."""

    def __init__(self, name=None, session=None, dependency_data_sources=None, subscribers=None, params=None):
        self.params = params or {}

    def run(self, request):
        return {"success": True, "status": "ready", "generation": "v1", "seen": self.params}


def test_instantiate_supports_the_current_plugin_api():
    instance = datasources.instantiate(V2DataSource, {"working_directory": "/srv"})
    assert instance._connection == {"working_directory": "/srv"}


def test_instantiate_falls_back_to_the_older_plugin_api():
    """A v1 datasource rejects connection= with a TypeError; the same dict must reach it as
    params= rather than the call failing."""
    instance = datasources.instantiate(V1DataSource, {"working_directory": "/srv"})
    assert instance.params == {"working_directory": "/srv"}


def test_instantiate_does_not_swallow_unrelated_type_errors():
    class Broken:
        def __init__(self, connection=None):
            raise TypeError("model must be a string, not int")

    with pytest.raises(TypeError, match="model must be a string"):
        datasources.instantiate(Broken, {})


# --- connection building --------------------------------------------------------------------


def test_connection_carries_the_fields_a_turn_already_has():
    connection = datasources.connection_for("claude", workspace="/srv/app", session_id="abc", timeout=1800)
    assert connection["working_directory"] == "/srv/app"
    assert connection["session_id"] == "abc"
    assert connection["timeout"] == 1800
    assert connection["auth_method"] == "existing_session"
    assert connection["resume_session"] is True
    # Never invent a sandbox for a provider that manages its own.
    assert "sandbox_mode" not in connection


def test_codex_gets_a_workspace_sandbox_by_default():
    assert datasources.connection_for("codex")["sandbox_mode"] == "workspace-write"


def test_empty_fields_are_omitted_rather_than_sent_blank():
    """A blank session_id must not be sent: datasources treat it as 'resume this session'."""
    connection = datasources.connection_for("claude", workspace="", session_id="")
    assert "session_id" not in connection
    assert "working_directory" not in connection


# --- resolution ------------------------------------------------------------------------------


def test_plugin_path_override_is_searched_first(monkeypatch, tmp_path):
    override = tmp_path / "custom"
    (override / "subjective_codex_datasource").mkdir(parents=True)
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_PLUGIN_PATH", str(override))
    roots = datasources.plugin_roots()
    assert roots[0] == override
    # The bundled submodules stay available as the fallback root.
    assert roots[-1] == datasources.SERVICE_ROOT / "plugins"


def test_unknown_provider_is_reported_clearly():
    with pytest.raises(LookupError, match="No datasource is registered"):
        datasources.load("telepathy")


def test_missing_module_names_where_it_looked(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_PLUGIN_PATH", str(tmp_path))
    datasources._paths_prepared = False
    with pytest.raises(LookupError) as error:
        datasources.load("cursor")
    assert "Looked in" in str(error.value)
    assert "git submodule update" in str(error.value)


def test_available_reports_every_provider_without_raising():
    report = datasources.available()
    assert set(report) == {"codex", "claude", "cursor", "localchats"}
    for key, entry in report.items():
        assert "found" in entry and "loaded" in entry
        if not entry["loaded"]:
            assert entry.get("error"), f"{key} failed to load without saying why"


# --- the submodules that are actually checked out ---------------------------------------------


@pytest.mark.parametrize("provider", ["codex", "claude", "localchats"])
def test_pinned_datasource_submodules_import(provider):
    """These three are pinned as submodules under plugins/. A failure here almost always
    means `git submodule update --init --recursive` has not been run."""
    report = datasources.available()
    if not report[provider]["found"]:
        pytest.skip(f"{provider} submodule is not checked out")
    datasource_class = datasources.load(provider)
    assert hasattr(datasource_class, "run"), f"{provider} datasource exposes no run()"


def test_a_pinned_datasource_can_be_constructed():
    """Construction is where the v1/v2 API split actually bites, so prove the adapter
    bridges it against a real pinned package rather than only the stand-ins above."""
    report = datasources.available()
    provider = next((key for key in ("codex", "claude") if report[key]["loaded"]), None)
    if provider is None:
        pytest.skip("no agent datasource submodule is checked out")
    instance = datasources.instantiate(datasources.load(provider), datasources.connection_for(provider))
    assert hasattr(instance, "run")


# --- can this datasource actually run a turn? ------------------------------------------------


def test_a_datasource_with_its_own_handler_supports_chat():
    class V2(V2DataSource):
        def handle_message(self, message, files=None):
            return {"success": True}

    assert datasources.supports_chat(V2)


def test_a_datasource_that_only_inherits_a_stub_does_not_support_chat():
    """The v1 generation predates the chat contract: those classes define _process_message
    and nothing else, so run() reaches the abstract base and returns a status object. A turn
    routed there silently does nothing, which is worse than a missing plugin — a missing
    plugin at least falls back to the exported bundle that works."""

    class SubjectiveDataSource:  # stands in for the abstract base
        def run(self, request):
            return {"success": True, "status": "ready"}

    class V1Legacy(SubjectiveDataSource):
        def _process_message(self, message):
            return message

    assert not datasources.supports_chat(V1Legacy)


def test_the_pinned_codex_datasource_is_detected_as_chat_incapable():
    """Pins the real-world case: the codex repo on GitHub is six months stale and has no
    run()/handle_message() of its own, so codex turns must go to the bundle."""
    report = datasources.available()
    if not report["codex"]["loaded"]:
        pytest.skip("codex submodule is not checked out")
    assert not datasources.supports_chat(datasources.load("codex"))


def test_the_claude_datasource_can_run_a_turn():
    report = datasources.available()
    if not report["claude"]["loaded"]:
        pytest.skip("claude submodule is not checked out")
    assert datasources.supports_chat(datasources.load("claude"))
