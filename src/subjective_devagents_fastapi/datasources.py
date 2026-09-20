"""Loads the existing Subjective datasource modules and calls them in-process.

These are the same datasources the monolithic web app already uses — `subjective_codex_datasource`,
`subjective_claude_cli_datasource`, `subjective_localchats_datasource` — vendored here as git
submodules under `plugins/` rather than reimplemented. Each exposes the same v2 contract:

    SubjectiveXDataSource(connection={...}).run(request_dict) -> result_dict

which is exactly the payload `llm_service` already speaks, so a datasource can be called
directly instead of paying for a Python interpreter start per turn.

Two paths exist on purpose:

- **In-process** (this module) for work running as the current OS user. No subprocess, no
  bundle unpacking, and exceptions surface as real tracebacks.
- **Subprocess through the su bridge** (`llm_service._run_pipeline_for_user`) for a
  "Continue as <other user>" conversation. That one cannot be replaced: running as another
  uid is the entire point, and an import cannot change the uid of this process.

Resolution order for a datasource package, first hit wins:

1. `SUBJECTIVE_DEVAGENTS_PLUGIN_PATH` — colon-separated roots, for pointing at a working
   copy (see the version-skew note in the README).
2. `plugins/` in this repo — the pinned submodules.
3. Already importable on `sys.path`.

Import failures are never fatal here: `available()` reports what loaded and `load()` raises
only when something actually asks for a missing datasource, so the service still starts and
serves history when a plugin is absent.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable

SERVICE_ROOT = Path(__file__).resolve().parents[2]

# provider key -> (package directory name, module file stem, class name)
DATASOURCE_MODULES = {
    "codex": ("subjective_codex_datasource", "SubjectiveCodexDataSource", "SubjectiveCodexDataSource"),
    "claude": ("subjective_claude_cli_datasource", "SubjectiveClaudeCliDataSource", "SubjectiveClaudeCliDataSource"),
    "cursor": ("subjective_cursor_datasource", "SubjectiveCursorDataSource", "SubjectiveCursorDataSource"),
    "grok": ("subjective_grok_cli_datasource", "SubjectiveGrokCliDataSource", "SubjectiveGrokCliDataSource"),
    "localchats": ("subjective_localchats_datasource", "SubjectiveLocalchatsDataSource", "SubjectiveLocalchatsDataSource"),
    "ssh": ("subjective_ssh_datasource", "SubjectiveSshDataSource", "SubjectiveSshDataSource"),
}

# The abstract base every datasource subclasses, plus the config and logger packages they
# import. Same names and layout as the monorepo's libs/dependencies/, so a checkout of this
# service resolves them identically to a checkout of the monorepo.
SUPPORT_PACKAGE_DIRS = (
    SERVICE_ROOT / "libs" / "dependencies" / "subjective-abstract-data-source-package",
    SERVICE_ROOT / "libs" / "dependencies" / "brainboost_data_source_logger_package",
    SERVICE_ROOT / "libs" / "dependencies" / "brainboost_configuration_package",
)

_lock = threading.RLock()
_class_cache: dict[str, type] = {}
_paths_prepared = False


def plugin_roots() -> list[Path]:
    roots: list[Path] = []
    override = os.environ.get("SUBJECTIVE_DEVAGENTS_PLUGIN_PATH", "")
    for entry in override.split(os.pathsep):
        entry = entry.strip()
        if entry:
            roots.append(Path(entry).expanduser())
    roots.append(SERVICE_ROOT / "plugins")
    return roots


def _prepare_sys_path() -> None:
    """Put the support packages on sys.path once, so `from subjective_abstract_data_source_package
    import SubjectiveDataSource` resolves the same way it does inside an exported bundle."""
    global _paths_prepared
    if _paths_prepared:
        return
    for directory in SUPPORT_PACKAGE_DIRS:
        if directory.is_dir():
            path = str(directory)
            if path not in sys.path:
                sys.path.insert(0, path)
    # A datasource may import a sibling helper by bare name (localchats does: `context`).
    for root in plugin_roots():
        if root.is_dir():
            for package in root.iterdir():
                if package.is_dir() and str(package) not in sys.path:
                    sys.path.append(str(package))
    _paths_prepared = True


def _find_module_file(package_dir: str, module_stem: str) -> Path | None:
    for root in plugin_roots():
        candidate = root / package_dir / f"{module_stem}.py"
        if candidate.is_file():
            return candidate
    return None


def load(provider: str) -> type:
    """Import a datasource class by provider key. Raises LookupError when unavailable."""
    key = str(provider or "").strip().lower()
    if key not in DATASOURCE_MODULES:
        raise LookupError(f"No datasource is registered for provider {provider!r}.")
    with _lock:
        cached = _class_cache.get(key)
        if cached is not None:
            return cached

        package_dir, module_stem, class_name = DATASOURCE_MODULES[key]
        _prepare_sys_path()
        module_file = _find_module_file(package_dir, module_stem)
        if module_file is None:
            searched = ", ".join(str(root / package_dir) for root in plugin_roots())
            raise LookupError(
                f"Datasource module {module_stem}.py for {key!r} was not found. Looked in: {searched}. "
                "Run `git submodule update --init --recursive`, or set "
                "SUBJECTIVE_DEVAGENTS_PLUGIN_PATH to a directory holding the plugin packages."
            )

        spec = importlib.util.spec_from_file_location(f"subjective_devagents_ds_{key}", module_file)
        if spec is None or spec.loader is None:
            raise LookupError(f"Could not load {module_file}.")
        module = importlib.util.module_from_spec(spec)
        # Register before exec so a datasource that imports itself by name does not re-enter.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:  # noqa: BLE001
            sys.modules.pop(spec.name, None)
            raise LookupError(f"Importing {module_file} failed: {error}") from error

        datasource_class = getattr(module, class_name, None)
        if datasource_class is None:
            raise LookupError(f"{module_file} does not define {class_name}.")
        _class_cache[key] = datasource_class
        return datasource_class


def available() -> dict[str, dict]:
    """Report which datasources can be imported right now, for /api/datasources and startup
    logging. Never raises: a missing plugin is a reported state, not a crash."""
    report: dict[str, dict] = {}
    for key, (package_dir, module_stem, _class_name) in DATASOURCE_MODULES.items():
        module_file = _find_module_file(package_dir, module_stem)
        entry: dict[str, Any] = {"package": package_dir, "found": module_file is not None}
        if module_file is not None:
            entry["path"] = str(module_file)
        try:
            load(key)
            entry["loaded"] = True
        except LookupError as error:
            entry["loaded"] = False
            entry["error"] = str(error)
        report[key] = entry
    return report


def supports_chat(datasource_class: type) -> bool:
    """Can this datasource actually run an agent turn?

    The fleet spans two generations, and the older one predates the chat contract
    entirely: those classes define `_process_message` and no `run`/`handle_message` of
    their own, so calling `run()` silently reaches the abstract base and returns a status
    object instead of executing anything. A turn routed there does nothing and reports
    success-shaped nonsense, which is worse than a missing plugin — a missing plugin at
    least falls back to the exported bundle that works.

    Checked against the class's own attributes, not the base's, so inheriting a stub does
    not count as support.
    """
    for attribute in ("handle_message", "run"):
        for klass in datasource_class.__mro__:
            if klass.__name__ in ("SubjectiveDataSource", "SubjectiveOnDemandDataSource", "object"):
                continue
            if attribute in vars(klass):
                return True
    return False


def instantiate(datasource_class: type, connection: dict | None = None):
    """Construct a datasource across both generations of the plugin API.

    The fleet is mid-migration: v2 datasources take `connection=` and read `self._connection`,
    while older ones take `params=` and read `self.params`. Both receive the same dict, so
    try the current shape first and fall back rather than pinning the service to one
    generation — the pinned submodules are not all on the same side of that line.
    """
    settings = dict(connection or {})
    try:
        return datasource_class(connection=settings)
    except TypeError as error:
        if "connection" not in str(error):
            raise
        return datasource_class(params=settings)


def run(
    provider: str,
    request: dict,
    connection: dict | None = None,
    on_instance: Callable[[Any], None] | None = None,
) -> dict:
    """Instantiate the datasource and run one request, returning its result dict.

    A fresh instance per call keeps this stateless: the datasources hold a session id and a
    lock on the instance, and conversations here are resumed by passing `session_id` in the
    request rather than by keeping a long-lived object per chat.

    `on_instance` is handed the live instance before the (blocking) call, which is the only
    moment a caller can reach it — that is how Stop gets hold of a turn's cancel().
    """
    datasource_class = load(provider)
    instance = instantiate(datasource_class, connection)
    if on_instance is not None:
        on_instance(instance)
    result = instance.run(dict(request or {}))
    if not isinstance(result, dict):
        return {"success": True, "status": "completed", "result": result}
    return result


def connection_for(provider: str, workspace: str = "", session_id: str = "", timeout: int | None = None,
                   extra: dict | None = None) -> dict:
    """Build the connection dict a datasource expects from the fields a turn already has."""
    connection: dict[str, Any] = {"auth_method": "existing_session", "resume_session": True}
    if workspace:
        connection["working_directory"] = workspace
    if session_id:
        connection["session_id"] = session_id
    if timeout:
        connection["timeout"] = int(timeout)
    if provider and str(provider).lower() == "codex":
        connection.setdefault("sandbox_mode", "workspace-write")
    if provider and str(provider).lower() == "ssh":
        connection.update(ssh_connection_from_workspace(workspace))
    connection.update(extra or {})
    return connection


# `[user@]host[:port][:/path]` or `ssh://[user@]host[:port]/path`.
_SSH_WORKSPACE_RE = re.compile(
    r"""^
    (?:(?P<username>[^@/\s]+)@)?
    (?P<host>\[[0-9A-Fa-f:]+\]|[^@:/\s]+)
    (?::(?P<port>\d+))?
    (?::?(?P<path>/\S*))?
    $""",
    re.VERBOSE,
)


def ssh_connection_from_workspace(workspace: str) -> dict:
    """Read an SSH target out of a conversation's workspace string.

    The SSH datasource needs a host and an account, and a conversation has exactly one
    free-text field that can carry them: its workspace. So an SSH chat's workspace is
    written `deploy@server.example.com:/srv/app` (or `ssh://deploy@server:2222/srv/app`)
    and is split here into the datasource's connection fields.

    This reaches only the fields that fit in one line, which means agent
    authentication — the default, and the only method that stores no secret. Keys,
    passwords and pinned fingerprints need somewhere to be entered and stored per
    connection; until that exists they can be supplied through `extra`.

    An unparseable workspace yields `{}` rather than a guess: the datasource then
    reports `configuration_required` naming what is missing, which is a better failure
    than connecting somewhere unintended.
    """
    text = str(workspace or "").strip()
    if not text:
        return {}
    if text.startswith("ssh://"):
        text = text[len("ssh://"):]
    # A plain local path is a workspace for the CLI datasources, not an SSH target.
    if text.startswith(("/", ".", "~")):
        return {}
    match = _SSH_WORKSPACE_RE.match(text)
    if not match:
        return {}
    connection: dict[str, Any] = {"host": match.group("host").strip("[]")}
    if match.group("username"):
        connection["username"] = match.group("username")
    if match.group("port"):
        connection["port"] = int(match.group("port"))
    # The datasource's own `working_directory`, not the raw workspace string that
    # connection_for already put there.
    connection["working_directory"] = match.group("path") or ""
    return connection
