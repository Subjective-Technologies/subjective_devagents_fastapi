#!/usr/bin/env python3
"""Subjective Chat host adapter for an unmodified exported pipeline bundle.

The exported launcher keeps its standard ``~/.<pipeline_name>`` contract. This
adapter temporarily presents Subjective Chat's application home as ``~`` only
while that launcher resolves storage paths, then restores the real user home so
provider CLIs continue finding their existing authentication and histories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import threading
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


STORAGE_KEYS = (
    "USERDATA_PATH",
    "LOG_PATH",
    "CONTEXT_STORAGE",
    "DATASOURCES_TEMP_STORAGE",
    "DS_CONNECTION_TMP_SPACE",
)


def load_launcher(run_py: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"subjective_chat_bundle_{run_py.parent.name}", run_py
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load exported launcher: {run_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_nested_environment(launcher: ModuleType) -> dict[str, str]:
    application_home = os.environ.get("SUBJECTIVE_CHAT_PIPELINE_PARENT_HOME", "").strip()
    if not application_home:
        raise RuntimeError("SUBJECTIVE_CHAT_PIPELINE_PARENT_HOME is required.")

    previous_home = os.environ.get("HOME")
    previous_profile = os.environ.get("USERPROFILE")
    try:
        os.environ["HOME"] = application_home
        os.environ["USERPROFILE"] = application_home
        env = launcher.resolve_userdata_paths()
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
        if previous_profile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = previous_profile

    for key in STORAGE_KEYS:
        value_key = {
            "USERDATA_PATH": "userdata_path",
            "LOG_PATH": "log_path",
            "CONTEXT_STORAGE": "context_storage",
            "DATASOURCES_TEMP_STORAGE": "datasources_temp_storage",
            "DS_CONNECTION_TMP_SPACE": "datasources_temp_storage",
        }[key]
        os.environ[key] = str(env[value_key])
    os.environ["DATASOURCES_PLUGIN_PATH"] = str(launcher.BUNDLE_ROOT / "plugins")
    return env


def execute(run_py: Path, request: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    launcher = load_launcher(run_py)
    env = resolve_nested_environment(launcher)
    launcher.add_plugin_sys_paths()
    launcher.init_bblogger(env)

    passphrase = os.environ.get("SUBJECTIVE_BUNDLE_PASSPHRASE", "").strip()
    if passphrase:
        launcher.materialize_per_sis(passphrase, Path(env["userdata_path"]))

    failures = [
        item for item in launcher.smoke_test_plugins()
        if item.get("status") != "imported"
    ]
    if failures:
        raise RuntimeError(f"Exported plugin import failed: {failures}")

    from subjective_abstract_data_source_package import SubjectivePipelineDataSource

    pipe_file = launcher.BUNDLE_ROOT / "pipelines" / f"{launcher.PIPELINE_NAME}.pipe"
    datasource = SubjectivePipelineDataSource(
        connection={
            "pipeline_name": launcher.PIPELINE_NAME,
            "pipeline_file": str(pipe_file),
        }
    )

    stop_timer = None
    if timeout is not None and timeout > 0:
        stop_timer = threading.Timer(timeout, datasource.stop)
        stop_timer.daemon = True
        stop_timer.start()

    def stop_pipeline(*_args: Any) -> None:
        datasource.stop()

    signal.signal(signal.SIGINT, stop_pipeline)
    try:
        result = datasource.run(request)
    finally:
        if stop_timer is not None:
            stop_timer.cancel()

    return {
        "success": True,
        "pipeline_name": launcher.PIPELINE_NAME,
        "pipeline_home": env["pipeline_home_path"],
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_py", type=Path)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("Pipeline request must be a JSON object.")
        response = execute(args.run_py.resolve(), request, args.timeout)
        print(json.dumps(response, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({
            "success": False,
            "status": "runtime_error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }, ensure_ascii=False))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
