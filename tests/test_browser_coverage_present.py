"""Guard against the browser tests skipping and the run still looking green.

Two of this service's defects were only ever caught in a browser: a pane
destroying the button being pressed, and the composer losing the cursor after
every send. Both are covered exclusively by the Selenium suites — and those skip
themselves when selenium or Chrome is absent, which is the correct behaviour on a
developer's machine and a silent hole anywhere it matters. A run on a box without
a browser reports success while testing neither.

So the requirement is made explicit rather than assumed. Set

    SUBJECTIVE_DEVAGENTS_REQUIRE_BROWSER=1

wherever browser coverage is supposed to happen — CI, a release check — and a
missing browser becomes a failure that names itself instead of a skip nobody
reads. Left unset, this says nothing.
"""
from __future__ import annotations

import importlib
import os
import shutil

import pytest

REQUIRED = (os.environ.get("SUBJECTIVE_DEVAGENTS_REQUIRE_BROWSER") or "").strip() == "1"

# The names Chrome and Chromium install under across the platforms this runs on.
CHROME_BINARIES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)


@pytest.mark.skipif(
    not REQUIRED,
    reason="set SUBJECTIVE_DEVAGENTS_REQUIRE_BROWSER=1 where browser coverage is expected",
)
def test_selenium_is_installed():
    try:
        importlib.import_module("selenium")
    except ImportError:  # pragma: no cover - the point of the test
        pytest.fail(
            "selenium is missing, so every browser test silently skipped. "
            "Install this service's dev extra: pip install -e '.[dev]'"
        )


@pytest.mark.skipif(
    not REQUIRED,
    reason="set SUBJECTIVE_DEVAGENTS_REQUIRE_BROWSER=1 where browser coverage is expected",
)
def test_a_browser_is_installed():
    found = [name for name in CHROME_BINARIES if shutil.which(name)]
    assert found, (
        "no Chrome or Chromium on PATH, so every browser test silently skipped — "
        "including the only coverage of the prompt-option buttons and the composer "
        f"keeping focus. Looked for: {', '.join(CHROME_BINARIES)}"
    )


@pytest.mark.skipif(
    not REQUIRED,
    reason="set SUBJECTIVE_DEVAGENTS_REQUIRE_BROWSER=1 where browser coverage is expected",
)
def test_the_client_bundle_is_checked_out():
    """The browser suites drive the real client, which is a sibling submodule.

    Without it they skip for a third reason that also reads as success.
    """
    from browser_support import is_client_bundle, resolve_client_root

    root = resolve_client_root()
    assert is_client_bundle(root), (
        f"no client bundle at {root}, so every browser test silently skipped. "
        "Run: git submodule update --init --recursive, or point "
        "SUBJECTIVE_DEVAGENTS_TEST_CLIENT_ROOT at the checkout."
    )
