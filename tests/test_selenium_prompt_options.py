"""Selenium end-to-end test: does clicking a prompt-option button (Yes / No / Other)
actually reach the backend?

This exists to reproduce, deterministically and without a live CLI turn, the bug report
"the chat prompts me to select an option and clicking it does nothing". It drives the real
browser bundle (web/index.html + webBridge.js + renderer.js, unmodified) against a real
instance of this FastAPI service, and uses the test-only /api/llm/test/inject-assistant-
message endpoint to get an assistant message with a yes/no question on demand — no waiting
on a real agent turn, no CLI credentials required.

Requires: `pip install selenium` (see pyproject.toml's `dev` extra) and a Chrome/Chromium
binary + matching chromedriver on PATH. Every test skips cleanly if either is missing, so
this file is safe to collect even where a browser isn't installed.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))
# The real client bundle, not a test fixture stub — the whole point is exercising the
# actual renderer.js/webBridge.js shipped to users.
CLIENT_ROOT = SERVICE_ROOT.parent.parent / "clients" / "desktop" / "subjective_chat"

selenium = pytest.importorskip("selenium", reason="pip install selenium to run the browser E2E tests")
requests = pytest.importorskip("requests")
uvicorn = pytest.importorskip("uvicorn")

from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.chrome.service import Service  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402

# Prefer a system-installed, apt-linked pair (correct shared libraries guaranteed) over
# Selenium Manager's own auto-download: on a minimal server image the auto-downloaded
# chromedriver has been observed to fail with "unexpectedly exited, status 127" — it
# fetched a binary, but this host is missing something it silently assumes exists.
_BROWSER_BINARIES = ("/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome")
_DRIVER_BINARIES = ("/usr/bin/chromedriver", "/usr/lib/chromium/chromedriver", "/usr/lib/chromium-browser/chromedriver")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    if not (CLIENT_ROOT / "web" / "index.html").is_file():
        pytest.skip(f"Client bundle not found at {CLIENT_ROOT} — is the checkout complete?")

    chat_home = tmp_path_factory.mktemp("chat-home")
    port = _free_port()
    os.environ["SUBJECTIVE_CHAT_HOME"] = str(chat_home)
    os.environ["SUBJECTIVE_CHAT_HISTORY_REFRESH_MS"] = "0"
    os.environ["SUBJECTIVE_DEVAGENTS_D1_SYNC"] = "0"
    os.environ["SUBJECTIVE_DEVAGENTS_ENABLE_TEST_ENDPOINTS"] = "1"
    os.environ["SUBJECTIVE_DEVAGENTS_CLIENT_ROOT"] = str(CLIENT_ROOT)
    # _stamp_api_base() only rewrites the page's <meta name="devagents-api"> / CSP when
    # this is set — even to "". Left unset, the page keeps whatever origin is hardcoded in
    # web/index.html (the real production API), the browser's CSP then blocks every fetch
    # to our local test server, and the UI looks permanently empty for reasons that have
    # nothing to do with whatever the test is actually exercising. Must be "" (same-origin),
    # not omitted.
    os.environ["SUBJECTIVE_DEVAGENTS_PUBLIC_API"] = ""
    os.environ.pop("SUBJECTIVE_CHAT_IRC", None)
    os.environ.pop("SUBJECTIVE_DEVAGENTS_API_TOKEN", None)

    from subjective_devagents_fastapi import app as app_module

    # TEST_ENDPOINTS_ENABLED is read from the env once at module import time. If another
    # test file in the same session imported app.py first (with the flag unset), the env
    # var above would be too late — so set the already-imported module's attribute
    # directly, which works regardless of import order.
    app_module.TEST_ENDPOINTS_ENABLED = True
    app = app_module.app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "FastAPI test server did not start in time"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _build_driver(*, mobile_emulation: dict | None = None):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if mobile_emulation is None:
        options.add_argument("--window-size=1280,900")
    else:
        # Unlike set_window_size (pixels only), this sets Chrome's actual mobile emulation:
        # a phone User-Agent, touch-event dispatch instead of mouse, and matching device
        # pixel ratio — a resized desktop window shares none of those with a real phone
        # browser, only the CSS viewport width.
        options.add_experimental_option("mobileEmulation", mobile_emulation)
    for candidate in _BROWSER_BINARIES:
        if Path(candidate).is_file():
            options.binary_location = candidate
            break
    service = None
    for candidate in _DRIVER_BINARIES:
        if Path(candidate).is_file():
            service = Service(executable_path=candidate)
            break
    try:
        return webdriver.Chrome(options=options, service=service) if service else webdriver.Chrome(options=options)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Chrome/chromedriver not available: {exc}")


@pytest.fixture()
def driver():
    instance = _build_driver()
    try:
        yield instance
    finally:
        instance.quit()


@pytest.fixture()
def mobile_driver():
    instance = _build_driver(mobile_emulation={"deviceName": "iPhone 12 Pro"})
    try:
        yield instance
    finally:
        instance.quit()


def _create_conversation(live_server: str) -> str:
    response = requests.post(
        f"{live_server}/api/llm/create-conversation",
        json={"provider": "codex", "name": "Selenium prompt-options check"},
        timeout=10,
    ).json()
    assert response.get("ok"), response
    return response["conversation"]["target"]


def _inject_question(live_server: str, target: str, text: str) -> None:
    response = requests.post(
        f"{live_server}/api/llm/test/inject-assistant-message",
        json={"target": target, "text": text},
        timeout=10,
    ).json()
    assert response.get("ok"), response


def _prompt_action_button(driver, label: str):
    for button in driver.find_elements(By.CSS_SELECTOR, ".prompt-actions .prompt-action"):
        try:
            if button.find_element(By.CSS_SELECTOR, ".prompt-action-label").text == label:
                return button
        except Exception:  # noqa: BLE001 - element mid-(re)render, try again next poll
            continue
    return None


def test_prompt_options_render_for_a_yes_no_question(live_server, driver):
    """The backend's promptOptions heuristic (services/.../prompt_options.py) must reach
    the DOM as real, labelled buttons — not just be computed and dropped."""
    target = _create_conversation(live_server)

    driver.get(live_server)
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.ID, "createLlmButton")))

    row = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]'))
    )
    row.click()

    _inject_question(live_server, target, "Should I proceed with this change?")

    wait.until(lambda d: _prompt_action_button(d, "Yes") is not None)
    assert _prompt_action_button(driver, "No") is not None
    assert _prompt_action_button(driver, "Other…") is not None


def _assert_yes_click_reaches_backend(driver, live_server: str) -> None:
    """Reproduces the reported bug directly: click Yes, and something must visibly happen.

    We do not require a real CLI login to pass — that would make the test depend on
    machine-specific credentials. What we assert is the actual failure mode reported:
    (1) the click must synchronously disable the buttons (proof sendAgentAction's handler
    fired at all — this is the part that was silent), and (2) the pending state must
    eventually clear (proof the round trip to /api/llm/send-message completed, whether the
    CLI itself then succeeded or cleanly failed) rather than hanging forever, which is what
    "I clicked Yes and nothing happened" looks like from the outside.
    """
    target = _create_conversation(live_server)

    driver.get(live_server)
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.ID, "createLlmButton")))

    # Below the 720px breakpoint the conversation list lives inside <dialog id="mobileDrawer">
    # (renderer.js moves the node there, matching real mobile navigation) — a <dialog>
    # without .showModal() is not "clickable" as far as the browser or Selenium are
    # concerned, so the real first step on a phone is opening it via the hamburger button.
    mobile_menu_buttons = driver.find_elements(By.ID, "mobileMenuButton")
    if mobile_menu_buttons and mobile_menu_buttons[0].is_displayed():
        mobile_menu_buttons[0].click()

    row = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]'))
    )
    row.click()

    _inject_question(live_server, target, "Should I proceed with this change?")
    wait.until(lambda d: _prompt_action_button(d, "Yes") is not None)

    yes_button = _prompt_action_button(driver, "Yes")
    assert yes_button.is_enabled(), "Yes button should start enabled"

    # A plain WebDriver .click() issues a trusted click via Chrome's own input pipeline
    # (not a synthetic dispatchEvent), so it is affected by real-world click-blockers the
    # same way a human's click would be: a transparent overlay, a mis-sized hit target, or
    # pointer-events:none from a CSS rule scoped to this viewport width all make Selenium's
    # click fail the same way. That is exactly what we want here — desktop-width vs.
    # mobile-width is the one variable changing between the two callers of this helper.
    yes_button.click()

    def _click_registered(d):
        button = _prompt_action_button(d, "Yes")
        # None means the transcript already re-rendered past this state — also proof the
        # click triggered real activity, not silence.
        return button is None or button.get_attribute("disabled") is not None

    WebDriverWait(driver, 5).until(_click_registered)

    def _round_trip_finished(d):
        return len(d.find_elements(By.CSS_SELECTOR, ".prompt-actions.pending")) == 0

    # _round_trip_finished already is the real proof: the pending state only clears once
    # the backend's /api/llm/send-message call has returned, success or failure — a silent
    # hang (the bug being chased) would leave it pending until this 60s wait times out. A
    # clean provider failure (e.g. no CLI installed/authenticated in this environment) does
    # NOT append to the stored transcript — chat_core.py broadcasts it as a 'system' event
    # instead — so message count is not a reliable second signal and asserting on it here
    # would fail on environment differences that have nothing to do with the bug in question.
    WebDriverWait(driver, 60).until(_round_trip_finished)

    conversation = requests.get(
        f"{live_server}/api/llm/conversation", params={"target": target}, timeout=10
    ).json()
    assert conversation.get("ok"), conversation


def test_clicking_yes_reaches_the_backend_instead_of_doing_nothing(live_server, driver):
    """Desktop-width viewport (this file's default driver size, 1280x900) — the width the
    user reported as broken."""
    driver.set_window_size(1280, 900)
    _assert_yes_click_reaches_backend(driver, live_server)


def test_clicking_yes_reaches_the_backend_on_real_mobile_emulation(live_server, mobile_driver):
    """Same exact check, under Chrome's actual mobile emulation (iPhone 12 Pro: phone
    User-Agent, touch-event dispatch instead of mouse, matching device pixel ratio) — not
    just a resized desktop window, which only matches a real phone's CSS viewport width and
    nothing else. If only one of the two click tests fails, that is real evidence of what
    differs between the reported-broken and reported-working cases; a plain resize could
    not have shown that, since it never diverged from desktop in anything but pixel count."""
    _assert_yes_click_reaches_backend(mobile_driver, live_server)
