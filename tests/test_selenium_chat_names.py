"""Assigned chat names stay synchronized across sidebar, header, and visible panes.

Uses the real FastAPI service and browser bundle through the existing isolated
prompt-options fixtures. No provider CLI call or production chat mutation is needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("selenium")
requests = pytest.importorskip("requests")

from selenium.webdriver import ActionChains  # noqa: E402
from selenium.common.exceptions import StaleElementReferenceException  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402

from test_selenium_prompt_options import driver, live_server  # noqa: E402, F401


def _create_chat(base_url: str, name: str) -> str:
    response = requests.post(
        f"{base_url}/api/llm/create-conversation",
        json={"provider": "codex", "name": name},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload.get("ok"), payload
    return payload["conversation"]["target"]


def _row_selector(target: str) -> str:
    return f'tr.conversation-row[data-target="{target}"]'


def _assert_labels(driver, wait, target: str, name: str, pane: int, *, active=False):
    selectors = [
        f"{_row_selector(target)} .conversation-label",
        f'.chat-frame[data-frame="{pane}"] .frame-title',
    ]
    if active:
        selectors.append("#topChannel")
    for selector in selectors:
        wait.until(lambda browser: browser.find_element(By.CSS_SELECTOR, selector).text == name)
    title = driver.find_element(By.CSS_SELECTOR, selectors[1])
    assert title.get_attribute("title") == name


def test_assigned_chat_names_update_all_panes_and_survive_refresh(live_server, driver):
    """Rename via UI and another client's API event; neither pane may retain old text."""
    run_id = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    first = _create_chat(live_server, "See agents protocol manager original prompt")
    second = _create_chat(live_server, "See agents protocol reverse engineer original prompt")
    first_name = "Manager renamed by user"
    second_name = "Reverse engineer renamed remotely"

    driver.get(live_server)
    wait = WebDriverWait(driver, 15, ignored_exceptions=(StaleElementReferenceException,))
    wait.until(EC.element_to_be_clickable((By.ID, "frameLayoutToggle"))).click()
    wait.until(EC.element_to_be_clickable((
        By.CSS_SELECTOR, '.frame-layout-cell[data-row="2"][data-col="2"]'
    ))).click()
    wait.until(lambda browser: len(browser.find_elements(By.CSS_SELECTOR, ".chat-frame")) == 4)
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, _row_selector(first)))).click()
    driver.find_element(By.CSS_SELECTOR, '.chat-frame[data-frame="1"] .frame-head').click()
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, _row_selector(second)))).click()
    _assert_labels(driver, wait, first, "See agents protocol manager original prompt", 0)
    _assert_labels(driver, wait, second, "See agents protocol reverse engineer original prompt", 1, active=True)

    # Native double-click exercises the shipped inline rename handler, not a stub or
    # direct mutation of renderer state. Existing chat remains in its original pane.
    row = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, _row_selector(first))))
    ActionChains(driver).double_click(row).perform()
    field = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".conversation-rename-input")))
    field.send_keys(Keys.CONTROL, "a")
    field.send_keys(first_name, Keys.ENTER)
    _assert_labels(driver, wait, first, first_name, 0, active=True)

    # Another connected client can rename an inactive but visible pane. Receiving the
    # broadcast must repaint that pane without switching focus or refreshing history.
    response = requests.post(
        f"{live_server}/api/llm/rename-conversation",
        json={"target": second, "name": second_name},
        timeout=10,
    )
    response.raise_for_status()
    assert response.json().get("ok"), response.json()
    _assert_labels(driver, wait, second, second_name, 1)
    _assert_labels(driver, wait, first, first_name, 0, active=True)

    # Reload discards in-memory labels. Both persisted API data and restored pane
    # layout must still use assigned titles, not their original prompt-derived names.
    driver.refresh()
    _assert_labels(driver, wait, first, first_name, 0, active=True)
    _assert_labels(driver, wait, second, second_name, 1)
    listed = requests.get(f"{live_server}/api/llm/conversations", timeout=10)
    listed.raise_for_status()
    labels = {item["target"]: item["label"] for item in listed.json()["conversations"]}
    assert labels[first] == first_name
    assert labels[second] == second_name

    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    artifacts = Path("/subjective/artifacts/selenium/devagents-chat-names/v1.0.0")
    artifacts.mkdir(parents=True, exist_ok=True)
    screenshot = artifacts / (
        f"{timestamp}__devagents-chat-names__v1.0.0__run-{run_id}"
        "__screenshot__001__renamed-panes-after-refresh.png"
    )
    assert driver.save_screenshot(str(screenshot))
    errors = [
        item for item in driver.get_log("browser")
        if item.get("level") == "SEVERE" and item.get("source") == "javascript"
    ]
    assert not errors, errors
