"""Real browser: clear, drag the same chat back, then reload; panes stay assigned."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
pytest.importorskip("selenium")
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from test_selenium_prompt_options import driver, live_server, CLIENT_ROOT
from test_selenium_chat_names import _create_chat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_deployment import verify


@pytest.fixture(scope="module", autouse=True)
def isolated_history():
    # Test chats are created below; do not scan this machine's provider histories.
    from subjective_devagents_fastapi.app import state
    original = state.refresh_history
    async def skip_scan(reason):
        return {"ok": True, "count": 0}
    state.refresh_history = skip_scan
    try:
        yield
    finally:
        state.refresh_history = original


def test_clear_stays_empty_after_drop_and_reload(live_server, driver):
    identity = verify(live_server, CLIENT_ROOT)
    assert identity["frontend"]["source"] and identity["backend"]["source"]
    first = _create_chat(live_server, "Clear this conversation")
    second = _create_chat(live_server, "Keep other conversation")
    for target, text in ((first, "old message to remove"), (second, "other history stays")):
        result = requests.post(live_server + "/api/llm/test/inject-assistant-message",
                               json={"target": target, "text": text}, timeout=10).json()
        assert result["ok"], result
    # Isolate the unrelated external task directory; all chat requests stay real.
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
        const nativeFetch = window.fetch;
        window.fetch = (url, ...args) => String(url) === 'https://taggy.subjectivetechnologies.com/api/state'
            ? Promise.resolve(new Response(JSON.stringify({tasks: [], tags: []}), {status: 200}))
            : nativeFetch(url, ...args);
    """})
    driver.get(live_server)
    wait = WebDriverWait(driver, 15)
    wait.until(EC.element_to_be_clickable((By.ID, "frameLayoutToggle"))).click()
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.frame-layout-cell[data-row="1"][data-col="2"]'))).click()

    def drop(target, pane):
        wait.until(lambda browser: browser.execute_script("""
            const row = document.querySelector('tr[data-target="' + CSS.escape(arguments[0]) + '"]');
            const frame = document.querySelector('.chat-frame[data-frame="' + arguments[1] + '"]');
            if (!row || !frame) return false;
            const transfer = new DataTransfer();
            row.dispatchEvent(new DragEvent('dragstart', {bubbles: true, dataTransfer: transfer}));
            frame.dispatchEvent(new DragEvent('drop', {bubbles: true, dataTransfer: transfer}));
            return true;
        """, target, pane))

    def transcript(pane):
        return driver.find_element(By.CSS_SELECTOR, f'.chat-frame[data-frame="{pane}"] .transcript').text

    drop(first, 0)
    drop(second, 1)
    wait.until(lambda _: "old message to remove" in transcript(0))
    wait.until(lambda _: "other history stays" in transcript(1))
    driver.find_element(By.CSS_SELECTOR, '.chat-frame[data-frame="0"] .frame-clear').click()
    wait.until(lambda _: "old message to remove" not in transcript(0))
    assert driver.find_element(By.CSS_SELECTOR, '.chat-frame[data-frame="0"] .frame-title').text == "Clear this conversation"
    assert "other history stays" in transcript(1)
    drop(first, 0)
    driver.refresh()
    wait.until(lambda _: driver.find_element(By.CSS_SELECTOR, '.chat-frame[data-frame="0"] .frame-title').text == "Clear this conversation")
    wait.until(lambda _: "other history stays" in transcript(1))
    assert "old message to remove" not in transcript(0)
    payload = requests.get(live_server + "/api/llm/conversation", params={"target": first}, timeout=10).json()
    assert payload["conversation"]["messages"] == []
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    folder = Path("/subjective/artifacts/selenium/clear-conversation/v1.0.0")
    folder.mkdir(parents=True, exist_ok=True)
    screenshot = folder / f"{stamp}__clear-conversation__v1.0.0__run-{stamp}__screenshot__001__after-clear-drop-reload.png"
    assert driver.save_screenshot(str(screenshot))
    assert not [row for row in driver.get_log("browser") if row.get("level") == "SEVERE" and row.get("source") == "javascript"]
