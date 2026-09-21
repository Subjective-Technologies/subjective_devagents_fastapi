"""Browser coverage for the SSH features: the connection manager, the chat kinds,
the markers that tell them apart, and the standalone console.

Driven against a real instance of this service serving the real client bundle —
the same arrangement as test_selenium_prompt_options.py, and for the same reason:
these are the files shipped to users, not a fixture copy of them.

**Not against the production dashboard**, deliberately. Reaching it needs an owner
Firebase sign-in driven through a third-party login page, the token expires in an
hour, and every run would create chats and open SSH connections on the live box.
A suite that can only run when someone has just signed in by hand, and that leaves
debris in production when it does, is not one anybody will keep running. What it
would catch over this — the dashboard's own iframe wrapper — is one integration
point, and the CSP that governs it is asserted in test_api.py already.

The one thing that cannot be faked is a real machine, so the tests that need one
are grouped at the end and skip themselves when it is not reachable.
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
CLIENT_ROOT = Path(os.environ.get(
    "SUBJECTIVE_DEVAGENTS_TEST_CLIENT_ROOT",
    str(SERVICE_ROOT.parent.parent / "clients" / "desktop" / "subjective_chat"),
))

selenium = pytest.importorskip("selenium", reason="pip install selenium to run the browser E2E tests")
requests = pytest.importorskip("requests")
uvicorn = pytest.importorskip("uvicorn")

from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402

# The chromedriver plumbing is solved once, next door; importing the fixture keeps
# one copy of it rather than two that can drift.
from test_selenium_prompt_options import driver  # noqa: E402,F401

TOKEN = "selenium-web-ui-token"
LIVE_HOST = os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_TEST_HOST", "subjective-agents")

# Enough of a key to pass the upload's header check without being one.
FAKE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def ui_server(tmp_path_factory):
    """A service with the SSH terminal on.

    It cannot share the neighbouring fixture: that one runs in development mode
    with no token, and the terminal refuses to enable without authentication —
    which is the interlock working, not a problem to route around.
    """
    if not (CLIENT_ROOT / "web" / "index.html").is_file():
        pytest.skip(f"Client bundle not found at {CLIENT_ROOT} — is the checkout complete?")

    home = tmp_path_factory.mktemp("ui-home")
    port = _free_port()
    # Restored on the way out: these are process-wide, and a module fixture that
    # leaves the SSH terminal switched on makes another module's "it is off unless
    # switched on" test pass for the wrong reason — or fail.
    saved_environment = dict(os.environ)
    os.environ.update({
        "SUBJECTIVE_CHAT_HOME": str(home),
        "SUBJECTIVE_CHAT_HISTORY_REFRESH_MS": "0",
        "SUBJECTIVE_DEVAGENTS_D1_SYNC": "0",
        "SUBJECTIVE_DEVAGENTS_ENABLE_TEST_ENDPOINTS": "1",
        "SUBJECTIVE_DEVAGENTS_CLIENT_ROOT": str(CLIENT_ROOT),
        "SUBJECTIVE_DEVAGENTS_PUBLIC_API": "",
        "SUBJECTIVE_DEVAGENTS_AUTH_MODE": "token",
        "SUBJECTIVE_DEVAGENTS_API_TOKEN": TOKEN,
        "SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED": "1",
        # Keys and config aliases are read from here, so the suite sees a directory
        # it controls instead of whichever developer's ~/.ssh is on the machine.
        "SUBJECTIVE_DEVAGENTS_SSH_DIR": str(tmp_path_factory.mktemp("dot-ssh")),
    })
    os.environ.pop("SUBJECTIVE_CHAT_IRC", None)

    from subjective_devagents_fastapi.app import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{base}/healthz", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        pytest.skip("the service did not start")
    yield base
    server.should_exit = True
    thread.join(timeout=10)
    os.environ.clear()
    os.environ.update(saved_environment)


def api(server: str, path: str, payload=None):
    """Call the service the way the browser does, token and all."""
    headers = {"X-DevAgents-Token": TOKEN}
    if payload is None:
        return requests.get(f"{server}{path}", headers=headers, timeout=15).json()
    return requests.post(f"{server}{path}", json=payload, headers=headers, timeout=60).json()


@pytest.fixture()
def ui(ui_server, driver, monkeypatch):  # noqa: F811 - `driver` is the imported fixture
    """The app, signed in, at a desktop size.

    The auth mode has to be re-pinned here. conftest sets development mode for
    every test with an autouse fixture, which runs after this module's server
    fixture and would otherwise put the running service into a mode where the SSH
    terminal refuses to start.
    """
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "token")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_API_TOKEN", TOKEN)
    driver.set_window_size(1400, 1000)
    # The bundle is served from disk at a URL that does not change between runs, so
    # a driver reused across tests would keep an edited script cached.
    try:
        driver.execute_cdp_cmd("Network.clearBrowserCache", {})
    except Exception:  # noqa: BLE001 - only Chrome has CDP; the wait below still applies
        pass
    # The page takes a token from the query once and keeps it; this is the flow the
    # operator uses too.
    driver.get(f"{ui_server}/?token={TOKEN}")
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "createLlmButton")))
    # The app is hidden until the page has signed in. Waiting for the button to be
    # *usable* rather than merely present is what distinguishes "loaded" from
    # "loaded and signed in", and the auth panel says which if it never happens.
    try:
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "createLlmButton"))
        )
    except Exception as error:  # noqa: BLE001
        status = driver.find_element(By.ID, "ownerAuthStatus").text
        raise AssertionError(f"the app never unlocked; the auth panel says: {status!r}") from error
    return driver


def reload(driver, server):
    """Reload the way an operator would: with the token.

    A plain refresh() will not do. The shared token is held for the life of the
    page and never stored, so a reload without it lands back on the sign-in panel
    — which is the intended behaviour, not a bug to work around here.
    """
    driver.get(f"{server}/?token={TOKEN}")
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "createLlmButton")))


def open_new_chat_dialog(driver):
    driver.find_element(By.ID, "createLlmButton").click()
    WebDriverWait(driver, 10).until(
        lambda d: d.find_element(By.ID, "llmDialog").get_attribute("open") is not None
    )


def select_agent(driver, value: str):
    from selenium.webdriver.support.ui import Select

    Select(driver.find_element(By.ID, "llmProviderSelect")).select_by_value(value)


def visible(driver, element_id: str) -> bool:
    return driver.find_element(By.ID, element_id).is_displayed()


# --- the new-chat dialog ---------------------------------------------------------


def test_ssh_is_offered_as_an_agent(ui):
    open_new_chat_dialog(ui)
    values = [o.get_attribute("value")
              for o in ui.find_elements(By.CSS_SELECTOR, "#llmProviderSelect option")]
    assert "ssh" in values, values


def test_the_ssh_datasource_asks_for_a_machine_instead_of_a_workspace(ui):
    open_new_chat_dialog(ui)
    select_agent(ui, "ssh")
    assert visible(ui, "sshConnectionRow"), "the connection picker should be shown"
    # "Run as" picks whose *local* login answers, which an SSH chat does not have.
    assert not visible(ui, "llmOsUserRow")


def test_an_agent_can_be_switched_to_run_over_ssh(ui):
    """The toggle that puts Claude on another machine."""
    open_new_chat_dialog(ui)
    select_agent(ui, "claude")
    assert not visible(ui, "sshConnectionRow")
    ui.find_element(By.ID, "llmOverSshToggle").click()
    assert visible(ui, "sshConnectionRow"), "picking SSH should reveal the connection picker"
    assert not visible(ui, "llmOsUserRow")
    # The workspace stays: for an agent it is the directory, now a remote one.
    assert ui.find_element(By.ID, "llmWorkspaceInput").is_displayed()
    # .text is the rendered text, and the stylesheet uppercases these labels.
    assert "REMOTE" in ui.find_element(By.ID, "llmWorkspaceLabel").text.upper()


# --- the connection form ----------------------------------------------------------


def _fill_connection(driver, name: str, host: str = "example.test", username: str = "deploy"):
    driver.find_element(By.ID, "sshHostInput").send_keys(host)
    driver.find_element(By.ID, "sshUsernameInput").send_keys(username)
    driver.find_element(By.ID, "sshNameInput").send_keys(name)


def test_a_connection_can_be_saved_and_comes_back_in_the_picker(ui, ui_server):
    open_new_chat_dialog(ui)
    select_agent(ui, "ssh")
    _fill_connection(ui, "Saved by selenium")
    ui.find_element(By.ID, "llmNameInput").send_keys("chat using a saved connection")
    ui.find_element(By.CSS_SELECTOR, "#llmForm button[type=submit]").click()

    WebDriverWait(ui, 15).until(
        lambda d: any(c["name"] == "Saved by selenium"
                      for c in api(ui_server, "/api/ssh/connections")["connections"])
    )
    saved = [c for c in api(ui_server, "/api/ssh/connections")["connections"]
             if c["name"] == "Saved by selenium"][0]
    assert saved["host"] == "example.test"
    assert saved["username"] == "deploy"
    # Never a secret, only whether one is set.
    assert saved["hasPassword"] is False


def test_a_private_key_can_be_uploaded_from_the_browser(ui, ui_server, tmp_path):
    """The server's own ~/.ssh holds the server's keys; without this a new machine
    cannot be reached at all."""
    key_file = tmp_path / "id_selenium"
    key_file.write_text(FAKE_KEY, encoding="utf-8")

    open_new_chat_dialog(ui)
    select_agent(ui, "ssh")
    # The input is hidden behind the Upload… button; a file input accepts a path
    # either way, which is how a file picker is driven.
    ui.find_element(By.ID, "sshKeyFileInput").send_keys(str(key_file))

    WebDriverWait(ui, 15).until(
        lambda d: any(k["name"] == "id_selenium" and k["source"] == "uploaded"
                      for k in api(ui_server, "/api/ssh/connections")["keys"])
    )
    status = ui.find_element(By.ID, "sshTestStatus").text
    assert "id_selenium" in status, status


def test_uploading_the_public_key_by_mistake_is_refused_with_advice(ui, tmp_path):
    """The easiest mistake to make, and it must not land in the key list."""
    public_key = tmp_path / "id_selenium.pub"
    public_key.write_text("ssh-ed25519 AAAAC3NzaC1 user@host\n", encoding="utf-8")

    open_new_chat_dialog(ui)
    select_agent(ui, "ssh")
    ui.find_element(By.ID, "sshKeyFileInput").send_keys(str(public_key))
    WebDriverWait(ui, 15).until(
        lambda d: "private key" in d.find_element(By.ID, "sshTestStatus").text
    )
    assert ".pub" in ui.find_element(By.ID, "sshTestStatus").text


def test_testing_a_connection_that_cannot_be_reached_says_so(ui):
    """The Test button exists so a wrong account or an unreachable host surfaces
    here instead of in a dead conversation."""
    open_new_chat_dialog(ui)
    select_agent(ui, "ssh")
    _fill_connection(ui, "Nowhere", host="203.0.113.1")  # TEST-NET-3, never routable
    ui.find_element(By.ID, "sshTestButton").click()
    WebDriverWait(ui, 60).until(
        lambda d: d.find_element(By.ID, "sshTestStatus").get_attribute("data-ok") == "false"
    )


# --- the chat list ------------------------------------------------------------------


def _row(driver, target: str):
    return driver.find_element(By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')


def test_a_local_chat_is_marked_as_running_here(ui, ui_server):
    target = api(ui_server, "/api/llm/create-conversation",
                 {"provider": "claude", "name": "Local chat"})["conversation"]["target"]
    reload(ui, ui_server)
    WebDriverWait(ui, 20).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')))
    marker = _row(ui, target).find_element(By.CSS_SELECTOR, ".conversation-location")
    assert "is-local" in marker.get_attribute("class")
    assert "this machine" in marker.get_attribute("title")


def test_an_ssh_chat_is_marked_with_the_machine_it_runs_on(ui, ui_server):
    target = api(ui_server, "/api/llm/create-conversation",
                 {"provider": "ssh", "name": "Remote chat", "workspace": "deploy@example.test"}
                 )["conversation"]["target"]
    reload(ui, ui_server)
    WebDriverWait(ui, 20).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')))
    marker = _row(ui, target).find_element(By.CSS_SELECTOR, ".conversation-location")
    assert "is-ssh" in marker.get_attribute("class")
    assert "deploy@example.test" in marker.get_attribute("title")


def test_deleting_a_chat_removes_its_row(ui, ui_server):
    target = api(ui_server, "/api/llm/create-conversation",
                 {"provider": "claude", "name": "Delete me"})["conversation"]["target"]
    reload(ui, ui_server)
    WebDriverWait(ui, 20).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')))

    assert api(ui_server, "/api/llm/delete-conversation", {"target": target})["ok"] is True
    reload(ui, ui_server)
    WebDriverWait(ui, 10).until(
        lambda d: not d.find_elements(By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')
    )


# --- how a reply is drawn -------------------------------------------------------------


def test_terminal_output_is_a_copyable_block_not_prose(ui, ui_server):
    """Markdown would reflow the runs of spaces that carry the layout, so an SSH
    reply goes into the same code card a fenced block gets — Copy, Wrap and all."""
    target = api(ui_server, "/api/llm/create-conversation",
                 {"provider": "ssh", "name": "Output shape", "workspace": "deploy@example.test"}
                 )["conversation"]["target"]
    listing = "AGENTS.md    libs         README.md\nartifacts    pipelines    run.sh"
    api(ui_server, "/api/llm/test/inject-assistant-message", {"target": target, "text": listing})

    reload(ui, ui_server)
    WebDriverWait(ui, 20).until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')))
    _row(ui, target).click()

    card = WebDriverWait(ui, 15).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, ".terminal-card")))
    # Starts wrapped: a terminal line is routinely wider than a bubble, and a
    # horizontal scrollbar in every reply is worse than a wrapped line.
    assert "wrapped" in card.get_attribute("class")
    labels = [b.text for b in card.find_elements(By.CSS_SELECTOR, "[data-code-action]")]
    assert "Copy" in labels and "Wrap" in labels
    # The spacing that carries the columns has to survive.
    assert "AGENTS.md    libs" in card.find_element(By.TAG_NAME, "code").get_attribute("textContent")


# --- the standalone console -------------------------------------------------------


def test_the_ssh_console_is_served_with_its_two_modes(ui_server, driver, monkeypatch):  # noqa: F811
    """The page that exists because the DevAgents client cannot drive a local
    backend: it pins the production API origin and gates itself behind Firebase."""
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_AUTH_MODE", "token")
    monkeypatch.setenv("SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED", "1")
    driver.get(f"{ui_server}/ssh")
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "host")))
    modes = [o.get_attribute("value") for o in driver.find_elements(By.CSS_SELECTOR, "#mode option")]
    assert modes == ["command", "pty"]
    assert driver.find_element(By.ID, "cmd").is_displayed()


# --- against a real machine ---------------------------------------------------------


def live_host_reachable() -> bool:
    import subprocess

    try:
        probe = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", LIVE_HOST, "true"],
            capture_output=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not live_host_reachable(), reason=f"{LIVE_HOST} is not reachable")
class TestAgainstARealMachine:
    def test_an_ssh_chat_runs_a_command_and_shows_its_output(self, ui, ui_server):
        target = api(ui_server, "/api/llm/create-conversation",
                     {"provider": "ssh", "name": "Live box", "workspace": LIVE_HOST}
                     )["conversation"]["target"]
        reload(ui, ui_server)
        WebDriverWait(ui, 20).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, f'tr.conversation-row[data-target="{target}"]')))
        _row(ui, target).click()

        composer = WebDriverWait(ui, 15).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, ".chat-frame.is-focused textarea, #composer textarea")))
        composer.click()
        composer.send_keys("hostname")
        composer.send_keys("")

        WebDriverWait(ui, 120).until(
            lambda d: LIVE_HOST in d.find_element(
                By.CSS_SELECTOR, ".chat-frame .transcript, #transcript").text
        )
        # And the cursor stays where it was, so the next command can just be typed.
        assert ui.switch_to.active_element == composer
