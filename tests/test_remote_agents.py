"""Running an agent CLI on another machine.

The CLI datasources were two things wound together: knowing how to invoke Claude
or Grok and parse what they say, and knowing how to run a process. Only the second
has to change for the agent to run elsewhere, so it was separated into a
CommandRunner the datasource is handed. These tests cover the wiring that chooses
which runner a turn gets, and the guarantee that nothing changes when it gets the
ordinary one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_ROOT / "src"))
# The abstract package is a vendored submodule, not an installed dependency of the
# test environment; datasources._prepare_sys_path does this at runtime.
for dependency in ("subjective-abstract-data-source-package",
                   "brainboost_configuration_package",
                   "brainboost_data_source_logger_package"):
    sys.path.insert(0, str(SERVICE_ROOT / "libs" / "dependencies" / dependency))

from subjective_devagents_fastapi import datasources  # noqa: E402
from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBJECTIVE_CHAT_HOME", str(tmp_path / "home"))
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return LlmService(root=str(tmp_path / "workspace"), app_home=str(tmp_path / "home"))


# --- the runner a datasource is handed --------------------------------------------


def test_a_datasource_runs_locally_by_default():
    """Nothing changes for a chat that never mentions SSH."""
    from subjective_abstract_data_source_package import LocalRunner, runner_from

    assert isinstance(runner_from({}), LocalRunner)
    assert isinstance(runner_from(None), LocalRunner)
    assert runner_from({}).is_remote is False


def test_a_supplied_runner_is_used_as_is():
    """How the service injects SSH without any CLI datasource importing SSH code."""
    from subjective_abstract_data_source_package import CommandRunner, runner_from

    class Elsewhere(CommandRunner):
        is_remote = True
        def popen(self, argv, *, env=None, cwd=""): raise NotImplementedError
        def run(self, argv, *, timeout, env=None, cwd=""): raise NotImplementedError
        def which(self, name, candidates=()): return None
        def is_file(self, path): return False
        def upload(self, local_path): return local_path

    supplied = Elsewhere()
    assert runner_from({"command_runner": supplied}) is supplied


def test_the_local_runner_behaves_as_subprocess_did():
    from subjective_abstract_data_source_package import LocalRunner

    runner = LocalRunner()
    assert runner.run(["echo", "hello"], timeout=10).stdout.strip() == "hello"
    assert runner.which("sh", ()).endswith("/sh")
    assert runner.which("no-such-binary-anywhere", ()) is None
    assert runner.is_file("/etc/hostname") is True
    # Locally an attachment needs no copying, so the path is unchanged.
    assert runner.upload("/tmp/x") == "/tmp/x"


def test_the_claude_datasource_takes_the_runner_it_is_given():
    """The conversion under test: no subprocess call of its own on the turn path."""
    report = datasources.available()
    if not report["claude"]["loaded"]:
        pytest.skip("claude submodule is not checked out")
    from subjective_abstract_data_source_package import LocalRunner

    instance = datasources.instantiate(datasources.load("claude"), {})
    assert isinstance(instance._runner, LocalRunner)


# --- which machine a chat is bound to ------------------------------------------------


def test_a_chat_remembers_the_machine_it_runs_on(service):
    """Stored rather than resolved, so the connection can change without the chat
    having to be recreated."""
    conversation = service.create_conversation(
        provider="claude", name="Remote", workspace="/srv/app", ssh_connection="abc123",
    )
    assert conversation["sshConnection"] == "abc123"
    # And it survives the normalising that every load and save goes through — a key
    # missing from there is silently dropped, which is how this first shipped.
    reloaded = service.find_conversation(conversation["target"])
    assert reloaded["sshConnection"] == "abc123"


def test_an_ordinary_chat_is_bound_to_no_machine(service):
    conversation = service.create_conversation(provider="claude", name="Local")
    assert conversation["sshConnection"] is None


def test_the_runner_reaches_the_datasource(service, monkeypatch):
    """End of the wiring: what connection_for passes is what the datasource gets."""
    seen = {}

    def fake_run(provider, payload, connection=None, on_instance=None):
        seen["connection"] = connection
        return {"success": True, "status": "completed", "response": "done"}

    monkeypatch.setattr(datasources, "run", fake_run)
    monkeypatch.setattr(datasources, "supports_chat", lambda cls: True)
    monkeypatch.setattr(datasources, "load", lambda provider: object)
    monkeypatch.setattr(datasources, "instantiate", lambda cls, connection=None: object())

    sentinel = object()
    monkeypatch.setattr(datasources, "ssh_runner_for", lambda home, identifier: sentinel)
    conversation = service.create_conversation(
        provider="claude", name="Remote", ssh_connection="abc123",
    )
    service.send_message(conversation["target"], "hello")
    assert seen["connection"]["command_runner"] is sentinel


def test_a_local_chat_is_given_no_runner(service, monkeypatch):
    seen = {}
    monkeypatch.setattr(datasources, "run", lambda p, payload, connection=None, on_instance=None:
                        (seen.update(connection=connection),
                         {"success": True, "status": "completed", "response": "done"})[1])
    monkeypatch.setattr(datasources, "supports_chat", lambda cls: True)
    monkeypatch.setattr(datasources, "load", lambda provider: object)
    monkeypatch.setattr(datasources, "instantiate", lambda cls, connection=None: object())

    conversation = service.create_conversation(provider="claude", name="Local")
    service.send_message(conversation["target"], "hello")
    assert "command_runner" not in seen["connection"]


def test_an_unreachable_machine_fails_rather_than_running_here(service, monkeypatch):
    """The dangerous failure: an agent silently running on the wrong computer."""
    def unavailable(home, identifier):
        raise LookupError("The SSH terminal is disabled.")

    monkeypatch.setattr(datasources, "ssh_runner_for", unavailable)
    monkeypatch.setattr(datasources, "supports_chat", lambda cls: True)
    monkeypatch.setattr(datasources, "load", lambda provider: object)
    ran = []
    monkeypatch.setattr(datasources, "run", lambda *args, **kwargs: ran.append(args))

    conversation = service.create_conversation(
        provider="claude", name="Remote", ssh_connection="gone",
    )
    outcome = service.send_message(conversation["target"], "hello")
    assert outcome["ok"] is False
    assert outcome["status"] == "configuration_required"
    assert "another machine" in outcome["error"]
    assert ran == []
