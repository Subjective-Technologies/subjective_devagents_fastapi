import pytest
from subjective_devagents_fastapi.activity_preview import safe_tool_detail

@pytest.mark.parametrize('command', [
    'redis-server --requirepass fixture7', 'kinit alice <<< fixture7',
    "pwsh -command 'Write-Output fixture7'",
    'curl https://example.com/fixture7', 'python script.py fixture7',
    'redis-cli -a fixture7 ping', 'sshpass -p fixture7 ssh host',
    'psql postgresql://alice:fixture7@db/app',
    "curl '-u' alice:fixture7 https://example.com",
    'curl -ualice:fixture7 https://example.com',
    'curl -\"\"u alice:fixture7 https://example.com',
    "python -c 'print(123)'",
    'cat /subjective/.env', 'curl -H "Authorization: Bearer fixture" /api',
    'curl -u admin:shortpass /api', 'MY_CONFIG=opaque-value npm run deploy',
    'echo ghp_abcdefghijklmnopqrstuvwxyz', 'echo -----BEGIN PRIVATE KEY-----',
    'printenv', 'curl https://user:pass@example.com', 'curl --data "sensitive payload" /api',
    'echo ' + 'x' * 40, 'echo ok; cat /subjective/.env' + 'x' * 600,
])
def test_sensitive_commands_hidden(command):
    preview = safe_tool_detail(command)
    assert preview != command and 'hidden' in preview
    assert 'fixture7' not in preview and 'shortpass' not in preview and 'alice:' not in preview

def test_known_environment_secret_and_bounded_preview(monkeypatch):
    monkeypatch.setenv('TEST_API_KEY', 'short-private-fixture')
    assert safe_tool_detail('echo short-private-fixture') == '[details hidden: sensitive data]'
    assert safe_tool_detail('npm test -- --run') == 'npm test -- --run'
    assert safe_tool_detail('cd /subjective/services/subjective_devagents_fastapi && pytest -q') == 'cd /subjective/services/subjective_devagents_fastapi && pytest -q'
    assert safe_tool_detail('pwd\nls') == 'Shell command [arguments hidden]'
    assert len(safe_tool_detail('ls ' + './src/abc.py ' * 100)) == 500
    assert safe_tool_detail(None) is None


def test_file_paths_are_details_not_shell_commands():
    assert safe_tool_detail('/subjective/services/example/src/main.py', command=False) == '/subjective/services/example/src/main.py'
    assert safe_tool_detail('cd /subjective/services/example && npm run build') == 'cd /subjective/services/example && npm run build'
    assert safe_tool_detail('git status --short') == 'git status --short'
    assert safe_tool_detail('python -m pytest -q tests/test_main.py') == 'python -m pytest -q tests/test_main.py'
