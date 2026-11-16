"""Runtime operations that wasted agent time in production (2026-11-16).

* Deploy restarts cancelled running agent turns (3 turns lost, one after 35 minutes).
* Release Agent ran `npm run deploy` without a Cloudflare token for a repository that its
  own CI deploys on every push to main.
"""
import subprocess

import pytest

from subjective_devagents_fastapi import deploy_methods as dm
from subjective_devagents_fastapi.language_models.contracts import ModelError
from test_model_api import Waiting, engine  # noqa: F401
from test_channel_coordination import bound  # noqa: F401

PIPELINE = """name: Pipeline
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: self-hosted
  deploy:
    if: github.ref == 'refs/heads/main'
    steps:
      - run: npx wrangler pages deploy dist --project-name x
"""


def repo(tmp_path, name, workflow=None, deploy_script=False):
    folder = tmp_path / name
    folder.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=folder, check=True)
    if workflow:
        (folder / '.github' / 'workflows').mkdir(parents=True)
        (folder / '.github' / 'workflows' / 'pipeline.yml').write_text(workflow)
    if deploy_script:
        (folder / 'package.json').write_text('{"scripts": {"deploy": "wrangler deploy"}}')
    return folder


def test_ci_deployed_repository_is_detected_and_named_in_the_prompt(tmp_path):
    ci = repo(tmp_path, 'pipelines', PIPELINE, deploy_script=True)
    manual = repo(tmp_path, 'tool', deploy_script=True)
    plain = repo(tmp_path, 'lib', 'on: [pull_request]\njobs: {}\n')
    assert dm.ci_deploy(ci) == 'CI deploys on push to main (.github/workflows/pipeline.yml)'
    assert dm.ci_deploy(manual) is None and dm.ci_deploy(plain) is None
    assert 'deploy script' in dm.manual_deploy(manual)
    block = dm.facts_block(f'Work in {ci}/src and {manual}/src/app.js, not {plain}')
    assert f'{ci}: CI deploys on push to main' in block and f'{manual}: package.json has a deploy script' in block
    assert str(plain) not in block


@pytest.mark.parametrize('command', ['npm run deploy', 'npx wrangler pages deploy dist', 'opennextjs-cloudflare deploy',
                                     'npm run build && npm run deploy'])
def test_local_deploy_of_a_ci_deployed_repository_is_refused(tmp_path, command):
    ci = repo(tmp_path, 'pipelines', PIPELINE, deploy_script=True)
    with pytest.raises(ModelError) as error:
        dm.refuse_local_deploy(command, str(ci))
    assert error.value.code == 'DEPLOY_VIA_CI' and 'Push the verified commit' in str(error.value)


def test_local_deploy_stays_allowed_where_ci_does_not_deploy(tmp_path, monkeypatch):
    manual = repo(tmp_path, 'tool', deploy_script=True)
    dm.refuse_local_deploy('npm run deploy', str(manual))
    ci = repo(tmp_path, 'pipelines', PIPELINE)
    dm.refuse_local_deploy('npm run build', str(ci))
    monkeypatch.setenv('SUBJECTIVE_ALLOW_LOCAL_DEPLOY', '1')
    dm.refuse_local_deploy('npm run deploy', str(ci))


def test_guard_refuses_local_deploy_before_any_lease_check(bound, tmp_path):
    session, _, identity = bound
    ci = repo(tmp_path / 'workspace', 'pipelines', PIPELINE)
    with pytest.raises(ModelError) as error:
        session.guard('Bash', {'command': f'cd {ci} && npm run deploy'})
    assert error.value.code == 'DEPLOY_VIA_CI'


def test_draining_runtime_refuses_new_work_and_a_restart_clears_it(engine):
    waiting = Waiting()
    engine.strategies['cli:claude'] = waiting
    conversation = engine.create('claude-a')
    engine.submit(conversation['id'], 'wait', 'one')
    assert waiting.started.wait(2)
    engine.drain_flag.touch()
    capacity = engine.capacity()
    assert capacity['draining'] and capacity['available'] == 0 and capacity['busy'] == 1
    engine.drain_flag.unlink()
    assert engine.capacity()['available'] == engine.max_workers - 1
    engine.drain_flag.touch()
    from subjective_devagents_fastapi.language_models.model import LanguageModel
    fresh = LanguageModel(engine.store, engine.identities, {}, max_workers=2)
    assert not fresh.drain_flag.exists() and not fresh.capacity()['draining']
    fresh.close()
    engine.close()


def test_drain_restart_script_is_valid_shell():
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'drain_restart.sh'
    assert subprocess.run(['bash', '-n', str(script)]).returncode == 0
    text = script.read_text()
    assert 'DRAINING' in text and 'systemctl restart' in text and 'DRAIN_MAX:-1500' in text
