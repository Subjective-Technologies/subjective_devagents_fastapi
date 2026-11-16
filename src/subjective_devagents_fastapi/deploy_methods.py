"""How each repository deploys, as a runtime fact instead of something agents rediscover.

Production 2026-11-16: Release Agent ran `npm run deploy` for subjective_web_pipelines,
which failed for lack of a Cloudflare token, although the repository's own GitHub Actions
pipeline deploys every push to main. A local deploy of a CI-deployed repository is also
unsafe: a stale local build can overwrite newer production.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEPLOY_COMMAND = re.compile(r'\b(?:(?:npm|pnpm|yarn)\s+(?:run\s+)?deploy|wrangler\s+(?:pages\s+)?deploy|'
                            r'opennextjs-cloudflare\s+deploy)\b')
_DEPLOY_STEP = re.compile(r'(?i)wrangler[^\n]*deploy|pages\s+deploy|opennextjs-cloudflare\s+deploy|'
                          r'npm\s+run\s+deploy|cloudflare/wrangler-action|^\s*(?:-\s*)?name:\s*deploy\b', re.M)
_ABS_PATH = re.compile(r'(/(?:[\w.@-]+/){1,8}[\w.@-]+)')


def repository_root(path):
    # As written (not symlink-resolved): agents and leases use the workspace's own spelling.
    current = Path(os.path.abspath(str(path)))
    try:
        current = current if current.is_dir() else current.parent
    except OSError:
        return None
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    return None


def ci_deploy(repo):
    """'CI on push to main (.github/workflows/x.yml)' when a workflow deploys pushes to main."""
    folder = Path(repo) / '.github' / 'workflows'
    try:
        files = sorted(folder.glob('*.y*ml'))
    except OSError:
        return None
    for workflow in files:
        try:
            text = workflow.read_text(errors='replace')[:200_000]
        except OSError:
            continue
        on_push = re.search(r'(?m)^\s*push\s*:', text) or re.search(r'(?m)^on\s*:\s*\[?[^\n]*\bpush\b', text)
        if on_push and re.search(r'\bmain\b|\bmaster\b', text) and _DEPLOY_STEP.search(text):
            return f'CI deploys on push to main ({workflow.relative_to(Path(repo))})'
    return None


def manual_deploy(repo):
    package = Path(repo) / 'package.json'
    try:
        text = package.read_text(errors='replace')
    except OSError:
        return None
    if '"deploy"' in text:
        credential = 'present' if os.environ.get('CLOUDFLARE_API_TOKEN') else 'absent on this runtime'
        return f'package.json has a deploy script (Cloudflare credentials {credential})'
    return None


def repositories_in(text, limit=8):
    roots = []
    for match in _ABS_PATH.finditer(text or ''):
        root = repository_root(match.group(1))
        if root and str(root) != '/' and root not in roots:
            roots.append(root)
        if len(roots) >= limit:
            break
    return roots


def facts_block(text):
    """Deploy facts for the repositories an assignment names, for the agent's prompt."""
    lines = []
    for repo in repositories_in(text):
        method = ci_deploy(repo) or manual_deploy(repo)
        if method:
            lines.append(f'- {repo}: {method}')
    if not lines:
        return ''
    return ('\n\nRUNTIME FACTS (detected from the repositories; authoritative):\nDeploy methods:\n' + '\n'.join(lines)
            + '\nFor a repository deployed by CI, publishing (push to main) starts the deploy: verify the live '
              'build instead of deploying from this host.')


def refuse_local_deploy(command, workdir):
    """ModelError when `command` deploys a CI-deployed repository from this host."""
    from .language_models.contracts import ModelError
    if not DEPLOY_COMMAND.search(command or '') or os.environ.get('SUBJECTIVE_ALLOW_LOCAL_DEPLOY') == '1':
        return
    repo = repository_root(workdir)
    method = ci_deploy(repo) if repo else None
    if method:
        raise ModelError('DEPLOY_VIA_CI', f'{repo} is deployed by CI ({method}). A local deploy can overwrite a newer '
                         'production build. Push the verified commit to main and verify the live build instead.')
