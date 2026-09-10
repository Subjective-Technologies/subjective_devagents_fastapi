# DevAgents production recovery and verification

The coding server `subjective-agents` uses a **system** unit, not the workstation's
user-unit template. Verified 2026-09-10:

- Unit: `subjective-devagents-fastapi.service`
- Run user: `subjective`, home `/home/subjective`
- Backend checkout: `/subjective/services/subjective_devagents_fastapi`
- Frontend and fallback bundles: `/subjective/clients/desktop/subjective_chat`
- API: `https://devagents-api.subjectivetechnologies.com:8080`
- Operator SSH account: `worktwins`; GitHub access works as `subjective`.

Grok's Linux sandbox requires the OS package `bubblewrap` (`bwrap`). Keep the
sandbox enabled; install its dependency rather than disabling enforcement. A
provider status of `ready` can still be followed by a billing failure on a real
turn; Grok returned HTTP 402 for exhausted usage balance during this recovery.

## Inspect before updating

```bash
systemctl show subjective-devagents-fastapi.service \
  -p MainPID -p User -p WorkingDirectory -p FragmentPath
sudo -u subjective -H git -C /subjective/services/subjective_devagents_fastapi status --short
sudo -u subjective -H git -C /subjective/services/subjective_devagents_fastapi submodule status
journalctl -u subjective-devagents-fastapi.service -n 100 --no-pager
```

Do not reset the parent `/subjective` repository to repair this service. Inspect
and preserve edits in each submodule too: a parent stash does not save their edits.
Tracked `__pycache__` changes can account for `-dirty`; inspect the actual diff.
Use `git pull --ff-only` and `git submodule update --init --recursive` as the run
user only after the intended code and submodule revisions have been published and
server-local edits have been preserved. A divergent checkout needs reconciliation,
not an automatic `reset --hard`.

Restart the exact system unit after updating Python sources or plugin pins:

```bash
sudo systemctl restart subjective-devagents-fastapi.service
systemctl is-active subjective-devagents-fastapi.service
curl -fsS -D - https://devagents-api.subjectivetechnologies.com:8080/healthz
```

`X-DevAgents-Version` identifies the **frontend checkout**. It must not be compared
to a backend commit. `X-DevAgents-Backend-Version` and `/api/version.backendVersion`
identify the backend checkout loaded at startup. A health response proves liveness,
not that a provider can send a message.

## Required live send gate

Run from a directory the `subjective` user can access. Name every provider required
for this deployment. A selected provider that is logged out or missing must fail,
not silently skip. Supply `SUBJECTIVE_DEVAGENTS_API_TOKEN` securely when enabled.

```bash
cd /subjective/services/subjective_devagents_fastapi
sudo -u subjective -H env \
  SUBJECTIVE_DEVAGENTS_LIVE_URL=https://devagents-api.subjectivetechnologies.com:8080 \
  SUBJECTIVE_DEVAGENTS_LIVE_PROVIDERS=claude,codex,cursor,grok \
  .venv/bin/python -m pytest tests/test_quick_reply_live.py -q --tb=short
```

This creates labelled `[smoke]` conversations and uses real provider turns. It
checks HTTP acceptance, an unpredictable marker in the assistant reply, WebSocket
busy/reply/completion events, and persisted transcript/session state. Claude also
resumes with an image and must recall the previous marker. No send function or CLI
is mocked. The image deliberately exercises `--add-dir`, whose variadic argument
parser consumed the prompt before the `--` terminator fix.

The manual GitHub Actions workflow `Live chat deployment gate` runs the same test
from outside the server. Configure environment `devagents-live`, variable
`DEVAGENTS_LIVE_URL`, and secret `DEVAGENTS_API_TOKEN`. Run after each deployment;
normal credential-free unit CI is not a substitute. No recurring billing schedule
is installed by this workflow.

## Attachment failure diagnosis

Claude declares `--add-dir <directories...>` and `--allowedTools <tools...>`.
Appending the prompt immediately after either option can consume it as another
option value. An image send then reports an empty prompt; an image resume can
instead report "No deferred tool marker found". The correct command ends with
`--`, then the complete prompt as one argument. Keep the pinned datasource,
frontend vendored copy, and exported Claude bundle consistent.

API authentication is separate from provider login. Before enabling the shared
token on an existing deployment, seed authorized browsers using the frontend's
token flow and configure the live-test secret; never embed the token in public
HTML or commit it. Dashboard login alone does not protect the public API listener.
