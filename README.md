# Subjective DevAgents FastAPI

Backend for **Subjective DevAgents** ("My Agents"). This service exists so the DevAgents
client can be **web only**: the browser holds the UI, and every piece of backend logic —
agent CLI orchestration, conversation history, `su` sessions — lives here in Python.

It is a port of the Node backend that used to live inside the client at
`clients/desktop/subjective_devagents/src`. Route paths, request bodies, response shapes
and the `/ws` event stream are **identical** to the old `webServer.js`, so the existing
browser UI runs against this unchanged — no frontend edits were needed.

## LanguageModel replacement engine

The `/api/language-model` API and existing `/api/llm/*` routes share one execution
engine. It provides independent CLI/API strategies, isolated accounts, local/SSH
execution, durable turns, and explicit account/provider handoffs. See
[LANGUAGE_MODEL.md](LANGUAGE_MODEL.md) for configuration and capabilities.

## Checkout

The datasources and shared packages are git submodules, laid out the same way as the
monorepo (`plugins/` for datasource packages, `libs/dependencies/` for shared ones):

```bash
git clone git@github.com:Subjective-Technologies/subjective_devagents_fastapi.git
cd subjective_devagents_fastapi
git submodule update --init --recursive
```

| Submodule | Purpose |
| --- | --- |
| `plugins/subjective_codex_datasource` | Codex CLI turns |
| `plugins/subjective_claude_cli_datasource` | Claude Code CLI turns |
| `plugins/subjective_localchats_datasource` | Imports existing chats across every user and IDE on the machine |
| `libs/dependencies/subjective-abstract-data-source-package` | `SubjectiveDataSource` base class |
| `libs/dependencies/brainboost_configuration_package` | Config the datasources import |
| `libs/dependencies/brainboost_data_source_logger_package` | Logger the datasources import |

## Standalone execution

Legacy datasource and exported-pipeline execution is no longer part of send paths.
Provider drivers own CLI/API command grammar and parsing; transports own local and SSH
processes. `GET /api/datasources` remains as a compatibility diagnostic and reports the
standalone backend. Shell and terminal SSH paths use the service-owned SSH runtime.

### ⚠️ The pinned submodules are behind the code in production

The datasource repos on GitHub are older than the vendored copies running today in
`clients/desktop/subjective_devagents/plugins/`:

| Datasource | GitHub HEAD | Vendored (running) |
| --- | --- | --- |
| `subjective_codex_datasource` | 417 lines, last pushed 2026-03-18, **v1 plugin API** | 832 lines, v2 API |
| `subjective_localchats_datasource` | 1383 lines, 2026-07-14 | 1579 lines |
| `subjective_claude_cli_datasource` | 1035 lines, 2026-08-11 | 1045 lines |

The adapter bridges the v1/v2 constructor split so the stale pins still load, but this is a
compatibility shim, not parity — the vendored copies carry six months of unpublished work.
**Publishing the vendored code to those repos and re-pinning is the real fix.** Until then,
point `SUBJECTIVE_DEVAGENTS_PLUGIN_PATH` at the client's `plugins/` directory to run what
production runs.

## Run

Production recovery, system-unit discovery, backend version checks, and the real
send deployment gate are documented in [deploy/README.md](deploy/README.md).

```bash
./run_subjective_devagents_fastapi.sh          # from the monorepo root
./run.sh                                       # from this repo
```

Binds `0.0.0.0:8080` so it does not collide with the web UI (`:80` / `:8787`).
Override with `SUBJECTIVE_CHAT_WEB_HOST` / `SUBJECTIVE_CHAT_WEB_PORT`.

or directly:

```bash
uvicorn subjective_devagents_fastapi.app:app --host 0.0.0.0 --port 8080
```

## What moved where

| Node (client, before) | Python (this service) |
| --- | --- |
| `src/webServer.js` (HTTP + `ws`) | `app.py` (FastAPI + WebSocket) |
| `src/chatCore.js` (IRC, DCC, LLM orchestration, Continue-as) | `chat_core.py` |
| `src/llmService.js` (history archive, sends, pipeline spawn) | `llm_service.py` |
| `src/suSessionManager.js` (`su` pty, provider login, bridge) | `su_session_manager.py` |
| `src/systemUsers.js` | `system_users.py` |
| `src/providerSessions.js` | `provider_sessions.py` |
| `src/cursorDomInject.js` (CDP DOM inject) | `cursor_inject.py` |
| `src/pipeline_runtime.py`, `pipeline_bridge_server.py`, `pty_fallback.py` | copied verbatim |

The Electron desktop app (`src/main.js`) still uses the Node modules directly, so they are
left in place. `src/webServer.js` is superseded by this service.

## Compatible on-disk state

This service reads and writes the **same files** as the Node backend, in the same formats,
so switching backends does not lose or rewrite anything:

- `$SUBJECTIVE_CHAT_HOME/llm-conversations.json` — state `version: 2`, same conversation
  records, `activeTargets` and `labelOverrides`.
- `$SUBJECTIVE_CHAT_HOME/settings.json` — same `{active, bookmarks}` shape.
- `$SUBJECTIVE_CHAT_HOME/.subjective_chat_history/context/` — same manifest and export files.
- Conversation `target` strings and generated IRC channel names are byte-identical
  (`tests/test_parity.py` asserts this against the live Node code), so existing chats keep
  their identity and channels.

## Environment

| Variable | Meaning |
| --- | --- |
| `SUBJECTIVE_CHAT_WEB_HOST` / `SUBJECTIVE_CHAT_WEB_PORT` | Bind address (launcher; default `0.0.0.0:8080`, away from the web UI) |
| `SUBJECTIVE_CHAT_HOME` | Application home holding conversation state (default `~/.subjective_chat`) |
| `SUBJECTIVE_DEVAGENTS_CLIENT_ROOT` | Where the browser UI files and `pipelines/` bundles live (default: the DevAgents client in this repo) |
| `SUBJECTIVE_CHAT_PYTHON` | Python used to run the exported CLI pipeline bundles |
| `SUBJECTIVE_CHAT_HISTORY_REFRESH_MS` | History rescan interval; `0` disables (default 10 min) |
| `SUBJECTIVE_CHAT_IRC` | `1` enables the optional IRC bridge (off by default) |
| `SUBJECTIVE_DEVAGENTS_CORS_ORIGINS` | Comma-separated origins when the UI is hosted separately (default `*`) |
| `SUBJECTIVE_DEVAGENTS_API_TOKEN` | **Set this.** Shared secret required on `/api/*`, `/ws`, `/ws/ssh`, `/docs`, `/openapi.json`. `run.sh` reads it from the monorepo's `.env` when the environment does not already set one. Unset leaves the API open to anonymous internet callers (logged loudly at startup). Seed a browser once with `?token=...` |
| `SUBJECTIVE_DEVAGENTS_EXEC_ENABLED` | `1` enables `POST /api/exec` (run a shell command on this host). Off by default, and **refused** unless `SUBJECTIVE_DEVAGENTS_API_TOKEN` is also set |
| `SUBJECTIVE_DEVAGENTS_EXEC_CWD` | Default working directory for `/api/exec` (default: `$HOME`) |
| `SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED` | `1` enables `/ws/ssh` (a live shell on another machine). Off by default, and **refused** unless `SUBJECTIVE_DEVAGENTS_API_TOKEN` is also set |
| `SUBJECTIVE_DEVAGENTS_SSH_MAX_SESSIONS` | Concurrent `/ws/ssh` sessions allowed (default 8) |
| `SUBJECTIVE_DEVAGENTS_SSH_IDLE_TIMEOUT` | Seconds an idle `/ws/ssh` session is kept before closing (default 1800) |
| `SUBJECTIVE_DEVAGENTS_PUBLIC_API` | Origin the served page should call. Empty = same-origin (this process). Unset leaves the checkout's default. Rewrites the page's `devagents-api` meta tag **and** its CSP `connect-src` at serve time |
| `CURSOR_CDP_PORT` | Cursor's remote-debugging port for DOM injection (default 9222) |

## Serving the UI

The Hetzner API host does **not** serve the browser UI (`SUBJECTIVE_DEVAGENTS_SERVE_UI=0`).
Web stays at the original endpoint: `https://devagents.subjectivetechnologies.com/`.
This process is the API: `/docs`, `/openapi.json`, `/api/*`, `/ws`.

`GET /` returns a JSON index pointing at `/docs` and the web URL.

To serve HTML from this process anyway, set `SUBJECTIVE_DEVAGENTS_SERVE_UI=1`. Files then
come from `$SUBJECTIVE_DEVAGENTS_CLIENT_ROOT/web` and `/src` under a fixed allowlist, plus
`/assets/**` with a resolved-path containment check.

When the UI is hosted separately, set `SUBJECTIVE_DEVAGENTS_CORS_ORIGINS` to that origin
(default `*`). The API surface is origin-independent.

HTML responses carry `Content-Security-Policy: frame-ancestors …` so the dashboard's
**My Agents** page can iframe this app across origins — a zone-wide Cloudflare rule stamps
`X-Frame-Options: SAMEORIGIN` on every HTML response, and `frame-ancestors` overrides it.

## Concurrency model

Every blocking operation — CLI subprocesses, `su` ptys, history scans, IRC socket I/O —
runs on a worker thread (`asyncio.to_thread`) or its own daemon thread, so a 30-minute
agent turn never stalls the event loop or another request.

`POST /api/llm/send-message` returns `{ok: true, pending: true}` immediately and delivers
the reply over the WebSocket (`llm-message` / `llm-busy` / `system`). Cloudflare and most
reverse proxies close idle HTTP at ~100s with a 524, while agent turns are allowed 30
minutes, so the answer cannot ride the request that started it.

## SSH terminal (`/ws/ssh`)

A WebSocket that opens a live shell on another machine through
`subjective_ssh_datasource`. One session per socket; shell state persists across
commands and output streams back as the remote writes it.

```
client → {"type":"auth","token":"…","user_id":"…","company_name":"…"}
client → {"type":"open","host":"subjective-agents"}
server → {"type":"hello"|"opening"|"ready", …}
client → {"type":"input","command":"ls -la"}
server → {"type":"output","stream":"stdout","data":"…"}  … then
server → {"type":"exit","code":0,"cwd":"/tmp"}
client → {"type":"interrupt"|"close"}
server → {"type":"closed","code":0} | {"type":"error","error":"…"}
```

`host` may be a `Host` alias from the service account's `~/.ssh/config`, in which case
no credential passes through this service at all. Keys and passwords may be supplied
per session instead; they never appear in argv, logs, the `opening` frame or any
result. Only the fields in `ssh_terminal.ALLOWED_CONNECTION_KEYS` are taken from the
browser — notably not `ssh_path`.

`GET /api/ssh/sessions` lists what is open, without credentials.

### Saved connections

`GET /api/ssh/connections` returns three things the picker shows together: connections
saved in `<app home>/ssh-connections.json`, private keys found in the service
account's `~/.ssh` (by path — contents are never read), and `Host` aliases from its
`~/.ssh/config`. Aliases are listed, never copied: an alias already carries its
account, port, key and options, and duplicating it here would be a second copy to
keep in step.

`POST /api/ssh/connections`, `.../update`, `.../delete` manage the saved ones, and
`.../test` opens a status check without starting a chat.

`POST /api/ssh/keys` takes a private key from the browser and writes it `0600` into
`<app home>/ssh-keys/`, after which it appears in the same picker and is used exactly
like one already in `~/.ssh` — by path, never read back. This is what makes a
deployed instance usable at all: the server's `~/.ssh` holds the *box's* keys, not
the operator's. The upload is refused unless it starts with a private-key header, so
the directory cannot become a general file drop and the commonest mistake (sending
the `.pub`) is caught with that advice. `POST /api/ssh/keys/delete` removes one, and
resolves the path against the upload directory so it can never reach a key in
`~/.ssh`.

A saved password or key passphrase is a plaintext secret on this server's disk. The
store is written `0600` atomically, `get()` is the only path that returns a secret
and is not reachable from the API, and every listing reports `hasPassword` /
`hasKeyPassphrase` booleans instead of the values. Agent and key authentication
store no secret at all and are the default.

The console at `/ssh` drives this socket and is the only way to exercise the service
on its own: the DevAgents client pins the production API origin in a meta tag and a
CSP and gates itself behind Firebase. Open it with the token already filled in —
`run.sh` prints this line at startup:

```bash
xdg-open "http://127.0.0.1:8090/ssh?token=$(grep -m1 '^SUBJECTIVE_DEVAGENTS_API_TOKEN=' /subjective/.env | cut -d= -f2-)"
```

A Firebase sign-in also authenticates, in every mode, when the caller sends the
user id, the token, and this server's company name. The token subject must be
that user id. The token expires in an hour and cannot be revoked before it
does, so use the static token locally.

The remote shell is closed when the socket closes, including when the browser tab
disappears: the endpoint's cleanup does no `await` before unregistering and closing,
because a client disconnect cancels the coroutine and the first `await` in a `finally`
would raise straight past it.

Not a terminal emulator — no pty, so `top`, `vim` and `sudo` prompts do not work, and
`interrupt` tears the connection down rather than sending SIGINT to the remote job.

## Security notes

- `/ws/ssh` is remote code execution on a **third** machine. It is off unless
  `SUBJECTIVE_DEVAGENTS_SSH_TERMINAL_ENABLED=1`, and refuses to switch on without an
  API token configured.
- The listener is **unauthenticated**; owner-only access is enforced upstream in the
  dashboard. Do not expose it publicly.
- Nothing here ever collects or types an OS password. `su` ptys are driven interactively by
  whoever is at the browser, through the terminal panel.
- `provider_sessions.py` only checks that credential files *exist* — it never reads token
  contents.
- The `su` bridge socket lives in `/tmp` with a random suffix; access is gated by a
  per-session shared secret, not filesystem permissions.

## The `devagents` D1 database

`schema/devagents.sql` defines a Cloudflare D1 database that records every conversation we
hold with any LLM under one roof, whether it was answered by a subscription seat (a CLI
logged in as an OS user) or a metered API key. A chat looks the same here regardless of
which vendor served it or how it was paid for.

The database id is `SUBJECTIVE_DEVAGENTS_D1_DATABASE_ID` on the host that owns
that Cloudflare account. It is not stored in this repository. The binding name
in `wrangler.toml` is `devagents`.

Re-applying either file is safe and expected: the schema is entirely `IF NOT EXISTS` and
the seed is entirely upserts, so this is also how you refresh prices or add a provider.

```bash
npx wrangler d1 execute devagents --remote --file schema/devagents.sql
npx wrangler d1 execute devagents --remote --file seeds/devagents_seed.sql
```

### One D1 constraint worth knowing

D1 builds SQLite with `SQLITE_MAX_COMPOUND_SELECT = 5`, so a view with six `UNION ALL`
branches is rejected at deploy time with `too many terms in compound SELECT` — and plain
SQLite, which allows 500, will happily run it locally first. There are six element types,
so `v_turn_timeline` is split: `v_turn_timeline_base` carries five branches and the public
view adds the sixth, since referencing a view costs one term instead of replaying its
branches. `test_no_view_exceeds_d1s_compound_select_limit` counts branches so this is
caught before a deploy rather than during one.

### Two levels of conversation id

`conversations.id` is **our** global id and never changes. `conversation_sessions` holds
the provider's **own** id (`provider_session_id`) — a codex/claude session uuid, a Cursor
composer id, an OpenAI response id — one row per provider-side session.

A conversation collects several of them because a session cannot always be carried
forward: "Continue as \<user\>" starts a fresh one (each CLI's session store lives in that
user's `$HOME`), and switching vendor starts another (`start_reason` records which). The
user keeps one chat; the vendors each see their own.

### Tables

| Table | Holds |
| --- | --- |
| `providers` | Vendors and CLIs we can route to, and which billing modes each allows |
| `models` | Model catalog with per-million-token prices used to compute cost |
| `provider_accounts` | A subscription seat or an API key — never the secret, only where it lives |
| `account_usage_windows` | Rolling plan allowance (a 5-hour cap), which token counts cannot express |
| `conversations` | The global conversation id, title, workspace, owner, rollups |
| `conversation_sessions` | The provider's own session id, model, account, and why it was started |
| `turns` | One user request and the whole agent loop answering it |
| `messages` | The chat text: what the user sent, what the LLM replied |
| `thoughts` | The reasoning steps reported before the response |
| `tool_calls` | Shell / file / web / MCP invocations made during a turn |
| `document_accesses` | Every document the chat read or wrote, indexed by path |
| `authorization_requests` | "May I do this?" and the human's decision, with scope |
| `user_prompts` | Questions the chat asked the human, and the answer |
| `attachments` | Images and files carried in or produced |
| `usage_records` | Tokens and cost **per model call** — a turn makes many |
| `import_sources` | Local transcript → conversation map, with the importer's fingerprint |

### Why the elements are separate tables

Every element carries `(turn_id, seq)`, so `v_turn_timeline` unions them into one ordered
stream — thinking, then the tool it ran, then the file it read, then the approval it asked
for, then the reply. That is why there is no separate events table shadowing them: nothing
can drift out of sync with itself.

Approvals and questions get their own tables rather than a flag on the tool call because
their decision arrives **later** than the request, and a turn legitimately parks in
`awaiting_input` / `awaiting_authorization` until a human answers. `v_pending_human_actions`
is the resulting inbox across every conversation. Document accesses are separate so "what
has any agent read under this path?" is one indexed query rather than a scan of tool
arguments — the question you actually ask when auditing an agent.

### Token accounting

`usage_records` is per **model call**, not per turn, because an agentic turn makes many
round-trips; `turns` carries the rollup. `billing_mode` is what unifies the two worlds: a
subscription call still reports tokens (worth tracking against the plan window) but bills
nothing per token, so only `api` rows move money. `v_conversation_costs` reports both
side by side. Cost is snapshotted into the row at write time along with
`price_snapshot_json`, so re-pricing a model later never rewrites past spend.

## D1 replica

Local JSON under `$SUBJECTIVE_CHAT_HOME` is still the live store. After each history
refresh the service upserts conversations, sessions, messages, and `import_sources`
fingerprints into the Cloudflare D1 database `devagents` over HTTPS (this host is not a
Worker, so the wrangler binding is unused).

Unchanged transcripts are skipped via `import_sources.fingerprint`. Set
`SUBJECTIVE_DEVAGENTS_D1_SYNC=0` to disable. Credentials come from the process
environment or `/subjective/.env` (`CLOUDFLARE_API_TOKEN_5` preferred).

```
GET  /api/d1/sync   last run
POST /api/d1/sync   run now
```

## Tests

```bash
python -m pytest tests/ -q
```

- `tests/test_parity.py` runs the **live Node implementation** (`tests/parity_dump.js`) and
  the Python port over the same fixed inputs and asserts identical output — provider
  labels, envelope stripping, IRC channel hashes, transcript handoff formatting, passwd
  parsing. Skips cleanly when `node` or the client checkout is absent.
- `tests/test_api.py` drives every route the browser calls, plus the WebSocket handshake,
  static stamping, asset traversal blocking, and that saving a config against a dead IRC
  host answers immediately instead of blocking on the TCP timeout.
- `tests/test_schema.py` runs the D1 schema against real SQLite with foreign keys enforced
  (D1 *is* SQLite) and asserts the guarantees the design exists to give: a global
  conversation id surviving provider and OS-user switches, a provider session id that two
  conversations cannot both claim, token rollups that add up, subscription usage that never
  moves money, a findable pending-approval inbox, and a conversation delete that leaves
  nothing behind — including rows with no turn, which imported history produces.
# Clear messages and deployment verification

`POST /api/llm/clear-conversation` clears a local chat's saved messages while retaining
its target, name, and pane assignment. It resets the local execution transcript and
native session, rejects busy chats, and broadcasts the empty conversation to connected
clients. Imported provider transcripts return `read_only`; provider-owned files are not
modified. A failed save leaves the displayed transcript intact. The D1 replica is updated
on a best-effort basis and its outcome is returned separately as `d1`.

The live DevAgents store is `${SUBJECTIVE_CHAT_HOME}/llm-conversations.json`, with
execution state in `${SUBJECTIVE_CHAT_HOME}/language_models.sqlite3` (unless
`SUBJECTIVE_LANGUAGE_MODELS_DB` overrides it). Imported exports live under
`${SUBJECTIVE_CHAT_HOME}/.subjective_chat_history/context`. D1 database `devagents`
contains replica tables including `conversations`, `conversation_sessions`, `turns`,
and `messages`; it is not the source used to reopen a pane.

HTML comments and `X-DevAgents-Version` / `X-DevAgents-Backend-Version` identify the
frontend and backend Git revisions. `X-DevAgents-Frontend-Source` and
`X-DevAgents-Backend-Source` identify exact source contents, including uncommitted edits.
Run this read-only gate against the deployed URL before browser acceptance:

```sh
python scripts/verify_deployment.py https://devagents.subjectivetechnologies.com \
  --client-root /subjective/clients/desktop/subjective_chat
```

It fails on missing or stale markers and also compares the actual served renderer.
The legacy production checkout and canonical `clients/desktop/subjective_devagents`
checkout must contain the same frontend changes. Source fingerprints are captured at
backend startup; restart is required after deployment. The browser regression uses a
temporary chat home, exercises two panes, clears one, drags it back, and reloads the page.
