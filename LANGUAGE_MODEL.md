# LanguageModel replacement engine

`LanguageModel` is a provider-neutral facade over `ModelStrategy`. `Cli`, `Api` and
`ApiAgent` implement that strategy; `CodexCli` and `ClaudeCli` implement the
`CliDriver` protocol. `LocalProcess` and `SshTunnel` implement `ProcessTransport`
independently of the provider. There are no imports from datasource plugins,
pipeline bundles, or the abstract datasource library in this package.

## CLI vs api_agent: why there are two execution paths

`Cli` spawns the vendor's own binary (`claude`, `codex`, `grok`, `cursor-agent`),
which is itself a full coding agent: it runs Bash, edits files, and reports its own
tool-use stream. That binary has to be installed, logged in, and kept compatible on
every execution host, which is the main source of operational unreliability this
strategy exists to retire.

`ApiAgent` replaces it for `claude`, `codex` and `grok` by talking straight to an
inference-only HTTP endpoint (for example the `CLIProxyAPI` described in
`/subjective/how_to_llm.md`) using the OpenAI Chat Completions tool-calling wire
format. That endpoint never touches the workspace — "the client runs tools, then
sends the result back" — so `ApiAgent` *is* the tool loop and tool executor now:
it declares `Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob` and `TodoWrite`, and runs
them itself through the same `LocalProcess`/`SshTunnel` transports and sudo/SSH
placement `Cli` uses, so remote and other-OS-user accounts work identically. One
`ApiAgent` instance serves every provider behind the proxy; step naming for the live
feed is shared with `ClaudeCli.tool_step` so the frontend did not need to change.
Cursor has no account behind the proxy, so it stays on `Cli` unconditionally.

Consequences worth knowing operationally:

- No native session: the proxy is stateless, so `ApiAgent` never returns a
  `session_id` and resends the full prior transcript (user/assistant pairs only,
  not intermediate tool calls) on every turn, same as the pre-existing `Api` class.
- Images are sent inline as OpenAI-style `image_url` content parts (base64 data URL), the
  same per-image/per-turn limits as everywhere else (`contracts.MAX_ATTACHMENT_BYTES`,
  `MAX_ATTACHMENTS`). Whether the selected model actually understands them is between it and
  the proxy; an unsupported combination surfaces as a normal provider error code, not a crash.
- A tool call outside the effective set (sandbox + `allowed_tools`) is not fatal:
  the model is told it is not permitted and can continue without it.
- `api_agent`'s URL validation is deliberately looser than `api`'s: it allows plain
  `http://` to any host, because the documented deployment target
  (`CLIProxyAPI`) runs TLS off by design and `api_base_url` is operator-configured,
  never client input. Prefer the loopback or Tailscale address over a public one
  when more than one is reachable from the execution host, since the bearer key
  travels in the clear.

The structural contract is canonical architecture v4.1.0, package
`package.devagents-language-model`. Python `Identity` and `TurnStore` implement
the model's `ExecutionIdentity` and `ConversationStore`; native session and
handoff records are persisted by the store. `CodexCli`/`ClaudeCli` implement the
model's Codex/Claude driver roles. This document explains usage and migration.

## What is implemented

- Non-PTY `codex exec --json` and `claude -p --output-format stream-json` execution.
- Explicit native session resume under the same immutable identity.
- Claude text deltas and Codex completed message events, persisted with event cursors.
- An OpenAI-compatible HTTP chat-completions strategy (separate API billing).
- Local subprocesses, optional noninteractive `sudo -n -u`, and SSH `-T` execution.
- Credential-isolated environments: only configured account paths reach the CLI.
  Ambient API keys, OAuth tokens, provider selectors and proxies are not inherited.
- SQLite-backed acceptance, idempotency, results, cancellation requests, history
  and handoff records. HTTP/WebSocket disconnects do not erase results.
- One active turn per conversation and one active CLI turn per account. Different
  accounts may run concurrently. Reservations work across processes sharing the
  same SQLite database. A second conversation using the same busy CLI account
  receives `409 identity_busy`; the caller can try again after completion.
- No automatic provider fallback, failed-turn replay, or restart replay.
- Explicit import of completed legacy text conversations.

## Boundaries of this release

Existing browser routes now adapt into this engine. Datasource and exported-pipeline
execution paths are retired from send handling; plugins remain on disk only for
historical import and unrelated compatibility surfaces.

Only text turns are supported. Attachments and interactive tool-approval exchanges
are rejected or reported as unsupported, never silently ignored. Codex defaults
to `read-only`; an operator can select `workspace-write`. Claude uses `dontAsk`
and the operator's `allowed_tools`; denied tool requests return `approval_required`.
There is no blanket permission bypass. CLI configuration and workspace files can
also influence provider behavior, so provision trusted account profiles.

The initial Codex adapter uses bounded `exec` rather than the richer app-server
protocol. A future app-server driver can fit behind the strategy boundary. It is
not implemented here. API requests return a complete text response rather than
token streaming; tool calls and truncated/non-text completions fail explicitly.

Process execution targets Linux/POSIX. Authentication is provisioned separately in
each configured home; sends never start login. This release does not provide an
OAuth callback/device-login UI. Authentication expiry and subscription limits
still require provider-specific account handling. Pin and test CLI versions on
the actual execution hosts before cutover. The test suite uses real processes
and HTTP fixtures but does not, by default, consume subscriptions.

`SshTunnel` means an SSH command channel, not TCP port forwarding. SSH keys and
verified host keys must already be provisioned for the service user. Prompts stay
on stdin; arguments and working directories are shell-quoted. The optional target
OS user must be reachable through preconfigured noninteractive sudo. Changing
directory happens under that target user, so private workspaces can work.

## Configure accounts

Copy and adapt [examples/language-models.json](examples/language-models.json).
The example is not activated automatically. Each account has its own `home` and
`config_dir`, both on its execution host. Provision these directories with the
correct ownership and log in using exactly those paths. Use absolute executable
paths when CLI installations depend on an interactive shell or Node version manager.

The identity id is an operator-assigned account label, not an OAuth subject verified
by this engine. Do not replace credentials in place while keeping the identity id.
Use a new configured identity and an explicit handoff. Do not alias homes or
credential directories through symlinks or different SSH names. The registry
rejects duplicate configured paths; it cannot prove remote filesystem identity.

The model, workspace, account paths, OS user, host and permission settings are
snapshotted for each turn. Editing configuration cannot silently retarget an
existing conversation: subsequent sends fail `identity_changed` until handoff.

Enable on the existing FastAPI process at its next controlled start:

```bash
export SUBJECTIVE_LANGUAGE_MODELS_CONFIG=/absolute/path/language-models.json
export SUBJECTIVE_LANGUAGE_MODELS_DB=/absolute/path/language-models.sqlite3
```

Without `SUBJECTIVE_LANGUAGE_MODELS_CONFIG`, new endpoints return `503 disabled`.
The database defaults to `${SUBJECTIVE_CHAT_HOME}/language_models.sqlite3`, where
the default chat home is `~/.subjective_chat`. It is separate from platform userdata
and legacy conversation files. New databases are created with mode `0600`.
Keep the database on a local filesystem supporting SQLite locking, not an object
storage mount; all API workers must share this one database for admission control.

An optional API identity (no tool use, single text response) uses this shape:

```json
{
  "id": "api-account",
  "strategy": "api",
  "provider": "openai-compatible",
  "model": "YOUR_MODEL_ID",
  "api_base_url": "https://YOUR_PROVIDER/v1",
  "api_key_env": "DEVAGENTS_PROVIDER_API_KEY"
}
```

The referenced environment variable is loaded by the operator's service environment,
not copied into the JSON file. This API strategy does not consume a CLI subscription.
Endpoints are operator-configured; HTTP clients cannot provide commands, credential
paths, URLs or OS users in a send request.

### Agentic API identity (`api_agent`): proxy-backed replacement for a CLI account

An `api_agent` identity keeps the coding-agent behavior (Bash, file edits, over SSH
or as another OS user) but gets inference from an HTTP endpoint instead of a locally
installed, logged-in CLI binary:

```json
{
  "id": "claude-alice",
  "strategy": "api_agent",
  "provider": "claude",
  "model": "claude-sonnet-4-5",
  "workspace": "/srv/workspaces/project",
  "home": "/srv/devagents/accounts/claude-alice",
  "os_user": "alice",
  "api_base_url": "http://127.0.0.1:8317/v1",
  "api_key_env": "SUBJECTIVE_DEVAGENTS_LLM_PROXY_API_KEY"
}
```

`provider` must be `claude`, `codex` or `grok` (the accounts `CLIProxyAPI` fronts;
there is no Cursor account behind it). `model` must be a current id from that
server's `GET /v1/models` — the catalog changes, so do not copy a stale id from
this document. `api_key_env` names the environment variable holding the proxy's
*client* key (`access.api-keys` in the proxy's config, per `/subjective/how_to_llm.md`),
set by the operator's service environment, never written into this file.

**Existing deployments with no `language-models.json` at all** (the common case —
see `SUBJECTIVE_LANGUAGE_MODELS_CONFIG` below) get `api_agent` identities built
automatically, per OS account, once the operator sets:

```bash
export SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL=http://127.0.0.1:8317/v1      # or the Tailscale/public address
export SUBJECTIVE_DEVAGENTS_LLM_PROXY_API_KEY=...                       # the proxy's client key, never the file itself
export SUBJECTIVE_DEVAGENTS_LLM_MODEL_CLAUDE=claude-sonnet-4-5          # a current id from GET /v1/models
export SUBJECTIVE_DEVAGENTS_LLM_MODEL_CODEX=...
export SUBJECTIVE_DEVAGENTS_LLM_MODEL_GROK=...
```

Until `SUBJECTIVE_DEVAGENTS_LLM_PROXY_URL` and the relevant per-provider model
variable are both set, that provider keeps using its existing CLI account —
nothing changes until an operator opts in. A conversation already bound to the
old CLI identity will not silently switch: `identity_changed` is returned until an
explicit handoff, exactly as for any other identity-configuration edit (see
"Configure accounts" above).

## Use the API

All routes inherit the existing FastAPI authentication policy. In production owner
mode, use the owner's Firebase bearer token. Automation tokens work only when
the existing service is explicitly configured to accept them.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/language-model/identities` | Configured identities and supported capabilities |
| POST | `/api/language-model/conversations` | Create with `identity_id` and optional `history` |
| GET | `/api/language-model/conversations/{id}` | Transcript, identity, native session, active turn and handoffs |
| POST | `/api/language-model/conversations/{id}/turns` | Submit `prompt`, unique `request_id`, optional `timeout` |
| GET | `/api/language-model/turns/{id}` | Durable state, result and error code |
| GET | `/api/language-model/turns/{id}/events?after=N` | Up to 1,000 events after the last sequence received |
| POST | `/api/language-model/turns/{id}/cancel` | Request cancellation; poll for the actual outcome |
| POST | `/api/language-model/conversations/{id}/handoff` | Switch to another configured `identity_id` |
| POST | `/api/language-model/turns/{id}/reconcile` | Operator recovery after independently confirming execution stopped |

Example create body:

```json
{"identity_id": "claude-alice"}
```

Example send body:

```json
{"prompt": "Explain this project", "request_id": "client-generated-unique-id", "timeout": 300}
```

Submission returns HTTP 202 with a turn `id`. Acknowledgment is not successful model
completion. Poll that turn until `completed`, `failed`, `cancelled`, `timed_out`, or
`interrupted`. Reusing the same `request_id` and payload returns the original turn,
even after restart; a different payload with that id returns 409. Never replace the
request id merely because the HTTP connection was lost.

Once the active turn finishes, hand off with:

```json
{"identity_id": "codex-remote"}
```

Handoff records the old binding and clears the native session. The next CLI turn
receives the completed user/assistant transcript as context in a new native session.
Returning to an earlier account also starts a new session with the current
transcript, preventing an old native session from silently missing intervening turns.
API turns always carry the full completed transcript. Context is capped at 512 KiB;
oversized transfers fail rather than silently truncate. Hidden reasoning, tool state,
attachments and uncompleted turns are not transferred. Verify workspace side effects
when recovering from a failed or cancelled tool-using turn.

## Failure and recovery

Normal shutdown requests cancellation and waits for workers to finish. Local
subprocess groups receive SIGINT followed by forced cleanup if needed. A crashed
server can leave `queued` or `running` records; startup deliberately does not
relaunch them or assume their CLI/SSH children are dead.

SSH interruption cannot prove that a remote CLI stopped. HTTP cancellation likewise
cannot prove that provider inference stopped. These return `interrupted` with an
outcome-unknown code and block further work for that conversation/account. Inspect
the execution host/provider and stop the old job before calling reconciliation with:

```json
{"confirmed_stopped": true}
```

Reconciliation does not execute or replay anything. It preserves an interrupted
record, clears native resume and allows a deliberate new request. It refuses an
actively executing worker owned by this process. Across processes, the operator
must independently verify that the owning worker and remote job stopped.

CLI diagnostics are not copied verbatim into logs because they can contain
credentials. Persisted `process_exit` events include exit code and transport status;
server logs include turn id, identity id, terminal status and stable error code.
Inspect provider authentication separately under the configured OS user/home when
`cli_exit_error` occurs. Transcript and text events are private application data.

## Migrate and retire datasources

1. Provision two isolated accounts and select required providers.
2. Run fixture tests and live gate below against deployment.
3. Export a complete legacy conversation. Convert it without modifying the source:

   ```bash
   .venv/bin/python -m subjective_devagents_fastapi.language_models.migration \
     conversation-export.json claude-alice > import-body.json
   ```

4. POST `import-body.json` to the new conversations endpoint. Imports require complete
   text-only user/assistant pairs. Failed turns, system-message interleaving,
   attachments or truncated exports require explicit handling first. Native session
   IDs are never imported across identities.
5. Migrate the consumer to turn IDs, result polling/event cursors and explicit handoff.
   Validate real account A -> account B -> provider C -> account A continuity.
6. Retain old data and datasource files for rollback/import. Remove them in a separate
   cleanup after live-provider matrix and consumer workflows pass.

No account login, service restart, production cutover, old-data deletion or plugin
removal is performed by installing this code.

## Verification

```bash
.venv/bin/python -m pytest tests/language_models -v -ra
```

Live deployment gate (consumes the selected accounts' actual allowance):

```bash
SUBJECTIVE_LANGUAGE_MODELS_LIVE_URL=http://127.0.0.1:8080 \
SUBJECTIVE_LANGUAGE_MODELS_LIVE_IDENTITIES=claude-alice,claude-bob,codex-remote,claude-alice \
.venv/bin/python -m pytest tests/language_models/test_live.py -v -ra
```

Supply `SUBJECTIVE_LANGUAGE_MODELS_LIVE_TOKEN` through the environment if required
by the target service. The gate retains its created conversation as evidence,
asserts native resume and marker retention across handoffs, and fails (not skips)
when a selected identity is unavailable. The default suite skips this gate when
the URL is absent. It does not yet cover real-provider restart/remote cancellation;
those remain required operational acceptance checks before legacy retirement.

Provider references: [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode),
[Claude headless execution](https://code.claude.com/docs/en/headless).
