# Production web routing

The public chat hostname must serve HTML and API responses from the same FastAPI
process. The previous route served HTML through the legacy Node backend, while
JavaScript called FastAPI directly. Node left the HTML instance placeholder
unstamped, so the version banner persisted after every reload.

Verified route on `subjective-agents`, 2026-09-11:

```text
Cloudflare -> HTTP :80 (subjective-chat-web-http.socket)
           -> systemd-socket-proxyd -> nginx 127.0.0.1:8788
           -> verified TLS 127.0.0.1:8080 (FastAPI)
Browser API and events -> devagents-api.subjectivetechnologies.com:8080
```

Cloudflare currently uses Flexible SSL. Do not change the whole zone to repair
one hostname. The local proxy validates the upstream certificate using the
`devagents-api.subjectivetechnologies.com` server name. The certificate does not
cover `devagents.subjectivetechnologies.com`, so a bare port override is insufficient.

FastAPI also needs `SUBJECTIVE_DEVAGENTS_SERVE_UI=1`; the previous production unit
explicitly disabled its UI. `90-serve-ui.conf` enables it and retains the existing
direct API origin, avoiding Cloudflare's long-request timeout on agent turns.
The proxy supports WebSocket upgrades and disables response buffering.

## Installation

Back up the existing HTTP service and any existing files at the destinations below.
Check for running agent turns before restarting FastAPI. Install Debian's `nginx`
package, then disable its default `nginx.service` to prevent contention with the
existing port 80 socket. This deployment uses its own `subjective-devagents-proxy`
unit and configuration; do not enable the default nginx unit.

Install these files as root, mode 0644:

| Source | Destination |
| --- | --- |
| `nginx.conf` | `/etc/subjective-devagents-proxy/nginx.conf` |
| `subjective-devagents-proxy.service` | `/etc/systemd/system/subjective-devagents-proxy.service` |
| `90-serve-ui.conf` | `/etc/systemd/system/subjective-devagents-fastapi.service.d/90-serve-ui.conf` |
| `subjective-chat-web-http.service` | `/etc/systemd/system/subjective-chat-web-http.service` |

Create parent directories first. Validate nginx configuration with
`nginx -t -c /etc/subjective-devagents-proxy/nginx.conf` (create
`/run/subjective-devagents-proxy` for this initial check). Reload systemd, restart
FastAPI, then start the dedicated proxy. Confirm `http://127.0.0.1:8788/` returns
HTML with a real `data-devagents-instance` before installing and restarting the
HTTP forwarder. The existing enabled HTTP socket activates its forwarder, which
requires the dedicated proxy; no additional boot listener is needed.

The legacy `subjective-chat-web.service` is no longer a routing dependency. Keep
it until direct port 8787 consumers have been checked; changing the public route
alone does not prove those consumers are absent.

## Verification

A credential-free curl of the public page must show the live commits:

```bash
curl -sS https://devagents.subjectivetechnologies.com/ | grep 'devagents-deploy'
curl -sS -D - -o /dev/null https://devagents-api.subjectivetechnologies.com:8080/healthz \
  | grep -i 'X-DevAgents'
```

Compare `/api/version` on both public hosts (authenticated): `version`,
`backendVersion`, and `instanceId` must agree with that HTML comment and with
the HTML document's `data-devagents-version` and `data-devagents-instance`.
Placeholders (`DEVAGENTS_DEPLOY_VERSION`, `DEVAGENTS_INSTANCE_ID`) must be gone.
Check actual public HTML and JavaScript, not only `/healthz`.

A fresh browser must keep the version banner hidden. After an idle backend restart,
an already-open tab should show the banner within two minutes. Reload must clear
it. Verify a WebSocket connection to both public origins and run
`tests/test_selenium_prompt_options.py` for desktop and actual mobile emulation.

## Rollback

The initial repair saved the previous HTTP unit under
`/var/backups/subjective-devagents-routing-20260911/`. Restore that unit to
`/etc/systemd/system/subjective-chat-web-http.service`, run `systemctl daemon-reload`,
ensure the legacy Node unit is running, and restart the HTTP forwarder. This
restores the previous public route without deleting conversation state. The
FastAPI API endpoint remains on TLS port 8080 throughout.

Authentication is separate: this routing fix does not enable direct execution or
change credentials. Owner login and token enforcement still require a browser
credential flow; dashboard login alone does not protect this API.

## Firebase owner access

Install `95-owner-auth.conf` in
`/etc/systemd/system/subjective-devagents-fastapi.service.d/95-owner-auth.conf`,
then reload systemd and restart the backend. Owner mode is also the code default:
unset static-token configuration no longer opens the API. Static secrets cannot
bypass owner mode. Direct execution remains disabled.

A valid, unexpired Firebase ID token must match this installation's
`SUBJECTIVE_DEVAGENTS_OWNER_UID` or the verified
`SUBJECTIVE_DEVAGENTS_OWNER_EMAIL`. The Firebase project is
`FIREBASE_PROJECT_ID`. JWT signature, issuer, audience, issue/expiry time and
authentication time are checked. Authentication uses Firebase's public signing
keys; it does not mint tokens or need account keys. None of those values are
stored in this repository.

The dashboard sends its current Firebase token to its exact DevAgents iframe via
`postMessage`. The frame validates sender origin/window, then asks FastAPI to
validate the token before loading chat code. Refresh requests return to the
dashboard's Firebase SDK. The standalone page also supports Firebase Google and
email/password sign-in. If Google rejects the standalone hostname because it is
not in Firebase's authorized-domain list, the page goes to dashboard My Agents;
a persistent "Continue in dashboard" link is also available. Existing dashboard
sessions work without adding a new Firebase authorized domain. Google sign-in for
new sessions still requires the hostname used for that sign-in to be authorized
in Firebase; no domain settings were changed by this deployment.

REST credentials use `Authorization: Bearer`. Owner WebSockets receive the token
in their first frame and join the event hub only after verification. Connections
close at token expiry and the browser reconnects with a refreshed token. HTTP
query-string credentials are not accepted. `/healthz`, `/api/auth/config`, and
the static login shell remain public; chat data, action endpoints, documentation,
and `/api/version` require authentication. CORS wraps authentication so browser
preflight succeeds and rejected requests remain visible as 401 responses.

Deploy the dashboard relay before enabling the backend gate. Refresh existing
browser tabs after deployment. Never enable `development` mode on the public
service; tests explicitly opt into it for credential-free fixtures. `token` mode
is an explicit compatibility mode for automation requiring a configured static
secret; it is not the production owner-only policy.

Firebase revocation is not checked through an admin API. Existing ID tokens can
remain usable until their normal expiration. Logout clears the client session
and unloads chat data; no refresh token or ID token is copied into application
localStorage, HTML, or URLs.
