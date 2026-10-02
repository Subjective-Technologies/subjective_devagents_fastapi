"""Saved SSH connections — host, port, account and credentials, kept between chats.

Until now an SSH chat's target was a string in the conversation's workspace, which
reaches a host, an account, a port and a path and nothing else. That is enough for
agent authentication and nothing else: a key, a password, a passphrase or a pinned
host key had nowhere to live. This is that missing place — the equivalent of the
saved-connection list in any SSH client.

Three sources feed the picker, and only one of them is stored here:

- **`~/.ssh/config` aliases.** Already a saved connection, written by the operator,
  with the account, port, key and options the real ssh command would use. Listed,
  never copied: duplicating a config entry into this store would be one more place
  to keep in step, and the whole point of an alias is that ssh already knows it.
- **Keys in `~/.ssh`.** Offered by path so a connection can name one. Key *contents*
  are never read here — the datasource hands the path to ssh, which is what a
  passphrase-protected key in an agent needs anyway.
- **Keys uploaded from a browser.** The server's own `~/.ssh` is the *server's*, not
  the operator's laptop, so on a deployed instance it holds nothing useful. An
  uploaded key is written into a private directory beside the store and then behaves
  exactly like one already in `~/.ssh`: named by path, contents never read again.
- **Connections saved here.** Everything else: a host and account typed in once.

**Secrets at rest.** A stored password or passphrase is a plaintext secret on the
server's disk, so the file is written 0600 inside a 0700 directory and never leaves
this module: every read path strips them, and the API returns connections without
them. That is weaker than a keychain and stronger than the alternative on offer
(retyping a password into every chat), so the UI says which fields are stored and
prefers agent and key authentication, which store no secret at all.
"""
from __future__ import annotations

import json
import os
import re
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Fields a caller may set. `id`, `createdAt` and `updatedAt` are ours.
EDITABLE_FIELDS = frozenset({
    "name",
    "host",
    "port",
    "username",
    "auth_method",
    "password",
    "private_key_path",
    "key_passphrase",
    "known_hosts_path",
    "host_key_fingerprint",
    "host_key_policy",
    "strict_host_key_checking",
    "working_directory",
    "shell",
    "term",
})
# Never returned by any read, and never logged.
SECRET_FIELDS = frozenset({"password", "key_passphrase"})
AUTH_METHODS = ("agent", "private_key_path", "password")
# accept-new trusts a machine's key the first time and refuses it if it ever
# changes. strict cannot reach a machine this server has never met, which is every
# new machine; off verifies nothing.
HOST_KEY_POLICIES = ("accept-new", "strict", "off")

# Long enough for a descriptive name, short enough to stay a label.
MAX_NAME = 120
MAX_VALUE = 4096
MAX_CONNECTIONS = 200

_lock = threading.RLock()

# Private keys are the files in ~/.ssh that are not these. Matched on the *prefix*
# for the well-known names, because an operator's directory is full of things like
# `config.backup.20251002_184950`, which an anchored exact match let through and
# offered as a key.
_NOT_A_KEY = re.compile(
    r"^(config|known_hosts|authorized_keys|environment|rc|agent|sshd?_config)"
    r"|\.(pub|ppk|old|bak|orig|backup|log|swp|tmp|json|txt|md)$"
    r"|\.backup",
    re.IGNORECASE,
)
_HOST_LINE = re.compile(r"^\s*Host\s+(?P<hosts>.+?)\s*$", re.IGNORECASE)


def store_path(app_home: str) -> Path:
    return Path(app_home) / "ssh-connections.json"


def ssh_dir() -> Path:
    """The service account's own ~/.ssh — whose keys and config it can actually use."""
    return Path(os.environ.get("SUBJECTIVE_DEVAGENTS_SSH_DIR") or (Path.home() / ".ssh"))


def uploaded_keys_dir(app_home: str) -> Path:
    """Where keys sent from a browser live. Private, and beside the store."""
    return Path(app_home) / "ssh-keys"


# What a private key file starts with. Checked so an upload cannot quietly turn this
# directory into a general-purpose file drop, and so that pasting the *public* key —
# the easiest mistake to make — is caught here rather than as an ssh failure later.
_PRIVATE_KEY_HEADER = re.compile(
    r"^-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|^PuTTY-User-Key-File-\d"
)
MAX_KEY_BYTES = 64 * 1024
_SAFE_KEY_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _key_display_name(name: str) -> str:
    """A browser's filename is untrusted input that becomes part of a path."""
    cleaned = _SAFE_KEY_NAME.sub("-", str(name or "").strip()).strip("-.")
    return (cleaned or "key")[:64]


def save_uploaded_key(app_home: str, name: str, content: str) -> dict:
    """Store a private key sent from a browser and return how to refer to it."""
    text = str(content or "")
    if len(text.encode("utf-8", "replace")) > MAX_KEY_BYTES:
        raise ValueError("That file is too large to be a private key.")
    stripped = text.strip()
    if not _PRIVATE_KEY_HEADER.match(stripped):
        raise ValueError(
            "That does not look like a private key. Choose the key file itself "
            "(id_ed25519), not the .pub next to it."
        )

    directory = uploaded_keys_dir(app_home)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(stat.S_IRWXU)
    # The filename is ours, not the browser's: a generated id keeps two uploads of
    # the same name apart and keeps any path trickery out of it entirely.
    display = _key_display_name(name)
    path = directory / f"{uuid.uuid4().hex}-{display}"
    # ssh refuses a key others can read, so it is written private from the start
    # rather than chmod'ed after a moment of being readable.
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(stripped + "\n")
    return {"path": str(path), "name": display, "source": "uploaded"}


def delete_uploaded_key(app_home: str, path: str) -> bool:
    """Remove an uploaded key. Only ever a file this module wrote."""
    directory = uploaded_keys_dir(app_home).resolve()
    try:
        target = Path(path).resolve()
        # Anything outside the upload directory is not ours to delete, however the
        # caller spelled the path.
        target.relative_to(directory)
    except (OSError, ValueError):
        return False
    try:
        target.unlink()
    except OSError:
        return False
    return True


# --- reading and writing the store ----------------------------------------------------


def _read(app_home: str) -> dict:
    path = store_path(app_home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"connections": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("connections"), list):
        return {"connections": []}
    return payload


def _write(app_home: str, payload: dict) -> None:
    """Write 0600, atomically, so a crash cannot leave a half-written secret file."""
    path = store_path(app_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    os.replace(temporary, path)


def public(connection: dict) -> dict:
    """A connection as the API and the UI may see it: never the secrets.

    Booleans say whether a secret is set, because a form has to show "password
    saved" without ever sending the password to the browser.
    """
    visible = {key: value for key, value in connection.items() if key not in SECRET_FIELDS}
    visible["hasPassword"] = bool(connection.get("password"))
    visible["hasKeyPassphrase"] = bool(connection.get("key_passphrase"))
    return visible


def list_connections(app_home: str) -> list[dict]:
    with _lock:
        return [public(item) for item in _read(app_home)["connections"]]


def get(app_home: str, connection_id: str) -> dict | None:
    """The full connection, secrets included — for opening a session, not for display."""
    with _lock:
        for item in _read(app_home)["connections"]:
            if item.get("id") == connection_id:
                return dict(item)
    return None


def _clean(payload: dict, existing: dict | None = None) -> dict:
    """Validate and normalise what a caller sent."""
    result = dict(existing or {})
    for key, value in (payload or {}).items():
        if key not in EDITABLE_FIELDS:
            continue
        if isinstance(value, str) and len(value) > MAX_VALUE:
            raise ValueError(f"{key} is too long.")
        result[key] = value

    host = str(result.get("host") or "").strip()
    if not host:
        raise ValueError("A host is required: a hostname, an address, or a ~/.ssh/config alias.")
    result["host"] = host
    result["name"] = (str(result.get("name") or "").strip() or host)[:MAX_NAME]

    if result.get("port") in (None, ""):
        result.pop("port", None)
    else:
        try:
            port = int(result["port"])
        except (TypeError, ValueError):
            raise ValueError("Port must be a number.") from None
        if not 1 <= port <= 65535:
            raise ValueError("Port must be between 1 and 65535.")
        result["port"] = port

    method = str(result.get("auth_method") or "agent").strip()
    if method not in AUTH_METHODS:
        raise ValueError(f"Authentication must be one of: {', '.join(AUTH_METHODS)}.")
    result["auth_method"] = method
    if method == "private_key_path" and not str(result.get("private_key_path") or "").strip():
        raise ValueError("Choose a key, or switch authentication to agent or password.")

    # An empty string means "clear this secret"; absent means "leave it as it was".
    for field in SECRET_FIELDS:
        if field in (payload or {}) and not payload[field]:
            result.pop(field, None)

    policy = str(result.get("host_key_policy") or "").strip()
    if policy and policy not in HOST_KEY_POLICIES:
        raise ValueError(f"Host key checking must be one of: {', '.join(HOST_KEY_POLICIES)}.")
    if policy:
        result["host_key_policy"] = policy
        # The boolean this replaced would otherwise keep overriding the choice.
        result.pop("strict_host_key_checking", None)
    elif "strict_host_key_checking" in result:
        result["strict_host_key_checking"] = bool(result["strict_host_key_checking"])
    return result


def create(app_home: str, payload: dict) -> dict:
    with _lock:
        stored = _read(app_home)
        if len(stored["connections"]) >= MAX_CONNECTIONS:
            raise ValueError(f"Too many saved connections (max {MAX_CONNECTIONS}).")
        connection = _clean(payload)
        connection["id"] = uuid.uuid4().hex
        connection["createdAt"] = connection["updatedAt"] = time.time()
        stored["connections"].append(connection)
        _write(app_home, stored)
        return public(connection)


def update(app_home: str, connection_id: str, payload: dict) -> dict:
    with _lock:
        stored = _read(app_home)
        for index, item in enumerate(stored["connections"]):
            if item.get("id") != connection_id:
                continue
            connection = _clean(payload, existing=item)
            connection["id"] = connection_id
            connection["updatedAt"] = time.time()
            stored["connections"][index] = connection
            _write(app_home, stored)
            return public(connection)
    raise LookupError("That saved connection no longer exists.")


def delete(app_home: str, connection_id: str) -> bool:
    with _lock:
        stored = _read(app_home)
        remaining = [item for item in stored["connections"] if item.get("id") != connection_id]
        if len(remaining) == len(stored["connections"]):
            return False
        stored["connections"] = remaining
        _write(app_home, stored)
        return True


# --- what the operator's own ~/.ssh already knows ---------------------------------------


def list_keys(app_home: str = "") -> list[dict]:
    """Private keys this server can use, by path. Contents are never read.

    Two places: the service account's `~/.ssh`, and the uploads directory. In
    `~/.ssh` a key is identified by exclusion — everything that is not config,
    known_hosts, a public key or a backup — because key files have no required
    naming beyond the id_* convention and an operator's key may be called anything.
    Uploads need no such guessing: everything in that directory was put there as a
    key and was checked when it arrived.
    """
    keys: list[dict] = []
    sources = [(ssh_dir(), "ssh_dir")]
    if app_home:
        sources.append((uploaded_keys_dir(app_home), "uploaded"))
    for directory, source in sources:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if source == "ssh_dir" and _NOT_A_KEY.search(entry.name):
                continue
            try:
                mode = stat.S_IMODE(entry.stat().st_mode)
            except OSError:
                continue
            keys.append(_key_entry(entry, mode, source))
    keys.sort(key=lambda key: (key["source"] != "uploaded", not key["hasPublicKey"], key["name"]))
    return keys


def _key_entry(entry: Path, mode: int, source: str) -> dict:
    return {
            "path": str(entry),
            # An uploaded file is named `<id>-<display>`; the id is bookkeeping and
            # would only clutter the picker.
            "name": entry.name.split("-", 1)[-1] if source == "uploaded" else entry.name,
            "source": source,
            # A private key almost always has a sibling .pub, so this sorts the
            # likely keys to the top without reading a single byte of any of them.
            # ssh refuses a key others can read, and saying so here beats a failed
            # connection with a message the operator never sees.
            "tooOpen": bool(mode & 0o077),
            "hasPublicKey": entry.with_suffix(entry.suffix + ".pub").exists()
            or entry.with_name(entry.name + ".pub").exists(),
    }


def list_config_hosts() -> list[dict]:
    """`Host` aliases from the service account's ~/.ssh/config.

    These are connections the operator already saved, in the place ssh itself reads,
    so they are offered as-is rather than imported: an alias carries its account,
    port, key and options, and copying those here would create a second copy to keep
    in step.

    Patterns are skipped. `Host *` and friends are defaults applied to other hosts,
    not machines anyone can connect to.
    """
    path = ssh_dir() / "config"
    hosts: list[dict] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return hosts
    for line in lines:
        match = _HOST_LINE.match(line)
        if not match:
            continue
        for alias in match.group("hosts").split():
            if any(character in alias for character in "*?!") or alias in seen:
                continue
            seen.add(alias)
            hosts.append({"host": alias, "name": alias, "source": "ssh_config"})
    return hosts


# --- turning any of the three into something the datasource can open --------------------


def connection_settings(app_home: str, reference: str) -> dict:
    """Resolve what a conversation stored into connection fields for the datasource.

    A conversation holds one string. It is either `saved:<id>`, naming a connection
    in this store, or a target written out longhand (`user@host:/path`, or an alias).
    Both are supported because the longhand form predates this store and still works
    without one.
    """
    reference = str(reference or "").strip()
    if reference.startswith("saved:"):
        connection = get(app_home, reference[len("saved:"):])
        if connection is None:
            raise ValueError(
                "That saved SSH connection no longer exists. Pick another one for this chat."
            )
        return {
            key: value
            for key, value in connection.items()
            if key in EDITABLE_FIELDS and key != "name" and value not in (None, "")
        }

    settings = ssh_connection_from_workspace(reference)
    if not settings.get("host"):
        raise ValueError(
            "This SSH chat has no host. Set its workspace to a saved connection, or to "
            "a machine — `user@host`, `user@host:/path`, or a Host alias from ~/.ssh/config."
        )
    return settings


def describe_reference(app_home: str, reference: str) -> str:
    """A label for a conversation's target, for logs and the UI."""
    reference = str(reference or "").strip()
    if reference.startswith("saved:"):
        connection = get(app_home, reference[len("saved:"):])
        if connection is None:
            return "(deleted connection)"
        account = f"{connection.get('username')}@" if connection.get("username") else ""
        return f"{connection.get('name')} ({account}{connection.get('host')})"
    return reference


_SSH_WORKSPACE_RE = re.compile(
    r"""^
    (?:(?P<username>[^@/\s]+)@)?
    (?P<host>\[[0-9A-Fa-f:]+\]|[^@:/\s]+)
    (?::(?P<port>\d+))?
    (?::?(?P<path>/\S*))?
    $""",
    re.VERBOSE,
)


def ssh_connection_from_workspace(workspace: str) -> dict:
    """Read an SSH target out of a conversation's workspace string.

    The SSH runtime needs a host and an account, and a conversation has exactly one
    free-text field that can carry them: its workspace. So an SSH chat's workspace is
    written `deploy@server.example.com:/srv/app` (or `ssh://deploy@server:2222/srv/app`)
    and is split here into the runtime's connection fields.

    This reaches only the fields that fit in one line, which means agent
    authentication — the default, and the only method that stores no secret. Keys,
    passwords and pinned fingerprints need somewhere to be entered and stored per
    connection; until that exists they can be supplied through `extra`.

    An unparseable workspace yields `{}` rather than a guess: the runtime then
    reports `configuration_required` naming what is missing, which is a better failure
    than connecting somewhere unintended.
    """
    text = str(workspace or "").strip()
    if not text:
        return {}
    if text.startswith("ssh://"):
        text = text[len("ssh://"):]
    # A plain local path is a workspace for the CLI runtimes, not an SSH target.
    if text.startswith(("/", ".", "~")):
        return {}
    match = _SSH_WORKSPACE_RE.match(text)
    if not match:
        return {}
    connection: dict[str, Any] = {"host": match.group("host").strip("[]")}
    if match.group("username"):
        connection["username"] = match.group("username")
    if match.group("port"):
        connection["port"] = int(match.group("port"))
    # The runtime's own `working_directory`, not the raw workspace string that
    # connection_for already put there.
    connection["working_directory"] = match.group("path") or ""
    return connection
