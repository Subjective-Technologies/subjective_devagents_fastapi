"""Operator-owned identity configuration, not client-supplied process arguments."""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .contracts import Identity, ModelError


def image_support(identity: Identity) -> bool:
    """Mirrors strategies.CliDriver.images: Codex needs local image files, Claude takes them inline."""
    if identity.strategy == "api_agent":
        # Inline base64 image content parts, same request regardless of ssh_host/os_user;
        # unlike CodexCli there is no local-file-path step that an SSH identity would break.
        return True
    if identity.strategy != "cli":
        return False
    return identity.provider == "claude" or (identity.provider == "codex" and not identity.ssh_host)


class IdentityRegistry:
    def __init__(self, identities: list[Identity]):
        self._identities: dict[str, Identity] = {}
        homes: set[tuple[str, str]] = set()
        configs: set[tuple[str, str]] = set()
        for identity in identities:
            if identity.sandbox not in {"read-only", "workspace-write"}:
                raise ValueError("Sandbox must be read-only or workspace-write")
            if any(not isinstance(tool, str) or not tool or "\x00" in tool or "," in tool
                   for tool in identity.allowed_tools):
                raise ValueError("Allowed tools must be individual nonempty tool specifications")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", identity.id):
                raise ValueError("Invalid identity id")
            if identity.id in self._identities:
                raise ValueError("Duplicate identity id")
            if identity.strategy not in {"cli", "api", "api_agent"}:
                raise ValueError("Unknown model strategy")
            if identity.os_user and not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*[$]?", identity.os_user):
                raise ValueError("Invalid OS user")
            if identity.strategy == "cli":
                if identity.provider not in {"codex", "claude", "cursor", "grok"}:
                    raise ValueError("Unsupported CLI provider")
                for value in (identity.home, identity.workspace, identity.config_dir):
                    if not value.startswith("/") or "\x00" in value or ".." in PurePosixPath(value).parts:
                        raise ValueError("CLI paths must be absolute paths without parent traversal")
                canonical = {"cursor": ".config/cursor", "grok": ".grok"}.get(identity.provider)
                if canonical and PurePosixPath(identity.config_dir) != PurePosixPath(identity.home) / canonical:
                    raise ValueError(f"{identity.provider} requires its canonical config directory under home")
                host = identity.ssh_host
                home_key = (host, str(PurePosixPath(identity.home)))
                config_key = (host, str(PurePosixPath(identity.config_dir)))
                if home_key in homes or config_key in configs:
                    raise ValueError("CLI identities must use separate homes and credential directories")
                homes.add(home_key)
                configs.add(config_key)
                if identity.ssh_host:
                    from .transports import SshTunnel
                    SshTunnel(identity.ssh_host, identity.os_user or None)
                if identity.api_key_env or identity.api_base_url:
                    raise ValueError("CLI subscription identities cannot select API credentials")
            elif identity.strategy == "api_agent":
                # Unlike "api", this strategy runs tools locally/remotely like a CLI identity
                # does, so it shares the CLI branch's execution-target checks instead of
                # forbidding them. The endpoint itself is inference-only (operator-trusted,
                # often a private non-TLS proxy); only credential-in-URL and malformed-URL
                # shapes are rejected here.
                if identity.provider not in {"codex", "claude", "grok"}:
                    raise ValueError("Unsupported api_agent provider")
                if (not identity.workspace.startswith("/") or "\x00" in identity.workspace
                        or ".." in PurePosixPath(identity.workspace).parts):
                    raise ValueError("api_agent workspace must be an absolute path without parent traversal")
                if identity.home and (not identity.home.startswith("/") or "\x00" in identity.home
                                      or ".." in PurePosixPath(identity.home).parts):
                    raise ValueError("api_agent home must be an absolute path without parent traversal")
                url = urlsplit(identity.api_base_url)
                if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                    raise ValueError("api_agent URL must be a plain http(s) origin with no embedded credentials")
                if not identity.model or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identity.api_key_env):
                    raise ValueError("api_agent identities require model and API-key environment variable name")
                if identity.ssh_host:
                    from .transports import SshTunnel
                    SshTunnel(identity.ssh_host, identity.os_user or None)
            else:
                url = urlsplit(identity.api_base_url)
                if (url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password
                        or url.query or url.fragment
                        or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
                    raise ValueError("API URL must use HTTPS (HTTP permitted only for loopback tests)")
                if not identity.model or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identity.api_key_env):
                    raise ValueError("API identities require model and API-key environment variable name")
                if identity.ssh_host or identity.os_user:
                    raise ValueError("API strategy does not execute through OS users or SSH")
            self._identities[identity.id] = identity

    def register_legacy(self, identity: Identity) -> None:
        """Bind an existing OS account. Store admission serializes shared credential paths."""
        IdentityRegistry([identity])  # Apply exactly the same field validation.
        previous = self._identities.get(identity.id)
        if previous is not None and previous != identity:
            raise ModelError("identity_changed", "Existing identity cannot be retargeted.")
        self._identities[identity.id] = identity

    @classmethod
    def load(cls, path: str) -> "IdentityRegistry":
        data = json.loads(Path(path).read_text())
        return cls([Identity(**item) for item in data["identities"]])

    def get(self, identifier: str) -> Identity:
        try:
            return self._identities[identifier]
        except KeyError as exc:
            raise ModelError("unknown_identity", "Identity is not configured on this server.") from exc

    def public(self) -> list[dict]:
        return [{"id": item.id, "provider": item.provider, "model": item.model,
                 "strategy": item.strategy, "remote": bool(item.ssh_host),
                 "capabilities": {"text": True, "attachments": image_support(item),
                                  "interactive_approvals": False, "native_resume": item.strategy == "cli"}}
                for item in self._identities.values()]
