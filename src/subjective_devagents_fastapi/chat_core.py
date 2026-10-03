"""The transport-independent backend: IRC client, LLM agent conversations, DCC file
transfer, and "Continue as..." su-session glue.

Python port of clients/desktop/subjective_devagents/src/chatCore.js. Every mutating
method is synchronous and safe to call from a worker thread; events are pushed out
through the `emit` callback the app wires to the WebSocket broadcaster.

IRC is optional (SUBJECTIVE_CHAT_IRC=1). My Agents is the LLM UI; the default IRC host
is often down, and reconnect storms painted the status bar as errors on every send.
"""
from __future__ import annotations

import hashlib
import os
import re
import socket
import ssl
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .llm_service import (
    LlmService,
    format_transcript_for_handoff,
    provider_label,
    tail_conversation_for_client,
)
from .prompt_options import annotate_message
from .provider_sessions import list_provider_sessions
from .su_session_manager import SuSessionManager
from .system_users import current_username, is_valid_username, list_system_users

LOGIN_PROVIDERS = {"codex", "claude", "cursor"}

DEFAULT_CONNECTION = {
    "name": "Subjective LAN",
    "host": "192.168.0.250",
    "port": 6667,
    "tls": False,
    "channel": "#subjective",
    "nick": "Tomasso",
    "realname": "Tomasso",
}

TLS_TOKEN_RE = re.compile(r"^(1|true|yes|on|tls|ssl)$", re.IGNORECASE)
IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|avif)$", re.IGNORECASE)


def is_tls_token(value: Any) -> bool:
    return bool(TLS_TOKEN_RE.match(str(value or "").strip()))


class ChatCore:
    def __init__(
        self,
        root: str | Path,
        app_home: str | Path | None = None,
        log: Callable[..., None] | None = None,
        emit: Callable[[dict], None] | None = None,
        default_connection: dict | None = None,
    ):
        self.root = str(root)
        self.application_home = str(
            app_home or os.environ.get("SUBJECTIVE_CHAT_HOME") or Path.home() / ".subjective_chat"
        )
        self.log_fn = log or (lambda level, message, details=None: None)
        self._emit = emit or (lambda event: None)

        self.default_connection = {**DEFAULT_CONNECTION, **(default_connection or {})}
        self.config = {
            "host": os.environ.get("IRC_HOST") or self.default_connection["host"],
            "port": int(os.environ.get("IRC_PORT") or self.default_connection["port"]),
            "tls": str(os.environ.get("IRC_TLS") or "false").lower() == "true",
            "channel": os.environ.get("IRC_CHANNEL") or self.default_connection["channel"],
            "nick": os.environ.get("IRC_NICK") or self.default_connection["nick"],
            "realname": os.environ.get("IRC_REALNAME") or os.environ.get("IRC_NICK") or self.default_connection["realname"],
            "dccPublicHost": os.environ.get("DCC_PUBLIC_HOST") or "",
            "downloadDir": str(Path(self.root) / os.environ["DCC_DOWNLOAD_DIR"])
            if os.environ.get("DCC_DOWNLOAD_DIR")
            else str(Path.home() / "Downloads"),
            "sentDir": str(Path(self.application_home) / "sent"),
        }

        self.bookmarks: list[dict] = []
        self.manual_disconnect = False
        self.socket: socket.socket | None = None
        self.connected = False
        self.connecting = False
        self.registered = False
        self.reconnect_timer: threading.Timer | None = None
        self.reconnect_attempt = 0
        self.line_buffer = ""
        self.current_nick = self.config["nick"]
        self.channel_users: dict[str, dict] = {}
        self.names_buffer: list[str] = []
        self.names_refresh_timer: threading.Timer | None = None
        self.nick_retry_timer: threading.Timer | None = None
        self.active_llm_irc_channels: dict[str, str] = {}
        self.shutting_down = False
        self._irc_fail_count = 0
        self._write_lock = threading.Lock()
        self._su_viewer_broadcasts: dict[str, Callable[[str], None]] = {}

        self.su_session_manager = SuSessionManager(log=self.log_fn)
        self.su_session_manager.on_status = lambda event: self.broadcast("switchuser-status", event)
        self.su_session_manager.on_provider_status = lambda event: self.broadcast("switchuser-provider-status", event)

        self.llm_service = LlmService(
            root=self.root,
            app_home=self.application_home,
            su_session_manager=self.su_session_manager,
            log=self.log_fn,
        )
        self.llm_service.on_progress = self._on_llm_progress
        for conversation in self.llm_service.list_conversations()["conversations"]:
            if conversation.get("activeOnIrc") and conversation.get("ircChannel"):
                self.active_llm_irc_channels[conversation["ircChannel"].lower()] = conversation["target"]

    def _on_llm_progress(self, progress: dict) -> None:
        if not progress.get("text"):
            return
        payload = {"target": progress.get("target") or "", "text": progress["text"]}
        if progress.get("diff"):
            payload["diff"] = progress["diff"]
        self.broadcast("llm-progress", payload)

    def log(self, level: str, message: str, details: dict | None = None) -> None:
        self.log_fn(level, message, details or {})

    def broadcast(self, event_type: str, payload: dict | None = None) -> None:
        try:
            self._emit({"type": event_type, **(payload or {})})
        except Exception:  # noqa: BLE001
            pass

    # --- settings -------------------------------------------------------------------

    def settings_path(self) -> str:
        return str(Path(self.application_home) / "settings.json")

    def public_config(self) -> dict:
        return {
            "host": self.config["host"],
            "port": self.config["port"],
            "tls": self.config["tls"],
            "channel": self.config["channel"],
            "nick": self.config["nick"],
            "realname": self.config["realname"],
            "downloadDir": self.config["downloadDir"],
            "bookmarks": self.bookmarks,
        }

    def normalize_connection(self, incoming: dict | None = None, fallback: dict | None = None) -> dict:
        incoming = incoming or {}
        fallback = fallback if fallback is not None else self.active_connection()
        host = str(incoming.get("host") or fallback.get("host") or self.default_connection["host"]).strip()
        try:
            port = int(incoming.get("port") or fallback.get("port") or self.default_connection["port"])
        except (TypeError, ValueError):
            port = self.default_connection["port"]
        channel = str(incoming.get("channel") or fallback.get("channel") or self.default_connection["channel"]).strip()
        nick = str(incoming.get("nick") or fallback.get("nick") or self.default_connection["nick"]).strip()
        realname = str(incoming.get("realname") or fallback.get("realname") or nick or self.default_connection["realname"]).strip()
        return {
            "name": str(incoming.get("name") or f"{host}:{port}").strip(),
            "host": host,
            "port": port if port > 0 else self.default_connection["port"],
            "tls": bool(incoming.get("tls")),
            "channel": channel if channel.startswith("#") else f"#{channel or self.default_connection['channel'].lstrip('#')}",
            "nick": nick,
            "realname": realname,
        }

    def active_connection(self) -> dict:
        return self.normalize_connection(self.config, self.default_connection)

    def apply_connection(self, connection: dict) -> None:
        nxt = self.normalize_connection(connection, self.active_connection())
        self.config.update(
            {
                "host": nxt["host"],
                "port": nxt["port"],
                "tls": nxt["tls"],
                "channel": nxt["channel"],
                "nick": nxt["nick"],
                "realname": nxt["realname"],
            }
        )
        self.current_nick = nxt["nick"]

    def ensure_bookmark(self, connection: dict) -> None:
        nxt = self.normalize_connection(connection)

        def key(item: dict) -> str:
            return f"{item['host']}:{item['port']}:{item['channel']}:{item['nick']}".lower()

        self.bookmarks = [nxt, *[item for item in self.bookmarks if key(item) != key(nxt)]][:24]

    def load_settings(self) -> None:
        from .llm_service import safe_json

        fallback_bookmark = self.normalize_connection({**self.default_connection, **self.active_connection()})
        self.bookmarks = [fallback_bookmark]
        saved = safe_json(self.settings_path())
        if saved:
            if saved.get("active"):
                self.apply_connection(saved["active"])
            if isinstance(saved.get("bookmarks"), list) and saved["bookmarks"]:
                self.bookmarks = [self.normalize_connection(item) for item in saved["bookmarks"]]
        self.ensure_bookmark(self.active_connection())

    def save_settings(self) -> None:
        import json

        payload = {"active": self.active_connection(), "bookmarks": self.bookmarks}
        Path(self.settings_path()).parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def emit_config(self) -> None:
        self.broadcast("config", self.public_config())

    def apply_connection_and_reconnect(self, connection: dict, should_save_bookmark: bool = True) -> None:
        self.apply_connection(connection)
        if should_save_bookmark:
            self.ensure_bookmark(self.active_connection())
        self.save_settings()
        self.emit_config()
        self.broadcast("system", {
            "target": self.config["channel"],
            "text": f"Connection set to {self.config['host']}:{self.config['port']} {self.config['channel']}",
        })
        self.manual_disconnect = False
        self._cancel_timer("reconnect_timer")
        if self.socket:
            self.broadcast("status", {"state": "reconnecting", "text": "Applying connection settings..."})
            self.disconnect_irc(False)
            threading.Timer(0.4, self.connect_irc).start()
        else:
            self.connect_irc()

    def parse_server_connection(self, args_text: str) -> dict | None:
        tokens = [token for token in str(args_text or "").strip().split() if token]
        if not tokens:
            return None

        use_tls = self.config["tls"]
        tls_index = next((index for index, token in enumerate(tokens) if is_tls_token(token)), -1)
        if tls_index != -1:
            use_tls = is_tls_token(tokens[tls_index])
            tokens.pop(tls_index)

        if not tokens:
            return None
        host = tokens.pop(0)

        port = self.config["port"]
        if tokens and tokens[0].isdigit():
            port = int(tokens.pop(0))

        channel = self.config["channel"]
        if tokens and tokens[0].startswith("#"):
            channel = tokens.pop(0)

        nick = self.config["nick"]
        if tokens and not tokens[0].isdigit() and not tokens[0].startswith("#"):
            nick = tokens.pop(0)

        realname = " ".join(tokens) if tokens else nick
        return self.normalize_connection(
            {**self.active_connection(), "host": host, "port": port, "channel": channel, "nick": nick, "realname": realname, "tls": use_tls}
        )

    # --- IRC wire protocol ------------------------------------------------------------

    def _cancel_timer(self, name: str) -> None:
        timer = getattr(self, name, None)
        if timer:
            timer.cancel()
        setattr(self, name, None)

    def write_irc(self, command: str) -> bool:
        with self._write_lock:
            if not self.socket or not self.connected:
                return False
            try:
                self.socket.sendall(f"{command}\r\n".encode("utf-8"))
            except OSError:
                return False
            return True

    def send_irc_text(self, target: str, text: str) -> bool:
        if not self.socket or not self.connected:
            return False
        clean = re.sub(r"[\r\n]+", " ", str(text or "")).strip()
        if not clean:
            return False
        chunks = [clean[index : index + 380] for index in range(0, len(clean), 380)] or [clean]
        sent = False
        for chunk in chunks:
            sent = self.write_irc(f"PRIVMSG {target} :{chunk.strip()}") or sent
        return sent

    def send_action(self, target: str, text: str) -> bool:
        return self.write_irc(f"PRIVMSG {target} :\x01ACTION {text}\x01")

    def connect_irc(self) -> None:
        """Starts connecting and returns immediately.

        The TCP/TLS handshake happens on its own thread: a dead IRC host would otherwise
        block the caller for the full connect timeout, and callers here are HTTP handlers
        (`/api/connect`, `/api/save-config`) that must answer right away — matching the
        non-blocking `net.connect` the Node backend used.
        """
        if self.connected or self.socket or self.connecting:
            return
        self.connecting = True
        threading.Thread(target=self._connect_irc_blocking, daemon=True).start()

    def _connect_irc_blocking(self) -> None:
        self.manual_disconnect = False
        self.registered = False
        self._cancel_timer("nick_retry_timer")
        Path(self.config["downloadDir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["sentDir"]).mkdir(parents=True, exist_ok=True)
        self.line_buffer = ""
        self.names_buffer = []
        self.channel_users = {}
        self.current_nick = self.config["nick"]

        attempt = self.reconnect_attempt or 0
        if attempt <= 1 or attempt % 10 == 0:
            self.log("warn" if attempt > 1 else "info", "Connecting to IRC", {
                "host": self.config["host"], "port": self.config["port"], "tls": self.config["tls"], "attempt": attempt,
            })
        try:
            raw = socket.create_connection((self.config["host"], self.config["port"]), timeout=20)
            if self.config["tls"]:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                raw = context.wrap_socket(raw, server_hostname=self.config["host"])
            raw.settimeout(None)
        except OSError as error:
            self._irc_fail_count += 1
            if self._irc_fail_count == 1 or self._irc_fail_count % 10 == 0:
                self.log("error", "IRC socket error", {
                    "message": str(error), "failCount": self._irc_fail_count,
                    "host": self.config["host"], "port": self.config["port"],
                })
            self.broadcast("status", {"state": "error", "text": str(error)})
            self.connecting = False
            self._schedule_reconnect()
            return

        self.socket = raw
        self.connecting = False
        self.connected = True
        self.reconnect_attempt = 0
        self._irc_fail_count = 0
        self.log("info", "IRC connected")
        self.broadcast("status", {"state": "connected", "text": f"Connected to {self.config['host']}:{self.config['port']}"})
        self.write_irc(f"NICK {self.config['nick']}")
        self.write_irc(f"USER {self.config['nick']} 0 * :{self.config['realname']}")
        threading.Thread(target=self._read_loop, args=(raw,), daemon=True).start()

    def _read_loop(self, sock: socket.socket) -> None:
        while True:
            try:
                chunk = sock.recv(8192)
            except OSError:
                chunk = b""
            if not chunk:
                break
            self.line_buffer += chunk.decode("utf-8", "replace")
            lines = re.split(r"\r?\n", self.line_buffer)
            self.line_buffer = lines.pop() or ""
            for line in lines:
                if line.strip():
                    try:
                        self._handle_irc_line(line)
                    except Exception as error:  # noqa: BLE001
                        self.log("error", "IRC line handler failed", {"message": str(error)})
        self._handle_irc_close()

    def _handle_irc_close(self) -> None:
        self.connected = False
        self.connecting = False
        self.registered = False
        self.socket = None
        self._cancel_timer("nick_retry_timer")
        if self.manual_disconnect or self._irc_fail_count <= 1 or self._irc_fail_count % 10 == 0:
            self.log("info", "IRC disconnected", {"manualDisconnect": self.manual_disconnect, "failCount": self._irc_fail_count})
        self.broadcast("status", {"state": "disconnected", "text": "Disconnected"})
        self._update_users()
        self._schedule_reconnect()

    def disconnect_irc(self, user_requested: bool = False) -> None:
        if user_requested:
            self.manual_disconnect = True
        if not self.socket:
            return
        if self.connected:
            self.write_irc("QUIT :Subjective Chat closed")
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.socket.close()
        except OSError:
            pass

    def reconnect_now(self) -> None:
        self.manual_disconnect = False
        self._cancel_timer("reconnect_timer")
        if self.socket:
            self.broadcast("status", {"state": "reconnecting", "text": "Reconnecting..."})
            self.disconnect_irc(False)
            threading.Timer(1.5, self.connect_irc).start()
            return
        self.connect_irc()

    def _schedule_reconnect(self) -> None:
        if self.shutting_down or self.manual_disconnect or self.reconnect_timer:
            return
        self.reconnect_attempt += 1
        delay = min(30.0, 1.0 * 2 ** min(self.reconnect_attempt, 5))
        self.broadcast("status", {"state": "reconnecting", "text": f"Reconnecting in {round(delay)}s"})

        def fire() -> None:
            self.reconnect_timer = None
            self.connect_irc()

        self.reconnect_timer = threading.Timer(delay, fire)
        self.reconnect_timer.daemon = True
        self.reconnect_timer.start()

    def shutdown(self) -> None:
        self.shutting_down = True
        self.llm_service.backend.close()
        self._cancel_timer("reconnect_timer")
        self._cancel_timer("nick_retry_timer")
        self.su_session_manager.disconnect_all()
        if self.socket:
            self.disconnect_irc(True)

    def _parse_irc_line(self, line: str) -> dict:
        rest = line
        prefix = ""
        if rest.startswith(":"):
            space = rest.find(" ")
            prefix = rest[1:space]
            rest = rest[space + 1 :]
        trailing_index = rest.find(" :")
        trailing = None
        if trailing_index != -1:
            trailing = rest[trailing_index + 2 :]
            rest = rest[:trailing_index]
        parts = rest.strip().split() if rest.strip() else []
        command = parts.pop(0) if parts else ""
        if trailing is not None:
            parts.append(trailing)
        return {"prefix": prefix, "command": command, "params": parts, "raw": line}

    @staticmethod
    def _nick_from_prefix(prefix: str) -> str:
        return prefix.split("!")[0] or prefix

    @staticmethod
    def _normalize_nick(raw_nick: str) -> str:
        return re.sub(r"^[~&@%+]+", "", raw_nick)

    def _user_record(self, raw_nick: str) -> dict:
        modes_match = re.match(r"^[~&@%+]+", raw_nick)
        nick = self._normalize_nick(raw_nick)
        return {"nick": nick, "modes": modes_match.group(0) if modes_match else "", "self": nick.lower() == self.current_nick.lower()}

    def _set_channel_user(self, raw_nick: str) -> None:
        record = self._user_record(raw_nick)
        if record["nick"]:
            self.channel_users[record["nick"].lower()] = record

    def _remove_channel_user(self, nick: str) -> None:
        self.channel_users.pop(nick.lower(), None)

    def _rename_channel_user(self, old_nick: str, new_nick: str) -> None:
        existing = self.channel_users.pop(old_nick.lower(), None)
        if existing:
            self.channel_users[new_nick.lower()] = {
                **existing,
                "nick": new_nick,
                "self": existing["self"] or new_nick == self.current_nick,
            }

    def _update_users(self) -> None:
        users = sorted(self.channel_users.values(), key=lambda user: user["nick"])
        self.broadcast("users", {"channel": self.config["channel"], "users": users})

    @staticmethod
    def _clean_channel_name(channel: Any) -> str:
        return re.sub(r"^:", "", str(channel or ""))

    def _is_configured_channel(self, channel: Any) -> bool:
        return self._clean_channel_name(channel).lower() == self.config["channel"].lower()

    def _request_names_soon(self) -> None:
        if not self.registered or self.names_refresh_timer:
            return

        def fire() -> None:
            self.names_refresh_timer = None
            if self.registered and self.connected:
                self.names_buffer = []
                self.write_irc(f"NAMES {self.config['channel']}")

        self.names_refresh_timer = threading.Timer(0.75, fire)
        self.names_refresh_timer.daemon = True
        self.names_refresh_timer.start()

    def _retry_configured_nick_soon(self) -> None:
        if self.nick_retry_timer or self.shutting_down:
            return

        def fire() -> None:
            self.nick_retry_timer = None
            if not self.connected or self.registered:
                return
            self.broadcast("status", {"state": "reconnecting", "text": f"Waiting for {self.config['nick']} to disconnect..."})
            self.write_irc(f"NICK {self.config['nick']}")

        self.nick_retry_timer = threading.Timer(5.0, fire)
        self.nick_retry_timer.daemon = True
        self.nick_retry_timer.start()

    def _conversation_target(self, target: str, sender: str) -> str:
        if target.lower() == self.current_nick.lower():
            return sender
        if target.lower() == self.config["channel"].lower():
            return self.config["channel"]
        return target

    def _channel_recipients(self) -> list[str]:
        return [user["nick"] for user in self.channel_users.values() if user["nick"].lower() != self.current_nick.lower()]

    def _handle_irc_line(self, line: str) -> None:
        message = self._parse_irc_line(line)
        command = message["command"]
        params = message["params"]

        if command == "PING":
            self.write_irc(f"PONG :{params[0] if params else ''}")
            return

        if command == "001":
            self.registered = True
            self._cancel_timer("nick_retry_timer")
            self.write_irc(f"JOIN {self.config['channel']}")
            self.write_irc(f"NAMES {self.config['channel']}")
            for channel in self.active_llm_irc_channels:
                self.write_irc(f"JOIN {channel}")
            self.broadcast("system", {"target": self.config["channel"], "text": f"Joined {self.config['channel']}"})
            return

        if command == "353":
            if len(params) > 2 and self._is_configured_channel(params[2]):
                self.names_buffer.extend([nick for nick in str(params[3] if len(params) > 3 else "").split() if nick])
            return

        if command == "366":
            if not params or not self._is_configured_channel(params[1] if len(params) > 1 else ""):
                return
            self.channel_users = {}
            for nick in self.names_buffer:
                self._set_channel_user(nick)
            self.names_buffer = []
            self._update_users()
            return

        if command == "433":
            self.broadcast("status", {"state": "reconnecting", "text": f"{self.config['nick']} is still disconnecting. Retrying..."})
            self._retry_configured_nick_soon()
            return

        if command == "JOIN":
            nick = self._nick_from_prefix(message["prefix"])
            if not params or not self._is_configured_channel(params[0]):
                return
            self._set_channel_user(nick)
            self._update_users()
            self._request_names_soon()
            self.broadcast("join", {"target": self.config["channel"], "nick": nick, "channel": self._clean_channel_name(params[0])})
            return

        if command == "PART":
            nick = self._nick_from_prefix(message["prefix"])
            if not params or not self._is_configured_channel(params[0]):
                return
            self._remove_channel_user(nick)
            self._update_users()
            self._request_names_soon()
            self.broadcast("part", {"target": self.config["channel"], "nick": nick, "text": params[-1] if params else ""})
            return

        if command == "QUIT":
            nick = self._nick_from_prefix(message["prefix"])
            self._remove_channel_user(nick)
            self._update_users()
            self._request_names_soon()
            self.broadcast("part", {"target": self.config["channel"], "nick": nick, "text": params[-1] if params else ""})
            return

        if command == "NICK":
            old_nick = self._nick_from_prefix(message["prefix"])
            new_nick = params[0] if params else ""
            if old_nick.lower() == self.current_nick.lower():
                self.current_nick = new_nick
            self._rename_channel_user(old_nick, new_nick)
            self._update_users()
            self._request_names_soon()
            self.broadcast("system", {"target": self.config["channel"], "text": f"{old_nick} is now {new_nick}"})
            return

        if command == "PRIVMSG":
            sender = self._nick_from_prefix(message["prefix"])
            target = params[0] if params else ""
            text = params[1] if len(params) > 1 else ""

            if text.startswith("\x01") and text.endswith("\x01"):
                self._handle_ctcp(sender, target, text[1:-1])
                return

            llm_target = self.active_llm_irc_channels.get(str(target).lower())
            if llm_target:
                if sender.lower() != self.current_nick.lower():
                    threading.Thread(
                        target=self._answer_llm_in_background, args=(llm_target, text, sender), daemon=True
                    ).start()
                return

            conversation = self._conversation_target(target, sender)
            self.broadcast("message", {
                "target": conversation,
                "from": sender,
                "text": text,
                "mine": sender.lower() == self.current_nick.lower(),
                "time": time.time() * 1000,
                "private": conversation == sender,
            })
            return

        if re.match(r"^\d+$", command):
            return
        self.broadcast("raw", {"text": line})

    def _answer_llm_in_background(self, target: str, text: str, sender: str) -> None:
        try:
            self.answer_llm_message(target, text, sender, mirror_to_irc=True)
        except Exception as error:  # noqa: BLE001
            self.log("error", "Failed to answer IRC LLM channel message", {"target": target, "from": sender, "message": str(error)})

    # --- DCC ---------------------------------------------------------------------------

    def _handle_ctcp(self, sender: str, target: str, body: str) -> None:
        if body == "VERSION":
            self.write_irc(f"NOTICE {sender} :\x01VERSION Subjective Chat web client\x01")
            return
        if not body.startswith("DCC "):
            return
        parts = self._split_dcc(body)
        if len(parts) < 5 or parts[1] != "SEND":
            return

        file_name = os.path.basename(parts[2])
        host = self._decode_dcc_address(parts[3])
        try:
            port = int(parts[4])
            size = int(parts[5]) if len(parts) > 5 else 0
        except ValueError:
            return
        if not file_name or not host or not port:
            return

        hinted_target = self._dcc_context_target(parts)
        conversation = hinted_target or self._conversation_target(target, sender)
        threading.Thread(
            target=self._receive_dcc_file,
            args=({"from": sender, "target": conversation, "fileName": file_name, "host": host, "port": port, "size": size},),
            daemon=True,
        ).start()

    @staticmethod
    def _split_dcc(body: str) -> list[str]:
        matches = re.findall(r'"[^"]+"|\S+', body)
        return [part[1:-1] if part.startswith('"') and part.endswith('"') else part for part in matches]

    @staticmethod
    def _decode_dcc_address(value: str) -> str:
        if re.match(r"^\d+$", value):
            number = int(value)
            return ".".join(str((number >> shift) & 255) for shift in (24, 16, 8, 0))
        return value

    @staticmethod
    def _encode_dcc_address(host: str) -> str:
        parts = host.split(".")
        if len(parts) != 4:
            return host
        try:
            numbers = [int(part) for part in parts]
        except ValueError:
            return host
        if any(number < 0 or number > 255 for number in numbers):
            return host
        return str((numbers[0] << 24) + (numbers[1] << 16) + (numbers[2] << 8) + numbers[3])

    @staticmethod
    def _dcc_context_target(parts: list[str]) -> str:
        from urllib.parse import unquote

        try:
            index = parts.index("SUBJECTIVE_TARGET")
        except ValueError:
            return ""
        if index + 1 >= len(parts):
            return ""
        try:
            return unquote(parts[index + 1])
        except Exception:  # noqa: BLE001
            return parts[index + 1]

    def _is_image_file(self, file_name: str) -> bool:
        if IMAGE_EXT_RE.search(file_name):
            return True
        if not Path(file_name).exists():
            return False
        try:
            with open(file_name, "rb") as handle:
                header = handle.read(16)
            if len(header) < 4:
                return False
            if header[:3] == b"\xff\xd8\xff":
                return True
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return True
            if header[:6] in (b"GIF87a", b"GIF89a"):
                return True
            if header[:2] == b"BM":
                return True
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return True
            if header[4:12] == b"ftypavif":
                return True
        except OSError as error:
            self.log("error", "Image type detection failed", {"fileName": file_name, "message": str(error)})
        return False

    @staticmethod
    def _safe_file_name(file_name: str) -> str:
        return re.sub(r"[^\w .()\[\]-]", "_", os.path.basename(file_name))

    @staticmethod
    def _unique_path(file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return str(path)
        index = 1
        while True:
            candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
            if not candidate.exists():
                return str(candidate)
            index += 1

    def _local_dcc_host(self) -> str:
        if self.config["dccPublicHost"]:
            return self.config["dccPublicHost"]
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            address = probe.getsockname()[0]
            probe.close()
            if address and not address.startswith("127."):
                return address
        except OSError:
            pass
        return self.config["host"]

    def _receive_dcc_file(self, transfer: dict) -> None:
        Path(self.config["downloadDir"]).mkdir(parents=True, exist_ok=True)
        output_path = self._unique_path(str(Path(self.config["downloadDir"]) / self._safe_file_name(transfer["fileName"])))
        self.broadcast("dcc-start", {
            "target": transfer["target"],
            "direction": "incoming",
            "from": transfer["from"],
            "fileName": os.path.basename(output_path),
            "size": transfer["size"],
            "image": self._is_image_file(output_path),
        })

        received = 0
        try:
            with socket.create_connection((transfer["host"], transfer["port"]), timeout=30) as incoming, open(
                output_path, "wb"
            ) as handle:
                while True:
                    chunk = incoming.recv(65536)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    incoming.sendall(struct.pack(">I", received & 0xFFFFFFFF))
                    self.broadcast("dcc-progress", {
                        "target": transfer["target"],
                        "direction": "incoming",
                        "fileName": os.path.basename(output_path),
                        "received": received,
                        "size": transfer["size"],
                    })
        except OSError as error:
            self.broadcast("dcc-error", {
                "target": transfer["target"], "direction": "incoming", "fileName": transfer["fileName"], "text": str(error),
            })
            return

        self.broadcast("dcc-complete", {
            "target": transfer["target"],
            "direction": "incoming",
            "from": transfer["from"],
            "fileName": os.path.basename(output_path),
            "filePath": output_path,
            "url": Path(output_path).as_uri(),
            "size": received,
            "image": self._is_image_file(output_path),
        })

    def send_dcc_buffer(self, target: str, buffer: bytes, file_name: str, mime_type: str = "") -> dict:
        if not self.registered:
            self.broadcast("system", {"target": target or self.config["channel"], "text": "Connect before sending an image."})
            return {"ok": False}

        conversation = target or self.config["channel"]
        recipients = self._channel_recipients() if conversation.startswith("#") else [conversation]
        if not recipients:
            self.broadcast("system", {"target": conversation, "text": "No DCC recipients are available."})

        Path(self.config["sentDir"]).mkdir(parents=True, exist_ok=True)
        clean_name = self._safe_file_name(file_name or f"pasted-{int(time.time() * 1000)}.png")
        saved_path = self._unique_path(str(Path(self.config["sentDir"]) / clean_name))
        with open(saved_path, "wb") as handle:
            handle.write(buffer)

        for recipient in recipients:
            self._announce_dcc_send(
                recipient, buffer, os.path.basename(saved_path), conversation if conversation.startswith("#") else ""
            )

        self.broadcast("dcc-complete", {
            "target": conversation,
            "direction": "outgoing",
            "from": self.current_nick,
            "fileName": os.path.basename(saved_path),
            "filePath": saved_path,
            "url": Path(saved_path).as_uri(),
            "size": len(buffer),
            "image": self._is_image_file(saved_path) or str(mime_type or "").startswith("image/"),
        })
        return {"ok": True, "recipients": len(recipients)}

    def _announce_dcc_send(self, recipient: str, buffer: bytes, file_name: str, source_target: str = "") -> None:
        from urllib.parse import quote

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        server.settimeout(120)
        port = server.getsockname()[1]

        def serve() -> None:
            try:
                peer, _ = server.accept()
            except OSError:
                server.close()
                return
            try:
                peer.sendall(buffer)
                self.broadcast("dcc-progress", {
                    "target": recipient, "direction": "outgoing", "fileName": file_name,
                    "received": len(buffer), "size": len(buffer),
                })
            except OSError as error:
                self.log("error", "DCC peer socket error", {"recipient": recipient, "fileName": file_name, "message": str(error)})
            finally:
                peer.close()
                server.close()

        threading.Thread(target=serve, daemon=True).start()
        host = self._local_dcc_host()
        quoted_name = f'"{file_name}"' if " " in file_name else file_name
        context = f" SUBJECTIVE_TARGET {quote(source_target)}" if source_target else ""
        self.write_irc(
            f"PRIVMSG {recipient} :\x01DCC SEND {quoted_name} {self._encode_dcc_address(host)} {port} {len(buffer)}{context}\x01"
        )

    # --- slash commands -----------------------------------------------------------

    def execute_slash_command(self, command_line: str, default_target: str = "") -> dict:
        raw = str(command_line or "").strip()
        if not raw:
            return {"ok": False}

        command = raw[1:] if raw.startswith("/") else raw
        space_index = command.find(" ")
        name = (command if space_index == -1 else command[:space_index]).lower()
        args_text = "" if space_index == -1 else command[space_index + 1 :].strip()
        fallback_target = default_target or self.config["channel"]

        def first_token(text: str) -> str:
            tokens = text.split()
            return tokens[0].strip() if tokens else ""

        if name == "server":
            nxt = self.parse_server_connection(args_text)
            if not nxt:
                return {"ok": False}
            self.apply_connection_and_reconnect(nxt, True)
            return {"ok": True, "focusTarget": nxt["channel"]}

        if name == "nick":
            next_nick = first_token(args_text)
            if not next_nick or not self.write_irc(f"NICK {next_nick}"):
                return {"ok": False}
            self.config["nick"] = next_nick
            self.current_nick = next_nick
            self.save_settings()
            self.emit_config()
            self.broadcast("system", {"target": self.config["channel"], "text": f"Nickname changed to {next_nick}"})
            return {"ok": True}

        if name == "join":
            next_channel = first_token(args_text)
            if not next_channel:
                return {"ok": False}
            channel_name = next_channel if next_channel.startswith("#") else f"#{next_channel}"
            self.config["channel"] = channel_name
            self.save_settings()
            self.emit_config()
            self.channel_users = {}
            self._update_users()
            if self.write_irc(f"JOIN {channel_name}"):
                self.write_irc(f"NAMES {channel_name}")
                self.broadcast("system", {"target": channel_name, "text": f"Joining {channel_name}"})
                return {"ok": True, "focusTarget": channel_name}
            return {"ok": False}

        if name == "part":
            next_token = first_token(args_text)
            target = next_token if next_token.startswith("#") else fallback_target
            reason = args_text[len(next_token) :].strip() if next_token.startswith("#") else args_text.strip()
            if not target:
                return {"ok": False}
            self.channel_users = {}
            self._update_users()
            if not self.write_irc(f"PART {target} :{reason}" if reason else f"PART {target}"):
                return {"ok": False}
            self.broadcast("system", {"target": target, "text": f"Parted {target}"})
            return {"ok": True}

        if name == "msg":
            target = first_token(args_text)
            text = args_text[len(target) :].strip() if target else ""
            if not target or not text or not self.write_irc(f"PRIVMSG {target} :{text}"):
                return {"ok": False}
            self.broadcast("message", {
                "target": target, "from": self.current_nick, "text": text, "mine": True,
                "time": time.time() * 1000, "private": not target.startswith("#"),
            })
            return {"ok": True, "focusTarget": target}

        if name == "query":
            target = first_token(args_text)
            text = args_text[len(target) :].strip() if target else ""
            if not target:
                return {"ok": False}
            if text and not self.write_irc(f"PRIVMSG {target} :{text}"):
                return {"ok": False}
            if text:
                self.broadcast("message", {
                    "target": target, "from": self.current_nick, "text": text, "mine": True,
                    "time": time.time() * 1000, "private": True,
                })
            return {"ok": True, "focusTarget": target}

        if name == "me":
            if not args_text or not self.send_action(fallback_target, args_text):
                return {"ok": False}
            self.broadcast("message", {
                "target": fallback_target, "from": self.current_nick,
                "text": f"* {self.current_nick} {args_text}", "mine": True,
                "time": time.time() * 1000, "private": not fallback_target.startswith("#"),
            })
            return {"ok": True}

        if name == "notice":
            target = first_token(args_text)
            text = args_text[len(target) :].strip() if target else ""
            if not target or not text or not self.write_irc(f"NOTICE {target} :{text}"):
                return {"ok": False}
            self.broadcast("system", {"target": target, "text": f"Notice to {target}: {text}"})
            return {"ok": True, "focusTarget": target}

        if name == "topic":
            if not args_text or not self.write_irc(f"TOPIC {fallback_target} :{args_text}"):
                return {"ok": False}
            self.broadcast("system", {"target": fallback_target, "text": f"Topic updated for {fallback_target}"})
            return {"ok": True}

        if name == "names":
            next_target = first_token(args_text)
            target = next_target if next_target.startswith("#") else fallback_target
            if not self.write_irc(f"NAMES {target}"):
                return {"ok": False}
            return {"ok": True, "focusTarget": target}

        if name == "whois":
            target = first_token(args_text)
            if not target or not self.write_irc(f"WHOIS {target}"):
                return {"ok": False}
            return {"ok": True, "focusTarget": target}

        if name == "away":
            if not self.write_irc(f"AWAY :{args_text}" if args_text else "AWAY"):
                return {"ok": False}
            self.broadcast("system", {
                "target": fallback_target, "text": f"Set away: {args_text}" if args_text else "Away cleared",
            })
            return {"ok": True}

        if name == "connect":
            self.reconnect_now()
            return {"ok": True}

        if name in ("disconnect", "quit"):
            self.disconnect_irc(True)
            return {"ok": True}

        if name == "raw":
            return {"ok": True} if self.write_irc(args_text or command) else {"ok": False}

        return {"ok": True} if self.write_irc(command) else {"ok": False}

    # --- LLM agent conversations ----------------------------------------------------

    def activate_llm_irc_channel(self, conversation: dict | None) -> dict:
        if not conversation or not conversation.get("ircChannel"):
            return {"ok": False, "error": "Conversation has no IRC channel."}
        self.active_llm_irc_channels[conversation["ircChannel"].lower()] = conversation["target"]
        if self.connected:
            self.write_irc(f"JOIN {conversation['ircChannel']}")
            self.write_irc(f"NAMES {conversation['ircChannel']}")
        return {"ok": True, "channel": conversation["ircChannel"], "connected": self.connected}

    @staticmethod
    def llm_error_text(label: str, response: dict) -> str:
        base = f"{label} error ({response.get('status') or 'error'}): {response.get('error') or 'Unknown error'}"
        if response.get("status") != "timeout":
            return base
        return (
            f"{base} {label} has no memory of this attempt - your next message starts a brand-new session from scratch."
        )

    def answer_llm_message(self, target: str, text: str, sender: str | None = None, mirror_to_irc: bool = True) -> dict:
        sender = sender or self.current_nick
        conversation = self.llm_service.find_conversation(target)
        if not conversation:
            return {"ok": False, "status": "not_found", "error": "Conversation was not found."}
        provider = conversation.get("provider") or "codex"
        label = provider_label(provider)
        self.broadcast("llm-busy", {"target": target, "busy": True})
        self.log("info", "LLM answer start", {
            "target": target, "provider": provider, "from": sender, "promptChars": len(str(text or "")),
        })
        try:
            response = self.llm_service.send_message(target, text, [])
            # Only echo the user's line once the turn actually happened: broadcasting on a
            # failed send published an unsent message as a confirmed one, which the renderer
            # then kept in place of its own optimistic bubble.
            if response.get("ok") and response.get("userMessage"):
                self.broadcast("llm-message", {"target": target, "message": response["userMessage"], "provider": provider, "from": sender})
            if response.get("ok"):
                self.broadcast("llm-message", {"target": target, "message": annotate_message(response["assistantMessage"], provider), "provider": provider})
                self.broadcast("llm-conversation", {"conversation": tail_conversation_for_client(response["conversation"])})
                if response.get("cursorIdeHint"):
                    self.broadcast("system", {
                        "target": target,
                        "text": (
                            "Web reply is from cursor-agent. Cursor will not show it while this tab stays open — it "
                            "overwrites the chat index from memory. Close agents_chat_dashboard in Cursor, click ↻ here, "
                            "then reopen the tab."
                        ),
                    })
                if mirror_to_irc and self.connected and conversation.get("ircChannel"):
                    self.send_irc_text(conversation["ircChannel"], f"{label}: {response['assistantMessage']['text']}")
            else:
                if self.llm_service.is_auth_failure(response) and not response.get("sessions"):
                    response["sessions"] = self.llm_service.list_provider_sessions(provider)
                self.log("error", "LLM answer failed", {
                    "target": target, "provider": provider, "status": response.get("status") or "",
                    "error": response.get("error") or "", "authFailure": self.llm_service.is_auth_failure(response),
                })
                error_text = self.llm_error_text(label, response)
                self.broadcast("system", {"target": target, "text": error_text})
                if self.llm_service.is_auth_failure(response):
                    sessions = response.get("sessions") or {}
                    self.broadcast("llm-choose-session", {
                        "target": target, "provider": provider, "text": text,
                        "currentUser": sessions.get("currentUser"), "sessions": sessions.get("users") or [],
                    })
                if mirror_to_irc and self.connected and conversation.get("ircChannel"):
                    self.send_irc_text(conversation["ircChannel"], error_text)
            return response
        finally:
            self.broadcast("llm-busy", {"target": target, "busy": False})

    def send_llm_message(self, target: str, text: str, files: list | None = None) -> dict:
        files = files or []
        conversation = self.llm_service.find_conversation(target)
        if self.connected and conversation and conversation.get("ircChannel"):
            self.send_irc_text(conversation["ircChannel"], text)
        if files:
            provider = (conversation or {}).get("provider") or "codex"
            label = provider_label(provider)
            self.broadcast("llm-busy", {"target": target, "busy": True})
            try:
                response = self.llm_service.send_message(target, text, files)
                if response.get("ok") and response.get("userMessage"):
                    self.broadcast("llm-message", {
                        "target": target, "message": response["userMessage"], "provider": provider, "from": self.current_nick,
                    })
                if response.get("ok"):
                    self.broadcast("llm-message", {"target": target, "message": annotate_message(response["assistantMessage"], provider), "provider": provider})
                    self.broadcast("llm-conversation", {"conversation": tail_conversation_for_client(response["conversation"])})
                    if self.connected and conversation and conversation.get("ircChannel"):
                        self.send_irc_text(conversation["ircChannel"], f"{label}: {response['assistantMessage']['text']}")
                else:
                    self.broadcast("system", {"target": target, "text": self.llm_error_text(label, response)})
                    if self.llm_service.is_auth_failure(response):
                        sessions = response.get("sessions") or {}
                        self.broadcast("llm-choose-session", {
                            "target": target, "provider": provider, "text": text,
                            "currentUser": sessions.get("currentUser"), "sessions": sessions.get("users") or [],
                        })
                return response
            finally:
                self.broadcast("llm-busy", {"target": target, "busy": False})
        try:
            return self.answer_llm_message(target, text, self.current_nick, True)
        except Exception as error:  # noqa: BLE001
            self.log("error", "LLM message crashed", {"target": target, "message": str(error)})
            return {"ok": False, "status": "error", "error": str(error)}

    # --- "Continue as..." su sessions -------------------------------------------------

    def switch_conversation_owner(self, target: str, username: str, provider: str) -> dict:
        try:
            lightweight = self.llm_service.find_conversation(target, include_messages=False)
            cross_provider = bool(lightweight) and bool(provider) and str(provider).lower() != str(lightweight["provider"]).lower()
            sibling = self.llm_service.create_sibling_for_user(target, username, provider if cross_provider else None)
            self.activate_llm_irc_channel(sibling)
            self.broadcast("llm-conversation", {"conversation": sibling})
            self.broadcast("llm-switch-user", {
                "sourceTarget": target, "target": sibling["target"], "osUser": username, "provider": sibling["provider"],
            })
            return {"conversation": sibling}
        except Exception as error:  # noqa: BLE001
            self.log("error", "Failed to create sibling conversation for OS user", {
                "username": username, "provider": provider, "message": str(error),
            })
            self.broadcast("system", {"target": target, "text": f"Could not continue as {username}: {error}"})
            return {"error": str(error)}

    def _handoff_in_background(self, target: str, handoff: str) -> None:
        try:
            self.send_llm_message(target, handoff)
        except Exception as error:  # noqa: BLE001
            self.log("error", "Continue-as handoff message failed", {"target": target, "message": str(error)})

    def continue_conversation_as_user(self, target: str, username: str, provider: str | None = None) -> dict:
        """`provider` is optional and defaults to the conversation's own provider — the
        auth-retry session picker calls this without it and must switch only the OS user,
        never the agent. "Continue as…" passes it to also switch which CLI answers."""
        conversation = self.llm_service.find_conversation(target)
        if not conversation:
            return {"ok": False, "error": "Conversation was not found."}
        if not is_valid_username(username):
            return {"ok": False, "error": "Invalid username."}
        target_provider = str(provider).lower() if provider and str(provider).lower() in LOGIN_PROVIDERS else conversation["provider"]

        if self.su_session_manager.is_provider_ready(username, target_provider):
            return {"ok": True, **self.switch_conversation_owner(target, username, target_provider)}

        self.broadcast("switchuser-terminal-needed", {"username": username, "provider": target_provider})

        def connect() -> None:
            try:
                self.su_session_manager.ensure_session(username)
                if target_provider in LOGIN_PROVIDERS:
                    self.su_session_manager.ensure_provider_login(username, target_provider)
                self.switch_conversation_owner(target, username, target_provider)
            except Exception as error:  # noqa: BLE001
                self.log("error", "Failed to connect as OS user", {
                    "username": username, "provider": target_provider, "message": str(error),
                })
                self.broadcast("switchuser-status", {"username": username, "status": "error", "error": str(error)})
                self.broadcast("system", {"target": target, "text": f"Could not continue as {username}: {error}"})

        threading.Thread(target=connect, daemon=True).start()
        existing = self.su_session_manager.get_status(username)
        return {"ok": True, "status": existing.get("status") or "connecting"}

    def attach_su_viewer_broadcast(self, username: str) -> None:
        """Wires a "Continue as..." pty's output into the same broadcast channel as every
        other event, tagged by username, so any connected browser tab can render it.
        Safe to call repeatedly — attach/detach key off the stored callback."""
        if username in self._su_viewer_broadcasts:
            return

        def viewer(chunk: str) -> None:
            self.broadcast("switchuser-pty-data", {"username": username, "chunk": chunk})

        self._su_viewer_broadcasts[username] = viewer
        self.su_session_manager.attach_viewer(username, viewer)

    def detach_su_viewer_broadcast(self, username: str) -> None:
        viewer = self._su_viewer_broadcasts.pop(username, None)
        if not viewer:
            return
        self.su_session_manager.detach_viewer(username, viewer)

    def build_context_menu_model(self, target: str) -> dict:
        """Plain-data description of the desktop app's native context menu, for the web
        frontend to render as an HTML dropdown."""
        conversation = self.llm_service.find_conversation(target)
        if not conversation or conversation.get("kind") != "llm":
            return {"renamable": False, "continueAs": None}
        if not conversation.get("writable"):
            return {"renamable": True, "continueAs": None}

        users_result = list_system_users()
        users = users_result["users"] if users_result["ok"] else []
        statuses = {entry["username"]: entry for entry in self.su_session_manager.list_statuses()}

        listed = list_provider_sessions()
        by_user = {user["username"]: user.get("providers") or {} for user in listed.get("users") or []}
        # "Continue as…" already picks an OS user; grouping by every writable agent CLI as
        # well turns it into a full (LLM × user) picker, so switching the model answering a
        # conversation doesn't require starting a brand new chat from scratch.
        current_owner = conversation.get("osUser") or current_username()
        providers = [
            {
                "key": key,
                "label": provider_label(key),
                "users": [
                    {
                        "username": user["username"],
                        "status": (statuses.get(user["username"]) or {}).get("status") or "idle",
                        "ready": bool((by_user.get(user["username"]) or {}).get(key, {}).get("ready")),
                        "unknown": bool((by_user.get(user["username"]) or {}).get(key, {}).get("unknown")),
                        "current": key == conversation["provider"] and user["username"] == current_owner,
                    }
                    for user in users
                ],
            }
            for key in sorted(LOGIN_PROVIDERS)
        ]
        active_sessions = [
            {"username": entry["username"], "status": entry["status"]}
            for entry in statuses.values()
            if entry["status"] not in ("idle", "disconnected")
        ]
        return {"renamable": True, "continueAs": {"providers": providers, "activeSessions": active_sessions}}
