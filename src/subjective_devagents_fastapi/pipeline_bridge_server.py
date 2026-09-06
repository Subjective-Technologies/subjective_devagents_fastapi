#!/usr/bin/env python3
"""Small command-execution bridge used by SuSessionManager (src/suSessionManager.js).

Why this exists: once Subjective Chat has `su - <user>`'d into another OS account's
authenticated shell (via a real pty, with the person typing that user's own password),
every *subsequent* one-shot `codex`/`claude` pipeline call for that user still needs to
run as that user. Piping each call's JSON request through the pty as literal keystrokes
is unreliable for anything beyond a few KB (Linux's canonical-mode tty line discipline
caps a single input line at ~4095 bytes - MAX_CANON - which realistic LLM prompts and
responses regularly exceed), and parsing ANSI/echo noise back out of a shell transcript
is brittle.

So instead: this script is started *once*, as a background child of that already
`su`'d shell (i.e. it only ever runs with credentials the human explicitly authenticated
for), and exposes a tiny local RPC service over a Unix domain socket. Subjective Chat's
main process then talks to it directly over that socket - a clean byte channel with no
tty framing limits - for every subsequent message in that user's conversations, without
re-invoking `su` or re-touching the pty.

Protocol: each message (both directions) is a 4-byte big-endian length prefix followed
by that many bytes of UTF-8 JSON.
  Request:  {"secret": "<token>", "argv": [...], "stdin": "<str>"}
            {"secret": "<token>", "action": "shutdown"}
  Response: {"ok": true, "stdout": "...", "stderr": "...", "returncode": 0}
            {"ok": false, "error": "..."}

The socket file is created world read/write (0o666) so the original OS user's process
(a different uid) can connect to a socket owned by this user - cross-connection access
control is enforced at the RPC layer via the shared `--secret`, generated fresh per
session by the Node side and passed only over the already-authenticated pty, not by
filesystem permissions on the socket itself.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time

IDLE_SHUTDOWN_SECONDS = 30 * 60
HEADER = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("peer closed connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(conn: socket.socket) -> dict:
    header = _recv_exact(conn, HEADER.size)
    (length,) = HEADER.unpack(header)
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise ValueError(f"refusing to read message of size {length}")
    payload = _recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))


def write_message(conn: socket.socket, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    conn.sendall(HEADER.pack(len(body)))
    conn.sendall(body)


class BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: "BridgeServer" = self.server  # type: ignore[assignment]
        try:
            request = read_message(self.request)
        except Exception as error:  # noqa: BLE001 - always reply, never crash the server
            try:
                write_message(self.request, {"ok": False, "error": f"bad request: {error}"})
            except OSError:
                pass
            return

        server.touch()
        if not hmac.compare_digest(str(request.get("secret", "")), server.secret):
            write_message(self.request, {"ok": False, "error": "bad secret"})
            return

        if request.get("action") == "shutdown":
            write_message(self.request, {"ok": True})
            server.request_shutdown()
            return

        argv = request.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            write_message(self.request, {"ok": False, "error": "argv must be a non-empty list of strings"})
            return
        stdin_payload = request.get("stdin", "")
        if not isinstance(stdin_payload, str):
            stdin_payload = json.dumps(stdin_payload)

        try:
            completed = subprocess.run(
                argv,
                input=stdin_payload,
                capture_output=True,
                text=True,
                timeout=request.get("timeout_seconds", 660),
            )
            write_message(self.request, {
                "ok": True,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode
            })
        except subprocess.TimeoutExpired as error:
            write_message(self.request, {"ok": False, "error": f"command timed out: {error}"})
        except Exception as error:  # noqa: BLE001 - always reply with a structured error
            write_message(self.request, {"ok": False, "error": str(error)})


class BridgeServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, secret: str):
        self.secret = secret
        self._last_activity = time.monotonic()
        self._lock = threading.Lock()
        super().__init__(socket_path, BridgeHandler)
        os.chmod(socket_path, 0o666)

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_activity

    def request_shutdown(self) -> None:
        threading.Thread(target=self.shutdown, daemon=True).start()


def watch_idle_timeout(server: BridgeServer, socket_path: str) -> None:
    while True:
        time.sleep(30)
        if server.idle_seconds() > IDLE_SHUTDOWN_SECONDS:
            server.request_shutdown()
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="Path of the Unix domain socket to listen on.")
    parser.add_argument("--secret", required=True, help="Shared secret required on every request.")
    args = parser.parse_args()

    socket_path = args.socket
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    server = BridgeServer(socket_path, args.secret)
    watchdog = threading.Thread(target=watch_idle_timeout, args=(server, socket_path), daemon=True)
    watchdog.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
