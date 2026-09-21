"""SSH chats: a conversation whose agent is a shell on another machine.

A DevAgents chat sends one message and waits for one reply. A terminal has neither —
it is a byte stream with no request, no response and no boundary between them. This
module is the adapter, and the whole design is about where that boundary comes from.

**Why a pty and not the framed command mode.** `open_shell()` already gives clean
per-command framing, and for a one-shot pipeline turn it is the right thing. But a
chat is a conversation: the interesting case is a command that *asks something back* —
`sudo` wanting a password, `claude` asking whether the folder is trusted, `apt`
asking to continue. With pipes those either fail outright or hang unanswerable. On a
tty they prompt, and the operator's next chat message is the answer. That is worth
the framing this file then has to reconstruct.

**How a turn ends.** The shell's prompt is made into a marker the moment the session
opens (`PS1`), so a turn normally ends the instant that marker comes back — exact,
not guessed. A full-screen program never returns to a prompt, so there is a second
rule: output that has been quiet for `QUIET_SECONDS` also ends the turn. The marker
is authoritative; quiescence is the fallback that keeps a TUI from hanging the chat
until the timeout.

**Echo is turned off** (`stty -echo`), because in a chat the operator's own words are
already on screen in their message bubble; echoing them back into the reply would
show every command twice.

Sessions are keyed by conversation, so `cd`, `export`, an activated virtualenv and a
half-finished `sudo` prompt all persist between messages of that chat — and a second
chat to the same host is a genuinely separate shell.
"""
from __future__ import annotations

import re
import threading
import time

from . import datasources, ssh_terminal

# The prompt this module installs. Distinctive enough that ordinary output will not
# contain it, and stripped before the reply is shown.
TURN_MARK = "__SUBJECTIVE_SSH_TURN__"
READY_MARK = "__SUBJECTIVE_SSH_READY__"

# The same markers as the shell must be *told* them. A tty echoes what is typed, and
# echo is still on while these very commands are being sent — so a marker written
# whole would come back in the echo and be matched before the shell had done
# anything, leaving every later turn reading the previous turn's output. Split here,
# joined by the shell's own string concatenation, so only the real prompt and the
# real output ever contain the whole marker.
_TURN_MARK_TYPED = '"__SUBJECTIVE_SSH""_TURN__"'
_READY_MARK_TYPED = '"__SUBJECTIVE_SSH""_READY__"'


# A turn ends when output has been this quiet, for programs that never return to a
# prompt. Long enough not to cut a command that pauses mid-work, short enough that a
# TUI does not feel hung.
QUIET_SECONDS = 1.2
# How long to wait for the shell to come up and accept its configuration.
SETUP_TIMEOUT = 30.0
DEFAULT_TURN_TIMEOUT = 900.0
# A chat bubble is not a terminal; beyond this the reply is unreadable anyway.
MAX_REPLY_CHARS = 200_000
# Wide, because output wraps at the terminal width and a chat has no horizontal scroll.
COLS = 200
ROWS = 50
IDLE_TIMEOUT_SECONDS = 3600

_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_OTHER = re.compile(r"\x1b[()#][0-9A-Za-z]|\x1b[=>]|\x1b[PX^_][^\x1b]*\x1b\\")


def clean_terminal_text(text: str) -> str:
    """Turn raw terminal output into something a chat bubble can show.

    Escape sequences are removed rather than interpreted: reproducing cursor motion
    would mean implementing a terminal emulator, and for the command output that
    makes up almost every turn, stripping is both correct and enough. A full-screen
    program's redraws survive as repeated text — ugly, but honest, and the pty
    console at /ssh renders those properly.
    """
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_CSI.sub("", text)
    text = _ANSI_OTHER.sub("", text)
    text = text.replace("\x1b", "").replace("\x07", "")
    # A tty ends lines with CRLF, and \r alone is a program redrawing the line it is
    # already on — keeping the last state of that line is what a terminal would show.
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[^\n]*\r(?!\n)", "", text)
    return text


class _Session:
    """One pty session plus the turn-taking built on top of it."""

    def __init__(self, target: str, connection: dict):
        self.target = target
        self.connection = dict(connection)
        self.lock = threading.Lock()
        self.pty = None
        self.buffer: list[str] = []
        self.buffer_lock = threading.Lock()
        self.last_data = 0.0
        self.last_active = time.time()
        self.closed_code: int | None = None
        self.opened_at = 0.0

    # -- plumbing ----------------------------------------------------------------

    def _on_event(self, event: dict) -> None:
        if event.get("type") == "data":
            with self.buffer_lock:
                self.buffer.append(event["data"])
                self.last_data = time.time()
        elif event.get("type") == "closed":
            self.closed_code = event.get("code")
            with self.buffer_lock:
                self.last_data = time.time()

    def _take(self) -> str:
        with self.buffer_lock:
            text, self.buffer = "".join(self.buffer), []
            return text

    def _peek_quiet_for(self) -> float:
        with self.buffer_lock:
            return time.time() - self.last_data if self.last_data else 0.0

    @property
    def alive(self) -> bool:
        return bool(self.pty and self.pty.alive)

    # -- lifecycle ---------------------------------------------------------------

    def open(self) -> None:
        """Start the shell and make its prompt into a turn marker."""
        _instance, session = ssh_terminal.open_session(
            self.connection, pty=True, cols=COLS, rows=ROWS
        )
        session.attach(self._on_event)
        started = session.start()
        if started.get("type") != "ready":
            session.close()
            raise RuntimeError(started.get("error") or "Could not open the SSH session.")
        self.pty = session
        self.opened_at = time.time()

        # `ready` means ssh started, not that the remote shell is reading yet. Anything
        # written before it is reading goes to the login process instead, so sync on a
        # round trip first.
        self.pty.write(f"echo {_READY_MARK_TYPED}\r")
        if not self._await(READY_MARK, SETUP_TIMEOUT):
            self.close()
            raise RuntimeError("The remote shell did not respond.")
        self._take()

        # PS1 is what makes a turn boundary exact instead of guessed. The rest keeps
        # the transcript readable: no echo of what the operator already typed, no
        # bracketed-paste escapes, no PROMPT_COMMAND writing a title on every turn.
        self.pty.write(
            "stty -echo 2>/dev/null; "
            "unset PROMPT_COMMAND 2>/dev/null; "
            "bind 'set enable-bracketed-paste off' 2>/dev/null; "
            f"PS1={_TURN_MARK_TYPED}; PS2=''\r"
        )
        if not self._await(TURN_MARK, SETUP_TIMEOUT):
            self.close()
            raise RuntimeError("The remote shell did not accept its configuration.")
        self._settle()

    def close(self) -> None:
        session, self.pty = self.pty, None
        if session is not None:
            session.close()

    def _settle(self, quiet: float = 0.15, limit: float = 3.0) -> None:
        """Discard whatever the terminal is still saying, then wait for it to stop.

        Draining once is not enough. A prompt can land in the gap between the drain
        and the next write, and the next turn then sees a marker it did not cause and
        ends immediately with no output — which is exactly how this first shipped:
        every reply was empty and the send looked successful.
        """
        deadline = time.time() + limit
        while time.time() < deadline:
            self._take()
            if self._peek_quiet_for() >= quiet:
                break
            time.sleep(0.05)
        self._take()

    def _await(self, marker: str, timeout: float) -> bool:
        """Wait for a marker during setup, leaving the buffer intact."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.buffer_lock:
                if marker in "".join(self.buffer):
                    return True
            if self.closed_code is not None:
                return False
            time.sleep(0.05)
        return False

    # -- turns -------------------------------------------------------------------

    def turn(self, text: str, timeout: float) -> dict:
        """Send one message and collect the reply."""
        self.last_active = time.time()
        # Anything from before this turn — the previous turn's prompt above all — is
        # not part of this reply, and a leftover marker would end the turn at once.
        self._settle()
        if not self.pty.write(text + "\r"):
            return {"success": False, "status": "closed", "error": "The remote shell has ended."}

        collected: list[str] = []
        deadline = time.time() + timeout
        ended = "timeout"
        while time.time() < deadline:
            chunk = self._take()
            if chunk:
                collected.append(chunk)
                if TURN_MARK in "".join(collected):
                    ended = "prompt"
                    break
            if self.closed_code is not None:
                ended = "closed"
                break
            # The fallback boundary, for a program that never returns to a prompt.
            if collected and self._peek_quiet_for() >= QUIET_SECONDS:
                ended = "quiet"
                break
            time.sleep(0.05)

        raw = "".join(collected)
        reply = clean_terminal_text(raw)
        # The marker is the boundary, not content. Everything after the last one is
        # the shell redrawing its prompt for the next turn.
        if TURN_MARK in reply:
            reply = reply.split(TURN_MARK)[0]
        reply = reply.strip("\n")
        truncated = len(reply) > MAX_REPLY_CHARS
        if truncated:
            reply = reply[:MAX_REPLY_CHARS] + f"\n… truncated at {MAX_REPLY_CHARS} characters."

        self.last_active = time.time()
        if ended == "closed":
            return {
                "success": False,
                "status": "closed",
                "response": reply,
                "error": f"The remote shell exited (code {self.closed_code}).",
            }
        if ended == "timeout":
            return {
                "success": False,
                "status": "timeout",
                "response": reply,
                "error": f"No prompt came back within {timeout:g}s. The command may still be running; "
                         "send an empty message to collect more output, or Stop to end it.",
            }
        return {
            "success": True,
            "status": "completed",
            "response": reply or "(no output)",
            "error": "",
            # `quiet` means a program is still on screen and has not returned to a
            # prompt — the next message will be read by *it*, not by the shell.
            "awaitingProgram": ended == "quiet",
            "truncated": truncated,
        }

    def interrupt(self) -> bool:
        return bool(self.pty and self.pty.interrupt())


class SshChatSessions:
    """The shell behind each SSH chat, one per conversation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}

    def _reap_idle(self) -> None:
        cutoff = time.time() - IDLE_TIMEOUT_SECONDS
        stale = [
            key for key, session in self._sessions.items()
            if session.last_active < cutoff or not session.alive
        ]
        for key in stale:
            session = self._sessions.pop(key, None)
            if session is not None:
                session.close()

    def send(self, target: str, connection: dict, text: str, timeout: float | None = None) -> dict:
        """Deliver one chat message to the shell for `target`, opening it if needed."""
        with self._lock:
            self._reap_idle()
            session = self._sessions.get(target)
            if session is not None and not session.alive:
                session.close()
                self._sessions.pop(target, None)
                session = None
            if session is None:
                session = _Session(target, connection)
                try:
                    session.open()
                except (RuntimeError, LookupError, ValueError) as error:
                    return {
                        "success": False,
                        "status": "connection_failed",
                        "response": "",
                        "error": str(error),
                    }
                self._sessions[target] = session

        # Outside the registry lock: a long turn in one chat must not block another.
        with session.lock:
            return session.turn(text, timeout or DEFAULT_TURN_TIMEOUT)

    def interrupt(self, target: str) -> bool:
        with self._lock:
            session = self._sessions.get(target)
        return bool(session and session.interrupt())

    def close(self, target: str) -> None:
        with self._lock:
            session = self._sessions.pop(target, None)
        if session is not None:
            session.close()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "target": key,
                    "connection": ssh_terminal.describe(session.connection),
                    "openedAt": session.opened_at,
                    "lastActive": session.last_active,
                    "alive": session.alive,
                }
                for key, session in self._sessions.items()
            ]

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


sessions = SshChatSessions()


def connection_for_conversation(conversation: dict, app_home: str = "") -> dict:
    """Where an SSH chat's target comes from: the conversation's workspace.

    A chat has one field that can name a machine, so that is where the reference
    lives. It holds either `saved:<id>` — a connection from the store, with its key,
    password and options — or a target written out longhand (`user@host:/path`, or a
    `~/.ssh/config` alias), which is what SSH chats used before the store existed and
    still works without one.
    """
    from . import ssh_connections

    reference = str(conversation.get("workspace") or "").strip()
    if not app_home:
        # No store to consult: only the longhand form can be resolved.
        settings = datasources.ssh_connection_from_workspace(reference)
        if not settings.get("host"):
            raise ValueError(
                "This SSH chat has no host. Set its workspace to a machine — "
                "`user@host`, `user@host:/path`, or a Host alias from ~/.ssh/config."
            )
        return settings
    return ssh_connections.connection_settings(app_home, reference)
