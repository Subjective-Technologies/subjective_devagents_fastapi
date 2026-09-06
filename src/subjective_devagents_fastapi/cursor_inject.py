"""Types a message into an open Cursor IDE tab over the Chrome DevTools Protocol.

Python port of clients/desktop/subjective_devagents/src/cursorDomInject.js. Requires
Cursor started with --remote-debugging-port. Synchronous; call from a worker thread.
"""
from __future__ import annotations

import json
import os
import re
import socket
import urllib.request

DEFAULT_CDP = int(os.environ.get("CURSOR_CDP_PORT") or 9222)


def cdp_port_open(port: int = DEFAULT_CDP, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def build_inject_expression(text: str) -> str:
    payload = json.dumps(str(text or ""))
    return (
        """(() => {
    const text = %s;
    const selectors = [
      '[data-lexical-editor="true"]',
      '.aislash-editor-input [contenteditable="true"]',
      '.composer-bar [contenteditable="true"]',
      '[class*="composer"] [contenteditable="true"]',
      '.ProseMirror',
      'textarea[placeholder*="Message"]',
      'textarea[placeholder*="Plan"]',
      'textarea'
    ];
    const nodes = [];
    for (const selector of selectors) {
      document.querySelectorAll(selector).forEach((node) => {
        if (!nodes.includes(node)) nodes.push(node);
      });
    }
    const visible = nodes.filter((node) => {
      const box = node.getBoundingClientRect();
      return box.width > 40 && box.height > 16;
    });
    const el = visible.at(-1);
    if (!el) return { ok: false, error: 'no composer input in DOM' };
    el.focus();
    if (el.isContentEditable) {
      document.execCommand('selectAll', false, null);
      const inserted = document.execCommand('insertText', false, text);
      if (!inserted) {
        el.textContent = text;
        el.dispatchEvent(new InputEvent('input', { bubbles: true, data: text, inputType: 'insertText' }));
      }
    } else {
      el.value = text;
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
    const enter = (type) => el.dispatchEvent(new KeyboardEvent(type, {
      key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true
    }));
    enter('keydown');
    enter('keyup');
    return { ok: true, tag: el.tagName, editable: el.isContentEditable };
  })()"""
        % payload
    )


def _get_json(url: str, timeout: float = 1.5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_cdp_targets(port: int = DEFAULT_CDP) -> list:
    targets = _get_json(f"http://127.0.0.1:{port}/json")
    return targets if isinstance(targets, list) else []


def pick_workbench_target(targets: list) -> dict | None:
    pages = [item for item in targets if item and item.get("type") == "page" and item.get("webSocketDebuggerUrl")]
    for item in pages:
        if re.search(r"workbench|cursor|vscode", f"{item.get('title', '')} {item.get('url', '')}", re.IGNORECASE):
            return item
    return pages[0] if pages else None


def evaluate_on_target(ws_url: str, expression: str, timeout: float = 8.0):
    from websockets.sync.client import connect

    with connect(ws_url, open_timeout=timeout, close_timeout=2) as ws:
        ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True, "awaitPromise": True},
        }))
        deadline = timeout
        while True:
            raw = ws.recv(timeout=deadline)
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if message.get("id") != 1:
                continue
            if message.get("error"):
                raise RuntimeError(message["error"].get("message") or "cdp error")
            result = message.get("result") or {}
            inner = result.get("result") or {}
            return inner.get("value")


def inject_into_cursor_dom(text: str, port: int | None = None) -> dict:
    cdp_port = int(port or DEFAULT_CDP)
    if not cdp_port_open(cdp_port):
        return {"ok": False, "error": f"Cursor CDP :{cdp_port} is off. Start Cursor with --remote-debugging-port={cdp_port}"}
    try:
        targets = list_cdp_targets(cdp_port)
        target = pick_workbench_target(targets)
        if not target:
            return {"ok": False, "error": f"Cursor CDP :{cdp_port} has no workbench page"}
        result = evaluate_on_target(target["webSocketDebuggerUrl"], build_inject_expression(text))
        if not result or result.get("ok") is not True:
            return {"ok": False, "error": (result or {}).get("error") or "DOM inject failed", "target": target.get("title")}
        return {"ok": True, "via": "cdp-dom", "target": target.get("title"), "detail": result}
    except Exception as error:  # noqa: BLE001
        message = str(error)
        if re.search(r"refused|timed? ?out|connect", message, re.IGNORECASE):
            return {"ok": False, "error": f"Cursor CDP :{cdp_port} is off. Start Cursor with --remote-debugging-port={cdp_port}"}
        return {"ok": False, "error": message}
