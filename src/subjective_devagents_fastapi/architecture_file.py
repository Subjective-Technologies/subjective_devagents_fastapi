"""The company's UML model on this server: /<company>/<company>/architecture/<company>.xmi.

DevAgents runs on the company's own server, so the file its business analyst and
architect agents write (workflow WORKFLOW_V2_PATHS: vm_root = /${company_name}/${company_name},
xmi_path = ${vm_root}/architecture/${company_name}.xmi) is a local file here. The
company is this server's company (auth.expected_company_name), never a caller value,
so the route cannot be pointed at another file.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from . import auth

MAX_BYTES = 16 * 1024 * 1024
_UNSAFE = re.compile(r"[^a-z0-9-]+")


def company_slug(name: str) -> str:
    """"CardNexia" -> "cardnexia". The slug the company's clone paths use."""
    slug = _UNSAFE.sub("-", (name or "").strip().lower()).strip("-")
    return "" if not slug or ".." in slug else slug[:80]


def xmi_path(company_name: str) -> Path | None:
    slug = company_slug(company_name)
    if not slug:
        return None
    # Tests point the root elsewhere; production reads from the filesystem root.
    root = Path(os.environ.get("SUBJECTIVE_DEVAGENTS_ARCHITECTURE_ROOT") or "/")
    return root / slug / slug / "architecture" / f"{slug}.xmi"


def read_company_xmi() -> tuple[int, dict]:
    """(status, body). A read that ran is 200: found false when there is no file yet."""
    company = auth.expected_company_name()
    path = xmi_path(company)
    if path is None:
        return 503, {"ok": False, "error": "company_not_configured"}
    shown = str(path)
    try:
        info = path.stat()
    except FileNotFoundError:
        return 200, {"ok": True, "found": False, "path": shown}
    except PermissionError:
        return 200, {"ok": False, "error": "permission_denied", "path": shown}
    if not path.is_file():
        return 200, {"ok": True, "found": False, "path": shown}
    if info.st_size > MAX_BYTES:
        return 200, {"ok": False, "error": "file_too_large", "path": shown, "size": info.st_size}
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
    except PermissionError:
        return 200, {"ok": False, "error": "permission_denied", "path": shown}
    except OSError:
        return 200, {"ok": False, "error": "read_failed", "path": shown}
    if len(raw) > MAX_BYTES:
        return 200, {"ok": False, "error": "file_too_large", "path": shown, "size": len(raw)}
    return 200, {
        "ok": True,
        "found": True,
        "company": company,
        "path": shown,
        "size": len(raw),
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(info.st_mtime))),
        "xmi_text": raw.decode("utf-8", errors="replace"),
    }
