"""The company's UML model on this server: /<company>/<company>/architecture/<company>.xmi,
or, when the folder holds dated versions, the latest
/<company>/<company>/architecture/YYYY_MM_DD_HH_MM_SS-<company>.xmi.

DevAgents runs on the company's own server, so the file its business analyst and
architect agents write (workflow WORKFLOW_V2_PATHS: vm_root = /${company_name}/${company_name},
xmi_path = ${vm_root}/architecture/${company_name}.xmi) is a local file here. The
company is this server's company (auth.expected_company_name). A caller may name
the company it means, but only to have the call refused when that is another
company: the file is always this server's, so the route cannot be pointed at
another file.

Subjective UML reads the model with ``read_company_xmi`` and saves edits with
``write_company_xmi``. A save writes a new dated version and names the version it
edited (``base_sha256``), so an edit made on a stale copy is refused rather than
hiding what the agents wrote since.
"""
from __future__ import annotations

import calendar
import hashlib
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from xml.parsers import expat

from . import auth

MAX_BYTES = 16 * 1024 * 1024
_UNSAFE = re.compile(r"[^a-z0-9-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# One save at a time in this process, so check-then-replace is not interleaved.
_WRITE_LOCK = threading.Lock()


def company_slug(name: str) -> str:
    """"CardNexia" -> "cardnexia". The slug the company's clone paths use."""
    slug = _UNSAFE.sub("-", (name or "").strip().lower()).strip("-")
    return "" if not slug or ".." in slug else slug[:80]


def company_root(company_name: str) -> Path | None:
    """/<company>/<company>: the company's clone on its own server (workflow vm_root)."""
    slug = company_slug(company_name)
    if not slug:
        return None
    # Tests point the root elsewhere; production reads from the filesystem root.
    root = Path(os.environ.get("SUBJECTIVE_DEVAGENTS_ARCHITECTURE_ROOT") or "/")
    return root / slug / slug


def xmi_path(company_name: str) -> Path | None:
    """The model's plain name: /<company>/<company>/architecture/<company>.xmi."""
    base = company_root(company_name)
    if base is None:
        return None
    return base / "architecture" / f"{base.name}.xmi"


def version_name_pattern(slug: str) -> re.Pattern[str]:
    """YYYY_MM_DD_HH_MM_SS-<company>.xmi: one saved version of the model."""
    return re.compile(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})-" + re.escape(slug) + r"\.xmi")


def latest_version_name(names, slug: str) -> str:
    """The newest YYYY_MM_DD_HH_MM_SS-<slug>.xmi among ``names``, or "".

    The timestamp is fixed-width, so the newest version sorts last.
    """
    pattern = version_name_pattern(slug)
    versions = sorted(name for name in names if pattern.fullmatch(name))
    return versions[-1] if versions else ""


def current_xmi_path(company_name: str) -> Path | None:
    """The file the editor opens: the latest YYYY_MM_DD_HH_MM_SS-<company>.xmi in the
    company's architecture folder, else <company>.xmi. Only names this module builds
    are considered, so a caller still cannot point the route at another file."""
    path = xmi_path(company_name)
    if path is None:
        return None
    try:
        names = [entry.name for entry in os.scandir(path.parent) if entry.is_file()]
    except OSError:
        return path
    latest = latest_version_name(names, path.parent.parent.name)
    return path.parent / latest if latest else path


_STAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"


def next_version_name(names, slug: str, now: float | None = None) -> str:
    """A new YYYY_MM_DD_HH_MM_SS-<slug>.xmi (UTC) that sorts after every version in
    ``names``, so the save is what the editor opens next even when the clock is
    behind a version already in the folder."""
    seconds = int(time.time() if now is None else now)
    latest = latest_version_name(names, slug)
    if latest:
        try:
            latest_seconds = int(calendar.timegm(time.strptime(latest[:19], _STAMP_FORMAT)))
        except ValueError:
            latest_seconds = seconds - 1
        seconds = max(seconds, latest_seconds + 1)
    return f"{time.strftime(_STAMP_FORMAT, time.gmtime(seconds))}-{slug}.xmi"


def _iso(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(seconds)))


def _target(requested_company: str) -> tuple[int, dict] | tuple[str, Path]:
    """(company, path) for this server, or (status, body) refusing the call."""
    company = auth.expected_company_name()
    path = current_xmi_path(company)
    if path is None:
        return 503, {"ok": False, "error": "company_not_configured"}
    requested = (requested_company or "").strip()
    if requested and company_slug(requested) != company_slug(company):
        # This server holds one company's files. Another company's model lives on its server.
        return 404, {"ok": False, "error": "company_mismatch", "company": company}
    return company, path


def read_company_xmi(requested_company: str = "") -> tuple[int, dict]:
    """(status, body). A read that ran is 200: found false when there is no file yet."""
    target = _target(requested_company)
    if isinstance(target[0], int):
        return target  # type: ignore[return-value]
    company, path = target
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
        "modified_at": _iso(info.st_mtime),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "xmi_text": raw.decode("utf-8", errors="replace"),
    }


def xml_problem(raw: bytes) -> str:
    """Empty when ``raw`` is one well-formed XML document; otherwise why not.

    Entity declarations are refused: a model never needs them, and they are how
    an XML document expands to gigabytes or reads other files.
    """
    parser = expat.ParserCreate()

    def refuse_entity(*_args):
        raise ValueError("entity declarations are not allowed")

    parser.EntityDeclHandler = refuse_entity
    parser.UnparsedEntityDeclHandler = refuse_entity
    parser.ExternalEntityRefHandler = lambda *_args: 0
    try:
        parser.Parse(raw, True)
    except ValueError as error:
        return str(error)
    except expat.ExpatError as error:
        return f"not well-formed XML: {expat.ErrorString(error.code)} at line {error.lineno}"
    return ""


def _current_sha(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            return hashlib.sha256(handle.read(MAX_BYTES + 1)).hexdigest()
    except FileNotFoundError:
        return None


def write_company_xmi(xmi_text, base_sha256=None, requested_company: str = "") -> tuple[int, dict]:
    """Save the model as a new version. (status, body).

    Each save writes a new /<company>/<company>/architecture/YYYY_MM_DD_HH_MM_SS-<company>.xmi
    (UTC, later than every version already there), which becomes the file the
    editor opens. Existing versions and <company>.xmi are never changed.

    ``base_sha256`` is the ``sha256`` the editor loaded: the save is refused with 409
    when the latest version has changed since. ``""`` means "I loaded no file":
    refused when one exists. ``None`` saves regardless.

    The new file appears complete (a temporary file in the same folder, then a
    link), so a reader never sees half a model. It takes the loaded file's mode.
    The change is not committed; the company's agents version the clone.
    """
    target = _target(requested_company)
    if isinstance(target[0], int):
        return target  # type: ignore[return-value]
    company, path = target
    shown = str(path)
    if not isinstance(xmi_text, str) or not xmi_text.strip():
        return 400, {"ok": False, "error": "xmi_text_required", "path": shown}
    if base_sha256 is not None and (not isinstance(base_sha256, str)
                                    or (base_sha256 and not _SHA256.fullmatch(base_sha256))):
        return 400, {"ok": False, "error": "invalid_base_sha256", "path": shown}
    raw = xmi_text.encode("utf-8")
    if len(raw) > MAX_BYTES:
        return 413, {"ok": False, "error": "file_too_large", "path": shown, "size": len(raw)}
    problem = xml_problem(raw)
    if problem:
        return 400, {"ok": False, "error": "invalid_xmi", "detail": problem, "path": shown}
    clone = path.parent.parent
    if not clone.is_dir():
        # The company repository is cloned by the dashboard; this route does not create it.
        return 404, {"ok": False, "error": "company_folder_missing", "path": shown}
    with _WRITE_LOCK:
        try:
            current = _current_sha(path)
        except PermissionError:
            return 403, {"ok": False, "error": "permission_denied", "path": shown}
        except OSError:
            return 500, {"ok": False, "error": "read_failed", "path": shown}
        if base_sha256 is not None and (current or "") != base_sha256:
            return 409, {"ok": False, "error": "conflict", "path": shown, "sha256": current or ""}
        folder = path.parent
        try:
            mode = path.stat().st_mode & 0o7777 if current is not None else 0o664
            folder.mkdir(exist_ok=True)
            names = [entry.name for entry in os.scandir(folder)]
            saved_path = folder / next_version_name(names, clone.name)
            handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{saved_path.name}.", suffix=".tmp")
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, mode)
                # A hard link never replaces a file: an existing version is left untouched.
                os.link(temporary, saved_path)
            finally:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        except FileExistsError:
            return 409, {"ok": False, "error": "conflict", "path": shown, "sha256": current or ""}
        except PermissionError:
            return 403, {"ok": False, "error": "permission_denied", "path": shown}
        except OSError:
            return 500, {"ok": False, "error": "write_failed", "path": shown}
        info = saved_path.stat()
    return 200, {
        "ok": True,
        "saved": True,
        "company": company,
        "path": str(saved_path),
        "previous_path": shown if current is not None else "",
        "size": len(raw),
        "modified_at": _iso(info.st_mtime),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "previous_sha256": current or "",
    }
