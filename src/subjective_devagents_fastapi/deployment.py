"""Content identity supplements Git HEAD when a deployment contains local edits."""
import hashlib
from pathlib import Path


def source_fingerprint(root: Path, directories: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for directory in directories:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".js", ".css", ".html", ".json"}:
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
