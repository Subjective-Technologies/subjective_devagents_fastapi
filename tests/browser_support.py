"""Where the browser tests find the client bundle they drive.

This lives in one place because it was duplicated, and the duplicate went stale:
the client moved from `clients/desktop/subjective_chat` to
`clients/desktop/subjective_devagents`, and both copies kept naming the old path.
Every browser test then skipped itself with "is the checkout complete?" on a
complete checkout, and the runs stayed green while covering nothing.

So the search is a list rather than a single guess, and a rename shows up as one
edit here instead of as silence.
"""
from __future__ import annotations

import os
from pathlib import Path

SERVICE_ROOT = Path(__file__).parent.parent
REPO_ROOT = SERVICE_ROOT.parent.parent

# Newest first. The last entry is the pre-2026_08_25 name, kept so an older
# checkout still runs.
CLIENT_CANDIDATES = (
    REPO_ROOT / "clients" / "desktop" / "subjective_devagents",
    REPO_ROOT / "clients" / "desktop" / "subjective_chat",
)


def is_client_bundle(root: Path) -> bool:
    return (root / "web" / "index.html").is_file()


def resolve_client_root() -> Path:
    """The client checkout to serve, or the best candidate to name in a skip.

    An explicit SUBJECTIVE_DEVAGENTS_TEST_CLIENT_ROOT always wins, even when it is
    wrong, so a deliberate override fails loudly instead of being second-guessed.
    """
    override = (os.environ.get("SUBJECTIVE_DEVAGENTS_TEST_CLIENT_ROOT") or "").strip()
    if override:
        return Path(override)
    for candidate in CLIENT_CANDIDATES:
        if is_client_bundle(candidate):
            return candidate
    return CLIENT_CANDIDATES[0]
