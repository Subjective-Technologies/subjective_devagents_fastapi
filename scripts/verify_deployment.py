"""Fail unless public deployment serves these exact frontend and backend sources.

Run after deployment, before browser tests. No credentials or chat writes required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))
from subjective_devagents_fastapi.deployment import source_fingerprint


def verify(base_url: str, client_root: Path) -> dict:
    headers = {"Cache-Control": "no-cache", "User-Agent": "Subjective-Deployment-Verification/1.0"}
    request = Request(base_url.rstrip("/") + f"/healthz?verify={time.time_ns()}", headers=headers)
    with urlopen(request, timeout=20) as response:
        headers = response.headers
        if not json.load(response).get("ok"):
            raise RuntimeError("Deployment health check failed")
    result = {}
    for name, root, directories, version_header in (
        ("frontend", client_root, ("src", "web"), "X-DevAgents-Version"),
        ("backend", SERVICE_ROOT, ("src",), "X-DevAgents-Backend-Version"),
    ):
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        version = headers.get(version_header, "")
        expected = source_fingerprint(root, directories)
        actual = headers.get(f"X-DevAgents-{name}-Source", "")
        if version.split("+")[-1].removesuffix("-dirty") != commit or actual != expected:
            raise RuntimeError(f"Stale {name}: served={version} source={actual or 'missing'}; expected commit={commit} source={expected}")
        result[name] = {"version": version, "source": actual}
    # Files are read at request time by the server. Check the actual shipped script
    # as well as startup metadata, catching edits made without a backend restart.
    request = Request(base_url.rstrip("/") + f"/renderer.js?verify={time.time_ns()}",
                      headers={"Cache-Control": "no-cache", "User-Agent": "Subjective-Deployment-Verification/1.0"})
    with urlopen(request, timeout=20) as response:
        actual_renderer = response.read()
    expected_renderer = (client_root / "src" / "renderer.js").read_text().replace(
        "DEVAGENTS_DEPLOY_VERSION", result["frontend"]["version"]
    ).encode()
    if hashlib.sha256(actual_renderer).digest() != hashlib.sha256(expected_renderer).digest():
        raise RuntimeError("Deployed renderer differs from the tested source")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--client-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.url, args.client_root), indent=2))
    except Exception as error:
        sys.exit(str(error))
