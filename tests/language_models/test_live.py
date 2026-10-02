"""Explicit deployment gate. Uses real configured accounts and subscription allowance.

No URL means skip. Configured accounts must succeed; failures are never skipped.
"""
import os
import time
import uuid

import httpx
import pytest

URL = os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_LIVE_URL", "").rstrip("/")
IDENTITIES = [value.strip() for value in os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_LIVE_IDENTITIES", "").split(",") if value.strip()]
pytestmark = pytest.mark.skipif(not URL, reason="Set SUBJECTIVE_LANGUAGE_MODELS_LIVE_URL for authenticated deployment testing")


def test_real_subscription_resume_and_identity_handoff():
    assert IDENTITIES, "Set SUBJECTIVE_LANGUAGE_MODELS_LIVE_IDENTITIES to required configured account ids"
    credential = os.environ.get("SUBJECTIVE_LANGUAGE_MODELS_LIVE_TOKEN", "")
    headers = {"Authorization": f"Bearer {credential}"} if credential else {}
    with httpx.Client(base_url=URL + "/api/language-model", headers=headers, timeout=20, trust_env=False) as client:
        response = client.post("/conversations", json={"identity_id": IDENTITIES[0]})
        assert response.status_code == 201, f"Create failed with HTTP {response.status_code}"
        conversation = response.json()["id"]
        marker = "LM_" + uuid.uuid4().hex[:12]

        def turn(prompt):
            response = client.post(f"/conversations/{conversation}/turns",
                                   json={"prompt": prompt, "request_id": str(uuid.uuid4()), "timeout": 120})
            assert response.status_code == 202, f"Submit failed with HTTP {response.status_code}"
            identifier = response.json()["id"]
            deadline = time.monotonic() + 135
            while time.monotonic() < deadline:
                response = client.get(f"/turns/{identifier}")
                assert response.status_code == 200
                result = response.json()
                if result["status"] not in {"queued", "running"}:
                    assert result["status"] == "completed", (identifier, result["status"], result["error_code"])
                    assert marker in result["text"], f"Missing remembered marker in turn {identifier}"
                    return result
                time.sleep(0.2)
            client.post(f"/turns/{identifier}/cancel")
            pytest.fail(f"Turn {identifier} timed out; inspect persisted result")

        first = turn(f"Remember this marker and reply with exactly {marker}. Do not use tools or edit files.")
        second = turn("Reply with exactly the marker from my previous message. Do not use tools or edit files.")
        assert first["session_id"] == second["session_id"]
        for identity in IDENTITIES[1:]:
            response = client.post(f"/conversations/{conversation}/handoff", json={"identity_id": identity})
            assert response.status_code == 200
            assert response.json()["session_id"] is None
            turn("Reply with exactly the marker from the earlier conversation. Do not use tools or edit files.")
