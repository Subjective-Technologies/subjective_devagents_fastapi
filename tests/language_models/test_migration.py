import pytest

from subjective_devagents_fastapi.language_models import ModelError
from subjective_devagents_fastapi.language_models.migration import import_payload


def test_legacy_export_transfers_text_but_not_native_identity():
    source = {"conversation": {"sessionId": "old-native", "provider": "claude", "osUser": "old-user",
                               "messages": [{"role": "user", "text": "question"},
                                            {"role": "assistant", "text": "answer"}]}}
    assert import_payload(source, "codex-new") == {
        "identity_id": "codex-new",
        "history": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    }
    assert source["conversation"]["sessionId"] == "old-native"


@pytest.mark.parametrize("source", [
    {}, [], {"messages": [{"role": "user", "text": "unfinished"}]},
    {"messages": [], "messagesTruncated": True},
    {"messages": [{"role": "user", "text": "image", "attachments": ["image"]},
                  {"role": "assistant", "text": "answer"}]},
    {"messages": [{"role": "system", "text": "notice"}, {"role": "assistant", "text": "answer"}]},
    {"messages": [{"role": "user", "text": "failed", "sendFailed": True},
                  {"role": "assistant", "text": "unrelated answer"}]},
])
def test_unsupported_migration_never_silently_drops_content(source):
    with pytest.raises(ModelError):
        import_payload(source, "new-account")
