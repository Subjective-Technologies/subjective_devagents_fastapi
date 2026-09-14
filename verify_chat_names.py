"""Read-only check: does every assigned chat name actually reach the listing?

Run as the account that owns the production state:
    sudo -u subjective /subjective/services/subjective_devagents_fastapi/.venv/bin/python \
        /subjective/services/subjective_devagents_fastapi/verify_chat_names.py

Never writes: _save_state is stubbed out so a verification run cannot touch live state.
"""
import sys

sys.path.insert(0, "/subjective/services/subjective_devagents_fastapi/src")
from subjective_devagents_fastapi.llm_service import LlmService  # noqa: E402


class ReadOnlyLlmService(LlmService):
    def _save_state(self) -> None:
        pass


service = ReadOnlyLlmService(
    root="/subjective/services/subjective_devagents_fastapi",
    app_home="/home/subjective/.subjective_chat",
)
listed = service.list_conversations()["conversations"]
shown_by_target = {item["target"]: item["label"] for item in listed}
print(f"conversations listed:      {len(listed)}")
print(f"assigned names recorded:   {len(service.label_overrides)}\n")

correct = wrong = 0
for target, assigned in service.label_overrides.items():
    shown = shown_by_target.get(target)
    if shown is None:
        continue  # an alias that deduplicated into another row; not listed on its own
    if shown == assigned:
        correct += 1
        print(f"  [ OK ] {assigned!r}")
    else:
        wrong += 1
        print(f"  [FAIL] assigned {assigned!r} but listing shows {shown!r}")

print(f"\n{correct} showing their assigned name, {wrong} still showing something else")
sys.exit(1 if wrong else 0)
