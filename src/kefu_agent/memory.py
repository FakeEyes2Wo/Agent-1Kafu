import json
import re
from pathlib import Path

from .config import get_settings


def _session_path(session_id: str) -> Path:
    settings = get_settings()
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", session_id)[:120]
    return settings.session_store_dir / f"{safe_id}.json"


def load_history(session_id: str) -> list[dict[str, str]]:
    path = _session_path(session_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_turn(session_id: str, question: str, answer: str) -> None:
    settings = get_settings()
    path = _session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(session_id)
    history.append({"question": question, "answer": answer})
    history = history[-settings.max_history_turns :]
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
