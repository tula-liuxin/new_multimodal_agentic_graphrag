import os, json, datetime
from typing import Dict, Any, Optional, List
from .utils import ensure_dir, now_ts_str, append_jsonl, list_turn_dirs

class ChatSession:
    def __init__(self, session_root: str, session_name: str | None = None, naming: str = "timestamp"):
        ensure_dir(session_root)
        ts = now_ts_str()
        if naming not in {"timestamp", "name_and_timestamp", "name_only"}:
            naming = "timestamp"
        if session_name:
            if naming == "name_only":
                folder = session_name
            elif naming == "name_and_timestamp":
                folder = f"{session_name}-{ts}"
            else:
                folder = f"session-{ts}"
        else:
            folder = f"session-{ts}"
        self.session_dir = os.path.join(session_root, folder)
        ensure_dir(self.session_dir)
        self.turn_idx = 1
        self.summary_path = os.path.join(self.session_dir, "summary.txt")
        self.transcript_path = os.path.join(self.session_dir, "transcript.jsonl")
        self.focus_path = os.path.join(self.session_dir, "focus.json")
        self.entities_path = os.path.join(self.session_dir, "entities.json")

    def new_turn_dir(self) -> str:
        d = os.path.join(self.session_dir, f"turn-{self.turn_idx:04d}")
        ensure_dir(d)
        self.turn_idx += 1
        return d

    def list_turn_dirs(self) -> List[str]:
        return list_turn_dirs(self.session_dir)

    def read_last_turn_summary(self) -> str:
        turns = self.list_turn_dirs()
        if not turns:
            return ""
        last = turns[-1]
        p = os.path.join(self.session_dir, last, "turn_summary.txt")
        if os.path.exists(p):
            try:
                return open(p, "r", encoding="utf-8").read()
            except Exception:
                return ""
        return ""

    def append_transcript(self, record: Dict[str, Any]) -> None:
        record["ts"] = datetime.datetime.now().isoformat()
        append_jsonl(self.transcript_path, [record])

    def read_chat_summary(self) -> str:
        if os.path.exists(self.summary_path):
            return open(self.summary_path, "r", encoding="utf-8").read()
        return ""

    def write_chat_summary(self, text: str) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(text or "")

    def get_focus(self) -> str:
        if os.path.exists(self.focus_path):
            try:
                return json.load(open(self.focus_path, "r", encoding="utf-8")).get("last_focus", "")
            except Exception:
                return ""
        return ""

    def set_focus(self, name: str) -> None:
        if not name:
            return
        with open(self.focus_path, "w", encoding="utf-8") as f:
            json.dump({"last_focus": name, "updated_at": datetime.datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    def get_entities(self) -> List[str]:
        if os.path.exists(self.entities_path):
            try:
                obj = json.load(open(self.entities_path, "r", encoding="utf-8"))
                return obj.get("stack", [])
            except Exception:
                return []
        return []

    def push_entities(self, new_entities: List[str], max_keep: int = 8) -> None:
        stack = self.get_entities()
        for e in new_entities:
            if e and e not in stack:
                stack.append(e)
        stack = stack[-max_keep:]
        with open(self.entities_path, "w", encoding="utf-8") as f:
            json.dump({"stack": stack, "updated_at": datetime.datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
