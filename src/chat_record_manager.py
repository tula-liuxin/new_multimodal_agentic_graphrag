import json, os, time
from pathlib import Path
from typing import Dict, Any, Optional

TS_FMT = "%Y%m%d-%H%M%S"

def _now_ts():
    return time.strftime(TS_FMT, time.localtime())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def start_chat_session(project_root: Path, base_dir: str = "chat_record", session_name: Optional[str] = None) -> Path:
    root = project_root / base_dir
    ensure_dir(root)
    ts = _now_ts()
    name = session_name or f"{ts}_chat"
    session_dir = root / name
    ensure_dir(session_dir)
    # 初始化文件
    (session_dir / "summary.txt").write_text("", encoding="utf-8")
    (session_dir / "transcript.jsonl").write_text("", encoding="utf-8")
    meta = {
        "session": name,
        "created_at": ts,
        "path": str(session_dir.resolve())
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_dir

def turn_dir(session_dir: Path, turn_index: int) -> Path:
    td = session_dir / f"turn_{turn_index:05d}"
    ensure_dir(td)
    return td

def append_transcript(session_dir: Path, record: Dict[str, Any]):
    line = json.dumps(record, ensure_ascii=False)
    with (session_dir / "transcript.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_turn_summary(td: Path, text: str):
    (td / "summary.txt").write_text(text or "", encoding="utf-8")

def write_turn_payload(td: Path, name: str, content: str):
    (td / name).write_text(content or "", encoding="utf-8")

def update_session_summary(session_dir: Path, text: str):
    (session_dir / "summary.txt").write_text(text or "", encoding="utf-8")
