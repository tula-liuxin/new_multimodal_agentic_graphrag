import os, re, json, datetime
from typing import List, Dict, Any, Iterable

SAFE_PREFIXES = ("\\\\?\\",)

def now_ts_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_display_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    for pref in SAFE_PREFIXES:
        if p.startswith(pref):
            p = p[len(pref):]
    return p.replace("/", "\\")

def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tokenize(s: str):
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", " ", s)
    return [t for t in s.strip().split() if t]

def score_keywords(text: str, keywords: List[str]) -> float:
    if not text or not keywords:
        return 0.0
    text_low = text.lower()
    hits = 0
    for k in set([k.lower() for k in keywords if k.strip()]):
        if k in text_low:
            hits += 1
    return hits / max(1, len(set(keywords)))

def merge_short_lines(lines, min_chars: int):
    buf, acc = [], ""
    for ln in lines:
        if len(acc) + len(ln) < min_chars:
            acc += (ln.strip() + " ")
        else:
            if acc.strip():
                buf.append(acc.strip())
                acc = ""
            if ln.strip():
                buf.append(ln.strip())
    if acc.strip():
        buf.append(acc.strip())
    return buf

def first_n_sentences(text: str, n: int = 3) -> str:
    import re
    parts = re.split(r"(?:\。|。|！|!|？|\?|\.|\n)", text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
        if len(out) >= n:
            break
    return "。".join(out) if out else text.strip()

def uniq_preserve(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def list_turn_dirs(session_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(session_dir) if d.startswith("turn-")])

def collect_turn_files_paths(session_dir: str) -> List[str]:
    out = []
    for tdir in list_turn_dirs(session_dir):
        p = os.path.join(session_dir, tdir, "turn_files", "rag_clean.jsonl")
        if os.path.exists(p):
            out.append(p)
    return out
