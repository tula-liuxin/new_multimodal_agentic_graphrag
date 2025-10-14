
import os, sys, json, hashlib, logging, time
from typing import Iterable, Dict, Any, List, Optional

try:
    import yaml
except Exception:
    yaml = None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def norm_win_abs(p: str) -> str:
    p = os.path.abspath(p)
    if p.startswith("\\\\?\\"):
        p = p[4:]
    return p

def ensure_dir(d: str) -> str:
    """mkdir -p，返回绝对 Windows 路径（去掉 \\?\ 前缀）。"""
    os.makedirs(d, exist_ok=True)
    return norm_win_abs(d)

def get_logger(name: str = "") -> logging.Logger:
    log = logging.getLogger(name or __name__)
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        h.setFormatter(fmt)
        log.addHandler(h)
    return log

def load_cfg(path: str = "config.yaml") -> Dict[str, Any]:
    cfg = {
        "root_dir": ".",
        "data_dir": "data",
        "enable_ocr": False,
        "ocr_backend": "rapidocr",
        "ollama_base_url": "http://127.0.0.1:11434",
        "embed_model": "bge-m3:latest",
        "clip_local": "",
        "image_batch": 32,
        "text_max_chars": 1200,
        "text_overlap": 100,
    }
    if os.path.exists(path) and yaml:
        with open(path, "r", encoding="utf-8") as f:
            u = yaml.safe_load(f) or {}
        if isinstance(u, dict):
            cfg.update(u)
    cfg["root_dir"] = norm_win_abs(cfg["root_dir"])
    cfg["data_dir"] = norm_win_abs(cfg["data_dir"])
    ensure_dir(cfg["data_dir"])
    return cfg

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path): return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path

def chunk_path(index_dir: str) -> str:
    return os.path.join(index_dir, "chunks.jsonl")

def chunk_emb_path(index_dir: str) -> str:
    return os.path.join(index_dir, "chunk_embeddings.f32")

def chunk_ids_path(index_dir: str) -> str:
    return os.path.join(index_dir, "chunk_ids.jsonl")

def image_list_path(index_dir: str) -> str:
    return os.path.join(index_dir, "images.jsonl")

def image_emb_path(index_dir: str) -> str:
    return os.path.join(index_dir, "image_embeddings.f32")

def links_path(index_dir: str) -> str:
    return os.path.join(index_dir, "links.jsonl")

def timeit(fn):
    def w(*a, **k):
        t0 = time.time()
        r = fn(*a, **k)
        dt = time.time() - t0
        return r, dt
    return w
