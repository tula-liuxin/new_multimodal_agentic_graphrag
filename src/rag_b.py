import os, json, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .ollama_client import OllamaClient

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

class RAGBIndex:
    def __init__(self, chat_root: Path, embed_model: str, ollama_host: Optional[str] = None):
        self.chat_root = chat_root
        self.data_dir = chat_root / "data"
        self.index_dir = chat_root / "index"
        self.embed_model = embed_model
        self.ollama_host = ollama_host
        self.chunks_path = self.data_dir / "chat_chunks.jsonl"
        self.vecs_path = self.index_dir / "text_vecs.npy"
        self.ids_path = self.index_dir / "text_ids.npy"
        self.map_path = self.index_dir / "id2row.json"
        self._ensure()

    def _ensure(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not self.chunks_path.exists():
            self.chunks_path.write_text("", encoding="utf-8")
        if not self.map_path.exists():
            self.map_path.write_text("{}", encoding="utf-8")

    def _load_map(self) -> Dict[str, int]:
        return json.loads(self.map_path.read_text(encoding="utf-8"))

    def _save_map(self, mp: Dict[str, int]):
        self.map_path.write_text(json.dumps(mp, ensure_ascii=False, indent=2), encoding="utf-8")

    def scan_and_append_new(self) -> int:
        existing_ids = set()
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj.get("id"))
                except Exception:
                    continue

        new_count = 0
        for session_dir in self.chat_root.iterdir():
            if not session_dir.is_dir() or session_dir.name in ("data", "index"):
                continue
            session_summary_file = session_dir / "summary.txt"
            session_summary = session_summary_file.read_text(encoding="utf-8") if session_summary_file.exists() else ""
            session_id = f"{session_dir.name}::session_summary"
            if session_summary and session_id not in existing_ids:
                rec = {
                    "id": session_id,
                    "path": str(session_summary_file.resolve()),
                    "title": f"[会话摘要] {session_dir.name}",
                    "rel": str(session_summary_file.relative_to(self.chat_root)),
                    "session": session_dir.name,
                    "turn": -1,
                    "text": session_summary,
                }
                with self.chunks_path.open("a", encoding="utf-8") as wf:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new_count += 1

            for td in sorted([p for p in session_dir.glob("turn_*") if p.is_dir()]):
                try:
                    turn_idx = int(td.name.split("_")[-1])
                except Exception:
                    continue
                ts = (td / "summary.txt").read_text(encoding="utf-8") if (td / "summary.txt").exists() else ""
                tid = f"{session_dir.name}::turn_{turn_idx:05d}::summary"
                if ts and tid not in existing_ids:
                    rec = {
                        "id": tid,
                        "path": str((td / "summary.txt").resolve()),
                        "title": f"[轮摘要] {session_dir.name} / turn {turn_idx}",
                        "rel": str((td / "summary.txt").relative_to(self.chat_root)),
                        "session": session_dir.name,
                        "turn": turn_idx,
                        "text": ts,
                    }
                    with self.chunks_path.open("a", encoding="utf-8") as wf:
                        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    new_count += 1
                for name in ("user.txt", "answer.txt"):
                    fp = td / name
                    if fp.exists():
                        cid = f"{session_dir.name}::turn_{turn_idx:05d}::{name}"
                        if cid not in existing_ids:
                            rec = {
                                "id": cid,
                                "path": str(fp.resolve()),
                                "title": f"[{name}] {session_dir.name} / turn {turn_idx}",
                                "rel": str(fp.relative_to(self.chat_root)),
                                "session": session_dir.name,
                                "turn": turn_idx,
                                "text": fp.read_text(encoding="utf-8"),
                            }
                            with self.chunks_path.open("a", encoding="utf-8") as wf:
                                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            new_count += 1
        return new_count

    def _load_index(self):
        id2row = self._load_map()
        if self.vecs_path.exists() and self.ids_path.exists() and id2row:
            vecs = np.load(self.vecs_path)
            ids = np.load(self.ids_path)
            return vecs, ids, id2row
        return np.empty((0,0), dtype=np.float32), np.empty((0,), dtype=np.int64), {}

    def _save_index(self, vecs, ids, id2row):
        np.save(self.vecs_path, vecs.astype(np.float32))
        np.save(self.ids_path, ids.astype(np.int64))
        self._save_map(id2row)

    def embed_incremental(self) -> int:
        id2row = self._load_map()
        rows = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append((i, obj))

        # 仅根据 id 判定是否已入库，避免重复
        pending = [(i, o) for (i,o) in rows if o["id"] not in id2row]

        if not pending:
            return 0

        client = OllamaClient(host=self.ollama_host, timeout=60)
        texts = [o["text"] for _,o in pending]
        embs = client.embeddings(model=self.embed_model, texts=texts)
        new_vecs = np.array(embs, dtype=np.float32)

        old_vecs, old_ids, old_map = self._load_index()
        if old_vecs.size == 0:
            vecs = new_vecs
            ids = np.array([i for i,_ in pending], dtype=np.int64)
            id2row = {}
        else:
            if old_vecs.shape[1] != new_vecs.shape[1]:
                vecs = new_vecs
                ids = np.array([i for i,_ in pending], dtype=np.int64)
                id2row = {}
            else:
                vecs = np.concatenate([old_vecs, new_vecs], axis=0)
                ids = np.concatenate([old_ids, np.array([i for i,_ in pending], dtype=np.int64)], axis=0)
                id2row = old_map

        start_row = len(id2row)
        for offset, (i, o) in enumerate(pending):
            id2row[o["id"]] = start_row + offset

        self._save_index(vecs, ids, id2row)
        return len(pending)

    def _cosine_topk(self, q: np.ndarray, M: np.ndarray, k: int):
        if M.size == 0:
            return []
        qn = q / (np.linalg.norm(q) + 1e-8)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        scores = Mn @ qn
        idx = np.argsort(-scores)[:k]
        return idx.tolist()

    def search(self, query: str, top_k: int = 8, scope: str = "session", session_name: Optional[str] = None):
        vecs, ids, id2row = self._load_index()
        if vecs.size == 0:
            return []
        client = OllamaClient(host=self.ollama_host, timeout=60)
        qv = np.array(client.embeddings(model=self.embed_model, texts=[query])[0], dtype=np.float32)
        top_idx = self._cosine_topk(qv, vecs, top_k*2)
        rows = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            all_lines = f.readlines()
        for j in top_idx:
            if j < 0 or j >= len(ids):
                continue
            row_id = int(ids[j])
            obj = json.loads(all_lines[row_id])
            if scope == "session" and session_name and obj.get("session") != session_name:
                continue
            rows.append({"row_id": row_id, "score": float(np.dot(vecs[j], qv)/(np.linalg.norm(vecs[j])*np.linalg.norm(qv)+1e-8)), **obj})
            if len(rows) >= top_k:
                break
        return rows
