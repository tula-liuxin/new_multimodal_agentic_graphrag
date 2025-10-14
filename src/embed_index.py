# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, time
from typing import List, Dict, Tuple, Optional
import numpy as np
from .ollama_client import OllamaClient

IDX_DIR = "index"
TEXT_VECS = os.path.join(IDX_DIR, "text_vecs.npy")
TEXT_IDS = os.path.join(IDX_DIR, "text_ids.jsonl")
META_JSON = os.path.join(IDX_DIR, "text_index_meta.json")

def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def _write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _read_meta() -> dict:
    if os.path.exists(META_JSON):
        try:
            return json.load(open(META_JSON, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_meta(meta: dict):
    _write_json(META_JSON, meta)

def load_text_index() -> Tuple[Optional[np.ndarray], List[Dict], dict]:
    """安全加载文本向量索引。返回 (vecs, ids_rows, meta)"""
    meta = _read_meta()
    ids_rows = _read_jsonl(TEXT_IDS)
    if not os.path.exists(TEXT_VECS):
        return None, ids_rows, meta
    try:
        vecs = np.load(TEXT_VECS, mmap_mode="r")
    except Exception:
        # 不可读则尝试普通读取
        vecs = np.load(TEXT_VECS)
    if vecs.ndim != 2 or vecs.shape[0] != len(ids_rows):
        # 形状不匹配，放弃语义召回
        return None, ids_rows, meta
    return vecs, ids_rows, meta

def _batched(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i+bs]

def build_text_index(chunks_path: str, embed_model: str, ollama_host: Optional[str], batch_size: int = 128, logger=None):
    """重建文本向量索引，写出 vecs/ids/meta。"""
    rows = _read_jsonl(chunks_path)
    if not rows:
        if logger: logger.warning("没有可用文本块，跳过文本向量构建。")
        return
    client = OllamaClient(host=ollama_host, timeout=120, retries=1, logger=logger)
    texts = [r.get("text","")[:1200] for r in rows]
    vecs_all = []
    if logger: logger.info(f"文本向量：共 {len(texts)} 块，batch={batch_size}，model={embed_model}")
    for batch in _batched(texts, batch_size):
        emb = client.embeddings(model=embed_model, texts=batch)
        if not emb:
            continue
        vecs_all.extend(emb)
    if not vecs_all:
        raise RuntimeError("未得到任何文本向量，请检查 Ollama embeddings 是否可用。")
    vecs = np.asarray(vecs_all, dtype=np.float32)
    os.makedirs(os.path.dirname(TEXT_VECS), exist_ok=True)
    np.save(TEXT_VECS, vecs)
    # 写 ids（与 chunks 对齐）
    with open(TEXT_IDS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r.get("id"), "path": r.get("path"), "title": r.get("title",""), "chunk_id": r.get("chunk_id",0)}, ensure_ascii=False) + "\n")
    _save_meta({"model": embed_model, "dim": int(vecs.shape[1]), "ts": time.time()})
    if logger: logger.info(f"文本索引完成：vecs={vecs.shape} → {TEXT_VECS}")
    return vecs
