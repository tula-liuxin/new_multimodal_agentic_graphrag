
import os, numpy as np
from typing import List, Dict, Any
from .utils import get_logger, load_cfg, chunk_path, chunk_emb_path, chunk_ids_path, read_jsonl
from .ollama_client import OllamaClient

def run_embed(cfg=None, logger=None, embed_model: str = None):
    log = logger or get_logger("embed")
    cfg = cfg or load_cfg()
    data_dir = cfg["data_dir"]
    embed_model = embed_model or cfg.get("embed_model", "bge-m3:latest")

    rows = read_jsonl(chunk_path(data_dir))
    if not rows:
        log.info("没有发现文本块（chunks.jsonl）。请先运行 ingest。")
        return

    client = OllamaClient(cfg.get("ollama_base_url", "http://127.0.0.1:11434"))
    texts = [r.get("text","") for r in rows]
    embs = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        vecs = client.embeddings(embed_model, batch)
        embs.extend(vecs)
        if (i//B) % 10 == 0:
            log.info(f"embedding 进度: {min(i+len(batch), len(texts))}/{len(texts)}")

    embs = np.array(embs, dtype=np.float32)
    embs.tofile(chunk_emb_path(data_dir))
    # Save ids in same order
    from .utils import write_jsonl
    ids = [{"id": r["id"], "rel": r["rel"], "chunk": r["chunk"]} for r in rows]
    write_jsonl(chunk_ids_path(data_dir), ids)
    log.info(f"文本嵌入完成：{embs.shape}")
