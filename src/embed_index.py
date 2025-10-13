import os, json, math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .utils import ensure_dir, get_logger, norm_win_abs
from .ollama_client import OllamaClient

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_embed(cfg, logger):
    data_dir = cfg["data_dir"]
    chunks_path = os.path.join(data_dir, "chunks.jsonl")
    ids_path = os.path.join(data_dir, "ids.json")
    meta_path = os.path.join(data_dir, "embed_meta.json")
    emb_npy = os.path.join(data_dir, "embeddings.npy")

    base = cfg["ollama_base_url"]
    model = cfg["embed_model"]
    batch = int(cfg.get("embed_batch", 128))
    concurrency = int(cfg.get("embed_concurrency", 2))
    timeout = int(cfg.get("embed_timeout", 120))
    keep_alive = cfg.get("embed_keep_alive", "10m")

    client = OllamaClient(base, timeout=timeout, keep_alive=keep_alive)
    texts = []
    ids = []
    for rec in read_jsonl(chunks_path):
        ids.append(rec["id"])
        texts.append(rec["text"] if rec["text"].strip() else " ")
    n = len(texts)
    if n == 0:
        logger.warning("chunks.jsonl 为空")
        return

    # 试探维度
    dim = len(client.embeddings(model, ["test"])[0])

    # memmap 逐批写入
    mm = np.memmap(emb_npy, dtype=np.float32, mode="w+", shape=(n, dim))
    order = list(range(0, n, batch))

    def work(start):
        end = min(n, start + batch)
        sub = texts[start:end]
        emb = client.embeddings(model, sub)
        return start, np.array(emb, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(work, s) for s in order]
        pbar = tqdm(total=n, desc="embedding", unit="chunk")
        for fut in as_completed(futures):
            s, arr = fut.result()
            mm[s:s+arr.shape[0], :] = arr
            pbar.update(arr.shape[0])
        pbar.close()
    mm.flush()

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model":model, "dim":dim, "count":n}, f, ensure_ascii=False, indent=2)
