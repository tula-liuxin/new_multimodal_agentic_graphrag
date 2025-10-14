
import os, sys, json, base64, argparse, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from .utils import (
    get_logger, load_cfg, read_jsonl, ensure_dir,
    chunk_path, chunk_ids_path, chunk_emb_path,
    image_list_path, image_emb_path, links_path, norm_win_abs
)
from .ollama_client import OllamaClient

def build_rel_index(chunks: List[Dict[str, Any]]):
    rel_to_idxs = {}
    for i, r in enumerate(chunks):
        rel = r.get("rel","")
        rel_to_idxs.setdefault(rel, []).append(i)
    # sort by chunk ordinal if present
    for rel, idxs in rel_to_idxs.items():
        idxs.sort(key=lambda i: chunks[i].get("chunk", 0))
    return rel_to_idxs

def pick_neighbors(idx: int, chunks: List[Dict[str, Any]], rel_to_idxs, k: int):
    if k <= 0: return []
    rel = chunks[idx].get("rel","")
    order = rel_to_idxs.get(rel, [])
    if not order: return []
    # find position of idx in order
    try:
        pos = order.index(idx)
    except ValueError:
        return []
    out = []
    # take k neighbors around pos
    left = pos - 1
    right = pos + 1
    while (left >= 0 or right < len(order)) and len(out) < k:
        if left >= 0:
            out.append(order[left]); left -= 1
            if len(out) >= k: break
        if right < len(order):
            out.append(order[right]); right += 1
    return out

def folder_siblings(rel: str, chunks: List[Dict[str, Any]], rel_to_idxs, k: int):
    if k <= 0: return []
    folder = os.path.dirname(rel)
    cands = []
    for r, idxs in rel_to_idxs.items():
        if os.path.dirname(r) == folder and r != rel:
            # pick the first chunk (0) if exists
            for i in idxs:
                if chunks[i].get("chunk", 0) == 0:
                    cands.append(i)
                    break
    return cands[:k]

def link_expansion(rel: str, links: List[Dict[str, Any]], rel_to_idxs, chunks: List[Dict[str, Any]], k: int = 3):
    if k <= 0 or not links: return []
    # build map
    src2dst = {row.get("src",""): row.get("dst",[]) for row in links}
    dsts = src2dst.get(rel, [])
    out = []
    for d in dsts:
        idxs = rel_to_idxs.get(d, [])
        if idxs:
            out.append(idxs[0])  # first chunk
        if len(out) >= k: break
    return out

def load_memmap(fp: str, dim: int = None):
    if not os.path.exists(fp): return None, 0, 0
    sz = os.path.getsize(fp)
    if dim is None:
        for d in (384, 512, 768, 1024, 1536):
            if sz % (4*d) == 0:
                n = sz // (4*d)
                return np.memmap(fp, dtype=np.float32, mode="r", shape=(n, d)), n, d
        return None, 0, 0
    else:
        n = sz // (4*dim)
        return np.memmap(fp, dtype=np.float32, mode="r", shape=(n, dim)), n, dim

def expand_query(q: str) -> List[str]:
    q = q.strip()
    toks = [q]
    for sep in ["，", "。", "；", ";", " ", "、"]:
        if sep in q:
            toks.extend([t for t in q.split(sep) if t])
    uniq, seen = [], set()
    for t in toks:
        t = t.strip()
        if not t or t in seen: continue
        seen.add(t); uniq.append(t)
    return uniq[:8]

def topk_dot(A: np.ndarray, q: np.ndarray, k: int = 20) -> List[Tuple[int, float]]:
    s = A @ q
    idx = np.argpartition(-s, min(k, len(s)-1))[:k]
    idx = idx[np.argsort(-s[idx])]
    return [(int(i), float(s[i])) for i in idx]

def b64image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def post_filter(question: str, cands: List[Dict[str, Any]], model: str, workers: int, timeout: int, base_url: str) -> List[Dict[str, Any]]:
    if not cands: return []
    client = OllamaClient(base_url)
    prompt_tpl = (
        "你是筛选助手。判断下方片段是否和用户问题相关，哪怕只有一点点相关也回答 YES，否则 NO。\n"
        "问题：{q}\n"
        "片段：{t}\n"
        "仅输出 YES 或 NO。"
    )
    kept = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2i = {}
        for i, c in enumerate(cands):
            t = c.get("text", "")[:1200] or c.get("ocr", "")[:1200]
            fut = ex.submit(client.generate, model, prompt_tpl.format(q=question, t=t), None, timeout)
            fut2i[fut] = i
        for fut in as_completed(fut2i):
            i = fut2i[fut]
            try:
                resp = fut.result()
            except Exception:
                continue
            if "YES" in (resp or "").upper():
                kept.append(cands[i])
    return kept

def main():
    log = get_logger("query")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart-query", action="store_true")
    ap.add_argument("--sqw", type=float, default=1.0)
    ap.add_argument("--text-first", type=int, default=6, dest="text_first")
    ap.add_argument("--wtext", type=float, default=1.0)
    ap.add_argument("--wimg", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--num-ctx", type=int, default=16000, dest="num_ctx")
    ap.add_argument("--ctx-chars", type=int, default=900, dest="ctx_chars")
    ap.add_argument("--neighbors", type=int, default=0)
    ap.add_argument("--folder-neighbors", type=int, default=0, dest="folder_neighbors")
    ap.add_argument("--embed-model", type=str, default="bge-m3:latest", dest="embed_model")
    ap.add_argument("--model", type=str, default="qwen3:4b-instruct-2507-fp16", help="最终回答模型")
    ap.add_argument("--post-filter", action="store_true", dest="post_filter")
    ap.add_argument("--pf-model", type=str, default="qwen2.5:3b-instruct", dest="pf_model")
    ap.add_argument("--pf-workers", type=int, default=4, dest="pf_workers")
    ap.add_argument("--pf-timeout", type=int, default=20, dest="pf_timeout")
    ap.add_argument("--pf-min-keep", type=int, default=6, dest="pf_min_keep")
    ap.add_argument("--vl-answer", action="store_true", dest="vl_answer")
    ap.add_argument("--max-images", type=int, default=3, dest="max_images")
    ap.add_argument("--gen-timeout", type=int, default=300, dest="gen_timeout")
    ap.add_argument("--debug-dir", required=True, type=str, dest="debug_dir")
    ap.add_argument("question", type=str)
    args = ap.parse_args()

    cfg = load_cfg()
    base_url = cfg.get("ollama_base_url", "http://127.0.0.1:11434")
    data_dir = cfg["data_dir"]
    ensure_dir(args.debug_dir)

    X, n, dim = load_memmap(chunk_emb_path(data_dir))
    ids = read_jsonl(chunk_ids_path(data_dir))
    chunks = read_jsonl(chunk_path(data_dir))
    if X is None or not ids or not chunks:
        log.warning("文本语义召回失败：缺少嵌入或索引文件（请先运行 python -m src.cli all）。")
        X = None
        n = dim = 0

    Xi, ni, dimi = load_memmap(image_emb_path(data_dir))
    images = read_jsonl(image_list_path(data_dir))

    client = OllamaClient(base_url)
    queries = [args.question]
    if args.smart_query:
        queries = expand_query(args.question)

    q_vecs = []
    for q in queries:
        try:
            v = client.embeddings(args.embed_model, [q])[0]
            q_vecs.append(np.asarray(v, dtype=np.float32))
        except Exception:
            continue
    if not q_vecs:
        print("（提示）生成模型调用失败，以下为检索到的上下文片段（供自查）：\n")
        return
    q_vec = np.mean(np.stack(q_vecs, axis=0), axis=0)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-6)

    text_hits = []
    if X is not None:
        text_hits = topk_dot(X, q_vec, k=max(args.top*5, 50))

    img_hits = []
    if Xi is not None:
        img_hits = topk_dot(Xi, q_vec, k=max(args.top*3, 24))

    cands = []
    T = args.text_first
    # --- neighbor & link expansions ---
    rel_to_idxs = build_rel_index(chunks) if chunks else {}
    link_rows = read_jsonl(links_path(data_dir))
    expanded = set(i for i,_ in text_hits[:max(T, args.top)])
    base_idxs = list(expanded)
    for idx in base_idxs:
        for j in pick_neighbors(idx, chunks, rel_to_idxs, args.neighbors):
            expanded.add(j)
        rel = chunks[idx].get("rel","") if chunks else ""
        for j in folder_siblings(rel, chunks, rel_to_idxs, args.folder_neighbors):
            expanded.add(j)
        for j in link_expansion(rel, link_rows, rel_to_idxs, chunks, k=3):
            expanded.add(j)
    # rebuild text hits restricted to expanded (keep original scores when available, else approximate with dot)
    forced = []
    if X is not None:
        # compute scores for any new idx not in original top
        orig_scores = {i:s for i,s in text_hits}
        for i in expanded:
            s = orig_scores.get(i, float(np.dot(np.asarray(X[i]), q_vec)))
            forced.append((i, s))
        forced.sort(key=lambda x: -x[1])
        text_hits = forced

    for (i, score) in text_hits[:max(T, args.top)]:
        meta = chunks[i]
        cands.append({
            "kind": "text",
            "score": float(score)*args.wtext,
            "path": meta.get("rel", ""),
            "text": meta.get("text", ""),
        })
    for (i, score) in img_hits[:args.top]:
        meta = images[i]
        cands.append({
            "kind": "image",
            "score": float(score)*args.wimg,
            "path": meta.get("rel", ""),
            "abs": meta.get("path",""),
            "ocr": meta.get("ocr",""),
        })

    if args.post_filter and cands:
        kept = post_filter(args.question, cands, args.pf_model, args.pf_workers, args.pf_timeout, base_url)
        if len(kept) < max(1, args.pf_min_keep):
            kept = sorted(cands, key=lambda x: -x["score"])[:args.pf_min_keep]
        cands = kept

    cands = sorted(cands, key=lambda x: -x["score"])[:args.top]

    ctx_lines = []
    refs = []
    imgs_b64 = []
    uniq_paths = []
    for c in cands:
        disp = c.get("path","").replace("\\\\?\\","")
        if disp not in uniq_paths:
            uniq_paths.append(disp)
            refs.append(disp)
        if c["kind"] == "text":
            txt = c.get("text","")[:args.ctx_chars]
            ctx_lines.append(f"[{len(refs)}] {disp}\n{txt}\n")
        else:
            ocr = c.get("ocr","")[:args.ctx_chars]
            ctx_lines.append(f"[{len(refs)}] {disp}\n(图片OCR)\n{ocr}\n")
            if args.vl_answer and len(imgs_b64) < args.max_images:
                abs_p = c.get("abs", disp)
                try:
                    imgs_b64.append(b64image(abs_p))
                except Exception:
                    pass

    with open(os.path.join(args.debug_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.debug_dir, "contexts.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ctx_lines))
    with open(os.path.join(args.debug_dir, "refs.json"), "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    if not cands:
        print("没有检索到与问题相关的材料，无法作答。")
        return

    sys_prompt = (
        "你是检索增强生成（RAG）的助手。根据“参考材料”回答用户问题。"
        "只回答你能从材料里找到的内容，必要时可以做简短推断。"
        "请在结尾列出编号化参考来源（例如：[1][3][5]）。"
    )
    context_block = "参考材料：\n" + "\n".join(ctx_lines) + "\n"
    full_prompt = f"{sys_prompt}\n问题：{args.question}\n{context_block}\n请给出结构化回答，最后附参考编号。"

    try:
        if args.vl_answer and imgs_b64:
            resp = OllamaClient(base_url).generate(model="qwen2.5vl:latest", prompt=full_prompt, images_b64=imgs_b64, timeout=args.gen_timeout)
        else:
            resp = OllamaClient(base_url).generate(model=args.model, prompt=full_prompt, images_b64=None, timeout=args.gen_timeout)
    except Exception:
        print("（提示）生成模型不可用，本次仅返回命中的参考材料。\n")
        print("参考：")
        for i, r in enumerate(refs, 1):
            print(f"[{i}] {r}")
        return

    print(resp.strip())
    print("\n参考：")
    for i, r in enumerate(refs, 1):
        print(f"[{i}] {r}")

if __name__ == "__main__":
    main()
