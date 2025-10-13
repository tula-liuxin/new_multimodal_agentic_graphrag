import argparse, os, json, re, math, time
import numpy as np
import joblib
from typing import List, Dict, Tuple
from .utils import load_cfg, get_logger, ensure_dir, norm_win_abs
from .ollama_client import OllamaClient
from .multimodal_fusion import fuse_scores, mmr_select, read_jsonl
import open_clip, torch
from PIL import Image

# --- Keyword expansion & boolean scoring helpers ---
AGI_SYNONYMS = {
    "AGI": ["AGI", "通用人工智能", "人工通用智能", "强人工智能", "通用智能"],
    "ASI": ["ASI", "超人工智能", "超强智能", "超级智能"],
    "LLM": ["LLM", "大型语言模型", "大语言模型", "大模型", "语言模型"],
}

def normalize_tokens(q: str):
    toks = re.findall(r"[一-龥A-Za-z0-9_]+", q)
    out = []
    for t in toks:
        if re.search(r"[A-Za-z]", t):
            out.append(t.upper())
        else:
            out.append(t)
    return out

def expand_terms(tokens):
    expanded = set(tokens)
    for t in list(tokens):
        if t.upper() in AGI_SYNONYMS:
            for s in AGI_SYNONYMS[t.upper()]:
                expanded.add(s.upper() if re.search(r"[A-Za-z]", s) else s)
        for k, syns in AGI_SYNONYMS.items():
            if t in syns:
                expanded.add(k)
    return list(expanded)

def make_combos(tokens, max_combo=2):
    toks = [t for t in tokens if len(t) >= 2]
    combos = set()
    n = len(toks)
    for i in range(n):
        for j in range(i+1, n):
            combos.add(tuple(sorted([toks[i], toks[j]])))
            if max_combo >= 3:
                for k in range(j+1, n):
                    combos.add(tuple(sorted([toks[i], toks[j], toks[k]])))
    return list(combos)

def bool_channel_score(text: str, tokens, combos):
    if not text:
        return 0.0
    t = text.upper()
    hits = sum(1 for tok in tokens if tok.upper() in t)
    frac = 0.0 if len(tokens)==0 else hits/len(tokens)
    combo_hits = 0
    for c in combos:
        if all(tok.upper() in t for tok in c):
            combo_hits += 1
    combo_frac = combo_hits / max(1, len(combos))
    return 0.6*frac + 0.4*combo_frac

def cosine_rows(A, b):
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    nb = np.linalg.norm(b) + 1e-9
    na = np.linalg.norm(A, axis=1) + 1e-9
    sim = (A @ b) / (na * nb)
    return sim

def smart_keywords(q: str, weight: float=1.2) -> List[str]:
    toks = re.findall(r"[一-龥A-Za-z0-9_]+", q)
    stop = {"的","了","和","与","及","在","有","是","我","我们","最近","计划"}
    kws = [t for t in toks if t not in stop and len(t) >= 2]
    return kws

def load_dense(data_dir):
    emb_path = os.path.join(data_dir, "embeddings.npy")
    meta_path = os.path.join(data_dir, "embed_meta.json")
    ids_path = os.path.join(data_dir, "ids.json")
    ids = json.load(open(ids_path, "r", encoding="utf-8"))
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    count, dim = int(meta.get("count", len(ids))), int(meta.get("dim", 0))
    arr = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(count, dim))
    return arr, ids

def load_tfidf(data_dir):
    vec = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.joblib"))
    X = joblib.load(os.path.join(data_dir, "tfidf_matrix.joblib"))
    return vec, X

def load_char(data_dir):
    vec = joblib.load(os.path.join(data_dir, "tfidf_char_vectorizer.joblib"))
    X = joblib.load(os.path.join(data_dir, "tfidf_char_matrix.joblib"))
    return vec, X

def load_chunks(data_dir):
    return [json.loads(l) for l in open(os.path.join(data_dir, "chunks.jsonl"), "r", encoding="utf-8").read().splitlines() if l.strip()]

def load_images(data_dir):
    p = os.path.join(data_dir, "images.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, "r", encoding="utf-8").read().splitlines() if l.strip()]

def load_link_graph(data_dir):
    p = os.path.join(data_dir, "link_graph.json")
    if os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))
    return {"nodes":{}, "edges":[]}

def image_query_to_scores(cfg, q: str, data_dir: str, images, wimg: float):
    ids_path = os.path.join(data_dir, "image_ids.json")
    emb_path = os.path.join(data_dir, "image_embeddings.npy")
    if not (os.path.exists(ids_path) and os.path.exists(emb_path)):
        return None, None, None
    img_ids = json.load(open(ids_path, "r", encoding="utf-8"))
    flat = np.memmap(emb_path, dtype=np.float32, mode="r")
    if len(img_ids) == 0:
        return None, None, None
    dim = flat.size // len(img_ids)
    img_vecs = flat.reshape(len(img_ids), dim)
    model_name = cfg.get("image_model_name", "ViT-L-14")
    pretrained = cfg.get("image_pretrained", "laion2b_s32b_b82k")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tok = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        txt = tok([q])
        txt = torch.tensor(txt).to(device) if not isinstance(txt, torch.Tensor) else txt.to(device)
        if hasattr(model, "encode_text"):
            tfeat = model.encode_text(txt)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            tfeat = tfeat.detach().cpu().float().numpy()[0]
        else:
            return None, None, None
    sims = cosine_rows(img_vecs, tfeat)
    return sims, img_ids, images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["tfidf","dense","hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.5, help="词面 TF-IDF 权重")
    parser.add_argument("--beta", type=float, default=0.3, help="字 n-gram 权重")
    parser.add_argument("--gamma", type=float, default=0.8, help="文本向量权重")
    parser.add_argument("--wimg", type=float, default=0.25, help="图像通道权重")
    parser.add_argument("--wbool", type=float, default=0.6, help="关键词/组合布尔通道权重")

    parser.add_argument("--mmr", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--link-hop", action="store_true")
    parser.add_argument("--link-depth", type=int, default=1)
    parser.add_argument("--folder-neighbors", type=int, default=0)

    parser.add_argument("--smart-query", action="store_true")
    parser.add_argument("--sqw", type=float, default=1.2)

    parser.add_argument("--strict-mode", choices=["off","exact","smart"], default="off")

    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=28000)
    parser.add_argument("--ctx-chars", type=int, default=1200)

    parser.add_argument("--wclause", type=float, default=0.8)
    parser.add_argument("--wn", type=float, default=0.9)
    parser.add_argument("--wq", type=float, default=1.0)

    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--model", default=None, help="用于回答/扩展的本地模型（Ollama tag）")

    parser.add_argument("question", nargs="+")

    args = parser.parse_args()
    cfg = load_cfg(args.config)
    logger = get_logger()

    q = " ".join(args.question)
    data_dir = cfg["data_dir"]
    input_dir = cfg["input_dir"]
    debug_dir = args.debug_dir or cfg.get("debug_dir")

    chunks = load_chunks(data_dir)
    if len(chunks) == 0:
        print("没有可用的 chunks（请先运行 ingest）。")
        return
    dense, ids = None, None
    if args.mode in ["dense","hybrid"]:
        dense, ids = load_dense(data_dir)
    vec_word, X_word = None, None
    vec_char, X_char = None, None
    if args.mode in ["tfidf","hybrid"]:
        vec_word, X_word = load_tfidf(data_dir)
        vec_char, X_char = load_char(data_dir)

    images = load_images(data_dir)
    link_graph = load_link_graph(data_dir)

    q_eff = q
    if args.smart_query:
        kws = smart_keywords(q, weight=args.sqw)
        if kws:
            q_eff = q + " " + " ".join(kws)

    # 关键词扩展（含 AGI/ASI 等）
    base_tokens = normalize_tokens(q_eff)
    expanded = expand_terms(base_tokens)
    combos = make_combos(expanded, max_combo=2)

    N = len(chunks)
    scores_word = None
    scores_char = None
    scores_dense = None
    scores_img = None

    if args.mode in ["tfidf","hybrid"]:
        vq = vec_word.transform([q_eff])
        scores_word = (X_word @ vq.T).toarray().ravel()
        vqc = vec_char.transform([q_eff])
        scores_char = (X_char @ vqc.T).toarray().ravel()

    if args.mode in ["dense","hybrid"]:
        client = OllamaClient(cfg["ollama_base_url"], timeout=int(cfg.get("embed_timeout",120)), keep_alive=cfg.get("embed_keep_alive","10m"))
        qemb = client.embeddings(cfg["embed_model"], [q_eff])[0]
        qemb = np.array(qemb, dtype=np.float32)
        scores_dense = cosine_rows(dense, qemb)

    img_info = None
    if args.mode == "hybrid" and not args.images_only and args.wimg > 0:
        s_img, img_ids, img_recs = image_query_to_scores(cfg, q_eff, data_dir, images, args.wimg)
        if s_img is not None:
            img_map = {}
            for sim, rec in zip(s_img, img_recs):
                img_map[rec["source_path"]] = float(sim)
            scores_img = np.zeros(N, dtype=np.float32)
            for i, rec in enumerate(chunks):
                sp = rec["source_path"]
                if sp in img_map:
                    scores_img[i] = img_map[sp]
                else:
                    base_dir = os.path.dirname(sp)
                    best = 0.0
                    for k, v in img_map.items():
                        if os.path.dirname(k) == base_dir:
                            best = max(best, v*0.8)
                    scores_img[i] = best
            img_info = (s_img, img_ids, img_recs)

    # 初选候选（减少布尔扫描开销）：从各通道取前 100 的并集
    prelim = set()
    if scores_word is not None: prelim.update(np.argsort(-scores_word)[:100])
    if scores_char is not None: prelim.update(np.argsort(-scores_char)[:100])
    if scores_dense is not None: prelim.update(np.argsort(-scores_dense)[:100])
    if not prelim:
        prelim = set(range(N))
    prelim = list(prelim)

    scores_bool = np.zeros(N, dtype=np.float32)
    for i in prelim:
        rec = chunks[i]
        scores_bool[i] = bool_channel_score(rec.get("text",""), expanded, combos)

    scores = fuse_scores(
        {"word":scores_word, "char":scores_char, "dense":scores_dense, "img":scores_img, "bool":scores_bool},
        {"word":args.alpha, "char":args.beta, "dense":args.gamma, "img":args.wimg, "bool":args.wbool}
    )
    if scores is None:
        print("没有可融合的分数，请确认已构建对应索引。")
        return

    order = np.argsort(-scores)

    if args.strict_mode == "exact":
        mask = np.array([ (re.search(re.escape(q), chunks[i]["text"]) is not None) for i in range(len(chunks)) ])
        order = [i for i in order if mask[i]]
    elif args.strict_mode == "smart":
        kws = expanded
        mask = np.array([ all(k.upper() in chunks[i]["text"].upper() for k in kws) for i in range(len(chunks)) ]) if kws else np.ones(len(chunks), dtype=bool)
        order = [i for i in order if mask[i]]

    idx_top = order[:min(200, len(order))]
    cands = [(i, float(scores[i])) for i in idx_top]
    selected = [i for (i, _) in cands[:args.top]]

    chosen_idx = selected[:args.top]
    chosen_chunks = [chunks[i] for i in chosen_idx]

    images_used = []
    if img_info is not None:
        s_img, img_ids, img_recs = img_info
        img_order = np.argsort(-s_img)[:args.top]
        images_used = [img_recs[i] for i in img_order if s_img[i] > 0]

    # 构造上下文
    total = 0
    ctx_parts = []
    for rec in chosen_chunks:
        t = rec["text"]
        if total + len(t) > args.num_ctx:
            t = t[:max(0, args.num_ctx-total)]
        ctx_parts.append(t)
        total += len(t)
        if total >= args.num_ctx:
            break
    for im in images_used:
        extras = []
        if im.get("caption"):
            extras.append(f"字幕：{im['caption']}")
        if im.get("ocr_text"):
            extras.append(f"OCR：{im['ocr_text'][:200]}")
        if extras:
            ctx_parts.append("【相关图片】" + "；".join(extras))
    ctx = "\n\n".join(ctx_parts)

    answer = ""
    if args.model:
        sys_prompt = "你是可靠的企业内检索与问答助手。回答要基于提供的上下文，若没有证据，请直说不知道。"
        client = OllamaClient(cfg["ollama_base_url"], timeout=int(cfg.get("embed_timeout",120)), keep_alive=cfg.get("embed_keep_alive","10m"))
        prompt = f"问题：{q}\n\n参考上下文：\n{ctx}\n\n请用中文直接回答。"
        try:
            answer = client.generate(args.model, prompt, temperature=0.2, system=sys_prompt)
        except Exception as e:
            answer = f"(生成失败，返回检索片段)\n\n" + ctx
    else:
        answer = ctx

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "query.txt"), "w", encoding="utf-8") as f:
            f.write(f"question: {q}\n")
            if args.smart_query:
                f.write(f"smart_keywords: {' '.join(smart_keywords(q))}\n")
            f.write(f"q_eff: {q_eff}\n")
            f.write(f"expanded_terms: {', '.join(expanded)}\n")
            f.write(f"combos: {json.dumps(combos, ensure_ascii=False)}\n")
        cands_json = []
        for i in chosen_idx:
            cands_json.append({
                "id": chunks[i]["id"],
                "source_path": chunks[i]["source_path"],
                "score": float(scores[i])
            })
        with open(os.path.join(debug_dir, "candidates.json"), "w", encoding="utf-8") as f:
            json.dump(cands_json, f, ensure_ascii=False, indent=2)
        pipe = {
            "mode": args.mode,
            "weights": {"alpha":args.alpha, "beta":args.beta, "gamma":args.gamma, "wimg":args.wimg, "wbool":args.wbool},
            "mmr": args.mmr, "neighbors": args.neighbors,
            "strict_mode": args.strict_mode,
            "top": args.top, "num_ctx": args.num_ctx, "ctx_chars": args.ctx_chars
        }
        with open(os.path.join(debug_dir, "pipeline.json"), "w", encoding="utf-8") as f:
            json.dump(pipe, f, ensure_ascii=False, indent=2)
        with open(os.path.join(debug_dir, "contexts.txt"), "w", encoding="utf-8") as f:
            f.write(ctx)
        if images_used:
            with open(os.path.join(debug_dir, "images.json"), "w", encoding="utf-8") as f:
                json.dump(images_used, f, ensure_ascii=False, indent=2)

    print(answer.strip())
    used_paths = set()
    for rec in chosen_chunks:
        used_paths.add(norm_win_abs(os.path.join(input_dir, rec["source_path"])))
    for im in images_used:
        used_paths.add(norm_win_abs(os.path.join(input_dir, im["source_path"])))
    if used_paths:
        print("\n参考：")
        for p in used_paths:
            print(p)

if __name__ == "__main__":
    main()
