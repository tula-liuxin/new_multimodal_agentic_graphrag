import argparse, os, json, re, math, time, sys
import numpy as np
import joblib
from typing import List, Dict, Tuple
from .utils import load_cfg, get_logger, ensure_dir, norm_win_abs
from .ollama_client import OllamaClient
from .multimodal_fusion import fuse_scores, mmr_select, read_jsonl
from .zh_utils import normalize_zh, is_cjk_name
import open_clip, torch
from PIL import Image

def cosine_rows(A, b):
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    nb = np.linalg.norm(b) + 1e-9
    na = np.linalg.norm(A, axis=1) + 1e-9
    sim = (A @ b) / (na * nb)
    return sim

CN_INSTR_STOP = {"所有记录","所有信息","全部信息","全部记录","总体","完整","汇总","总结","整理","如何","怎么","请","帮我","需要","希望","想要","请给出","的","关于","相关","最近","本周","本月","下一步","目标","安排","计划","日程","行程","ToDo","待办","OKR"}

def smart_keywords(q: str, weight: float=1.2) -> List[str]:
    toks = re.findall(r"[一-龥A-Za-z0-9_]+", q)
    kws = [t for t in toks if t not in CN_INSTR_STOP and len(t) >= 2]
    return kws

def robust_json_find(s: str):
    m = re.search(r'\{.*\}', s, flags=re.S)
    if not m:
        return None
    txt = m.group(0)
    try:
        return json.loads(txt)
    except Exception:
        txt = re.sub(r',\s*}', '}', txt)
        txt = re.sub(r',\s*]', ']', txt)
        try:
            return json.loads(txt)
        except Exception:
            return None

def llm_expand_terms(cfg, model_tag: str, question: str):
    client = OllamaClient(cfg["ollama_base_url"], timeout=int(cfg.get("embed_timeout",120)), keep_alive=cfg.get("embed_keep_alive","10m"))
    sys_prompt = "你是检索前的查询理解器。仅输出JSON，不要解释。"
    prompt = (
        "请阅读用户问题，抽取可用于检索的关键信息。"
        "把指令性词（如“所有记录/全部信息/请/如何”等）忽略，只保留用于命中的实体、别名、关键词、可能的时间限定。"
        "输出JSON，字段："
        "{"
        "  \"keywords\": [\"必搜词\",\"核心实体\"],"
        "  \"synonyms\": [\"别名或同义词\"],"
        "  \"exclude\": [\"应当排除的词\"],"
        "  \"time\": \"可选时间范围描述，如 2025年10月/最近一周\""
        "}"
        f"问题：{question}"
    )
    try:
        out = client.generate(model_tag, prompt, temperature=0.1, system=sys_prompt)
    except Exception as e:
        return None
    obj = robust_json_find(out or "")
    if not obj:
        return None
    def norm(ts):
        res = []
        for t in ts or []:
            t = str(t).strip()
            if not t: continue
            if t in CN_INSTR_STOP: continue
            res.append(t)
        return res
    return {
        "keywords": norm(obj.get("keywords")),
        "synonyms": norm(obj.get("synonyms")),
        "exclude": norm(obj.get("exclude")),
        "time": (obj.get("time") or "").strip()
    }

def load_dense(data_dir):
    emb_path = os.path.join(data_dir, "embeddings.npy")
    meta_path = os.path.join(data_dir, "embed_meta.json")
    ids_path = os.path.join(data_dir, "ids.json")
    ids = json.load(open(ids_path, "r", encoding="utf-8"))
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if os.path.exists(meta_path) else {"count": len(ids), "dim": 0}
    file_size = os.path.getsize(emb_path)
    dim = int(meta.get("dim") or 0)
    if dim <= 0:
        commons = [3072, 2048, 1536, 1024, 768, 512, 384]
        dim = next((d for d in commons if file_size % (4*d) == 0), 768)
    count_from_size = file_size // (4 * dim)
    count_meta = int(meta.get("count", len(ids)))
    count = min(count_from_size, count_meta, len(ids))
    try:
        arr = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(count, dim))
    except (OSError, ValueError):
        flat = np.memmap(emb_path, dtype=np.float32, mode="r")
        total = flat.size
        count_from_flat = total // dim
        count = min(count, count_from_flat, len(ids))
        arr = flat[:count*dim].reshape(count, dim)
    if len(ids) != count:
        ids = ids[:count]
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
        if t in CN_INSTR_STOP:
            continue
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
    frac = hits / len(tokens) if len(tokens) else 0.0
    combo_hits = 0
    for c in combos:
        if all(tok.upper() in t for tok in c):
            combo_hits += 1
    combo_frac = combo_hits / max(1, len(combos))
    return 0.6 * frac + 0.4 * combo_frac

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
    sims = (img_vecs @ tfeat) / ((np.linalg.norm(img_vecs, axis=1) + 1e-9) * (np.linalg.norm(tfeat) + 1e-9))
    return sims, img_ids, images

def build_context(chosen_chunks: List[Dict], images_used: List[Dict], cfg, num_ctx: int, ctx_chars: int):
    ctx_parts = []
    total = 0
    for rec in chosen_chunks:
        t = rec["text"]
        if total + len(t) > num_ctx:
            t = t[:max(0, num_ctx-total)]
        ctx_parts.append(t[:ctx_chars])
        total += len(t)
        if total >= num_ctx:
            break
    for im in images_used:
        extras = []
        if im.get("caption"):
            extras.append(f"字幕：{im['caption']}")
        if im.get("ocr_text"):
            extras.append(f"OCR：{im['ocr_text'][:200]}")
        if extras:
            ctx_parts.append("【相关图片】" + "；".join(extras))
    return "\n\n".join(ctx_parts)

def hyde_expand(cfg, model_tag: str, question: str):
    client = OllamaClient(cfg["ollama_base_url"], timeout=int(cfg.get("embed_timeout",120)), keep_alive=cfg.get("embed_keep_alive","10m"))
    sys_prompt = "请撰写一段与用户问题高度相关的说明性段落（用于检索，不必真实），不超过200字。"
    try:
        txt = client.generate(model_tag, question, temperature=0.2, system=sys_prompt)
        return txt.strip()
    except Exception:
        return ""

def prf_terms(vec, X, top_idx, k=10):
    # 取初选文档的 tf-idf 权重总和，挑前 k 个词作为扩展
    if len(top_idx) == 0:
        return []
    sub = X[top_idx]
    s = np.asarray(sub.sum(axis=0)).ravel()
    try:
        feats = vec.get_feature_names_out()
    except Exception:
        return []
    order = np.argsort(-s)[:k]
    return [feats[i] for i in order]

def fuzzy_scores(prelim_idx, chunks, q):
    try:
        from rapidfuzz import fuzz
    except Exception:
        return np.zeros(len(chunks), dtype=np.float32)
    nq = normalize_zh(q)
    scores = np.zeros(len(chunks), dtype=np.float32)
    for i in prelim_idx:
        t = normalize_zh(chunks[i].get("text","")[:500])
        scores[i] = fuzz.partial_ratio(nq, t) / 100.0
    return scores

def cross_rerank(pairs, model_name="BAAI/bge-reranker-v2-m3"):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    ce = CrossEncoder(model_name, trust_remote_code=True)
    scores = ce.predict(pairs, show_progress_bar=False)
    return np.array(scores, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["tfidf","dense","hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.5, help="词面 TF-IDF 权重")
    parser.add_argument("--beta", type=float, default=0.3, help="字 n-gram 权重")
    parser.add_argument("--gamma", type=float, default=0.8, help="文本向量权重")
    parser.add_argument("--wimg", type=float, default=0.25, help="图像通道权重")
    parser.add_argument("--wbool", type=float, default=0.6, help="关键词/组合布尔通道权重")
    parser.add_argument("--wfuzzy", type=float, default=0.5, help="模糊匹配通道权重")
    parser.add_argument("--images-only", action="store_true")

    parser.add_argument("--mmr", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--link-hop", action="store_true")
    parser.add_argument("--link-depth", type=int, default=1)
    parser.add_argument("--folder-neighbors", type=int, default=0)

    parser.add_argument("--smart-query", action="store_true")
    parser.add_argument("--sqw", type=float, default=1.2)
    parser.add_argument("--llm-expand", action="store_true", help="使用本地 LLM 抽取关键词/别名/时间等后再检索")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE：先生成一段假想文档再做向量检索")
    parser.add_argument("--prf", action="store_true", help="启用 PRF：用初选文档自动扩展查询词")

    parser.add_argument("--strict-mode", choices=["off","exact","smart"], default="off")

    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=28000)
    parser.add_argument("--ctx-chars", type=int, default=1200)

    parser.add_argument("--wclause", type=float, default=0.8)
    parser.add_argument("--wn", type=float, default=0.9)
    parser.add_argument("--wq", type=float, default=1.0)

    parser.add_argument("--cross-rerank", action="store_true", help="使用交叉编码器重排 TopK")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-k", type=int, default=50)

    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--model", default=None, help="用于回答/扩展/LLM抽取的本地模型（Ollama tag）")

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

    # LLM 预处理
    llm_info = None
    if args.llm_expand and args.model:
        llm_info = llm_expand_terms(cfg, args.model, q)

    q_eff = q
    seed_terms = []
    if llm_info:
        seed_terms += llm_info.get("keywords", [])
        seed_terms += llm_info.get("synonyms", [])
        seed_terms = [t for t in seed_terms if t not in CN_INSTR_STOP]
        if seed_terms:
            q_eff = q + " " + " ".join(seed_terms)

    if args.smart_query:
        kws = smart_keywords(q, weight=args.sqw)
        if kws:
            q_eff = q_eff + " " + " ".join(kws)

    # HyDE：合成段落并一起编码
    hyde_txt = ""
    if args.hyde and args.model and args.mode in ["dense","hybrid"]:
        hyde_txt = hyde_expand(cfg, args.model, q)
        if hyde_txt:
            q_eff = q_eff + " " + hyde_txt

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
    if args.mode == "hybrid" and not getattr(args, "images_only", False) and args.wimg > 0:
        s_img, img_ids, img_recs = image_query_to_scores(cfg, q_eff, data_dir, images, args.wimg)
        if s_img is not None:
            img_map = {rec["source_path"]: float(sim) for sim, rec in zip(s_img, img_recs)}
            scores_img = np.zeros(N, dtype=np.float32)
            for i, rec in enumerate(chunks):
                sp = rec["source_path"]
                base_dir = os.path.dirname(sp)
                if sp in img_map:
                    scores_img[i] = img_map[sp]
                else:
                    best = 0.0
                    for k, v in img_map.items():
                        if os.path.dirname(k) == base_dir:
                            best = max(best, v*0.8)
                    scores_img[i] = best
            img_info = (s_img, img_ids, img_recs)

    if getattr(args, "images_only", False):
        s_img, img_ids, img_recs = image_query_to_scores(cfg, q_eff, data_dir, images, args.wimg)
        if s_img is None:
            print("未找到图像索引（先运行 imgindex）。")
            return
        order = np.argsort(-s_img)[:args.top]
        print("图片检索 Top:")
        for i in order:
            rec = img_recs[i]
            abs_path = norm_win_abs(os.path.join(cfg["input_dir"], rec["source_path"]))
            print(f"{abs_path}  (score={s_img[i]:.4f})")
        return

    # Boolean channel
    base_tokens = normalize_tokens(q_eff)
    if llm_info:
        base_tokens += [t for t in (llm_info.get("keywords", []) + llm_info.get("synonyms", [])) if t not in CN_INSTR_STOP]
    expanded = expand_terms(base_tokens)
    combos = make_combos(expanded, max_combo=2)

    prelim = set()
    if scores_word is not None: prelim.update(np.argsort(-scores_word)[:200])
    if scores_char is not None: prelim.update(np.argsort(-scores_char)[:200])
    if scores_dense is not None: prelim.update(np.argsort(-scores_dense)[:200])
    if not prelim:
        prelim = set(range(N))
    prelim = list(prelim)

    scores_bool = np.zeros(N, dtype=np.float32)
    for i in prelim:
        rec = chunks[i]
        scores_bool[i] = bool_channel_score(rec.get("text",""), expanded, combos)

    # Fuzzy channel on prelim
    scores_fuzzy = fuzzy_scores(prelim, chunks, q)

    scores = fuse_scores(
        {"word":scores_word, "char":scores_char, "dense":scores_dense, "img":scores_img, "bool":scores_bool, "fuzzy":scores_fuzzy},
        {"word":args.alpha, "char":args.beta, "dense":args.gamma, "img":args.wimg, "bool":args.wbool, "fuzzy":args.wfuzzy}
    )
    if scores is None:
        print("没有可融合的分数，请确认已构建对应索引。")
        return

    order = np.argsort(-scores)

    # PRF：基于初选做二次词扩展并重算 lexical
    if args.prf and args.mode in ["tfidf","hybrid"]:
        idx_seed = order[: min(100, len(order))]
        extra_terms = prf_terms(vec_word, X_word, idx_seed, k=10)
        if extra_terms:
            q_eff2 = q_eff + " " + " ".join(extra_terms)
            vq2 = vec_word.transform([q_eff2])
            scores_word2 = (X_word @ vq2.T).toarray().ravel()
            vqc2 = vec_char.transform([q_eff2])
            scores_char2 = (X_char @ vqc2.T).toarray().ravel()
            scores = fuse_scores(
                {"word":scores_word2, "char":scores_char2, "dense":scores_dense, "img":scores_img, "bool":scores_bool, "fuzzy":scores_fuzzy},
                {"word":args.alpha, "char":args.beta, "dense":args.gamma, "img":args.wimg, "bool":args.wbool, "fuzzy":args.wfuzzy}
            )
            order = np.argsort(-scores)

    # strict filters
    if False:  # keep code path for completeness; currently we rely on expansions
        pass

    # choose top
    idx_top = order[:min(max(args.top*6, 60), len(order))]
    if args.cross_rerank:
        pairs = []
        for i in idx_top[:args.rerank_k]:
            txt = chunks[i]["text"][:args.ctx_chars]
            pairs.append((q, txt))
        rr = cross_rerank(pairs, model_name=args.reranker)
        if rr is not None:
            # rerank only a subset; pick final top by reranker
            order_local = np.argsort(-rr)[:args.top]
            chosen_idx = [idx_top[:args.rerank_k][j] for j in order_local]
        else:
            chosen_idx = list(idx_top[:args.top])
    else:
        chosen_idx = list(idx_top[:args.top])

    chosen_chunks = [chunks[i] for i in chosen_idx]

    images_used = []
    if img_info is not None:
        s_img, img_ids, img_recs = img_info
        img_order = np.argsort(-s_img)[:args.top]
        images_used = [img_recs[i] for i in img_order if s_img[i] > 0]

    ctx = build_context(chosen_chunks, images_used, cfg, args.num_ctx, args.ctx_chars)

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
            if llm_info:
                f.write(f"llm_keywords: {', '.join(llm_info.get('keywords', []))}\n")
                f.write(f"llm_synonyms: {', '.join(llm_info.get('synonyms', []))}\n")
                f.write(f"llm_time: {llm_info.get('time','')}\n")
            if hyde_txt:
                f.write(f"hyde: {hyde_txt}\n")
            f.write(f"q_eff: {q_eff}\n")
            f.write(f"expanded_terms: {', '.join(expand_terms(normalize_tokens(q_eff)))}\n")
        cands_json = []
        for i in chosen_idx:
            cands_json.append({
                "id": chunks[i]["id"],
                "source_path": chunks[i]["source_path"]
            })
        with open(os.path.join(debug_dir, "candidates.json"), "w", encoding="utf-8") as f:
            json.dump(cands_json, f, ensure_ascii=False, indent=2)
        pipe = {
            "mode": args.mode,
            "weights": {"alpha":args.alpha, "beta":args.beta, "gamma":args.gamma, "wimg":args.wimg, "wbool":args.wbool, "wfuzzy":args.wfuzzy},
            "mmr": args.mmr, "neighbors": args.neighbors,
            "strict_mode": args.strict_mode,
            "top": args.top, "num_ctx": args.num_ctx, "ctx_chars": args.ctx_chars,
            "llm_expand": args.llm_expand, "hyde": args.hyde, "prf": args.prf,
            "cross_rerank": args.cross_rerank, "reranker": args.reranker, "rerank_k": args.rerank_k
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
