import argparse, os, json, re, math, time
import numpy as np
import joblib
from typing import List, Dict, Tuple
from .utils import load_cfg, get_logger, ensure_dir, norm_win_abs
from .ollama_client import OllamaClient
from .multimodal_fusion import fuse_scores, mmr_select, read_jsonl
import open_clip, torch
from PIL import Image

def cosine_rows(A, b):
    # A: (N,D), b: (D,)
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    nb = np.linalg.norm(b) + 1e-9
    na = np.linalg.norm(A, axis=1) + 1e-9
    sim = (A @ b) / (na * nb)
    return sim

def smart_keywords(q: str, weight: float=1.2) -> List[str]:
    # 低成本解析：中文分块 + 英文词
    toks = re.findall(r"[一-龥A-Za-z0-9_]+", q)
    # 简单去停用
    stop = {"的","了","和","与","及","在","有","是","我","我们","最近","计划"}
    kws = [t for t in toks if t not in stop and len(t) >= 2]
    return kws

def load_dense(data_dir):
    emb = np.memmap(os.path.join(data_dir, "embeddings.npy"), dtype=np.float32, mode="r")
    ids = json.load(open(os.path.join(data_dir, "ids.json"), "r", encoding="utf-8"))
    return emb, ids

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
    # 使用同一 CLIP 文本编码器
    ids_path = os.path.join(data_dir, "image_ids.json")
    emb_path = os.path.join(data_dir, "image_embeddings.npy")
    if not (os.path.exists(ids_path) and os.path.exists(emb_path)):
        return None, None, None
    img_ids = json.load(open(ids_path, "r", encoding="utf-8"))
    img_vecs = np.memmap(emb_path, dtype=np.float32, mode="r")
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

def build_context(chosen_chunks: List[Dict], images_used: List[Dict], cfg, num_ctx: int, ctx_chars: int):
    # 拼接文本上下文（限制字符数），图像信息作为附加注释
    ctx_parts = []
    total = 0
    for rec in chosen_chunks:
        t = rec["text"]
        if total + len(t) > num_ctx:
            t = t[:max(0, num_ctx-total)]
        ctx_parts.append(t)
        total += len(t)
        if total >= num_ctx:
            break
    # 附带图片 OCR/字幕简述
    for im in images_used:
        extras = []
        if im.get("caption"):
            extras.append(f"字幕：{im['caption']}")
        if im.get("ocr_text"):
            extras.append(f"OCR：{im['ocr_text'][:200]}")
        if extras:
            ctx_parts.append("【相关图片】" + "；".join(extras))
    return "\n\n".join(ctx_parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["tfidf","dense","hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.5, help="词面 TF-IDF 权重")
    parser.add_argument("--beta", type=float, default=0.3, help="字 n-gram 权重")
    parser.add_argument("--gamma", type=float, default=0.8, help="文本向量权重")
    parser.add_argument("--wimg", type=float, default=0.25, help="图像通道权重")
    parser.add_argument("--images-only", action="store_true")

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

    # 载入资源
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

    # Query 扩展与 strict 规则
    q_eff = q
    if args.smart_query:
        kws = smart_keywords(q, weight=args.sqw)
        if kws:
            q_eff = q + " " + " ".join(kws)

    # 计算分数
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
        # 获取 query 向量（用文本嵌入模型）
        from .ollama_client import OllamaClient
        client = OllamaClient(cfg["ollama_base_url"], timeout=int(cfg.get("embed_timeout",120)), keep_alive=cfg.get("embed_keep_alive","10m"))
        qemb = client.embeddings(cfg["embed_model"], [q_eff])[0]
        qemb = np.array(qemb, dtype=np.float32)
        scores_dense = cosine_rows(dense, qemb)

    img_info = None
    if args.mode == "hybrid" and not args.images_only and args.wimg > 0:
        s_img, img_ids, img_recs = image_query_to_scores(cfg, q_eff, data_dir, images, args.wimg)
        if s_img is not None:
            # 将图片分数映射到与文本 chunks 同一空间：通过 source_path 关联
            img_map = {}
            for sim, rec in zip(s_img, img_recs):
                img_map[rec["source_path"]] = float(sim)
            scores_img = np.zeros(N, dtype=np.float32)
            for i, rec in enumerate(chunks):
                sp = rec["source_path"]
                # 同目录/同名关联（启发式）：若文本来自与图片相邻文件，给一点加成
                base_dir = os.path.dirname(sp)
                if sp in img_map:
                    scores_img[i] = img_map[sp]
                else:
                    # 简单邻近：同目录中的图片最大分
                    best = 0.0
                    for k, v in img_map.items():
                        if os.path.dirname(k) == base_dir:
                            best = max(best, v*0.8)
                    scores_img[i] = best
            img_info = (s_img, img_ids, img_recs)

    if args.images_only:
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

    # 融合
    scores = fuse_scores(
        {"word":scores_word, "char":scores_char, "dense":scores_dense, "img":scores_img},
        {"word":args.alpha, "char":args.beta, "dense":args.gamma, "img":args.wimg}
    )
    if scores is None:
        print("没有可融合的分数，请确认已构建对应索引。")
        return

    order = np.argsort(-scores)
    # strict 过滤
    if args.strict_mode == "exact":
        mask = np.array([ (re.search(re.escape(q), chunks[i]["text"]) is not None) for i in range(len(chunks)) ])
        order = [i for i in order if mask[i]]
    elif args.strict_mode == "smart":
        kws = smart_keywords(q)
        mask = np.array([ all(k in chunks[i]["text"] for k in kws) for i in range(len(chunks)) ]) if kws else np.ones(len(chunks), dtype=bool)
        order = [i for i in order if mask[i]]

    # MMR 去冗
    idx_top = order[:min(200, len(order))]
    # 简单相似度矩阵（用 word+char 稀疏近似）
    # 为降低开销，这里只用稠密融合后的标量，构造近似“相似度”
    sim_mat = np.zeros((len(chunks), len(chunks)), dtype=np.float32)
    cands = [(i, float(scores[i])) for i in idx_top]
    selected = [i for (i, _) in cands[:args.top]]
    # NOTE: 为简化，暂不计算片段间相似度矩阵；可在将来替换为余弦相似度

    chosen_idx = selected[:args.top]
    chosen_chunks = [chunks[i] for i in chosen_idx]

    # 找到涉及到的图片（按同目录关联 & 最高分）
    images_used = []
    if img_info is not None:
        s_img, img_ids, img_recs = img_info
        # 取前若干张
        img_order = np.argsort(-s_img)[:args.top]
        images_used = [img_recs[i] for i in img_order if s_img[i] > 0]

    # 构造上下文
    ctx = build_context(chosen_chunks, images_used, cfg, args.num_ctx, args.ctx_chars)

    # 生成回答（可选）
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
        # 直接输出上下文片段
        answer = ctx

    # Debug 落盘
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "query.txt"), "w", encoding="utf-8") as f:
            f.write(f"question: {q}\n")
            if args.smart_query:
                f.write(f"smart_keywords: {' '.join(smart_keywords(q))}\n")
            f.write(f"q_eff: {q_eff}\n")
        # 候选
        cands_json = []
        for i in chosen_idx:
            cands_json.append({
                "id": chunks[i]["id"],
                "source_path": chunks[i]["source_path"],
                "score": float(scores[i])
            })
        with open(os.path.join(debug_dir, "candidates.json"), "w", encoding="utf-8") as f:
            json.dump(cands_json, f, ensure_ascii=False, indent=2)
        # pipeline
        pipe = {
            "mode": args.mode,
            "weights": {"alpha":args.alpha, "beta":args.beta, "gamma":args.gamma, "wimg":args.wimg},
            "mmr": args.mmr, "neighbors": args.neighbors,
            "strict_mode": args.strict_mode,
            "top": args.top, "num_ctx": args.num_ctx, "ctx_chars": args.ctx_chars
        }
        with open(os.path.join(debug_dir, "pipeline.json"), "w", encoding="utf-8") as f:
            json.dump(pipe, f, ensure_ascii=False, indent=2)
        # contexts
        with open(os.path.join(debug_dir, "contexts.txt"), "w", encoding="utf-8") as f:
            f.write(ctx)
        # images
        if images_used:
            with open(os.path.join(debug_dir, "images.json"), "w", encoding="utf-8") as f:
                json.dump(images_used, f, ensure_ascii=False, indent=2)

    # 输出答案 + 参考
    print(answer.strip())
    # 参考只出现一次：列出实际用到的文本/图片绝对 Windows 路径
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
