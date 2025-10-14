# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, time, math, re
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    from rapidfuzz.fuzz import partial_ratio
except Exception:
    partial_ratio = None

from .ollama_client import OllamaClient
from .embed_index import load_text_index

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _write_json(path: str, data):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _read_jsonl(path: str) -> List[Dict]:
    out = []
    if not os.path.exists(path): return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _norm_path(p: str) -> str:
    if not p: return p
    p = p.replace("\\\\?\\", "").replace("\\?", "")
    return os.path.abspath(p)

def _keywords_llm_clean(question: str, qc_model: str, qc_prompt_file: str, host: Optional[str], timeout: int, logger=None) -> Dict:
    prompt = open(qc_prompt_file, "r", encoding="utf-8").read() if os.path.exists(qc_prompt_file) else "请从用户问题中抽取关键词并输出JSON。"
    payload = f"{prompt}\n\n用户问题：{question}\n只输出JSON："
    cli = OllamaClient(host=host, timeout=timeout, retries=1, logger=logger)
    jr = cli.generate(model=qc_model, prompt=payload, stream=False, options={"num_predict": 256, "temperature": 0.1})
    txt = jr.get("response","").strip()
    # 容错提取 JSON
    m = re.search(r'\{.*\}', txt, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            # 关键词去重
            kws = []
            for k in obj.get("keywords", []):
                if k and k not in kws: kws.append(k)
            obj["keywords"] = kws
            exp = []
            for k in obj.get("expanded", []):
                if k and k not in kws and k not in exp: exp.append(k)
            obj["expanded"] = exp
            return obj
        except Exception:
            pass
    # 回退启发式
    toks = re.findall(r'[\u4e00-\u9fa5A-Za-z0-9_]+', question)
    toks = [t for t in toks if t not in ("我的","我们","帮我","请","所有","记录","是什么","说了什么","有哪些","怎么","如何")]
    # 去重
    dedup = []
    for t in toks:
        if t not in dedup: dedup.append(t)
    return {"clean":" ".join(dedup[:8]), "keywords": dedup[:8], "expanded": []}

def _boolean_score(s: str, kw: List[str]) -> float:
    if not s: return 0.0
    s_low = s.lower()
    sc = 0.0
    for k in kw:
        if not k: continue
        k_low = k.lower()
        if k_low in s_low:
            sc += 1.0
        else:
            # 简单模糊 + optional rapidfuzz
            bonus = 0.0
            if len(k_low) >= 2 and (k_low[:2] in s_low or k_low[-2:] in s_low):
                bonus = 0.3
            if partial_ratio is not None:
                try:
                    pr = partial_ratio(k_low, s_low)  # 0~100
                    bonus = max(bonus, pr/200.0)      # up to +0.5
                except Exception:
                    pass
            sc += bonus
    return sc

def _rank_semantic(vecs: np.ndarray, qvec: np.ndarray, topk: int) -> List[Tuple[int, float]]:
    # 安全余弦
    if vecs is None or qvec is None: return []
    if vecs.ndim != 2 or qvec.ndim != 1 or vecs.shape[1] != qvec.shape[0]:
        return []
    nv = np.linalg.norm(vecs, axis=1) + 1e-6
    nq = np.linalg.norm(qvec) + 1e-6
    sims = (vecs @ qvec) / (nv * nq)
    topk = min(topk, len(sims))
    if topk <= 0: return []
    top_idx = np.argpartition(-sims, topk-1)[:topk]
    pairs = [(int(i), float(sims[i])) for i in top_idx]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def _load_links(path: str) -> Dict[str, List[str]]:
    """读取 links.jsonl，返回无向邻接表"""
    g = defaultdict(list)
    rows = _read_jsonl(path)
    for r in rows:
        a = _norm_path(r.get("src","")); b = _norm_path(r.get("dst",""))
        if not a or not b: continue
        g[a].append(b); g[b].append(a)
    return g

def _expand_paths(base_paths: List[str], link_graph: Dict[str, List[str]], link_depth: int,
                  expand_parent: bool, sibling_span: int) -> List[str]:
    expanded = set(base_paths)
    # link hops
    frontier = list(base_paths)
    for _ in range(max(0, link_depth)):
        newf = []
        for p in frontier:
            for nb in link_graph.get(p, []):
                if nb not in expanded:
                    expanded.add(nb); newf.append(nb)
        frontier = newf
    # 父级 + 兄弟
    for p in list(expanded):
        if expand_parent:
            parent = _norm_path(os.path.dirname(p))
            if parent: expanded.add(parent)
        if sibling_span>0:
            try:
                folder = os.path.dirname(p)
                files = []
                for fn in os.listdir(folder):
                    files.append(_norm_path(os.path.join(folder, fn)))
                files.sort()
                if p in files:
                    j = files.index(p)
                    l = max(0, j-sibling_span); r = min(len(files), j+sibling_span+1)
                    for sibl in files[l:r]:
                        expanded.add(sibl)
            except Exception:
                pass
    return sorted(expanded)

def _build_answer_with_refs(chunks: List[Dict], kept_idx: List[int], top_n: int) -> Tuple[str, List[str]]:
    refs = []
    used = set()
    lines = []
    for i in kept_idx[:top_n]:
        if i<0 or i>=len(chunks): continue
        p = _norm_path(chunks[i].get("path",""))
        if not p or p in used: continue
        used.add(p); refs.append(p)
    # 仅构造引用段；真正答案交给 LLM
    ref_block = "\n".join([f"[{i}] {p}" for i,p in enumerate(refs, start=1)])
    return ref_block, refs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--smart-query", action="store_true")
    ap.add_argument("--sqw", type=float, default=1.2)
    ap.add_argument("--text-first", type=int, default=6)
    ap.add_argument("--wtext", type=float, default=1.0)
    ap.add_argument("--wimg", type=float, default=0.25)
    ap.add_argument("--wbool", type=float, default=0.7)
    ap.add_argument("--wfuzzy", type=float, default=0.6)
    ap.add_argument("--neighbors", type=int, default=1)
    ap.add_argument("--folder-neighbors", type=int, default=1)
    ap.add_argument("--link-hop", action="store_true")
    ap.add_argument("--link-depth", type=int, default=1)
    ap.add_argument("--expand-parent", action="store_true")
    ap.add_argument("--sibling-span", type=int, default=1)
    ap.add_argument("--post-filter", action="store_true")
    ap.add_argument("--pf-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--pf-workers", type=int, default=4)
    ap.add_argument("--pf-timeout", type=int, default=15)
    ap.add_argument("--pf-min-keep", type=int, default=6)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--num-ctx", type=int, default=20000)
    ap.add_argument("--ctx-chars", type=int, default=1200)
    ap.add_argument("--embed-model", default="bge-m3:latest")
    ap.add_argument("--gen-timeout", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--vl-answer", action="store_true")
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--llm-num-ctx", type=int, default=8192)
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--debug-dir", default="debug/session")
    ap.add_argument("--model", default="qwen3:4b-instruct-2507-fp16")
    # 关键词清洗
    ap.add_argument("--llm-query-clean", action="store_true")
    ap.add_argument("--qc-model", default="qwen3:4b-instruct-2507-fp16")
    ap.add_argument("--qc-timeout", type=int, default=60)
    ap.add_argument("--qc-max-keys", type=int, default=8)
    ap.add_argument("--qc-prompt-file", default="prompts/keyword_extractor_zh.txt")

    ap.add_argument("question")
    args = ap.parse_args()

    dbg = args.debug_dir
    _ensure_dir(dbg)

    # Step 0: 关键词清洗
    if args.llm_query_clean:
        try:
            qc = _keywords_llm_clean(args.question, args.qc_model, args.qc_prompt_file, args.ollama_host, args.qc_timeout)
        except Exception as e:
            qc = {"error": str(e)}
        _write_json(os.path.join(dbg, "stage0_keywords.json"), {"question": args.question, "qc": qc})
        if isinstance(qc, dict) and "keywords" in qc:
            kw = [x for x in (qc.get("keywords", []) + qc.get("expanded", [])) if x]
            # 去重
            tmp=[]; [tmp.append(x) for x in kw if x not in tmp]
            kw = tmp[:args.qc_max_keys]
        else:
            kw = []
    else:
        kw = []

    print("[阶段] 召回（RAG）开始 ...")
    t0 = time.time()

    # 读取 chunks
    chunks = _read_jsonl("chunks.jsonl") or _read_jsonl(os.path.join("data","chunks.jsonl"))
    # 加载文本向量索引
    text_vecs, text_ids, meta = load_text_index()

    # 计算查询向量（仅当语义索引可用且维度匹配）
    qvec = None
    if text_vecs is not None:
        try:
            cli = OllamaClient(host=args.ollama_host, timeout=90)
            qtext = " ".join(kw) if kw else args.question
            qvecs = cli.embeddings(model=args.embed_model, texts=[qtext])
            if qvecs and len(qvecs[0]) == text_vecs.shape[1]:
                qvec = np.asarray(qvecs[0], dtype=np.float32)
            else:
                print("[警告] 查询向量维度与索引不匹配，将跳过语义召回。")
        except Exception as e:
            print(f"[警告] 查询向量计算失败，将跳过语义召回：{e}")

    # 语义召回（按 top*3，上限为全部）
    sem_rank = []
    if qvec is not None:
        want = max(args.top*3, min(len(chunks), 5000))
        sem_rank = _rank_semantic(text_vecs, qvec, topk=want)
    _write_json(os.path.join(dbg, "stage1_sem_text.json"), [{"idx": int(i), "score": s} for i,s in sem_rank])

    # 布尔/模糊（覆盖全部文档，支持 very large top）
    bool_scores = []
    if chunks:
        for i, r in enumerate(chunks):
            s = (r.get("title","") + " " + r.get("text",""))[:2000]
            sc = _boolean_score(s, kw if kw else [args.question])
            if sc > 0:
                bool_scores.append((i, sc))
    bool_scores.sort(key=lambda x: x[1], reverse=True)
    # 不再截断到固定 200
    _write_json(os.path.join(dbg, "stage1_bool.json"), [{"idx":int(i), "score":float(s)} for i,s in bool_scores])

    # 合并打分
    scores = defaultdict(float)
    for i,s in sem_rank:
        scores[i] += float(args.wtext) * s
    for i,s in bool_scores:
        scores[i] += float(args.wbool) * s

    # 选 topK（按要求 honor 超大 top，但保护性上限为文档总数）
    cand_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    limit = min(len(cand_sorted), max(args.top*4, args.top))  # 放宽到 4x top
    cand_idx = [i for i,_ in cand_sorted[:limit]]

    # 链接图扩展（如果存在）+ 父级/兄弟
    link_graph = _load_links("links.jsonl") if os.path.exists("links.jsonl") else _load_links(os.path.join("data","links.jsonl"))
    base_paths = [_norm_path(chunks[i].get("path","")) for i in cand_idx if i < len(chunks)]
    if args.link_depth>0 and link_graph:
        expanded_paths = _expand_paths(base_paths, link_graph, args.link_depth if args.link_hop else 0, args.expand_parent, args.sibling_span)
    else:
        expanded_paths = _expand_paths(base_paths, defaultdict(list), 0, args.expand_parent, args.sibling_span)
    _write_json(os.path.join(dbg, "stage1_expanded_paths.json"), expanded_paths)

    # 构造候选（只保留真实存在的文件路径）
    cand_set = set(base_paths) | set(expanded_paths)
    path2first = {}
    cands = []
    for i in cand_idx:
        if i>=len(chunks): continue
        p = _norm_path(chunks[i].get("path",""))
        if not p: continue
        if p not in cand_set: continue
        if p in path2first: continue
        path2first[p] = i
        r = chunks[i]
        cands.append({
            "id": r.get("id"), "path": p, "title": r.get("title",""),
            "chunk_id": r.get("chunk_id",0), "text": r.get("text",""),
            "score": float(scores.get(i,0.0))
        })
    _write_json(os.path.join(dbg, "candidates.json"), cands)

    print(f"[INFO] 召回候选数={len(cands)}，耗时{time.time()-t0:.2f}s")

    # Step: 并行过滤（尽量沿用你现有的 post_filter；没有就直接保留前K）
    kept = cands
    try:
        if args.post_filter:
            from .post_filter import parallel_filter  # 你的实现
            kept = parallel_filter(cands, model=args.pf_model, workers=args.pf_workers,
                                   timeout=args.pf_timeout, min_keep=args.pf_min_keep,
                                   ollama_host=args.ollama_host, debug_dir=dbg)
    except Exception as e:
        _write_json(os.path.join(dbg, "post_filter_error.json"), {"error": str(e)})
        # 回退：保留前K
        pass
    _write_json(os.path.join(dbg, "filtered.json"), kept)

    # 最终回答（使用 LLM 生成；失败时优雅降级）
    print("[阶段] 最终回答 生成 ...")
    # 先构造引用与上下文
    # 将 kept 的文本截断拼接，控制在 num_ctx / ctx_chars 限制内
    ctx_chunks = []
    total_chars = 0
    for r in kept:
        t = (r.get("title","") + "\n" + r.get("text",""))[:max(128, args.ctx_chars)]
        ctx_chunks.append(t)
        total_chars += len(t)
        if total_chars >= args.num_ctx:
            break

    ref_block, refs = _build_answer_with_refs(kept, list(range(len(kept))), top_n=len(kept))

    sys_prompt = (
        "你是严谨的企业级助手。请基于给定上下文回答用户问题，先列出要点再归纳，"
        "末尾必须只出现一次“参考：”段，按[1]开始的编号，路径必须为可直接打开的 Windows 绝对路径，且不得包含 \\\\?\\ 或 \\?。"
    )
    user_question = args.question
    context_text = "\n\n---\n\n".join(ctx_chunks)

    prompt = f"""[系统规则]\n{sys_prompt}\n\n[用户问题]\n{user_question}\n\n[已检索上下文（节选）]\n{context_text}\n\n[引用清单]\n{ref_block}\n\n请根据上下文作答；不要杜撰不存在的信息。"""

    answer_txt = None
    try:
        cli = OllamaClient(host=args.ollama_host, timeout=args.gen_timeout, retries=1)
        jr = cli.generate(model=args.model, prompt=prompt, stream=False,
                          options={"temperature": args.temperature, "num_ctx": args.llm_num_ctx, "num_predict": 512})
        answer_txt = jr.get("response","").strip()
    except Exception as e:
        _write_json(os.path.join(dbg, "gen_error.json"), {"error": str(e)})

    if not answer_txt:
        print("\n（提示）生成模型不可用，本次仅返回命中的参考材料。\n")
        print("参考：")
        for i,p in enumerate(refs, start=1):
            print(f"[{i}] {p}")
        # 也写入文件
        with open(os.path.join(dbg, "answer.txt"), "w", encoding="utf-8") as f:
            f.write("（提示）生成模型不可用，本次仅返回命中的参考材料。\n\n参考：\n" + "\n".join([f"[{i}] {p}" for i,p in enumerate(refs, start=1)]))
        return

    # 规范化"参考"段：如果模型生成了多个或不合规，我们强制覆写/附加一个正确的参考段
    norm = answer_txt.strip()
    # 去掉重复“参考”段标题（仅保留模型正文），然后附上我们的标准参考
    norm = re.sub(r"\n+参考[:：].*", "", norm, flags=re.S)
    final = norm + "\n\n参考：\n" + "\n".join([f"[{i}] {p}" for i,p in enumerate(refs, start=1)])

    with open(os.path.join(dbg, "answer.txt"), "w", encoding="utf-8") as f:
        f.write(final)

    print(final)

if __name__ == "__main__":
    main()
