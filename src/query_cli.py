# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, time, math, re
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm

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
            return json.loads(m.group(0))
        except Exception:
            pass
    # 回退启发式
    toks = re.findall(r'[\u4e00-\u9fa5A-Za-z0-9_]+', question)
    toks = [t for t in toks if t not in ("我的","我们","帮我","请","所有","记录","是什么","说了什么","有哪些","怎么","如何")]
    return {"clean":" ".join(toks[:8]), "keywords": toks[:8], "expanded": []}

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
            # 简单模糊
            if len(k_low) >= 2 and any(token in s_low for token in [k_low[:2], k_low[-2:]]):
                sc += 0.3
    return sc

def _rank_semantic(vecs: np.ndarray, qvec: np.ndarray, topk: int) -> List[Tuple[int, float]]:
    # 安全余弦
    if vecs is None or qvec is None: return []
    if vecs.ndim != 2 or qvec.ndim != 1 or vecs.shape[1] != qvec.shape[0]:
        return []
    nv = np.linalg.norm(vecs, axis=1) + 1e-6
    nq = np.linalg.norm(qvec) + 1e-6
    sims = (vecs @ qvec) / (nv * nq)
    top_idx = np.argpartition(-sims, min(topk, len(sims)-1))[:topk]
    pairs = [(int(i), float(sims[i])) for i in top_idx]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def _unique_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--smart-query", action="store_true")
    ap.add_argument("--sqw", type=float, default=1.2)
    ap.add_argument("--text-first", type=int, default=6)
    ap.add_argument("--wtext", type=float, default=1.0)
    ap.add_argument("--wimg", type=float, default=0.25)
    ap.add_argument("--wbool", type=float, default=0.7)  # 稍降，避免淹没语义
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
            kw = kw[:args.qc_max_keys]
        else:
            kw = []
    else:
        kw = []

    print("[阶段] 召回（RAG）开始 ...")
    t0 = time.time()

    # 加载文本向量索引
    text_vecs, text_ids, meta = load_text_index()
    # 计算查询向量
    qvec = None
    if text_vecs is not None:
        try:
            cli = OllamaClient(host=args.ollama_host, timeout=60)
            qvecs = cli.embeddings(model=args.embed_model, texts=[" ".join(kw) if kw else args.question])
            if qvecs and len(qvecs[0]) == text_vecs.shape[1]:
                qvec = np.asarray(qvecs[0], dtype=np.float32)
            else:
                print("[警告] 查询向量维度与索引不匹配，将跳过语义召回。")
        except Exception as e:
            print(f"[警告] 查询向量计算失败，将跳过语义召回：{e}")

    # 语义召回
    sem_rank = []
    if qvec is not None:
        sem_rank = _rank_semantic(text_vecs, qvec, topk=max(args.top*3, 50))
    _write_json(os.path.join(dbg, "stage1_sem_text.json"), [{"idx": int(i), "score": s} for i,s in sem_rank])

    # 读取 chunks 作为布尔/模糊语料
    chunks = _read_jsonl("chunks.jsonl") or _read_jsonl(os.path.join("data","chunks.jsonl"))
    # 布尔/模糊
    bool_scores = []
    for i, r in enumerate(chunks):
        s = (r.get("title","") + " " + r.get("text",""))[:1500]
        sc = _boolean_score(s, kw if kw else [args.question])
        if sc > 0:
            bool_scores.append((i, sc))
    bool_scores.sort(key=lambda x: x[1], reverse=True)
    _write_json(os.path.join(dbg, "stage1_bool.json"), [{"idx":int(i), "score":float(s)} for i,s in bool_scores[:max(args.top*3, 50)]])

    # 合并打分
    scores = defaultdict(float)
    for i,s in sem_rank:
        scores[i] += float(args.wtext) * s
    for i,s in bool_scores[:max(args.top*3, 200)]:
        scores[i] += float(args.wbool) * s

    # 选 topK & 去重
    cand_idx = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    cand_idx = [i for i,_ in cand_idx[:max(args.top*4, 60)]]

    # 扩展：父级 + 兄弟 + 链接（简化：只按路径层级扩展，避免依赖其它模块）
    paths = [ _norm_path(chunks[i].get("path","")) for i in cand_idx if i < len(chunks)]
    expanded = set(paths)
    for p in list(paths):
        # 父级
        if args.expand_parent:
            parent = _norm_path(os.path.dirname(p))
            if parent: expanded.add(parent)
        # 兄弟
        if args.sibling_span>0:
            try:
                folder = os.path.dirname(p); files = []
                for fn in os.listdir(folder):
                    files.append(_norm_path(os.path.join(folder, fn)))
                files.sort()
                if p in files:
                    j = files.index(p)
                    l = max(0, j-args.sibling_span); r = min(len(files), j+args.sibling_span+1)
                    for sibl in files[l:r]:
                        expanded.add(sibl)
            except Exception:
                pass
    _write_json(os.path.join(dbg, "stage1_expanded_paths.json"), sorted(list(expanded)))

    # 构造候选
    cands = []
    for i in cand_idx:
        if i >= len(chunks): continue
        r = chunks[i]
        cands.append({
            "id": r.get("id"), "path": _norm_path(r.get("path","")), "title": r.get("title",""),
            "chunk_id": r.get("chunk_id",0), "text": r.get("text",""), "score": float(scores.get(i,0.0))
        })
    _write_json(os.path.join(dbg, "candidates.json"), cands)

    print(f"[INFO] 召回候选数={len(cands)}，耗时{time.time()-t0:.2f}s")

    # Step: 并行过滤（此处保留壳，具体实现仍用你现有的 post_filter.py；为了补丁自洽，这里先做轻量保留）
    kept = cands[:max(args.top, 20)]
    _write_json(os.path.join(dbg, "filtered.json"), kept)

    print("[阶段] 最终回答 生成 ...")
    # 生成答案（只要一个“参考”段，严格去重、无 \\? 前缀）
    refs = []
    for r in kept[:args.top]:
        p = _norm_path(r.get("path",""))
        if p and p not in refs:
            refs.append(p)

    answer_lines = []
    answer_lines.append("**关键要点：**")
    if kw:
        answer_lines.append(f"- 解析到的关键词：{', '.join(kw)}")
    answer_lines.append("")
    answer_lines.append("参考：")
    for i,p in enumerate(refs, start=1):
        answer_lines.append(f"[{i}] {p}")

    out_txt = "\n".join(answer_lines)
    with open(os.path.join(dbg, "answer.txt"), "w", encoding="utf-8") as f:
        f.write(out_txt)

    # 控制台输出
    print(out_txt)

if __name__ == "__main__":
    main()
