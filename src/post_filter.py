# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, base64, re, concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
from .utils import load_cfg, get_logger, ensure_dir

def _b64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""

def _ollama_generate(base_url: str, model: str, prompt: str, images: List[str]=None, timeout:int=60) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if images:
        payload["images"] = images
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response","")

def _clean(p: str) -> str:
    return p.replace("\\?\\", "").replace("/", "\\")

PF_SYS = "你是一个过滤器。判断下面候选内容是否与提问相关联（哪怕部分相关也算）。只回复一个词：YES 或 NO。"

def _pf_prompt(q: str, text: str, path: str) -> str:
    return f"{PF_SYS}\n\n提问：{q}\n候选（截断展示）：{text[:800]}\n来源：{path}\n\n只回复 YES 或 NO。"

def _judge_one(args) -> Tuple[bool, Dict[str, Any]]:
    base_url, model, timeout, question, cand = args
    text = cand.get("text","")
    path = _clean(cand.get("path",""))
    prompt = _pf_prompt(question, text, path)
    try:
        resp = _ollama_generate(base_url, model, prompt, images=None, timeout=timeout).strip().upper()
    except Exception:
        resp = "NO"
    ok = ("YES" in resp) and ("NO" not in resp[:5])
    return ok, {"path": path, "text": text, "id": cand.get("id") or cand.get("rid")}

def _load_candidates(debug_dir: Path, data_dir: Path, question: str, top:int=50) -> List[Dict[str,Any]]:
    cj = debug_dir / "candidates.json"
    if cj.exists():
        try:
            return json.loads(cj.read_text("utf-8"))
        except Exception:
            pass
    chunks = data_dir / "chunks.jsonl"
    if not chunks.exists():
        return []
    kws = [w for w in re.split(r"[\s、，,。；;:?？!！]+", question) if w]
    scored = []
    with chunks.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text","")
            score = sum(text.count(k) for k in kws)
            if score>0:
                scored.append((score, {"id": obj.get("id"), "path": obj.get("path"), "text": text}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _,b in scored[:top]]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--pf-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--pf-workers", type=int, default=6)
    ap.add_argument("--pf-timeout", type=int, default=15)
    ap.add_argument("--pf-min-keep", type=int, default=6)
    ap.add_argument("--final-model", default="qwen3:4b-instruct-2507-fp16")
    ap.add_argument("--vl-final", action="store_true")
    ap.add_argument("--max-final-images", type=int, default=4)
    ap.add_argument("--ctx-chars", type=int, default=1200)
    ap.add_argument("--num-ctx", type=int, default=16000)
    args = ap.parse_args()

    cfg = load_cfg()
    logger = get_logger()
    base_url = cfg.get("ollama_base_url", "http://127.0.0.1:11434")
    debug_dir = Path(args.debug_dir); ensure_dir(debug_dir)
    data_dir = Path(cfg["data_dir"])

    cands = _load_candidates(debug_dir, data_dir, args.question, top=60)
    if not cands:
        print("（提示）未发现候选。")
        return

    tasks = [(base_url, args.pf_model, args.pf_timeout, args.question, c) for c in cands]
    kept: List[Dict[str,Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.pf_workers) as ex:
        for ok, item in ex.map(_judge_one, tasks):
            if ok:
                kept.append(item)
    if len(kept) < args.pf_min_keep:
        kept = cands[:args.pf_min_keep]

    (debug_dir / "filtered_candidates.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), "utf-8")

    ctx = []
    refs = []
    total = 0
    for i, c in enumerate(kept, start=1):
        t = c.get("text","")[:args.ctx_chars]
        ctx.append(f"[{i}] {t}")
        refs.append(f"[{i}] {_clean(c.get('path',''))}")
        total += len(t)
        if total > args.num_ctx:
            break
    (debug_dir / "contexts.txt").write_text("\n\n".join(ctx), "utf-8")

    sys_prompt = "你是检索增强问答助手。使用给定的参考片段回答，条理清晰，并在答案末尾附上编号化“参考”。"
    user_prompt = f"问题：{args.question}\n\n参考片段：\n" + "\n\n".join(ctx) + "\n\n请基于参考作答。"

    images_b64: List[str] = []
    if args.vl_final and args.max_final_images>0:
        for c in kept:
            p = str(c.get("path",""))
            if p.lower().endswith(('.png','.jpg','.jpeg','.webp')):
                b = _b64_image(p)
                if b:
                    images_b64.append(b)
                if len(images_b64) >= args.max_final_images:
                    break

    try:
        resp = _ollama_generate(base_url, args.final_model, f"{sys_prompt}\n\n{user_prompt}", images=images_b64 if images_b64 else None, timeout=120)
    except Exception:
        resp = "（提示）生成模型不可用。仅返回参考。"

    final_md = resp.strip() + "\n\n参考：\n" + "\n".join(refs)
    (debug_dir / "final_answer.md").write_text(final_md, "utf-8")
    print(final_md)

if __name__ == "__main__":
    main()
