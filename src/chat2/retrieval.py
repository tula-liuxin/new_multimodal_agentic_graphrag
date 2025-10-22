import os, json, subprocess, shlex, re, sys
from typing import Dict, Any, List

def _write_text(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: f.write(s or "")

ASI_ALIASES = ["ASI","asi","人工超智能","超人工智能","人工通用智能","AGI","agi","通用人工智能"]

def _uniq(seq): 
    seen=set(); out=[]
    for x in seq:
        if x and x not in seen: out.append(x); seen.add(x)
    return out

def expand_keywords(keywords: List[str]) -> List[str]:
    out = list(keywords or [])
    if any(k.lower() == "asi" for k in (keywords or [])):
        out.extend(ASI_ALIASES)
    return _uniq(out)

def run_query_cli(question: str, turn_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    debug_dir = os.path.abspath(turn_dir)
    flags = list(cfg.get("query_cli_flags", []))
    cmd = [sys.executable, "-X", "utf8", "-m", "src.query_cli"] + flags + ["--debug-dir", debug_dir, question]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")

    cwd = os.path.abspath(cfg.get("project_root", "."))

    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
        console_path = os.path.join(turn_dir, "console.log")
        _write_text(console_path, (p.stdout or "") + "\n\n[stderr]\n" + (p.stderr or ""))
        rc = p.returncode
    except Exception as e:
        console_path = os.path.join(turn_dir, "console.log")
        _write_text(console_path, f"[exception] {e!r}")
        rc = -1

    def _exists(rel): 
        path = os.path.join(turn_dir, rel) if not os.path.isabs(rel) else rel
        return path if os.path.exists(path) else None

    return {
        "returncode": rc,
        "python_exec": sys.executable,
        "built_cmd": " ".join(shlex.quote(c) for c in cmd),
        "answer_txt": _exists("answer.txt"),
        "filtered_json": _exists("filtered.json"),
        "candidates_json": _exists("candidates.json"),
        "stage0_keywords": _exists("stage0_keywords.json"),
        "console_log": console_path,
    }

def _score_keywords(text: str, keywords: List[str]) -> float:
    if not text or not keywords: return 0.0
    t=text.lower(); ks=set(k.lower() for k in keywords if k.strip())
    hits=sum(1 for k in ks if k in t)
    return hits/max(1,len(ks))

def _entity_hits(text: str, entities: List[str]) -> int:
    if not text or not entities: return 0
    return sum(1 for e in entities if e and e in text)

def _merge_short_lines(lines, min_chars):
    buf=[]; acc=""
    for ln in lines:
        if len(acc)+len(ln)<min_chars: acc += (ln.strip()+" ")
        else:
            if acc.strip(): buf.append(acc.strip()); acc=""
            if ln.strip(): buf.append(ln.strip())
    if acc.strip(): buf.append(acc.strip())
    return buf

def build_turn_files(turn_dir: str, cfg: Dict[str, Any], entities: List[str], keywords: List[str]):
    import json
    filtered = None
    for name in ["filtered.json","candidates.json","stage1_sem_text.json"]:
        p=os.path.join(turn_dir, name)
        if os.path.exists(p):
            try: filtered=json.load(open(p,"r",encoding="utf-8")); break
            except: pass
    filtered = filtered or []
    include_scores = bool(cfg.get("turn_files", {}).get("include_scores", True))
    min_chars = int(cfg.get("turn_files", {}).get("min_chars", 50))
    max_items = int(cfg.get("turn_files", {}).get("max_items", 60))
    exp_keywords = expand_keywords(keywords or [])

    rows=[]
    for it in filtered:
        if isinstance(it, dict) and not it.get("keep", True): 
            continue
        txt = (it.get("clean_text") or it.get("text") or "").strip()
        if not txt: continue
        base_score = float(it.get("score") or 0.0)
        kw = _score_keywords(txt, exp_keywords)
        eh = _entity_hits(txt, entities)
        dual = 0.3 if (len(entities)>=2 and all(e in txt for e in entities[:2])) else 0.0
        final = base_score*0.5 + kw*0.4 + eh*0.2 + dual
        rows.append({"text":txt, "path":it.get("path",""), "score":final})
    rows.sort(key=lambda r:r["score"], reverse=True)

    lines = _merge_short_lines([r["text"] for r in rows], min_chars=min_chars)
    out=[]
    for i, ln in enumerate(lines[:max_items]):
        base=rows[min(i,len(rows)-1)] if rows else {}
        rec={"path": base.get("path",""), "text": ln}
        if include_scores and rows: rec["score_hint"]=round(base.get("score",0.0),4)
        out.append(rec)

    out_jsonl=os.path.join(turn_dir,"turn_files","rag_clean.jsonl")
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl,"w",encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    md=["# 清洗后的检索片段 (rag_clean)"]
    for idx, r in enumerate(out,1):
        md.append(f"\n## [{idx}] {os.path.basename(r.get('path',''))}\n{r.get('text','').strip()}")
    out_md=os.path.join(turn_dir,"turn_files","rag_clean.md")
    _write_text(out_md,"\n".join(md))

    return (out, out_md)

def search_history(session_dir: str, keywords: List[str], cfg: Dict[str, Any]):
    return {"turn_files":[]}
