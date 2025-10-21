import os, json, subprocess, shlex, re, sys
from typing import Dict, Any, List
from .utils import ensure_dir, read_jsonl, write_text, safe_display_path, score_keywords, merge_short_lines, uniq_preserve

ASI_ALIASES = ["ASI","asi","人工超智能","超人工智能","人工通用智能","AGI","agi","通用人工智能"]

def expand_keywords(keywords: List[str]) -> List[str]:
    out = list(keywords or [])
    if any(k.lower() == "asi" for k in keywords or []):
        out.extend(ASI_ALIASES)
    return uniq_preserve(out)

def run_query_cli(question: str, turn_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    debug_dir = os.path.abspath(turn_dir)
    flags = list(cfg.get("query_cli_flags", []))
    cmd = [sys.executable, "-X", "utf8", "-m", "src.query_cli"] + flags + ["--debug-dir", debug_dir, question]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    cwd = os.path.abspath(cfg.get("project_root", "."))
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=None, cwd=cwd)
        console_path = os.path.join(turn_dir, "console.log")
        write_text(console_path, (p.stdout or "") + "\n\n[stderr]\n" + (p.stderr or ""))
        rc = p.returncode
    except Exception as e:
        console_path = os.path.join(turn_dir, "console.log")
        write_text(console_path, f"[exception] {e!r}")
        rc = -1

    out = {
        "returncode": rc,
        "python_exec": sys.executable,
        "built_cmd": " ".join(shlex.quote(c) for c in cmd),
        "answer_txt": os.path.join(turn_dir, "answer.txt"),
        "filtered_json": os.path.join(turn_dir, "filtered.json"),
        "candidates_json": os.path.join(turn_dir, "candidates.json"),
        "stage0_keywords": os.path.join(turn_dir, "stage0_keywords.json"),
        "console_log": console_path,
    }
    for k in list(out.keys()):
        if k in {"returncode","built_cmd","python_exec"}: continue
        if not os.path.exists(out[k]):
            out[k] = None

    if not out.get("answer_txt") or (out.get("answer_txt") and not os.path.exists(os.path.join(turn_dir, "answer.txt"))):
        try:
            ctext = open(console_path, "r", encoding="utf-8").read()
            rec = extract_answer_from_console(ctext)
            if rec.strip():
                ap = os.path.join(turn_dir, "answer.txt")
                write_text(ap, rec)
                out["answer_txt"] = ap
        except Exception:
            pass
    return out

def extract_answer_from_console(console_text: str) -> str:
    if not console_text:
        return ""
    m = re.search(r"(要点：[\s\S]+)$", console_text)
    if m:
        return m.group(1).strip()[-4000:]
    m2 = re.search(r"(参考：[\s\S]+)$", console_text)
    if m2:
        return m2.group(1).strip()[-4000:]
    return console_text.strip()[-2000:]

def _load_json(path: str):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None

def _contains_all_entities(text: str, entities: List[str]) -> bool:
    t = text or ""
    return all((e and e in t) for e in entities)

def _entity_hit_count(text: str, entities: List[str]) -> int:
    t = text or ""
    return sum(1 for e in entities if e and e in t)

def _extract_clean_snippets(filtered_items: List[Dict[str, Any]], max_items: int, min_chars: int, include_scores: bool, entities: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for it in filtered_items or []:
        if isinstance(it, dict) and not it.get("keep", True):
            continue
        txt = (isinstance(it, dict) and (it.get("clean_text") or it.get("text"))) or ""
        txt = (txt or "").strip()
        if not txt:
            continue
        src = safe_display_path((it.get("path") if isinstance(it, dict) else "") or "")
        reason = (it.get("reason") if isinstance(it, dict) else "") or ""
        base_score = float((it.get("score") if isinstance(it, dict) else 0.0) or 0.0)
        kw_score = score_keywords(txt, keywords)
        ent_hits = _entity_hit_count(txt, entities)
        dual_bonus = 0.25 if (len(entities) >= 2 and _contains_all_entities(txt, entities[:2])) else 0.0
        final_score = base_score*0.5 + kw_score*0.4 + ent_hits*0.2 + dual_bonus
        rec = {"path": src, "text": txt, "reason": reason, "score": final_score}
        rows.append(rec)
    rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    lines = merge_short_lines([r["text"] for r in rows], min_chars=min_chars)
    out = []
    for i, ln in enumerate(lines[:max_items]):
        base = rows[min(i, len(rows)-1)] if rows else {"path": ""}
        rec = {"path": base.get("path",""), "text": ln}
        if include_scores and rows:
            rec["score_hint"] = round(base.get("score", 0.0), 4)
        out.append(rec)
    return out

def build_turn_files(turn_dir: str, cfg: Dict[str, Any], entities: List[str], keywords: List[str]):
    fpath = os.path.join(turn_dir, "filtered.json")
    if os.path.exists(fpath):
        try:
            filtered = json.load(open(fpath, "r", encoding="utf-8"))
        except Exception:
            filtered = []
    else:
        filtered = _load_json(os.path.join(turn_dir, "candidates.json")) or _load_json(os.path.join(turn_dir, "stage1_sem_text.json")) or []
    include_scores = bool(cfg.get("turn_files", {}).get("include_scores", True))
    min_chars = int(cfg.get("turn_files", {}).get("min_chars", 50))
    max_items = int(cfg.get("turn_files", {}).get("max_items", 60))
    exp_keywords = expand_keywords(keywords or [])
    rows = _extract_clean_snippets(filtered, max_items=max_items, min_chars=min_chars, include_scores=include_scores, entities=entities or [], keywords=exp_keywords)

    out_jsonl = os.path.join(turn_dir, "turn_files", "rag_clean.jsonl")
    ensure_dir(os.path.dirname(out_jsonl))
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = ["# 清洗后的检索片段 (rag_clean)"]
    for idx, r in enumerate(rows, 1):
        md.append(f"\n## [{idx}] {os.path.basename(r.get('path',''))}\n{r.get('text','').strip()}")
    out_md = os.path.join(turn_dir, "turn_files", "rag_clean.md")
    write_text(out_md, "\n".join(md))
    return (rows, out_md)

def search_history(session_dir: str, keywords: List[str], cfg: Dict[str, Any]):
    exp_keywords = expand_keywords(keywords or [])
    hits = []
    max_hits = int(cfg.get("history_search", {}).get("max_hits", 12))
    th = float(cfg.get("history_search", {}).get("score_threshold", 0.2))
    for tdir in sorted([d for d in os.listdir(session_dir) if d.startswith("turn-")]):
        tf_jsonl = os.path.join(session_dir, tdir, "turn_files", "rag_clean.jsonl")
        if not os.path.exists(tf_jsonl):
            continue
        for row in read_jsonl(tf_jsonl):
            s = score_keywords(row.get("text",""), exp_keywords)
            if s >= th:
                hits.append({"where": f"{tdir}/turn_files", "score": s, "text": row.get("text","")})
                if len(hits) >= max_hits:
                    return {"turn_files": hits}
    for tdir in sorted([d for d in os.listdir(session_dir) if d.startswith("turn-")]):
        ts = os.path.join(session_dir, tdir, "turn_summary.txt")
        if not os.path.exists(ts):
            continue
        txt = open(ts, "r", encoding="utf-8").read()
        s = score_keywords(txt, exp_keywords)
        if s >= th:
            hits.append({"where": f"{tdir}/turn_summary", "score": s, "text": txt})
            if len(hits) >= max_hits:
                return {"turn_files": hits}
    cs = os.path.join(session_dir, "summary.txt")
    if os.path.exists(cs):
        txt = open(cs, "r", encoding="utf-8").read()
        s = score_keywords(txt, exp_keywords)
        if s >= th:
            hits.append({"where": f"chat_summary", "score": s, "text": txt})
    return {"turn_files": hits}
