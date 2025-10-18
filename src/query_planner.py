import json, re
from pathlib import Path
from typing import Optional, Dict, Any, List
from .ollama_client import OllamaClient

def _safe_json_line(s: str) -> Optional[dict]:
    s = s.strip()
    # 取第一行的 JSON 或全量 JSON
    if "\n" in s:
        first = s.splitlines()[0].strip()
        if first.startswith("{") and first.endswith("}"):
            try:
                return json.loads(first)
            except Exception:
                pass
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def _dedup_list(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        x = x.strip()
        if not x or x in seen: 
            continue
        seen.add(x)
        out.append(x)
    return out

def _make_planned_query(plan: dict, cleaned_query: str) -> str:
    # 优先以 rag_a.search_terms 合成
    terms = plan.get("rag_a",{}).get("search_terms") or []
    if not terms:
        terms = [cleaned_query]
    # 清除重复“与 我 关系”
    def fix(s: str) -> str:
        s = re.sub(r"(\s*与\s*我\s*关系)+$", " 与 我 关系", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    terms = [fix(t) for t in terms]
    terms = _dedup_list(terms)
    return " ".join(terms)

def plan_query(user_q: str, cleaned_query: str,
               ollama_host: str,
               model: str,
               prompt_file: Optional[str]=None) -> Dict[str, Any]:
    prompt = ""
    if prompt_file and Path(prompt_file).exists():
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    ins = f"原始：{user_q}\n清洗：{cleaned_query}\n请生成 JSON 规划："
    client = OllamaClient(host=ollama_host, timeout=60)
    resp = client.generate(model=model, prompt=(prompt + "\n\n" + ins).strip(), stream=False)
    text = (resp.get("response") or resp.get("message") or "").strip()

    data = _safe_json_line(text) or {}
    # 兜底设置
    data.setdefault("use_rag_a", True)
    data.setdefault("use_rag_b", True if len(cleaned_query) <= 8 else False)
    data.setdefault("keywords", [cleaned_query])
    data.setdefault("rag_a", {}).setdefault("search_terms", [cleaned_query])
    data.setdefault("rag_a", {}).setdefault("bool_terms", [])
    data.setdefault("rag_b", {}).setdefault("search_terms", [cleaned_query])

    # 去重/清洗
    data["keywords"] = _dedup_list([k.strip() for k in data.get("keywords", [])])
    data["rag_a"]["search_terms"] = _dedup_list([k.strip() for k in data["rag_a"].get("search_terms", [])])
    data["rag_a"]["bool_terms"] = _dedup_list([k.strip() for k in data["rag_a"].get("bool_terms", [])])
    data["rag_b"]["search_terms"] = _dedup_list([k.strip() for k in data["rag_b"].get("search_terms", [])])

    # 生成 planned_query 串（供 RAG_A 用）
    data["planned_query"] = _make_planned_query(data, cleaned_query)
    return data
