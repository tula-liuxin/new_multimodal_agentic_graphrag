import json, re
from pathlib import Path
from typing import Optional
from .ollama_client import OllamaClient

SELF_TOKENS = {"我","本人","我们","吾","自己"}
REL_TOKENS = {"关系","相干","关联","相关","联系"}

def _json_or_text(text: str):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    return text

def _heuristic_merge(original: str, cleaned: str) -> str:
    has_self = any(tok in original for tok in SELF_TOKENS)
    has_rel = any(tok in original for tok in REL_TOKENS)
    if has_self and has_rel:
        cand = cleaned
        if not cand or cand in REL_TOKENS:
            m = re.findall(r"[\u4e00-\u9fa5]{2,6}", original)
            blacklist = {"什么","是谁","关系","相关","记录","总结","查下","看看","三观","前面","内容"}
            m = [x for x in m if x not in blacklist]
            if m:
                m.sort(key=len, reverse=True)
                cand = m[0]
        if not cand:
            cand = cleaned
        if cand:
            cleaned = f"{cand} 与 我 关系"
    return cleaned

def _dedup_tail_rel(s: str) -> str:
    # 去除重复的“与 我 关系”尾缀
    s = re.sub(r"(\s*与\s*我\s*关系){2,}$", " 与 我 关系", s)
    # 去重多空格
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_query(user_query: str,
                    ollama_host: str,
                    model: str,
                    prompt_file: Optional[str]=None) -> str:
    prompt = ""
    if prompt_file and Path(prompt_file).exists():
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    ins = f"输入：{user_query}\n请输出清洗后的检索查询："
    client = OllamaClient(host=ollama_host, timeout=60)
    resp = client.generate(model=model, prompt=(prompt + "\n\n" + ins).strip(), stream=False)
    text = (resp.get("response") or resp.get("message") or "").strip()
    data = _json_or_text(text)

    if isinstance(data, dict):
        cleaned = (data.get("search") or data.get("query") or "").strip()
    else:
        cleaned = str(data).splitlines()[0].strip()

    cleaned = _heuristic_merge(user_query, cleaned)
    cleaned = _dedup_tail_rel(cleaned)

    return cleaned if cleaned else user_query[:120]
