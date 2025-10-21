import json, re
from pathlib import Path
from typing import Dict, Any
from .ollama_client import OllamaClient

def _extract_json(text: str) -> Dict[str, Any]:
    # 尽可能鲁棒地从文本中取出第一个 JSON 对象
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        # 去掉可能的尾逗号等
        cleaned = re.sub(r',\s*\}', '}', m.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

def route_intent(user_query: str, *, ollama_host: str, model: str, prompt_file: str) -> Dict[str, Any]:
    router_prompt = Path(prompt_file).read_text(encoding='utf-8')
    sys = router_prompt
    ins = f"用户输入：{user_query}\n请给出 JSON："
    client = OllamaClient(host=ollama_host, timeout=60)
    resp = client.generate(model=model, prompt=f"{sys}\n\n{ins}", stream=False)
    text = resp.get("response") or resp.get("message") or ""
    data = _extract_json(text) or {}
    # 默认兜底
    data.setdefault("use_rag_a", False)
    data.setdefault("use_rag_b", True)      # 倾向 session 级记忆检索
    data.setdefault("operation", "retrieve")
    data.setdefault("scope", "session")
    data.setdefault("reason", "兜底：缺省使用 RAG_B session 检索")
    return data
