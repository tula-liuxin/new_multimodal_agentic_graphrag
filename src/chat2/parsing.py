import re, json
from typing import Dict, Any, List
from .llm_client import ollama_generate

GENERIC = {"所有","帮我","我的","一下","如何","怎么","哪些","那个","这个","请","谢谢","关系","是谁","是什么"}
PLURAL_PN = ["他们","她们","TA们","ta们","二人","两人","两者","两个人","他们两个"]

def _rule_parse(question: str, last_entities: List[str]) -> Dict[str, Any]:
    q = question.strip().lower()
    need_history = any(w in q for w in ["之前","上轮","history","历史","前面","本次对话"])
    need_rag = not any(w in q for w in ["不用检索","no rag","不要查","不需要查询"])
    tokens = re.findall(r"[\u4e00-\u9fff]{1,10}|[a-zA-Z0-9_\.\-]{2,}", question)
    keywords = [t for t in tokens if t not in GENERIC][:12]
    entities = [t for t in keywords if re.fullmatch(r"[\u4e00-\u9fff]{2,8}", t)]
    coref = {}
    if any(p in question for p in PLURAL_PN) and len(last_entities) >= 2:
        coref = {"pronoun":"他们","targets": last_entities[-2:]}
    intent = "summarize_conversation" if any(k in q for k in ["总结本次对话","总结这次对话","回顾对话","对话总结"]) else "qa"
    actions = ["answer"]
    if intent == "qa" and need_rag: actions.insert(0, "retrieve")
    if intent == "summarize_conversation": actions = ["summarize_conversation"]
    return {
        "intent": intent,
        "need_rag": bool(need_rag),
        "need_history": bool(need_history or intent=="summarize_conversation"),
        "history_scopes": ["turn_files","turn_summary","chat_summary"] if (need_history or intent=="summarize_conversation") else [],
        "actions": actions,
        "entities": entities,
        "keywords": keywords,
        "coref": coref,
        "pairs": [],
        "topics": []
    }

def _json_from_text(txt: str) -> Dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            fixed = re.sub(r",\s*\}", "}", m.group(0))
            fixed = re.sub(r",\s*\]", "]", fixed)
            return json.loads(fixed)
        except Exception:
            return {}

def _prompt(question: str, last_focus: str, chat_summary: str, last_turn_summary: str, pronouns: List[str], last_entities: List[str]) -> str:
    return f'''
你是“问题解析器”。只输出 JSON，不要多余字符。

字段：
- intent: "qa" | "summarize_conversation" | "analyze_relation" | "list_entities" 等
- need_rag: bool
- need_history: bool
- history_scopes: ["turn_files","turn_summary","chat_summary"] 的子集
- actions: 例如 ["retrieve","answer"]、或 ["summarize_conversation"]
- entities: ["实体1","实体2",...]
- keywords: ["关键词1","关键词2",...]
- coref: 代词消解
  - 单数: {{"pronoun":"她","target":"刘晓玲"}}
  - 复数: {{"pronoun":"他们","targets":["刘晓玲","莫维芬"]}}
- pairs: 若涉及关系或比较: [{{"a":"刘晓玲","b":"莫维芬","relation":"关系"}}]
- topics: 主题词

上下文：
- last_focus: {last_focus or "（无）"}
- last_entities: {last_entities or []}
- chat_summary: {(chat_summary or "")[:300] or "（无）"}
- last_turn_summary: {(last_turn_summary or "")[:300] or "（无）"}

用户问题：{question}

严格 JSON：
'''

def _llm_parse(question: str, cfg: Dict[str, Any], last_focus: str, chat_summary: str, last_turn_summary: str, last_entities: List[str]) -> Dict[str, Any]:
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    model = cfg.get("model_parser", "qwen3:4b-instruct-2507-fp16")
    timeout = int(cfg.get("llm_timeout", 45))
    num_ctx = int(cfg.get("llm_max_ctx", 4096))
    pronouns = cfg.get("coref_pronouns") or ["她","他","TA","ta","她们","他们","两者","两人","他们两个"]
    p = _prompt(question, last_focus, chat_summary, last_turn_summary, pronouns, last_entities)
    out = ollama_generate(host=host, model=model, prompt=p, timeout=timeout, num_ctx=num_ctx, temperature=0.0)
    obj = _json_from_text(out)
    if not isinstance(obj, dict) or not obj:
        return {}
    obj.setdefault("intent", "qa")
    obj.setdefault("need_rag", True)
    obj.setdefault("need_history", (obj.get("intent") == "summarize_conversation"))
    obj.setdefault("history_scopes", ["turn_files","turn_summary","chat_summary"] if obj.get("need_history") else [])
    obj.setdefault("actions", ["retrieve","answer"] if obj.get("intent")=="qa" and obj.get("need_rag") else ["answer"])
    obj.setdefault("entities", [])
    obj.setdefault("keywords", [])
    obj.setdefault("coref", {})
    obj.setdefault("pairs", [])
    obj.setdefault("topics", [])
    return obj

def parse_question(question: str, cfg: Dict[str, Any], last_focus: str = "", chat_summary: str = "", last_turn_summary: str = "", last_entities: List[str] = None) -> Dict[str, Any]:
    last_entities = last_entities or []
    if cfg.get("enable_llm_parser", True):
        try:
            obj = _llm_parse(question, cfg, last_focus, chat_summary, last_turn_summary, last_entities)
            if obj:
                return obj
        except Exception:
            pass
    return _rule_parse(question, last_entities)
