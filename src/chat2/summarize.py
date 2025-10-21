from typing import Dict, Any, List, Tuple
import os, glob, json
from .utils import first_n_sentences, read_jsonl, collect_turn_files_paths
from .llm_client import ollama_generate

def summarize_turn(answer_text: str, cleaned_notes: str, cfg: Dict[str, Any], question: str = "") -> str:
    header = f"Q：{question.strip()}" if question else ""
    body = (answer_text or "").strip()
    addon = (cleaned_notes or "").strip()
    merged = "\n".join([p for p in [header, body, addon] if p])
    return first_n_sentences(merged, 6)

def _collect_turn_summaries(session_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(session_dir, "turn-*", "turn_summary.txt")))
    out = []
    for p in paths:
        try:
            out.append(open(p, "r", encoding="utf-8").read().strip())
        except Exception:
            continue
    return out

def summarize_chat_scan(session_dir: str, cfg: Dict[str, Any]) -> str:
    items = _collect_turn_summaries(session_dir)
    if not items:
        return ""
    keep = int(cfg.get("chat_summary_max_turns", 24))
    items = items[-keep:]
    joined = "\n".join(items)
    return first_n_sentences(joined, 10)

def _gather_conversation_materials(session_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    turns_keep = int(cfg.get("summarize_context_turns", 18))
    summs = []
    paths = sorted(glob.glob(os.path.join(session_dir, "turn-*", "turn_summary.txt")))[-turns_keep:]
    for p in paths:
        try:
            summs.append(open(p, "r", encoding="utf-8").read().strip())
        except: pass
    chat_summary = ""
    if cfg.get("summarize_include_chat_summary", True):
        cs = os.path.join(session_dir, "summary.txt")
        if os.path.exists(cs):
            chat_summary = open(cs, "r", encoding="utf-8").read().strip()
    tf_snippets = []
    if cfg.get("summarize_include_turn_files", True):
        k = int(cfg.get("summarize_turn_files_per_turn", 2))
        for tf_path in collect_turn_files_paths(session_dir):
            try:
                rows = read_jsonl(tf_path)[:k]
                for r in rows:
                    tf_snippets.append(r.get("text",""))
            except: pass
    return {"turn_summaries": summs, "chat_summary": chat_summary, "turn_files": tf_snippets}

def summarize_conversation(session_dir: str, cfg: Dict[str, Any], question: str) -> str:
    mats = _gather_conversation_materials(session_dir, cfg)
    parts = []
    if mats.get("chat_summary"):
        parts.append("【会话滚动摘要（历史）】\n" + mats["chat_summary"])
    if mats.get("turn_summaries"):
        parts.append("【近几轮摘要（按时间序）】\n- " + "\n- ".join(mats["turn_summaries"]))
    if mats.get("turn_files"):
        parts.append("【检索片段摘录】\n- " + "\n- ".join(mats["turn_files"][:40]))
    prompt = f"""你是一个会议记录与对话归纳助手。
请将下面提供的历史摘要/最近若干轮摘要/检索片段进行**去重合并**，生成「本次对话至今」的完整总结：

要求：
- 按主题分段（人物/关系、RAG与索引、配置与报错、哲学/个人观念、待办）
- 保留关键证据与来源线索（如出现具体文件名/路径/指令，可简要保留）
- 不要遗漏早先出现过的重要人物与问题（尤其：刘晓玲、莫维芬、ASI 等）
- 语言简洁、编号清晰；长度不超过 800 字

用户请求：{question}

素材：
{chr(10).join(parts)}

请直接输出总结。
"""
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    model = cfg.get("model_summarize", "qwen2.5:3b-instruct")
    timeout = int(cfg.get("llm_timeout", 45))
    num_ctx = max(4096, int(cfg.get("llm_max_ctx", 4096)))
    return ollama_generate(host=host, model=model, prompt=prompt, timeout=timeout, num_ctx=num_ctx, temperature=0.0)
