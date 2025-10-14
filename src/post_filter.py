# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import *
from .ollama_client import OllamaClient

FILTER_PROMPT = """你是检索候选的“保留判别器”。
任务：判断【候选内容】与【问题】是否存在**任何相关性**（包括别称、代词、上下文线索、模糊表达）。
规则：
- 宁可多留，不要错杀；
- 如果候选基本是链接或占位符，可标记为“弱相关-需邻居”，仍然保留但优先级较低；
- 输出 JSON：{"keep": true/false, "reason": "简要中文原因"}。
问题：{question}
候选内容：{candidate}
只输出 JSON。
"""

def parallel_filter(question: str, candidates: List[Dict], host: str, model: str, workers: int = 6, timeout: int = 15, min_keep: int = 6, logger=None):
    cli = OllamaClient(host=host, timeout=timeout, logger=logger)
    results = []

    def judge(c):
        txt = c.get("text") or c.get("ocr_text") or ""
        prompt = FILTER_PROMPT.format(question=question, candidate=txt[:1200])
        try:
            resp = cli.generate(model=model, prompt=prompt, stream=False)
            data = json.loads(resp.strip().splitlines()[-1])
            keep = bool(data.get("keep", True))
            reason = data.get("reason", "")
        except Exception:
            keep, reason = True, "解析失败，保守保留"
        c["keep"] = keep
        c["reason"] = reason
        c["clean_text"] = txt.strip()
        return c

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(judge, c) for c in candidates]
        for fu in as_completed(futs):
            try:
                results.append(fu.result())
            except Exception as e:
                if logger: logger.warning(f"过滤器异常：{e}")

    kept = [r for r in results if r.get("keep")]
    if len(kept) < min_keep:
        # 降阈：强制保留得分高的前若干
        rest = [r for r in results if not r.get("keep")]
        rest.sort(key=lambda x: -float(x.get("score", 0)))
        kept.extend(rest[:max(0, min_keep - len(kept))])
    return kept, results
