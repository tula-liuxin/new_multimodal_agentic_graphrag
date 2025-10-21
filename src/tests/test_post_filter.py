# -*- coding: utf-8 -*-
import types
from src.post_filter import parallel_filter
import src.post_filter as pf

class DummyClient:
    def __init__(self, *a, **k): pass
    def generate(self, *a, **k):
        # 返回 keep=false 的 JSON，迫使走 min_keep 兜底
        return '{"keep": false, "reason": "irrelevant"}'

def test_post_filter_min_keep(monkeypatch):
    # 替换 OllamaClient 为 DummyClient
    monkeypatch.setattr(pf, "OllamaClient", DummyClient)
    question = "元宇宙 架构师"
    cands = [{"id":str(i), "text":"nothing", "score":0.1*i} for i in range(4)]
    kept, allres = parallel_filter(question, cands, host="http://127.0.0.1:11434", model="qwen2.5:3b-instruct",
                                   workers=2, timeout=5, min_keep=3, logger=None)
    assert len(kept) >= 3
