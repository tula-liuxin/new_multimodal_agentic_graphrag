# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
from .utils import *
def cross_rerank(query: str, candidates: List[Tuple[str, str, float]], model_name: str = "BAAI/bge-reranker-v2-m3", topk: int = 50, logger=None):
    """使用 Sentence-Transformers CrossEncoder 做重排；失败则原样返回。"""
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:
        if logger: logger.warning(f"CrossEncoder 不可用：{e}")
        return candidates[:topk]
    try:
        pairs = [(query, c[1]) for c in candidates]
        ce = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        scores = ce.predict(pairs, convert_to_tensor=False, show_progress_bar=False)
        rescored = [(c[0], c[1], float(s)) for c, s in zip(candidates, scores)]
        rescored.sort(key=lambda x: -x[2])
        return rescored[:topk]
    except Exception as e:
        if logger: logger.warning(f"重排失败：{e}")
        return candidates[:topk]
