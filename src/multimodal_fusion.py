import os, json, math
import numpy as np
import joblib
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_embeddings(path):
    return np.memmap(path, dtype=np.float32, mode="r")

def unit_rows(X):
    # X 两维时行归一化
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n

def cosine(a, b):
    a = unit_rows(a)
    b = unit_rows(b)
    return (a @ b.T).squeeze()

def mmr_select(cands: List[Tuple[int, float]], sim_mat: np.ndarray, k: int=5, lambda_: float=0.7):
    # cands: (idx, score) 初始评分，高到低
    selected = []
    remaining = list(cands)
    if not remaining:
        return selected
    # 首选最高
    selected.append(remaining.pop(0))
    while remaining and len(selected) < k:
        best = None
        best_val = -1e9
        for idx, sc in remaining:
            # 与已选的最大相似度
            rep = max(sim_mat[idx, s[0]] for s in selected) if selected else 0.0
            val = lambda_ * sc - (1 - lambda_) * rep
            if val > best_val:
                best_val = val
                best = (idx, sc)
        remaining.remove(best)
        selected.append(best)
    return selected

def fuse_scores(scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    # 将各通道分数线性融合（归一化到 0..1）
    total = None
    for key, arr in scores.items():
        if arr is None: 
            continue
        w = float(weights.get(key, 0.0))
        if w == 0.0: 
            continue
        # 归一化
        arr = arr.astype(np.float32)
        if arr.size == 0:
            continue
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        if total is None:
            total = w * arr
        else:
            total += w * arr
    return total

def expand_neighbors(cand_idx: List[int], file_ids: List[str], chunks_file_map: Dict[int, str],
                     link_graph: Dict, folder_neighbors: int=0):
    # link-hop/同目录扩展的最小实现在 query_cli 中完成；此处保留接口占位
    return list(cand_idx)

