# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, re
from typing import Dict, List, Set
from tqdm import tqdm

from .utils import jsonl_read

def _ngrams_cn(text: str, nmin: int = 1, nmax: int = 3) -> List[str]:
    if not text:
        return []
    # 去除空白与控制符
    s = re.sub(r"\s+", "", text)
    grams = []
    L = len(s)
    for n in range(max(1, nmin), max(nmin, nmax) + 1):
        for i in range(0, L - n + 1):
            grams.append(s[i:i+n])
    return grams

def build_char_index(input_jsonl: str,
                     index_dir: str = "index",
                     nmin: int = 1,
                     nmax: int = 3) -> str:
    os.makedirs(index_dir, exist_ok=True)
    rows = jsonl_read(input_jsonl) if os.path.exists(input_jsonl) else []
    inv: Dict[str, Set[str]] = {}
    for r in tqdm(rows, desc='[CharIndex] 建立倒排', unit='chunk'):
        cid = r.get("id") or f"{r.get('path','')}|{r.get('chunk_id','0')}"
        txt = r.get("text", "")
        for g in _ngrams_cn(txt, nmin, nmax):
            inv.setdefault(g, set()).add(cid)

    # 转换为列表并写出
    out = {k: sorted(list(v)) for k, v in inv.items()}
    out_path = os.path.join(index_dir, "char_index.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nmin": nmin, "nmax": nmax, "index": out}, f, ensure_ascii=False)
    return out_path
