# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Dict, List, Set
from .utils import *

def build_graph(links_jsonl: str, out_path: str):
    edges = jsonl_read(links_jsonl)
    adj: Dict[str, Set[str]] = {}
    for e in edges:
        s = norm_path(e["src"]); d = norm_path(e["dst"])
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()) # ensure node exists
    json.dump({k:list(v) for k,v in adj.items()}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out_path

def expand_neighbors(adj_path: str, seeds: List[str], hop: int = 1, folder_neighbors: int = 0) -> List[str]:
    adj = json.load(open(adj_path, "r", encoding="utf-8"))
    cur = set(seeds); seen = set(seeds)
    for _ in range(hop):
        nxt = set()
        for u in list(cur):
            for v in adj.get(u, []):
                if v not in seen: nxt.add(v)
        seen.update(nxt); cur = nxt
    # 同文件夹邻居
    if folder_neighbors > 0:
        folders = {os.path.dirname(p) for p in seeds}
        for f in folders:
            try:
                files = [norm_path(os.path.join(f, x)) for x in os.listdir(f)]
            except Exception:
                files = []
            for p in files[:folder_neighbors]:
                seen.add(p)
    return list(seen)
