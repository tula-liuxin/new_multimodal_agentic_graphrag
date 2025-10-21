# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, hashlib
from typing import List, Dict
from bs4 import BeautifulSoup

def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()

def ingest_html_to_chunks(root_dir: str, out_chunks: str):
    rows = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith((".html",".htm")):
                p = os.path.join(base, fn)
                html = _read_text(p)
                soup = BeautifulSoup(html, "lxml")
                # 去 script/style
                for tag in soup(["script","style","nav"]):
                    tag.decompose()
                title = (soup.title.get_text(" ", strip=True) if soup.title else os.path.basename(p))
                text = soup.get_text("\n", strip=True)
                if len(text) < 20:
                    # 占位页不写 chunk
                    continue
                rid = f"{_sha1(p)}|{0}"
                rows.append({"id": rid, "path": os.path.abspath(p), "title": title, "chunk_id": 0, "text": text[:4000]})
    os.makedirs(os.path.dirname(out_chunks), exist_ok=True)
    with open(out_chunks, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
