import os, json, re
from collections import defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_audit(cfg, logger, term: str, alt: str=None, trad: bool=False):
    input_dir = cfg["input_dir"]
    data_dir = cfg["data_dir"]
    chunks_path = os.path.join(data_dir, "chunks.jsonl")

    raw_files_hit = set()
    chunk_files_hit = set()
    chunks_total_hits = 0

    patt = re.compile(re.escape(term))
    patt_alt = re.compile(re.escape(alt)) if alt else None

    # raw 级命中（粗略）
    for dirpath, _, fns in os.walk(input_dir):
        for fn in fns:
            if patt.search(fn) or (patt_alt and patt_alt.search(fn)):
                rel = os.path.relpath(os.path.join(dirpath, fn), input_dir).replace("/", "\\")
                raw_files_hit.add(rel)

    # chunk 级命中
    for rec in read_jsonl(chunks_path):
        text = rec.get("text","")
        if patt.search(text) or (patt_alt and patt_alt.search(text)):
            chunk_files_hit.add(rec["source_path"])
            chunks_total_hits += 1

    print("=== 文本审计 ===")
    print(f"raw_files_hit: {len(raw_files_hit)} 示例: {list(raw_files_hit)[:5]}")
    print(f"chunk_files_hit: {len(chunk_files_hit)} 示例: {list(chunk_files_hit)[:5]}")
    print(f"chunks_total_hits: {chunks_total_hits}")

    if len(raw_files_hit) > len(chunk_files_hit)*3:
        print("建议：检查编码/抽取/OCR 设置，可能存在 raw≫chunk 的丢失。")
