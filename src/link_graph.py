
import os, re
from .utils import get_logger, load_cfg, chunk_path, links_path, read_jsonl, write_jsonl

MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

def run_links(cfg=None, logger=None):
    log = logger or get_logger("links")
    cfg = cfg or load_cfg()
    rows = read_jsonl(chunk_path(cfg["data_dir"]))
    if not rows:
        log.info("没有发现文本块（chunks.jsonl）。请先运行 ingest。")
        return

    by_rel = {}
    for r in rows:
        rel = r.get("rel", "")
        txt = r.get("text", "")
        for m in MD_LINK.finditer(txt):
            href = m.group(2)
            by_rel.setdefault(rel, []).append(href)

    out = []
    for src, dsts in by_rel.items():
        out.append({"src": src, "dst": sorted(set(dsts))})
    write_jsonl(links_path(cfg["data_dir"]), out)
    log.info(f"links 构建完成：{len(out)} 个节点。")
