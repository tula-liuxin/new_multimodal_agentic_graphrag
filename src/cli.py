# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse
from .utils import load_yaml, setup_logger, norm_path
from .ingest import ingest
from .embed_index import build_text_index
from .image_index import build_image_index
from .char_index import build_char_index
from .link_graph import build_graph

def _device_auto():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def main():
    ap = argparse.ArgumentParser(prog="cli.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default="config.yaml", help="YAML 配置文件路径")

    # ingest
    p_ing = sub.add_parser("ingest", parents=[common])
    p_ing.add_argument("--enable-ocr", dest="enable_ocr", action="store_true")
    p_ing.add_argument("--no-ocr", dest="enable_ocr", action="store_false")
    p_ing.add_argument("--incremental", action="store_true", help="仅处理变更文件/文件夹")
    p_ing.set_defaults(enable_ocr=None)

    # embed
    p_emb = sub.add_parser("embed", parents=[common])
    p_emb.add_argument("--embed-model", type=str, default=None)
    p_emb.add_argument("--incremental", action="store_true", help="仅向量化新增/变更文本")

    # imgindex
    p_img = sub.add_parser("imgindex", parents=[common])
    p_img.add_argument("--clip-local", type=str, default=None)
    p_img.add_argument("--incremental", action="store_true", help="仅向量化新增/变更图片")

    # charindex
    p_char = sub.add_parser("charindex", parents=[common])

    # links -> graph
    p_links = sub.add_parser("links", parents=[common])
    sub.add_parser("graph", parents=[common])

    # all
    p_all = sub.add_parser("all", parents=[common])
    p_all.add_argument("--embed-model", type=str, default=None)
    p_all.add_argument("--clip-local", type=str, default=None)
    p_all.add_argument("--enable-ocr", dest="enable_ocr", action="store_true")
    p_all.add_argument("--no-ocr", dest="enable_ocr", action="store_false")
    p_all.add_argument("--incremental", action="store_true", help="对全流程启用增量处理")
    p_all.set_defaults(enable_ocr=None)

    args = ap.parse_args()
    cfg = load_yaml(args.config)

    log_dir = cfg.get("debug_dir", "debug")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "console.log"))

    roots = [norm_path(p) for p in cfg.get("roots", [])]
    data_dir = cfg.get("data_dir", "data")
    os.makedirs(data_dir, exist_ok=True)

    ollama_host = cfg.get("ollama", {}).get("host", "http://127.0.0.1:11434")
    index_dir = cfg.get("index", {}).get("save_dir", "index")
    os.makedirs(index_dir, exist_ok=True)

    # switches / params
    enable_ocr = cfg.get("enable_ocr", False) if getattr(args, "enable_ocr", None) is None else bool(getattr(args, "enable_ocr"))
    embed_model = (getattr(args, "embed_model", None)) or cfg.get("models", {}).get("embed_model", "bge-m3:latest")
    clip_local = (getattr(args, "clip_local", None)) or cfg.get("image", {}).get("clip_local", None)
    incremental = bool(getattr(args, "incremental", False))

    # Ingest
    if args.cmd in ("ingest", "all"):
        print(f"[阶段] Ingest 开始 ... （增量={incremental}）")
        ingest(roots=roots,
               data_dir=data_dir,
               enable_ocr=enable_ocr,
               ocr_tesseract_cmd=cfg.get("ocr_tesseract_cmd"),
               ocr_merge_into_text=cfg.get("ocr_merge_into_text", True),
               logger=logger,
               io_workers=cfg.get("io_workers", 8),
               incremental=incremental)

    # Text embeddings
    if args.cmd in ("embed", "all"):
        print(f"[阶段] 文本向量 开始 ... （增量={incremental}）")
        build_text_index(
            os.path.join(data_dir, "chunks.jsonl"),
            index_dir=index_dir,
            host=ollama_host,
            model=embed_model,
            batch=cfg.get("index", {}).get("batch", 64),
            use_faiss=True,
            logger=logger,
            incremental=incremental,
            delta_json=os.path.join(data_dir, "delta.json")
        )

    # Image embeddings
    if args.cmd in ("imgindex", "all"):
        print(f"[阶段] 图像向量 开始 ... （增量={incremental}）")
        build_image_index(
            os.path.join(data_dir, "images.jsonl"),
            index_dir=index_dir,
            clip_local=clip_local,
            device=_device_auto(),
            logger=logger,
            batch=cfg.get("image", {}).get("batch", 32),
            incremental=incremental,
            delta_json=os.path.join(data_dir, "delta.json")
        )

    # Char/N-gram index（体量小，直接重建）
    if args.cmd in ("charindex", "all"):
        print("[阶段] 字符倒排 开始 ...")
        build_char_index(
            os.path.join(data_dir, "chunks.jsonl"),
            index_dir,
            cfg.get("char_index", {}).get("n_min", 1),
            cfg.get("char_index", {}).get("n_max", 3)
        )

    # Links & graph
    if args.cmd in ("links", "all"):
        print("[阶段] 链接图 开始 ...")
        build_graph(
            os.path.join(data_dir, "links.jsonl"),
            os.path.join(data_dir, "adj.json")
        )

if __name__ == "__main__":
    main()
