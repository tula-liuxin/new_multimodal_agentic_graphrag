
import os, glob
from typing import Dict, Any, List
from .utils import get_logger, load_cfg, ensure_dir, sha1, safe_rel, chunk_path, image_list_path
from .file_readers import read_text_auto, ocr_image, is_image
from .text_splitter import split_text

def run_ingest(cfg=None, logger=None):
    log = logger or get_logger("ingest")
    cfg = cfg or load_cfg()
    root = cfg["root_dir"]
    data_dir = cfg["data_dir"]
    enable_ocr = bool(cfg.get("enable_ocr", False))
    ocr_backend = cfg.get("ocr_backend", "rapidocr")

    all_files = []
    for ext in ["**/*.md", "**/*.txt", "**/*.html", "**/*.htm", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.webp", "**/*.bmp"]:
        all_files.extend(glob.glob(os.path.join(root, ext), recursive=True))

    text_rows: List[Dict[str, Any]] = []
    img_rows: List[Dict[str, Any]] = []

    for i, p in enumerate(all_files):
        rp = safe_rel(p, root)
        rid = sha1(rp)
        if is_image(p):
            text = ""
            if enable_ocr:
                text = ocr_image(p, backend=ocr_backend) or ""
            img_rows.append({"id": rid, "path": p, "rel": rp, "ocr": text})
        else:
            text = read_text_auto(p)
            chunks = split_text(text, cfg.get("text_max_chars", 1200), cfg.get("text_overlap", 100))
            for j, c in enumerate(chunks):
                cid = sha1(f"{rp}:{j}")
                text_rows.append({"id": cid, "doc": rid, "rel": rp, "chunk": j, "text": c})

        if (i+1) % 500 == 0:
            log.info(f"ingest 进度: {i+1}/{len(all_files)}")

    ensure_dir(data_dir)
    from .utils import write_jsonl
    write_jsonl(chunk_path(data_dir), text_rows)
    write_jsonl(image_list_path(data_dir), img_rows)

    log.info(f"ingest 完成：文本块 {len(text_rows)}，图片文件 {len(img_rows)}")
