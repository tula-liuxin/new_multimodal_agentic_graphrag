import os, json, re
from typing import List, Dict
from .utils import ensure_dir, sha1, norm_win_abs
from tqdm import tqdm
from .file_readers import read_text_auto, ocr_image
from PIL import Image

TEXT_EXT = {".md",".markdown",".txt",".html",".htm",".rst",".tex",".csv",".tsv",".json",".yaml",".yml",".ini",".log",".xml",".xlsx",".xls",".docx",".pptx",".pdf"}
IMG_EXT = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff",".gif"}

def iter_chunks(text: str, max_chars=1200, overlap=200):
    if not text:
        return
    overlap = max(0, min(overlap, max_chars // 2))
    # 按段落切，再对超长段落滑窗；逐条 yield，避免一次性占用内存
    paras = (p.strip() for p in re.split(r"\n\s*\n", text))
    for p in paras:
        if not p:
            continue
        if len(p) <= max_chars:
            yield p
        else:
            s = 0
            step = max_chars - overlap
            if step <= 0:
                step = max_chars
            L = len(p)
            while s < L:
                e = min(L, s + max_chars)
                yield p[s:e]
                if e >= L:
                    break
                s += step

def walk_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)

def run_ingest(cfg, logger):
    input_dir = cfg["input_dir"]
    data_dir = cfg["data_dir"]
    ensure_dir(data_dir)

    chunks_path = os.path.join(data_dir, "chunks.jsonl")
    images_path = os.path.join(data_dir, "images.jsonl")

    max_chars = int(cfg.get("max_chars", 1200))
    overlap = int(cfg.get("overlap_chars", 200))

    enable_ocr = bool(cfg.get("enable_ocr", False))
    ocr_lang = cfg.get("ocr_lang", "chi_sim+eng")
    ocr_psm = int(cfg.get("ocr_psm", 6))

    cnt_txt = 0
    cnt_img = 0

    # 预扫描文件以显示进度
    all_files = list(walk_files(input_dir))
    with open(chunks_path, "w", encoding="utf-8") as f_txt, open(images_path, "w", encoding="utf-8") as f_img:
        for p in tqdm(all_files, desc="ingest 扫描", unit="file"):
            ext = os.path.splitext(p)[1].lower()
            rel = os.path.relpath(p, input_dir).replace("/", "\\")
            if ext in TEXT_EXT:
                try:
                    text = read_text_auto(p)
                except Exception as e:
                    # 文件可能在扫描后被移动/删除，或长路径/权限问题
                    print(f"[warn] 无法读取文本文件: {p} ({e})")
                    continue
                i = 0
                for ch in iter_chunks(text, max_chars, overlap):
                    rid = sha1(f"{rel}:{i}")
                    rec = {
                        "id": rid,
                        "source_path": rel,
                        "chunk_index": i,
                        "text": ch if ch.strip() else " ",
                        "modality": ext.lstrip('.')
                    }
                    f_txt.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    i += 1
                cnt_txt += 1
            elif ext in IMG_EXT:
                try:
                    from PIL import Image
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception:
                    w, h = 0, 0
                ocr_text = ""
                if enable_ocr:
                    ocr_text = ocr_image(p, lang=ocr_lang, psm=ocr_psm)
                rid = sha1(f"{rel}:image")
                rec = {
                    "id": rid,
                    "source_path": rel,
                    "width": w,
                    "height": h,
                    "modality": "image",
                    "ocr_text": ocr_text,
                    "caption": ""
                }
                f_img.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt_img += 1
            else:
                continue

    logger.info(f"ingest 完成：文本文件 {cnt_txt}，图片文件 {cnt_img}")
