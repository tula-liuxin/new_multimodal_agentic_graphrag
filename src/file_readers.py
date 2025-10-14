
import os, re
from typing import Optional, Tuple
from .utils import get_logger

log = get_logger("file_readers")

def read_text_auto(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".txt", ".md", ".markdown", ".html", ".htm"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        log.warning(f"[warn] 无法读取文本文件: {path} ({e})")
        return ""

def ocr_image(path: str, backend: str = "rapidocr") -> str:
    text = ""
    try:
        if backend == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            res, _ = engine(path)
            if res:
                text = "\n".join([x[1] for x in res if len(x) >= 2])
                return text
    except Exception as e:
        log.warning(f"[warn] RapidOCR 失败({e})，尝试 tesseract...")
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path).convert("RGB")
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        log.warning(f"[warn] tesseract OCR 失败: {e}")
        return ""

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]

def clean_display_path(p: str) -> str:
    p = p.replace("\\\\?\\", "")
    return p
