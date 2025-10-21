# -*- coding: utf-8 -*-
r"""
通用工具：路径归一化、jsonl 读写、分块、HTML 清洗封装、链接解析、并发、日志等。
"""
from __future__ import annotations
import os, re, sys, json, time, math, base64, hashlib, mimetypes, urllib.parse, threading
from typing import Iterable, List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from bs4 import BeautifulSoup
import regex as reg
import yaml
import logging
from logging import handlers
from PIL import Image
from io import BytesIO

# ------------- 日志 -------------
def setup_logger(log_path: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("graphrag")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = handlers.RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=2, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

# ------------- 配置 -------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ------------- 路径规范化 -------------
_WIN_PREFIXES = (r"\\?\ ", r"\\?\\".strip(), r"\?")

def norm_path(p: str) -> str:
    r"""将各种奇怪前缀/相对路径统一为 Windows 绝对路径，去掉 \?\ 与 \? 等。"""
    if not p:
        return p
    p = p.replace("/", "\\")
    for bad in ["\\\\?\\", "\\?"]:
        if p.startswith(bad):
            p = p[len(bad):]
    p = os.path.abspath(p)
    # 清理重复的反斜杠
    p = re.sub(r"\\\\+", r"\\", p)
    return p

def to_io_path(p: str) -> str:
    r"""返回用于文件I/O的安全路径：当长度过长时自动加上 \\?\ 前缀（仅内部 I/O 使用）。"""
    if not p:
        return p
    p2 = p.replace("/", "\\")
    # 绝对化
    p2 = os.path.abspath(p2)
    # 如果太长且未带前缀，则添加 \\?\
    try:
        if len(p2) >= 240 and not p2.startswith("\\\\?\\"):
            p2 = "\\\\?\\" + p2
    except Exception:
        pass
    return p2


def is_html(path: str) -> bool:
    return Path(path).suffix.lower() in {".html", ".htm"}

def is_text(path: str) -> bool:
    return Path(path).suffix.lower() in {".md", ".txt"}

def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

def is_pdf(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"

def read_file(path: str) -> str:
    with open(to_io_path(path), "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def sha1_of_path(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(to_io_path(path), "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return hashlib.sha1(path.encode("utf-8", "ignore")).hexdigest()

def jsonl_write(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def jsonl_read(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

# ------------- 文本分块（中文友好） -------------
def split_text_cn(text: str, max_chars: int = 1200) -> List[str]:
    """按段落与句子边界拆分，优先中文标点；保证每块不超过 max_chars。"""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    # 先按段落再细分
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    for p in paras:
        buf = ""
        for sent in reg.split(r"(?<=[。！？!?；;：:])", p):
            if len(buf) + len(sent) > max_chars and buf:
                chunks.append(buf.strip())
                buf = sent
            else:
                buf += sent
        if buf.strip():
            chunks.append(buf.strip())
    # 再做二次切片
    final = []
    for ch in chunks:
        if len(ch) <= max_chars:
            final.append(ch)
        else:
            for i in range(0, len(ch), max_chars):
                final.append(ch[i:i+max_chars])
    return final

# ------------- HTML 清洗/链接抽取 -------------
def html_to_text_and_links(html: str, base_dir: str) -> Tuple[str, List[Tuple[str, str]]]:
    """返回 纯文本 与 链接 (href/src, type=link|image)。"""
    soup = BeautifulSoup(html, "lxml")
    # 删除噪声
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    # 文本
    body_text = soup.get_text("\n", strip=True)
    text = (title + "\n" + body_text).strip() if title else body_text
    # 链接
    links = []
    for a in soup.find_all("a", href=True):
        links.append((a["href"], "link"))
    for img in soup.find_all("img", src=True):
        links.append((img["src"], "image"))
    # 归一化相对路径
    normed = []
    for url, tp in links:
        url = urllib.parse.unquote(url)
        if re.match(r"^[a-zA-Z]+://", url):
            # 远程 URL 不索引为本地路径，只保留为原始字符串
            normed.append((url, tp))
        else:
            p = url.replace("/", "\\")
            normed.append((os.path.normpath(os.path.join(base_dir, p)), tp))
    return text, normed

# ------------- Markdown 链接抽取 -------------
_MD_LINK = re.compile(r'!\[.*?\]\((.*?)\)|\[(?:.*?)\]\((.*?)\)')

def markdown_links(md: str, base_dir: str) -> List[Tuple[str, str]]:
    out = []
    for m in _MD_LINK.finditer(md):
        img, link = m.groups()
        if img:
            path = urllib.parse.unquote(img)
            if not re.match(r"^[a-zA-Z]+://", path):
                path = os.path.normpath(os.path.join(base_dir, path.replace("/", "\\")))
            out.append((path, "image"))
        elif link:
            path = urllib.parse.unquote(link)
            if not re.match(r"^[a-zA-Z]+://", path):
                path = os.path.normpath(os.path.join(base_dir, path.replace("/", "\\")))
            out.append((path, "link"))
    return out

# ------------- 并发 -------------
def run_threaded(func, items: Iterable[Any], max_workers: int = 8, desc: str = "") -> List[Any]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(func, it) for it in items]
        for fu in as_completed(futs):
            try:
                results.append(fu.result())
            except Exception as e:
                results.append(e)
    return results

# ------------- 简单 TopK -------------
import numpy as np

def topk_ip(query: np.ndarray, mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """内积 TopK，返回 (scores, idxs)；向量均已 L2 归一化。"""
    sims = (mat @ query.astype(np.float32))
    idx = np.argpartition(-sims, kth=min(k-1, sims.shape[0]-1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return sims[idx], idx

# ------------- Prompt 模板 -------------
SMART_QUERY_PROMPT = """你是检索改写器。将用户问题改写为有利召回的短语与关键词（中文为主），规则：
1) 去掉“的所有记录/说了什么/描述了什么/有哪些/如何/怎么”等行为性描述，保留“实体+主题词”；
2) 自动补充别名、同义词、中文简繁与英文名；
3) 生成 3~5 条子查询（含 1~2 个中文 char-gram 片段），每条不超过 12 个字；
仅返回 JSON 数组，如：["元宇宙 架构师", "Metaverse 架构", "莫维芬", "刘晓玲", "元宇宙 架构 布局"]
用户问题：{question}"""

def make_reference_block(ref_paths: List[str]) -> str:
    """生成“参考”段，编号从 [1] 开始，路径为绝对路径且去除奇怪前缀。"""
    uniq = []
    seen = set()
    for p in ref_paths:
        p = norm_path(p)
        if p not in seen:
            uniq.append(p); seen.add(p)
    if not uniq:
        return ""
    lines = ["参考："]
    for i, p in enumerate(uniq, 1):
        lines.append(f"[{i}] {p}")
    return "\n".join(lines)

def sanitize_answer(answer: str) -> str:
    """清理重复“参考”段，去掉多余的编号重复。"""
    # 只保留最后一个“参考：”之后的内容由我们拼接
    answer = re.sub(r"\n?参考[:：][\s\S]*$", "", answer.strip())
    return answer

