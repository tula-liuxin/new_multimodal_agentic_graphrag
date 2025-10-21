# -*- coding: utf-8 -*-
from __future__ import annotations
from bs4 import BeautifulSoup
import re

def notion_html_to_text(html: str) -> str:
    """Notion HTML 专用清理：移除 script/style/nav，保留正文，合并段落。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    # Notion 标题一般在 h1 或 title
    title = ""
    h1 = soup.find(["h1", "h2"])
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    text = soup.get_text("\n", strip=True)
    if title and not text.startswith(title):
        text = f"{title}\n{text}"
    # 合理压缩空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
