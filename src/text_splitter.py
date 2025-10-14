
from typing import List

def split_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    n = len(text)
    if max_chars <= 0: max_chars = 1000
    if overlap < 0: overlap = 0
    step = max(1, max_chars - overlap)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end >= n: break
        start += step
    return chunks
