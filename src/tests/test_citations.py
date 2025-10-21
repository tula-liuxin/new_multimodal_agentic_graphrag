# -*- coding: utf-8 -*-
from src.utils import make_reference_block, sanitize_answer

def test_reference_block_and_sanitize():
    refs = [r"\\?\C:\A\a.html", r"C:\A\b.html", r"C:\A\a.html"]
    blk = make_reference_block(refs)
    assert "[1]" in blk and "[2]" in blk
    ans = "正文\n参考：\n[1] X\n[2] Y"
    out = sanitize_answer(ans)
    assert "参考" not in out
