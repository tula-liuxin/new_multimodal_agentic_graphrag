# -*- coding: utf-8 -*-
from src.utils import norm_path

def test_norm_path():
    p = norm_path(r"\\?\C:\Temp\foo\..\bar\index.html")
    assert p.lower().endswith(r"c:\temp\bar\index.html")
    assert "\\?\\" not in p and r"\?" not in p
