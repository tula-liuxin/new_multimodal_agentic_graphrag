# -*- coding: utf-8 -*-
import os, tempfile
from src.utils import html_to_text_and_links

def test_html_links():
    html = """<html><head><title>T</title></head>
    <body><a href="A/B C.html">x</a><img src="img/k.png"/></body></html>"""
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td,"x.html"),"w",encoding="utf-8").write(html)
        txt, links = html_to_text_and_links(html, td)
        assert "T" in txt
        paths = [l[0] for l in links]
        assert any(p.endswith(os.path.join("A", "B C.html")) for p in paths)
        assert any(p.endswith(os.path.join("img", "k.png")) for p in paths)
