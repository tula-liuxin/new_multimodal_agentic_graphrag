# -*- coding: utf-8 -*-
import os, tempfile, json, shutil
from src.ingest import ingest

def test_ingest_outputs():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "root"); os.makedirs(d)
        # html
        open(os.path.join(d,"a.html"),"w",encoding="utf-8").write("<html><head><title>A</title></head><body><p>中文段落一。</p><a href='b.html'>b</a></body></html>")
        # md
        open(os.path.join(d,"note.md"),"w",encoding="utf-8").write("这是笔记。\n\n[link](a.html)")
        # img
        from PIL import Image
        im = Image.new("RGB",(32,32),(255,255,255)); im.save(os.path.join(d,"x.png"))
        chunks, images, links, nodes = ingest([d], os.path.join(td,"data"), enable_ocr=False, ocr_tesseract_cmd="")
        assert os.path.exists(chunks) and os.path.getsize(chunks)>0
        assert os.path.exists(images) and os.path.getsize(images)>0
        assert os.path.exists(links) and os.path.getsize(links)>0
