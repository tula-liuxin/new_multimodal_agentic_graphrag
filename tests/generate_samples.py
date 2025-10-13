# 生成最小样例数据到 config.input_dir 下（若不存在则创建）
import os, io
from PIL import Image, ImageDraw, ImageFont
import yaml

def load_cfg(p="config.yaml"):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    cfg = load_cfg("config.yaml")
    inp = cfg["input_dir"]
    base = os.path.join(inp, "General")
    ensure_dir(base)

    md = os.path.join(base, "样例文档.md")
    if not os.path.exists(md):
        with open(md, "w", encoding="utf-8") as f:
            f.write("# 计划\n\n我最近计划去杭州与罗建祥见面，讨论GraphRAG与多模态检索。\n\n参见[相关图片](pic.png)。")

    html = os.path.join(base, "index.html")
    if not os.path.exists(html):
        with open(html, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>目录</h1><a href='样例文档.md'>样例文档</a></body></html>")

    imgp = os.path.join(base, "pic.png")
    if not os.path.exists(imgp):
        img = Image.new("RGB", (512, 256), (255,255,255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "会议日程：GraphRAG 讨论", fill=(0,0,0))
        d.text((10, 60), "联系人：罗建祥", fill=(0,0,0))
        img.save(imgp)

    print(f"已生成样例：{base}")

if __name__ == "__main__":
    main()
