# 生成最小样例数据到 config.input_dir 下
import os
from PIL import Image, ImageDraw
import yaml

def load_cfg(p="config.yaml"):
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    cfg = load_cfg("config.yaml")
    inp = cfg["input_dir"]
    base = os.path.join(inp, "General")
    ensure_dir(base)

    md = os.path.join(base, "样例文档-AGI.md")
    if not os.path.exists(md):
        with open(md, "w", encoding="utf-8") as f:
            f.write("# AGI/ASI 路线\n\n通用人工智能（AGI）与超人工智能（ASI）的发展路径、方法与控制问题。\n\n参见[相关图片](pic_agi.png)。")

    imgp = os.path.join(base, "pic_agi.png")
    if not os.path.exists(imgp):
        img = Image.new("RGB", (640, 360), (255,255,255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "通用人工智能（AGI）研究路线", fill=(0,0,0))
        d.text((10, 50), "超人工智能（ASI）控制问题", fill=(0,0,0))
        img.save(imgp)

    print(f"已生成样例：{base}")

if __name__ == "__main__":
    main()
