import os, json
from .ollama_client import OllamaClient

def run_captions(cfg, logger):
    # 可选：为图片生成描述字幕（写回 images.jsonl 的 caption 字段）
    data_dir = cfg["data_dir"]
    images_path = os.path.join(data_dir, "images.jsonl")

    base = cfg["ollama_base_url"]
    model = cfg.get("caption_model", "qwen2.5vl:latest")
    client = OllamaClient(base)

    recs = []
    with open(images_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))

    changed = False
    with open(images_path, "w", encoding="utf-8") as f:
        for rec in recs:
            if not rec.get("caption"):
                prompt = f"请简洁用中文描述这张图片的核心内容，50字以内：{rec['source_path']}"
                try:
                    caption = client.generate(model, prompt)
                except Exception as e:
                    caption = ""
                rec["caption"] = caption
                changed = True
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed
