import os, json, re, numpy as np

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_imgaudit(cfg, logger, term: str, mode: str="clip"):
    data_dir = cfg["data_dir"]
    input_dir = cfg["input_dir"]
    images_path = os.path.join(data_dir, "images.jsonl")
    img_ids_path = os.path.join(data_dir, "image_ids.json")
    img_emb_path = os.path.join(data_dir, "image_embeddings.npy")

    recs = list(read_jsonl(images_path))
    print(f"images 总数：{len(recs)}")

    if mode in ["ocr","caption"]:
        cnt = 0
        for r in recs:
            txt = (r.get("ocr_text","") if mode=="ocr" else r.get("caption","")) or ""
            if term in txt:
                cnt += 1
        print(f"{mode} 覆盖命中：{cnt}")
    elif mode == "clip":
        # 仅统计是否已经有 embedding
        if not (os.path.exists(img_ids_path) and os.path.exists(img_emb_path)):
            print("未找到 image 索引（先运行：python -m src.cli imgindex）")
            return
        ids = json.load(open(img_ids_path, "r", encoding="utf-8"))
        emb = np.memmap(img_emb_path, dtype=np.float32, mode="r")
        print(f"image_ids: {len(ids)}，embeddings shape: {emb.shape}")
        print("示例路径：")
        for r in recs[:5]:
            print(os.path.join(input_dir, r["source_path"]))
