import os, json, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from .utils import get_logger
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, base_dir: str, image_records):
        self.base_dir = base_dir
        self.recs = image_records
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, idx):
        rec = self.recs[idx]
        path = os.path.join(self.base_dir, rec["source_path"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (255,255,255))
        return img, idx

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def run_imgindex(cfg, logger):
    data_dir = cfg["data_dir"]
    input_dir = cfg["input_dir"]
    images_path = os.path.join(data_dir, "images.jsonl")
    ids_path = os.path.join(data_dir, "image_ids.json")
    emb_npy = os.path.join(data_dir, "image_embeddings.npy")

    model_name = cfg.get("image_model_name", "ViT-L-14")
    pretrained = cfg.get("image_pretrained", "laion2b_s32b_b82k")
    batch = int(cfg.get("image_batch", 64))
    precision = cfg.get("image_precision", "fp16")
    workers = int(cfg.get("image_workers", 6))

    recs = list(read_jsonl(images_path))
    if len(recs) == 0:
        logger.warning("images.jsonl 为空")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    ds = ImageDataset(input_dir, recs)
    def collate(batch):
        imgs = [preprocess(x[0]) for x in batch]
        idxs = [x[1] for x in batch]
        return torch.stack(imgs, dim=0), torch.tensor(idxs, dtype=torch.long)

    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)

    # 推断维度
    with torch.no_grad():
        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        if precision == "fp16" and device == "cuda":
            with torch.cuda.amp.autocast():
                feat0 = model.encode_image(x0)
        else:
            feat0 = model.encode_image(x0)
        dim = feat0.shape[-1]

    mm = np.memmap(emb_npy, dtype=np.float32, mode="w+", shape=(len(recs), dim))

    offset = 0
    with torch.no_grad():
        pbar = tqdm(total=len(recs), desc="image encode", unit="img")
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            if precision == "fp16" and device == "cuda":
                with torch.cuda.amp.autocast():
                    feats = model.encode_image(imgs)
            else:
                feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.detach().cpu().float().numpy()
            for bi, ridx in enumerate(idxs.tolist()):
                mm[ridx, :] = feats[bi]
            offset += imgs.shape[0]
            pbar.update(imgs.shape[0])
        pbar.close()

    mm.flush()
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump([r["id"] for r in recs], f, ensure_ascii=False, indent=2)
