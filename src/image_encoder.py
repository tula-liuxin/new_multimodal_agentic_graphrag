
import os, numpy as np
from typing import List, Dict, Any
from PIL import Image, UnidentifiedImageError
from .utils import get_logger, load_cfg, image_list_path, image_emb_path

def _load_openclip(clip_local: str = ""):
    try:
        import open_clip
        if clip_local and os.path.isdir(clip_local):
            model, _, preprocess = open_clip.create_model_and_transforms_from_pretrained(clip_local)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        return model, preprocess
    except Exception as e:
        return None, None

def run_imgindex(cfg=None, logger=None):
    log = logger or get_logger("imgindex")
    cfg = cfg or load_cfg()
    data_dir = cfg["data_dir"]
    clip_local = cfg.get("clip_local", "")

    from .utils import read_jsonl
    rows = read_jsonl(image_list_path(data_dir))
    if not rows:
        log.info("没有发现图片列表（images.jsonl）。请先运行 ingest。")
        return

    model, preprocess = _load_openclip(clip_local)
    if model is None:
        log.warning("open_clip 不可用，跳过图片索引。")
        return

    try:
        import torch
    except Exception:
        log.warning("未安装 torch，跳过图片索引。")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    embs = []
    B = cfg.get("image_batch", 32)
    batch_imgs = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            path = r["path"]
            try:
                img = Image.open(path).convert("RGB")
            except UnidentifiedImageError:
                continue
            except Exception:
                continue
            batch_imgs.append(preprocess(img).unsqueeze(0))
            if len(batch_imgs) == B or i == len(rows)-1:
                x = torch.cat(batch_imgs, dim=0).to(device)
                with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                    feats = model.encode_image(x)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                embs.append(feats.cpu().float().numpy())
                batch_imgs = []
                if (i+1) % 200 == 0:
                    log.info(f"image encode 进度: {i+1}/{len(rows)}")

    if not embs:
        log.warning("没有生成任何图片向量。")
        return

    arr = np.concatenate(embs, axis=0).astype(np.float32)
    arr.tofile(image_emb_path(data_dir))
    log.info(f"图片索引完成：{arr.shape}")
