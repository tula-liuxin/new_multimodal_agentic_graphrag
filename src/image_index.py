# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json
from typing import List, Tuple, Set
import numpy as np
from PIL import Image

import torch
import open_clip
from tqdm import tqdm

from .utils import jsonl_read, to_io_path

def _autocast_ctx(device: str):
    """Return a context manager for autocast that is future-proof."""
    try:
        from torch.amp import autocast as amp_autocast
        return amp_autocast(device if device in ("cuda","cpu") else "cuda")
    except Exception:
        return torch.cuda.amp.autocast(enabled=(device=="cuda"))

def _load_model(clip_local: str|None, device: str):
    model_name = "ViT-L-14"
    pretrained = clip_local if (clip_local and os.path.exists(clip_local)) else "laion2b_s32b_b82k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess

def _safe_open_rgb(path: str) -> Image.Image:
    im = Image.open(to_io_path(path))
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def _read_image_paths(images_jsonl: str) -> List[str]:
    rows = jsonl_read(images_jsonl) if os.path.exists(images_jsonl) else []
    return [r.get("path") for r in rows if r.get("path")]

def _load_prev(index_dir: str):
    try:
        vec = np.load(os.path.join(index_dir, "image_vecs.npy"))
        ids = json.load(open(os.path.join(index_dir, "image_ids.json"), "r", encoding="utf-8"))
        return vec, ids
    except Exception:
        return np.zeros((0, 768), dtype=np.float32), []

def build_image_index(images_jsonl: str,
                      index_dir: str = "index",
                      clip_local: str|None = None,
                      device: str = "cuda",
                      logger=None,
                      batch: int = 32,
                      incremental: bool = False,
                      delta_json: str = "data/delta.json") -> Tuple[str, str]:
    os.makedirs(index_dir, exist_ok=True)
    paths = _read_image_paths(images_jsonl)
    if logger: logger.info(f"图像向量：共 {len(paths)} 张，device={device}, batch={batch}")

    prev_vecs, prev_ids = _load_prev(index_dir)
    prev_map = {pid:i for i,pid in enumerate(prev_ids)}
    changed_paths: Set[str] = set()
    removed_paths: Set[str] = set()

    if incremental and os.path.exists(delta_json):
        d = json.load(open(delta_json, "r", encoding="utf-8"))
        changed_paths = set(d.get("changed_paths", []))
        removed_paths = set(d.get("deleted_paths", []))

    try:
        model, preprocess = _load_model(clip_local, device)
    except Exception as e:
        if logger: logger.warning(f"OpenCLIP 加载失败（降级 CPU/预处理简单转换）：{e}")
        device = "cpu"
        model, preprocess = _load_model(None, device)

    # 需要编码的列表
    need_encode = paths if not incremental else [p for p in paths if (p in changed_paths or p not in prev_map)]
    # 待删除 id
    to_delete = set(pid for pid in prev_ids if pid not in paths or pid in removed_paths)

    all_vecs: List[np.ndarray] = []
    all_ids: List[str] = []

    # 先保留旧的、且不在删除名单、且不在更新名单的
    keep_old = []
    keep_ids = []
    for pid, idx in zip(prev_ids, range(len(prev_ids))):
        if (pid in to_delete) or (pid in set(need_encode)):
            continue
        keep_old.append(prev_vecs[idx:idx+1])
        keep_ids.append(pid)

    if keep_old:
        all_vecs.append(np.concatenate(keep_old, axis=0))
        all_ids.extend(keep_ids)

    # 新计算
    for i in tqdm(range(0, len(need_encode), batch), desc='[图像索引] 批处理', unit='batch'):
        batch_paths = need_encode[i:i+batch]
        imgs = []
        ok_ids = []
        for p in tqdm(batch_paths, desc='[图像索引] 加载预处理', leave=False):
            try:
                im = _safe_open_rgb(p)
                imgs.append(preprocess(im))
                ok_ids.append(p)
            except Exception as ex:
                if logger: logger.warning(f"图像读取失败（跳过）: {p} | {ex}")
        if not imgs: continue
        ims = torch.stack(imgs).to(device)
        with torch.no_grad(), _autocast_ctx(device):
            feats = model.encode_image(ims)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs = feats.detach().cpu().numpy().astype(np.float32)
        all_vecs.append(vecs); all_ids.extend(ok_ids)

    if all_vecs:
        arr = np.concatenate(all_vecs, axis=0)
    else:
        arr = np.zeros((0, 768), dtype=np.float32)

    vec_path = os.path.join(index_dir, "image_vecs.npy")
    ids_path = os.path.join(index_dir, "image_ids.json")
    np.save(vec_path, arr)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False)

    if logger: logger.info(f"图像索引完成：vecs={arr.shape}, ids={len(all_ids)} → {vec_path}")
    return vec_path, ids_path
