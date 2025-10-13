import os, re, json, hashlib, logging, pathlib, sys, socket
from typing import Dict, Any
import yaml

ENV_PREFIX = "NMAGR_"

def norm_win_abs(p: str) -> str:
    if not p:
        return p
    p = os.path.abspath(p)
    p = p.replace('/', '\\')
    return p

def win_long(path: str) -> str:
    # Windows 长路径支持
    if os.name != "nt":
        return path
    if not path:
        return path
    p = os.path.abspath(path)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):  # UNC
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p

def load_cfg(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    for k in list(cfg.keys()):
        env_key = ENV_PREFIX + k
        if env_key in os.environ:
            val = os.environ[env_key]
            if isinstance(cfg[k], bool):
                cfg[k] = str(val).lower() in ["1","true","yes","y","on"]
            elif isinstance(cfg[k], int):
                try: cfg[k] = int(val)
                except: pass
            elif isinstance(cfg[k], float):
                try: cfg[k] = float(val)
                except: pass
            else:
                cfg[k] = val
    for key in ["project_root", "input_dir", "data_dir", "debug_dir"]:
        if key in cfg:
            cfg[key] = norm_win_abs(cfg[key])
    return cfg

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_logger(name: str = "nmagr", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', errors='ignore')).hexdigest()

def is_port_open(host="127.0.0.1", port=11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False
