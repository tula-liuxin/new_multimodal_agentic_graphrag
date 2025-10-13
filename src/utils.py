import os, re, json, hashlib, logging, pathlib, sys, socket
from typing import Dict, Any
import yaml

ENV_PREFIX = "NMAGR_"

def norm_win_abs(p: str) -> str:
    # 将任意传入路径规范为绝对 Windows 路径（反斜杠）
    if not p:
        return p
    p = os.path.abspath(p)
    p = p.replace('/', '\\')
    # 若缺盘符（在 WSL/非 Win 环境），仍用反斜杠形式，用户在 Windows 运行会正确
    return p

def load_cfg(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    # 环境变量覆盖
    for k in list(cfg.keys()):
        env_key = ENV_PREFIX + k
        if env_key in os.environ:
            val = os.environ[env_key]
            # 简单转换：true/false/int/float
            if isinstance(cfg[k], bool):
                cfg[k] = str(val).lower() in ["1", "true", "yes", "y", "on"]
            elif isinstance(cfg[k], int):
                try: cfg[k] = int(val)
                except: cfg[k] = cfg[k]
            elif isinstance(cfg[k], float):
                try: cfg[k] = float(val)
                except: cfg[k] = cfg[k]
            else:
                cfg[k] = val
    # 规范路径
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


def win_long(path: str) -> str:
    # 为 Windows 启用长路径前缀，兼容 UNC
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
