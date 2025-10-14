
import os
from .utils import get_logger, load_cfg

def run_charindex(cfg=None, logger=None):
    log = logger or get_logger("charindex")
    cfg = cfg or load_cfg()
    mark = os.path.join(cfg["data_dir"], "charindex.done")
    with open(mark, "w") as f:
        f.write("ok")
    log.info("charindex 完成（占位实现）。")
