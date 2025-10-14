
import argparse
from .utils import get_logger, load_cfg
from .ingest import run_ingest
from .embed_index import run_embed
from .image_encoder import run_imgindex
from .char_index import run_charindex
from .link_graph import run_links

def main():
    log = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["ingest", "embed", "imgindex", "charindex", "links", "all"], help="要执行的任务")
    args = parser.parse_args()

    log.info(f"Running task: {args.task}")
    cfg = load_cfg()

    if args.task == "ingest":
        run_ingest(cfg, log)
    elif args.task == "embed":
        run_embed(cfg, log)
    elif args.task == "imgindex":
        run_imgindex(cfg, log)
    elif args.task == "charindex":
        run_charindex(cfg, log)
    elif args.task == "links":
        run_links(cfg, log)
    elif args.task == "all":
        run_ingest(cfg, log)
        run_embed(cfg, log)
        run_imgindex(cfg, log)
        run_charindex(cfg, log)
        run_links(cfg, log)

if __name__ == "__main__":
    main()
