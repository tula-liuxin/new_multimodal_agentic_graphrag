import argparse, os
from .utils import load_cfg, get_logger
from .ingest import run_ingest
from .embed_index import run_embed
from .tfidf_index import run_tfidf
from .char_index import run_char_tfidf
from .links import run_links
from .graph_build import run_graph
from .image_encoder import run_imgindex
from .audit import run_audit
from .imgaudit import run_imgaudit

def main():
    parser = argparse.ArgumentParser(description="new_multimodal_agentic_graphrag: CLI 入口")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="采集/切片")
    p_ing.add_argument("--config", default="config.yaml")

    p_emb = sub.add_parser("embed", help="文本嵌入（Ollama）")
    p_emb.add_argument("--config", default="config.yaml")

    p_img = sub.add_parser("imgindex", help="图像索引（OpenCLIP）")
    p_img.add_argument("--config", default="config.yaml")

    p_tfidf = sub.add_parser("charindex", help="词面/字 n-gram TF-IDF 索引")
    p_tfidf.add_argument("--config", default="config.yaml")

    p_links = sub.add_parser("links", help="解析 .md/.html 链接")
    p_links.add_argument("--config", default="config.yaml")

    p_graph = sub.add_parser("graph", help="构建 GraphRAG 最小图谱")
    p_graph.add_argument("--config", default="config.yaml")

    p_audit = sub.add_parser("audit", help="文本审计")
    p_audit.add_argument("--config", default="config.yaml")
    p_audit.add_argument("--term", required=True)
    p_audit.add_argument("--alt", default=None)
    p_audit.add_argument("--trad", action="store_true")

    p_iaudit = sub.add_parser("imgaudit", help="图像审计")
    p_iaudit.add_argument("--config", default="config.yaml")
    p_iaudit.add_argument("--term", required=True)
    p_iaudit.add_argument("--mode", choices=["ocr","caption","clip"], default="clip")

    args = parser.parse_args()
    cfg = load_cfg(args.config)
    logger = get_logger()

    if args.cmd == "ingest":
        run_ingest(cfg, logger)
    elif args.cmd == "embed":
        run_embed(cfg, logger)
    elif args.cmd == "imgindex":
        run_imgindex(cfg, logger)
    elif args.cmd == "charindex":
        run_tfidf(cfg, logger)
        run_char_tfidf(cfg, logger)
    elif args.cmd == "links":
        run_links(cfg, logger)
    elif args.cmd == "graph":
        run_graph(cfg, logger)
    elif args.cmd == "audit":
        run_audit(cfg, logger, term=args.term, alt=args.alt, trad=args.trad)
    elif args.cmd == "imgaudit":
        run_imgaudit(cfg, logger, term=args.term, mode=args.mode)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
