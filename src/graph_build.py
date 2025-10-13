import os, json

def run_graph(cfg, logger):
    data_dir = cfg["data_dir"]
    link_graph = os.path.join(data_dir, "link_graph.json")
    out_graph = os.path.join(data_dir, "graph.json")
    # 最小实现：用 link_graph 直接作为 GraphRAG 的图（节点/边）
    try:
        with open(link_graph, "r", encoding="utf-8") as f:
            g = json.load(f)
        with open(out_graph, "w", encoding="utf-8") as fo:
            json.dump(g, fo, ensure_ascii=False, indent=2)
        logger.info("graph.json 已生成（最小实现：拷贝 link_graph）。")
    except Exception as e:
        logger.warning(f"graph 构建失败：{e}")
