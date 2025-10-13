import os, re, json
from bs4 import BeautifulSoup
from .file_readers import read_text_fallback
from tqdm import tqdm

MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

def parse_md_links(text: str):
    links = []
    for m in MD_LINK.finditer(text):
        title = m.group(1)
        url = m.group(2)
        links.append((title, url))
    return links

def parse_html_links(raw: str):
    soup = BeautifulSoup(raw, "lxml")
    out = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            out.append((a.get_text(" ", strip=True), href))
    return out

def run_links(cfg, logger):
    input_dir = cfg["input_dir"]
    data_dir = cfg["data_dir"]
    graph_path = os.path.join(data_dir, "link_graph.json")
    nodes = {}
    edges = []

    # 预扫描以显示进度
    file_list = []
    for dirpath, _, filenames in os.walk(input_dir):
        for fn in filenames:
            file_list.append(os.path.join(dirpath, fn))
    for p in tqdm(file_list, desc="links 扫描", unit="file"):
        rel = os.path.relpath(p, input_dir).replace("/", "\\")
        ext = os.path.splitext(p)[1].lower()
        links = []
        if ext in [".md",".markdown",".txt",".rst",".tex"]:
            raw = read_text_fallback(p)
            links = parse_md_links(raw)
        elif ext in [".html",".htm"]:
            raw = read_text_fallback(p)
            links = parse_html_links(raw)
        nodes[rel] = {"path": rel}
        for _, href in links:
            if "://" in href:
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(rel), href)).replace("/", "\\")
            edges.append({"source": rel, "target": target})
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
