# -*- coding: utf-8 -*-
import os, sys, argparse, json, re

def list_html(root):
    htmls = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if re.search(r"\.(html?|htm)$", f, re.I):
                htmls.append(os.path.join(dirpath, f))
    return set(map(os.path.abspath, htmls))

def read_chunks_paths(chunks_path):
    paths = set()
    if not os.path.exists(chunks_path):
        return paths
    with open(chunks_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                p = obj.get("path") or obj.get("src") or ""
                if p: paths.add(os.path.abspath(p))
            except Exception:
                pass
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_html = list_html(args.root)
    covered = read_chunks_paths(args.chunks)

    missing = sorted(all_html - covered)
    report = {
        "total_html": len(all_html),
        "with_chunks": len(covered & all_html),
        "missing": missing[:1000],  # cap for readability
        "missing_count": len(missing),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)
    print(f"[coverage] total={report['total_html']} with_chunks={report['with_chunks']} missing={report['missing_count']}")
    if report["missing_count"]>0:
        print("[coverage] sample missing:")
        for p in report["missing"][:10]:
            print("  ", p)

if __name__ == "__main__":
    main()
