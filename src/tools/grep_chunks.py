# -*- coding: utf-8 -*-
import os, sys, argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kw = args.keyword
    hits = []
    paths = set()

    with open(args.chunks, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "")
            if kw in text:
                hits.append(obj)
                p = obj.get("path","")
                if p: paths.add(p)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for h in hits:
            fout.write(json.dumps(h, ensure_ascii=False) + "\n")

    # summary to console
    print(f"[grep] keyword='{kw}' hits={len(hits)} unique_files={len(paths)}")
    for i,p in enumerate(sorted(paths)):
        if i<20:
            print("  ", p)

if __name__ == "__main__":
    main()
