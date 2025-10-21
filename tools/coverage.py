import argparse, os, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seen = set()
    with open(a.chunks, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                sp = j.get("source_path") or j.get("path")
                if sp:
                    seen.add(sp)
            except Exception:
                pass

    all_files = []
    for dirpath, _, filenames in os.walk(a.root):
        for fn in filenames:
            if fn.lower().endswith((".html",".htm",".md",".txt",".pdf",".doc",".docx",".png",".jpg",".jpeg",".bmp",".gif",".webp",".tif",".tiff")):
                all_files.append(os.path.join(dirpath, fn))

    missing = [p for p in all_files if p not in seen]

    out = {"total": len(all_files), "covered": len(seen), "missing": len(missing), "missing_list": missing[:2000]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"covered": len(seen), "missing": len(missing)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
