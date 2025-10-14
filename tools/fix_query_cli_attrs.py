# -*- coding: utf-8 -*-
from pathlib import Path
import re

def patch_file(path: Path):
    txt = path.read_text("utf-8")

    txt = re.sub(r"args\.vl-answer", r"getattr(args, 'vl_answer', False)", txt)
    txt = re.sub(r"args\.text-first", r"getattr(args, 'text_first', 0)", txt)

    txt = re.sub(r"args\.disable-clip", r"getattr(args, 'disable_clip', False)", txt)
    txt = re.sub(r"args\.disable_clip", r"getattr(args, 'disable_clip', False)", txt)

    if "getattr(args, 'images_only', False)" not in txt:
        txt = re.sub(r"(def main\(\):\n\s+.*?args = parser\.parse_args\(\)\n)", 
                     r"\1    args.images_only = getattr(args, 'images_only', False)\n    args.vl_answer = getattr(args, 'vl_answer', False)\n    args.text_first = getattr(args, 'text_first', 0)\n",
                     txt, flags=re.DOTALL)

    path.write_text(txt, "utf-8")

def main():
    q = Path("C:\\MyNotion\\new_multimodal_agentic_graphrag\\src\\query_cli.py")
    if not q.exists():
        print("未找到", q)
        return
    patch_file(q)
    print("已修补", q)

if __name__ == "__main__":
    main()
