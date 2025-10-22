import os, json, tempfile

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def atomic_write_text(path, text):
    ensure_dir(os.path.dirname(path))
    fd,tmp=tempfile.mkstemp(prefix=".tmp_",suffix=".txt",dir=os.path.dirname(path))
    with os.fdopen(fd,"w",encoding="utf-8") as f: f.write(text or "")
    os.replace(tmp,path)

def atomic_write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    fd,tmp=tempfile.mkstemp(prefix=".tmp_",suffix=".json",dir=os.path.dirname(path))
    with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2)
    os.replace(tmp,path)

def append_jsonl(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path,"a",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def first_n_sentences(text, n=3):
    import re; parts=re.split(r"(?:。|！|!|？|\?|\.|\n)", text or ""); out=[]
    for p in parts:
        p=p.strip()
        if p: out.append(p)
        if len(out)>=n: break
    return "。".join(out) if out else (text or "").strip()
