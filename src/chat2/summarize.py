from .utils import first_n_sentences
import glob, os

def summarize_turn(answer_text, cleaned_notes, cfg, question=""):
    header=f"Q：{question.strip()}" if question else ""
    merged="\n".join([p for p in [header,(answer_text or "").strip(),(cleaned_notes or "").strip()] if p])
    return first_n_sentences(merged,6)

def summarize_chat_scan(session_dir, cfg):
    paths=sorted(glob.glob(os.path.join(session_dir,"turn-*","turn_summary.txt")))
    items=[]
    for p in paths:
        try: items.append(open(p,"r",encoding="utf-8").read().strip())
        except: pass
    if not items: return ""
    keep=int(cfg.get("chat_summary_max_turns",24))
    items=items[-keep:]
    return first_n_sentences("\n".join(items),10)

def summarize_conversation(session_dir, cfg, question):
    return "（占位：会话总结，由模型生成；此版本保留入口）"
