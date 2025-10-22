import os, sys, json, argparse, yaml
from .session import ChatSession
from .parsing import parse_question
from .coref import rewrite_with_coref
from .retrieval import run_query_cli, build_turn_files, search_history
from .summarize import summarize_turn, summarize_chat_scan, summarize_conversation
from .utils import atomic_write_text as write_text

def load_cfg(path):
    with open(path,"r",encoding="utf-8") as f: 
        return yaml.safe_load(f) or {}

def main():
    ap=argparse.ArgumentParser(description="Chat v2 - fixed15")
    ap.add_argument("--config", required=True)
    ap.add_argument("--session", default="")
    a=ap.parse_args()
    cfg=load_cfg(os.path.abspath(a.config))
    print(f"[config] using: {os.path.abspath(a.config)}")
    print(f"[config] flags({len(cfg.get('query_cli_flags',[]))}): {cfg.get('query_cli_flags',[])}")

    sess=ChatSession(session_root=cfg.get("session_root","chat_sessions"),
                     session_name=(a.session or None),
                     naming=cfg.get("session_naming","timestamp"),
                     cfg=cfg)
    print(f"[Chat v2] Session at: {sess.session_dir}")

    while True:
        try: q=input("\n你：").strip()
        except (EOFError, KeyboardInterrupt): print("\n[Chat v2] Bye."); break
        if not q: continue
        if q.lower() in {"exit","quit","bye","q"}: print("[Chat v2] Bye."); break

        turn_dir=sess.new_turn_dir()
        print(f"[turn] {os.path.basename(turn_dir)}")

        last_focus=sess.get_focus()
        chat_summary=sess.read_chat_summary()
        last_turn_summary=sess.read_last_turn_summary()
        last_entities=sess.get_entities()

        parsed=parse_question(q, cfg, last_focus, chat_summary, last_turn_summary, last_entities)

        keywords=parsed.get("keywords",[])
        entities=parsed.get("entities",[])
        actions=parsed.get("actions",[])
        intent=parsed.get("intent","qa")

        if entities: sess.push_entities(entities)

        coref=parsed.get("coref") or {}
        focus_target=(coref.get("target") or (entities[0] if entities else "") or last_focus)
        if focus_target: sess.set_focus(focus_target)

        effective_question=rewrite_with_coref(q, coref, fallback_entities=sess.get_entities())

        sess.append_transcript({"role":"user","text":q,"parsed":parsed,"effective_question":effective_question})

        if intent=="summarize_conversation":
            answer_text=summarize_conversation(sess.session_dir, cfg, effective_question).strip() or "（未能生成会话总结）"
        else:
            rag=run_query_cli(effective_question, turn_dir, cfg)
            answer_text=""
            if rag.get("answer_txt") and os.path.exists(rag["answer_txt"]):
                answer_text=open(rag["answer_txt"],"r",encoding="utf-8").read().strip()
            if not answer_text:
                answer_text="（未获得 RAG 结果）"

        rows, rag_md=build_turn_files(turn_dir, cfg, entities=entities or sess.get_entities(), keywords=keywords)
        cleaned=open(rag_md,"r",encoding="utf-8").read() if os.path.exists(rag_md) else ""
        turn_summary=summarize_turn(answer_text, cleaned, cfg, question=effective_question)
        write_text(os.path.join(turn_dir,"turn_summary.txt"), turn_summary)

        chat_summary_new=summarize_chat_scan(sess.session_dir, cfg)
        sess.write_chat_summary(chat_summary_new)

        with open(os.path.join(turn_dir,"parsed.json"),"w",encoding="utf-8") as f:
            json.dump({"parsed":parsed,"keywords":keywords,"entities":entities,"effective_question":effective_question}, f, ensure_ascii=False, indent=2)

        write_text(os.path.join(turn_dir,"answer.txt"), answer_text)

        print("\n—— 回答 ——\n"); print(answer_text)
        print("\n—— 本轮摘要（turn_summary）——\n"); print(turn_summary)

        sess.append_transcript({"role":"assistant","text":answer_text,"turn_summary":turn_summary})

if __name__=='__main__':
    main()
