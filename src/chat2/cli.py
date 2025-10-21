import os, sys, json, argparse, yaml
from typing import Dict, Any, List
from .session import ChatSession
from .parsing import parse_question
from .coref import rewrite_with_coref
from .retrieval import run_query_cli, build_turn_files, search_history, ASI_ALIASES
from .summarize import summarize_turn, summarize_chat_scan, summarize_conversation
from .utils import write_text

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def maybe_append_asi_synonyms(q: str, keywords: List[str], actions: List[str]) -> str:
    need_retrieve = any(a in {"retrieve","rag"} for a in actions or [])
    if need_retrieve and any(k.lower() == "asi" for k in (keywords or [])):
        hint = "（同义词：ASI、asi、人工超智能、超人工智能、人工通用智能）"
        if hint not in q:
            q = q + hint
    return q

def main():
    ap = argparse.ArgumentParser(description="Chat v2 - multi-turn shell (fixed14)")
    ap.add_argument("--config", required=True, help="path to chat2_config.yaml")
    ap.add_argument("--session", default="", help="optional session name (prefix)")
    ap.add_argument("--max-turns", type=int, default=0, help="max turns (0 = unlimited)")
    args = ap.parse_args()

    cfg_path_abs = os.path.abspath(args.config)
    cfg = load_cfg(cfg_path_abs)
    print(f"[config] using: {cfg_path_abs}")
    eff_flags = cfg.get("query_cli_flags", [])
    print(f"[config] flags({len(eff_flags)}): {eff_flags}")

    session_root = cfg.get("session_root", "chat_sessions")
    sess = ChatSession(session_root=session_root, session_name=(args.session or None),
                       naming=cfg.get("session_naming","timestamp"))
    print(f"[Chat v2] Session at: {sess.session_dir}")

    turns = 0
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat v2] Bye.")
            break
        if not q:
            continue
        if q.lower() in {"exit","quit","bye","q"}:
            print("[Chat v2] Bye.")
            break

        turn_dir = sess.new_turn_dir()
        print(f"[turn] {os.path.basename(turn_dir)}")

        last_focus = sess.get_focus()
        chat_summary = sess.read_chat_summary()
        last_turn_summary = sess.read_last_turn_summary()
        last_entities = sess.get_entities()

        parsed = parse_question(q, cfg, last_focus=last_focus, chat_summary=chat_summary, last_turn_summary=last_turn_summary, last_entities=last_entities)

        need_rag_cfg = bool(cfg.get("default_enable_rag", True))
        need_hist_cfg = bool(cfg.get("default_enable_history_search", True))
        need_rag = bool(parsed.get("need_rag", need_rag_cfg))
        need_hist = bool(parsed.get("need_history", need_hist_cfg))
        keywords: List[str] = parsed.get("keywords", [])
        entities: List[str] = parsed.get("entities", [])
        actions: List[str] = parsed.get("actions", [])
        intent: str = parsed.get("intent", "qa")

        if entities:
            sess.push_entities(entities)

        coref = parsed.get("coref") or {}
        focus_target = (coref.get("target") or (entities[0] if entities else "") or last_focus)
        if focus_target:
            sess.set_focus(focus_target)

        effective_question = rewrite_with_coref(q, coref, fallback_entities=sess.get_entities())
        effective_question = maybe_append_asi_synonyms(effective_question, keywords, actions)

        sess.append_transcript({"role": "user", "text": q, "parsed": parsed, "effective_question": effective_question})

        if intent == "summarize_conversation":
            answer_text = summarize_conversation(sess.session_dir, cfg, question=effective_question).strip()
            if not answer_text:
                answer_text = "（未能生成会话总结）"
            write_text(os.path.join(turn_dir, "answer.txt"), answer_text)
            turn_files_rows, rag_md = build_turn_files(turn_dir, cfg, entities=entities or sess.get_entities(), keywords=keywords)
            cleaned_notes = ""
            if rag_md and os.path.exists(rag_md):
                cleaned_notes = open(rag_md, "r", encoding="utf-8").read()
            turn_summary = summarize_turn(answer_text, cleaned_notes, cfg, question=effective_question)
            write_text(os.path.join(turn_dir, "turn_summary.txt"), turn_summary)
            chat_summary_new = summarize_chat_scan(sess.session_dir, cfg)
            sess.write_chat_summary(chat_summary_new)

            stage_paths = {
                "turn_dir": turn_dir,
                "effective_question": effective_question,
                "answer_txt": os.path.join(turn_dir, "answer.txt"),
                "returncode": 0,
                "python_exec": sys.executable,
                "built_cmd": "(conversation summarizer; no query_cli)",
                "filtered_json": os.path.join(turn_dir, "filtered.json") if os.path.exists(os.path.join(turn_dir, "filtered.json")) else None,
                "candidates_json": os.path.join(turn_dir, "candidates.json") if os.path.exists(os.path.join(turn_dir, "candidates.json")) else None,
                "stage0_keywords": os.path.join(turn_dir, "stage0_keywords.json") if os.path.exists(os.path.join(turn_dir, "stage0_keywords.json")) else None,
                "console_log": os.path.join(turn_dir, "console.log") if os.path.exists(os.path.join(turn_dir, "console.log")) else None,
                "turn_files_jsonl": os.path.join(turn_dir, "turn_files", "rag_clean.jsonl"),
                "turn_files_md": os.path.join(turn_dir, "turn_files", "rag_clean.md"),
                "turn_summary": os.path.join(turn_dir, "turn_summary.txt"),
                "chat_summary": sess.summary_path,
                "transcript": sess.transcript_path,
                "parsed_json": os.path.join(turn_dir, "parsed.json"),
            }
            with open(os.path.join(turn_dir, "stage_paths.json"), "w", encoding="utf-8") as f:
                json.dump(stage_paths, f, ensure_ascii=False, indent=2)

            print("\n—— 回答 ——\n")
            print(answer_text)
            print("\n—— 中间产物路径 ——\n")
            for k, v in stage_paths.items():
                print(f"{k}: {v}")
            print("\n—— 本轮摘要（turn_summary）——\n")
            print(turn_summary)

            sess.append_transcript({"role": "assistant", "text": answer_text, "turn_summary": turn_summary})
            turns += 1
            if args.max_turns and turns >= args.max_turns:
                print("[Chat v2] 达到最大轮数，结束。")
                break
            continue

        rag: Dict[str, Any] = {}
        if cfg.get("force_rag", True) or need_rag:
            rag = run_query_cli(effective_question, turn_dir, cfg)

        hist_ctx: Dict[str, Any] = {}
        if need_hist:
            hist_ctx = search_history(sess.session_dir, keywords, cfg)

        turn_files_rows, rag_md = build_turn_files(turn_dir, cfg, entities=entities or sess.get_entities(), keywords=keywords)

        answer_text = ""
        if rag.get("answer_txt") and os.path.exists(rag["answer_txt"]):
            answer_text = open(rag["answer_txt"], "r", encoding="utf-8").read().strip()

        if (not answer_text) and cfg.get("require_rag_success", True):
            blocks = []
            for i, r in enumerate(turn_files_rows[:8], 1):
                blocks.append(f"[{i}] {r.get('text','')[:400]}")
            if blocks:
                answer_text = "根据检索片段整理：\n" + "\n".join(blocks)

        if not answer_text:
            buf = ["（未获得 RAG 生成结果，以下为历史命中片段）"]
            for h in hist_ctx.get("turn_files", [])[:5]:
                buf.append(f"[{h.get('where')}] {h.get('text','')[:400]}")
            answer_text = "\n".join(buf).strip()

        cleaned_notes = ""
        if rag_md and os.path.exists(rag_md):
            cleaned_notes = open(rag_md, "r", encoding="utf-8").read()

        turn_summary = summarize_turn(answer_text, cleaned_notes, cfg, question=effective_question)
        write_text(os.path.join(turn_dir, "turn_summary.txt"), turn_summary)

        chat_summary_new = summarize_chat_scan(sess.session_dir, cfg)
        sess.write_chat_summary(chat_summary_new)

        with open(os.path.join(turn_dir, "parsed.json"), "w", encoding="utf-8") as f:
            json.dump({
                "parsed": parsed,
                "keywords": keywords,
                "entities": entities,
                "effective_question": effective_question,
                "used_history": hist_ctx,
                "focus": sess.get_focus(),
                "last_entities": sess.get_entities()
            }, f, ensure_ascii=False, indent=2)

        write_text(os.path.join(turn_dir, "answer.txt"), answer_text)

        stage_paths = {
            "turn_dir": turn_dir,
            "effective_question": effective_question,
            "answer_txt": rag.get("answer_txt"),
            "returncode": rag.get("returncode"),
            "python_exec": rag.get("python_exec"),
            "built_cmd": rag.get("built_cmd"),
            "filtered_json": rag.get("filtered_json"),
            "candidates_json": rag.get("candidates_json"),
            "stage0_keywords": rag.get("stage0_keywords"),
            "console_log": rag.get("console_log"),
            "turn_files_jsonl": os.path.join(turn_dir, "turn_files", "rag_clean.jsonl"),
            "turn_files_md": os.path.join(turn_dir, "turn_files", "rag_clean.md"),
            "turn_summary": os.path.join(turn_dir, "turn_summary.txt"),
            "chat_summary": sess.summary_path,
            "transcript": sess.transcript_path,
            "parsed_json": os.path.join(turn_dir, "parsed.json"),
        }
        with open(os.path.join(turn_dir, "stage_paths.json"), "w", encoding="utf-8") as f:
            json.dump(stage_paths, f, ensure_ascii=False, indent=2)

        print("\n—— 回答 ——\n")
        print(answer_text)

        print("\n—— 中间产物路径 ——\n")
        for k, v in stage_paths.items():
            print(f"{k}: {v}")

        print("\n—— 本轮摘要（turn_summary）——\n")
        print(turn_summary)

        sess.append_transcript({"role": "assistant", "text": answer_text, "turn_summary": turn_summary})

        turns += 1
        if args.max_turns and turns >= args.max_turns:
            print("[Chat v2] 达到最大轮数，结束。")
            break

if __name__ == "__main__":
    main()
