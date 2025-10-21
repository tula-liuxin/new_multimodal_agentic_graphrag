import argparse, os, sys, json, subprocess
from pathlib import Path
from typing import List
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from .ollama_client import OllamaClient
from .intent_router import route_intent
from .chat_record_manager import (
    start_chat_session, turn_dir, append_transcript,
    write_turn_summary, write_turn_payload, update_session_summary
)
from .rag_b import RAGBIndex
from .query_normalizer import normalize_query
from .query_planner import plan_query

EXIT_CMDS = {"/exit", "exit", "/quit", "quit", ":q", "q", "/q"}

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding='utf-8'))

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _now_ts():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def summarize_text(client: OllamaClient, model: str, history_text: str, sys_prompt_file: Path | None) -> str:
    sys_prompt = ""
    if sys_prompt_file and sys_prompt_file.exists():
        sys_prompt = sys_prompt_file.read_text(encoding="utf-8")
    prompt = f"""你是对话摘要助手。请在不丢失关键信息的前提下，给出精炼、要点化中文摘要（可分点）。
文本：
{history_text}
"""
    resp = client.generate(model=model, prompt=(sys_prompt + "\n\n" + prompt).strip(), stream=False)
    return resp.get("response") or resp.get("message") or ""

def build_session_context(session_dir: Path, keep_last_turns: int = 20, max_chars: int = 20000) -> str:
    turns = sorted([p for p in session_dir.glob("turn_*") if p.is_dir()])
    turns = turns[-keep_last_turns:]
    buf = []
    for td in turns:
        u = (td/"user.txt").read_text(encoding="utf-8") if (td/"user.txt").exists() else ""
        a = (td/"answer.txt").read_text(encoding="utf-8") if (td/"answer.txt").exists() else ""
        s = (td/"summary.txt").read_text(encoding="utf-8") if (td/"summary.txt").exists() else ""
        buf.append(f"[User]\n{u}\n[Assistant]\n{a}\n[TurnSummary]\n{s}\n")
    text = "\n\n".join(buf)[-max_chars:]
    return text

def run_rag_a(question: str, turn_dir_path: Path, rag_a_flags: List[str]):
    cmd = [sys.executable, "-m", "src.query_cli"] + rag_a_flags + ["--debug-dir", str(turn_dir_path.resolve()), question]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
    answer_file = turn_dir_path / "answer.txt"
    if answer_file.exists():
        return answer_file.read_text(encoding="utf-8")
    return proc.stdout or proc.stderr

def main():
    ap = argparse.ArgumentParser(description="RAG_A + RAG_B 多轮对话编排器")
    ap.add_argument("--config", default="chat_orchestrator.yaml", help="配置文件路径（相对项目根或绝对路径）")
    ap.add_argument("--session", default=None, help="可选：自定义会话名（默认=时间戳_chat）")
    args = ap.parse_args()

    project_root = _project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_yaml(cfg_path)

    ollama_host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    decision_model = cfg.get("decision_model", "qwen3:4b-instruct-2507-fp16")
    summarizer_model = cfg.get("summarizer_model", "qwen3:4b-instruct-2507-fp16")
    embed_model = cfg.get("embed_model", "bge-m3:latest")
    chat_base_dir = cfg.get("chat_base_dir", "chat_record")
    normalizer_model = cfg.get("normalizer",{}).get("model", summarizer_model)
    normalizer_prompt = cfg.get("normalizer",{}).get("prompt_file", str(project_root / "prompts" / "query_normalizer_zh.txt"))
    planner_model = cfg.get("planner",{}).get("model", summarizer_model)
    planner_prompt = cfg.get("planner",{}).get("prompt_file", str(project_root / "prompts" / "plan_router_zh.txt"))

    session_dir = start_chat_session(project_root, chat_base_dir, args.session)
    print(f"[INFO] New session: {session_dir}")
    print("[HINT] 输入 /exit 退出。")

    ragb = RAGBIndex(chat_root=(project_root / chat_base_dir), embed_model=embed_model, ollama_host=ollama_host)

    client = OllamaClient(host=ollama_host, timeout=90)
    intent_prompt_file = (project_root / "prompts" / "intent_router_zh.txt")
    summary_sys_prompt = (project_root / "prompts" / "summary_system_zh.txt") if (project_root / "prompts" / "summary_system_zh.txt").exists() else None

    turn_idx = 0
    while True:
        try:
            user_q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exit.")
            break
        if not user_q:
            print("[INFO] Empty input, exit.")
            break
        if user_q.lower() in EXIT_CMDS:
            print("[INFO] Exit by command.")
            break

        turn_idx += 1
        td = turn_dir(session_dir, turn_idx)
        (td/"user.txt").write_text(user_q, encoding="utf-8")

        print("[STAGE] route ...", end="", flush=True)
        intent = route_intent(user_q, ollama_host=ollama_host, model=decision_model, prompt_file=str(intent_prompt_file))
        print(" ok")

        print("[STAGE] normalize query ...", end="", flush=True)
        clean_q = normalize_query(user_q, ollama_host=ollama_host, model=normalizer_model, prompt_file=normalizer_prompt)
        (td/"clean_query.txt").write_text(clean_q, encoding="utf-8")
        print(f" ok ({clean_q})")

        print("[STAGE] plan ...", end="", flush=True)
        plan = plan_query(user_q, clean_q, ollama_host=ollama_host, model=planner_model, prompt_file=planner_prompt)
        (td/"planned_query.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        planned_q_str = plan.get("planned_query") or clean_q
        (td/"planned_query.txt").write_text(planned_q_str, encoding="utf-8")
        print(f" ok ({planned_q_str})")

        print("[STAGE] RAG_B update ...", end="", flush=True)
        ragb.scan_and_append_new()
        ragb.embed_incremental()
        print(" ok")

        answer = ""
        used = {"rag_a": False, "rag_b": False}
        ctx_for_answer = ""

        # 由 planner 决定是否用 RAG_B / RAG_A
        if plan.get("use_rag_b"):
            print("[STAGE] RAG_B search ...", end="", flush=True)
            used["rag_b"] = True
            scope = cfg.get("rag_b",{}).get("scope","session")
            session_name = session_dir.name if scope == "session" else None
            top_k = int(cfg.get("rag_b",{}).get("top_k",8))
            # 优先使用 planner 的搜索词
            q_b = " ".join(plan.get("rag_b",{}).get("search_terms", [clean_q])) or clean_q
            hits = ragb.search(q_b, top_k=top_k, scope=scope, session_name=session_name)
            (td/"rag_b_hits.json").write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f" ok ({len(hits)} hits)")

            if hits:
                ctx_for_answer = "\n\n".join([f"[{h.get('title')}]\n{h.get('text')}" for h in hits])
            else:
                print("[STAGE] RAG_B no hits, using recent session context ...", end="", flush=True)
                ctx_for_answer = build_session_context(session_dir, keep_last_turns=int(cfg.get("summary",{}).get("keep_last_turns",20)), max_chars=int(cfg.get("summary",{}).get("memory_max_chars",20000)))
                print(" ok")

        if plan.get("use_rag_a"):
            print("[STAGE] RAG_A query ...", end="", flush=True)
            used["rag_a"] = True
            rag_a_flags_cfg = cfg.get("rag_a", {}).get("query_cli_flags", [])
            flat_flags: List[str] = []
            for item in rag_a_flags_cfg:
                if isinstance(item, list):
                    flat_flags.extend([str(x) for x in item])
                else:
                    flat_flags.append(str(item))
            # 用 planner 的合成检索串
            a = run_rag_a(planned_q_str, td, flat_flags)
            if not answer or intent.get("operation") in ("answer","both"):
                answer = a
            print(" ok")

        if not (used["rag_a"] or used["rag_b"]):
            print("[STAGE] LLM direct answer ...", end="", flush=True)
            client = OllamaClient(host=ollama_host, timeout=60)
            resp = client.generate(model=summarizer_model, prompt=user_q, stream=False)
            answer = resp.get("response") or resp.get("message") or ""
            print(" ok")

        if ctx_for_answer and plan.get("use_rag_b"):
            # 将基于 RAG_B 的上下文与 RAG_A 的答案融合（简要整合）
            client = OllamaClient(host=ollama_host, timeout=60)
            fuse_prompt = f"""基于聊天上下文与外部检索结果，给出最终回答：
【用户问题】
{user_q}

【规划关键词】
{', '.join(plan.get('keywords', []))}

【聊天上下文/检索片段】
{ctx_for_answer}

【外部检索回答（如有）】
{answer}
"""
            resp2 = client.generate(model=summarizer_model, prompt=fuse_prompt, stream=False)
            merged = resp2.get("response") or resp2.get("message") or ""
            if merged:
                answer = merged

        (td/"answer.txt").write_text(answer, encoding="utf-8")

        print("[STAGE] turn summary ...", end="", flush=True)
        session_ctx = build_session_context(
            session_dir,
            keep_last_turns=int(cfg.get("summary",{}).get("keep_last_turns",20)),
            max_chars=int(cfg.get("summary",{}).get("memory_max_chars",20000))
        )
        client = OllamaClient(host=ollama_host, timeout=90)
        turn_summary = summarize_text(
            client, summarizer_model,
            f"[当前问] {user_q}\n[清洗查询] {clean_q}\n[规划串] {planned_q_str}\n[当前答]\n{answer}\n\n[上下文]\n{session_ctx}",
            sys_prompt_file=( _project_root() / "prompts" / "summary_system_zh.txt")
        )
        write_turn_summary(td, turn_summary)
        print(" ok")

        print("[STAGE] session summary ...", end="", flush=True)
        old_session_summary = (session_dir/"summary.txt").read_text(encoding="utf-8") if (session_dir/"summary.txt").exists() else ""
        client = OllamaClient(host=ollama_host, timeout=90)
        session_summary = summarize_text(
            client, summarizer_model,
            f"[旧摘要]\n{old_session_summary}\n\n[新增轮摘要]\n{turn_summary}",
            sys_prompt_file=( _project_root() / "prompts" / "summary_system_zh.txt")
        )
        update_session_summary(session_dir, session_summary)
        print(" ok")

        rec = {
            "ts": _now_ts(),
            "turn": turn_idx,
            "user": user_q,
            "clean_query": clean_q,
            "planned_query": planned_q_str,
            "plan": plan,
            "answer": answer,
            "turn_summary": turn_summary,
            "used": used,
        }
        append_transcript(session_dir, rec)

        print(f"\n助手: {answer}\n")
        print(f"[turn_{turn_idx:05d} 摘要已写入] {td/'summary.txt'}")

        print("[STAGE] RAG_B refresh ...", end="", flush=True)
        ragb.scan_and_append_new()
        ragb.embed_incremental()
        print(" ok")

    print(f"[INFO] 会话摘要: {(session_dir/'summary.txt').resolve()}")
    print(f"[INFO] 会话转写: {(session_dir/'transcript.jsonl').resolve()}")

if __name__ == "__main__":
    main()
