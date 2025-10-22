#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版多轮对话脚本：
1. 每轮先调用 LLM（按配置）解析意图/动作、管理多轮记忆。
2. 根据解析结果判断是否触发 RAG（query_cli.py）或历史检索（跨会话）。
3. 每轮写出 turn 目录 + round_xx_summary / round_xx_rag 兼容文件。

运行示例：
    python scripts/multi_round_query_chat.py --session demo_chat
          （可选 --config chat2_config.yaml --max-rounds 30）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat2.session import ChatSession
from src.chat2.parsing import parse_question
from src.chat2.coref import rewrite_with_coref
from src.chat2.retrieval import run_query_cli, build_turn_files
from src.chat2.summarize import summarize_turn, summarize_chat_scan
from src.ollama_client import OllamaClient
from src.rag_b import RAGBIndex


DEFAULT_FLAGS = [
    "--mode", "hybrid",
    "--smart-query",
    "--sqw", "1.2",
    "--text-first", "6",
    "--wtext", "1.0",
    "--wimg", "0.25",
    "--wbool", "0.7",
    "--wfuzzy", "0.6",
    "--neighbors", "1",
    "--folder-neighbors", "2",
    "--link-hop",
    "--link-depth", "1",
    "--expand-parent",
    "--sibling-span", "1",
    "--post-filter",
    "--pf-model", "qwen2.5:3b-instruct",
    "--pf-workers", "6",
    "--pf-timeout", "15",
    "--pf-min-keep", "6",
    "--top", "2",
    "--num-ctx", "20000",
    "--ctx-chars", "1200",
    "--embed-model", "bge-m3:latest",
    "--gen-timeout", "180000",
    "--temperature", "0.2",
    "--llm-query-clean",
    "--qc-model", "qwen3:4b-instruct-2507-fp16",
    "--qc-timeout", "90",
    "--qc-max-keys", "8",
    "--qc-prompt-file", "prompts/keyword_extractor_zh.txt",
    "--llm-num-ctx", "48192",
    "--gen-max-tokens", "12000",
    "--ollama-host", "http://127.0.0.1:11434",
    "--model", "qwen3:4b-instruct-2507-fp16",
]

SMALLTALK_PROMPT = """\
你是对话助手。请在尊重上下文的基础上用中文简洁回复用户。
- 如提供了“历史上下文”，请挑选与当前问题最相关的信息。
- 如果上下文中没有答案，可以礼貌说明并询问是否需要检索知识库。
- 回复不超过 8 句，保持口吻自然。
"""


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_flags(flags_cfg: Optional[List[Any]]) -> List[str]:
    if not flags_cfg:
        return DEFAULT_FLAGS
    merged: List[str] = []
    for item in flags_cfg:
        if isinstance(item, str):
            merged.append(item)
        elif isinstance(item, list):
            merged.extend(str(x) for x in item)
        else:
            merged.append(str(item))
    return merged


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def gather_session_context(session: ChatSession, keep_turns: int = 3) -> str:
    data = []
    session_path = Path(session.session_dir)
    turn_dirs = sorted([d for d in session_path.glob("turn-*") if d.is_dir()])
    turn_dirs = turn_dirs[-keep_turns:]
    for td in turn_dirs:
        user_txt = (td / "user.txt").read_text("utf-8") if (td / "user.txt").exists() else ""
        ans_txt = (td / "answer.txt").read_text("utf-8") if (td / "answer.txt").exists() else ""
        summ_txt = (td / "turn_summary.txt").read_text("utf-8") if (td / "turn_summary.txt").exists() else ""
        if not any([user_txt.strip(), ans_txt.strip(), summ_txt.strip()]):
            continue
        data.append(f"[User]\n{user_txt.strip()}\n[Assistant]\n{ans_txt.strip()}\n[TurnSummary]\n{summ_txt.strip()}")
    chat_summary = session.read_chat_summary()
    if chat_summary:
        data.append(f"[ChatSummary]\n{chat_summary.strip()}")
    return "\n\n".join(data[-keep_turns:]).strip()


def gather_history_hits(ragb: Optional[RAGBIndex], keywords: List[str], effective_question: str,
                        session_name: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not ragb:
        return []
    query_text = " ".join([kw for kw in keywords if kw]) or effective_question
    if not query_text.strip():
        return []
    top_k = int(cfg.get("history_search", {}).get("max_hits", 8))
    scope = cfg.get("history_search", {}).get("scope", "global")
    try:
        hits = ragb.search(query_text, top_k=top_k, scope=scope, session_name=session_name)
    except Exception:
        hits = []
    threshold = float(cfg.get("history_search", {}).get("score_threshold", 0.0))
    return [h for h in hits if h.get("score", 0.0) >= threshold]


def render_history_hits(hits: List[Dict[str, Any]], limit: int = 6) -> str:
    if not hits:
        return ""
    parts = ["[历史检索片段]"]
    for h in hits[:limit]:
        parts.append(
            f"- ({h.get('session','?')} #{h.get('turn','?')}) {h.get('title','')}\n  {h.get('text','')[:400]}"
        )
    return "\n".join(parts)


def llm_smalltalk_answer(client: OllamaClient, model: str, question: str,
                         context: str, timeout: int, max_ctx: int) -> str:
    prompt = SMALLTALK_PROMPT.strip() + "\n\n"
    if context:
        prompt += f"历史上下文：\n{context.strip()}\n\n"
    prompt += f"用户问题：{question.strip()}\n\n请直接作答："
    resp = client.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options={"temperature": 0.2, "num_ctx": max_ctx, "timeout": timeout},
    )
    return (resp.get("response") or resp.get("message") or "").strip()


def llm_decide_actions(
    client: OllamaClient,
    model: str,
    question: str,
    parsed: Dict[str, Any],
    session_hint: str,
    default_need_rag: bool,
    default_need_history: bool,
    timeout: int,
    max_ctx: int,
) -> Dict[str, Any]:
    prompt = textwrap.dedent(
        f"""\
        你是“对话策略分析器”。请综合用户问题、解析结果与已有上下文，决定是否需要调用知识库检索（RAG）以及是否需要读取历史对话。

        用户问题：
        {question}

        解析结果（JSON）：
        {json.dumps(parsed, ensure_ascii=False, indent=2)}

        最近会话上下文摘要（可能为空）：
        {session_hint or "（无）"}

        请输出严格 JSON：
        {{
          "use_rag": true/false,
          "use_history": true/false,
          "reason": "简短中文说明，20~80字"
        }}
        """
    ).strip()

    try:
        resp = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.0, "num_ctx": max_ctx, "timeout": timeout, "num_predict": 256},
        )
        obj = _extract_json(resp.get("response") or resp.get("message") or "")
    except Exception:
        obj = {}

    if not isinstance(obj, dict):
        obj = {}

    return {
        "use_rag": bool(obj.get("use_rag", default_need_rag)),
        "use_history": bool(obj.get("use_history", default_need_history)),
        "reason": obj.get("reason") or "",
    }


def write_round_compat_files(session_dir: Path, round_idx: int, summary_text: str, rag_text: str) -> None:
    prefix = f"round_{round_idx:02d}"
    summary_path = session_dir / f"{prefix}_summary.txt"
    rag_path = session_dir / f"{prefix}_rag.txt"
    ensure_dirs([summary_path, rag_path])
    summary_body = textwrap.dedent(
        f"""\
        回合编号: {round_idx}

        快速摘要:
        {summary_text.strip() or "（无摘要）"}
        """
    ).strip() + "\n"
    rag_body = textwrap.dedent(
        f"""\
        回合编号: {round_idx}

        精炼检索内容:
        {rag_text.strip() or "（未触发检索）"}
        """
    ).strip() + "\n"
    summary_path.write_text(summary_body, encoding="utf-8")
    rag_path.write_text(rag_body, encoding="utf-8")


def save_parsed(turn_dir: Path, parsed: Dict[str, Any], effective_question: str) -> None:
    out = {
        "parsed": parsed,
        "effective_question": effective_question,
        "saved_at": datetime.now().isoformat(),
    }
    with (turn_dir / "parsed.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮对话脚本（LLM 解析 + RAG 编排）")
    parser.add_argument("--config", default="chat2_config.yaml", help="配置文件（默认为 chat2_config.yaml）")
    parser.add_argument("--session", required=True, help="会话名：用于 chat_sessions 下命名")
    parser.add_argument("--max-rounds", type=int, default=20, help="最多轮数（默认 20）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_config(cfg_path)

    session_root = cfg.get("session_root", "chat_sessions")
    session_root_path = (project_root / session_root) if not os.path.isabs(session_root) else Path(session_root)
    session_root_path.mkdir(parents=True, exist_ok=True)

    session = ChatSession(
        session_root=str(session_root_path),
        session_name=args.session,
        naming=cfg.get("session_naming", "name_and_timestamp"),
        cfg=cfg,
    )
    print(f"[INFO] 会话目录：{session.session_dir}")
    print("[HINT] 输入 exit/quit 可结束会话。")

    ollama_host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    client = OllamaClient(host=ollama_host, timeout=int(cfg.get("llm_timeout", 45)))
    parser_model = cfg.get("model_parser", "qwen3:4b-instruct-2507-fp16")
    answer_model = cfg.get("model", cfg.get("model_coref", parser_model))
    decision_model = cfg.get("decision_model", parser_model)
    llm_timeout = int(cfg.get("llm_timeout", 45))
    llm_max_ctx = int(cfg.get("llm_max_ctx", 4096))

    query_flags = flatten_flags(cfg.get("query_cli_flags"))

    ragb: Optional[RAGBIndex] = None
    try:
        ragb = RAGBIndex(
            chat_root=session_root_path,
            embed_model=cfg.get("embed_model", "bge-m3:latest"),
            ollama_host=ollama_host,
        )
        ragb.scan_and_append_new()
        ragb.embed_incremental()
    except Exception as e:
        print(f"[WARN] 初始化 RAG_B 索引失败：{e}")
        ragb = None

    history_keywords: List[str] = []
    print("-" * 60)

    for round_idx in range(1, args.max_rounds + 1):
        try:
            user_prompt = input(f"[Round {round_idx}] 请输入问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 用户中断，结束会话。")
            break
        if not user_prompt:
            print("[INFO] 空输入，结束会话。")
            break
        if user_prompt.lower() in {"exit", "quit", ":q", "/exit", "/quit"}:
            print("[INFO] 结束会话。")
            break

        turn_dir = Path(session.new_turn_dir())
        (turn_dir / "user.txt").write_text(user_prompt, encoding="utf-8")

        last_focus = session.get_focus()
        chat_summary = session.read_chat_summary()
        last_turn_summary = session.read_last_turn_summary()
        parsed = parse_question(
            user_prompt,
            cfg,
            last_focus=last_focus,
            chat_summary=chat_summary,
            last_turn_summary=last_turn_summary,
            last_entities=session.get_entities(),
        )
        keywords = parsed.get("keywords", []) or []
        entities = parsed.get("entities", []) or []
        actions = parsed.get("actions", []) or []
        intent = parsed.get("intent", "qa")
        need_rag = bool(parsed.get("need_rag", cfg.get("default_enable_rag", True)))
        refer_history = bool(parsed.get("need_history") or cfg.get("default_enable_history_search", True))

        session_hint = gather_session_context(session, keep_turns=int(cfg.get("decision_history_turns", 3)))
        action_decision = llm_decide_actions(
            client=client,
            model=decision_model,
            question=user_prompt,
            parsed=parsed,
            session_hint=session_hint,
            default_need_rag=need_rag,
            default_need_history=refer_history,
            timeout=llm_timeout,
            max_ctx=llm_max_ctx,
        )
        need_rag = bool(action_decision.get("use_rag", need_rag))
        refer_history = bool(action_decision.get("use_history", refer_history))
        decision_reason = action_decision.get("reason") or ""

        if intent != "qa":
            need_rag = False

        if cfg.get("force_rag"):
            need_rag = True

        if decision_reason:
            print(f"[策略] {decision_reason}（RAG={'是' if need_rag else '否'}, 历史={'是' if refer_history else '否'}）")
        else:
            print(f"[策略] RAG={'是' if need_rag else '否'}, 历史={'是' if refer_history else '否'}")

        for ent in entities:
            if ent and ent not in session.get_entities():
                session.push_entities([ent])
        effective_question = rewrite_with_coref(
            user_prompt,
            parsed.get("coref") or {},
            fallback_entities=session.get_entities(),
        )
        coref_info = parsed.get("coref") or {}
        if isinstance(coref_info, dict) and coref_info.get("target"):
            session.set_focus(coref_info.get("target"))
        elif entities:
            session.set_focus(entities[0])

        save_parsed(turn_dir, parsed, effective_question)

        history_context = ""
        history_hits: List[Dict[str, Any]] = []
        if refer_history:
            history_hits = gather_history_hits(
                ragb,
                keywords=keywords or history_keywords,
                effective_question=effective_question,
                session_name=Path(session.session_dir).name,
                cfg=cfg,
            )
            history_context = "\n".join(
                [
                    textwrap.dedent(
                        f"""\
                        历史摘要:
                        {summarize_chat_scan(session.session_dir, cfg)}
                        """
                    ).strip()
                ]
            ).strip()
            hits_text = render_history_hits(history_hits)
            if hits_text:
                history_context = (history_context + "\n\n" + hits_text).strip()
        else:
            history_context = gather_session_context(session)

        answer_text = ""
        rag_clean_md = ""
        rag_clean_markdown_path: Optional[Path] = None

        if need_rag:
            print("[INFO] 触发 RAG 检索 ...")
            # 将 flags 写入临时 cfg 副本，保持 run_query_cli 调用逻辑
            cfg_for_query = dict(cfg)
            cfg_for_query["query_cli_flags"] = query_flags
            rag_result = run_query_cli(effective_question, str(turn_dir), cfg_for_query)
            answer_file = rag_result.get("answer_txt")
            if answer_file and Path(answer_file).exists():
                answer_text = Path(answer_file).read_text(encoding="utf-8").strip()
            else:
                answer_text = Path(turn_dir / "answer.txt").read_text(encoding="utf-8").strip() if (turn_dir / "answer.txt").exists() else ""
            if not answer_text:
                answer_text = "(提示) RAG 调用未返回答案，请检查 query_cli 输出。"

            rows, rag_md = build_turn_files(str(turn_dir), cfg, entities=entities or session.get_entities(), keywords=keywords)
            if rag_md and Path(rag_md).exists():
                rag_clean_md = Path(rag_md).read_text(encoding="utf-8")
                rag_clean_markdown_path = Path(rag_md)
            else:
                rag_clean_md = "(提示) 未生成 RAG 片段文件。"
        else:
            print("[INFO] 解析为无需 RAG，调用 LLM 直接应答 ...")
            answer_text = llm_smalltalk_answer(
                client,
                model=answer_model,
                question=effective_question,
                context=history_context,
                timeout=llm_timeout,
                max_ctx=llm_max_ctx,
            )
            if not answer_text:
                answer_text = "（提示）无法生成回答，请稍后重试或考虑开启检索。"
            rag_clean_md = history_context or "本轮未触发 RAG；已使用会话上下文作答。"

        (turn_dir / "answer.txt").write_text(answer_text, encoding="utf-8")
        if rag_clean_markdown_path is None:
            rag_clean_markdown_path = Path(turn_dir / "turn_files" / "rag_clean.md")
            ensure_dirs([rag_clean_markdown_path])
            rag_clean_markdown_path.write_text(rag_clean_md, encoding="utf-8")

        turn_summary = summarize_turn(answer_text, rag_clean_md, cfg, question=effective_question)
        (turn_dir / "turn_summary.txt").write_text(turn_summary, encoding="utf-8")

        new_chat_summary = summarize_chat_scan(session.session_dir, cfg)
        session.write_chat_summary(new_chat_summary)

        session.append_transcript({"role": "user", "text": user_prompt, "parsed": parsed, "effective_question": effective_question})
        session.append_transcript({"role": "assistant", "text": answer_text, "turn_summary": turn_summary})

        write_round_compat_files(Path(session.session_dir), round_idx, turn_summary, rag_clean_md)

        history_keywords.extend([kw for kw in keywords if kw and kw not in history_keywords])

        print("-" * 60)
        print(f"[Round {round_idx}] 摘要:\n{textwrap.fill(turn_summary, width=70)}")
        print(f"[Round {round_idx}] 答复:\n{textwrap.fill(answer_text, width=70)}")
        print(f"[Round {round_idx}] 精炼检索内容写入: {rag_clean_markdown_path}")
        if history_hits:
            print(f"[Round {round_idx}] 命中历史片段 {len(history_hits)} 条。")
        print("-" * 60)

        if ragb:
            try:
                ragb.scan_and_append_new()
                ragb.embed_incremental()
            except Exception:
                pass

    print("[INFO] 会话结束，感谢使用。")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    main()
