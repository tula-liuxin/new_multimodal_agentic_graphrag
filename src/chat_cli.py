
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, time, subprocess, re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from .ollama_client import OllamaClient
except Exception:
    from src.ollama_client import OllamaClient  # type: ignore

def now_str():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class ChatConfig:
    ollama_host: Optional[str] = "http://127.0.0.1:11434"
    session_name: Optional[str] = None
    base_dir: str = "chat_sessions"
    query_cli_flags: List[str] = None
    summarizer_model: str = "qwen3:4b-instruct-2507-fp16"
    summarizer_timeout: int = 60
    memory_max_chars: int = 2000
    history_ctx_max_chars: int = 8000
    include_history_in_question: bool = True

    def ensure(self):
        if self.query_cli_flags is None:
            self.query_cli_flags = []

class ChatSession:
    def __init__(self, cfg: ChatConfig):
        cfg.ensure()
        self.cfg = cfg
        session = cfg.session_name or time.strftime("session-%Y%m%d-%H%M%S")
        self.session_dir = Path(cfg.base_dir)/session
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.session_dir/"summary.txt"
        self.summary_hist_path = self.session_dir/"summary_history.jsonl"
        self.summary_hist_plain = self.session_dir/"history_context.txt"
        self.transcript_path = self.session_dir/"transcript.jsonl"
        for p in (self.summary_path, self.summary_hist_path, self.summary_hist_plain, self.transcript_path):
            if not p.exists():
                p.write_text("" if p.suffix!=".jsonl" else "", encoding="utf-8")
        self.turn = 0

    def _append_transcript(self, role: str, text: str, meta: Dict[str,Any] = None):
        rec = {"time": now_str(), "role": role, "text": text}
        if meta: rec["meta"] = meta
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    def _read_latest_summary(self) -> str:
        try:
            return self.summary_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _aggregate_history_ctx(self) -> str:
        items: List[str] = []
        try:
            if self.summary_hist_path.exists():
                with open(self.summary_hist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            obj = json.loads(line)
                            t = obj.get("turn")
                            s = (obj.get("summary") or "").strip()
                            if s:
                                items.append(f"[T{t}] {s}")
                        except Exception:
                            pass
        except Exception:
            pass
        max_chars = int(self.cfg.history_ctx_max_chars or 8000)
        acc: List[str] = []
        total = 0
        for seg in reversed(items):
            add = len(seg) + 1
            if total + add > max_chars:
                break
            acc.append(seg)
            total += add
        acc = list(reversed(acc))
        text = "\n".join(acc).strip()
        try:
            self.summary_hist_plain.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return text

    def _write_summary(self, single_summary: str, turn_dir: Path):
        s = (single_summary or "").strip()
        if len(s) > self.cfg.memory_max_chars:
            s = s[-self.cfg.memory_max_chars:]
        self.summary_path.write_text(s, encoding="utf-8")
        with open(self.summary_hist_path, "a", encoding="utf-8") as f:
            rec = {"time": now_str(), "turn": self.turn, "summary": s}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (turn_dir/"summary.txt").write_text(s, encoding="utf-8")
        _ = self._aggregate_history_ctx()

    def _summarize_llm(self, model: str, user: str, assistant: str) -> str:
        prev = self._read_latest_summary()
        sys_prompt = (
            "你是对话记忆压缩器。请基于【旧摘要】与【本轮新增】重写新的精简摘要，"
            "用于指导下一轮 RAG 检索。保留关键信息：主题/实体/限制/未解决目标；去除客套，输出≤{limit}字符。"
        ).format(limit=self.cfg.memory_max_chars)
        prompt = f"{sys_prompt}\n\n[旧摘要]\n{prev}\n\n[本轮新增]\n用户: {user}\n助手: {assistant}\n\n请给出新的摘要："
        cli = OllamaClient(host=self.cfg.ollama_host, timeout=self.cfg.summarizer_timeout)
        jr = cli.generate(model=model, prompt=prompt, stream=False, options={"temperature":0.2,"num_predict":512})
        summ = (jr.get("response","") or "").strip()
        summ = re.sub(r"^摘要[:：]\s*", "", summ, flags=re.I).strip()
        return summ

    def _summarize_heuristic(self, user: str, assistant: str) -> str:
        ans = (assistant or "").strip()
        m = re.search(r"(关键要点[:：][\s\S]+?)(?:\n\n|$)", ans)
        base = m.group(1).strip() if m else ""
        if not base:
            base = ans[:400]
        base = re.sub(r"\n{3,}", "\n\n", base)
        summ = f"Q: {user}\n{base}"
        return summ.strip()

    def chat_once(self, user_msg: str, extra_query_flags: List[str] = None) -> str:
        self.turn += 1
        turn_dir = self.session_dir/f"turn-{self.turn:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        self._append_transcript("user", user_msg)

        history_ctx = self._aggregate_history_ctx() if self.cfg.include_history_in_question else ""
        if history_ctx:
            composed_question = f"[对话历史摘要]\n{history_ctx}\n\n[当前问题]\n{user_msg}"
        else:
            composed_question = user_msg
        (turn_dir/"composed_question.txt").write_text(composed_question, encoding="utf-8")

        flags = list(self.cfg.query_cli_flags or [])
        if self.cfg.ollama_host:
            flags += ["--ollama-host", self.cfg.ollama_host]
        flags += ["--debug-dir", str(turn_dir)]
        if extra_query_flags:
            flags += extra_query_flags

        cmd = [sys.executable, "-m", "src.query_cli"] + flags + [composed_question]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        (turn_dir/"cmd.txt").write_text(" ".join(cmd), encoding="utf-8")

        print(f"[Chat] 第 {self.turn} 轮：正在检索与生成 ...")
        print(f"[Chat] 历史摘要用于本轮：{len(history_ctx)} chars")
        print(f"[Chat] CMD: {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
            (turn_dir/"console.log").write_text((p.stdout or "") + "\n===STDERR===\n" + (p.stderr or ""), encoding="utf-8")
        except Exception as e:
            (turn_dir/"console.log").write_text("子进程调用失败: "+str(e), encoding="utf-8")

        ans_path = turn_dir/"answer.txt"
        if ans_path.exists():
            answer = ans_path.read_text(encoding="utf-8")
        else:
            msg = ["（提示）本轮生成失败。请查看 turn 目录下的 console.log 与 gen_error.json。"]
            ge = turn_dir/"gen_error.json"
            if ge.exists():
                try:
                    data = json.loads(ge.read_text(encoding="utf-8"))
                    reason = data.get("error") or data.get("message") or ""
                    if reason:
                        msg.append("\n[gen_error.json 摘要]\n"+reason)
                except Exception:
                    pass
            try:
                cl = (turn_dir/"console.log").read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(cl[-40:])
                if tail.strip():
                    msg.append("\n[console.log 尾部]\n"+tail)
            except Exception:
                pass
            answer = "\n".join(msg)

        self._append_transcript("assistant", answer, meta={"turn_dir": str(turn_dir)})

        summ = ""
        try:
            summ = self._summarize_llm(self.cfg.summarizer_model, user_msg, answer)
        except Exception:
            pass
        if not summ:
            summ = self._summarize_heuristic(user_msg, answer)
        self._write_summary(summ, turn_dir)

        print(answer)
        print("\n—— 以上为第 {} 轮，已写入：\n- 滚动摘要：{}\n- 历史摘要：{}\n- 本轮问句：{}\n".format(
            self.turn, self.summary_path, self.summary_hist_path, turn_dir/"composed_question.txt"
        ))
        return answer

def load_chat_config(path: Optional[str]) -> ChatConfig:
    import yaml
    cfg = ChatConfig()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg.ollama_host = data.get("ollama_host", cfg.ollama_host)
        cfg.session_name = data.get("session_name", cfg.session_name)
        cfg.base_dir = data.get("base_dir", cfg.base_dir)
        cfg.query_cli_flags = data.get("query_cli_flags", cfg.query_cli_flags)
        cfg.summarizer_model = data.get("summarizer_model", cfg.summarizer_model)
        cfg.summarizer_timeout = int(data.get("summarizer_timeout", cfg.summarizer_timeout))
        cfg.memory_max_chars = int(data.get("memory_max_chars", cfg.memory_max_chars))
        cfg.history_ctx_max_chars = int(data.get("history_ctx_max_chars", cfg.history_ctx_max_chars))
        cfg.include_history_in_question = bool(data.get("include_history_in_question", cfg.include_history_in_question))
    cfg.ensure()
    return cfg

def main():
    import argparse
    ap = argparse.ArgumentParser(description="多轮对话 Chat 壳（每轮自动 RAG；滚动摘要 + 历史摘要拼接）")
    ap.add_argument("--config", default="chat_config.yaml")
    ap.add_argument("--session", default=None, help="指定会话名；默认按时间戳新建（想续聊请带上这个）")
    ap.add_argument("--once", action="store_true", help="只进行一轮（便于脚本集成）")
    ap.add_argument("--question", default=None, help="与 --once 配合使用：单轮问题文本")
    args = ap.parse_args()

    cfg = load_chat_config(args.config)
    if args.session: cfg.session_name = args.session
    sess = ChatSession(cfg)

    if args.once:
        q = args.question or "你好"
        sess.chat_once(q)
        return

    print("=== 多轮对话开始（输入 exit/quit 结束） ===")
    print(f"(会话目录：{sess.session_dir})")
    try:
        while True:
            try:
                user_msg = input("\n你：").strip()
            except EOFError:
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("exit","quit",":q","q"):
                print("结束。会话保存在：{}".format(sess.session_dir))
                break
            sess.chat_once(user_msg)
    except KeyboardInterrupt:
        print("\n中断。会话保存在：{}".format(sess.session_dir))

if __name__ == "__main__":
    main()
