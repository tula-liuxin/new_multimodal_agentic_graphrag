# Chat Record Orchestrator （RAG_A + RAG_B）

> 仅新增文件；不修改现有模块行为。将本补丁解压到项目根（`C:\MyNotion\new_multimodal_agentic_graphrag`）。

## 目录新增
- `prompts/intent_router_zh.txt`：用于路由判断的提示词
- `chat_orchestrator.yaml`：对话编排器配置
- `src/intent_router.py`：LLM 路由判定
- `src/chat_record_manager.py`：chat_record 目录/摘要/转写管理
- `src/rag_b.py`：RAG_B（基于聊天记录的增量检索/索引）
- `src/agent_chat.py`：**主入口**（交互式多轮对话、按需调用 RAG_A / RAG_B）
- `scripts/chat_record.ps1`：PowerShell 启动脚本
- `src/tests/test_agent_chat.py`：最小导入测试
- `README_CHAT_RECORD.md`：本文档

## 使用方法
1. 进入项目根：`C:\MyNotion\new_multimodal_agentic_graphrag`
2. 运行：
   ```powershell
   .\scripts\chat_record.ps1       # 默认时间戳会话
   .\scripts\chat_record.ps1 -Session "my-session"  # 自定义会话名
   ```
3. 会生成目录：
   ```
   chat_record/
     20251018-130000_chat/           # 或 --Session 指定的名称
       summary.txt                   # 会话摘要（每轮更新）
       transcript.jsonl              # 全量转写（增量追加）
       turn_00001/
         user.txt
         answer.txt
         summary.txt                 # 本轮摘要
         rag_b_hits.json             # 如使用了 RAG_B，记录命中
       turn_00002/
         ...
     data/chat_chunks.jsonl          # RAG_B 全局语料（增量）
     index/text_vecs.npy, text_ids.npy, id2row.json
   ```

## 运行逻辑
- 每轮：
  1) 用 `prompts/intent_router_zh.txt` 让本地 LLM 判定是否调用 RAG_A / RAG_B；
  2) 若用 RAG_B：先增量维护 `chat_record/data/chat_chunks.jsonl` 与向量索引，检索后生成总结/回答；
  3) 若用 RAG_A：调用 `python -m src.query_cli ...`（flags 来自 `chat_orchestrator.yaml`），`--debug-dir` 指向本轮目录；
  4) 产出本轮 `answer.txt`；随后用本地 LLM 生成 **turn 摘要** 和 **session 摘要**（会覆盖更新）；
  5) 记录 `transcript.jsonl`；最后再把本轮产物增量写入 RAG_B 语料与向量索引。

## 配置项要点
- `ollama_host`：统一传给路由/摘要/嵌入；默认 `http://127.0.0.1:11434`
- `decision_model` / `summarizer_model`：默认 `qwen3:4b-instruct-2507-fp16`
- `embed_model`：默认 `bge-m3:latest`（与 RAG_A 保持一致维度）
- `rag_a.query_cli_flags`：按行列出，支持布尔开关和键值参数；会拼到命令行
- `summary.keep_last_turns` / `summary.memory_max_chars`：控制 session 摘要上下文窗口

## 兼容性 & 契约
- 不修改任何既有函数签名/默认行为；所有新增输出都落盘到 `chat_record/`
- RAG_B 离线优先、可增量；向量维度不匹配时会自动重建索引（仅 RAG_B 内部，不影响 RAG_A）
- Windows 路径对外输出为绝对路径（由 Python 的 `resolve()` 保证，不含 `\\?\` 前缀）

## 常见问题
- **首次运行很慢？** 需要首次创建 RAG_B 向量索引；后续增量会很快。
- **维度不匹配？** RAG_B 会丢弃旧索引并用当前嵌入模型重建，不影响 RAG_A。
- **只想总结历史，不查外部？** 直接输入“总结前面几轮聊天”的自然语言即可，路由器会选择 RAG_B。

祝使用愉快！
