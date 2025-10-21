# Chat 多轮对话壳（增量新增，不改现有代码）

## 启动
```powershell
# 可选：指定会话名
.\scripts\chat.ps1 -Session "my-notion-chat"
# 或直接：
python -m src.chat_cli --config chat_config.yaml
```

## 目录结构
```
chat_sessions/
  <session>/
    summary.txt            # 滚动摘要（供下一轮参考）
    transcript.jsonl       # 对话明细（逐轮追加）
    turn-0001/
      answer.txt           # 本轮最终回答（含唯一参考段）
      console.log          # 本轮 query_cli 控制台输出
      ...                  # 其余 debug 产物均由 query_cli 写入
    turn-0002/
      ...
```

## 原理
- 每轮 `src.chat_cli` 会把 `summary.txt`（滚动摘要）和你的本轮问题拼成**组合问题**，调用 `python -m src.query_cli`；
- `query_cli` 完整执行 RAG 三段式并生成答案；
- `chat_cli` 读取 `answer.txt` 回显；
- 紧接着 `chat_cli` 用本地 LLM（`summarizer_model`）将**历史对话+本轮回答**压成新的 `summary.txt`，供下一轮使用；
- 整个过程中不改任何现有文件。

## 常用调整
- 修改 `chat_config.yaml` 中的 `query_cli_flags` 即可改变检索/过滤/生成参数；
- 如果希望关键词清洗（llm-query-clean），只需在 `query_cli_flags` 里加入：
  `--llm-query-clean --qc-model qwen3:4b-instruct-2507-fp16 --qc-timeout 90 --qc-max-keys 8 --qc-prompt-file prompts/keyword_extractor_zh.txt`；
  **注意**：开启关键词清洗时，组合问题里的“[对话摘要]”可能被当做检索文本；如果发现召回偏移，可在 flags 中**先不启用**。

## 退出
在交互模式下输入 `exit` / `quit` / `:q` / `q`。
