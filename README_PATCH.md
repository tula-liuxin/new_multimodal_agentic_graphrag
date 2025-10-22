# Chat v2 fixed15 Patch (LLM 解析 + 指代消解 + UTF‑8 + 摘要滚动 + 检索增强)

将本压缩包内容拷贝到你的仓库根目录（`new_multimodal_agentic_graphrag/`）即可：

- `src/chat2/*.py`：替换/新增模块（不影响旧 `src.query_cli`）
- `chat2_config.yaml`：示例配置（可直接用或对比你的配置修改）
- `scripts/chat_v2.bat`：Windows 一键启动（避免 PowerShell 执行策略限制）

## 快速使用
1. 解压到仓库根目录。
2. 运行：
   ```bat
   scripts\chat_v2.bat
   ```
   或：
   ```bat
   .venv\Scripts\python.exe -m src.chat2.cli --config chat2_config.yaml --session "my-session"
   ```
3. 每次会话会创建：`chat_sessions/<name>-YYYYMMDD-HHMMSS/`，
   每轮生成：`turn-0001/turn_summary.txt`、`turn_files/rag_clean.*`、`answer.txt`、`parsed.json`、`console.log`。

## 主要改动
- LLM 结构化解析：`intent/need_rag/keywords/entities/coref/pairs/topics/actions`
- 代词消解（“她/他们/两人”等）→ `effective_question` 自动替换为实体名
- UTF‑8 严格：子进程 `-X utf8` + 环境变量，避免 cp1252 报错
- 每轮/会话摘要：`turn_summary.txt` & `summary.txt` 每轮更新
- 检索增强：`top=4`、ASI 同义词扩展、双实体奖励；不改老 `query_cli`
