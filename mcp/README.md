
# MCP Server for new_multimodal_agentic_graphrag

本目录提供一个 **MCP (Model Context Protocol)** 服务器，将你的项目暴露为 MCP 工具，便于在 **Claude Desktop / Amazon Q CLI / 其他 MCP 客户端**中直接调用流水线与查询。

## 安装

1) 在你的虚拟环境中安装 MCP Python SDK（包含 CLI）：

```powershell
pip install "mcp[cli]"
```

> 参考：官方 Python SDK 文档（含 `mcp` 命令与 FastMCP）citeturn2view0

2) （可选）在项目根目录设置环境变量，指定根与 Python：

```powershell
$env:GRAPHRAG_ROOT = "C:\MyNotion\new_multimodal_agentic_graphrag"
$env:GRAPHRAG_PY   = "$env:VIRTUAL_ENV\Scripts\python.exe"
```

## 本地开发运行

```powershell
# 在项目根目录执行
mcp dev .\mcp\graphrag_server.py
```

开发模式会使用 **stdio 传输**运行服务器，方便在命令行里观察日志与调试。

## Claude Desktop 集成（Windows）

1) 运行一条命令自动写入配置：

```powershell
mcp install .\mcp\graphrag_server.py
```

2) 重启 Claude Desktop。此后在 Claude 的「工具」里可以看到 **graphrag-mcp**。

> 参考：MCP Python SDK 的 CLI `mcp dev / mcp install` 用法与 Claude Desktop 集成指引。citeturn2view0turn4search9

## 暴露的工具

- `health(ollama_host="http://127.0.0.1:11434")`  
  返回 Python/平台/CUDA/Ollama 可用性；并对 `/api/generate` 做 1 次探活。

- `list_models()`  
  解析 `ollama list` 输出，返回本地可用模型表。

- `rag_all(cli_args: str = "", timeout_sec: int = 0)`  
  等同命令：`python -m src.cli all`，并将 `cli_args` 原样拼接。例如：  
  `--enable-ocr --embed-model "bge-m3:latest" --clip-local "C:\...\CLIP-ViT-L-14-laion2B-s32B-b82K"`。

- `rag_query(question: str, cli_args: str = "", timeout_sec: int = 0)`  
  等同命令：`python -m src.query_cli <cli_args> "<question>"`。给你最大灵活度填写之前的全部检索/过滤/生成参数。

- `tail_debug(debug_dir: str, n: int = 2000)`  
  读取 `debug_dir` 下的 `console.log/pipeline.json/error.log/contexts.txt` 中存在的第一个文件，返回尾部 `n` 字符，便于在 MCP 客户端中快速查看。

> 你可以直接在 MCP 客户端里调用这些工具；例如在 Claude 里发起一个工具调用：
> `rag_query(question="刘晓玲的所有记录", cli_args="--mode hybrid --smart-query --post-filter --pf-model 'qwen2.5:3b-instruct' --top 20 --model 'qwen3:4b-instruct-2507-fp16' --debug-dir 'C:\\MyNotion\\new_multimodal_agentic_graphrag\\debug\\moweifen-01'")`

## 典型命令（PowerShell，可复制）

**开发/调试服务器：**
```powershell
mcp dev .\mcp\graphrag_server.py
```

**安装到 Claude Desktop：**
```powershell
mcp install .\mcp\graphrag_server.py
```

**（任意 MCP 客户端）调用流水线：**
- `rag_all`：`cli_args` 里放入你原本在命令行里的 flags
- `rag_query`：把问题与 `cli_args` 一起传入（包含 smart-query、post-filter、vl-answer、neighbors 等所有参数）

## 安全说明

- 该服务器通过子进程调用你项目中的 CLI，**仅在本机使用**；请勿暴露到公网。
- `rag_query`/`rag_all` 的 `cli_args` 会直接传给命令行，请仅向可信客户端暴露。

