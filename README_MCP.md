# MCP quick usage (GraphRAG)

## Dev inspector
```powershell
# From repo root
.\scripts\mcp.ps1 dev
```
Then open the MCP Inspector and connect.

## Install into Claude Desktop
```powershell
.\scripts\mcp.ps1 install
```

## Tools exposed
- `health()` → basic env info (python/torch/cuda/ollama)
- `run_cli(module: str, args: str)` → run project CLIs, e.g.:
  - `module="src.cli"` `args="all --enable-ocr"`
  - `module="src.query_cli"` `args="--mode hybrid --smart-query \"刘晓玲\""`

All stdout/stderr are saved to `debug/mcp/run_<ts>.log`.
