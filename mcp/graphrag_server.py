# -*- coding: utf-8 -*-
"""
MCP server exposing new_multimodal_agentic_graphrag as tools.

Tools:
- health(): report environment info
- run_cli(module: str, args: str): run "python -m <module> <args>" in project root

Usage (dev):
  mcp dev mcp/graphrag_server.py
  # or run directly:
  python mcp/graphrag_server.py
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

# ✅ Correct import: stdio transport is handled by FastMCP.run()
from mcp.server.fastmcp import FastMCP

APP_NAME = "graphrag-mcp"
mcp = FastMCP(APP_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@mcp.tool()
def health() -> Dict[str, Any]:
    """Return basic environment/diagnostic info for the server."""
    info: Dict[str, Any] = {
        "app": APP_NAME,
        "cwd": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "venv": os.environ.get("VIRTUAL_ENV") or "",
    }
    # Optional torch / cuda
    try:
        import torch  # type: ignore
        info["torch"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch"] = f"unavailable ({e.__class__.__name__})"
    # Optional: ollama version
    try:
        out = subprocess.check_output(["ollama", "version"], text=True, timeout=5)
        info["ollama_version"] = out.strip()
    except Exception as e:
        info["ollama_version"] = f"unavailable ({e.__class__.__name__})"
    return info

@mcp.tool()
def run_cli(module: str, args: str) -> Dict[str, Any]:
    """
    Run a project CLI module via the current Python, inside the project root.

    Parameters
    ----------
    module : str
        e.g. "src.cli" or "src.query_cli"
    args : str
        Full argument string appended to the module, e.g. 'all --enable-ocr --embed-model "bge-m3:latest"'

    Returns
    -------
    dict : {ok, code, cmd, elapsed_s, stdout_tail, stderr_tail, debug_file}
    """
    start = time.time()
    py = sys.executable or "python"
    cmd = [py, "-m", module] + shlex.split(args)
    # Ensure PATH contains project root
    env = os.environ.copy()
    # Working directory: project root (two parents up from this file)
    cwd = str(PROJECT_ROOT)
    # Make a debug output file (stdout+stderr tee) under project debug/mcp
    debug_dir = PROJECT_ROOT / "debug" / "mcp"
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_path = debug_dir / (f"run_{int(start)}.log")
    ok = False
    code = -1
    stdout_tail = ""
    stderr_tail = ""
    try:
        with open(log_path, "w", encoding="utf-8", errors="ignore") as fh:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = proc.communicate()
            fh.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            fh.write("=== STDOUT ===\n")
            fh.write(out or "")
            fh.write("\n=== STDERR ===\n")
            fh.write(err or "")
            code = proc.returncode
            ok = (code == 0)
            # Return last 4000 chars to avoid flooding the client
            stdout_tail = (out or "")[-4000:]
            stderr_tail = (err or "")[-4000:]
    except Exception as e:
        stderr_tail = f"{e.__class__.__name__}: {e}"
    elapsed = round(time.time() - start, 3)
    return {
        "ok": ok,
        "code": code,
        "cmd": " ".join(shlex.quote(x) for x in cmd),
        "elapsed_s": elapsed,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "debug_file": str(log_path),
    }

def main() -> None:
    # Default transport is stdio; you can override with transport="streamable-http"
    mcp.run()

if __name__ == "__main__":
    main()
