
Overlay v2 (path fix + robust answer)
=====================================

What this overlay changes
-------------------------
1) Reference paths now join with `input_dir` (user data root), not project root.
2) Adds Windows long-path support (\\?\ prefix) for very long paths.
3) Ensures contexts.txt is always written; "参考：" appears once with [1]..[N].
4) Stronger "文本优先 + 图片补齐" fusion. Images searched via OCR/caption.
5) LLM is always attempted unless you pass --quiet-fallback; otherwise graceful fallback.

Install
-------
1) Unzip this file to your project root:
   C:\MyNotion\new_multimodal_agentic_graphrag\
   It will overwrite:
     - src\query_cli.py
     - src\utils.py

2) Make sure config.yaml has:
   input_dir: C:\MyNotion\MyNotion20251007

3) Run a quick sanity test:
   python -u -m src.query_cli --smart-query --text-first 6 --wtext 1.0 --wimg 0.25 ^
     --top 12 --num-ctx 16000 --ctx-chars 900 ^
     --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\sanity" ^
     --model "qwen3:4b-instruct-2507-fp16" ^
     "莫维芬的所有记录"

Notes
-----
- If you still see "生成模型不可用"，检查 Ollama 是否在运行：
  $env:OLLAMA_HOST="http://127.0.0.1:11434"; ollama serve
- Very long paths will be printed with \\?\ prefix; Explorer 有时不能直接打开，先复制路径到地址栏再回车即可。
