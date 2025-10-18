\
    Param(
      [string]$Root = "C:\MyNotion\MyNotion20251007",
      [string]$OutDir = "data",
      [string]$Dbg = "C:\MyNotion\new_multimodal_agentic_graphrag\debug\session-01",
      [string]$ClipLocal = "C:\MyNotion\new_multimodal_agentic_graphrag\models\openclip\CLIP-ViT-L-14-laion2B-s32B-b82K",
      [string]$Ollama = "http://127.0.0.1:11434"
    )

    $dest = Join-Path $PSScriptRoot "_cheatsheet.txt"

    @"
# 0) Quick sanity (optional)
python -V
ollama list

# 1) Ingest (incremental, keep debug, no thin filter)
python -m src.cli ingest --roots "$Root" --out-dir "$OutDir" --incremental --enable-ocr --min-chars 0 --no-thin-filter --save-why-drop --debug-dir "$Dbg"

# 2) Build indices (text, image, char, graph)
python -m src.cli embed --out-dir "$OutDir" --incremental --embed-model "bge-m3:latest" --ollama-host "$Ollama"
python -m src.cli imgindex --out-dir "$OutDir" --incremental --clip-local "$ClipLocal"
python -m src.cli charindex --out-dir "$OutDir" --incremental
python -m src.cli graph --out-dir "$OutDir"

# 3) Text query (hybrid)
python -m src.query_cli --mode hybrid --smart-query --text-first 6 --wtext 1.0 --wimg 0.25 --wbool 0.8 --wfuzzy 0.8 --neighbors 2 --folder-neighbors 2 --link-hop --link-depth 2 --expand-parent --sibling-span 1 --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 --top 20 --num-ctx 20000 --ctx-chars 1200 --embed-model "bge-m3:latest" --gen-timeout 180 --temperature 0.2 --llm-query-clean --qc-model "qwen3:4b-instruct-2507-fp16" --qc-timeout 60 --qc-max-keys 8 --qc-prompt-file "prompts\keyword_extractor_zh.txt" --llm-num-ctx 8192 --ollama-host "$Ollama" --debug-dir "$Dbg" --model "qwen3:4b-instruct-2507-fp16" ""

# 4) Multimodal query (image-first)
python -m src.query_cli --mode hybrid --smart-query --text-first 0 --wtext 0.25 --wimg 1.0 --wbool 0.6 --wfuzzy 0.6 --neighbors 2 --folder-neighbors 2 --link-hop --link-depth 2 --expand-parent --sibling-span 1 --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 --top 8 --num-ctx 12000 --ctx-chars 700 --embed-model "bge-m3:latest" --vl-answer --max-images 4 --gen-timeout 150 --temperature 0.2 --llm-num-ctx 6144 --llm-query-clean --qc-model "qwen3:4b-instruct-2507-fp16" --qc-timeout 60 --qc-max-keys 8 --qc-prompt-file "prompts\keyword_extractor_zh.txt" --ollama-host "$Ollama" --debug-dir "$Dbg" --model "qwen2.5vl:latest" "8分档"

# 5) Quick audit helpers
.\scripts\quick_audit.ps1 -Root "$Root" -OutDir "$OutDir" -Dbg "$Dbg" -Keyword "刘晓玲"
"@ | Set-Content -Path $dest -Encoding UTF8

Write-Host "== 已生成: $dest =="
