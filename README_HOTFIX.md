# Graphrag Hotfix — Usage

## 1) Install / Overwrite
- Unzip this package into your project root (`C:\MyNotion\new_multimodal_agentic_graphrag`).
- Only the files listed in `PATCH_NOTES.txt` are included, safe to overwrite.

## 2) Rebuild indices (incremental safe)
```powershell
python -m src.cli ingest --roots "C:\MyNotion\MyNotion20251007" --out-dir "data" --incremental --enable-ocr --min-chars 0 --no-thin-filter --save-why-drop --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\session-01"

python -m src.cli embed --out-dir "data" --incremental --embed-model "bge-m3:latest" --ollama-host "http://127.0.0.1:11434"
python -m src.cli imgindex --out-dir "data" --incremental --clip-local "C:\MyNotion\new_multimodal_agentic_graphrag\models\openclip\CLIP-ViT-L-14-laion2B-s32B-b82K"
python -m src.cli charindex --out-dir "data" --incremental
python -m src.cli graph --out-dir "data"
```

## 3) Query examples
- Text-first:
```powershell
python -m src.query_cli --mode hybrid --smart-query --text-first 6 --wtext 1.0 --wimg 0.25 --wbool 0.8 --wfuzzy 0.8 --neighbors 2 --folder-neighbors 2 --link-hop --link-depth 2 --expand-parent --sibling-span 1 --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 --top 20 --num-ctx 20000 --ctx-chars 1200 --embed-model "bge-m3:latest" --gen-timeout 180 --temperature 0.2 --llm-query-clean --qc-model "qwen3:4b-instruct-2507-fp16" --qc-timeout 60 --qc-max-keys 8 --qc-prompt-file "prompts\keyword_extractor_zh.txt" --llm-num-ctx 8192 --ollama-host "http://127.0.0.1:11434" --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\session-01" --model "qwen3:4b-instruct-2507-fp16" "刘晓玲"
```
- Image-first:
```powershell
python -m src.query_cli --mode hybrid --smart-query --text-first 0 --wtext 0.25 --wimg 1.0 --wbool 0.6 --wfuzzy 0.6 --neighbors 2 --folder-neighbors 2 --link-hop --link-depth 2 --expand-parent --sibling-span 1 --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 --top 8 --num-ctx 12000 --ctx-chars 700 --embed-model "bge-m3:latest" --vl-answer --max-images 4 --gen-timeout 150 --temperature 0.2 --llm-num-ctx 6144 --llm-query-clean --qc-model "qwen3:4b-instruct-2507-fp16" --qc-timeout 60 --qc-max-keys 8 --qc-prompt-file "prompts\keyword_extractor_zh.txt" --ollama-host "http://127.0.0.1:11434" --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\session-01" --model "qwen2.5vl:latest" "8分档"
```

## 4) Audit helper
```powershell
python -m tools.coverage --root "C:\MyNotion\MyNotion20251007" --chunks "data\chunks.jsonl" --out "C:\MyNotion\new_multimodal_agentic_graphrag\debug\ingest-audit\audit_coverage.json"
```
