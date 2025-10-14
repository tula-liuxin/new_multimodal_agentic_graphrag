# tri-stage-patch

这是一个“可覆盖的小补丁”，解决：
- `charindex` 无 `run_charindex`；
- 缺少 `link_graph`；
- 新增 `post_filter`（并行轻量 LLM 过滤 + 最终回答，多模态可选）；
- 附带 `tools/fix_query_cli_attrs.py` 以修复 `query_cli.py` 里 `vl-answer/text-first/images_only` 之类属性名不匹配。

## 覆盖 & 运行

```powershell
# 在项目根目录
$env:PYTHONPATH = "$PWD"

# 1) 覆盖本补丁的 src/ 与 tools/ 到项目同名目录

# 2)（推荐）修补 query_cli.py
python tools/fix_query_cli_attrs.py

# 3) 预处理
python -m src.cli ingest
python -m src.cli embed
python -m src.cli imgindex
python -m src.cli charindex
python -m src.cli links
# graph 可选
```

## 三段式（示例）

### 第一步：RAG 召回（仍用你的 query_cli，写出 candidates.json）
```powershell
python -u -m src.query_cli `
  --smart-query --sqw 1.2 `
  --text-first 6 --wtext 1.0 --wimg 0.25 `
  --top 20 --num-ctx 20000 --ctx-chars 1200 `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\liuxiaoling-01" `
  --model "qwen3:4b-instruct-2507-fp16" `
  "刘晓玲的所有记录"
```

### 第二 & 三步：并行过滤 +（可多模态）最终回答（新的 post_filter）
```powershell
python -u -m src.post_filter `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\liuxiaoling-01" `
  --question "刘晓玲的所有记录" `
  --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 `
  --final-model "qwen3:4b-instruct-2507-fp16" `
  --ctx-chars 1200 --num-ctx 20000
# 若要多模态最终回答：
#   --final-model "qwen2.5vl:latest" --vl-final --max-final-images 4
```
输出：
- `debug\filtered_candidates.json`
- `debug\contexts.txt`
- `debug\final_answer.md`（答案末尾自带**编号引用**，且已去除路径里的“\\?\”）
