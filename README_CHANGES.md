
# 补丁说明（v5）

本补丁只包含修改的文件：
- `src/ollama_client.py`：加固本地调用（关闭代理/重试/超时/兼容 embeddings 参数），支持 `images`。
- `src/embed_index.py`：文本向量写入 `index/text_vecs.npy`，并保存 `index/text_meta.json`（含维度）；进度条。
- `src/image_index.py`：修复 `tqdm` 引用、Pillow 截断图容错、AMP API；进度条。
- `src/char_index.py`：修复 `tqdm` 引用；进度条。
- `src/query_cli.py`：
  - 关键词 LLM 清洗（`--llm-query-clean`、`--qc-*`、`--qc-prompt-file`）。
  - 语义召回维度校验（不匹配自动跳过，避免 matmul size 0）。
  - 布尔+模糊增强；邻居扩展包含：链接 hop / 同文件夹 / 父级 / 兄弟。
  - 中间结果落盘：`stage0_keywords.json`、`stage1_sem_text.json`、`stage1_bool.json`、`stage1_expanded_paths.json`、`candidates.json`、`filtered.json`、`answer.txt`、`gen_error.json`。
  - 只保留一个“参考：”段；路径去除 `\\?\`/`\?`。
  - tqdm 进度输出。
- `prompts/keyword_extractor_zh.txt`：关键词剔除 Prompt。

## 快速应用
```powershell
# 1) 解压补丁到项目根（会覆盖同名文件）
Expand-Archive -Path .\bugfix_all_in_one_20251014_v5.zip -DestinationPath . -Force

# 2) 如曾出现“向量维度不匹配/为空”，请重建文本索引：
python -c 'from src.embed_index import build_text_index; build_text_index("chunks.jsonl", embed_model="bge-m3:latest", ollama_host="http://127.0.0.1:11434", batch_size=128); print("✅ 文本索引已重建")'
```

## 查询示例
```powershell
python -m src.query_cli `
  --mode hybrid --smart-query --sqw 1.2 `
  --text-first 6 --wtext 1.0 --wimg 0.25 --wbool 0.7 --wfuzzy 0.6 `
  --neighbors 2 --folder-neighbors 2 --link-hop --link-depth 2 `
  --expand-parent --sibling-span 1 `
  --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 `
  --top 50 --num-ctx 20000 --ctx-chars 1200 `
  --embed-model "bge-m3:latest" `
  --gen-timeout 180 --temperature 0.2 --llm-num-ctx 8192 `
  --llm-query-clean --qc-model "qwen3:4b-instruct-2507-fp16" --qc-timeout 60 --qc-max-keys 8 `
  --qc-prompt-file "prompts\keyword_extractor_zh.txt" `
  --ollama-host "http://127.0.0.1:11434" `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\example" `
  --model "qwen3:4b-instruct-2507-fp16" `
  "刘晓玲"
```
