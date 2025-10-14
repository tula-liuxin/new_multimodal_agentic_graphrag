
# Unified One-Step Query Patch

把 **检索 → 并行过滤 → 最终回答** 合成一个命令（保持你原来的调用习惯）：

## 1) 覆盖文件
解压此压缩包，把 `src/query_cli.py` 覆盖到你的项目同名文件。

## 2) 用法示例

### 文本最终回答
```powershell
python -u -m src.query_cli `
  --smart-query --sqw 1.2 `
  --text-first 6 --wtext 1.0 --wimg 0.25 `
  --top 20 --num-ctx 20000 --ctx-chars 1200 `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\liuxiaoling-01" `
  --embed-model "bge-m3:latest" `
  --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 `
  --final-model "qwen3:4b-instruct-2507-fp16" `
  "刘晓玲的所有记录"
```

### 多模态最终回答（把命中的图片也喂给 LLM）
```powershell
python -u -m src.query_cli `
  --smart-query --sqw 1.2 `
  --text-first 6 --wtext 1.0 --wimg 0.25 `
  --vl-answer --max-images 4 `
  --top 20 --num-ctx 20000 --ctx-chars 1200 `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\metaverse-vl" `
  --embed-model "bge-m3:latest" `
  --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 `
  --final-model "qwen2.5vl:latest" `
  "元宇宙Metaverse架构师 说了什么"
```

## 3) 输出位置
- `final_answer.md`：带编号内联引用 + 「参考」编号列表（不会重复、不会出现 `\\?\`）
- `contexts.txt`：最终喂给 LLM 的上下文
- `candidates.json`：初始候选
- `filtered_candidates.json`：并行过滤后保留的候选

## 4) 备注
- 查询向量默认用 `bge-m3:latest`。如与你的 `embed` 阶段不一致，可用 `--embed-model` 指定。
- 并行过滤使用 `qwen2.5:3b-instruct`（非常快），只要“有一点点相关性”就保留。
- 最终回答默认为 `qwen3:4b-instruct-2507-fp16`；多模态用 `qwen2.5vl:latest`。

如需把过滤阈值进一步放宽/收紧，修改 `--pf-min-keep` 即可。
