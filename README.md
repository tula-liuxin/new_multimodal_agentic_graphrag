# new_multimodal_agentic_graphrag

一个可在 **Windows (PowerShell) + Python 3.12** 直接跑通的 **多模态 Agentic GraphRAG** 最小可用项目（文本 + 图片（OCR/VLM）），
对 **Notion HTML 导出** 做了专门适配。默认 **离线优先**，本地模型使用 **Ollama (127.0.0.1:11434)** 与 **OpenCLIP**。

> 目标：三段式流水线 —— **召回 → 并行过滤 → 最终回答（带编号引用）**。  
> 重要：所有可调项都在 `config.yaml` 或 CLI 参数中。**不要修改源码就能跑通**。

---

## 0. 运行环境与预置

- OS: Windows 10/11 (PowerShell)
- Python: 3.12.x 64-bit（建议在 `C:\MyNotion\.venv`）
- GPU: NVIDIA（可用 CUDA；自动 CPU fallback）
- 本地模型服务：**Ollama**（已安装并存在如下模型）
  - `qwen2.5:3b-instruct`（轻量并行过滤器）
  - `qwen3:4b-instruct-2507-fp16`（默认最终文本 LLM）
  - `qwen2.5vl:latest`（最终多模态 LLM，可选）
  - `bge-m3:latest`（文本向量）
  - `nomic-embed-text:latest`（备选文本向量）
- OpenCLIP：`CLIP-ViT-L-14-laion2B-s32B-b82K`（可通过 `--clip-local` 指定本地离线目录）
- OCR：默认 **关闭**；开启后使用 **Tesseract**（需本地安装并在 `config.yaml.ocr_tesseract_cmd` 指定路径）。

> 参考：Ollama 官方 API 对 **/api/generate**（支持 `images` base64）与 **/api/embed**（嵌入）详解。citeturn4view0

---

## 1. 一键安装与环境检测

**（推荐）**在 `PowerShell` 执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd <你要放项目的目录>
# 解压后进入目录
.\scripts\quickstart.ps1
```

脚本会：
- 创建并激活 venv、安装 `requirements.txt`；
- 进行 **环境检测**（Ollama 端口、CUDA、Tesseract、长路径策略等）；
- 演示 **ingest → embed/imgindex/charindex → graph → query** 的全流程。

> 如果网络不稳，`pip` 安装失败，请多次重试或使用你本地的 wheel 缓存。模型文件本项目 **不从网络下载**。

---

## 2. 目录结构

```
new_multimodal_agentic_graphrag/
├─ README.md
├─ requirements.txt
├─ config.yaml
├─ .env.example
├─ scripts/
│  ├─ quickstart.ps1
│  └─ sanity_checks.ps1
└─ src/
   ├─ cli.py
   ├─ utils.py
   ├─ ingest.py
   ├─ html_cleaner.py
   ├─ embed_index.py
   ├─ image_index.py
   ├─ char_index.py
   ├─ link_graph.py
   ├─ rerank.py
   ├─ ollama_client.py
   ├─ post_filter.py
   ├─ query_cli.py
   └─ tests/
      ├─ test_paths.py
      ├─ test_html_links.py
      ├─ test_ingest_outputs.py
      ├─ test_post_filter.py
      └─ test_citations.py
```

---

## 3. `config.yaml` 关键项

- `roots`: 数据根目录列表（支持多个 Notion 导出目录）。
- `enable_ocr`: 是否对图片/PDF 做 OCR（默认 false）。
- `ocr_tesseract_cmd`: Windows 下 `tesseract.exe` 绝对路径（启用 OCR 时必填）。
- `ollama.host`: 默认 `http://127.0.0.1:11434`。
- `models`: 嵌入、过滤、多模态等模型名。
- `openclip.local_path`: OpenCLIP 本地目录。
- `concurrency`: 并行度配置。

---

## 4. 一步跑通（示例命令）

> 你可以直接复制以下块到 PowerShell（修改路径）。

### 4.1 一把梭构建

```powershell
python -m src.cli all --clip-local "C:\MyNotion\new_multimodal_agentic_graphrag\models\openclip\CLIP-ViT-L-14-laion2B-s32B-b82K"
```

### 4.2 文本查询

```powershell
python -m src.query_cli `
  --mode hybrid --smart-query --sqw 1.2 `
  --text-first 6 --wtext 1.0 --wimg 0.25 `
  --neighbors 2 --folder-neighbors 2 --link-hop --link-depth 2 `
  --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --pf-timeout 15 --pf-min-keep 6 `
  --top 20 --num-ctx 20000 --ctx-chars 1200 `
  --embed-model "bge-m3:latest" `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\moweifen-01" `
  --model "qwen3:4b-instruct-2507-fp16" `
  "刘晓玲的所有记录"
```

### 4.3 多模态问答（启用 VLM）

```powershell
python -m src.query_cli `
  --mode hybrid --smart-query --sqw 1.2 `
  --vl-answer --max-images 3 --gen-timeout 60 `
  --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 `
  --top 12 --num-ctx 16000 --ctx-chars 900 `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\metaverse-vl" `
  --model "qwen2.5vl:latest" `
  "元宇宙Metaverse架构师 说了什么"
```

---

## 5. 三段式流水线说明

1) **RAG 召回**：文本语义（Ollama `/api/embed`）+ 文件名/路径布尔 + 中文 char/n-gram 模糊 +（可选）OpenCLIP 图像向量；
   对命中节点进行 **link hop** 与 **同文件夹邻居** 扩展，输出 `debug\candidates.json`。  
   > `/api/generate` 的 `images` 字段用于多模态；`/api/embed` 支持单条或批量输入。citeturn4view0

2) **并行过滤 & 预处理**：以 `qwen2.5:3b-instruct` 做 **“任意相关即保留”** 判别（并发可配），清洗噪声/重复，输出 `debug\filtered.json`。

3) **最终回答**：文本或多模态（`--vl-answer`）生成答案，**严格只在末尾输出一个“参考”段**，编号从 `[1]` 开始，
   路径为 **Windows 绝对路径**，去除 `\\?\` / `\?` 等前缀；如 LLM 不可用则优雅降级并输出参考编号。

---

## 6. 常见报错与修复

- **Ollama 500 或超时**：自动重试与降级；请先执行 `.\scripts\sanity_checks.ps1` 查看端口/模型。
- **TesseractNotFoundError**：请安装 Windows 版 Tesseract，并将 `config.yaml.ocr_tesseract_cmd` 指向 `tesseract.exe`。详见修复思路。citeturn0search14turn0search9
- **OpenCLIP 离线加载**：通过 `--clip-local` 传入 checkpoint 路径（`open_clip.create_model_and_transforms(..., pretrained=<ckpt_path>)` 支持本地权重）。citeturn0search2turn5search10
- **FAISS 不可用**：自动回退到 **纯 NumPy TopK** 检索，功能不受影响（仅速度略慢）。

---

## 7. 内置 Prompt 模板

**Smart-Query（检索预处理）**  
见 `src/utils.py: SMART_QUERY_PROMPT`。 

**Post-Filter（并行判别，宁可多留）**  
见 `src/post_filter.py: FILTER_PROMPT`。

**Final-Answer（严格单一“参考”段）**  
见 `src/query_cli.py: FINAL_ANSWER_PROMPT`。

---

## 8. 自测与单元测试

运行：

```powershell
python -m pytest -q
```

覆盖：路径归一化、HTML/MD 链接解析、ingest 产物格式、并行过滤保留策略、引用编号去重等。

---

## 9. 许可

MIT。仅示范工程结构与实现模式，请按需二次开发。
