# new_multimodal_agentic_graphrag

多模态（文本 + 图像）的本地知识库检索与问答流水线，支持 GraphRAG、层级/目录检索、link-hop、中文长句/人名召回、绝对 Windows 路径回显、Debug 落盘。

- 目标 OS：**Windows 11**
- 运行环境：**PowerShell + Python 3.12**
- 推荐硬件：**RTX 4060 Ti 16GB + i5-13600KF + 32GB RAM**
- 默认用户数据目录：`C:\MyNotion\MyNotion20251007`
- 默认项目目录：`C:\MyNotion\new_multimodal_agentic_graphrag`

> 所有输出中的文件路径均为**绝对 Windows 路径**（反斜杠 `\`）。

---

## 0. 安装依赖（Windows 一键参考）

### 0.1 Python 与虚拟环境
```powershell
# 建议 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 0.2 CUDA + PyTorch
> 建议 CUDA 12.1；若本机已有 CUDA/cuDNN，可直接安装与之匹配的 torch 轮子。以下为官方示例（请按需替换版本）：
```powershell
# 官方示例（请以 https://pytorch.org/get-started/locally/ 实际为准）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 0.3 其余依赖
```powershell
pip install -r requirements.txt
```

### 0.4 可选：Tesseract OCR（中文 + 英文）
1. 安装包（Windows）：https://github.com/UB-Mannheim/tesseract/wiki
2. 安装后将 `tesseract.exe` 所在目录（例如 `C:\Program Files\Tesseract-OCR`）加入 PATH。
3. 中文语言包：安装 `chi_sim`（简体）与 `eng`。安装完成后确保 `tessdata` 下有 `chi_sim.traineddata`。

### 0.5 启动 Ollama（本机大模型服务）
```powershell
# 新窗口中常驻
$env:OLLAMA_HOST="127.0.0.1:11434"
$env:OLLAMA_NUM_PARALLEL="2"
$env:OLLAMA_KEEP_ALIVE="10m"
ollama serve
```
> 需要模型：
> - 文本嵌入：`nomic-embed-text:latest`
> -（可选）文本生成/重写/扩展：`qwen3:4b-instruct-2507-fp16`（或 `qwen3:latest`）
> -（可选）多模态字幕：`qwen2.5vl:latest` 或任何本地可用的 VL 模型

查看已有模型：
```powershell
ollama list
```

---

## 1. 配置

编辑 `config.yaml`（所有字段都可被环境变量覆盖），关键默认值：

```yaml
project_root: "C:\\MyNotion\\new_multimodal_agentic_graphrag"
input_dir:    "C:\\MyNotion\\MyNotion20251007"
data_dir:     "C:\\MyNotion\\new_multimodal_agentic_graphrag\\data"
debug_dir:    "C:\\MyNotion\\new_multimodal_agentic_graphrag\\debug"

# OCR/字幕
enable_ocr: false
ocr_lang: "chi_sim+eng"
ocr_psm: 6
enable_image_captions: false
caption_model: "qwen2.5vl:latest"

# Ollama
ollama_base_url: "http://127.0.0.1:11434"
embed_model: "nomic-embed-text"
embed_batch: 128
embed_concurrency: 2
embed_num_thread: 8
embed_keep_alive: "10m"
embed_timeout: 120

# OpenCLIP 图像向量
image_model_name: "ViT-L-14"
image_pretrained: "laion2b_s32b_b82k"
image_batch: 64
image_precision: "fp16"
image_workers: 6

# 索引/检索
max_chars: 1200
overlap_chars: 200
tfidf_max_features: 100000
char_ngram_range: [2, 5]
```

> 环境变量覆盖示例（PowerShell）：
```powershell
$env:NMAGR_input_dir="C:\Another\Folder"
$env:NMAGR_enable_ocr="true"
```

---

## 2. 从 0 到查询（必跑流程）

> **确保已启动 Ollama 服务（见 0.5）。**

```powershell
# 1) 采集（文本+图片；可选开启 OCR/字幕）
python -m src.cli ingest

# 2) 文本嵌入
python -m src.cli embed

# 3) 图像嵌入（OpenCLIP）
python -m src.cli imgindex

# 4) 字 n-gram（中文增强）
python -m src.cli charindex

# 5) 链接图（跨文档）
python -m src.cli links

# 6) （可选）图谱
python -m src.cli graph

# 7) 统一查询（保持参数形态，可按需调整数值）
python -m src.query_cli `
   --link-hop --link-depth 2 --folder-neighbors 1 `
   --mode hybrid --alpha 0.5 --beta 0.3 --gamma 0.8 `
   --mmr 0.5 --neighbors 1 `
   --smart-query --sqw 1.2 `
   --strict-mode off `
   --wclause 0.8 --wn 0.9 --wq 1.0 `
   --wimg 0.25 `
   --top 3 --num-ctx 28192 --ctx-chars 1200 `
   --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\recall-boost-01" `
   --model "qwen3:4b-instruct-2507-fp16" `
   "我最近计划"
```

---

## 3. 索引与产物（`data\` 下生成）
- `chunks.jsonl`：文本切片
- `images.jsonl`：图片清单 + 尺寸 +（可选）OCR/字幕
- `embeddings.npy`：文本嵌入（float32, memmap）
- `ids.json` / `embed_meta.json`：文本 ID 对齐信息与元数据
- `tfidf_vectorizer.joblib` / `tfidf_matrix.joblib`：词面 TF-IDF
- `tfidf_char_vectorizer.joblib` / `tfidf_char_matrix.joblib`：字 n-gram TF-IDF
- `image_embeddings.npy` / `image_ids.json`：图像向量与 ID
- `link_graph.json`：跨文档链接图
- `graph.json`：GraphRAG 图谱（最小实现）

---

## 4. 审计与可观测性
- 文本审计：
```powershell
python -m src.audit --term "罗建祥" --alt "罗健祥" --trad
```
- 图像审计：
```powershell
python -m src.imgaudit --term "发票" --mode clip
```

打开 `--debug-dir` 目录可获得：
- `pipeline.json`：参数与计数
- `candidates.json`：各通道候选与融合分数
- `contexts.txt`：拼接给 LLM 的上下文
- `query.txt`：原始 query、smart 解析关键词、（可选）LLM 扩展
- `images.json`：图像召回详情（相似度、路径、OCR/字幕是否使用）

---

## 5. 常见错误排查
- **/api/embeddings 连接失败**：
  - 确认 `ollama serve` 正在运行；确认 `OLLAMA_HOST` 指向 `127.0.0.1:11434`；
  - `curl http://127.0.0.1:11434/api/tags` 试通；
  - 防火墙/代理是否阻断本地端口。

- **ids 不对齐**：
  - 删除 `data\embeddings.npy` 与 `data\ids.json` 后重建；
  - 确认 `chunks.jsonl` 每一行都有非空文本（空文本会被处理为 `" "` 以对齐）。

- **显存不足（CUDA OOM）**：
  - 将 `image_batch` 调小（如 32）；将 `image_precision` 切换 `fp32`；
  - 降低 `embed_batch`，将 `embed_concurrency` 与 `OLLAMA_NUM_PARALLEL` 调小。

- **端口占用**：
  - 修改 `OLLAMA_HOST` 为其他端口（例如 `127.0.0.1:11500`）。

---

## 6. 许可证
MIT
