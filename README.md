# new_multimodal_agentic_graphrag (Quality+OCR Upgrade)

本版在你现有项目基础上做了**检索/回答质量增强**与**OCR 打开**：
- 默认开启 **OCR**，并把 **图片的 OCR 文本作为文本 chunk** 一并入库（真正实现“文本+图片（非仅 OCR）”融合召回）。
- 新增 **关键词与同义词扩展**（含 AGI/ASI/LLM 的中英别名），支持**关键词组合（pair）布尔通道**参与融合打分，显著提升“概念词/缩写”的召回率与精度。
- `strict-mode smart` 更聪明：基于扩展后的关键词集合做 AND 过滤。
- 进度条、长路径、坏图回退、并发与 memmap 等之前优化全部保留。

---

## 0) 安装与运行环境
与之前一致（Windows 11 + PowerShell + Python 3.12）。依赖见 `requirements.txt`。

Ollama 建议模型：
- 嵌入：`nomic-embed-text:latest`
- 生成（可选）：`qwen3:4b-instruct-2507-fp16`
- 多模态字幕（可选）：`qwen2.5vl:latest`

---

## 1) 配置（已默认开启 OCR）
`config.yaml` 关键项：
```yaml
enable_ocr: true
ocr_lang: "chi_sim+eng"
ocr_psm: 6

# 中文更强：
tfidf_max_features: 200000
char_ngram_range: [2, 6]
```

> 本版在 ingest 时会将 `images.jsonl` 的 OCR 文本**同步写入** `chunks.jsonl`（modality:`image_ocr`），因此后续 `embed/tfidf/charindex` 都会覆盖这些 OCR 文本。

---

## 2) 从 0 到检索（建议全量重建）
```powershell
# 0) 启动 Ollama（另一个窗口常驻）
$env:OLLAMA_HOST="127.0.0.1:11434"
$env:OLLAMA_NUM_PARALLEL="2"
$env:OLLAMA_KEEP_ALIVE="10m"
ollama serve

# 1) ingest（默认开启 OCR）
python -m src.cli ingest

# 2) 文本嵌入（带进度）
python -m src.cli embed

# 3) 图像向量（OpenCLIP；带进度；已修复 Windows DataLoader 闭包问题）
python -m src.cli imgindex

# 4) 字 n-gram（带进度）
python -m src.cli charindex

# 5) 链接图
python -m src.cli links

# 6) （可选）图谱
python -m src.cli graph
```

---

## 3) 新查询建议（AGI/ASI 类问题更准）
新增 `--wbool`（关键词/组合布尔通道权重，默认 0.6），并内置 AGI/ASI 中文同义词扩展：

```powershell
python -m src.query_cli `
  --mode hybrid --alpha 0.4 --beta 0.6 --gamma 1.0 `
  --wimg 0.10 --wbool 0.8 `
  --top 10 --mmr 0.35 --neighbors 2 `
  --smart-query --sqw 1.3 `
  --strict-mode off `
  --ctx-chars 800 --num-ctx 22000 `
  --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\agi-tune-01" `
  "asi\\agi 的路径、方法、思路、路线、理论、技术，以及对他们的控制？"
```

> 说明：查询会自动扩展为：`AGI/ASI/通用人工智能/强人工智能/超人工智能/超级智能/…`，并把**成对组合**（pair）用于布尔通道打分，融合到最终排序。`debug\query.txt` 会记录 `expanded_terms` 与 `combos`，可审计。

---

## 4) 质量提升思路（已经内置）
- **多通道融合**：词面 TF-IDF、字 n-gram、语义向量、图像向量、关键词组合布尔（新）。
- **中文增强**：字符 n-gram 扩到 `[2,6]`；空白/编码容错；HTML 标题/目录抽取。
- **图片可见性**：OCR 文本进入文本通道（不依赖字幕也能命中）；CLIP 跨模态检索保留。
- **可观测性**：`debug/` 落盘记录关键参数与候选；可快速调权重。

---

## 5) 常见问答
- **为什么之前没命中 AGI/ASI？**  
  原因通常是：你的库里中文文档更常写“通用人工智能/强人工智能”等，而纯 `AGI/ASI` 英文缩写较少；现已自动**同义词扩展**与**组合布尔**打分，大幅提升召回。
- **OCR 很多图片文本也参与检索了吗？**  
  是的。打开 `enable_ocr: true` 后，图片的 OCR 文本会在 `chunks.jsonl` 生成对应记录（`modality: image_ocr`），与普通文本一样被嵌入与 TF-IDF 化。

---

## 6) 速度与稳定性
- ingest/嵌入/图像索引均带进度条；并发+memmap；OpenCLIP AMP 混精度。  
- Windows 长路径 `\\?\` 兼容；坏图回退白图不阻塞。

---

## 7) 许可证
MIT
