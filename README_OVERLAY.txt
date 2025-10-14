
覆盖包说明（仅替换 src/query_cli.py）
===================================

把本包解压到：
C:\MyNotion\new_multimodal_agentic_graphrag\
它会覆盖：src\query_cli.py

主要改动：
- 文本、图片分池检索，文本优先（--text-first）
- 参考只输出一次，附编号与绝对路径
- contexts.txt 必写（图片会写 OCR/字幕/文件名摘要）
- --vl-answer 时，即使初选没有图，也会强制从图通道补图进 qwen2.5vl

示例：
powershell> python -u -m src.query_cli --mode hybrid --smart-query --sqw 1.2 --text-first 6 --wtext 1.0 --wimg 0.25 --top 12 --num-ctx 16000 --ctx-chars 900 --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\moweifen-text" --model "qwen3:4b-instruct-2507-fp16" "刘晓玲的所有记录"

VL 多模态：
powershell> python -u -m src.query_cli --mode hybrid --smart-query --text-first 4 --wtext 1.0 --wimg 0.35 --vl-answer --max-images 4 --clip-local "C:\MyNotion\new_multimodal_agentic_graphrag\models\openclip\CLIP-ViT-L-14-laion2B-s32B-b82K" --top 10 --num-ctx 16000 --ctx-chars 900 --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\metaverse-vl" --model "qwen2.5vl:latest" "元宇宙Metaverse架构师 说了什么"
