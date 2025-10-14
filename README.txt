
nmagr_overlay_v4
================

修复点：
1) argparse 属性名：--vl-answer / --text-first / --disable-clip 全部映射为 args.vl_answer / args.text_first / args.disable_clip，避免 AttributeError。
2) 新增并行 LLM 相关性“初筛”模块（src/llm_filter.py），默认开启，模型 qwen2.5:3b-instruct，--pf-workers 控制并发。
3) 参考路径统一去除 \\?\ 前缀并转为可点击的 Windows 绝对路径。
4) query_cli.py 提供前端封装：若仓库有更完整的 pipeline，将优先调用（import .query_pipeline），否则用 debug/candidates.json 演示 post-filter。

安装：
- 将 zip 解压后，把里面的 src 目录直接覆盖 C:\MyNotion\new_multimodal_agentic_graphrag\src

建议命令：
python -u -m src.query_cli --smart-query --sqw 1.2 --text-first 6 --wtext 1.0 --wimg 0.25 --post-filter --pf-model "qwen2.5:3b-instruct" --pf-workers 6 --top 20 --num-ctx 20000 --ctx-chars 1200 --debug-dir "C:\MyNotion\new_multimodal_agentic_graphrag\debug\liuxiaoling-01" --model "qwen3:4b-instruct-2507-fp16" "刘晓玲的所有记录"
