# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts ingestion, indexing, querying, and chat entry points (`cli.py`, `query_cli.py`, `agent_chat.py`); keep shared helpers in `utils.py`.
- `src/tests/` contains pytest suites covering ingestion outputs, HTML linking, post-filter logic, and agent chat flows.
- Runtime artifacts live in `data/`, `index/`, and `debug/`; treat them as disposable and exclude them from commits.
- `scripts/` offers PowerShell helpers such as `quickstart.ps1`, `commands_cheatsheet.ps1`, and `sanity_checks.ps1` for orchestrating end-to-end runs.
- Prompt templates reside in `prompts/`; MCP automation assets ship under `mcpsrc运维包/`; chat transcripts drop into `chat_sessions/`.

## Build, Test, and Development Commands
- Install dependencies with `python -m pip install -r requirements.txt` inside a Python 3.12 virtual environment defined by `.env.example`.
- Drive ingestion and indexing via `python -m src.cli ingest|embed|imgindex|charindex|graph --config config.yaml`; use `python -m src.cli all --incremental` for an end-to-end refresh.
- Execute hybrid retrieval locally with `python -m src.query_cli --mode hybrid --post-filter --debug-dir debug\session-01 --model "qwen3:4b-instruct-2507-fp16"`; adjust embeddings (`--embed-model`, `--clip-local`) to match local GPU assets.
- For scripted pipelines, reuse `.\scripts\quickstart.ps1` or tailor `commands_cheatsheet.ps1` parameters to your workspace paths and Ollama endpoint.

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indentation, snake_case module and function names, and descriptive constants (see `SMART_QUERY_PROMPT` in `src/utils.py`).
- Prefer type hints on new public APIs and keep CLI flags aligned with existing argparse naming.
- Route logging through `utils.setup_logger` to maintain rotating file handlers under `debug/console.log`.
- Store configuration in lowercase YAML keys; new prompt files should mirror the existing `keyword_extractor_zh.txt` naming pattern.

## Testing Guidelines
- Run `python -m pytest -q` before raising a pull request; target the `src/tests/` package layout (`test_<feature>.py`).
- Expand coverage by mirroring existing fixtures when adding ingestion or post-filter logic; capture failing artifacts in a fresh `debug/<case>` folder.
- Use `python tools/coverage.py --root <source_root> --chunks data\chunks.jsonl --out debug\coverage.json` to audit content coverage after large ingests.

## Commit & Pull Request Guidelines
- Keep commits focused and imperatively titled (`multi-turn chat`, `snapshot: <timestamp>` reflects current history); avoid bundling unrelated fixes.
- PR descriptions should summarize the pipeline touched, list commands executed (ingest, embed, query, tests), and note any new models, ports, or environment variables.
- Link related issues, attach representative `debug/` snippets or screenshots for UI-affecting changes, and highlight updates to `requirements.txt` or configuration files for reviewer visibility.
