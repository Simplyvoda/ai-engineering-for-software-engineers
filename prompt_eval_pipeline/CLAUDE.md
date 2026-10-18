# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A script-based pipeline that compares system prompts for a sentiment classifier (positive / negative / neutral) on rental reviews, using OpenAI `gpt-4o-mini`. It is a learning project: the owner often prefers to write code themselves, so check whether they want an explanation or an implementation before making large changes.

## Commands

All scripts use paths relative to the current directory, so run them from the repo root.

```bash
source .venv/bin/activate          # this folder's own env; the parent folder has a different .venv without these packages
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
python evaluation_system.py        # 30 API calls; overwrites evaluation_results.json, adds runs/run_<timestamp>.json
python analyze_result.py           # no API; reads evaluation_results.json + test_cases.csv
python visualize.py                # no API; reads runs/, writes charts/accuracy_over_time.png
python ab_test.py                  # 80 API calls; writes ab_test_results.json
```

There is no test suite, linter or build step.

### Checking changes without an API key

`evaluation_system.py` and `ab_test.py` each create a module-level `client = OpenAI()`, which raises at import time if `OPENAI_API_KEY` is unset. To exercise the code offline, set a dummy key, import the module, and replace `module.client` with a fake whose `chat.completions.create(...)` returns saved outputs (for example, those in `evaluation_results.json`, keyed by system prompt text and review text). Do this in a copy of the folder so `runs/` and `evaluation_results.json` are not polluted with fake runs.

## Architecture

The scripts form a pipeline and communicate only through files:

```
test_cases.csv + prompts/*.txt
        │
evaluation_system.py ──► evaluation_results.json   (latest run only, flat list of results)
        │            └─► runs/run_<YYYYMMDD_HHMMSS>.json   (history: model, prompt versions, summaries, results)
        │
        ├── analyze_result.py   reads evaluation_results.json
        ├── visualize.py        reads runs/
        └── ab_test.py          imports MODEL, load_prompts, ReviewClassifierEvaluator
```

- **Prompts are data, not code.** Each file in `prompts/` is one system prompt; the file name minus `.txt` is the prompt name used everywhere else. Adding a prompt means adding a file. `prompt_version()` is a hash of the text, stored with each run so results can be tied to the exact prompt wording.
- **`metrics.py` is pure functions with no I/O or API calls.** `extract_label()` turns a free-text model reply into one of the three labels or `"unknown"` (e.g. the model answering "mixed"). It uses the single mentioned label if there is only one, otherwise the last cue phrase such as "Overall sentiment: X" or "leaning towards X". New reply formats may need new cues in `_CUE_RE`.
- **Two scoring systems coexist on every result.** The original fields (`exact_match`, `semantic_similarity`, `score`) are kept so `analyze_result.py` keeps working, but `semantic_similarity` is known to be flawed: its fallback matches the last three letters of the expected label, so "negative" earns partial credit when "positive" was expected. The trustworthy fields are `predicted_label`, `label_match` and the per-label precision/recall/F1. Ranking in `compare_prompts`, `print_failures`, the chart and the A/B test all use the label-based metric.
- **Column naming differs between files.** The review id is `id` in `test_cases.csv` / `TestCase` and `test_id` in results. `difficulty` and `category` exist only in the CSV; `analyze_result.py` joins them in by id.
- **`TestCase(**row)` requires CSV headers to match the dataclass fields exactly**; adding a CSV column means adding a field.
- **`ab_test.py` simulates production traffic** by replaying the test reviews through a random A/B split. `get_winner()` refuses to name a winner below `min_samples` per arm or when a two-proportion z-test is not significant.
- **`visualize.py`** assigns each prompt a fixed colour, marker and dash pattern in name order; dashes exist so that prompts with identical accuracy remain visible when their lines overlap.

## Known limits

- Only 10 test cases, so one changed answer moves accuracy by 10 points; treat prompt rankings as indicative.
- Runs before versioning was added were overwritten; `runs/` starts from the 2026-10-08 16:27 run.
