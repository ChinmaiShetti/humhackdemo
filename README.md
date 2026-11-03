# HumanAlign-RAG

Neighbour-grounded query judging for human-aligned synthetic evaluation of multilingual and code-mixed retrieval.

Status: Phase 0 (setup). The spec is in [docs/HUMANALIGN_RAG_BUILD_SPEC.md](docs/HUMANALIGN_RAG_BUILD_SPEC.md), and the execution plan and status tracker are in [docs/PLAN.md](docs/PLAN.md).

## Setup (Windows, Python 3.11, CUDA)

```
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
copy .env.example .env
.venv\Scripts\ha doctor
```

`ha doctor` must report all checks OK. Set `PYTHONUTF8=1` as a user environment variable (for Devanagari I/O).

## Reproducing results

To be filled in from Phase 8 onwards: a map from each table and figure to the command that produces it. Every number comes from `results/registry.jsonl`.
