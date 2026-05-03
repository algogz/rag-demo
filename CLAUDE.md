# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Demo collection for text embedding and reranking models. Each script is self-contained: it loads a model, runs example queries, and prints ranked results. All models are downloaded to `./models/` on first run (git-ignored).

## Commands

```bash
uv sync                    # Install dependencies
uv run python bge-demo.py  # Run BGE demos (see README for --mode flags)
uv run python qwen-demo.py # Run Qwen3 demos (see README for --mode flags)
```

Both scripts accept `--mode` to select backend/demo type and `--interactive` for an interactive REPL.

## Architecture

Two standalone scripts with no shared modules:

- **bge-demo.py** — BAAI/bge-m3 (dense/sparse/ColBERT embedding) and BAAI/bge-reranker-v2-m3 (cross-encoder reranking). Uses `FlagEmbedding` library for the primary backend, `transformers` as alternative.
- **qwen-demo.py** — Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B. Uses `sentence-transformers` as primary backend, `transformers` as alternative. Qwen3 reranker uses causal LM logits over "yes"/"no" tokens.

Both scripts share the same pattern: `model_path()` resolves local cache vs HF download, `DEMO_QUERIES` provides bilingual (EN/ZH) test data, `print_ranking()` formats output.

## Key Dependencies

- `FlagEmbedding` — BGE model loading and scoring
- `sentence-transformers` — Qwen3 SentenceTransformer/CrossEncoder backends
- `transformers` — lower-level model backends for both BGE and Qwen3
- Python 3.14, managed via `uv`
