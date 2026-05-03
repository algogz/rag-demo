# BGM Demo

Demos for text embedding and reranking models.

## Setup

```bash
uv sync
```

Models are downloaded automatically to `./models/` on first run.

## BGE Demo

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — multilingual embedding (dense, sparse, ColBERT).
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — lightweight multilingual reranker.

```bash
# Embedding demos
uv run python bge-demo.py --mode embedding           # Dense + Sparse + ColBERT
uv run python bge-demo.py --mode embedding-score      # Combined pair scoring

# Reranker demos
uv run python bge-demo.py                            # FlagEmbedding reranker (default)
uv run python bge-demo.py --mode reranker-tf         # Transformers reranker

# Interactive mode
uv run python bge-demo.py --interactive
uv run python bge-demo.py --interactive --mode embedding
```

## Qwen3 Embedding & Reranker

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — text embedding (100+ languages, 1024-dim).
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) — text reranking (100+ languages).

```bash
# Embedding demos
uv run python qwen-demo.py                           # Sentence Transformers (default)
uv run python qwen-demo.py --mode embedding           # Transformers backend

# Reranker demos
uv run python qwen-demo.py --mode reranker-st         # Sentence Transformers
uv run python qwen-demo.py --mode reranker            # Transformers backend

# Interactive mode
uv run python qwen-demo.py --interactive
uv run python qwen-demo.py --interactive --mode reranker-st
```

## Options

### bge-demo.py

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--mode` | `embedding`, `embedding-score`, `reranker`, `reranker-tf` | `reranker` | Model and backend |
| `--interactive` | — | `false` | Interactive mode |

### qwen-demo.py

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--mode` | `embedding-st`, `embedding`, `reranker-st`, `reranker` | `embedding-st` | Model and backend |
| `--interactive` | — | `false` | Interactive mode |
