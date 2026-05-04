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

## Qwen3-VL Embedding & Reranker

[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) — multimodal embedding (text + image + video, 2048-dim).
[Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) — multimodal reranker for cross-modal retrieval.

```bash
# Embedding demo (text, image, and mixed inputs)
uv run python qwen-vl-demo.py

# Reranker demo
uv run python qwen-vl-demo.py --mode reranker-st

# Interactive mode
uv run python qwen-vl-demo.py --interactive
uv run python qwen-vl-demo.py --interactive --mode reranker-st
```

## Qwen3-VL Image Search

[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B), [sqlite-vec](https://github.com/asg017/sqlite-vec).

Two-stage retrieval: vector similarity search followed by cross-encoder reranking.

```bash
# Embed images (file or directory, recursive)
uv run python img-search.py embed /path/to/images

# Semantic search (with reranking)
uv run python img-search.py search "a sunset over the ocean"

# Vector search only (skip reranking)
uv run python img-search.py search "风景照片" --no-rerank --top-k 10
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

### qwen-vl-demo.py

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--mode` | `embedding-st`, `reranker-st` | `embedding-st` | Model type |
| `--interactive` | — | `false` | Interactive mode (supports `img:<url>` and `mix:<url> <text>` docs) |

### img-search.py

| Command | Flags | Description |
|---------|-------|-------------|
| `embed <path>` | — | Embed images from file or directory (skips already embedded) |
| `search <desc>` | `--top-k N` | Search with reranking (default top-k: 20) |
| `search <desc>` | `--no-rerank` | Vector similarity only, skip reranker |
