# BGE-Reranker-v2-m3 Demo

A demo for [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — a lightweight multilingual reranker model.

## Setup

```bash
# Install dependencies
uv sync
```

The model (~568MB) is automatically downloaded to `~/.cache/huggingface/hub/` on first run.

## Usage

### Built-in Demos

Runs 3 multilingual examples (English, Chinese, French) with pre-defined queries and documents:

```bash
# FlagEmbedding backend (default, recommended)
uv run python main.py

# HuggingFace Transformers backend
uv run python main.py --backend transformers
```

Output example:

```
Example 1: What is pandas in Python?
──────────────────────────────────────────────────────────────────────
  >>> #1  score=0.9948  pandas is a fast, powerful, flexible and easy to use open source...
     #2  score=0.8231  NumPy is a library for the Python programming language...
     #3  score=0.3124  The giant panda (Ailuropoda melanoleuca)...
     #4  score=0.0012  Hi, how are you doing today?
```

### Interactive Mode

Enter your own query and documents for live reranking:

```bash
uv run python main.py --interactive
```

Type `quit` to exit.

## Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--backend` | `flagembedding`, `transformers` | `flagembedding` | Backend library |
| `--interactive` | - | `false` | Enable interactive mode |
