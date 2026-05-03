"""
BGE Demo — Embedding & Reranking
- BAAI/bge-m3: dense / sparse / ColBERT embedding
- BAAI/bge-reranker-v2-m3: cross-encoder reranking
Models are stored in ./models/ directory.
"""

import argparse
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


def model_path(model_id: str) -> str:
    """Return local path under ./models/ if already downloaded, else the HF model ID."""
    local = MODEL_DIR / model_id.replace("/", "--")
    if local.exists():
        return str(local)
    os.makedirs(MODEL_DIR, exist_ok=True)
    return model_id


DEMO_QUERIES = [
    {
        "query": "What is pandas in Python?",
        "documents": [
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool built on top of the Python programming language.",
            "Hi, how are you doing today?",
            "NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.",
        ],
    },
    {
        "query": "什么是机器学习？",
        "documents": [
            "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进。",
            "深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的模式识别任务。",
            "今天天气非常好，适合出去散步。",
            "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。",
        ],
    },
]


def print_ranking(label, query, scores, documents):
    ranked = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
    print(f"\n{'─' * 70}")
    print(f"{label} | Query: {query}")
    print(f"{'─' * 70}")
    for rank, (score, doc) in enumerate(ranked, 1):
        marker = ">>>" if rank == 1 else "   "
        print(f"  {marker} #{rank}  score={score:.4f}  {doc[:80]}...")


# ─── Embedding Demo ────────────────────────────────────────────────────────────


def run_embedding_demo():
    """Embedding demo using BGE-M3 (dense, sparse, ColBERT)."""
    from FlagEmbedding import BGEM3FlagModel

    print("=" * 70)
    print("BGE-M3 Embedding Demo")
    print("=" * 70)

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    model = BGEM3FlagModel(mp, use_fp16=True, cache_dir=str(MODEL_DIR))

    for example in DEMO_QUERIES:
        query_emb = model.encode([example["query"]], return_dense=True, return_sparse=True, return_colbert_vecs=True)
        doc_emb = model.encode(example["documents"], return_dense=True, return_sparse=True, return_colbert_vecs=True)

        # Dense similarity
        dense_scores = (query_emb["dense_vecs"] @ doc_emb["dense_vecs"].T)[0].tolist()

        # Sparse (lexical matching) scores
        sparse_scores = [
            model.compute_lexical_matching_score(query_emb["lexical_weights"][0], doc_emb["lexical_weights"][j])
            for j in range(len(example["documents"]))
        ]

        # ColBERT scores
        colbert_scores = [
            model.colbert_score(query_emb["colbert_vecs"][0], doc_emb["colbert_vecs"][j]).item()
            for j in range(len(example["documents"]))
        ]

        print_ranking("Dense", example["query"], dense_scores, example["documents"])
        print_ranking("Sparse", example["query"], sparse_scores, example["documents"])
        print_ranking("ColBERT", example["query"], colbert_scores, example["documents"])

    # Show token weights for the first query
    print(f"\n{'─' * 70}")
    print("Sparse token weights (first query):")
    print(f"{'─' * 70}")
    query_emb = model.encode([DEMO_QUERIES[0]["query"]], return_sparse=True)
    tokens = model.convert_id_to_token(query_emb["lexical_weights"])
    for i, tw in enumerate(tokens):
        print(f"  {tw}")


def run_embedding_score_demo():
    """Compute combined scores for text pairs using BGE-M3."""
    from FlagEmbedding import BGEM3FlagModel

    print("=" * 70)
    print("BGE-M3 Pair Scoring Demo")
    print("=" * 70)

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    model = BGEM3FlagModel(mp, use_fp16=True, cache_dir=str(MODEL_DIR))

    for example in DEMO_QUERIES:
        pairs = [[example["query"], doc] for doc in example["documents"]]
        result = model.compute_score(
            pairs,
            max_passage_length=512,
            weights_for_different_modes=[0.4, 0.2, 0.4],  # dense, sparse, colbert
        )

        print_ranking("Combined (dense+sparse+colbert)", example["query"], result["colbert+sparse+dense"], example["documents"])
        print(f"  Dense scores:   {[f'{s:.4f}' for s in result['dense']]}")
        print(f"  Sparse scores:  {[f'{s:.4f}' for s in result['sparse']]}")
        print(f"  ColBERT scores: {[f'{s:.4f}' for s in result['colbert']]}")


# ─── Reranker Demo ─────────────────────────────────────────────────────────────


def run_reranker_demo(normalize: bool = True):
    from FlagEmbedding import FlagReranker

    print("=" * 70)
    print("BGE-Reranker-v2-m3 Demo (FlagEmbedding backend)")
    print("=" * 70)

    mp = model_path(RERANKER_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    reranker = FlagReranker(mp, use_fp16=True, cache_dir=str(MODEL_DIR))

    for example in DEMO_QUERIES:
        pairs = [[example["query"], doc] for doc in example["documents"]]
        scores = reranker.compute_score(pairs, normalize=normalize)
        print_ranking("Reranker", example["query"], scores, example["documents"])


def run_reranker_transformers_demo(normalize: bool = True):
    import math

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 70)
    print("BGE-Reranker-v2-m3 Demo (HuggingFace Transformers backend)")
    print("=" * 70)

    mp = model_path(RERANKER_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    tokenizer = AutoTokenizer.from_pretrained(mp, cache_dir=str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(mp, cache_dir=str(MODEL_DIR))
    model.eval()

    for example in DEMO_QUERIES:
        pairs = [[example["query"], doc] for doc in example["documents"]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1).float().tolist()

        if normalize:
            scores = [1 / (1 + math.exp(-s)) for s in scores]

        print_ranking("Reranker", example["query"], scores, example["documents"])


# ─── Interactive Mode ──────────────────────────────────────────────────────────


def run_interactive(mode: str):
    if mode == "embedding":
        from FlagEmbedding import BGEM3FlagModel
        mp = model_path(EMBEDDING_MODEL_ID)
        print(f"\nLoading embedding model from: {mp} ...")
        model = BGEM3FlagModel(mp, use_fp16=True, cache_dir=str(MODEL_DIR))
    elif mode == "reranker":
        from FlagEmbedding import FlagReranker
        mp = model_path(RERANKER_MODEL_ID)
        print(f"\nLoading reranker model from: {mp} ...")
        reranker = FlagReranker(mp, use_fp16=True, cache_dir=str(MODEL_DIR))

    print("\nInteractive BGE Demo (type 'quit' to exit)")
    print("=" * 70)

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        print("Enter documents (one per line, empty line to finish):")
        documents = []
        while True:
            doc = input(f"  [{len(documents) + 1}] ").strip()
            if not doc:
                break
            documents.append(doc)

        if not documents:
            print("No documents provided, skipping.")
            continue

        if mode == "embedding":
            query_emb = model.encode([query])
            doc_emb = model.encode(documents)
            scores = (query_emb["dense_vecs"] @ doc_emb["dense_vecs"].T)[0].tolist()
        else:
            pairs = [[query, doc] for doc in documents]
            scores = reranker.compute_score(pairs, normalize=True)

        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
        print(f"\nResults for: {query}")
        for rank, (score, doc) in enumerate(ranked, 1):
            marker = ">>>" if rank == 1 else "   "
            print(f"  {marker} #{rank}  score={score:.4f}  {doc}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="BGE Demo — Embedding & Reranking")
    parser.add_argument(
        "--mode",
        choices=["embedding", "embedding-score", "reranker", "reranker-tf"],
        default="reranker",
        help=(
            "'embedding' = BGE-M3 dense/sparse/colbert demo, "
            "'embedding-score' = BGE-M3 pair scoring demo, "
            "'reranker' = BGE-Reranker FlagEmbedding (default), "
            "'reranker-tf' = BGE-Reranker Transformers"
        ),
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(mode=args.mode if args.mode in ("embedding", "reranker") else "reranker")
    elif args.mode == "embedding":
        run_embedding_demo()
    elif args.mode == "embedding-score":
        run_embedding_score_demo()
    elif args.mode == "reranker":
        run_reranker_demo()
    elif args.mode == "reranker-tf":
        run_reranker_transformers_demo()


if __name__ == "__main__":
    main()
