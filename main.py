"""
BGE-Reranker-v2-m3 Demo
Demonstrates multilingual reranking using BAAI/bge-reranker-v2-m3.
Supports two backends: FlagEmbedding (recommended) and HuggingFace Transformers.
"""

import argparse
import sys

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
    {
        "query": "Comment faire un gâteau au chocolat?",
        "documents": [
            "Pour faire un gâteau au chocolat, vous aurez besoin de chocolat noir, de beurre, d'œufs, de sucre et de farine.",
            "La Tour Eiffel a été construite par Gustave Eiffel pour l'Exposition universelle de 1889.",
            "Le machine learning est un domaine de l'intelligence artificielle qui permet aux ordinateurs d'apprendre sans être explicitement programmés.",
            "Faire du sport régulièrement est important pour maintenir une bonne santé.",
        ],
    },
]


def run_flagembedding_demo(normalize: bool = True):
    """Run demo using the FlagEmbedding library (recommended approach)."""
    from FlagEmbedding import FlagReranker

    print("=" * 70)
    print("BGE-Reranker-v2-m3 Demo (FlagEmbedding backend)")
    print("=" * 70)
    print("\nLoading model BAAI/bge-reranker-v2-m3 ...")

    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    for i, example in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Example {i}: {example['query']}")
        print(f"{'─' * 70}")

        pairs = [[example["query"], doc] for doc in example["documents"]]
        scores = reranker.compute_score(pairs, normalize=normalize)

        ranked = sorted(
            zip(scores, example["documents"], strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        for rank, (score, doc) in enumerate(ranked, 1):
            marker = ">>>" if rank == 1 else "   "
            print(f"  {marker} #{rank}  score={score:.4f}  {doc[:80]}...")


def run_transformers_demo(normalize: bool = True):
    """Run demo using HuggingFace transformers directly."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 70)
    print("BGE-Reranker-v2-m3 Demo (HuggingFace Transformers backend)")
    print("=" * 70)
    print("\nLoading model BAAI/bge-reranker-v2-m3 ...")

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
    model.eval()

    for i, example in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Example {i}: {example['query']}")
        print(f"{'─' * 70}")

        pairs = [[example["query"], doc] for doc in example["documents"]]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1).float().tolist()

        if normalize:
            import math
            scores = [1 / (1 + math.exp(-s)) for s in scores]

        ranked = sorted(
            zip(scores, example["documents"], strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        for rank, (score, doc) in enumerate(ranked, 1):
            marker = ">>>" if rank == 1 else "   "
            print(f"  {marker} #{rank}  score={score:.4f}  {doc[:80]}...")


def run_interactive(backend: str = "flagembedding"):
    """Interactive mode: enter your own query and documents."""
    if backend == "flagembedding":
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        model.eval()

    print("\nInteractive Reranker (type 'quit' to exit)")
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

        pairs = [[query, doc] for doc in documents]

        if backend == "flagembedding":
            scores = reranker.compute_score(pairs, normalize=True)
        else:
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
                raw = model(**inputs, return_dict=True).logits.view(-1).float().tolist()
                import math
                scores = [1 / (1 + math.exp(-s)) for s in raw]

        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(
            zip(scores, documents, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        print(f"\nResults for: {query}")
        for rank, (score, doc) in enumerate(ranked, 1):
            marker = ">>>" if rank == 1 else "   "
            print(f"  {marker} #{rank}  score={score:.4f}  {doc}")


def main():
    parser = argparse.ArgumentParser(description="BGE-Reranker-v2-m3 Demo")
    parser.add_argument(
        "--backend",
        choices=["flagembedding", "transformers"],
        default="flagembedding",
        help="Backend library to use (default: flagembedding)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive(backend=args.backend)
    elif args.backend == "flagembedding":
        run_flagembedding_demo()
    else:
        run_transformers_demo()


if __name__ == "__main__":
    main()
