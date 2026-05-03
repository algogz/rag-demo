"""
Qwen3-VL Embedding & Reranker Demo
Demonstrates multimodal (text + image) embedding and reranking using
Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B.
Models are stored in ./models/ directory.
"""

import argparse
import os
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).parent / "models"

EMBEDDING_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"

DEMO_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def model_path(model_id: str) -> str:
    """Return local path under ./models/ if already downloaded, else the HF model ID."""
    local = MODEL_DIR / model_id.replace("/", "--")
    if local.exists():
        return str(local)
    os.makedirs(MODEL_DIR, exist_ok=True)
    return model_id


def format_doc(doc) -> str:
    """Format a document (str, URL, or dict) for display."""
    if isinstance(doc, str):
        if doc.startswith(("http://", "https://")):
            return f"[image] {doc}"
        return doc
    if isinstance(doc, dict):
        parts = []
        if "text" in doc:
            parts.append(doc["text"])
        if "image" in doc:
            img = doc["image"]
            parts.append(f"[image: ...{img[-40:]}]")
        return " | ".join(parts) if parts else str(doc)
    return str(doc)


DEMO_QUERIES = [
    {
        "task": "Retrieve images or text relevant to the user's query.",
        "query": "A woman playing with her dog on a beach at sunset.",
        "documents": [
            "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
            DEMO_IMAGE_URL,
            {
                "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
                "image": DEMO_IMAGE_URL,
            },
            "A bustling city skyline illuminated by neon lights reflecting off a river at night.",
        ],
    },
    {
        "task": "Retrieve images or text relevant to the user's query.",
        "query": "一只猫坐在窗台上晒太阳",
        "documents": [
            "一只橘色的猫咪正趴在窗台上，阳光洒在它的毛发上，看起来非常惬意。",
            "今天天气非常好，适合出去散步。",
            {
                "text": "一只黑白相间的猫趴在窗台上看风景。",
                "image": DEMO_IMAGE_URL,
            },
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        ],
    },
]


# ─── Embedding Demo ────────────────────────────────────────────────────────────


def run_embedding_demo():
    """Embedding demo using Sentence Transformers."""
    from sentence_transformers import SentenceTransformer

    print("=" * 70)
    print("Qwen3-VL-Embedding-2B Demo (Sentence Transformers)")
    print("=" * 70)

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    model = SentenceTransformer(mp, cache_folder=str(MODEL_DIR))

    for i, example in enumerate(DEMO_QUERIES, 1):
        query_emb = model.encode([example["query"]], prompt=example["task"])
        doc_emb = model.encode(example["documents"])
        similarity = model.similarity(query_emb, doc_emb)[0].tolist()

        print_ranking("Embedding", example["query"], similarity, example["documents"], i)


# ─── Reranker Demo ─────────────────────────────────────────────────────────────


def run_reranker_demo():
    """Reranker demo using Sentence Transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    print("=" * 70)
    print("Qwen3-VL-Reranker-2B Demo (Sentence Transformers)")
    print("=" * 70)

    mp = model_path(RERANKER_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    model = CrossEncoder(mp, cache_folder=str(MODEL_DIR))

    for i, example in enumerate(DEMO_QUERIES, 1):
        pairs = [(example["query"], doc) for doc in example["documents"]]
        scores = model.predict(pairs, prompt=example["task"], activation_fn=torch.nn.Sigmoid()).tolist()

        print_ranking("Reranker", example["query"], scores, example["documents"], i)


# ─── Shared Helpers ────────────────────────────────────────────────────────────


def print_ranking(mode, query, scores, documents, example_idx=None):
    ranked = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
    label = f"{mode} Example {example_idx}" if example_idx else mode
    print(f"\n{'─' * 70}")
    print(f"{label} | Query: {query}")
    print(f"{'─' * 70}")
    for rank, (score, doc) in enumerate(ranked, 1):
        marker = ">>>" if rank == 1 else "   "
        doc_str = format_doc(doc)
        display = doc_str if len(doc_str) <= 80 else doc_str[:80] + "..."
        print(f"  {marker} #{rank}  score={score:.4f}  {display}")


# ─── Interactive Mode ──────────────────────────────────────────────────────────


def run_interactive(mode: str):
    if mode in ("embedding", "embedding-st"):
        from sentence_transformers import SentenceTransformer
        mp = model_path(EMBEDDING_MODEL_ID)
        print(f"\nLoading embedding model from: {mp} ...")
        model = SentenceTransformer(mp, cache_folder=str(MODEL_DIR))
    else:
        from sentence_transformers import CrossEncoder
        mp = model_path(RERANKER_MODEL_ID)
        print(f"\nLoading reranker model from: {mp} ...")
        model = CrossEncoder(mp, cache_folder=str(MODEL_DIR))

    print("\nInteractive Qwen3-VL Demo (type 'quit' to exit)")
    print("Document prefixes: 'img:<url>' = image, 'mix:<url> <text>' = text+image")
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
            if doc.startswith("img:"):
                documents.append(doc[4:])
            elif doc.startswith("mix:"):
                parts = doc[4:].split(" ", 1)
                documents.append({"text": parts[1] if len(parts) > 1 else "", "image": parts[0]})
            else:
                documents.append(doc)

        if not documents:
            print("No documents provided, skipping.")
            continue

        if mode in ("embedding", "embedding-st"):
            query_emb = model.encode([query], prompt="Retrieve relevant content.")
            doc_emb = model.encode(documents)
            scores = model.similarity(query_emb, doc_emb)[0].tolist()
        else:
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(
                pairs, prompt="Retrieve relevant content.", activation_fn=torch.nn.Sigmoid()
            ).tolist()

        ranked = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
        print(f"\nResults for: {query}")
        for rank, (score, doc) in enumerate(ranked, 1):
            marker = ">>>" if rank == 1 else "   "
            print(f"  {marker} #{rank}  score={score:.4f}  {format_doc(doc)}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Embedding & Reranker Demo")
    parser.add_argument(
        "--mode",
        choices=["embedding-st", "reranker-st"],
        default="embedding-st",
        help=(
            "Demo mode: "
            "'embedding-st' = Sentence Transformers embedding (default), "
            "'reranker-st' = Sentence Transformers reranker"
        ),
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(mode=args.mode)
    elif args.mode == "embedding-st":
        run_embedding_demo()
    elif args.mode == "reranker-st":
        run_reranker_demo()


if __name__ == "__main__":
    main()
