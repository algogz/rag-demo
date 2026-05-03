"""
Qwen3 Embedding & Reranker Demo
Demonstrates text embedding and reranking using Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B.
Models are stored in ./models/ directory.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DIR = Path(__file__).parent / "models"

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"


def model_path(model_id: str) -> str:
    """Return local path under ./models/ if already downloaded, else the HF model ID."""
    local = MODEL_DIR / model_id.replace("/", "--")
    if local.exists():
        return str(local)
    os.makedirs(MODEL_DIR, exist_ok=True)
    return model_id


def get_detailed_instruct(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery:{query}"


DEMO_QUERIES = [
    {
        "task": "Given a web search query, retrieve relevant passages that answer the query",
        "query": "What is pandas in Python?",
        "documents": [
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool built on top of the Python programming language.",
            "Hi, how are you doing today?",
            "NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.",
        ],
    },
    {
        "task": "根据搜索查询，检索相关的回答段落",
        "query": "什么是机器学习？",
        "documents": [
            "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进。",
            "深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的模式识别任务。",
            "今天天气非常好，适合出去散步。",
            "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。",
        ],
    },
]


# ─── Embedding Demo ────────────────────────────────────────────────────────────


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def run_embedding_demo_sentence_transformers():
    """Embedding demo using Sentence Transformers (recommended)."""
    from sentence_transformers import SentenceTransformer

    print("=" * 70)
    print("Qwen3-Embedding-0.6B Demo (Sentence Transformers)")
    print("=" * 70)

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    model = SentenceTransformer(mp, cache_folder=str(MODEL_DIR))

    for i, example in enumerate(DEMO_QUERIES, 1):
        query_emb = model.encode([example["query"]], prompt_name="query")
        doc_emb = model.encode(example["documents"])
        similarity = model.similarity(query_emb, doc_emb)[0].tolist()

        print_ranking("Embedding", example["query"], similarity, example["documents"], i)


def run_embedding_demo_transformers():
    """Embedding demo using HuggingFace Transformers."""
    from transformers import AutoModel, AutoTokenizer

    print("=" * 70)
    print("Qwen3-Embedding-0.6B Demo (HuggingFace Transformers)")
    print("=" * 70)

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="left", cache_dir=str(MODEL_DIR))
    model = AutoModel.from_pretrained(mp, cache_dir=str(MODEL_DIR))
    model.eval()

    for i, example in enumerate(DEMO_QUERIES, 1):
        instructed_query = get_detailed_instruct(example["task"], example["query"])
        all_texts = [instructed_query] + example["documents"]

        batch_dict = tokenizer(all_texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_emb = embeddings[:1]
        doc_emb = embeddings[1:]
        similarity = (query_emb @ doc_emb.T)[0].tolist()

        print_ranking("Embedding", example["query"], similarity, example["documents"], i)


# ─── Reranker Demo ─────────────────────────────────────────────────────────────


def format_rerank_instruction(instruction, query, doc):
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"Instruct: {instruction}\nQuery: {query}\nDocument: {doc}"


def run_reranker_demo_sentence_transformers():
    """Reranker demo using Sentence Transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    print("=" * 70)
    print("Qwen3-Reranker-0.6B Demo (Sentence Transformers)")
    print("=" * 70)

    mp = model_path(RERANKER_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    model = CrossEncoder(mp, cache_folder=str(MODEL_DIR))

    for i, example in enumerate(DEMO_QUERIES, 1):
        pairs = [(example["query"], doc) for doc in example["documents"]]
        scores = model.predict(pairs, activation_fn=torch.nn.Sigmoid()).tolist()

        print_ranking("Reranker", example["query"], scores, example["documents"], i)


def run_reranker_demo_transformers():
    """Reranker demo using HuggingFace Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Qwen3-Reranker-0.6B Demo (HuggingFace Transformers)")
    print("=" * 70)

    mp = model_path(RERANKER_MODEL_ID)
    print(f"\nLoading model from: {mp} ...")
    tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="left", cache_dir=str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(mp, cache_dir=str(MODEL_DIR)).eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    for i, example in enumerate(DEMO_QUERIES, 1):
        pairs = [format_rerank_instruction(example["task"], example["query"], doc) for doc in example["documents"]]

        inputs = tokenizer(
            pairs, padding=False, truncation="longest_first",
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for idx, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

        with torch.no_grad():
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vec = batch_scores[:, token_true_id]
            false_vec = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vec, true_vec], dim=1)
            batch_scores = F.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

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
        print(f"  {marker} #{rank}  score={score:.4f}  {doc[:80]}...")


# ─── Interactive Mode ──────────────────────────────────────────────────────────


def run_interactive(mode: str):
    if mode in ("embedding", "embedding-st"):
        from sentence_transformers import SentenceTransformer
        mp = model_path(EMBEDDING_MODEL_ID)
        print(f"\nLoading embedding model from: {mp} ...")
        model = SentenceTransformer(mp, cache_folder=str(MODEL_DIR))
    elif mode == "reranker-st":
        from sentence_transformers import CrossEncoder
        mp = model_path(RERANKER_MODEL_ID)
        print(f"\nLoading reranker model from: {mp} ...")
        model = CrossEncoder(mp, cache_folder=str(MODEL_DIR))
    else:
        # transformers reranker
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mp = model_path(RERANKER_MODEL_ID)
        print(f"\nLoading reranker model from: {mp} ...")
        tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="left", cache_dir=str(MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(mp, cache_dir=str(MODEL_DIR)).eval()

    print("\nInteractive Qwen3 Demo (type 'quit' to exit)")
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

        if mode in ("embedding", "embedding-st"):
            query_emb = model.encode([query], prompt_name="query")
            doc_emb = model.encode(documents)
            scores = model.similarity(query_emb, doc_emb)[0].tolist()
        elif mode == "reranker-st":
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(pairs, activation_fn=torch.nn.Sigmoid()).tolist()
        else:
            pairs = [format_rerank_instruction(None, query, doc) for doc in documents]
            token_false_id = model.tokenizer.convert_tokens_to_ids("no") if hasattr(model, 'tokenizer') else tokenizer.convert_tokens_to_ids("no")
            token_true_id = model.tokenizer.convert_tokens_to_ids("yes") if hasattr(model, 'tokenizer') else tokenizer.convert_tokens_to_ids("yes")
            max_length = 8192
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

            inputs = tokenizer(
                pairs, padding=False, truncation="longest_first",
                return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
            )
            for idx, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][idx] = prefix_tokens + ele + suffix_tokens
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

            with torch.no_grad():
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vec = batch_scores[:, token_true_id]
                false_vec = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vec, true_vec], dim=1)
                batch_scores = F.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()

        ranked = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
        print(f"\nResults for: {query}")
        for rank, (score, doc) in enumerate(ranked, 1):
            marker = ">>>" if rank == 1 else "   "
            print(f"  {marker} #{rank}  score={score:.4f}  {doc}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Embedding & Reranker Demo")
    parser.add_argument(
        "--mode",
        choices=["embedding", "embedding-st", "reranker", "reranker-st"],
        default="embedding-st",
        help=(
            "Demo mode: "
            "'embedding-st' = Sentence Transformers embedding (default), "
            "'embedding' = Transformers embedding, "
            "'reranker-st' = Sentence Transformers reranker, "
            "'reranker' = Transformers reranker"
        ),
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(mode=args.mode)
    elif args.mode == "embedding-st":
        run_embedding_demo_sentence_transformers()
    elif args.mode == "embedding":
        run_embedding_demo_transformers()
    elif args.mode == "reranker-st":
        run_reranker_demo_sentence_transformers()
    elif args.mode == "reranker":
        run_reranker_demo_transformers()


if __name__ == "__main__":
    main()
