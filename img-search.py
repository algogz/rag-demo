"""
Image Embedding & Semantic Search with Qwen3-VL and sqlite-vec.
Uses Qwen3-VL-Embedding for image embeddings, Qwen3-VL-Reranker for reranking,
and sqlite-vec for vector storage.

Usage:
    uv run python img-search.py embed <path>   # Embed images in path
    uv run python img-search.py search <desc>  # Semantic search by description
"""

import argparse
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
from pillow_heif import register_heif_opener

register_heif_opener()

MODEL_DIR = Path(__file__).parent / "models"
DB_PATH = Path(__file__).parent / "image_vectors.db"
EMBED_DIM = 2048

EMBEDDING_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}


def model_path(model_id: str) -> str:
    local = MODEL_DIR / model_id.replace("/", "--")
    if local.exists():
        return str(local)
    os.makedirs(MODEL_DIR, exist_ok=True)
    return model_id


def serialize_f32(vector: np.ndarray) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector.tolist())


def init_db() -> sqlite3.Connection:
    import sqlite_vec

    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("""
        CREATE TABLE IF NOT EXISTS images (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL
        )
    """)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
            embedding float[{EMBED_DIM}]
        )
    """)
    db.commit()
    return db


def scan_images(path: str) -> list[Path]:
    p = Path(path).resolve()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    if not p.is_dir():
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


# ─── Embed Command ──────────────────────────────────────────────────────────────


def cmd_embed(path: str):
    from sentence_transformers import SentenceTransformer

    images = scan_images(path)
    if not images:
        print(f"No images found in {path}")
        return

    print(f"Found {len(images)} images")
    db = init_db()

    existing = {row[0] for row in db.execute("SELECT path FROM images").fetchall()}
    new_images = [img for img in images if str(img) not in existing]

    if not new_images:
        print("All images already embedded")
        db.close()
        return

    print(f"New images to embed: {len(new_images)}")

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"Loading embedding model from: {mp} ...")
    model = SentenceTransformer(mp, cache_folder=str(MODEL_DIR))

    from PIL import Image

    batch_size = 4
    for i in range(0, len(new_images), batch_size):
        batch = new_images[i : i + batch_size]
        paths = [str(img) for img in batch]
        n_batch = i // batch_size + 1
        n_total = (len(new_images) - 1) // batch_size + 1
        print(f"  Batch {n_batch}/{n_total} ({len(batch)} images) ...")

        pil_images = [Image.open(p) for p in paths]
        embeddings = model.encode(pil_images, normalize_embeddings=True, show_progress_bar=False)

        for path_str, emb in zip(paths, embeddings):
            cur = db.execute(
                "INSERT INTO images (path, filename) VALUES (?, ?)",
                [path_str, Path(path_str).name],
            )
            db.execute(
                "INSERT INTO vec_embeddings (rowid, embedding) VALUES (?, ?)",
                [cur.lastrowid, serialize_f32(emb)],
            )
        db.commit()

    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"Done. Total images in DB: {total}")
    db.close()


# ─── Search Command ─────────────────────────────────────────────────────────────


def cmd_search(query: str, top_k: int = 20, no_rerank: bool = False):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    db = init_db()
    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if total == 0:
        print("No images in database. Run 'embed <path>' first.")
        db.close()
        return

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"Loading embedding model from: {mp} ...")
    embed_model = SentenceTransformer(mp, cache_folder=str(MODEL_DIR))

    print(f"\nQuery: {query}")
    print(f"Database: {total} images")
    print("─" * 70)

    query_emb = embed_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vec = query_emb[0]

    # KNN search via sqlite-vec (L2 distance, lower = more similar)
    rows = db.execute(
        """
        SELECT vec.rowid, vec.distance, img.path, img.filename
        FROM vec_embeddings vec
        JOIN images img ON vec.rowid = img.rowid
        WHERE vec.embedding MATCH ? AND k = ?
        ORDER BY vec.distance
        """,
        [serialize_f32(query_vec), top_k],
    ).fetchall()

    db.close()

    if not rows:
        print("No results found")
        return

    if no_rerank:
        # Convert L2 distance to cosine similarity (embeddings are L2-normalized)
        # L2^2 = 2(1 - cos_sim) => cos_sim = 1 - L2^2/2
        scores = [max(0.0, 1.0 - row[1] ** 2 / 2) for row in rows]
        ranked = sorted(
            zip(scores, rows, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
    else:
        from PIL import Image

        mp = model_path(RERANKER_MODEL_ID)
        print(f"Loading reranker model from: {mp} ...")
        reranker = CrossEncoder(mp, cache_folder=str(MODEL_DIR))

        pairs = [(query, Image.open(row[2])) for row in rows]
        raw_scores = reranker.predict(pairs)

        # Normalize to [0, 1] via min-max
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        flat = [float(s) for s in raw_scores]
        min_s, max_s = min(flat), max(flat)
        if max_s > min_s:
            scores = [(s - min_s) / (max_s - min_s) for s in flat]
        else:
            scores = [1.0 for _ in flat]

        ranked = sorted(
            zip(scores, rows, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

    for rank, (score, (_, _, path, filename)) in enumerate(ranked, 1):
        marker = ">>>" if rank == 1 else "   "
        print(f"  {marker} #{rank:<3} score={score:.4f}  {path}")

    print("─" * 70)


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Image Embedding & Search with Qwen3-VL and sqlite-vec"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emb = sub.add_parser("embed", help="Embed images from a path")
    emb.add_argument("path", help="Path to image file or directory")

    srch = sub.add_parser("search", help="Semantic search for images")
    srch.add_argument("desc", help="Text description to search for")
    srch.add_argument("--top-k", type=int, default=20, help="Candidates from vector search (default: 20)")
    srch.add_argument("--no-rerank", action="store_true", help="Skip reranking, use vector similarity only")

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args.path)
    elif args.command == "search":
        cmd_search(args.desc, top_k=args.top_k, no_rerank=args.no_rerank)


if __name__ == "__main__":
    main()
