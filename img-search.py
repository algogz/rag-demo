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
import tempfile
import time
from pathlib import Path

import numpy as np
from pillow_heif import register_heif_opener

register_heif_opener()

MODEL_DIR = Path(__file__).parent / "models"
DB_PATH = Path(__file__).parent / "image_vectors.db"
EMBED_DIM = 2048

EMBEDDING_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".heic",
}


def model_path(model_id: str) -> str:
    local = MODEL_DIR / model_id.replace("/", "--")
    if local.exists():
        return str(local)
    os.makedirs(MODEL_DIR, exist_ok=True)
    return model_id


def resolve_device(device: str | None = None) -> str:
    """Resolve the best available device. Priority: user-specified > cuda > mps > cpu."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_dtype(device: str):
    """Use float16 on GPU/MPS for faster inference, float32 on CPU."""
    import torch

    return torch.float16 if device in ("cuda", "mps") else torch.float32


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


def cmd_embed(path: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    dev = resolve_device(device)
    print(f"Device: {dev}")

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
    model = SentenceTransformer(
        mp,
        cache_folder=str(MODEL_DIR),
        device=dev,
        model_kwargs={"torch_dtype": _model_dtype(dev)},
    )

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    _EMBED_MAX_LONG_EDGE = 768

    def _resize_for_embed(img: Image.Image) -> Image.Image:
        w, h = img.size
        long = max(w, h)
        if long <= _EMBED_MAX_LONG_EDGE:
            return img.convert("RGB")
        scale = _EMBED_MAX_LONG_EDGE / long
        return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS).convert(
            "RGB"
        )

    batch_size = 32
    for i in range(0, len(new_images), batch_size):
        batch = new_images[i : i + batch_size]
        paths = [str(img) for img in batch]
        n_batch = i // batch_size + 1
        n_total = (len(new_images) - 1) // batch_size + 1
        print(f"  Batch {n_batch}/{n_total} ({len(batch)} images) ...")

        pil_images = [_resize_for_embed(Image.open(p)) for p in paths]
        try:
            embeddings = model.encode(
                pil_images, normalize_embeddings=True, show_progress_bar=False
            )
            ok_pairs = list(zip(paths, embeddings))
        except Exception as exc:
            print(f"    Batch encode failed ({exc}), retrying one-by-one ...")
            ok_pairs = []
            for p, img in zip(paths, pil_images):
                try:
                    emb = model.encode(
                        [img], normalize_embeddings=True, show_progress_bar=False
                    )[0]
                    ok_pairs.append((p, emb))
                except Exception as e:
                    print(f"    SKIP {p}: {e}")

        for path_str, emb in ok_pairs:
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


# ─── Search Core ────────────────────────────────────────────────────────────────


def _knn_search(query_vec: np.ndarray, top_k: int) -> list[tuple]:
    """Vector KNN search. Returns list of (rowid, distance, path, filename)."""
    db = init_db()
    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if total == 0:
        db.close()
        return []

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
    return rows


_RERANK_MAX_PIXELS = 147456  # ~384x384, enough for relevance scoring


def _resize_for_rerank(img) -> "Image.Image":
    """Resize image so total pixels <= _RERANK_MAX_PIXELS, keeping aspect ratio."""
    w, h = img.size
    if w * h <= _RERANK_MAX_PIXELS:
        return img
    import math

    scale = math.sqrt(_RERANK_MAX_PIXELS / (w * h))
    return img.resize((int(w * scale), int(h * scale)))


def _rerank(
    query, rows: list[tuple], reranker, rerank_k: int = 10
) -> list[tuple[float, str, str]]:
    """Rerank top-N results. Returns list of (score, path, filename)."""
    from PIL import Image

    candidates = rows[:rerank_k]
    pairs = [(query, _resize_for_rerank(Image.open(row[2]))) for row in candidates]
    raw_scores = reranker.predict(pairs, batch_size=10)

    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    flat = [float(s) for s in raw_scores]
    min_s, max_s = min(flat), max(flat)
    if max_s > min_s:
        scores = [(s - min_s) / (max_s - min_s) for s in flat]
    else:
        scores = [1.0 for _ in flat]

    ranked = sorted(
        zip(scores, candidates, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )
    return [(s, row[2], row[3]) for s, row in ranked]


def _distance_scores(rows: list[tuple]) -> list[tuple[float, str, str]]:
    """Convert L2 distances to cosine similarity. Returns (score, path, filename)."""
    results = [(max(0.0, 1.0 - row[1] ** 2 / 2), row[2], row[3]) for row in rows]
    return sorted(results, key=lambda x: x[0], reverse=True)


# ─── Search Command ─────────────────────────────────────────────────────────────


def cmd_search(
    query: str, top_k: int = 20, no_rerank: bool = False, device: str | None = None
):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    dev = resolve_device(device)
    print(f"Device: {dev}")

    db = init_db()
    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    db.close()
    if total == 0:
        print("No images in database. Run 'embed <path>' first.")
        return

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"Loading embedding model from: {mp} ...")
    embed_model = SentenceTransformer(
        mp,
        cache_folder=str(MODEL_DIR),
        device=dev,
        model_kwargs={"torch_dtype": _model_dtype(dev)},
    )

    print(f"\nQuery: {query}")
    print(f"Database: {total} images")
    print("─" * 70)

    query_emb = embed_model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    rows = _knn_search(query_emb[0], top_k)

    if not rows:
        print("No results found")
        return

    if no_rerank:
        ranked = _distance_scores(rows)
    else:
        mp = model_path(RERANKER_MODEL_ID)
        print(f"Loading reranker model from: {mp} ...")
        reranker = CrossEncoder(
            mp,
            cache_folder=str(MODEL_DIR),
            device=dev,
            model_kwargs={"torch_dtype": _model_dtype(dev)},
        )
        ranked = _rerank(query, rows, reranker)

    for rank, (score, path, filename) in enumerate(ranked, 1):
        marker = ">>>" if rank == 1 else "   "
        print(f"  {marker} #{rank:<3} score={score:.4f}  {path}")

    print("─" * 70)


# ─── Serve Command ─────────────────────────────────────────────────────────────


def cmd_serve(
    port: int = 7860,
    no_rerank: bool = False,
    device: str | None = None,
    rerank_k: int = 10,
):
    import gradio as gr
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    dev = resolve_device(device)
    print(f"Device: {dev}")

    db = init_db()
    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if total == 0:
        db.close()
        print("No images in database. Run 'embed <path>' first.")
        return

    # Collect unique parent directories for Gradio allowed_paths
    dirs = {
        str(Path(row[0]).parent)
        for row in db.execute("SELECT path FROM images").fetchall()
    }
    db.close()

    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"Loading embedding model from: {mp} ...")
    embed_model = SentenceTransformer(
        mp,
        cache_folder=str(MODEL_DIR),
        device=dev,
        model_kwargs={"torch_dtype": _model_dtype(dev)},
    )

    reranker = None
    if not no_rerank:
        from sentence_transformers import CrossEncoder

        mp = model_path(RERANKER_MODEL_ID)
        print(f"Loading reranker model from: {mp} ...")
        reranker = CrossEncoder(
            mp,
            cache_folder=str(MODEL_DIR),
            device=dev,
            model_kwargs={"torch_dtype": _model_dtype(dev)},
        )

    heic_cache = Path(tempfile.mkdtemp(prefix="img_search_heic_"))

    def _ensure_browser_compatible(path: str) -> str:
        """Convert HEIC to JPEG for browser display; return original path otherwise."""
        if Path(path).suffix.lower() != ".heic":
            return path
        jpeg_path = heic_cache / (Path(path).stem + ".jpg")
        if not jpeg_path.exists():
            Image.open(path).convert("RGB").save(jpeg_path, "JPEG", quality=90)
        return str(jpeg_path)

    def do_search(text_query: str, image_file, top_k: int):
        if not text_query and image_file is None:
            return [], "Please enter a text query or upload an image."

        t0 = time.perf_counter()

        # Build query vector
        if image_file is not None:
            query_input = Image.open(image_file)
        else:
            query_input = text_query

        query_emb = embed_model.encode(
            [query_input], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = query_emb[0]
        print(f"  [timing] embedding: {time.perf_counter() - t0:.3f}s")

        t1 = time.perf_counter()
        rows = _knn_search(query_vec, top_k)
        print(f"  [timing] vector search: {time.perf_counter() - t1:.3f}s")
        if not rows:
            return [], "No results found."

        if reranker is not None:
            t2 = time.perf_counter()
            query_str = text_query if text_query else "[image query]"
            ranked = _rerank(query_str, rows, reranker, rerank_k=rerank_k)
            print(
                f"  [timing] reranking ({len(rows[:rerank_k])}/{len(rows)} candidates): {time.perf_counter() - t2:.3f}s"
            )
        else:
            ranked = _distance_scores(rows)

        print(f"  [timing] total: {time.perf_counter() - t0:.3f}s")

        gallery = [
            (_ensure_browser_compatible(item[1]), f"#{i + 1}  score: {item[0]:.4f}")
            for i, item in enumerate(ranked)
        ]
        summary = f"Found {len(ranked)} results"
        return gallery, summary

    with gr.Blocks(title="Image Search") as app:
        gr.Markdown("# Image Semantic Search")
        gr.Markdown("Search by text description or upload a reference image.")

        with gr.Row():
            text_input = gr.Textbox(
                label="Text Query", placeholder="Describe what you're looking for..."
            )
            image_input = gr.Image(label="Image Query (optional)", type="filepath")

        top_k = gr.Slider(5, 50, value=20, step=5, label="Top K")

        search_btn = gr.Button("Search", variant="primary")
        status = gr.Markdown()

        gallery = gr.Gallery(
            label="Results",
            columns=4,
            height="auto",
            object_fit="contain",
            show_label=True,
        )

        search_btn.click(
            fn=do_search,
            inputs=[text_input, image_input, top_k],
            outputs=[gallery, status],
        )

    print(f"\nLaunching at http://localhost:{port}")
    app.launch(server_port=port, allowed_paths=list(dirs) + [str(heic_cache)])


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Image Embedding & Search with Qwen3-VL and sqlite-vec"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emb = sub.add_parser("embed", help="Embed images from a path")
    emb.add_argument("path", help="Path to image file or directory")
    emb.add_argument(
        "--device", help="Compute device: cuda, mps, cpu (default: auto-detect)"
    )

    srch = sub.add_parser("search", help="Semantic search for images")
    srch.add_argument("desc", help="Text description to search for")
    srch.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Candidates from vector search (default: 20)",
    )
    srch.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking, use vector similarity only",
    )
    srch.add_argument(
        "--device", help="Compute device: cuda, mps, cpu (default: auto-detect)"
    )

    srv = sub.add_parser("serve", help="Launch web UI for image search")
    srv.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)"
    )
    srv.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking, use vector similarity only",
    )
    srv.add_argument(
        "--device", help="Compute device: cuda, mps, cpu (default: auto-detect)"
    )
    srv.add_argument(
        "--rerank-k",
        type=int,
        default=10,
        help="Number of candidates to rerank (default: 10)",
    )

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args.path, device=getattr(args, "device", None))
    elif args.command == "search":
        cmd_search(
            args.desc,
            top_k=args.top_k,
            no_rerank=args.no_rerank,
            device=getattr(args, "device", None),
        )
    elif args.command == "serve":
        cmd_serve(
            port=args.port,
            no_rerank=args.no_rerank,
            device=getattr(args, "device", None),
            rerank_k=args.rerank_k,
        )


if __name__ == "__main__":
    main()
