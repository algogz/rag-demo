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
import warnings
from pathlib import Path

import numpy as np
from pillow_heif import register_heif_opener

warnings.filterwarnings("ignore", message="Palette images with Transparency")

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
    # Check huggingface_hub cache layout: models/<org>--<repo>/snapshots/<hash>/
    cache_dir = MODEL_DIR / f"models--{model_id.replace('/', '--')}"
    if cache_dir.exists():
        import glob

        snaps = glob.glob(str(cache_dir / "snapshots" / "*"))
        if snaps:
            return snaps[0]
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
        img = _safe_to_rgb(img)
        w, h = img.size
        long = max(w, h)
        if long <= _EMBED_MAX_LONG_EDGE:
            return img
        scale = _EMBED_MAX_LONG_EDGE / long
        return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    batch_size = 32
    t0 = time.perf_counter()
    failed: list[tuple[str, str]] = []

    def _fmt_dur(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    for i in range(0, len(new_images), batch_size):
        batch = new_images[i : i + batch_size]
        paths = [str(img) for img in batch]
        n_batch = i // batch_size + 1
        n_total = (len(new_images) - 1) // batch_size + 1
        t_batch = time.perf_counter()

        # Load images, skip unreadable files
        loaded: list[tuple[str, Image.Image]] = []
        for p in paths:
            try:
                loaded.append((p, _resize_for_embed(Image.open(p))))
            except Exception as e:
                failed.append((p, str(e)))
                print(f"    SKIP {p}: {e}")

        if not loaded:
            continue

        load_paths, pil_images = zip(*loaded)
        try:
            embeddings = model.encode(
                list(pil_images), normalize_embeddings=True, show_progress_bar=False
            )
            ok_pairs = list(zip(load_paths, embeddings))
        except Exception as exc:
            print(f"    Batch encode failed ({exc}), retrying one-by-one ...")
            ok_pairs = []
            for p, img in zip(load_paths, pil_images):
                try:
                    emb = model.encode(
                        [img], normalize_embeddings=True, show_progress_bar=False
                    )[0]
                    ok_pairs.append((p, emb))
                except Exception as e:
                    failed.append((p, str(e)))
                    print(f"    SKIP {p}: {e}")

        batch_time = time.perf_counter() - t_batch
        elapsed = time.perf_counter() - t0
        avg = elapsed / n_batch
        eta = avg * (n_total - n_batch)

        print(
            f"  [{time.strftime('%H:%M:%S')}] Batch {n_batch}/{n_total} ({len(batch)} images) "
            f"| {batch_time:.1f}s | {_fmt_dur(elapsed)} elapsed, ~{_fmt_dur(eta)} remaining"
        )

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
    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for p, reason in failed:
            print(f"  {p}: {reason}")
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


def _safe_to_rgb(img: "Image.Image") -> "Image.Image":
    """Convert any PIL image to RGB, handling palette/GIF with transparency safely."""
    if img.mode in ("RGBA", "LA", "PA"):
        return img.convert("RGB")
    if img.mode == "P":
        return img.convert("RGBA").convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _resize_for_rerank(img: "Image.Image", max_pixels: int) -> "Image.Image":
    """Resize image so total pixels <= max_pixels, keeping aspect ratio."""
    import math

    from PIL import Image

    img = _safe_to_rgb(img)
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / (w * h))
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ─── Gallery Modal Enhancement (CSS + JS) ──────────────────────────────────────

_GALLERY_CSS = """
.ge-1x1-btn {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
    cursor: pointer;
}
.ge-open-btn {
    font-size: 15px !important;
    text-decoration: none !important;
    line-height: 1;
    cursor: pointer;
}
/* ===== Compact top panel ===== */
.top-bar { gap: 8px !important; align-items: stretch !important; }
.top-bar .image-upload { height: 100% !important; }
.top-bar .image-upload > div { height: 100% !important; min-height: unset !important; }
/* ===== Compact rerank row ===== */
.rerank-row { gap: 8px !important; }
/* Force radio buttons horizontal */
.rerank-row fieldset > div { flex-direction: row !important; flex-wrap: nowrap !important; gap: 8px !important; }
/* Compact number inputs */
.compact-num input { padding: 4px 6px !important; font-size: 0.85em !important; height: 32px !important; }
"""

_GALLERY_JS = """
(function() {
  function enhance(img) {
    if (!img.isConnected) return;
    var preview = img.closest('.preview');
    if (!preview || preview.querySelector('.ge-marker')) return;
    var mediaBtn = preview.querySelector('.media-button');
    var toolbar = preview.querySelector('.icon-button-wrapper');
    if (!mediaBtn || !toolbar) return;

    var marker = document.createElement('span');
    marker.className = 'ge-marker';
    marker.style.display = 'none';
    preview.appendChild(marker);

    var btn1x1 = document.createElement('button');
    btn1x1.className = 'icon-button ge-1x1-btn';
    btn1x1.textContent = '1:1';
    btn1x1.title = 'Show original size';
    btn1x1.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      if (img.dataset.geZoom === '1x1') {
        img.style.flex = '';
        img.style.objectFit = '';
        img.style.width = '';
        img.style.height = '';
        mediaBtn.style.overflow = '';
        img.dataset.geZoom = '';
        btn1x1.textContent = '1:1';
        btn1x1.title = 'Show original size';
      } else {
        mediaBtn.style.overflow = 'auto';
        img.style.flex = 'none';
        img.style.objectFit = 'none';
        img.style.width = img.naturalWidth + 'px';
        img.style.height = img.naturalHeight + 'px';
        img.dataset.geZoom = '1x1';
        btn1x1.textContent = 'Fit';
        btn1x1.title = 'Fit to view';
      }
    });

    var openLink = document.createElement('a');
    openLink.className = 'icon-button ge-open-btn';
    openLink.textContent = '\\u2197';
    openLink.title = 'Open in new window';
    openLink.target = '_blank';
    openLink.rel = 'noopener noreferrer';
    openLink.href = img.src;
    openLink.addEventListener('click', function(e) { e.stopPropagation(); });

    toolbar.insertBefore(openLink, toolbar.firstChild);
    toolbar.insertBefore(btn1x1, toolbar.firstChild);

    var imgObs = new MutationObserver(function() {
      openLink.href = img.src;
      if (img.dataset.geZoom === '1x1') {
        img.style.flex = '';
        img.style.objectFit = '';
        img.style.width = '';
        img.style.height = '';
        mediaBtn.style.overflow = '';
        img.dataset.geZoom = '';
        btn1x1.textContent = '1:1';
      }
    });
    imgObs.observe(img, { attributes: true, attributeFilter: ['src'] });
  }

  var observer = new MutationObserver(function() {
    var img = document.querySelector('[data-testid="detailed-image"]');
    if (img && !img.dataset.geEnhanced) {
      img.dataset.geEnhanced = 'true';
      setTimeout(function() { enhance(img); }, 50);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
"""

_RERANK_S1_PIXELS = 112896  # 336x336 — coarse filter
_RERANK_S1_K = 30
_RERANK_S2_PIXELS = 450816  # 672x672 — fine scoring
_RERANK_S2_K = 10


def _score_pairs(
    query, candidates, reranker, max_pixels: int, batch_size: int
) -> list[float]:
    """Score (query, image) pairs with the reranker. Returns normalized scores."""
    from PIL import Image

    pairs = [
        (query, _resize_for_rerank(Image.open(row[2]), max_pixels))
        for row in candidates
    ]
    raw = reranker.predict(pairs, batch_size=batch_size)
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    flat = [float(s) for s in raw]
    min_s, max_s = min(flat), max(flat)
    if max_s > min_s:
        return [(s - min_s) / (max_s - min_s) for s in flat]
    return [1.0 for _ in flat]


def _rerank(
    query,
    rows: list[tuple],
    reranker=None,
    *,
    mode: str = "2-stage",
    s1_k: int = 30,
    s2_k: int = 10,
    rerank_k: int = 10,
) -> list[tuple[float, str, str]]:
    """Unified rerank with mode: 'none', '1-stage', or '2-stage'."""
    import time

    mode = mode.lower()
    if mode == "none":
        return _distance_scores(rows[:rerank_k])

    if mode == "1-stage":
        candidates = rows[:s1_k]
        t0 = time.perf_counter()
        scores = _score_pairs(
            query, candidates, reranker, _RERANK_S2_PIXELS, batch_size=10
        )
        print(
            f"    [timing] 1-stage ({len(candidates)} imgs @ 672px): {time.perf_counter() - t0:.3f}s"
        )
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
        return [(s, row[2], row[3]) for s, row in ranked[:rerank_k]]

    # 2-stage
    s1_candidates = rows[:s1_k]
    t1 = time.perf_counter()
    s1_scores = _score_pairs(
        query, s1_candidates, reranker, _RERANK_S1_PIXELS, batch_size=20
    )
    print(
        f"    [timing] stage1 ({len(s1_candidates)} imgs @ 336px): {time.perf_counter() - t1:.3f}s"
    )
    s1_ranked = sorted(
        zip(s1_scores, s1_candidates, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )

    s2_candidates = [row for _, row in s1_ranked[:s2_k]]
    t2 = time.perf_counter()
    s2_scores = _score_pairs(
        query, s2_candidates, reranker, _RERANK_S2_PIXELS, batch_size=10
    )
    print(
        f"    [timing] stage2 ({len(s2_candidates)} imgs @ 672px): {time.perf_counter() - t2:.3f}s"
    )
    s2_ranked = sorted(
        zip(s2_scores, s2_candidates, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )
    return [(s, row[2], row[3]) for s, row in s2_ranked[:rerank_k]]


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
    rows = _knn_search(query_emb[0], max(top_k, _RERANK_S1_K))

    if not rows:
        print("No results found")
        return

    if no_rerank:
        ranked = _rerank(query, rows, mode="none", rerank_k=top_k)
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

    def do_search(
        text_query: str,
        image_file,
        rerank_mode: str,
        s1_k: int,
        s2_k: int,
        rerank_k: int,
    ):
        if not text_query and image_file is None:
            return [], "Please enter a text query or upload an image."

        t0 = time.perf_counter()

        # Build query vector
        if image_file is not None:
            query_input = _safe_to_rgb(Image.open(image_file))
        else:
            query_input = text_query

        query_emb = embed_model.encode(
            [query_input], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = query_emb[0]
        print(f"  [timing] embedding: {time.perf_counter() - t0:.3f}s")

        t1 = time.perf_counter()
        knn_k = max(s1_k, rerank_k)
        rows = _knn_search(query_vec, knn_k)
        print(f"  [timing] vector search (k={knn_k}): {time.perf_counter() - t1:.3f}s")
        if not rows:
            return [], "No results found."

        query_str = text_query if text_query else "[image query]"

        # Enforce: candidates >= stage2 >= results
        rerank_k = max(1, min(rerank_k, s2_k, s1_k))
        s2_k = max(rerank_k, min(s2_k, s1_k))

        # Disable reranker when not loaded
        effective_mode = rerank_mode if reranker is not None else "none"

        ranked = _rerank(
            query_str,
            rows,
            reranker,
            mode=effective_mode,
            s1_k=s1_k,
            s2_k=s2_k,
            rerank_k=rerank_k,
        )

        elapsed = time.perf_counter() - t0
        print(f"  [timing] total: {elapsed:.3f}s")

        gallery = [
            (_ensure_browser_compatible(item[1]), f"#{i + 1}  score: {item[0]:.4f}")
            for i, item in enumerate(ranked)
        ]
        summary = f"Found {len(ranked)} results in {elapsed:.1f}s"
        return gallery, summary

    with gr.Blocks(title="Image Search") as app:
        # gr.Markdown("### Image Semantic Search")

        # --- Top input panel ---
        with gr.Row(elem_classes=["top-bar"]):
            # Col 1: query + rerank controls (5/7)
            with gr.Column(scale=5, min_width=300):
                text_input = gr.Textbox(
                    label="Query",
                    placeholder="Describe what you're looking for...",
                    lines=1,
                    max_lines=2,
                    show_label=False,
                )
                with gr.Row(elem_classes=["rerank-row"]):
                    rerank_mode = gr.Radio(
                        choices=["2-stage", "1-stage", "None"],
                        value="2-stage",
                        label="Rerank",
                        show_label=True,
                        scale=3,
                    )
                    s1_k_input = gr.Number(
                        value=_RERANK_S1_K,
                        label="Candidates",
                        minimum=5,
                        maximum=100,
                        step=5,
                        scale=1,
                        min_width=90,
                        elem_classes=["compact-num"],
                    )
                    s2_k_input = gr.Number(
                        value=_RERANK_S2_K,
                        label="Stage 2",
                        minimum=5,
                        maximum=50,
                        step=5,
                        scale=1,
                        min_width=90,
                        elem_classes=["compact-num"],
                    )
                    rerank_k_input = gr.Number(
                        value=10,
                        label="Results",
                        minimum=1,
                        maximum=50,
                        step=1,
                        scale=1,
                        min_width=90,
                        elem_classes=["compact-num"],
                    )
                search_btn = gr.Button(
                    "\U0001f50d Search", variant="primary", scale=0, min_width=90
                )
            # Col 2: image input (2/7)
            image_input = gr.Image(
                label="Image",
                type="filepath",
                scale=2,
                show_label=True,
                sources=["upload"],
                elem_classes=["image-upload"],
            )

        status = gr.Markdown("")

        # Show/hide Stage 2 control based on mode
        rerank_mode.change(
            fn=lambda m: gr.update(visible=(m == "2-stage")),
            inputs=[rerank_mode],
            outputs=[s2_k_input],
        )

        # Enforce s1_k >= s2_k >= rerank_k on input change
        def _clamp_s1(s1, s2, rerank_k):
            s1 = max(s1, 1)
            s2 = min(s2, s1)
            rerank_k = min(rerank_k, s2)
            return gr.update(value=s1), gr.update(value=s2), gr.update(value=rerank_k)

        def _clamp_s2(s1, s2, rerank_k):
            s2 = max(s2, 1)
            s1 = max(s1, s2)
            rerank_k = min(rerank_k, s2)
            return gr.update(value=s1), gr.update(value=s2), gr.update(value=rerank_k)

        def _clamp_rerank_k(s1, s2, rerank_k):
            rerank_k = max(rerank_k, 1)
            s2 = max(s2, rerank_k)
            s1 = max(s1, s2)
            return gr.update(value=s1), gr.update(value=s2), gr.update(value=rerank_k)

        all_three = [s1_k_input, s2_k_input, rerank_k_input]
        s1_k_input.change(fn=_clamp_s1, inputs=all_three, outputs=all_three)
        s2_k_input.change(fn=_clamp_s2, inputs=all_three, outputs=all_three)
        rerank_k_input.change(fn=_clamp_rerank_k, inputs=all_three, outputs=all_three)

        gallery = gr.Gallery(
            label="Results (click image to zoom)",
            columns=5,
            height="85vh",
            object_fit="contain",
            show_label=True,
        )

        search_btn.click(
            fn=do_search,
            inputs=[
                text_input,
                image_input,
                rerank_mode,
                s1_k_input,
                s2_k_input,
                rerank_k_input,
            ],
            outputs=[gallery, status],
        )

    print(f"\nLaunching at http://localhost:{port}")
    app.launch(
        server_port=port,
        allowed_paths=list(dirs) + [str(heic_cache)],
        css=_GALLERY_CSS,
        js=_GALLERY_JS,
    )


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
        )


if __name__ == "__main__":
    main()
