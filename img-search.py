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
.search-row { gap: 12px !important; align-items: center !important; }
.search-status { font-size: 0.85em !important; opacity: 0.7; }
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
    query: str,
    top_k: int = 20,
    no_rerank: bool = False,
    device: str | None = None,
    person: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
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

    # Build metadata filters
    filter_rowids = _parse_metadata_filters(
        person_filter=person or "",
        date_from=date_from or "",
        date_to=date_to or "",
    )
    if filter_rowids is not None:
        print(f"Metadata filter: {len(filter_rowids)} candidates")

    print(f"\nQuery: {query}")
    print(f"Database: {total} images")
    print("─" * 70)

    query_emb = embed_model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    rows = _knn_search_filtered(
        query_emb[0], max(top_k, _RERANK_S1_K), filter_rowids
    )

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


# ─── Apple Photos Metadata Indexer ─────────────────────────────────────────────

APPLE_PHOTOS_DB = "database/Photos.sqlite"
CORE_DATA_EPOCH = 978307200  # seconds between 1970-01-01 and 2001-01-01


def _open_apple_photos_db(library_path: str) -> sqlite3.Connection:
    """Open Apple Photos SQLite DB. Handles SMB mounts via immutable=1."""
    db_path = Path(library_path) / APPLE_PHOTOS_DB
    if not db_path.exists():
        print(f"Error: {db_path} not found")
        sys.exit(1)
    return sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)


def _create_metadata_tables(db: sqlite3.Connection):
    """Create metadata tables if they don't exist."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS photo_metadata (
            rowid INTEGER PRIMARY KEY,
            asset_pk INTEGER,
            date_created REAL,
            latitude REAL,
            longitude REAL,
            timezone_name TEXT,
            moment_title TEXT,
            camera_model TEXT
        );

        CREATE TABLE IF NOT EXISTS photo_persons (
            rowid INTEGER NOT NULL,
            person_name TEXT NOT NULL,
            PRIMARY KEY (rowid, person_name)
        );

        CREATE TABLE IF NOT EXISTS photo_scenes (
            rowid INTEGER NOT NULL,
            scene_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            PRIMARY KEY (rowid, scene_id)
        );

        CREATE INDEX IF NOT EXISTS idx_meta_date ON photo_metadata(date_created);
        CREATE INDEX IF NOT EXISTS idx_meta_latlon ON photo_metadata(latitude, longitude);
        CREATE INDEX IF NOT EXISTS idx_persons_name ON photo_persons(person_name);
        CREATE INDEX IF NOT EXISTS idx_scenes_rowid ON photo_scenes(rowid);
    """)
    db.commit()


def cmd_index_photos(library_path: str):
    """Index Apple Photos metadata into image_vectors.db."""
    apple_db = _open_apple_photos_db(library_path)
    vec_db = init_db()
    _create_metadata_tables(vec_db)

    # Build path→rowid map from our image vectors DB
    # Extract "originals/X/filename.ext" suffix for matching
    rows = vec_db.execute("SELECT rowid, path FROM images").fetchall()
    path_to_rowid: dict[str, int] = {}
    for rowid, path in rows:
        # Extract the originals/... suffix
        parts = path.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if p == "originals" and i + 2 < len(parts):
                suffix = f"originals/{parts[i+1]}/{parts[i+2]}"
                path_to_rowid[suffix] = rowid
                break

    print(f"Loaded {len(path_to_rowid)} image paths from vector DB")

    # Build asset_pk→rowid mapping via ZDIRECTORY/ZFILENAME
    print("Matching Apple Photos assets to vector DB...")
    asset_rows = apple_db.execute("""
        SELECT a.Z_PK, a.ZDIRECTORY, a.ZFILENAME,
               a.ZDATECREATED, a.ZLATITUDE, a.ZLONGITUDE,
               a.ZMOMENT, a.ZEXTENDEDATTRIBUTES, a.ZADDITIONALATTRIBUTES
        FROM ZASSET a
    """).fetchall()

    matched = 0
    asset_to_rowid: dict[int, int] = {}

    for pk, directory, filename, date_created, lat, lon, moment_fk, ext_fk, add_fk in asset_rows:
        suffix = f"originals/{directory}/{filename}"
        rowid = path_to_rowid.get(suffix)
        if rowid is not None:
            asset_to_rowid[pk] = rowid
            matched += 1

    print(f"Matched {matched} assets ({len(asset_rows)} total in Photos DB)")

    if not matched:
        print("No matches found — check library path")
        apple_db.close()
        vec_db.close()
        return

    # Phase 1: Photo metadata (date, GPS, camera)
    print("Indexing photo metadata (date, GPS, camera)...")
    moment_cache: dict[int, tuple] = {}
    moment_rows = apple_db.execute("SELECT Z_PK, ZTITLE FROM ZMOMENT").fetchall()
    for pk, title in moment_rows:
        moment_cache[pk] = (title,)

    ext_cache: dict[int, str] = {}
    ext_rows = apple_db.execute("SELECT Z_PK, ZCAMERAMODEL FROM ZEXTENDEDATTRIBUTES").fetchall()
    for pk, model in ext_rows:
        ext_cache[pk] = model or ""

    add_cache: dict[int, tuple] = {}
    add_rows = apple_db.execute("SELECT Z_PK, ZTIMEZONENAME FROM ZADDITIONALASSETATTRIBUTES").fetchall()
    for pk, tz in add_rows:
        add_cache[pk] = (tz,)

    vec_db.execute("DELETE FROM photo_metadata")
    batch = []
    for pk, rowid in asset_to_rowid.items():
        # Find the original asset row data
        # We need to look up by pk — iterate asset_rows again (use dict)
        pass

    # Rebuild with dict for O(1) lookup
    asset_data: dict[int, tuple] = {}
    for pk, directory, filename, date_created, lat, lon, moment_fk, ext_fk, add_fk in asset_rows:
        if pk in asset_to_rowid:
            rowid = asset_to_rowid[pk]
            moment_title = moment_cache.get(moment_fk, (None,))[0] if moment_fk else None
            camera_model = ext_cache.get(ext_fk, "")
            tz_name = add_cache.get(add_fk, (None,))[0] if add_fk else None
            # Skip placeholder coordinates (-180, -180)
            if lat == -180.0 or lon == -180.0:
                lat, lon = None, None
            batch.append((rowid, pk, date_created, lat, lon, tz_name, moment_title, camera_model))

    vec_db.executemany(
        "INSERT OR REPLACE INTO photo_metadata (rowid, asset_pk, date_created, latitude, longitude, timezone_name, moment_title, camera_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        batch,
    )
    vec_db.commit()
    print(f"  Indexed {len(batch)} photo metadata records")

    # Phase 2: Persons
    print("Indexing persons...")
    face_rows = apple_db.execute("""
        SELECT f.ZASSETFORFACE, COALESCE(p.ZDISPLAYNAME, p.ZFULLNAME)
        FROM ZDETECTEDFACE f
        JOIN ZPERSON p ON f.ZPERSONFORFACE = p.Z_PK
        WHERE COALESCE(p.ZDISPLAYNAME, p.ZFULLNAME) IS NOT NULL
          AND COALESCE(p.ZDISPLAYNAME, p.ZFULLNAME) != ''
    """).fetchall()

    person_batch = []
    for asset_pk, name in face_rows:
        rowid = asset_to_rowid.get(asset_pk)
        if rowid is not None:
            person_batch.append((rowid, name))

    vec_db.execute("DELETE FROM photo_persons")
    vec_db.executemany(
        "INSERT OR IGNORE INTO photo_persons (rowid, person_name) VALUES (?, ?)",
        person_batch,
    )
    vec_db.commit()
    print(f"  Indexed {len(person_batch)} person-face links")

    # Person name stats
    name_stats = vec_db.execute("""
        SELECT person_name, COUNT(DISTINCT rowid) as cnt
        FROM photo_persons GROUP BY person_name ORDER BY cnt DESC LIMIT 20
    """).fetchall()
    for name, cnt in name_stats:
        print(f"    {name}: {cnt} photos")

    # Phase 3: Scene classifications
    print("Indexing scene classifications (this may take a minute)...")
    vec_db.execute("DELETE FROM photo_scenes")

    scene_rows = apple_db.execute("""
        SELECT sc.ZASSETATTRIBUTES, sc.ZSCENEIDENTIFIER, sc.ZCONFIDENCE
        FROM ZSCENECLASSIFICATION sc
        WHERE sc.ZCONFIDENCE >= 0.3
    """).fetchall()

    # Need to map add_attr_pk → asset_pk → rowid
    add_to_asset: dict[int, int] = {}
    add_asset_rows = apple_db.execute("SELECT Z_PK, ZASSET FROM ZADDITIONALASSETATTRIBUTES").fetchall()
    for add_pk, asset_pk in add_asset_rows:
        add_to_asset[add_pk] = asset_pk

    scene_batch = []
    for add_pk, scene_id, confidence in scene_rows:
        asset_pk = add_to_asset.get(add_pk)
        if asset_pk is not None:
            rowid = asset_to_rowid.get(asset_pk)
            if rowid is not None:
                scene_batch.append((rowid, scene_id, confidence))

    vec_db.executemany(
        "INSERT OR IGNORE INTO photo_scenes (rowid, scene_id, confidence) VALUES (?, ?, ?)",
        scene_batch,
    )
    vec_db.commit()
    print(f"  Indexed {len(scene_batch)} scene classifications")

    # Summary
    meta_count = vec_db.execute("SELECT COUNT(*) FROM photo_metadata").fetchone()[0]
    person_photo_count = vec_db.execute("SELECT COUNT(DISTINCT rowid) FROM photo_persons").fetchone()[0]
    scene_photo_count = vec_db.execute("SELECT COUNT(DISTINCT rowid) FROM photo_scenes").fetchone()[0]
    print(f"\nDone! Metadata indexed for {meta_count} photos")
    print(f"  {person_photo_count} photos with named persons")
    print(f"  {scene_photo_count} photos with scene classifications")

    apple_db.close()
    vec_db.close()


# ─── Metadata-Aware Search ─────────────────────────────────────────────────────

# Scene ID to label mapping (common Apple Photos scene identifiers)
# See: https://developer.apple.com/documentation/photokit/phphotossceneclassification


def _metadata_filter(
    db: sqlite3.Connection,
    persons: list[str] | None = None,
    person_match: str = "all",
    date_from: float | None = None,
    date_to: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> set[int] | None:
    """Return set of rowids matching metadata filters, or None if no filters."""
    has_filter = any(v is not None for v in [persons, date_from, date_to, lat_min])
    if not has_filter:
        return None

    conditions = ["1=1"]
    params: list = []

    if persons:
        placeholders = ",".join("?" * len(persons))
        if len(persons) == 1 or person_match == "any":
            # OR: match photos with any of the listed persons
            conditions.append(
                f"rowid IN (SELECT rowid FROM photo_persons WHERE person_name IN ({placeholders}))"
            )
        else:
            # AND: match photos with ALL listed persons
            conditions.append(
                f"rowid IN (SELECT rowid FROM photo_persons WHERE person_name IN ({placeholders}) GROUP BY rowid HAVING COUNT(DISTINCT person_name) = {len(persons)})"
            )
        params.extend(persons)

    if date_from is not None:
        conditions.append("date_created >= ?")
        params.append(date_from)

    if date_to is not None:
        conditions.append("date_created <= ?")
        params.append(date_to)

    if lat_min is not None:
        conditions.append("latitude >= ? AND latitude <= ?")
        params.extend([lat_min, lat_max])

    if lon_min is not None:
        conditions.append("longitude >= ? AND longitude <= ?")
        params.extend([lon_min, lon_max])

    where = " AND ".join(conditions)
    rows = db.execute(
        f"SELECT rowid FROM photo_metadata WHERE {where}", params
    ).fetchall()
    return {r[0] for r in rows}


def _knn_search_filtered(
    query_vec: np.ndarray,
    top_k: int,
    allowed_rowids: set[int] | None = None,
) -> list[tuple]:
    """Vector KNN search with optional metadata pre-filter.
    Returns list of (rowid, distance, path, filename).

    Strategy: for small filter sets (< 500), compute distances for all
    filtered images directly. For larger sets, use KNN-then-filter.
    """
    import sqlite_vec

    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if total == 0:
        db.close()
        return []

    if allowed_rowids is not None:
        if not allowed_rowids:
            db.close()
            return []
        filter_size = len(allowed_rowids)

        if filter_size <= 500:
            # Small filter: score ALL filtered images directly
            placeholders = ",".join("?" * filter_size)
            rows = db.execute(
                f"""
                SELECT vec.rowid, vec.distance, img.path, img.filename
                FROM vec_embeddings vec
                JOIN images img ON vec.rowid = img.rowid
                WHERE vec.embedding MATCH ? AND k = ?
                AND vec.rowid IN ({placeholders})
                ORDER BY vec.distance
                """,
                [serialize_f32(query_vec), filter_size, *allowed_rowids],
            ).fetchall()
        else:
            # Large filter: KNN-then-filter
            search_k = min(max(top_k * 10, filter_size + top_k), total, 4096)
            rows = db.execute(
                """
                SELECT vec.rowid, vec.distance, img.path, img.filename
                FROM vec_embeddings vec
                JOIN images img ON vec.rowid = img.rowid
                WHERE vec.embedding MATCH ? AND k = ?
                ORDER BY vec.distance
                """,
                [serialize_f32(query_vec), search_k],
            ).fetchall()
            rows = [r for r in rows if r[0] in allowed_rowids][:top_k]
    else:
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


# ─── LLM Query Planner ─────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a photo search query planner. Decompose the user's natural language query into structured filters for searching a photo library.

## Available Filter Fields

- **persons**: list of person names from the known persons list below
- **person_match**: "all" or "any" — whether ALL persons must appear in the same photo, or ANY one is enough. Default "all". Use "all" for queries like "A和B的合照" (photos with both A and B). Use "any" for queries like "A或B的照片" (photos with A or B).
- **date_from**: ISO date string (YYYY-MM-DD), inclusive
- **date_to**: ISO date string (YYYY-MM-DD), inclusive
- **location**: free-text place name (will be geocoded to GPS bounds)
- **semantic_query**: the visual/semantic part of the query (what the photo looks like)

## Rules

1. Extract person names from the query. If a name closely resembles a known person (e.g. partial match), include it anyway — fuzzy matching will be applied.
2. Resolve relative dates to absolute dates (current date: {today}). Examples:
   - "去年" → date_from={last_year}-01-01, date_to={last_year}-12-31
   - "上个月" → previous month range
   - "2023年夏天" → date_from=2023-06-01, date_to=2023-08-31
3. Extract location names as-is (e.g., "西藏", "广州", "Lhasa").
4. Put the visual/semantic description into semantic_query (e.g., "旅游", "birthday party", "sunset").
5. If a field is not mentioned in the query, omit it from the output.
6. Return ONLY a JSON object, no explanation.

## Known Persons

{persons}
"""

_PLANNER_MODEL = "minimax-2.7"


def _extract_text_from_response(resp) -> str:
    """Extract text from an Anthropic API response, skipping thinking blocks."""
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def _get_known_persons() -> list[str]:
    """Get list of known person names from the metadata DB."""
    if not DB_PATH.exists():
        return []
    db = sqlite3.connect(str(DB_PATH))
    try:
        rows = db.execute(
            "SELECT DISTINCT person_name FROM photo_persons ORDER BY person_name"
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        db.close()


def _resolve_persons(query_persons: list[str]) -> list[str]:
    """Resolve query person names to known persons with fuzzy matching.

    For each name: exact match first, then edit-distance fallback on
    shared surname+middle character. Returns resolved names; unmatched
    names are dropped with a warning.
    """
    known = _get_known_persons()
    resolved = []
    for name in query_persons:
        if name in known:
            resolved.append(name)
            continue
        # Prefix match: "王耀" matches "王耀东", "王耀华", etc.
        prefix_matches = [p for p in known if p.startswith(name) or name.startswith(p)]
        if len(prefix_matches) == 1:
            print(f"  [person] fuzzy match: {name!r} → {prefix_matches[0]!r}")
            resolved.append(prefix_matches[0])
            continue
        if len(prefix_matches) > 1:
            # Multiple prefix matches — ambiguous, skip
            print(f"  [person] ambiguous match: {name!r} → {prefix_matches}, skipped")
            continue
        # Substring match as last resort
        substr_matches = [p for p in known if name in p or p in name]
        if len(substr_matches) == 1:
            print(f"  [person] substring match: {name!r} → {substr_matches[0]!r}")
            resolved.append(substr_matches[0])
            continue
        if len(substr_matches) > 1:
            print(f"  [person] ambiguous substring: {name!r} → {substr_matches}, skipped")
            continue
        # Edit-distance fallback: find closest match within threshold
        # Useful for single-character typos in Chinese names (e.g. 王耀军 vs 王耀东)
        best_dist = len(name)
        best_count = 0
        best = None
        for candidate in known:
            if abs(len(candidate) - len(name)) > 1:
                continue
            dist = sum(a != b for a, b in zip(name, candidate))
            extra = abs(len(name) - len(candidate))
            dist += extra
            if dist < best_dist:
                best_dist = dist
                best = candidate
                best_count = 1
            elif dist == best_dist:
                best_count += 1
        if best and best_dist <= 1 and best_count == 1:
            print(f"  [person] edit-distance match: {name!r} → {best!r} (dist={best_dist})")
            resolved.append(best)
            continue
        print(f"  [person] unknown: {name!r}, no match in {len(known)} known persons")
    return resolved


# Module-level LLM config (set by CLI args or env vars)
_llm_base_url: str | None = None
_llm_api_key: str | None = None


def plan_query(query: str) -> dict:
    """Use LLM to decompose a natural language query into structured filters.

    Returns dict with optional keys: persons, date_from, date_to, location, semantic_query.
    """
    import anthropic
    import datetime

    today = datetime.date.today()
    last_year = today.year - 1
    persons = _get_known_persons()

    system = _PLANNER_SYSTEM.format(
        today=today.isoformat(),
        last_year=last_year,
        persons=", ".join(persons),
    )

    kwargs: dict = {}
    if _llm_base_url:
        kwargs["base_url"] = _llm_base_url
    if _llm_api_key:
        kwargs["api_key"] = _llm_api_key
    client = anthropic.Anthropic(**kwargs)
    resp = client.messages.create(
        model=_PLANNER_MODEL,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": query}],
    )

    import json

    text = _extract_text_from_response(resp).strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(text)
    print(f"  [planner] {query} → {result}")
    return result


def _plan_to_filters(plan: dict) -> tuple[set[int] | None, str, str]:
    """Convert planner output to (allowed_rowids, semantic_query, warning).

    Returns (rowid_set or None, semantic_query_string, warning_string).
    """
    raw_persons = plan.get("persons")
    person_match = plan.get("person_match", "all")
    date_from_str = plan.get("date_from")
    date_to_str = plan.get("date_to")
    location = plan.get("location")
    semantic = plan.get("semantic_query", "")

    # Build semantic query from the parts
    parts = [semantic]
    if location:
        parts.append(location)
    semantic_full = " ".join(p for p in parts if p)

    # Resolve person names with fuzzy matching
    persons = _resolve_persons(raw_persons) if raw_persons else None
    person_warning = ""
    if raw_persons and not persons:
        names_str = ", ".join(raw_persons)
        person_warning = f"⚠ Person not found: {names_str}. "

    # Resolve date filters
    ts_from = _parse_date(date_from_str, start_of_day=True) if date_from_str else None
    ts_to = _parse_date(date_to_str, start_of_day=False) if date_to_str else None

    # Resolve location to center point
    center = _geocode(location) if location else None

    # Apply filters (without location first, to check if person/date filters alone yield anything)
    has_filter = any(v is not None and v != [] for v in [
        persons, ts_from, ts_to, center
    ])

    if not has_filter:
        return None, semantic_full, person_warning

    db = sqlite3.connect(str(DB_PATH))
    base_kwargs: dict = {}
    if persons:
        base_kwargs["persons"] = persons
        base_kwargs["person_match"] = person_match
    if ts_from is not None:
        base_kwargs["date_from"] = ts_from
    if ts_to is not None:
        base_kwargs["date_to"] = ts_to

    if not center:
        rowids = _metadata_filter(db, **base_kwargs)
        db.close()
        return rowids, semantic_full, person_warning

    # Progressive location expansion: 1km → 2km → 10km → 20km → 30km → 50km → 100km
    import math
    for radius_km in _LOCATION_RADII_KM:
        lat_range, lon_range = _km_to_bbox(center[0], center[1], radius_km)
        kwargs = {
            **base_kwargs,
            "lat_min": lat_range[0], "lat_max": lat_range[1],
            "lon_min": lon_range[0], "lon_max": lon_range[1],
        }
        rowids = _metadata_filter(db, **kwargs)
        n = len(rowids) if rowids else 0
        lat_span_km = (lat_range[1] - lat_range[0]) * 111
        lon_span_km = (lon_range[1] - lon_range[0]) * 111 * math.cos(math.radians(center[0]))
        if n > 0:
            print(f"  [geo-filter] {location} ±{radius_km}km ({lat_span_km:.0f}×{lon_span_km:.0f}km) → {n} candidates")
            db.close()
            return rowids, semantic_full, person_warning
        print(f"  [geo-filter] {location} ±{radius_km}km ({lat_span_km:.0f}×{lon_span_km:.0f}km) → 0 candidates")

    # No candidates within 100km — give up on location, try without it
    print(f"  [geo-filter] {location}: no candidates within 100km, dropping location filter")
    rowids = _metadata_filter(db, **base_kwargs) if base_kwargs else None
    db.close()
    return rowids, semantic_full, person_warning


_LOCATION_RADII_KM = [1, 2, 10, 20, 30, 50, 100]
_geocode_cache: dict[str, tuple[float, float] | None] = {}
_geocode_last_request: float = 0.0


def _geocode(place_name: str) -> tuple[float, float] | None:
    """Geocode a place name to center (lat, lon).

    Uses Nominatim (OpenStreetMap) free geocoding API.
    Caches results and enforces 1 req/s rate limit per Nominatim policy.
    """
    import json
    import time
    import urllib.request
    import urllib.parse

    if place_name in _geocode_cache:
        cached = _geocode_cache[place_name]
        if cached:
            print(f"  [geocode] {place_name} → ({cached[0]:.4f}, {cached[1]:.4f}) (cached)")
        return cached

    # Enforce Nominatim rate limit: max 1 request per second
    global _geocode_last_request
    elapsed = time.monotonic() - _geocode_last_request
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    params = urllib.parse.urlencode({
        "q": place_name,
        "format": "json",
        "limit": 5,
        "polygon_text": 0,
        "accept-language": "zh",
        "countrycodes": "cn",
    })
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": "img-search/1.0"},
    )
    results = None
    for attempt in range(3):
        try:
            import ssl
            ctx = ssl.create_default_context()
            _geocode_last_request = time.monotonic()
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                results = json.loads(resp.read())
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(1.1)
                continue
            print(f"  [geocode] failed for '{place_name}': {e}")
            _geocode_cache[place_name] = None
            return None

    if not results:
        print(f"  [geocode] no results for '{place_name}'")
        _geocode_cache[place_name] = None
        return None

    # Pick best result: prefer higher importance, then presence of bounding box
    results.sort(key=lambda r: (
        float(r.get("importance") or 0),
        bool(r.get("boundingbox")),
    ), reverse=True)
    r = results[0]
    lat = float(r["lat"])
    lon = float(r["lon"])
    display = r.get("display_name", "").split(",")[0]
    print(f"  [geocode] {place_name} → ({lat:.4f}, {lon:.4f}) {display}")
    _geocode_cache[place_name] = (lat, lon)
    return lat, lon


def _km_to_bbox(
    center_lat: float, center_lon: float, radius_km: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Convert center point + radius in km to (lat_range, lon_range)."""
    import math
    d_lat = radius_km / 111.0
    d_lon = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    return (center_lat - d_lat, center_lat + d_lat), (center_lon - d_lon, center_lon + d_lon)


def _parse_metadata_filters(
    person_filter: str = "",
    date_from: str = "",
    date_to: str = "",
) -> set[int] | None:
    """Parse UI filter inputs into a set of allowed rowids, or None if no filters."""
    persons = [p.strip() for p in person_filter.split(",") if p.strip()] if person_filter else []
    ts_from = _parse_date(date_from, start_of_day=True) if date_from else None
    ts_to = _parse_date(date_to, start_of_day=False) if date_to else None

    if not persons and ts_from is None and ts_to is None:
        return None

    db = sqlite3.connect(str(DB_PATH))
    rowids = _metadata_filter(db, persons=persons or None, date_from=ts_from, date_to=ts_to)
    db.close()
    return rowids


def _parse_date(date_str: str, start_of_day: bool = True) -> float | None:
    """Parse YYYY-MM-DD to Apple Core Data timestamp (UTC-based)."""
    import datetime

    try:
        dt = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d")
        if not start_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        unix_ts = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        return unix_ts - CORE_DATA_EPOCH
    except (ValueError, TypeError):
        return None


# ─── Serve Command ─────────────────────────────────────────────────────────────


def cmd_smart_search(
    query: str,
    top_k: int = 10,
    no_rerank: bool = False,
    device: str | None = None,
):
    """Search with LLM-powered query planning."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    dev = resolve_device(device)
    print(f"Device: {dev}")

    db = init_db()
    total = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    db.close()
    if total == 0:
        print("No images in database. Run 'embed <path>' first.")
        return

    # Step 1: LLM query planning
    print(f"\nQuery: {query}")
    print("─" * 70)
    plan = plan_query(query)
    print(f"Plan: {plan}")

    # Step 2: Convert plan to filters + semantic query
    filter_rowids, semantic_query, warning = _plan_to_filters(plan)
    if warning:
        print(f"Warning: {warning}")
    if filter_rowids is not None:
        print(f"Metadata filter: {len(filter_rowids)} candidates")

    search_text = semantic_query or query
    print(f"Semantic search: '{search_text}'")

    # Step 3: Load embedding model and search
    mp = model_path(EMBEDDING_MODEL_ID)
    print(f"Loading embedding model from: {mp} ...")
    embed_model = SentenceTransformer(
        mp,
        cache_folder=str(MODEL_DIR),
        device=dev,
        model_kwargs={"torch_dtype": _model_dtype(dev)},
    )

    query_emb = embed_model.encode(
        [search_text], normalize_embeddings=True, show_progress_bar=False
    )
    rows = _knn_search_filtered(
        query_emb[0], max(top_k, _RERANK_S1_K), filter_rowids
    )

    if not rows:
        print("No results found")
        return

    if no_rerank:
        ranked = _rerank(search_text, rows, mode="none", rerank_k=top_k)
    else:
        mp = model_path(RERANKER_MODEL_ID)
        print(f"Loading reranker model from: {mp} ...")
        reranker = CrossEncoder(
            mp,
            cache_folder=str(MODEL_DIR),
            device=dev,
            model_kwargs={"torch_dtype": _model_dtype(dev)},
        )
        ranked = _rerank(search_text, rows, reranker)

    for rank, (score, path, filename) in enumerate(ranked, 1):
        marker = ">>>" if rank == 1 else "   "
        print(f"  {marker} #{rank:<3} score={score:.4f}  {path}")

    print("─" * 70)


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
        rows = _knn_search_filtered(query_vec, knn_k, None)
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

    def do_smart_search(text_query: str, rerank_mode: str, rerank_k: int):
        """Smart search using LLM query planning."""
        if not text_query:
            return [], "Please enter a query."

        t0 = time.perf_counter()

        # Step 1: LLM plan
        try:
            plan = plan_query(text_query)
        except Exception as e:
            return [], f"Query planning failed: {e}"

        filter_rowids, semantic_query, warning = _plan_to_filters(plan)
        search_text = semantic_query or text_query

        filter_info = f" | {len(filter_rowids)} candidates" if filter_rowids else ""
        plan_str = ", ".join(f"{k}={v}" for k, v in plan.items() if v)
        warn_prefix = warning or ""

        # Step 2: Embed and search
        query_emb = embed_model.encode(
            [search_text], normalize_embeddings=True, show_progress_bar=False
        )
        rows = _knn_search_filtered(query_emb[0], max(rerank_k, _RERANK_S1_K), filter_rowids)
        if not rows:
            return [], f"No results. Plan: {plan_str}"

        effective_mode = rerank_mode if reranker is not None else "none"
        ranked = _rerank(
            search_text, rows, reranker,
            mode=effective_mode, s1_k=_RERANK_S1_K, s2_k=_RERANK_S2_K, rerank_k=rerank_k,
        )

        elapsed = time.perf_counter() - t0
        print(f"  [timing] total: {elapsed:.3f}s")

        gallery = [
            (_ensure_browser_compatible(item[1]), f"#{i + 1}  score: {item[0]:.4f}")
            for i, item in enumerate(ranked)
        ]
        summary = f"{warn_prefix}Found {len(ranked)} in {elapsed:.1f}s | {plan_str}{filter_info}"
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
                with gr.Row(elem_classes=["search-row"]):
                    search_btn = gr.Button(
                        "\U0001f50d Search", variant="primary", scale=0, min_width=120
                    )
                    smart_btn = gr.Button(
                        "\U0001f9e0 Smart", variant="secondary", scale=0, min_width=120
                    )
                    status = gr.Markdown("", elem_classes=["search-status"])
            # Col 2: image input (2/7)
            image_input = gr.Image(
                label="Image",
                type="filepath",
                scale=2,
                show_label=True,
                sources=["upload"],
                elem_classes=["image-upload"],
            )

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

        smart_btn.click(
            fn=do_smart_search,
            inputs=[text_input, rerank_mode, rerank_k_input],
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


def _set_llm_config(args):
    """Set module-level LLM config from CLI args, falling back to env vars."""
    global _llm_base_url, _llm_api_key
    _llm_base_url = getattr(args, "llm_base_url", None) or os.environ.get("ANTHROPIC_BASE_URL")
    _llm_api_key = getattr(args, "llm_api_key", None) or os.environ.get("ANTHROPIC_API_KEY")


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
    srch.add_argument("--person", help="Filter by person name (comma-sep for multiple)")
    srch.add_argument("--date-from", help="Filter: start date (YYYY-MM-DD)")
    srch.add_argument("--date-to", help="Filter: end date (YYYY-MM-DD)")

    idx = sub.add_parser("index-photos", help="Index Apple Photos metadata")
    idx.add_argument("library_path", help="Path to .photoslibrary bundle")

    smart = sub.add_parser("smart-search", help="NL search with LLM query planning")
    smart.add_argument("query", help="Natural language query (e.g. '王梓涵去年在广州的照片')")
    smart.add_argument("--top-k", type=int, default=10, help="Number of results")
    smart.add_argument("--no-rerank", action="store_true", help="Skip reranking")
    smart.add_argument("--device", help="Compute device: cuda, mps, cpu")
    smart.add_argument("--llm-base-url", help="Anthropic-compatible API base URL")
    smart.add_argument("--llm-api-key", help="API key for the LLM service")

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
    srv.add_argument("--llm-base-url", help="Anthropic-compatible API base URL")
    srv.add_argument("--llm-api-key", help="API key for the LLM service")

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args.path, device=getattr(args, "device", None))
    elif args.command == "index-photos":
        cmd_index_photos(args.library_path)
    elif args.command == "search":
        cmd_search(
            args.desc,
            top_k=args.top_k,
            no_rerank=args.no_rerank,
            device=getattr(args, "device", None),
            person=getattr(args, "person", None),
            date_from=getattr(args, "date_from", None),
            date_to=getattr(args, "date_to", None),
        )
    elif args.command == "serve":
        _set_llm_config(args)
        cmd_serve(
            port=args.port,
            no_rerank=args.no_rerank,
            device=getattr(args, "device", None),
        )
    elif args.command == "smart-search":
        _set_llm_config(args)
        cmd_smart_search(
            args.query,
            top_k=args.top_k,
            no_rerank=args.no_rerank,
            device=getattr(args, "device", None),
        )


if __name__ == "__main__":
    main()
