# Photo Intelligence Search — Implementation Plan

## Goal

Enable complex queries like "王耀军去年在西藏旅游的照片" by combining Apple Photos structured metadata (persons, GPS, dates, scenes) with VL embedding/reranking.

## Architecture

```
User Query (NL, bilingual)
         │
         ▼
  ┌──────────────────┐
  │  LLM Query Planner│  ← Decomposes NL → structured filters + semantic intent
  └──────┬───────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
Structured    Semantic
Filters       Text
    │          │
    ▼          │
┌────────────┐ │
│  SQL Filter │ │  ← Narrow 84K → ~hundreds
│  (metadata) │ │
└─────┬──────┘ │
      │        │
      ▼        ▼
  ┌────────────────┐
  │  Vector Search  │  ← KNN on filtered subset
  │  + Reranker     │
  └────────────────┘
         │
         ▼
     Ranked Results
```

## Implementation Phases

### Phase 1 — Metadata Indexer

Build a metadata layer in `image_vectors.db` linking image rows to Apple Photos structured data.

- New tables: `photo_metadata`, `photo_persons`, `photo_scenes`
- Path matching: `images.path` ↔ `ZASSET` via `originals/{ZDIRECTORY}/{ZFILENAME}`
- Single CLI command: `img-search.py index-photos <library_path>`

### Phase 2 — Person + Date Filter

- Extend `_knn_search` to accept optional person/date filters
- SQL pre-filter narrows candidates before vector search
- Handles queries like "王耀军去年" (person + date range)

### Phase 3 — Location Filter

- Geocoding service for place names → GPS bounding boxes
- Reverse-geocode existing `ZMOMENT` titles as fallback
- Handles "在西藏" → lat/lng bounding box filter

### Phase 4 — LLM Query Planner

- NL query → structured JSON filters + semantic text
- Prompt engineering for bilingual (EN/ZH) queries
- Integrates with the filter-then-rank pipeline

### Phase 5 — Scene + Keyword Filtering (Bonus)

- Scene classification labels from `ZSCENECLASSIFICATION` (3.8M rows)
- User keywords from `ZKEYWORD`
- Additional structured filter dimensions

## Key Technical Details

- **Timestamps**: Apple Core Data = seconds since 2001-01-01. Convert: `+ 978307200`
- **Path resolution**: `originals/{ZDIRECTORY}/{ZFILENAME}` → match against `images.path`
- **Geocoding**: Nominatim (free) or pre-computed location clusters from 84K geotagged photos
- **LLM Planner**: Any small model works; structured output with JSON schema
