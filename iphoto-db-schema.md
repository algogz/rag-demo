# Apple Photos Library Database Schema

**Database:** `iPhoto.photoslibrary/database/Photos.sqlite` (SQLite 3.x, ~4 GB)
**Core Data version:** Schema 4, page size 8192

## Core Data Conventions

Apple Photos uses **Core Data** with SQLite as the backing store.

| Convention | Description |
|------------|-------------|
| `Z_PK` | Auto-increment primary key |
| `Z_ENT` | Entity type ID, maps to table name via `Z_PRIMARYKEY` |
| `Z_OPT` | Core Data optimization flag |
| Timestamps | Seconds since **2001-01-01 00:00:00 UTC**. Convert: `+ 978307200` → Unix epoch |
| Foreign keys | Named after target entity (e.g. `ZASSET.ZMOMENT → ZMOMENT.Z_PK`) |
| `Z_FOK_*` | "Fake opposite key" — Core Data denormalization for inverse relationships |

## Entity Registry

`Z_PRIMARYKEY` maps `Z_ENT` values to table names:

| Z_ENT | Table |
|------:|-------|
| 1 | AdditionalAssetAttributes |
| 2 | AlbumList |
| 3 | **Asset** |
| 23 | **DetectedFace** |
| 24 | DetectedFaceGroup |
| 25 | DetectedFaceprint |
| 28 | **ExtendedAttributes** |
| 32 | **GenericAlbum** |
| 33 | Album |
| 52 | **Keyword** |
| 56 | **Memory** |
| 58 | **Moment** |
| 59 | **Person** |
| 62 | PhotosHighlight |
| 64 | **SceneClassification** |

Full list has 75+ entity types.

## Entity Relationship Diagram

```
ZGENERICALBUM (albums, folders, smart albums)
  ├── Z_33ASSETS        ── many-to-many ──► ZASSET
  ├── Z_32ALBUMLISTS    ── parent/child ──► ZGENERICALBUM (folder hierarchy)
  └── ZKIND: 1506=moment grouping, 2=smart, etc.

ZASSET (the central table — every photo/video)
  ├── ZEXTENDEDATTRIBUTES       (1:1) ── EXIF: camera, lens, aperture, ISO, shutter
  ├── ZADDITIONALASSETATTRIBUTES (1:1) ── file metadata: original filename, size, play count, title
  ├── ZMOMENT                   (N:1) ──► ZMOMENT (time/place grouping)
  ├── ZMEDIAANALYSISATTRIBUTES  (1:1) ── ML analysis state
  ├── ZCOMPUTEDATTRIBUTES       (1:1) ── curation/aesthetic scores
  ├── Z_1KEYWORDS (via ZADDITIONALASSETATTRIBUTES) ──► ZKEYWORD
  │
  ├── ZDIRECTORY + ZFILENAME    ── file path under originals/
  ├── ZKIND                     ── 0=photo, 1=video
  ├── ZFAVORITE, ZHIDDEN, ZTRASHEDSTATE
  ├── ZDATECREATED, ZLATITUDE, ZLONGITUDE
  ├── ZWIDTH, ZHEIGHT, ZDURATION
  └── ZUNIFORMTYPEIDENTIFIER    ── UTI (public.jpeg, public.heic, etc.)

ZMOMENT (auto-generated time+place clusters)
  ├── ZTITLE, ZSUBTITLE        ── e.g. "Guangzhou, December 2014"
  ├── ZSTARTDATE, ZENDDATE
  ├── ZAPPROXIMATELATITUDE/LONGITUDE
  └── ZHIGHLIGHT ──► ZPHOTOSHIGHLIGHT

ZPHOTOSHIGHLIGHT (day/month/year groupings for the Photos tab)
  ├── ZKEYASSETPRIVATE/SHARED   ──► ZASSET (cover photo)
  ├── ZPARENTPHOTOSHIGHLIGHT    ──► self (day → month → year hierarchy)
  └── ZKIND, ZCATEGORY, ZMOOD

ZDETECTEDFACE (individual face detection in a photo)
  ├── ZASSETFORFACE             ──► ZASSET (which photo)
  ├── ZPERSONFORFACE            ──► ZPERSON (identified person)
  ├── ZCENTERX, ZCENTERY, ZSIZE ── face bounding box (normalized 0-1)
  ├── ZHASSMILE, ZGAZETYPE, ZGENDERTYPE, ZAGETYPE, etc.
  └── ZFACECROP                 ──► FaceCrop (thumbnail)

ZPERSON (identified people)
  ├── ZDISPLAYNAME, ZFULLNAME
  ├── ZFACECOUNT
  ├── ZTYPE: 0=detected, 1=verified (user-confirmed)
  └── ZKEYFACE ──► ZDETECTEDFACE (representative face)

ZSCENECLASSIFICATION (ML scene/object detection per photo)
  ├── ZASSETATTRIBUTES          ──► ZADDITIONALASSETATTRIBUTES
  ├── ZSCENEIDENTIFIER          ── integer code for detected scene
  ├── ZCONFIDENCE               ── detection confidence (0.0-1.0)
  └── ~3.8M rows — every photo has dozens of scene labels

ZMEMORY (auto-generated "Memories" collections)
  ├── ZTITLE, ZSUBTITLE, ZCATEGORY
  ├── Z_3MEMORIESBEINGCURATEDASSETS ──► ZASSET (many-to-many)
  └── ZSCORE, ZFEATUREDSTATE

ZKEYWORD (user tags)
  └── Z_1KEYWORDS ──► ZADDITIONALASSETATTRIBUTES (many-to-many)
```

## Key Tables

### ZASSET — The Central Table

134 columns. Every photo and video is one row.

**Important columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Z_PK` | INTEGER | Primary key |
| `ZKIND` | INTEGER | 0=photo, 1=video |
| `ZFAVORITE` | INTEGER | 1=hearted |
| `ZHIDDEN` | INTEGER | 1=hidden |
| `ZTRASHEDSTATE` | INTEGER | Non-zero=trashed |
| `ZDATECREATED` | TIMESTAMP | Core Data timestamp (see conversion above) |
| `ZLATITUDE` / `ZLONGITUDE` | FLOAT | GPS coordinates |
| `ZWIDTH` / `ZHEIGHT` | INTEGER | Pixel dimensions |
| `ZDURATION` | FLOAT | Video duration in seconds |
| `ZDIRECTORY` | VARCHAR | Hash bucket folder (single char: 0-9, A-F) |
| `ZFILENAME` | VARCHAR | UUID-based filename with extension |
| `ZUNIFORMTYPEIDENTIFIER` | VARCHAR | UTI: `public.jpeg`, `public.heic`, `com.apple.quicktime-movie`, etc. |
| `ZUUID` | VARCHAR | Canonical UUID |
| `ZCLOUDASSETGUID` | VARCHAR | iCloud sync identifier (NULL if local-only) |
| `ZOVERALLAESTHETICSCORE` | FLOAT | ML aesthetic score |
| `ZCURATIONSCORE` | FLOAT | Curation ranking |
| `ZICONICSCORE` | FLOAT | "Iconic" quality score |
| `ZPROMOTIONSCORE` | FLOAT | For Memories promotion |
| `ZORIENTATION` | INTEGER | EXIF orientation value |
| `ZMOMENT` | INTEGER | FK → ZMOMENT.Z_PK |
| `ZEXTENDEDATTRIBUTES` | INTEGER | FK → ZEXTENDEDATTRIBUTES.Z_PK |
| `ZADDITIONALASSETATTRIBUTES` | INTEGER | FK → ZADDITIONALASSETATTRIBUTES.Z_PK |

### ZEXTENDEDATTRIBUTES — EXIF Data (1:1 with ZASSET)

| Column | Type | Description |
|--------|------|-------------|
| `ZCAMERAMAKE` | VARCHAR | e.g. "Apple" |
| `ZCAMERAMODEL` | VARCHAR | e.g. "iPhone 15 Pro Max" |
| `ZLENSMODEL` | VARCHAR | e.g. "iPhone X back dual camera 4mm f/1.8" |
| `ZFOCALLENGTH` | FLOAT | Focal length in mm |
| `ZFOCALLENGTHIN35MM` | INTEGER | 35mm-equivalent focal length |
| `ZAPERTURE` | FLOAT | f-number |
| `ZSHUTTERSPEED` | FLOAT | Shutter speed (reciprocal, e.g. 0.008 = 1/125) |
| `ZISO` | INTEGER | ISO value |
| `ZFLASHFIRED` | INTEGER | 1=flash fired |
| `ZWHITEBALANCE` | INTEGER | White balance mode |
| `ZMETERINGMODE` | INTEGER | Metering mode |
| `ZEXPOSUREBIAS` | FLOAT | Exposure compensation |
| `ZBITRATE` | FLOAT | Video bitrate |
| `ZFPS` | FLOAT | Video frame rate |
| `ZCODEC` | VARCHAR | Video codec |
| `ZSAMPLERATE` | INTEGER | Audio sample rate |
| `ZTRACKFORMAT` | INTEGER | Audio track format |
| `ZASSET` | INTEGER | FK → ZASSET.Z_PK (inverse) |

### ZADDITIONALASSETATTRIBUTES — Extended Metadata (1:1 with ZASSET)

| Column | Type | Description |
|--------|------|-------------|
| `ZORIGINALFILENAME` | VARCHAR | Original filename at import |
| `ZORIGINALFILESIZE` | INTEGER | File size in bytes |
| `ZORIGINALWIDTH` / `ZORIGINALHEIGHT` | INTEGER | Original dimensions |
| `ZTITLE` | VARCHAR | User-assigned title |
| `ZPLAYCOUNT` | INTEGER | Number of plays |
| `ZVIEWCOUNT` | INTEGER | Number of views |
| `ZTIMEZONEOFFSET` | INTEGER | Timezone offset in seconds |
| `ZTIMEZONENAME` | VARCHAR | e.g. "Asia/Shanghai" |
| `ZIMPORTEDBYBUNDLEIDENTIFIER` | VARCHAR | App that imported the asset |
| `ZIMPORTSESSIONID` | VARCHAR | Import session reference |
| `ZSCENEANALYSISVERSION` | INTEGER | ML scene analysis version |
| `ZSCENEPRINT` | INTEGER | FK → scene print data |
| `ZEDITEDIPTCATTRIBUTES` | INTEGER | FK → IPTC metadata |
| `ZASSETDESCRIPTION` | INTEGER | FK → description |
| `ZREVERSELOCATIONDATA` | BLOB | Reverse geocoded place names |
| `ZFACEREGIONS` | BLOB | Face region data |
| `ZPLACEANNOTATIONDATA` | BLOB | Place annotation |
| `ZOBJECTSALIENCYRECTSDATA` | BLOB | Object saliency detection |

### ZGENERICALBUM — Albums, Folders, Smart Albums

| Column | Type | Description |
|--------|------|-------------|
| `ZTITLE` | VARCHAR | Album name |
| `ZKIND` | INTEGER | Album type (see below) |
| `ZCACHEDCOUNT` | INTEGER | Total asset count (cached) |
| `ZCACHEDPHOTOSCOUNT` / `ZCACHEDVIDEOSCOUNT` | INTEGER | Photo/video counts |
| `ZPARENTFOLDER` | INTEGER | FK → parent ZGENERICALBUM |
| `ZISPINNED` | INTEGER | 1=pinned in sidebar |
| `ZTRASHEDSTATE` | INTEGER | 1=deleted |
| `ZCREATIONDATE` | TIMESTAMP | Album creation date |
| `ZSTARTDATE` / `ZENDDATE` | TIMESTAMP | Date range of contents |
| `ZCUSTOMSORTKEY` | INTEGER | Sort order |
| `ZCUSTOMSORTASCENDING` | INTEGER | Sort direction |

**ZKIND values observed:**

| ZKIND | Meaning | Count |
|------:|---------|------:|
| 1506 | Auto-generated moment groupings | 6,341 |
| 2 | Smart albums | 1,336 |
| 1510 | Special album type | 963 |
| 4000 | Other special type | 2 |
| 1507, 1552, 1600+ | Various album subtypes | 1 each |

### ZDETECTEDFACE — Face Detections

| Column | Type | Description |
|--------|------|-------------|
| `ZASSETFORFACE` | INTEGER | FK → ZASSET.Z_PK |
| `ZPERSONFORFACE` | INTEGER | FK → ZPERSON.Z_PK |
| `ZCENTERX` / `ZCENTERY` | FLOAT | Face center (normalized 0-1) |
| `ZSIZE` | FLOAT | Face size (normalized) |
| `ZBODYCENTERX/Y`, `ZBODYWIDTH/HEIGHT` | FLOAT | Body bounding box |
| `ZHASSMILE` | INTEGER | 1=smile detected |
| `ZGAZETYPE` | INTEGER | Gaze direction |
| `ZGENDERTYPE` | INTEGER | Gender classification |
| `ZAGETYPE` | INTEGER | Age bracket |
| `ZFACIALHAIRTYPE` | INTEGER | Facial hair |
| `ZGLASSESTYPE` | INTEGER | Glasses detection |
| `ZFACEEXPRESSIONTYPE` | INTEGER | Expression type |
| `ZQUALITY` | FLOAT | Detection quality score |
| `ZSOURCEWIDTH` / `ZSOURCEHEIGHT` | INTEGER | Image dimensions at detection time |
| `ZFACECROP` | INTEGER | FK → FaceCrop (thumbnail) |
| `ZFACEPRINT` | INTEGER | FK → face embedding data |
| `ZDURATION` / `ZSTARTTIME` | FLOAT | For video face tracks |

### ZPERSON — Identified People

| Column | Type | Description |
|--------|------|-------------|
| `ZDISPLAYNAME` | VARCHAR | Display name (user-assigned or suggested) |
| `ZFULLNAME` | VARCHAR | Full name |
| `ZFACECOUNT` | INTEGER | Number of associated faces |
| `ZTYPE` | INTEGER | 0=detected, 1=verified (user-confirmed) |
| `ZVERIFIEDTYPE` | INTEGER | Verification status |
| `ZKEYFACE` | INTEGER | FK → ZDETECTEDFACE (representative face) |
| `ZMERGETARGETPERSON` | INTEGER | FK → ZPERSON (merge target) |
| `ZGENDERTYPE` | INTEGER | Gender classification |
| `ZAGETYPE` | INTEGER | Age bracket |
| `ZPERSONUUID` | VARCHAR | Persistent UUID |

### ZMOMENT — Time/Place Clusters

| Column | Type | Description |
|--------|------|-------------|
| `ZTITLE` | VARCHAR | e.g. "Guangzhou" |
| `ZSUBTITLE` | VARCHAR | e.g. "December 2014" |
| `ZSTARTDATE` / `ZENDDATE` | TIMESTAMP | Time range |
| `ZAPPROXIMATELATITUDE` / `ZLONGITUDE` | FLOAT | Approximate location |
| `ZCACHEDCOUNT` | INTEGER | Asset count |
| `ZCACHEDPHOTOSCOUNT` / `ZCACHEDVIDEOSCOUNT` | INTEGER | Photo/video counts |
| `ZHIGHLIGHT` | INTEGER | FK → ZPHOTOSHIGHLIGHT |
| `ZTIMEZONEOFFSET` | INTEGER | Timezone offset |
| `ZUUID` | VARCHAR | Persistent UUID |

### ZSCENECLASSIFICATION — ML Scene Labels

| Column | Type | Description |
|--------|------|-------------|
| `ZSCENEIDENTIFIER` | INTEGER | Scene class ID |
| `ZASSETATTRIBUTES` | INTEGER | FK → ZADDITIONALASSETATTRIBUTES |
| `ZCONFIDENCE` | FLOAT | Detection confidence (0.0-1.0) |
| `ZCLASSIFICATIONTYPE` | INTEGER | Classification type |
| `ZPACKEDBOUNDINGBOXRECT` | INTEGER | Bounding box (for object detection) |
| `ZSTARTTIME` / `ZDURATION` | FLOAT | For video frame ranges |

~3.8M rows. Every photo typically has dozens of scene/object labels.

### ZMEMORY — Auto-Generated Memories

| Column | Type | Description |
|--------|------|-------------|
| `ZTITLE` | VARCHAR | Memory title |
| `ZSUBTITLE` | VARCHAR | Memory subtitle |
| `ZCATEGORY` | INTEGER | Memory category |
| `ZSCORE` | FLOAT | Relevance score |
| `ZFAVORITE` | INTEGER | 1=user favorited |
| `ZFEATUREDSTATE` | INTEGER | Featured status |
| `ZREJECTED` | INTEGER | 1=user dismissed |
| `ZKEYASSET` | INTEGER | FK → ZASSET (cover) |
| `ZCREATIONDATE` | TIMESTAMP | When memory was generated |
| `ZLASTVIEWEDDATE` | TIMESTAMP | Last viewed |
| `ZSTARTDATE` / `ZENDDATE` | TIMESTAMP | Date range of assets |

### ZPHOTOSHIGHLIGHT — Photos Tab Groupings

| Column | Type | Description |
|--------|------|-------------|
| `ZTITLE` | VARCHAR | Group title |
| `ZSUBTITLE` | VARCHAR | Group subtitle |
| `ZKIND` | INTEGER | Highlight kind |
| `ZCATEGORY` | INTEGER | Category |
| `ZMOOD` | INTEGER | Mood classification |
| `ZKEYASSETPRIVATE` / `ZKEYASSETSHARED` | INTEGER | FK → ZASSET (cover photos) |
| `ZPARENTPHOTOSHIGHLIGHT` | INTEGER | FK → self (day → month → year) |
| `ZSTARTDATE` / `ZENDDATE` | TIMESTAMP | Date range |
| `ZASSETSCOUNT` | INTEGER | Total assets |
| `ZVISIBILITYSTATE` | INTEGER | Visibility |

### ZKEYWORD — User Tags

| Column | Type | Description |
|--------|------|-------------|
| `ZTITLE` | VARCHAR | Keyword text |
| `ZSHORTCUT` | VARCHAR | Keyboard shortcut |
| `ZUUID` | VARCHAR | Persistent UUID |

Only 16 keywords in this library. Linked to assets via `Z_1KEYWORDS` junction table.

## Junction Tables (Many-to-Many)

| Table | Links |
|-------|-------|
| `Z_33ASSETS` | Album ↔ Asset (`Z_33ALBUMS`, `Z_3ASSETS`) |
| `Z_32ALBUMLISTS` | Album ↔ Sub-album (`Z_32ALBUMS`, `Z_2ALBUMLISTS`) |
| `Z_1KEYWORDS` | Keyword ↔ AssetAttributes (`Z_52KEYWORDS`, `Z_1ASSETATTRIBUTES`) |
| `Z_3MEMORIESBEINGCURATEDASSETS` | Memory ↔ Asset |
| `Z_3MEMORIESBEINGKEYASSETS` | Memory ↔ Key Asset |
| `Z_3SUGGESTIONSBEINGKEYASSETS` | Suggestion ↔ Key Asset |
| `Z_59MERGECANDIDATES` | Person ↔ Merge Candidate Person |

## File Path Resolution

Assets live on disk at:

```
iPhoto.photoslibrary/originals/{ZDIRECTORY}/{ZFILENAME}
```

- `ZDIRECTORY` is a single hex char (`0`–`9`, `A`–`F`) acting as a hash bucket
- `ZFILENAME` is a UUID with extension (e.g. `A3296E6C-F15E-4913-A9F5-20199C17A8F2.jpeg`)
- Each bucket contains ~5,000–5,400 files

Example: `ZDIRECTORY='A'`, `ZFILENAME='A3296E6C...jpeg'` → `originals/A/A3296E6C...jpeg`

## Timestamp Conversion

Apple Core Data stores timestamps as seconds since **2001-01-01 00:00:00 UTC**.

```sql
-- Convert to human-readable in SQLite
datetime(ZDATECREATED + 978307200, 'unixepoch')

-- In Python
import datetime
dt = datetime.datetime(2001, 1, 1) + datetime.timedelta(seconds=core_data_timestamp)

-- The constant
978307200  # seconds between 1970-01-01 and 2001-01-01
```

## Relationship Patterns

| Pattern | Mechanism | Example |
|---------|-----------|---------|
| 1:1 | Direct FK column | `ZASSET.ZEXTENDEDATTRIBUTES → ZEXTENDEDATTRIBUTES.Z_PK` |
| N:1 | Direct FK column | `ZASSET.ZMOMENT → ZMOMENT.Z_PK` |
| Many-to-many | Junction table `Z_{IDs}` | `Z_33ASSETS` links albums to assets |
| Self-referencing | FK to same table | `ZGENERICALBUM.ZPARENTFOLDER → ZGENERICALBUM.Z_PK` |
| Hierarchy | Parent FK chain | `ZPHOTOSHIGHLIGHT`: day → month → year |

## Key Record Counts

| Table | Count |
|-------|------:|
| ZASSET | 84,357 |
| ZEXTENDEDATTRIBUTES | 84,357 (1:1 with ZASSET) |
| ZADDITIONALASSETATTRIBUTES | 84,357 (1:1 with ZASSET) |
| ZGENERICALBUM | 8,691 |
| ZPERSON | 193,149 |
| ZDETECTEDFACE | 145,813 |
| ZSCENECLASSIFICATION | 3,798,707 |
| ZMOMENT | 10,282 |
| ZMEMORY | 1,684 |
| ZPHOTOSHIGHLIGHT | ~varies |
| ZKEYWORD | 16 |
| ZCLOUDMASTER | 0 (local-only library) |
