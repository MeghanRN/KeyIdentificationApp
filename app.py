"""
Key Identification App (Streamlit)
----------------------------------
Scan a key (via camera or image upload) and identify which saved key it matches.
Create and manage a private database of your own keys.

Quick start
===========
1) (Recommended) Create & activate a virtual environment
   python -m venv .venv
   # Windows: .venv\\Scripts\\activate   |   macOS/Linux: source .venv/bin/activate

2) Install dependencies
   pip install -r requirements.txt

   If you don't want a separate file, install directly:
   pip install streamlit pillow imagehash numpy pandas

3) Run the app
   streamlit run app.py

Data & security
===============
- Your data lives locally in the ./data folder next to this script.
- Database file: data/keys.db
- Images saved under: data/images/
- Exports go to: data/exports/
- Keep this folder private. This app is meant for *personal* record-keeping only.

Notes on identification
=======================
This app uses perceptual image hashing (aHash / pHash / dHash / wHash) to find the closest
match from your own saved keys. It does *not* decode key bitting or claim to identify
manufacturer/lock model. Treat results as best-effort similarity matching.
"""

from __future__ import annotations
import io
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import imagehash
import streamlit as st
from cv_key import extract_shape_features, ShapeFeatures, match_score  # new

# ---------------------------
# Constants & Paths
# ---------------------------
APP_TITLE = "Key Identifier"
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "images"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "keys.db"

HASH_SIZE = 12  # larger values => more detailed hashes (default 8). 12 -> 144-bit
DEFAULT_MATCH_TOPK = 5
DEFAULT_ACCEPT_THRESHOLD = 35  # lower = more similar (sum of Hamming distances)
THUMBNAIL_SIZE = (220, 220)

# Ensure folders exist
for p in [DATA_DIR, IMG_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Data layer
# ---------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS keys (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    purpose TEXT NOT NULL,
    description TEXT,
    tags TEXT,
    image_path TEXT NOT NULL,
    ahash TEXT,
    phash TEXT,
    dhash TEXT,
    whash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_SQL_SHAPES = """
CREATE TABLE IF NOT EXISTS key_shapes (
    key_id INTEGER PRIMARY KEY,
    svg TEXT,
    hu TEXT,           -- JSON list[float]
    fourier TEXT,      -- JSON list[float]
    contour TEXT,      -- JSON list[[x,y], ...] length ~256
    width INTEGER,
    height INTEGER,
    FOREIGN KEY(key_id) REFERENCES keys(id) ON DELETE CASCADE
);
"""

@dataclass
class KeyRecord:
    id: int
    name: str
    purpose: str
    description: str
    tags: str
    image_path: str
    ahash: Optional[str]
    phash: Optional[str]
    dhash: Optional[str]
    whash: Optional[str]
    created_at: str

# --- Optional live capture with overlay ---
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    import cv2  # you already use cv2 inside cv_key; keep this import here too for overlay drawing
    STREAM_WEBRTC_OK = True
except Exception:
    STREAM_WEBRTC_OK = False

import threading
_frame_lock = threading.Lock()
_last_frame = None

class FramingOverlay(VideoProcessorBase):
    def recv(self, frame):
        global _last_frame
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Draw a translucent guide box (centered)
        box_w, box_h = int(0.7 * w), int(0.35 * h)
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2
        x1, y1 = x0 + box_w, y0 + box_h

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), thickness=2)
        # center crosshair
        cv2.line(overlay, (x0, (y0 + y1)//2), (x1, (y0 + y1)//2), (255, 255, 255), 1)
        cv2.line(overlay, ((x0 + x1)//2, y0), ((x0 + x1)//2, y1), (255, 255, 255), 1)

        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        with _frame_lock:
            _last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(SCHEMA_SQL_SHAPES)

def insert_key_shape(key_id: int, svg: Optional[str], feats: Dict[str, object]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO key_shapes (key_id, svg, hu, fourier, contour, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key_id,
                svg,
                json.dumps(feats["hu"]),
                json.dumps(feats["fourier"]),
                json.dumps(feats["contour"]),
                int(feats["width"]),
                int(feats["height"]),
            ),
        )
        conn.commit()


def fetch_key_shapes() -> List[Dict[str, object]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT k.id, k.name, k.purpose, k.image_path,
                   s.svg, s.hu, s.fourier, s.contour, s.width, s.height
            FROM keys k
            JOIN key_shapes s ON s.key_id = k.id
            ORDER BY k.created_at DESC
        """)
        rows = cur.fetchall()
    out = []
    for row in rows:
        (kid, name, purpose, image_path, svg, hu, fourier, contour, w, h) = row
        out.append({
            "id": kid,
            "name": name,
            "purpose": purpose,
            "image_path": image_path,
            "svg": svg,
            "hu": json.loads(hu),
            "fourier": json.loads(fourier),
            "contour": json.loads(contour),
            "width": w,
            "height": h,
        })
    return out

def insert_key(name: str, purpose: str, description: str, tags: str, image_path: str, 
               hash_dict: Dict[str, Optional[str]]) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO keys (name, purpose, description, tags, image_path, ahash, phash, dhash, whash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(), purpose.strip(), (description or "").strip(), (tags or "").strip(),
                image_path,
                hash_dict.get("ahash"), hash_dict.get("phash"), hash_dict.get("dhash"), hash_dict.get("whash"),
            ),
        )
        conn.commit()
        return cur.lastrowid
    
def _pow2_le(n: int) -> int:
    """Return the largest power of two <= n, with a minimum of 2."""
    if n < 2:
        return 2
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p

def delete_key(key_id: int) -> None:
    with get_conn() as conn:
        # fetch image path to delete the file too
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM keys WHERE id=?", (key_id,))
        row = cur.fetchone()
        if row and row[0]:
            try:
                Path(row[0]).unlink(missing_ok=True)
            except Exception:
                pass
        cur.execute("DELETE FROM keys WHERE id=?", (key_id,))
        conn.commit()


def fetch_keys() -> List[KeyRecord]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, purpose, description, tags, image_path, ahash, phash, dhash, whash, created_at FROM keys ORDER BY created_at DESC")
        rows = conn.cursor().fetchall() if False else cur.fetchall()
    return [KeyRecord(*row) for row in rows]


def fetch_keys_df() -> pd.DataFrame:
    rows = fetch_keys()
    if not rows:
        return pd.DataFrame(columns=["id", "name", "purpose", "description", "tags", "image_path", "created_at"])  # hashes hidden by default
    df = pd.DataFrame([{
        "id": r.id,
        "name": r.name,
        "purpose": r.purpose,
        "description": r.description,
        "tags": r.tags,
        "image_path": r.image_path,
        "ahash": r.ahash,
        "phash": r.phash,
        "dhash": r.dhash,
        "whash": r.whash,
        "created_at": r.created_at,
    } for r in rows])
    return df


# ---------------------------
# Image & Hash utilities
# ---------------------------

def _open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    # Normalize: auto-orient + crop borders + resize for consistency (not strictly needed for hashes)
    img = ImageOps.exif_transpose(img)
    return img


def _save_image(img: Image.Image, suggested_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch for ch in suggested_name if ch.isalnum() or ch in ("-", "_"))[:40]
    fname = f"{ts}_{safe or 'key'}.jpg"
    out_path = IMG_DIR / fname
    img.save(out_path, format="JPEG", quality=92)
    return str(out_path)


def compute_hashes(img: Image.Image, hash_size: int = HASH_SIZE) -> Dict[str, str]:
    # Convert to a smaller, contrast-normalized image to stabilize hashing
    base = ImageOps.autocontrast(img)

    # a/p/d hash_size can be any int; wHash must be power of two.
    wsize = _pow2_le(hash_size)

    return {
        "ahash": str(imagehash.average_hash(base, hash_size=hash_size)),
        "phash": str(imagehash.phash(base, hash_size=hash_size)),
        "dhash": str(imagehash.dhash(base, hash_size=hash_size)),
        "whash": str(imagehash.whash(base, hash_size=wsize)),
    }


def hamming_distance(hex_a: Optional[str], hex_b: Optional[str]) -> Optional[int]:
    if not hex_a or not hex_b:
        return None
    try:
        return imagehash.hex_to_hash(hex_a) - imagehash.hex_to_hash(hex_b)
    except Exception:
        return None


def combined_distance(q_hashes: Dict[str, str], r: KeyRecord) -> Tuple[int, Dict[str, Optional[int]]]:
    comps = {
        "ahash": hamming_distance(q_hashes.get("ahash"), r.ahash),
        "phash": hamming_distance(q_hashes.get("phash"), r.phash),
        "dhash": hamming_distance(q_hashes.get("dhash"), r.dhash),
        "whash": hamming_distance(q_hashes.get("whash"), r.whash),
    }
    # sum only valid distances
    valid = [v for v in comps.values() if isinstance(v, int)]
    total = int(sum(valid)) if valid else 10**9
    return total, comps


def find_best_matches(q_hashes: Dict[str, str], top_k: int = DEFAULT_MATCH_TOPK) -> List[Tuple[KeyRecord, int, Dict[str, Optional[int]]]]:
    records = fetch_keys()
    scored = []
    for r in records:
        total, comps = combined_distance(q_hashes, r)
        scored.append((r, total, comps))
    scored.sort(key=lambda x: x[1])
    return scored[:top_k]


# ---------------------------
# UI Helpers
# ---------------------------

def key_card(rec: KeyRecord):
    cols = st.columns([1, 2])
    with cols[0]:
        try:
            st.image(rec.image_path, caption=f"#{rec.id}", use_container_width=True)
        except Exception:
            st.write("(image missing)")
    with cols[1]:
        st.markdown(f"**{rec.name}**")
        st.caption(f"Purpose: {rec.purpose}")
        if rec.description:
            st.write(rec.description)
        if rec.tags:
            st.write(f"Tags: `{rec.tags}`")
        st.caption(f"Added: {rec.created_at}")
        del_col, _ = st.columns([1, 4])
        with del_col:
            if st.button("Delete", type="secondary", key=f"del_{rec.id}"):
                delete_key(rec.id)
                st.success(f"Deleted key #{rec.id}")
                st.experimental_rerun()


def show_match_row(r: KeyRecord, total: int, comps: Dict[str, Optional[int]]):
    with st.expander(f"#{r.id} — {r.name}  (score {total})", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            try:
                st.image(r.image_path, use_container_width=True)
            except Exception:
                st.write("(image missing)")
        with c2:
            st.markdown(f"**Name:** {r.name}")
            st.write(f"**Purpose:** {r.purpose}")
            if r.description:
                st.write(f"**Description:** {r.description}")
            if r.tags:
                st.write(f"**Tags:** `{r.tags}`")
            st.caption("Per-hash distances (lower is more similar):")
            st.write({k: (v if v is not None else "-") for k, v in comps.items()})


# ---------------------------
# Streamlit App
# ---------------------------
init_db()

st.set_page_config(page_title=APP_TITLE, page_icon="🔑", layout="wide")
st.title(APP_TITLE)
st.caption("Personal key registry with quick image-based matching (local & private).")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Accept match if combined score ≤",
        min_value=5,
        max_value=100,
        value=DEFAULT_ACCEPT_THRESHOLD,
        help=(
            "Lower score = more similar. This is the sum of distances across multiple perceptual hashes.\n"
            "Start around 35–45; adjust based on your photos and lighting."
        ),
    )
    topk = st.slider("Show top K candidates", 1, 15, DEFAULT_MATCH_TOPK)
    st.divider()
    st.caption("Database path")
    st.code(str(DB_PATH))

    # NOTE: fixed indentation to spaces (no tabs)

# Tabs across the main page
# (Keep this *outside* the sidebar container for clarity.)
tabs = st.tabs(["📷 Scan & Identify", "➕ Add Key", "🗂️ My Keys", "⤴️ Export / Import"])  # noqa: E101

# --- Tab: Scan & Identify ---
with tabs[0]:
    st.subheader("Scan / Upload a key image")
    st.write("Use your webcam or upload a clear photo of one side of the key on a plain background.")

    cam = st.camera_input("Take a photo (webcam)")
    upl = st.file_uploader("…or upload an image", type=["jpg", "jpeg", "png"]) 

    target_img = None
    source_name = None
    if cam is not None:
        target_img = _open_image(cam.getvalue())
        source_name = "camera"
    elif upl is not None:
        target_img = _open_image(upl.getvalue())
        source_name = upl.name

    if target_img is not None:
        st.image(target_img, caption="Scanned image", width=400)
        # ---------- Shape-based matching (primary) ----------
        shape_records = fetch_key_shapes()
        if not shape_records:
            st.info("No shape features yet. Add keys or run the maintenance task in Export/Import to backfill.")
        else:
            try:
                q_feats, q_svg, dbg = extract_shape_features(target_img)
                st.image(dbg, caption="Detected silhouette & contour", width=400)

                # Score against all stored shapes
                scored = []
                for rec in shape_records:
                    cand = ShapeFeatures(
                        width=rec["width"], height=rec["height"],
                        hu=rec["hu"], fourier=rec["fourier"], contour=rec["contour"]
                    )
                    s = match_score(q_feats, cand)  # lower is better
                    scored.append((rec, s))
                scored.sort(key=lambda x: x[1])

                # Controls for shape ranking
                shape_thresh = st.slider("Shape match threshold (lower is better)", 0.0, 3.0, 0.65, 0.01,
                                        help="Tune until your true matches are ≤ this value.")
                topk_shape = st.slider("Top K (shape)", 1, 15, 5)

                top_shape = scored[:topk_shape]
                if top_shape:
                    best_rec, best_score = top_shape[0]
                    verdict = "✅ Likely match" if best_score <= shape_thresh else "⚠️ Uncertain match"
                    st.markdown(f"### {verdict}: **{best_rec['name']}** (shape score {best_score:.3f})")
                    with st.expander("Top candidates (shape)"):
                        for rec, s in top_shape:
                            st.write(f"- #{rec['id']} — **{rec['name']}** (purpose: {rec['purpose']}) — score **{s:.3f}**")
                            if rec.get("svg"):
                                st.markdown(
                                    f"""<div style='background:#fafafa;border:1px solid #eee;padding:8px'>{rec['svg']}</div>""",
                                    unsafe_allow_html=True
                                )
            except Exception as e:
                st.warning(f"Shape matching failed: {e}")

        # ---------- Perceptual-hash matching (fallback/secondary) ----------

        hashes = compute_hashes(target_img)
        matches = find_best_matches(hashes, top_k=topk)

        if not matches:
            st.info("No keys in the database yet. Add a few under **Add Key** first.")
        else:
            best = matches[0]
            best_rec, best_score, best_comps = best
            verdict = "✅ Likely match" if best_score <= threshold else "⚠️ Uncertain match"
            st.markdown(f"### {verdict}: **{best_rec.name}** (score {best_score})")
            st.caption("Scores are lower for better matches. Adjust the threshold in the sidebar if needed.")

            show_match_row(*best)
            if len(matches) > 1:
                st.markdown("#### Other candidates")
                for r, total, comps in matches[1:]:
                    show_match_row(r, total, comps)

            st.divider()
            st.markdown("**Add this scan to your database?**")
            with st.form("save_scanned"):
                name = st.text_input("Name", value=f"Scan from {source_name}")
                purpose = st.text_input("What is this key for? (e.g., Front Door)")
                description = st.text_area("Description (optional)")
                tags = st.text_input("Tags (comma-separated)")
                submitted = st.form_submit_button("Save scan as a new key")
                if submitted:
                    img_path = _save_image(target_img, suggested_name=name or "scan")
                    new_id = insert_key(name, purpose or "(unspecified)", description, tags, img_path, hashes)
                    st.success(f"Saved new key #{new_id}")
                    # Compute and store shape features for this new key
                    try:
                        feats, svg, _dbg = extract_shape_features(Image.open(img_path).convert("RGB"))
                        insert_key_shape(
                            new_id,
                            svg,
                            {
                                "hu": feats.hu,
                                "fourier": feats.fourier,
                                "contour": feats.contour,
                                "width": feats.width,
                                "height": feats.height,
                            },
                        )
                    except Exception as e:
                        st.warning(f"Saved key #{new_id}, but shape features failed: {e}")

# --- Tab: Add Key ---
with tabs[1]:
    st.subheader("Add a new key to your database")
    st.write("Capture or upload a photo and fill in the details.")
    st.markdown("##### Framing guide (align the key inside the box, blade horizontal)")
    use_live = st.toggle("Use live capture with overlay", value=False)
    if use_live and STREAM_WEBRTC_OK:
        ctx = webrtc_streamer(
            key="key-cam",
            video_processor_factory=FramingOverlay,
            media_stream_constraints={"video": True, "audio": False},
        )
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Capture still from live video"):
                with _frame_lock:
                    frame = None if _last_frame is None else _last_frame.copy()
                if frame is not None:
                    add_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    source_name = "live-capture"
                    st.success("Captured!")

                else:
                    st.warning("No frame available yet.")
    else:
        if use_live and not STREAM_WEBRTC_OK:
            st.info("Install live capture deps: pip install streamlit-webrtc av opencv-python-headless")

    add_cam = st.camera_input("Take a photo (webcam)", key="add_cam")
    add_upl = st.file_uploader("…or upload an image", type=["jpg", "jpeg", "png"], key="add_upl")

    add_img = None
    if add_cam is not None:
        add_img = _open_image(add_cam.getvalue())
    elif add_upl is not None:
        add_img = _open_image(add_upl.getvalue())

    with st.form("add_form"):
        name = st.text_input("Name", placeholder="e.g., Silver Kwikset")
        purpose = st.text_input("What is this key for?", placeholder="e.g., Front Door")
        description = st.text_area("Description (optional)")
        tags = st.text_input("Tags (comma-separated)")
        submitted = st.form_submit_button("Add key")

        if submitted:
            if add_img is None:
                st.error("Please capture or upload an image first.")
            elif not name or not purpose:
                st.error("Please provide at least a name and purpose.")
            else:
                hashes = compute_hashes(add_img)
                path = _save_image(add_img, suggested_name=name)
                new_id = insert_key(name, purpose, description, tags, path, hashes)
                st.success(f"Added key #{new_id}")

                # Compute and store shape features for this new key
                try:
                    feats, svg, _dbg = extract_shape_features(Image.open(path).convert("RGB"))
                    insert_key_shape(
                        new_id,
                        svg,
                        {
                            "hu": feats.hu,
                            "fourier": feats.fourier,
                            "contour": feats.contour,
                            "width": feats.width,
                            "height": feats.height,
                        },
                    )
                except Exception as e:
                    st.warning(f"Added key #{new_id}, but shape features failed: {e}")

# --- Tab: My Keys ---
with tabs[2]:
    st.subheader("Your keys")
    df = fetch_keys_df()
    if df.empty:
        st.info("No keys yet. Add some in the **Add Key** tab.")
    else:
        # Search / filter
        q = st.text_input("Search by name / purpose / tags")
        view = df.copy()
        if q:
            ql = q.lower()
            mask = (
                view["name"].str.lower().str.contains(ql) |
                view["purpose"].str.lower().str.contains(ql) |
                view["tags"].fillna("").str.lower().str.contains(ql)
            )
            view = view[mask]

        # Grid of cards
        for _, row in view.iterrows():
            rec = KeyRecord(
                id=int(row["id"]), name=row["name"], purpose=row["purpose"], description=row["description"],
                tags=row["tags"], image_path=row["image_path"], ahash=row.get("ahash"), phash=row.get("phash"),
                dhash=row.get("dhash"), whash=row.get("whash"), created_at=row["created_at"],
            )
            key_card(rec)

        with st.expander("Table view"):
            st.dataframe(df[["id", "name", "purpose", "tags", "created_at"]], use_container_width=True)

# --- Tab: Export / Import ---
with tabs[3]:
    st.subheader("Export / Import")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Export your database**")
        if st.button("Export to CSV"):
            df = fetch_keys_df()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = EXPORT_DIR / f"keys_{ts}.csv"
            df.to_csv(out, index=False)
            st.success(f"Exported to {out}")
        if st.button("Export to JSON"):
            df = fetch_keys_df()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = EXPORT_DIR / f"keys_{ts}.json"
            df.to_json(out, orient="records", indent=2)
            st.success(f"Exported to {out}")

    with c2:
        st.markdown("**Import from CSV/JSON**")
        imp = st.file_uploader("Upload a CSV or JSON previously exported by this app", type=["csv", "json"], key="imp")
        if imp is not None:
            try:
                if imp.type == "application/json" or imp.name.lower().endswith(".json"):
                    loaded = json.load(io.BytesIO(imp.getvalue()))
                    df = pd.DataFrame(loaded)
                else:
                    df = pd.read_csv(imp)
                required_cols = {"name", "purpose", "image_path"}
                if not required_cols.issubset(df.columns):
                    st.error(f"Missing required columns: {sorted(required_cols - set(df.columns))}")
                else:
                    # We only import references; images must be present at the referenced paths.
                    count = 0
                    for _, row in df.iterrows():
                        try:
                            # If hashes are missing, recompute from image path if available
                            ah, ph, dh, wh = row.get("ahash"), row.get("phash"), row.get("dhash"), row.get("whash")
                            img_path = str(row["image_path"]).strip()
                            if (not ah or not ph or not dh or not wh) and Path(img_path).exists():
                                img = Image.open(img_path).convert("RGB")
                                h = compute_hashes(img)
                            else:
                                h = {"ahash": ah, "phash": ph, "dhash": dh, "whash": wh}
                            insert_key(
                                str(row.get("name", "(unnamed)")),
                                str(row.get("purpose", "(unspecified)")),
                                str(row.get("description", "")),
                                str(row.get("tags", "")),
                                img_path,
                                h,
                            )
                            count += 1
                        except Exception as e:
                            st.warning(f"Skipped one row due to error: {e}")
                    st.success(f"Imported {count} records.")
            except Exception as e:
                st.error(f"Import failed: {e}")
    with st.expander("Shape features maintenance", expanded=False):
        if st.button("Compute/refresh shape features for all keys"):
            df_all = fetch_keys_df()
            ok, fail = 0, 0
            for _, r in df_all.iterrows():
                try:
                    p = str(r["image_path"])
                    feats, svg, _ = extract_shape_features(Image.open(p).convert("RGB"))
                    insert_key_shape(
                        int(r["id"]),
                        svg,
                        {
                            "hu": feats.hu,
                            "fourier": feats.fourier,
                            "contour": feats.contour,
                            "width": feats.width,
                            "height": feats.height,
                        },
                    )
                    ok += 1
                except Exception as e:
                    fail += 1
            st.success(f"Shape features updated. OK: {ok}, failed: {fail}")

st.divider()
st.caption("Tip: Use consistent lighting and a plain background when photographing keys to improve matching quality.")

# ---------------------------
# Lightweight self-tests (only run when RUN_KEY_APP_TESTS=1)
# ---------------------------


def _run_self_tests() -> None:
    """Minimal unit-style checks that don't touch the database or filesystem."""
    # Create two simple images in-memory
    img1 = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img2 = Image.new("RGB", (64, 64), color=(129, 128, 128))

    # compute_hashes returns 4 hex strings
    h1 = compute_hashes(img1)
    h2 = compute_hashes(img2)
    assert set(h1.keys()) == {"ahash", "phash", "dhash", "whash"}
    assert all(isinstance(v, str) and len(v) > 0 for v in h1.values())

    # wHash length must correspond to a power-of-two size
    expected_bits = _pow2_le(HASH_SIZE) ** 2            # bits in wHash
    assert len(h1["whash"]) * 4 == expected_bits        # hex chars * 4 = bits

    # Hamming distance should be 0 against itself
    for k in h1:
        assert hamming_distance(h1[k], h1[k]) == 0

    # Combined distance with identical hashes => total 0
    dummy_record = KeyRecord(
        id=0,
        name="test",
        purpose="test",
        description="",
        tags="",
        image_path="",
        ahash=h1["ahash"],
        phash=h1["phash"],
        dhash=h1["dhash"],
        whash=h1["whash"],
        created_at=str(datetime.now()),
    )
    total, comps = combined_distance(h1, dummy_record)
    assert total == 0
    assert all(v == 0 for v in comps.values())

    # Distances between slightly different images should be small but >= 0
    td, _ = combined_distance(h2, dummy_record)
    assert isinstance(td, int) and td >= 0

    print("All self-tests passed ✔️")


if __name__ == "__main__" and os.getenv("RUN_KEY_APP_TESTS") == "1":
    _run_self_tests()