"""
Key Identification App (Streamlit)
----------------------------------
Scan a key with a unified live camera (with overlay), identify which saved key it matches,
and manage a private database of your own keys.

Quick start
===========
1) (Recommended) Create & activate a virtual environment
   python -m venv .venv
   # Windows: .venv\\Scripts\\activate   |   macOS/Linux: source .venv/bin/activate

2) Install dependencies
   pip install streamlit pillow imagehash numpy pandas opencv-python-headless rembg streamlit-webrtc av

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
This app now prioritizes shape-based matching (silhouette and contour descriptors) and
also shows perceptual hash results as a fallback.
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

# --- Optional libs for live camera & CV ---
try:
    import cv2  # pip install opencv-python-headless
except Exception:
    cv2 = None

try:
    from rembg import remove as rembg_remove  # pip install rembg
except Exception:
    rembg_remove = None

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    STREAM_WEBRTC_OK = True
except Exception:
    STREAM_WEBRTC_OK = False

import threading

# ---------------------------
# Constants & Paths
# ---------------------------
APP_TITLE = "Key Identifier"
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "images"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "keys.db"

HASH_SIZE = 12  # a/p/d hashes use this; wHash will use largest power-of-two <= HASH_SIZE
DEFAULT_MATCH_TOPK = 5
DEFAULT_ACCEPT_THRESHOLD = 35  # lower = more similar (sum of Hamming distances)

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

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(SCHEMA_SQL_SHAPES)

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

def delete_key(key_id: int) -> None:
    with get_conn() as conn:
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
        rows = cur.fetchall()
    return [KeyRecord(*row) for row in rows]

def fetch_keys_df() -> pd.DataFrame:
    rows = fetch_keys()
    if not rows:
        return pd.DataFrame(columns=["id", "name", "purpose", "description", "tags", "image_path", "created_at"])
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

# ---------------------------
# Image & Hash utilities
# ---------------------------
def _open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img

def _save_image(img: Image.Image, suggested_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch for ch in suggested_name if ch.isalnum() or ch in ("-", "_"))[:40]
    fname = f"{ts}_{safe or 'key'}.jpg"
    out_path = IMG_DIR / fname
    img.save(out_path, format="JPEG", quality=92)
    return str(out_path)

def _pow2_le(n: int) -> int:
    """Return the largest power of two <= n, with a minimum of 2."""
    if n < 2:
        return 2
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p

def compute_hashes(img: Image.Image, hash_size: int = HASH_SIZE) -> Dict[str, str]:
    # Normalize for more stable hashing
    base = ImageOps.autocontrast(img)
    wsize = _pow2_le(hash_size)  # wHash must be a power of two
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
# Shape pipeline (embedded, previously cv_key.py)
# ---------------------------
@dataclass
class ShapeFeatures:
    width: int
    height: int
    hu: List[float]            # 7 Hu moments (log-scaled)
    fourier: List[float]       # DFT magnitude of contour signature (first 32 coeffs)
    contour: List[Tuple[int, int]]  # Resampled contour (e.g., 256 points)

def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. Run: pip install opencv-python-headless")

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def remove_background(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (bgr_image, binary_mask) where mask==255 roughly equals key.
    Prefers rembg if available; falls back to adaptive thresholding.
    """
    _ensure_cv2()
    bgr = _pil_to_bgr(img)

    if rembg_remove is not None:
        rgba = rembg_remove(img)  # PIL Image RGBA with alpha
        rgba = np.array(rgba.convert("RGBA"))
        alpha = rgba[:, :, 3]
        mask = (alpha > 16).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only largest component (key)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return bgr, np.zeros_like(mask)
    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return bgr, clean

def _rotate_to_canonical(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate so the longest axis is horizontal. Returns (rotated_mask, contour).
    """
    _ensure_cv2()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask, np.empty((0, 1, 2), dtype=np.int32)
    cnt = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    w, h = rect[1]
    if w < h:
        angle = angle + 90.0

    (hgt, wid) = mask.shape[:2]
    M = cv2.getRotationMatrix2D((wid / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(mask, M, (wid, hgt), flags=cv2.INTER_NEAREST, borderValue=0)

    cnts2, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt2 = max(cnts2, key=cv2.contourArea) if cnts2 else cnt
    return rotated, cnt2

def _resample_contour(cnt: np.ndarray, n: int = 256) -> np.ndarray:
    pts = cnt[:, 0, :].astype(np.float32)
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    d = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1] if s[-1] > 0 else 1.0
    targets = np.linspace(0, total, n, endpoint=False)
    resampled = []
    j = 1
    for t in targets:
        while j < len(s) and s[j] < t:
            j += 1
        j = min(j, len(s) - 1)
        t0, t1 = s[j - 1], s[j]
        if t1 - t0 == 0:
            p = pts[j].copy()
        else:
            alpha = (t - t0) / (t1 - t0)
            p = (1 - alpha) * pts[j - 1] + alpha * pts[j]
        resampled.append(p)
    return np.array(resampled, dtype=np.float32)

def _hu_moments(cnt: np.ndarray) -> List[float]:
    m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    out = []
    for h in hu:
        v = 0.0 if h == 0 else -np.sign(h) * np.log10(abs(h))
        out.append(float(v))
    return out

def _fourier_signature(resampled_cnt: np.ndarray, k: int = 32) -> List[float]:
    z = resampled_cnt[:, 0] + 1j * resampled_cnt[:, 1]
    z = z - z.mean()
    denom = np.linalg.norm(z) or 1.0
    z = z / denom
    F = np.fft.fft(z)
    mag = np.abs(F)[1:k + 1]
    mag = mag / (np.linalg.norm(mag) or 1.0)
    return [float(x) for x in mag]

def _contour_to_svg_path(resampled_cnt: np.ndarray) -> str:
    pts = resampled_cnt.astype(int).tolist()
    if not pts:
        return ""
    d = [f"M {pts[0][0]} {pts[0][1]}"]
    for x, y in pts[1:]:
        d.append(f"L {x} {y}")
    d.append("Z")
    return " ".join(d)

def extract_shape_features(img: Image.Image) -> Tuple[ShapeFeatures, Optional[str], Image.Image]:
    """
    Returns (features, svg_text, debug_overlay) where svg_text may be None.
    """
    _ensure_cv2()
    bgr, mask = remove_background(img)
    if mask.sum() == 0:
        raise RuntimeError("No key-like silhouette found. Try a cleaner background.")

    rot_mask, cnt = _rotate_to_canonical(mask)
    if cnt.size == 0:
        raise RuntimeError("Failed to extract contour.")

    resampled = _resample_contour(cnt, n=256)
    hu = _hu_moments(cnt)
    fourier = _fourier_signature(resampled, k=32)

    h, w = rot_mask.shape[:2]
    feats = ShapeFeatures(
        width=int(w), height=int(h), hu=hu, fourier=fourier,
        contour=[(int(x), int(y)) for x, y in resampled]
    )

    # SVG silhouette
    path_d = _contour_to_svg_path(resampled)
    svg = None
    if path_d:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{path_d}" fill="black"/></svg>'

    # Debug overlay
    dbg = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [cnt], -1, (0, 255, 0), 2)
    dbg_pil = _bgr_to_pil(dbg)
    return feats, svg, dbg_pil

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def match_score(query: ShapeFeatures, candidate: ShapeFeatures) -> float:
    """
    Lower is better—combine several signals into one score:
      - cv2.matchShapes on contours
      - Cosine distance of Fourier magnitudes
      - L2 distance on Hu moments
    """
    _ensure_cv2()
    q_cnt = np.array(query.contour, dtype=np.int32).reshape(-1, 1, 2)
    c_cnt = np.array(candidate.contour, dtype=np.int32).reshape(-1, 1, 2)

    ms = cv2.matchShapes(q_cnt, c_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    fq = np.array(query.fourier, dtype=np.float32)
    fc = np.array(candidate.fourier, dtype=np.float32)
    cos = _cosine(fq, fc)
    hu_q = np.array(query.hu, dtype=np.float32)
    hu_c = np.array(candidate.hu, dtype=np.float32)

    w1, w2, w3 = 0.6, 0.3, 0.1
    score = w1 * float(ms) + w2 * (1.0 - cos) + w3 * float(np.linalg.norm(hu_q - hu_c))
    return float(score)

# ---------------------------
# Unified live capture with overlay
# ---------------------------
_frame_lock = threading.Lock()
_last_frame = None

class FramingOverlay(VideoProcessorBase if STREAM_WEBRTC_OK else object):
    def recv(self, frame):  # type: ignore[override]
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
        cv2.line(overlay, (x0, (y0 + y1)//2), (x1, (y0 + y1)//2), (255, 255, 255), 1)
        cv2.line(overlay, ((x0 + x1)//2, y0), ((x0 + x1)//2, y1), (255, 255, 255), 1)

        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        with _frame_lock:
            _last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def capture_image_overlay(key: str, title: str) -> Optional[Image.Image]:
    """
    Shows a live video with a framing overlay and a Capture button.
    Returns a PIL Image once captured, or None otherwise.
    """
    if not STREAM_WEBRTC_OK:
        st.error(
            "Live capture is not available. Install dependencies:\n"
            "pip install streamlit-webrtc av opencv-python-headless"
        )
        return None
    if cv2 is None:
        st.error("OpenCV is required. Install: pip install opencv-python-headless")
        return None

    st.markdown("##### Framing guide (align the key inside the box, blade horizontal)")
    ctx = webrtc_streamer(
        key=f"webrtc_{key}",
        video_processor_factory=FramingOverlay,
        media_stream_constraints={"video": True, "audio": False},
    )

    if st.button(f"📸 Capture {title}", key=f"btn_capture_{key}"):
        with _frame_lock:
            frame = None if _last_frame is None else _last_frame.copy()
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.session_state[f"{key}_img"] = img
            st.success("Captured!")
        else:
            st.warning("No frame available yet. Try again.")

    img = st.session_state.get(f"{key}_img")
    if img is not None:
        st.image(img, caption="Captured", width=400)
    return img

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
                st.rerun()

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
st.caption("Personal key registry with shape-based matching (local & private).")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Accept match if combined score ≤ (perceptual-hash fallback)",
        min_value=5,
        max_value=100,
        value=DEFAULT_ACCEPT_THRESHOLD,
        help="Lower score = more similar. Sum of distances across multiple perceptual hashes.",
    )
    topk = st.slider("Show top K candidates (hash)", 1, 15, DEFAULT_MATCH_TOPK)
    st.divider()
    st.caption("Database path")
    st.code(str(DB_PATH))

tabs = st.tabs(["📷 Scan & Identify", "➕ Add Key", "🗂️ My Keys", "⤴️ Export / Import"])

# --- Tab: Scan & Identify ---
with tabs[0]:
    st.subheader("Scan a key")
    target_img = capture_image_overlay(key="scan", title="scan")
    if target_img is not None:
        # ---------- Shape-based matching (primary) ----------
        shape_records = fetch_key_shapes()
        if not shape_records:
            st.info("No shape features yet. Add keys or run the maintenance task in Export/Import to backfill.")
        else:
            try:
                q_feats, q_svg, dbg = extract_shape_features(target_img)
                st.image(dbg, caption="Detected silhouette & contour", width=420)

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
                                    f"<div style='background:#fafafa;border:1px solid #eee;padding:8px'>{rec['svg']}</div>",
                                    unsafe_allow_html=True
                                )
            except Exception as e:
                st.warning(f"Shape matching failed: {e}")

        # ---------- Perceptual-hash matching (fallback/secondary) ----------
        hashes = compute_hashes(target_img)
        matches = find_best_matches(hashes, top_k=topk)

        if matches:
            best = matches[0]
            best_rec, best_score, best_comps = best
            verdict = "✅ Likely (hash)" if best_score <= threshold else "⚠️ Uncertain (hash)"
            st.markdown(f"### {verdict}: **{best_rec.name}** (hash score {best_score})")
            st.caption("Hash scores are lower for better matches.")
            show_match_row(*best)
            if len(matches) > 1:
                st.markdown("#### Other candidates (hash)")
                for r, total, comps in matches[1:]:
                    show_match_row(r, total, comps)
        else:
            st.info("No keys in the database yet. Add a few under **Add Key** first.")

        st.divider()
        st.markdown("**Add this scan to your database?**")
        with st.form("save_scanned"):
            name = st.text_input("Name", value="Scan")
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
    st.write("Capture a photo and fill in the details.")
    add_img = capture_image_overlay(key="add", title="new key")

    with st.form("add_form"):
        name = st.text_input("Name", placeholder="e.g., Silver Kwikset")
        purpose = st.text_input("What is this key for?", placeholder="e.g., Front Door")
        description = st.text_area("Description (optional)")
        tags = st.text_input("Tags (comma-separated)")
        submitted = st.form_submit_button("Add key")

        if submitted:
            if add_img is None:
                st.error("Please capture an image first.")
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
                    count = 0
                    for _, row in df.iterrows():
                        try:
                            ah, ph, dh, wh = row.get("ahash"), row.get("phash"), row.get("dhash"), row.get("whash")
                            img_path = str(row["image_path"]).strip()
                            if (not ah or not ph or not dh or not wh) and Path(img_path).exists():
                                img = Image.open(img_path).convert("RGB")
                                h = compute_hashes(img)
                            else:
                                h = {"ahash": ah, "phash": ph, "dhash": dh, "whash": wh}
                            new_id = insert_key(
                                str(row.get("name", "(unnamed)")),
                                str(row.get("purpose", "(unspecified)")),
                                str(row.get("description", "")),
                                str(row.get("tags", "")),
                                img_path,
                                h,
                            )
                            # Try to compute & store shape features during import
                            try:
                                if Path(img_path).exists():
                                    feats, svg, _ = extract_shape_features(Image.open(img_path).convert("RGB"))
                                    insert_key_shape(
                                        int(new_id),
                                        svg,
                                        {
                                            "hu": feats.hu,
                                            "fourier": feats.fourier,
                                            "contour": feats.contour,
                                            "width": feats.width,
                                            "height": feats.height,
                                        },
                                    )
                            except Exception:
                                pass
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
                except Exception:
                    fail += 1
            st.success(f"Shape features updated. OK: {ok}, failed: {fail}")

st.divider()
st.caption("Tip: Use consistent lighting and a plain background when photographing keys to improve matching quality.")

# ---------------------------
# Lightweight self-tests (only run when RUN_KEY_APP_TESTS=1)
# ---------------------------
def _run_self_tests() -> None:
    """Minimal unit-style checks that don't touch the database or filesystem."""
    # Hashing tests
    img1 = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img2 = Image.new("RGB", (64, 64), color=(129, 128, 128))
    h1 = compute_hashes(img1)
    h2 = compute_hashes(img2)
    assert set(h1.keys()) == {"ahash", "phash", "dhash", "whash"}
    assert all(isinstance(v, str) and len(v) > 0 for v in h1.values())
    expected_bits = _pow2_le(HASH_SIZE) ** 2
    assert len(h1["whash"]) * 4 == expected_bits
    for k in h1:
        assert hamming_distance(h1[k], h1[k]) == 0
    dummy = KeyRecord(0, "t", "t", "", "", "", h1["ahash"], h1["phash"], h1["dhash"], h1["whash"], str(datetime.now()))
    total, comps = combined_distance(h1, dummy)
    assert total == 0 and all(v == 0 for v in comps.values())
    td, _ = combined_distance(h2, dummy)
    assert isinstance(td, int) and td >= 0

    # Shape pipeline smoke test (skip if cv2 missing)
    if cv2 is not None:
        base = Image.new("L", (256, 128), 0)
        rect = Image.new("L", (160, 60), 255)
        base.paste(rect, (48, 34))
        rect_rgb = base.convert("RGB")
        rect2 = rect_rgb.rotate(3, expand=True)
        f1, _, _ = extract_shape_features(rect_rgb)
        f2, _, _ = extract_shape_features(rect2)
        s = match_score(f1, f2)
        assert s < 0.2, f"Shape score too large for similar objects: {s}"

    print("All self-tests passed ✔️")

if __name__ == "__main__" and os.getenv("RUN_KEY_APP_TESTS") == "1":
    _run_self_tests()