"""
Key Identification App (Streamlit)
----------------------------------
Unified, robust key identification using advanced computer-vision:
- Background removal (U²-Net via `rembg` if available, else OpenCV fallback)
- Canonicalized silhouette (rotation/scale normalized)
- Elliptic Fourier Descriptors (EFD), Hu (and optional Zernike) moments
- Blade bitting profile (1-D curve, matched with DTW)
- Edge/contour distance (Hausdorff/Chamfer fallback)
- ORB local features as a tie-breaker
- Optional perceptual-hash fallback for legacy compatibility

Quick start
===========
1) (Recommended) Create & activate a virtual environment
   python -m venv .venv
   # Windows: .venv\\Scripts\\activate   |   macOS/Linux: source .venv/bin/activate

2) Install dependencies
   pip install streamlit pillow numpy pandas imagehash opencv-python scikit-image rembg

3) Run the app
   streamlit run app.py

Data & security
===============
- Your data lives locally in ./data next to this script.
- Database: data/keys.db
- Saved images: data/images/
- Exports: data/exports/
- This is a personal tool; keep the folder private.

Notes
=====
- The fused shape pipeline is the primary matcher.
- Hash scores (a/p/d/w) are shown only as an auxiliary view.
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

# --- CV pipeline (new unified descriptors & scoring) ---
#   Save this as `cv_key.py` from the previous message I sent.
from cv_key import (
    extract_and_describe,
    ShapeFeatures,
    match_score,
)

# ---------------------------
# Constants & Paths
# ---------------------------
APP_TITLE = "Key Identifier"
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "images"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "keys.db"

HASH_SIZE = 12         # for a/p/d hashes (wHash auto-adjusted to power of two)
DEFAULT_TOPK = 5
DEFAULT_SHAPE_THRESHOLD = 0.85   # lower = more similar (tune to your images)
DEFAULT_HASH_THRESHOLD = 35      # for legacy hash sum distance (fallback/aux)

# Ensure folders exist
for p in [DATA_DIR, IMG_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Database schema & helpers
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

# Shapes table: keep legacy cols (hu/fourier/contour/width/height/svg)
# and add richer fields: zernike, bitting, orb_kp, orb_desc
SCHEMA_SQL_SHAPES = """
CREATE TABLE IF NOT EXISTS key_shapes (
    key_id INTEGER PRIMARY KEY,
    svg TEXT,
    hu TEXT,
    fourier TEXT,
    contour TEXT,
    width INTEGER,
    height INTEGER,
    zernike TEXT,
    bitting TEXT,
    orb_kp INTEGER,
    orb_desc TEXT,
    FOREIGN KEY(key_id) REFERENCES keys(id) ON DELETE CASCADE
);
"""

def _migrate_shapes_table() -> None:
    # Ensure columns exist if DB was created by an older app version.
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(key_shapes)")
        cols = {r[1] for r in cur.fetchall()}
        alter_stmts = []
        if "zernike" not in cols:
            alter_stmts.append("ALTER TABLE key_shapes ADD COLUMN zernike TEXT;")
        if "bitting" not in cols:
            alter_stmts.append("ALTER TABLE key_shapes ADD COLUMN bitting TEXT;")
        if "orb_kp" not in cols:
            alter_stmts.append("ALTER TABLE key_shapes ADD COLUMN orb_kp INTEGER;")
        if "orb_desc" not in cols:
            alter_stmts.append("ALTER TABLE key_shapes ADD COLUMN orb_desc TEXT;")
        for stmt in alter_stmts:
            cur.execute(stmt)
        conn.commit()

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
    _migrate_shapes_table()

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
                name.strip(),
                purpose.strip(),
                (description or "").strip(),
                (tags or "").strip(),
                image_path,
                hash_dict.get("ahash"),
                hash_dict.get("phash"),
                hash_dict.get("dhash"),
                hash_dict.get("whash"),
            ),
        )
        conn.commit()
        return cur.lastrowid

def insert_key_shape(key_id: int, feats: ShapeFeatures, svg: Optional[str] = None) -> None:
    payload = {
        "hu": json.dumps(feats.hu),
        "fourier": json.dumps(feats.efd),       # store EFD in 'fourier' column for compatibility
        "contour": json.dumps(feats.contour),
        "width": int(feats.width),
        "height": int(feats.height),
        "zernike": json.dumps(feats.zernike) if feats.zernike is not None else None,
        "bitting": json.dumps(feats.bitting),
        "orb_kp": int(feats.orb_kp),
        "orb_desc": json.dumps(feats.orb_desc.tolist()) if feats.orb_desc is not None else None,
        "svg": svg,
    }
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO key_shapes
              (key_id, svg, hu, fourier, contour, width, height, zernike, bitting, orb_kp, orb_desc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key_id,
                payload["svg"],
                payload["hu"],
                payload["fourier"],
                payload["contour"],
                payload["width"],
                payload["height"],
                payload["zernike"],
                payload["bitting"],
                payload["orb_kp"],
                payload["orb_desc"],
            ),
        )
        conn.commit()

def fetch_keys() -> List[KeyRecord]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, purpose, description, tags, image_path, ahash, phash, dhash, whash, created_at "
            "FROM keys ORDER BY created_at DESC"
        )
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
                   s.svg, s.hu, s.fourier, s.contour, s.width, s.height,
                   s.zernike, s.bitting, s.orb_kp, s.orb_desc
            FROM keys k
            JOIN key_shapes s ON s.key_id = k.id
            ORDER BY k.created_at DESC
        """)
        rows = cur.fetchall()
    out = []
    for row in rows:
        (kid, name, purpose, image_path,
         svg, hu, fourier, contour, w, h,
         zernike, bitting, orb_kp, orb_desc) = row
        out.append({
            "id": kid, "name": name, "purpose": purpose, "image_path": image_path,
            "svg": svg,
            "hu": json.loads(hu) if hu else None,
            "fourier": json.loads(fourier) if fourier else None,
            "contour": json.loads(contour) if contour else None,
            "width": int(w), "height": int(h),
            "zernike": json.loads(zernike) if zernike else None,
            "bitting": json.loads(bitting) if bitting else None,
            "orb_kp": int(orb_kp) if orb_kp is not None else 0,
            "orb_desc": np.array(json.loads(orb_desc), dtype=np.uint8) if orb_desc else None,
        })
    return out

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

# ---------------------------
# Image utilities & hashes (aux)
# ---------------------------
def _open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return ImageOps.exif_transpose(img)

def _save_image(img: Image.Image, suggested_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch for ch in suggested_name if ch.isalnum() or ch in ("-", "_"))[:40]
    fname = f"{ts}_{safe or 'key'}.jpg"
    out_path = IMG_DIR / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=92)
    return str(out_path)

def _pow2_le(n: int) -> int:
    if n < 2:
        return 2
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p

def compute_hashes(img: Image.Image, hash_size: int = HASH_SIZE) -> Dict[str, str]:
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

def combined_hash_distance(q_hashes: Dict[str, str], r: KeyRecord) -> Tuple[int, Dict[str, Optional[int]]]:
    comps = {
        "ahash": hamming_distance(q_hashes.get("ahash"), r.ahash),
        "phash": hamming_distance(q_hashes.get("phash"), r.phash),
        "dhash": hamming_distance(q_hashes.get("dhash"), r.dhash),
        "whash": hamming_distance(q_hashes.get("whash"), r.whash),
    }
    valid = [v for v in comps.values() if isinstance(v, int)]
    total = int(sum(valid)) if valid else 10**9
    return total, comps

# ---------------------------
# Unified capture (no overlay)
# ---------------------------
def capture_image_simple(key: str) -> Optional[Image.Image]:
    st.markdown("#### Capture or upload a key photo")
    cam = st.camera_input("Take a photo", key=f"cam_{key}")
    upl = st.file_uploader("…or upload a photo", type=["jpg", "jpeg", "png"], key=f"upl_{key}")

    img = None
    if cam is not None:
        img = _open_image(cam.getvalue())
    elif upl is not None:
        img = _open_image(upl.getvalue())

    if img is not None:
        st.image(img, caption="Preview", width=420)

    return img

# ---------------------------
# UI helpers
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

def show_hash_row(r: KeyRecord, total: int, comps: Dict[str, Optional[int]]):
    with st.expander(f"#{r.id} — {r.name}  (hash score {total})", expanded=False):
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
# App
# ---------------------------
init_db()
st.set_page_config(page_title=APP_TITLE, page_icon="🔑", layout="wide")
st.title(APP_TITLE)
st.caption("Fused shape-based matching with optional perceptual-hash view (all data stays local).")

with st.sidebar:
    st.header("Matching Settings")
    shape_thresh = st.slider(
        "Accept if fused shape score ≤",
        min_value=0.10, max_value=2.50, value=DEFAULT_SHAPE_THRESHOLD, step=0.01,
        help="Lower is better. Tune based on your photos; 0.6–1.2 often works well."
    )
    topk_shape = st.slider("Top K (shape)", 1, 20, DEFAULT_TOPK)

    st.divider()
    st.header("Aux (hash) Settings")
    hash_thresh = st.slider(
        "Accept if combined hash distance ≤",
        min_value=5, max_value=100, value=DEFAULT_HASH_THRESHOLD
    )
    topk_hash = st.slider("Top K (hash)", 1, 20, 5)

    st.divider()
    st.caption("Database path")
    st.code(str(DB_PATH))

tabs = st.tabs(["📷 Scan & Identify", "➕ Add Key", "🗂️ My Keys", "⤴️ Export / Import"])

# --- Tab: Scan & Identify ---
with tabs[0]:
    st.subheader("Scan a key")
    target_img = capture_image_simple(key="scan")

    if target_img is not None:
        # Compute features & debug viz
        try:
            q_feats, q_dbg = extract_and_describe(target_img)
            st.image(q_dbg, caption="Canonical silhouette & contour", width=420)
        except Exception as e:
            st.error(f"Failed to extract features: {e}")
            st.stop()

        # Match against all stored shapes
        shape_records = fetch_key_shapes()
        if not shape_records:
            st.info("No shape features yet. Add keys or run backfill under Export/Import.")
        else:
            scored = []
            for rec in shape_records:
                if not (rec.get("contour") and rec.get("fourier") and rec.get("hu")):
                    continue
                cand = ShapeFeatures(
                    width=int(rec["width"]), height=int(rec["height"]),
                    contour=rec["contour"],
                    # If DB has these extra fields, use them; else graceful defaults
                    efd=rec["fourier"],
                    hu=rec["hu"],
                    zernike=rec.get("zernike"),
                    bitting=rec.get("bitting") or [0.0]*160,
                    orb_kp=int(rec.get("orb_kp", 0)),
                    orb_desc=rec.get("orb_desc"),
                    mask_bbox=(0, 0, int(rec["width"]), int(rec["height"]))
                )
                s = match_score(q_feats, cand)  # lower is better
                scored.append((rec, s))
            scored.sort(key=lambda x: x[1])

            if scored:
                top = scored[:topk_shape]
                best_rec, best_score = top[0]
                verdict = "✅ Likely match" if best_score <= shape_thresh else "⚠️ Uncertain match"
                st.markdown(f"### {verdict}: **{best_rec['name']}** — fused shape score **{best_score:.3f}**")
                with st.expander("Top candidates (shape)"):
                    for rec, s in top:
                        st.write(f"- #{rec['id']} — **{rec['name']}** (purpose: {rec['purpose']}) — score **{s:.3f}**")
                        if rec.get("svg"):
                            st.markdown(
                                f"<div style='background:#fafafa;border:1px solid #eee;padding:8px'>{rec['svg']}</div>",
                                unsafe_allow_html=True
                            )
            else:
                st.info("No comparable shape records yet. Add more keys.")

        st.divider()

        # Auxiliary perceptual-hash view (not used for primary verdict)
        hashes = compute_hashes(target_img)
        hash_candidates = []
        for r in fetch_keys():
            total, comps = combined_hash_distance(hashes, r)
            hash_candidates.append((r, total, comps))
        hash_candidates.sort(key=lambda x: x[1])

        if hash_candidates:
            best_r, best_total, best_comps = hash_candidates[0]
            verdict = "✅ Likely (hash)" if best_total <= hash_thresh else "⚠️ Uncertain (hash)"
            st.markdown(f"#### {verdict}: **{best_r.name}** — combined hash distance {best_total}")
            show_hash_row(best_r, best_total, best_comps)
            if len(hash_candidates) > 1:
                with st.expander("Other candidates (hash)"):
                    for r, total, comps in hash_candidates[1:topk_hash]:
                        show_hash_row(r, total, comps)

        st.divider()
        st.markdown("**Add this scan to your database?**")
        with st.form("save_scanned"):
            name = st.text_input("Name", value="Scan")
            purpose = st.text_input("What is this key for? (e.g., Front Door)")
            description = st.text_area("Description (optional)")
            tags = st.text_input("Tags (comma-separated)")
            submitted = st.form_submit_button("Save scan as a new key")
            if submitted:
                path = _save_image(target_img, suggested_name=name or "scan")
                new_id = insert_key(name, purpose or "(unspecified)", description, tags, path, hashes)
                try:
                    feats, _dbg = extract_and_describe(Image.open(path).convert("RGB"))
                    insert_key_shape(new_id, feats, svg=None)
                    st.success(f"Saved new key #{new_id} with shape features.")
                except Exception as e:
                    st.warning(f"Saved key #{new_id}, but shape features failed: {e}")

# --- Tab: Add Key ---
with tabs[1]:
    st.subheader("Add a new key to your database")
    st.write("Capture a photo and fill in the details.")
    add_img = capture_image_simple(key="add")

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
                try:
                    feats, _dbg = extract_and_describe(Image.open(path).convert("RGB"))
                    insert_key_shape(new_id, feats, svg=None)
                    st.success(f"Added key #{new_id} with shape features.")
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
                id=int(row["id"]),
                name=row["name"],
                purpose=row["purpose"],
                description=row["description"],
                tags=row["tags"],
                image_path=row["image_path"],
                ahash=row.get("ahash"),
                phash=row.get("phash"),
                dhash=row.get("dhash"),
                whash=row.get("whash"),
                created_at=row["created_at"],
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
                            try:
                                if Path(img_path).exists():
                                    feats, _ = extract_and_describe(Image.open(img_path).convert("RGB"))
                                    insert_key_shape(new_id, feats, svg=None)
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
                    feats, _ = extract_and_describe(Image.open(p).convert("RGB"))
                    insert_key_shape(int(r["id"]), feats, svg=None)
                    ok += 1
                except Exception:
                    fail += 1
            st.success(f"Shape features updated. OK: {ok}, failed: {fail}")

st.divider()
st.caption("Tip: Place the key on a plain, contrasting background. Keep the blade roughly horizontal.")

# ---------------------------
# Lightweight self-tests (RUN_KEY_APP_TESTS=1)
# ---------------------------
def _run_self_tests() -> None:
    # Hash pipeline quick checks
    img1 = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img2 = Image.new("RGB", (64, 64), color=(129, 128, 128))
    h1 = compute_hashes(img1)
    h2 = compute_hashes(img2)
    assert set(h1.keys()) == {"ahash", "phash", "dhash", "whash"}
    expected_bits = _pow2_le(HASH_SIZE) ** 2
    assert len(h1["whash"]) * 4 == expected_bits
    for k in h1:
        assert hamming_distance(h1[k], h1[k]) == 0
    t0, _ = combined_hash_distance(h1, KeyRecord(0, "t","t","","","", h1["ahash"],h1["phash"],h1["dhash"],h1["whash"], str(datetime.now())))
    assert t0 == 0

    # Shape pipeline smoke test (two near rectangles)
    base = Image.new("L", (256, 128), 0)
    rect = Image.new("L", (160, 60), 255)
    base.paste(rect, (48, 34))
    rect_rgb = base.convert("RGB")
    rect2 = rect_rgb.rotate(3, expand=True)
    f1, _ = extract_and_describe(rect_rgb)
    f2, _ = extract_and_describe(rect2)
    s = match_score(f1, f2)
    assert s < 0.4, f"shape score too large for similar objects: {s}"
    print("All self-tests passed ✔️")

if __name__ == "__main__" and os.getenv("RUN_KEY_APP_TESTS") == "1":
    _run_self_tests()