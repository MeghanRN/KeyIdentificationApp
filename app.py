"""
Key Identification App — Fused Outline Matcher
==============================================
- Outline only: Otsu → largest contour → PCA-canonicalize → resample
- Descriptors: radial, curvature, fourier-magnitude
- Plus chamfer on rasterized outline
- One fused score (lower is better). Tunable threshold.

Setup
-----
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install streamlit pillow numpy pandas opencv-python
streamlit run app.py
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
import streamlit as st

from cv_key import (
    extract_outline_features,
    OutlineFeatures,
    fused_outline_distance,
    SIG_LEN,
)

# ---------- constants / paths ----------
APP_TITLE = "Key Identifier (Fused Outline)"
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "images"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "keys.db"

DEFAULT_TOPK = 5
DEFAULT_ACCEPT_THRESHOLD = 0.85  # lower = more similar

for p in [DATA_DIR, IMG_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- DB ----------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS keys (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    purpose TEXT NOT NULL,
    description TEXT,
    tags TEXT,
    image_path TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_SQL_SHAPES = """
CREATE TABLE IF NOT EXISTS key_shapes (
    key_id INTEGER PRIMARY KEY,
    signature TEXT,   -- JSON (dict) {"radial":[...], "curv":[...], "fourier":[...]} OR legacy list
    contour TEXT,     -- JSON list[[x,y],...], canonical
    width INTEGER,
    height INTEGER,
    svg TEXT,
    FOREIGN KEY(key_id) REFERENCES keys(id) ON DELETE CASCADE
);
"""

def _migrate_shapes_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(key_shapes)")
        cols = {r[1] for r in cur.fetchall()}
        adds = []
        if "signature" not in cols: adds.append("ALTER TABLE key_shapes ADD COLUMN signature TEXT;")
        if "contour" not in cols:   adds.append("ALTER TABLE key_shapes ADD COLUMN contour TEXT;")
        if "width" not in cols:     adds.append("ALTER TABLE key_shapes ADD COLUMN width INTEGER;")
        if "height" not in cols:    adds.append("ALTER TABLE key_shapes ADD COLUMN height INTEGER;")
        if "svg" not in cols:       adds.append("ALTER TABLE key_shapes ADD COLUMN svg TEXT;")
        for s in adds: cur.execute(s)
        conn.commit()

@dataclass
class KeyRecord:
    id: int
    name: str
    purpose: str
    description: str
    tags: str
    image_path: str
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

def insert_key(name: str, purpose: str, description: str, tags: str, image_path: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO keys (name, purpose, description, tags, image_path) VALUES (?, ?, ?, ?, ?)",
            (name.strip(), purpose.strip(), (description or "").strip(), (tags or "").strip(), image_path),
        )
        conn.commit()
        return cur.lastrowid

def insert_shape(key_id: int, feats: OutlineFeatures, svg: Optional[str] = None) -> None:
    sig = feats.signature
    # Backward-compatible: ensure dict in JSON
    payload = {
        "signature": json.dumps(sig),
        "contour": json.dumps(feats.contour),
        "width": int(feats.width),
        "height": int(feats.height),
        "svg": svg,
    }
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO key_shapes (key_id, signature, contour, width, height, svg)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key_id, payload["signature"], payload["contour"], payload["width"], payload["height"], payload["svg"]),
        )
        conn.commit()

def fetch_keys() -> List[KeyRecord]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, purpose, description, tags, image_path, created_at FROM keys ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [KeyRecord(*row) for row in rows]

def fetch_keys_df() -> pd.DataFrame:
    rows = fetch_keys()
    if not rows:
        return pd.DataFrame(columns=["id", "name", "purpose", "description", "tags", "image_path", "created_at"])
    return pd.DataFrame([{
        "id": r.id, "name": r.name, "purpose": r.purpose, "description": r.description,
        "tags": r.tags, "image_path": r.image_path, "created_at": r.created_at
    } for r in rows])

def fetch_shapes() -> List[Dict[str, object]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT k.id, k.name, k.purpose, k.image_path,
                   s.signature, s.contour, s.width, s.height, s.svg
            FROM keys k
            JOIN key_shapes s ON s.key_id = k.id
            ORDER BY k.created_at DESC
        """)
        rows = cur.fetchall()
    out = []
    for (kid, name, purpose, image_path, signature, contour, w, h, svg) in rows:
        sig = None
        try:
            sig = json.loads(signature) if signature else None
            # migrate legacy list -> dict
            if isinstance(sig, list):
                sig = {"radial": sig}
        except Exception:
            sig = None
        out.append({
            "id": kid,
            "name": name,
            "purpose": purpose,
            "image_path": image_path,
            "signature": sig,
            "contour": json.loads(contour) if contour else None,
            "width": int(w), "height": int(h),
            "svg": svg,
        })
    return out

def delete_key(key_id: int) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM keys WHERE id=?", (key_id,))
        row = cur.fetchone()
        if row and row[0]:
            try: Path(row[0]).unlink(missing_ok=True)
            except Exception: pass
        cur.execute("DELETE FROM keys WHERE id=?", (key_id,))
        conn.commit()

# ---------- image helpers ----------
def _open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return ImageOps.exif_transpose(img)

def _save_image(img: Image.Image, suggested_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch for ch in suggested_name if ch.isalnum() or ch in ("-", "_"))[:40]
    fname = f"{ts}_{safe or 'key'}.jpg"
    out_path = IMG_DIR / fname
    img.save(out_path, format="JPEG", quality=92)
    return str(out_path)

# ---------- UI helpers ----------
def capture_image_simple(key: str) -> Optional[Image.Image]:
    st.markdown("#### Capture or upload a key photo")
    cam = st.camera_input("Take a photo", key=f"cam_{key}")
    upl = st.file_uploader("…or upload a photo", type=["jpg", "jpeg", "png"], key=f"upl_{key}")

    img = _open_image(cam.getvalue()) if cam is not None else (_open_image(upl.getvalue()) if upl is not None else None)
    if img is not None:
        st.image(img, caption="Preview", width=420)
    return img

def key_card(rec: KeyRecord, shape_row: Optional[Dict[str, object]] = None):
    cols = st.columns([1, 2])
    with cols[0]:
        try:
            st.image(rec.image_path, caption=f"#{rec.id}", use_container_width=True)
        except Exception:
            st.write("(image missing)")
    with cols[1]:
        st.markdown(f"**{rec.name}**")
        st.caption(f"Purpose: {rec.purpose}")
        if rec.description: st.write(rec.description)
        if rec.tags: st.write(f"Tags: `{rec.tags}`")
        st.caption(f"Added: {rec.created_at}")

        with st.expander("Outline descriptors"):
            if not shape_row or not shape_row.get("signature"):
                st.info("No outline data stored yet.")
            else:
                sig = shape_row["signature"]
                if isinstance(sig, dict):
                    if sig.get("radial"):
                        st.write(f"radial: {len(sig['radial'])} samples")
                        try: st.line_chart(pd.Series(sig["radial"]), height=120)
                        except Exception: pass
                    if sig.get("curv"):
                        st.write(f"curvature: {len(sig['curv'])} samples")
                        try: st.line_chart(pd.Series(sig["curv"]), height=120)
                        except Exception: pass
                    if sig.get("fourier"):
                        st.write(f"fourier: {len(sig['fourier'])} coeffs")
                else:
                    # legacy
                    st.write(f"signature: {len(sig)} samples")
                    try: st.line_chart(pd.Series(sig), height=120)
                    except Exception: pass

        del_col, _ = st.columns([1, 4])
        with del_col:
            if st.button("Delete", type="secondary", key=f"del_{rec.id}"):
                delete_key(rec.id)
                st.success(f"Deleted key #{rec.id}")
                st.experimental_rerun()

# ---------- app ----------
init_db()
st.set_page_config(page_title=APP_TITLE, page_icon="🔑", layout="wide")
st.title(APP_TITLE)
st.caption("Fused outline matching (radial + curvature + fourier + chamfer). Lower score = closer outline.")

with st.sidebar:
    st.header("Settings")
    thresh = st.slider(
        "Accept if fused distance ≤",
        min_value=0.10, max_value=2.50, value=DEFAULT_ACCEPT_THRESHOLD, step=0.01,
        help="0.0 ≈ identical. 0.5–1.2 is a typical match band depending on photos."
    )
    topk = st.slider("Top K candidates", 1, 20, DEFAULT_TOPK)
    st.divider()
    st.caption("Database path")
    st.code(str(DB_PATH))

tabs = st.tabs(["📷 Scan & Identify", "➕ Add Key", "🗂️ My Keys", "⤴️ Export / Import"])

# --- Scan & Identify ---
with tabs[0]:
    st.subheader("Scan a key")
    target_img = capture_image_simple(key="scan")

    if target_img is not None:
        try:
            q_feats, dbg = extract_outline_features(target_img)
            st.image(dbg, caption="Detected mask & resampled outline", width=420)
        except Exception as e:
            st.error(f"Failed to extract outline: {e}")
            st.stop()

        shape_rows = fetch_shapes()
        if not shape_rows:
            st.info("No keys with outline data yet. Add a key first.")
        else:
            scored = []
            for rec in shape_rows:
                try:
                    sig = rec.get("signature")
                    if not sig: continue
                    # migrate legacy single list -> dict
                    if isinstance(sig, list):
                        sig = {"radial": sig}
                    cand = OutlineFeatures(
                        width=rec["width"], height=rec["height"],
                        contour=rec["contour"],
                        signature=sig,
                    )
                    d, comps = fused_outline_distance(q_feats, cand)
                    scored.append((rec, d, comps))
                except Exception:
                    continue
            scored.sort(key=lambda x: x[1])

            if scored:
                best_rec, best_d, comps = scored[0]
                verdict = "✅ Likely match" if best_d <= thresh else "⚠️ Uncertain match"
                st.markdown(f"### {verdict}: **{best_rec['name']}** — distance **{best_d:.3f}**")
                with st.expander("Distance breakdown (best)"):
                    st.write({k: round(v, 4) for k, v in comps.items()})
                with st.expander("Top candidates"):
                    for rec, d, _ in scored[:topk]:
                        st.write(f"- #{rec['id']} — **{rec['name']}** (purpose: {rec['purpose']}) — dist **{d:.3f}**")

        st.divider()
        st.markdown("**Add this scan to your database?**")
        with st.form("save_scan"):
            name = st.text_input("Name", value="Scan")
            purpose = st.text_input("What is this key for? (e.g., Front Door)")
            description = st.text_area("Description (optional)")
            tags = st.text_input("Tags (comma-separated)")
            submitted = st.form_submit_button("Save as new key")
            if submitted:
                path = _save_image(target_img, suggested_name=name or "scan")
                new_id = insert_key(name, purpose or "(unspecified)", description, tags, path)
                try:
                    feats, _ = extract_outline_features(Image.open(path).convert("RGB"))
                    insert_shape(new_id, feats, svg=None)
                    st.success(f"Saved key #{new_id} with outline data.")
                except Exception as e:
                    st.warning(f"Saved key #{new_id}, but outline processing failed: {e}")

# --- Add Key ---
with tabs[1]:
    st.subheader("Add a new key")
    add_img = capture_image_simple(key="add")

    with st.form("add_form"):
        name = st.text_input("Name", placeholder="e.g., Silver Key")
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
                path = _save_image(add_img, suggested_name=name)
                new_id = insert_key(name, purpose, description, tags, path)
                try:
                    feats, _ = extract_outline_features(Image.open(path).convert("RGB"))
                    insert_shape(new_id, feats, svg=None)
                    st.success(f"Added key #{new_id} with outline data.")
                except Exception as e:
                    st.warning(f"Added key #{new_id}, but outline processing failed: {e}")

# --- My Keys ---
with tabs[2]:
    st.subheader("Your keys")
    df = fetch_keys_df()
    if df.empty:
        st.info("No keys yet. Add some in the **Add Key** tab.")
    else:
        shapes = fetch_shapes()
        by_id = {r["id"]: r for r in shapes}

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
                id=int(row["id"]), name=row["name"], purpose=row["purpose"],
                description=row["description"], tags=row["tags"],
                image_path=row["image_path"], created_at=row["created_at"]
            )
            key_card(rec, by_id.get(rec.id))

        with st.expander("Table view"):
            st.dataframe(df[["id", "name", "purpose", "tags", "created_at"]], use_container_width=True)

# --- Export / Import ---
with tabs[3]:
    st.subheader("Export / Import")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Export**")
        if st.button("Export KEYS to CSV"):
            df = fetch_keys_df()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = EXPORT_DIR / f"keys_{ts}.csv"
            df.to_csv(out, index=False)
            st.success(f"Exported KEYS to {out}")
        if st.button("Export SHAPES to JSON"):
            rows = fetch_shapes()
            # ensure JSON-serializable
            serial = []
            for r in rows:
                s = dict(r)
                serial.append(s)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = EXPORT_DIR / f"key_shapes_{ts}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2)
            st.success(f"Exported SHAPES to {out}")

    with c2:
        st.markdown("**Import**")
        imp_keys = st.file_uploader("Import KEYS (CSV/JSON exported by this app)", type=["csv", "json"], key="imp_keys")
        if imp_keys is not None:
            try:
                if imp_keys.type == "application/json" or imp_keys.name.lower().endswith(".json"):
                    loaded = json.load(io.BytesIO(imp_keys.getvalue()))
                    df = pd.DataFrame(loaded)
                else:
                    df = pd.read_csv(imp_keys)

                required = {"name", "purpose", "image_path"}
                if not required.issubset(df.columns):
                    st.error(f"Missing required columns: {sorted(required - set(df.columns))}")
                else:
                    count = 0
                    for _, row in df.iterrows():
                        try:
                            new_id = insert_key(
                                str(row.get("name", "(unnamed)")),
                                str(row.get("purpose", "(unspecified)")),
                                str(row.get("description", "")),
                                str(row.get("tags", "")),
                                str(row.get("image_path", "")).strip(),
                            )
                            p = str(row.get("image_path", "")).strip()
                            if p and Path(p).exists():
                                try:
                                    feats, _ = extract_outline_features(Image.open(p).convert("RGB"))
                                    insert_shape(new_id, feats, svg=None)
                                except Exception:
                                    pass
                            count += 1
                        except Exception as e:
                            st.warning(f"Skipped a row: {e}")
                    st.success(f"Imported {count} KEYS.")
            except Exception as e:
                st.error(f"Import KEYS failed: {e}")

        imp_shapes = st.file_uploader("Import SHAPES (JSON exported by this app)", type=["json"], key="imp_shapes")
        if imp_shapes is not None:
            try:
                rows = json.load(io.BytesIO(imp_shapes.getvalue()))
                count = 0
                with get_conn() as conn:
                    for r in rows:
                        try:
                            key_id = int(r["id"])
                            cur = conn.cursor()
                            cur.execute("SELECT 1 FROM keys WHERE id=?", (key_id,))
                            if not cur.fetchone(): continue
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO key_shapes
                                  (key_id, signature, contour, width, height, svg)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    key_id,
                                    json.dumps(r.get("signature")),
                                    json.dumps(r.get("contour")),
                                    int(r.get("width", 0)),
                                    int(r.get("height", 0)),
                                    r.get("svg"),
                                ),
                            )
                            count += 1
                        except Exception as e:
                            st.warning(f"Skipped a row: {e}")
                    conn.commit()
                st.success(f"Imported/updated {count} SHAPES rows.")
            except Exception as e:
                st.error(f"Import SHAPES failed: {e}")

st.divider()
st.caption("Tips: plain background, full key in frame. Rotation doesn’t need to be perfect.")

# ---------- (optional) self-test ----------
def _run_self_tests():
    base = Image.new("L", (320, 160), 0)
    rect = Image.new("L", (170, 60), 255)
    base.paste(rect, (75, 50))
    img1 = base.convert("RGB")
    img2 = img1.rotate(10, expand=True)
    f1, _ = extract_outline_features(img1)
    f2, _ = extract_outline_features(img2)
    d, comps = fused_outline_distance(f1, f2)
    assert d < 1.0, f"distance too large for similar shapes: {d} {comps}"
    print("Self-test passed:", d, comps)

if __name__ == "__main__" and os.getenv("RUN_KEY_APP_TESTS") == "1":
    _run_self_tests()