# Key Identification App — Fused Outline Matcher

A dead-simple, reliable way to identify **your own keys** by comparing the **outline** of a new photo with keys you’ve saved before. No heavy ML. No cloud. Just OpenCV + NumPy + Streamlit — local and fast.
---

## Features

* **One capture flow** (camera or upload) with a single control.
* **Outline tracing**: Otsu threshold → largest contour → resample → PCA canonicalization.
* **Tiny descriptors** (all 1-D, lightweight):

  * **Radial** signature (centroid → boundary distances).
  * **Curvature** signature (turning angle along the outline).
  * **Fourier magnitude** of the complex outline (shift-invariant).
* **Chamfer distance** on rasterized outlines as a sanity check.
* **Single fused score** (lower = better) with sensible defaults.
* Local **SQLite** database; JSON export/import for portability.

---

## Quick Start

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2) Install dependencies
pip install streamlit pillow numpy pandas opencv-python

# 3) Run the app
streamlit run app.py
```

> On servers/containers, you can use `opencv-python-headless` instead of `opencv-python`.

---

## How It Works

1. **Segment** the key with Otsu thresholding and morphology; keep the **largest** contour.
2. **Resample** the contour to a fixed length (default `256` points).
3. **Canonicalize**: rotate to the principal axis (PCA) and scale to a standard height.
4. **Describe** the outline with:

   * `radial` — distance from centroid along the boundary (zero-mean, unit-norm).
   * `curv` — smoothed turning angle between boundary points (zero-mean, unit-norm).
   * `fourier` — first 32 low-frequency magnitude coefficients of the complex outline (normalized).
5. **Compare** a query to every saved key using:

   * Circular correlation for `radial` and `curv` (also checks mirrored order).
   * L2 distance for `fourier`.
   * **Chamfer** on rasterized outlines (256×256) as a tie-breaker.
6. **Fuse** distances with weights:

   ```
   score = 0.45*radial + 0.25*curv + 0.20*fourier + 0.10*chamfer
   ```

   **Lower is better.** Identical outlines approach 0.

---

## Using the App

### Add keys (build your database)

1. Go to **“➕ Add Key”**.
2. Capture/upload a clear photo of **one key** on a plain background.
3. Enter **Name** and **Purpose**, then **Add key**.
4. The app saves the image and stores the **outline descriptor**.

### Identify a key

1. Go to **“📷 Scan & Identify”**.
2. Capture/upload a new photo of a key.
3. The app shows:

   * The detected mask + resampled outline (debug image).
   * The **best match** with its fused distance.
   * Top candidates and a breakdown of component distances.

### Manage / Export / Import

* **“🗂️ My Keys”** lists your keys; expand *Outline descriptors* to see stored signals.
* **Export** keys (CSV) and shapes (JSON).
* **Import** keys (CSV/JSON) and shapes (JSON).

  * When importing keys, if `image_path` exists, the app computes & stores outline descriptors automatically.

---

## Thresholds & Tuning

* Sidebar default **accept threshold** = **0.85**.

  * Typical matches land between **0.5–1.2**, depending on photos.
  * Lower = stricter (fewer false positives, more false negatives).
* **Top K** controls how many candidates to display.

If your images are very consistent, try **0.6–0.9**. If they vary a lot, you may prefer **1.0–1.4**.

---

## Image Tips (matters a lot)

* Use a **plain, contrasting background** (white paper works great).
* Keep the **entire key** in frame; avoid hands/fingers overlapping the outline.
* Diffuse lighting (avoid harsh glare).
* Orientation doesn’t need to be perfect — the app auto-aligns.

---

## Project Structure

```
.
├── app.py        # Streamlit UI + SQLite wiring + import/export
├── cv_key.py     # Outline extraction, descriptors, and fused distance
└── data/
    ├── images/   # Saved images
    ├── exports/  # CSV/JSON exports
    └── keys.db   # SQLite database
```

---

## Database Schema

### `keys`

| column      | type | notes                       |
| ----------- | ---- | --------------------------- |
| id (PK)     | int  | autoincrement               |
| name        | text | user-provided               |
| purpose     | text | user-provided               |
| description | text | optional                    |
| tags        | text | optional                    |
| image\_path | text | path to saved JPEG          |
| created\_at | text | default `CURRENT_TIMESTAMP` |

### `key_shapes`

| column    | type | notes                                                        |
| --------- | ---- | ------------------------------------------------------------ |
| key\_id   | int  | 1:1 with `keys.id`                                           |
| signature | text | JSON dict: `{"radial":[...], "curv":[...], "fourier":[...]}` |
| contour   | text | JSON list of `[x, y]` points (canonicalized)                 |
| width     | int  | original mask width                                          |
| height    | int  | original mask height                                         |
| svg       | text | reserved; not used in this build                             |

> **Legacy import**: if a prior export has `signature` as a **list**, it’s treated as `{"radial": list}` automatically.

---

## Import/Export Formats

### Exported SHAPES JSON (example)

```json
[
  {
    "id": 12,
    "name": "Black house key",
    "purpose": "Front door",
    "image_path": "data/images/20250811_143210_black.jpg",
    "signature": {
      "radial": [ ... 256 floats ... ],
      "curv":   [ ... 256 floats ... ],
      "fourier":[ ... 32 floats  ... ]
    },
    "contour": [[x, y], [x, y], ...],
    "width": 1280,
    "height": 720,
    "svg": null
  }
]
```

### Imported KEYS CSV

Must include at least: `name`, `purpose`, `image_path`.
If `image_path` exists locally, the app computes the outline descriptors on import.

---

## Troubleshooting

* **“No key outline found”**
  The background is too busy or the key is partly out of frame. Use plain paper and include the full key.

* **Wrong matches**

  * Lower the accept threshold (stricter).
  * Add 2–3 representative photos per key (different lighting/backgrounds).
  * Ensure keys are not overlapping other objects.

* **OpenCV build errors on servers**
  Try:

  ```bash
  pip uninstall opencv-python
  pip install opencv-python-headless
  ```

* **Switching cameras**
  `st.camera_input` uses your **browser’s** selected camera. Change it in site permissions (e.g., Chrome: Settings → Privacy & Security → Site Settings → Camera), then reload.

* **Module changes not showing**
  Stop the app (Ctrl+C) and run `streamlit run app.py` again to clear caches.

---

## Advanced (optional)

Edit constants in `cv_key.py`:

```python
SIG_LEN   = 256   # samples along outline (try 512 for tighter matching)
FOURIER_K = 32    # low-frequency magnitude count (16..48 is typical)
CANVAS    = 256   # raster size for chamfer distance
```

Change fusion weights inside `fused_outline_distance`:

```python
w_rad, w_curv, w_four, w_ch = 0.45, 0.25, 0.20, 0.10
```

> Heavier `radial` emphasizes global silhouette; heavier `curv` emphasizes tooth edges/shoulders.

---

## Self-Test

You can run a quick smoke test:

```bash
# Windows
set RUN_KEY_APP_TESTS=1 && streamlit run app.py
# macOS/Linux
RUN_KEY_APP_TESTS=1 streamlit run app.py
```

The test creates two similar rectangles and ensures the fused distance is small.

---

## Privacy

All data lives under `./data/` on your machine:

* DB: `data/keys.db`
* Images: `data/images/`
* Exports: `data/exports/`

No images or descriptors are uploaded anywhere.

---

## Limitations & Notes

* This identifies keys **by visual outline** against **your** saved set.
  It does **not** decode bitting depths, keyway types, or manufacturer models.
* Very similar blanks may require tighter photos and a slightly lower threshold.
* Mirroring is handled automatically (we check both directions).

---

## Roadmap (nice-to-haves)

* In-app **outline overlay** guide for consistent framing.
* Batch re-compute button for shapes (maintenance).
* Optional **mobile PWA** wrapper for a more camera-native feel.

---

## License

You own your data. Use responsibly and comply with local laws.
Built with Streamlit, NumPy, OpenCV, Pillow, and SQLite.
