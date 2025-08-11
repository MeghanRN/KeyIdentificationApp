# Key Identification App

Scan a key (camera or upload) and identify which **saved** key it matches. Create and manage a private database of your own keys. All data stays local.

This version uses a **single, unified identification pipeline** that’s robust to background, rotation, and small pose/scale changes—no need to take “the exact same photo” anymore.

---

# Quick start

1. (Recommended) Create & activate a virtual environment

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies
   (you can paste this to a `requirements.txt` or install directly)

   ```bash
   pip install streamlit pillow numpy pandas imagehash opencv-python scikit-image rembg
   ```

   * `rembg` is optional but strongly recommended for best background removal.
   * If your platform needs it, you can use `opencv-python-headless` instead of `opencv-python`.

3. Run the app

   ```bash
   streamlit run app.py
   ```

---

# What’s new (tech highlights)

* **Unified pipeline** for every image:

  * **Background removal** using **U²-Net** via `rembg` (falls back to robust OpenCV thresholding if `rembg` isn’t available).
  * **Canonicalization**: auto-rotates the silhouette with PCA so the blade is horizontal, flips so the blade points right, and normalizes scale.
  * **Descriptors (features) per key**:

    * **Elliptic Fourier Descriptors (EFD)** of the closed contour (rotation/scale invariant global shape).
    * **Hu** + **Zernike** moments on the normalized mask (global shape invariants).
    * **Bitting profile**: a 1-D curve of tooth depth along the blade, matched with **DTW** (Dynamic Time Warping).
    * **Edge/contour distance**: **Hausdorff** (or Chamfer fallback) between silhouettes.
    * **Local features**: **ORB** descriptors on edges as a tie-breaker.
  * **Two-stage retrieval**:

    1. Coarse shortlist with global shape (EFD + Hu + Zernike).
    2. Re-rank with Hausdorff/Chamfer + DTW (bitting) + ORB inlier ratio.
  * **Score fusion**: components are combined with tuned weights into one final score (lower is better).
* **Database schema** extended to store **Zernike**, **bitting**, and **ORB** so you get the **full fused score immediately** for past and new keys.
* **One capture flow** (camera or upload) — no more parallel code paths.

---

# How it works

1. **Segmentation**
   We remove the background with U²-Net (`rembg`) if available; otherwise, an OpenCV fallback:

   * Gray → CLAHE → adaptive threshold → morphology → keep largest component.

2. **Canonicalization**

   * PCA finds the major axis → rotate to horizontal.
   * Flip if needed so the blade faces right.
   * Normalize to a standard height (scale invariance).

3. **Feature extraction**

   * **EFD**: 24 harmonics (configurable) → rotation/scale/translation invariant contour signature.
   * **Hu moments** (on binary mask, log-scaled).
   * **Zernike moments** (requires `scikit-image`) — optional, but we store when available.
   * **Bitting profile**: sample tooth depth along the blade (right \~60% of mask) into a normalized 1-D vector.
   * **ORB**: detect/describe edge keypoints for local similarity.

4. **Matching & score**

   * Compute distances:

     * `efd`: L2(EFD vectors)
     * `hu`: L2(Hu vectors)
     * `zern`: L2(Zernike vectors) if both keys have it
     * `haus`: Hausdorff (or Chamfer-like fallback) between silhouettes
     * `dtw`: DTW distance between bitting curves
     * `orb`: 1 − inlier\_ratio (Lowe ratio test on ORB matches)
   * Fuse with weights (defaults):

     ```
     0.30*efd + 0.12*hu + 0.08*zern + 0.30*haus + 0.15*dtw + 0.05*orb
     ```
   * Lower is better. You can tune the **accept threshold** in the sidebar.

5. **Auxiliary view (legacy)**
   We still compute perceptual hashes (a/p/d/w) as a secondary/diagnostic view; the **primary verdict** is the fused shape score.

---

# Project structure

```
.
├── app.py                # Streamlit app (UI + DB)
├── cv_key.py             # CV pipeline: segmentation, features, scoring
├── data/
│   ├── images/           # saved key images
│   ├── exports/          # CSV/JSON exports
│   └── keys.db           # SQLite database
└── requirements.txt      # dependencies (optional)
```

---

# Database

We use SQLite (`data/keys.db`) with two tables:

## `keys`

| column                  | type | note                        |
| ----------------------- | ---- | --------------------------- |
| id (PK)                 | int  | autoincrement               |
| name                    | text | user-provided               |
| purpose                 | text | user-provided               |
| description             | text | optional                    |
| tags                    | text | optional                    |
| image\_path             | text | local path to saved JPEG    |
| ahash/phash/dhash/whash | text | perceptual hash hex strings |
| created\_at             | text | default `CURRENT_TIMESTAMP` |

## `key_shapes`

| column       | type | note                                                    |
| ------------ | ---- | ------------------------------------------------------- |
| key\_id PK   | int  | 1:1 with `keys.id`                                      |
| svg          | text | optional (silhouette rendering)                         |
| hu           | text | JSON list\[float]                                       |
| fourier      | text | JSON list\[float] (EFD vector)                          |
| contour      | text | JSON list\[\[x,y], ...] (canonicalized)                 |
| width/height | int  | canonical mask size                                     |
| zernike      | text | JSON list\[float] (optional; requires `scikit-image`)   |
| bitting      | text | JSON list\[float] 1-D depth curve                       |
| orb\_kp      | int  | number of ORB keypoints                                 |
| orb\_desc    | text | JSON list of ORB descriptors (uint8 rows); can be large |

> On upgrade, the app will **migrate** the `key_shapes` table to add missing columns.

---

# Usage

## 1) Add a key

* Go to **“➕ Add Key”**.
* Capture or upload a photo (one unified control).
* Fill in **Name** and **Purpose** (required).
* Submit to save:

  * The original image is saved in `data/images/`.
  * The app computes and stores **all** shape features (EFD, Hu, Zernike, bitting, ORB).

## 2) Identify a key

* Go to **“📷 Scan & Identify”**.
* Capture or upload a photo of a **single** key on a plain background.
* The app shows:

  * Canonicalized silhouette preview.
  * **Best match** and **score** (lower is better).
  * **Top candidates** and a **breakdown** of component distances for the best candidate.
  * (Optional) Perceptual-hash comparison as a secondary view.

## 3) Manage your keys

* **“🗂️ My Keys”** shows your saved keys.
* Expand **Shape features** to see stored descriptors; bitting curve is plotted if available.
* Delete a key from this screen (removes the image and DB row).

## 4) Export / Import

* Export **KEYS** to CSV/JSON and **SHAPES** to JSON.
* Import **KEYS** CSV/JSON (recomputes shape features for any rows that have accessible images).
* Import **SHAPES** JSON to backfill features (requires corresponding key rows to exist).
* **Maintenance**: “Compute/refresh ALL” recomputes *full* shape features for every key in the DB.

---

# Image capture tips

* Use a **plain, contrasting background** (paper, desk mat).
* Keep the **blade roughly horizontal**. The app auto-rotates and flips, but good framing helps.
* Avoid heavy glare; diffuse light is best.
* Keep the entire key in frame; crop out any second object.

---

# Settings & thresholds

In the sidebar:

* **Fused shape accept threshold** (default `0.85`):
  Lower is stricter. Typical good range is **0.6–1.2** depending on your photos.

* **Top K (shape)**: how many candidates to show.

* **Aux hash threshold** (default `35`) & **Top K (hash)**: diagnostic/secondary view.

> Tuning tip: Add 5–10 keys first, test a few scans, then adjust the fused threshold until true matches land **below** it and non-matches stay **above** it.

---

# Troubleshooting

* **“ImportError: cannot import name `extract_and_describe` from `cv_key`”**
  Ensure you copied the **new** `cv_key.py` (the one that defines `extract_and_describe`).
  If you can’t update immediately, there’s a compatibility shim you can drop into `app.py` to wrap the old `extract_shape_features`.

* **`rembg` not installed or fails**
  The app automatically falls back to an OpenCV thresholding pipeline. For best results, `pip install rembg`.

* **OpenCV conflicting builds**
  Some environments prefer `opencv-python-headless`. Uninstall `opencv-python` and install `opencv-python-headless`.

* **Zernike not computed**
  Requires `scikit-image`. Install it and recompute features from **Export/Import → “Compute/refresh ALL”**.

* **Score seems too high/low**
  Adjust the **fused threshold** in the sidebar. Make sure training photos are clean, single-key, and reasonably close.

* **Switching cameras**
  `st.camera_input` uses the **browser’s** selected camera. Change it in your browser’s site settings (Chrome: `chrome://settings/content/camera`, Safari: Settings → Websites → Camera, etc.), then reload.

---

# Privacy & data

* Everything is local to the `data/` folder:

  * DB at `data/keys.db`
  * Images under `data/images/`
  * Exports under `data/exports/`
* Keep this folder private. The app is for **personal** record-keeping only.

---

# Development & tests

* Minimal self-tests exist and can be run with:

  ```bash
  # Windows
  set RUN_KEY_APP_TESTS=1 && streamlit run app.py
  # macOS/Linux
  RUN_KEY_APP_TESTS=1 streamlit run app.py
  ```

  They check basic hashing and a shape smoke test.

* Recommended Python versions: 3.9–3.11 (works on newer too, but CV stacks vary by platform).

---

# Roadmap (optional ideas)

* **Small learned re-ranker** (Siamese / triplet) on your private dataset for even tighter ranking.
* **SVG export** of canonical silhouette (currently stored as contour; can render on demand).
* **Mobile PWA wrapper** for a more camera-native feel.

---

# FAQ

**Q: Can it identify the brand/lock model or decode bitting numbers?**
A: No. It performs **visual similarity matching** against *your* saved keys. It doesn’t decode keyways/bitting specs.

**Q: Do I have to use `rembg`?**
A: No, but you’ll get more robust segmentation with it.

**Q: Can I store thousands of keys?**
A: Technically yes; precomputing and caching features is designed for that. For very large sets, consider periodic exports and SSD storage.

---

# License & acknowledgments

* Built with **Streamlit**, **OpenCV**, **scikit-image**, **rembg (U²-Net)**, and **Pillow**.
* You own your data; keep it private and comply with local laws.