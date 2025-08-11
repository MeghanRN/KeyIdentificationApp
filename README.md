# Key Identification App — README

A local-first Streamlit app to **scan/upload a key photo** and **identify** which of your saved keys it matches—plus a simple database to **add, search, and manage** your own keys. No cloud, no accounts; everything stays on your machine.

---

## What this app does (and doesn’t)

* **Does:** compares a new photo of a key against *your* saved key photos using **perceptual image hashing**; shows the best matches with a numeric similarity score; lets you add, delete, search, export, and import your key records.
* **Doesn’t:** decode bitting, cut keys, or identify manufacturers/lock models. Results are **similarity-based**, not forensic identification.

---

## Quick start

### Requirements

* **Python** 3.9+
* macOS, Windows, or Linux
* Camera access (optional, for webcam capture)

### Install

```bash
# optional: create a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install deps
pip install -r requirements.txt
# or, install directly:
pip install streamlit pillow imagehash numpy pandas
```

**requirements.txt** (optional)

```
streamlit
pillow
ImageHash
numpy
pandas
```

> `ImageHash` brings in PyWavelets for wavelet hashing automatically.

### Run

```bash
streamlit run app.py
```

* Your browser will open to the app UI.
* If webcam capture is blocked, allow camera access in your browser.

---

## How to use the app

### Tabs overview

1. **📷 Scan & Identify**

   * Capture a photo via **webcam** or **upload** a JPG/PNG.
   * The app computes four perceptual hashes and finds the **Top-K** closest matches from your database.
   * You’ll see a **verdict**:

     * **✅ Likely match** if the combined score ≤ your threshold
     * **⚠️ Uncertain match** otherwise
   * Expand each candidate to see per-hash distances.
   * Optionally **save** the scanned photo as a new key (fill name/purpose/tags).

2. **➕ Add Key**

   * Add a new key record by webcam or upload, fill details, and save.
   * Hashes are computed automatically and stored alongside the image.

3. **🗂️ My Keys**

   * Search by **name**, **purpose**, or **tags**.
   * View each key as a card (image + metadata) and **delete** if needed.
   * Expand “Table view” for a sortable grid.

4. **⤴️ Export / Import**

   * **Export** your database to **CSV** or **JSON** (saved under `data/exports/`).
   * **Import** previously exported CSV/JSON.

     * Required columns: `name`, `purpose`, `image_path`
     * Optional: `description`, `tags`, `ahash`, `phash`, `dhash`, `whash`, `created_at`
     * If hashes are missing but the image file exists, the app **recomputes** hashes.

### Settings (left sidebar)

* **Accept match if combined score ≤ N**
  Default **35**. Lower is stricter (more similar).
  Start around **35–45** and tune based on your photos/lighting.
* **Top-K candidates**
  Default **5**. Shows the top N matches for review.
* Database path is shown for reference.

---

## Data layout & privacy

All data is saved under `./data` next to `app.py`:

* **SQLite DB**: `data/keys.db`
* **Images**: `data/images/` (JPEG files saved on add/scan)
* **Exports**: `data/exports/` (CSV/JSON)

The app is **local-only**. It does not send images or data to any server.

---

## How identification works (the tech)

The app uses **perceptual image hashing** via the `ImageHash` library:

* **aHash** (Average Hash): averages luminance and thresholds pixels.
* **pHash** (Perceptual Hash): DCT-based; robust to small changes.
* **dHash** (Difference Hash): looks at gradients/adjacent pixel differences.
* **wHash** (Wavelet Hash): wavelet transforms; robust to lighting changes.

For a scanned photo, we compute all four hashes (hex strings) and compare them with each record’s stored hashes using **Hamming distance**:

* **Per-hash distance** = number of differing bits between two hashes (lower = more similar).
* **Combined score** = sum of valid per-hash distances across the four hashes:

  ```
  combined = ahash_dist + phash_dist + dhash_dist + whash_dist
  ```

  If a stored hash is missing for a record, it’s skipped in the sum.

We then **sort ascending** by the combined score and show the top matches.

### Why multiple hashes?

Each hash is robust to different perturbations (noise, rotation, lighting). Summing several distances gives a more stable similarity metric than relying on a single hash.

---

## Tips for better matches

* Photograph keys **face-on** on a **plain, high-contrast background**.
* Keep lighting consistent; avoid heavy shadows or glare.
* Use similar **scale and orientation** for new scans vs. saved images.
* Save **both sides** of a key as separate records if needed (e.g., unique markings).

---

## Database schema

```sql
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
```

---

## Export/Import formats

### CSV example

```csv
id,name,purpose,description,tags,image_path,ahash,phash,dhash,whash,created_at
1,Front Door,Home Entry,,home,./data/images/20250101_key1.jpg,ff00...,aa11...,bb22...,cc33...,2025-01-01T12:34:56
```

### JSON example

```json
[
  {
    "name": "Garage",
    "purpose": "Garage Door",
    "description": "",
    "tags": "home,outdoor",
    "image_path": "./data/images/20250101_garage.jpg",
    "ahash": "ff00...",
    "phash": "aa11...",
    "dhash": "bb22...",
    "whash": "cc33...",
    "created_at": "2025-01-02T08:00:00"
  }
]
```

**Import rules**

* Must include: `name`, `purpose`, `image_path`.
* If any hash is missing and `image_path` exists, the app **recomputes** it.
* Import does **not** copy image files; it references the paths you provide.

---

## Testing & diagnostics

### Built-in self-tests (don’t touch your DB)

Run minimal checks for hashing/distance logic:

```bash
RUN_KEY_APP_TESTS=1 python app.py
```

You should see: `All self-tests passed ✔️`

### Common issues & fixes

* **`TabError: inconsistent use of tabs and spaces`**
  The app uses **spaces-only** indentation. Ensure your editor converts tabs to spaces. (This repo/version already standardizes indentation.)

* **`ModuleNotFoundError: No module named 'imagehash'`**
  Install dependencies: `pip install -r requirements.txt` (or `pip install ImageHash`).

* **Webcam not available / permission denied**

  * Use Chrome/Edge/Firefox, and allow camera access when prompted.
  * Or skip webcam and use **Upload**.

* **Blank or wrong matches**

  * Check lighting/background.
  * Raise/lower the threshold in the sidebar.
  * Add more high-quality reference photos.

* **Images missing after import**
  Import expects `image_path` to exist on disk. Copy image files to the target machine and update paths if needed.

---

## Performance notes

* Hash computation is fast (tiny downsampled images + DCT/wavelet).
* Database reads are lightweight; everything is local SQLite.
* Large collections: consider pruning duplicates and keeping images ≤ \~2–3 MB.

---

## Security & privacy

* **Local-only** app: no network calls.
* Keep `./data/` private (especially images).
* If you back up or share the DB, remember it references **real photos of your keys**.

---

## Roadmap ideas (optional)

* Optional **feature matching** (ORB/SIFT) for a second-stage re-rank.
* Bulk image import & auto-tagging.
* Duplicate detector / “are these the same key?” helper.
* Mobile packaging (Streamlit Community Cloud or PWA wrapper).

---

## Contributing & coding style

* Indentation: **spaces only** (no tabs).
* Keep UI elements inside the correct container (tabs belong in the main area, not the sidebar).
* Prefer small, composable helpers in the image/hash layer.

---

## License & disclaimer

This tool is intended for **personal record-keeping**. Do not use it for security-sensitive decision making. The similarity scores are heuristic and may be wrong.