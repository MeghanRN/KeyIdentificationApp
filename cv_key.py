# cv_key.py
from __future__ import annotations
import io
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
try:
    import cv2  # pip install opencv-python-headless
except Exception as e:
    cv2 = None

try:
    from rembg import remove as rembg_remove  # pip install rembg
except Exception:
    rembg_remove = None


@dataclass
class ShapeFeatures:
    width: int
    height: int
    hu: List[float]            # 7 Hu moments (log-scaled)
    fourier: List[float]       # DFT magnitude of contour signature (first 32 coeffs)
    contour: List[Tuple[int, int]]  # Resampled contour (e.g., 256 points)


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV not installed. Run: pip install opencv-python-headless"
        )


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def remove_background(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (bgr_image, binary_mask) where mask==255 roughly equals key.
    Prefers rembg if available; falls back to thresholding.
    """
    _ensure_cv2()
    bgr = _pil_to_bgr(img)

    if rembg_remove is not None:
        rgba = rembg_remove(img)  # PIL Image with alpha
        rgba = np.array(rgba.convert("RGBA"))
        alpha = rgba[:, :, 3]
        mask = (alpha > 16).astype(np.uint8) * 255
    else:
        # Simple fallback: adaptive threshold on grayscale + morphology
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (the key)
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

    rect = cv2.minAreaRect(cnt)  # (center, (w, h), angle)
    angle = rect[2]
    # OpenCV angle quirks: normalize to make the long side horizontal
    w, h = rect[1]
    if w < h:
        angle = angle + 90.0

    (hgt, wid) = mask.shape[:2]
    M = cv2.getRotationMatrix2D((wid / 2.0, hgt / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(mask, M, (wid, hgt), flags=cv2.INTER_NEAREST, borderValue=0)

    cnts2, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt2 = max(cnts2, key=cv2.contourArea) if cnts2 else cnt
    return rotated, cnt2


def _resample_contour(cnt: np.ndarray, n: int = 256) -> np.ndarray:
    """Resample contour to n points uniformly along arc length."""
    pts = cnt[:, 0, :].astype(np.float32)
    # Ensure closed loop
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    # cumulative arc length
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
    # Log-scale with sign to stabilize
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
    mag = np.abs(F)[1:k + 1]  # skip DC
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
    feats = ShapeFeatures(width=int(w), height=int(h), hu=hu, fourier=fourier,
                          contour=[(int(x), int(y)) for x, y in resampled])

    # Build a tiny SVG (viewBox is mask size)
    path_d = _contour_to_svg_path(resampled)
    svg = None
    if path_d:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{path_d}" fill="black"/></svg>'

    # Debug overlay image: contour drawn on rotated mask
    dbg = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [cnt], -1, (0, 255, 0), 2)
    dbg_pil = _bgr_to_pil(dbg)
    return feats, svg, dbg_pil


# -------- Matching --------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def match_score(query: ShapeFeatures, candidate: ShapeFeatures) -> float:
    """
    Lower is better—combine several signals into one score.
    Uses:
      - cv2.matchShapes on reconstructed contours
      - Cosine distance of Fourier magnitudes
      - L2 distance on Hu moments
    """
    _ensure_cv2()
    q_cnt = np.array(query.contour, dtype=np.int32).reshape(-1, 1, 2)
    c_cnt = np.array(candidate.contour, dtype=np.int32).reshape(-1, 1, 2)

    ms = cv2.matchShapes(q_cnt, c_cnt, cv2.CONTOURS_MATCH_I1, 0.0)  # ~[0..inf], 0=identical
    fq = np.array(query.fourier, dtype=np.float32)
    fc = np.array(candidate.fourier, dtype=np.float32)
    cos = _cosine(fq, fc)
    hu_q = np.array(query.hu, dtype=np.float32)
    hu_c = np.array(candidate.hu, dtype=np.float32)

    # Weighted sum of normalized components
    # (tuned conservatively; you can tweak in UI later)
    w1, w2, w3 = 0.6, 0.3, 0.1
    score = w1 * float(ms) + w2 * (1.0 - cos) + w3 * float(np.linalg.norm(hu_q - hu_c))
    return float(score)