# cv_key.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

@dataclass
class OutlineFeatures:
    width: int
    height: int
    contour: List[List[int]]     # Nx2 integer points (clockwise or ccw)
    signature: List[float]       # 1-D normalized radial signature (length SIG_LEN)

SIG_LEN = 256  # number of samples along the outline

# ---------------------------
# Core extraction
# ---------------------------
def _to_bgr(img) -> np.ndarray:
    """Accepts PIL.Image or NumPy; returns OpenCV BGR uint8."""
    if hasattr(img, "convert"):
        arr = np.array(img.convert("RGB"))
        return arr[:, :, ::-1].copy()
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            return img.copy()
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    raise ValueError("Unsupported image format")

def _binarize(img_bgr: np.ndarray) -> np.ndarray:
    """Otsu binarization with small denoise + closing; ensures key is white."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # Otsu
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background is white (common), invert so key is white
    if np.mean(thr) > 127:
        thr = 255 - thr
    # Close to fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    return thr

def _largest_contour(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("No key outline found. Try a plainer background and full key in frame.")
    return max(cnts, key=cv2.contourArea).squeeze(1)  # Nx2

def _uniform_resample(contour: np.ndarray, n: int = SIG_LEN) -> np.ndarray:
    """Resample polyline to 'n' points with uniform arc-length spacing."""
    pts = contour.astype(np.float64)
    # ensure closed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    L = s[-1]
    if L <= 0:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0, L, n, endpoint=False)
    x = np.interp(u, s, pts[:, 0])
    y = np.interp(u, s, pts[:, 1])
    return np.stack([x, y], axis=1)  # n x 2

def _radial_signature(points: np.ndarray) -> np.ndarray:
    """Distance of each resampled point to centroid; normalized to zero-mean, unit-norm."""
    cx, cy = points[:,0].mean(), points[:,1].mean()
    r = np.sqrt((points[:,0] - cx)**2 + (points[:,1] - cy)**2)
    # normalize
    r = r.astype(np.float64)
    r -= r.mean()
    denom = np.linalg.norm(r)
    r = r / (denom + 1e-12)
    return r

def extract_outline_signature(pil_image) -> Tuple[OutlineFeatures, np.ndarray]:
    """
    Main entry: returns OutlineFeatures and a debug BGR image with the outline drawn.
    """
    img = _to_bgr(pil_image)
    mask = _binarize(img)
    cnt = _largest_contour(mask)
    rs = _uniform_resample(cnt, SIG_LEN)
    sig = _radial_signature(rs)

    feats = OutlineFeatures(
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        contour=rs.astype(int).tolist(),
        signature=sig.tolist(),
    )

    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [rs.astype(np.int32)], -1, (0, 255, 0), 1)
    return feats, dbg

# ---------------------------
# Matching (single, simple metric)
# ---------------------------
def _best_circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns max circular correlation between two unit-norm, zero-mean vectors.
    Uses FFT-based circular cross-correlation: O(n log n).
    """
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    corr = np.fft.ifft(fa * np.conj(fb)).real
    return float(np.max(corr))

def outline_distance(query: OutlineFeatures, candidate: OutlineFeatures) -> float:
    """
    Lower is better. Range is ~[0, 4].
    If signatures are identical, distance ~ 0.
    We compute 2 - 2*max_corr for both candidate and its reversed version.
    """
    a = np.asarray(query.signature, dtype=np.float64)
    b = np.asarray(candidate.signature, dtype=np.float64)

    # They should already be zero-mean, unit-norm, but re-guard:
    def nz(z):
        z = z - z.mean()
        n = np.linalg.norm(z)
        return z / (n + 1e-12)

    a = nz(a); b = nz(b)
    corr1 = _best_circular_corr(a, b)
    corr2 = _best_circular_corr(a, b[::-1])  # mirror
    corr = max(corr1, corr2)

    # If both unit-norm, ||a-b_shift||^2 = 2 - 2*(a·b_shift)
    dist = 2.0 - 2.0 * corr
    # In case of tiny negative due to float errors:
    if dist < 0:
        dist = 0.0
    return float(dist)