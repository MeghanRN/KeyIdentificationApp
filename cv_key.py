# cv_key.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

SIG_LEN = 1024       # samples along the outline for 1-D descriptors
FOURIER_K = 32      # how many low-frequency magnitudes to keep (skip DC)
CANVAS = 256        # raster size for chamfer

@dataclass
class OutlineFeatures:
    width: int
    height: int
    contour: List[List[int]]           # Nx2 canonicalized, scaled to CANVAS height
    signature: Dict[str, List[float]]  # {"radial":[...], "curv":[...], "fourier":[...]}

# ---------- helpers ----------
def _to_bgr(img) -> np.ndarray:
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure key is white
    if np.mean(thr) > 127: thr = 255 - thr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    return thr

def _largest_contour(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("No key outline found. Use a plainer background and include the full key.")
    return max(cnts, key=cv2.contourArea).squeeze(1)  # Nx2

def _uniform_resample(contour: np.ndarray, n: int = SIG_LEN) -> np.ndarray:
    pts = contour.astype(np.float64)
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
    return np.stack([x, y], axis=1)

def _pca_canonicalize(points: np.ndarray) -> np.ndarray:
    """Rotate by PCA to horizontal, then scale so height==CANVAS, keep aspect, center."""
    pts = points.astype(np.float64)
    c = pts.mean(axis=0)
    X = pts - c
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # principal axis
    ang = np.arctan2(v[1], v[0])
    ca, sa = np.cos(-ang), np.sin(-ang)
    R = np.array([[ca, -sa], [sa, ca]])
    Xr = X @ R.T

    # uniform scale to CANVAS height
    min_y, max_y = Xr[:, 1].min(), Xr[:, 1].max()
    h = max(1e-6, (max_y - min_y))
    s = CANVAS / h
    Xs = Xr * s

    # center to canvas
    minx, maxx = Xs[:, 0].min(), Xs[:, 0].max()
    miny, maxy = Xs[:, 1].min(), Xs[:, 1].max()
    W = int(np.ceil(maxx - minx)) + 6
    H = int(np.ceil(maxy - miny)) + 6
    Xs[:, 0] -= (minx + maxx) / 2.0
    Xs[:, 1] -= (miny + maxy) / 2.0
    # map to pixel coords (centered)
    Xs[:, 0] += W / 2.0
    Xs[:, 1] += H / 2.0
    return Xs

def _radial_signature(points: np.ndarray) -> np.ndarray:
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    r = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    r = r.astype(np.float64)
    r -= r.mean()
    r /= (np.linalg.norm(r) + 1e-12)
    return r

def _curvature_signature(points: np.ndarray) -> np.ndarray:
    p = points.astype(np.float64)
    dp = np.diff(np.vstack([p, p[0]]), axis=0)  # cyclic diff
    ang = np.arctan2(dp[:, 1], dp[:, 0])
    ang_unw = np.unwrap(ang)
    kappa = np.diff(np.hstack([ang_unw, ang_unw[0]]))  # turning angle
    # smooth a bit
    kappa = cv2.GaussianBlur(kappa.reshape(-1, 1), (5, 1), 0).ravel()
    kappa -= kappa.mean()
    kappa /= (np.linalg.norm(kappa) + 1e-12)
    return kappa

def _fourier_mag_descriptor(points: np.ndarray, k: int = FOURIER_K) -> np.ndarray:
    p = points.astype(np.float64)
    z = (p[:, 0] - p[:, 0].mean()) + 1j * (p[:, 1] - p[:, 1].mean())
    F = np.fft.fft(z)
    mag = np.abs(F)[1:k + 1]  # skip DC
    mag /= (np.linalg.norm(mag) + 1e-12)
    return mag

def _rasterize(contour: np.ndarray, size: int = CANVAS) -> np.ndarray:
    # center to a fixed canvas
    pts = contour.copy()
    minx, maxx = pts[:, 0].min(), pts[:, 0].max()
    miny, maxy = pts[:, 1].min(), pts[:, 1].max()
    scale = size / max(maxx - minx, maxy - miny, 1e-6)
    pts[:, 0] = (pts[:, 0] - (minx + maxx) / 2.0) * scale + size / 2.0
    pts[:, 1] = (pts[:, 1] - (miny + maxy) / 2.0) * scale + size / 2.0

    mask = np.zeros((size, size), np.uint8)
    cv2.polylines(mask, [pts.astype(np.int32)], isClosed=True, color=255, thickness=1)
    return mask

def _best_circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    corr = np.fft.ifft(fa * np.conj(fb)).real
    return float(np.max(corr))

def _chamfer_distance(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    # mean nearest-edge distance (symmetrized)
    a = (a_mask > 0).astype(np.uint8) * 255
    b = (b_mask > 0).astype(np.uint8) * 255
    da = cv2.distanceTransform(255 - a, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(255 - b, cv2.DIST_L2, 3)
    ya, xa = np.where(a > 0)
    yb, xb = np.where(b > 0)
    if len(xa) == 0 or len(xb) == 0:
        return 1e6
    d_ab = float(np.mean(db[ya, xa]))
    d_ba = float(np.mean(da[yb, xb]))
    return 0.5 * (d_ab + d_ba)

# ---------- public API ----------
def extract_outline_features(pil_image) -> Tuple[OutlineFeatures, np.ndarray]:
    img = _to_bgr(pil_image)
    mask = _binarize(img)
    cnt = _largest_contour(mask)
    rs = _uniform_resample(cnt, SIG_LEN)
    canon = _pca_canonicalize(rs)                  # rotation/scale normalized
    sig_r = _radial_signature(canon)
    sig_c = _curvature_signature(canon)
    sig_f = _fourier_mag_descriptor(canon, FOURIER_K)

    feats = OutlineFeatures(
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        contour=canon.astype(int).tolist(),
        signature={
            "radial": sig_r.tolist(),
            "curv": sig_c.tolist(),
            "fourier": sig_f.tolist(),
        },
    )

    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [rs.astype(np.int32)], -1, (0, 255, 0), 1)
    return feats, dbg

def fused_outline_distance(q: OutlineFeatures, c: OutlineFeatures) -> Tuple[float, Dict[str, float]]:
    # fetch and guard
    def vec(d: Dict[str, List[float]], key: str) -> np.ndarray:
        v = d.get(key)
        return np.asarray(v, dtype=np.float64) if v is not None else None

    qr, qc, qf = vec(q.signature, "radial"), vec(q.signature, "curv"), vec(q.signature, "fourier")
    cr, cc, cf = vec(c.signature, "radial"), vec(c.signature, "curv"), vec(c.signature, "fourier")

    # normalize helpers
    def nz(z):
        z = z - z.mean()
        n = np.linalg.norm(z)
        return z / (n + 1e-12)

    # radial (circular corr + mirror)
    d_rad = 1.0
    if qr is not None and cr is not None and len(qr) == len(cr):
        a, b = nz(qr), nz(cr)
        corr = max(_best_circular_corr(a, b), _best_circular_corr(a, b[::-1]))
        d_rad = max(0.0, 2.0 - 2.0 * corr)   # 0..~4

    # curvature
    d_curv = 1.0
    if qc is not None and cc is not None and len(qc) == len(cc):
        a, b = nz(qc), nz(cc)
        corr = max(_best_circular_corr(a, b), _best_circular_corr(a, b[::-1]))
        d_curv = max(0.0, 2.0 - 2.0 * corr)

    # fourier magnitude (L2, already shift-invariant)
    d_four = 1.0
    if qf is not None and cf is not None and len(qf) == len(cf):
        a = qf / (np.linalg.norm(qf) + 1e-12)
        b = cf / (np.linalg.norm(cf) + 1e-12)
        d_four = float(np.linalg.norm(a - b))  # 0..2

    # chamfer on rasterized canonical contours (cheap & useful sanity)
    qmask = _rasterize(np.array(q.contour, np.float64), CANVAS)
    cmask = _rasterize(np.array(c.contour, np.float64), CANVAS)
    d_ch = _chamfer_distance(qmask, cmask) / (CANVAS * 0.05)  # scale ~0..~1

    # weights (tuneable)
    w_rad, w_curv, w_four, w_ch = 0.45, 0.25, 0.20, 0.10
    fused = w_rad * d_rad + w_curv * d_curv + w_four * d_four + w_ch * d_ch

    return float(fused), {"radial": float(d_rad), "curv": float(d_curv), "fourier": float(d_four), "chamfer": float(d_ch)}