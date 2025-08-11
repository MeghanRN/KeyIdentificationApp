# cv_key.py
from __future__ import annotations
import io
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

# --- Optional: high-quality background removal via U^2-Net (rembg) ---
try:
    from rembg import remove as rembg_remove  # pip install rembg
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

# --- Optional: Zernike moments (scikit-image) ---
try:
    from skimage.measure import moments_zernike  # pip install scikit-image
    _HAS_ZERNIKE = True
except Exception:
    _HAS_ZERNIKE = False

# --------------------------
# Data structures
# --------------------------
@dataclass
class ShapeFeatures:
    # Canonicalized geometry
    width: int
    height: int
    contour: List[List[int]]            # Nx2 int points, canonicalized CCW
    mask_bbox: Tuple[int, int, int, int]# (x,y,w,h) in canonical frame

    # Global descriptors
    efd: List[float]                    # Elliptic Fourier descriptor (normalized)
    hu: List[float]                     # 7 Hu moments (log scale)
    zernike: Optional[List[float]]      # optional Zernike (n<=8) or None

    # Blade/bitting signature
    bitting: List[float]                # 1-D depth curve (length K, normalized)

    # Local feature summary
    orb_kp: int                         # keypoint count
    orb_desc: Optional[np.ndarray]      # ORB descriptor matrix (uint8)

def _imread_pil_to_cv(img_pil) -> np.ndarray:
    # Accept PIL.Image or NumPy; convert to OpenCV BGR
    if hasattr(img_pil, "convert"):
        img = np.array(img_pil.convert("RGB"))[:, :, ::-1].copy()
    else:
        # Assume already BGR uint8
        img = img_pil.copy()
    return img

# --------------------------
# 1) Segmentation
# --------------------------
def _segment_key(img_bgr: np.ndarray) -> np.ndarray:
    """
    Return binary mask (uint8 0/255) of the key.
    Prefer rembg(U^2-Net); otherwise use robust OpenCV fallback.
    """
    if _HAS_REMBG:
        try:
            out = rembg_remove(img_bgr[:, :, ::-1])  # expects RGB
            if out.ndim == 3 and out.shape[2] == 4:  # RGBA
                alpha = out[:, :, 3]
                mask = (alpha > 10).astype(np.uint8) * 255
            else:
                gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            # post-process: close holes & keep largest component
            mask = _clean_mask(mask)
            return mask
        except Exception:
            pass

    # Fallback: CLAHE + adaptive threshold + largest component
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    # Invert if needed (key should be white)
    if np.mean(thr) < 127:
        thr = 255 - thr
    mask = _clean_mask(thr)
    return mask

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # Keep largest component
    num, lbls = cv2.connectedComponents(mask)
    if num <= 1:
        return mask
    areas = [(lbls == i).sum() for i in range(1, num)]
    i_max = 1 + int(np.argmax(areas))
    return ((lbls == i_max).astype(np.uint8) * 255)

# --------------------------
# 2) Canonicalization
# --------------------------
def _canonicalize(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """
    Align major axis horizontally (blade right), scale s.t. major-axis length==1 in normalized space.
    Returns: bin_mask (HxW uint8), contour (Nx2 float), bbox (x,y,w,h)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found.")
    cnt = max(contours, key=cv2.contourArea).squeeze(1)

    # PCA orientation
    data = cnt.astype(np.float32)
    mean, eigvecs = cv2.PCACompute(data, mean=None, maxComponents=2)
    major = eigvecs[0]  # unit vector
    angle = math.degrees(math.atan2(major[1], major[0]))

    # Rotate around center to make major axis horizontal
    (h, w) = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    contours2,_ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt2 = max(contours2, key=cv2.contourArea).squeeze(1)

    x,y,ww,hh = cv2.boundingRect(cnt2)
    roi = rot[y:y+hh, x:x+ww]

    # Decide blade direction: assume blade extends to the right (longer, thinner side).
    # If not, flip horizontally.
    left_area = (roi[:, :ww//2] > 0).sum()
    right_area = (roi[:, ww//2:] > 0).sum()
    if left_area > right_area:
        roi = cv2.flip(roi, 1)
        cnt2[:,0] = (w - cnt2[:,0])  # approximate flip for contour (ok for shape descriptors)

    # Normalize size to fixed height while preserving aspect
    target_h = 300
    scale = target_h / roi.shape[0]
    roi_n = cv2.resize(roi, (int(roi.shape[1]*scale), target_h), interpolation=cv2.INTER_NEAREST)

    # Recompute contour in normalized image
    cs,_ = cv2.findContours(roi_n, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_n = max(cs, key=cv2.contourArea).squeeze(1).astype(np.float32)

    return roi_n, cnt_n, (0, 0, roi_n.shape[1], roi_n.shape[0])

# --------------------------
# 3) Descriptors
# --------------------------
def _hu_moments(mask: np.ndarray) -> List[float]:
    m = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    # log scale for stability
    hu = [-np.sign(v)*np.log10(abs(v)+1e-12) for v in hu]
    return list(hu)

def _zernike(mask: np.ndarray, radius: int = 64, n: int = 8) -> Optional[List[float]]:
    if not _HAS_ZERNIKE:
        return None
    # place mask inside circle of 'radius'
    h, w = mask.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    y0 = (s - h)//2
    x0 = (s - w)//2
    sq[y0:y0+h, x0:x0+w] = mask
    sq = cv2.resize(sq, (2*radius, 2*radius), interpolation=cv2.INTER_NEAREST)
    # Normalize to [0,1]
    img = (sq > 0).astype(np.float32)
    # Return magnitudes for orders up to n
    feats = []
    for order in range(0, n+1):
        for repetition in range(order+1):
            if (order - repetition) % 2 == 0:
                feats.append(abs(moments_zernike(img, radius, order, repetition)))
    return feats

def _elliptic_fourier_descriptors(contour: np.ndarray, harmonics: int = 20) -> List[float]:
    """
    Simple EFD implementation: compute coefficients and normalize for translation/scale/rotation.
    Reference: Kuhl & Giardina 1982.
    """
    pts = contour.astype(np.float64)
    # Ensure closed contour
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # Differential
    dxy = np.diff(pts, axis=0)
    dt = np.sqrt((dxy**2).sum(1))
    t = np.hstack([[0.0], np.cumsum(dt)])
    T = t[-1] if t[-1] > 0 else 1.0

    # EFD coefficients (a_n, b_n, c_n, d_n), normalized by first harmonic for invariance
    coeffs = []
    for n in range(1, harmonics+1):
        cn = (2*np.pi*n)/T
        cos_ct = np.cos(cn*t[:-1]); sin_ct = np.sin(cn*t[:-1])
        dx = dxy[:,0]; dy = dxy[:,1]
        a_n = (1/(cn**2*T)) * np.sum(dx * (sin_ct[1:] - sin_ct[:-1]))
        b_n = (1/(cn**2*T)) * np.sum(dx * (-cos_ct[1:] + cos_ct[:-1]))
        c_n = (1/(cn**2*T)) * np.sum(dy * (sin_ct[1:] - sin_ct[:-1]))
        d_n = (1/(cn**2*T)) * np.sum(dy * (-cos_ct[1:] + cos_ct[:-1]))
        coeffs.append([a_n, b_n, c_n, d_n])
    coeffs = np.array(coeffs)

    # Normalize by first harmonic magnitude to remove scale/rotation
    a1,b1,c1,d1 = coeffs[0]
    norm = math.sqrt(a1*a1 + b1*b1 + c1*c1 + d1*d1) + 1e-12
    efd_norm = (coeffs / norm).flatten()
    return efd_norm.tolist()

def _blade_bitting_profile(mask: np.ndarray, samples: int = 160) -> List[float]:
    """
    Assume canonicalized mask with blade to the right.
    Crop right ~60% (blade) and sample vertical distance from spine to teeth edge.
    """
    h, w = mask.shape
    x0 = int(0.4 * w)
    roi = mask[:, x0:]
    # Distance from top to first background (spine) and bottom to first background (teeth)
    # We'll use bottom depth (teeth profile)
    depth = []
    col_idx = np.linspace(0, roi.shape[1]-1, samples).astype(int)
    for x in col_idx:
        col = roi[:, x]
        # from bottom up
        ys = np.where(col[::-1] > 0)[0]
        depth_px = float(ys[0]) if ys.size else 0.0
        depth.append(depth_px)
    # Normalize to [0,1] by max depth
    m = max(depth) if max(depth) > 0 else 1.0
    depth = [d/m for d in depth]
    return depth

def _orb_features(img_bin: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
    # Use edges to encourage structural keypoints
    edges = cv2.Canny(img_bin, 50, 150)
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    kp, desc = orb.detectAndCompute(edges, None)
    return len(kp), desc

# --------------------------
# Public API
# --------------------------
def extract_and_describe(pil_image) -> Tuple[ShapeFeatures, np.ndarray]:
    """
    Main entry: pass a PIL.Image (or NumPy BGR).
    Returns (features, debug_viz_bgr)
    """
    img = _imread_pil_to_cv(pil_image)
    mask = _segment_key(img)
    can_mask, can_cnt, bbox = _canonicalize(mask)

    efd = _elliptic_fourier_descriptors(can_cnt, harmonics=24)
    hu = _hu_moments(can_mask)
    zernike = _zernike(can_mask, radius=96, n=8)
    bitting = _blade_bitting_profile(can_mask, samples=160)
    kp_count, desc = _orb_features(can_mask)

    feats = ShapeFeatures(
        width=int(can_mask.shape[1]),
        height=int(can_mask.shape[0]),
        contour=can_cnt.astype(int).tolist(),
        mask_bbox=bbox,
        efd=efd,
        hu=hu,
        zernike=zernike,
        bitting=bitting,
        orb_kp=kp_count,
        orb_desc=desc
    )

    # Debug viz: overlay contour on mask
    dbg = cv2.cvtColor(can_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [np.array(feats.contour, np.int32)], -1, (0,255,0), 1)
    return feats, dbg

# --------------------------
# Matching & scoring
# --------------------------
def _zscore(x: float, mean: float, std: float) -> float:
    return (x - mean) / (std + 1e-9)

def _dtw(a: np.ndarray, b: np.ndarray, w: Optional[int]=None) -> float:
    """Classic DTW (Sakoe–Chiba band if w provided). Lower is better."""
    n, m = len(a), len(b)
    if w is None:
        w = max(n, m)
    w = max(w, abs(n - m))
    inf = 1e12
    D = np.full((n+1, m+1), inf, dtype=np.float32)
    D[0,0] = 0.0
    for i in range(1, n+1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end+1):
            cost = abs(a[i-1] - b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m] / (n + m))

def _hausdorff(maskA: np.ndarray, maskB: np.ndarray) -> float:
    """Approximate Hausdorff via OpenCV extractor if available; else fallback with distance transform."""
    try:
        extractor = cv2.createHausdorffDistanceExtractor()
        cntsA,_ = cv2.findContours(maskA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cntsB,_ = cv2.findContours(maskB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cA = max(cntsA, key=cv2.contourArea)
        cB = max(cntsB, key=cv2.contourArea)
        return float(extractor.computeDistance(cA, cB))
    except Exception:
        # Fallback: symmetric chamfer-like distance using distance transform
        def chamfer(a, b):
            dt = cv2.distanceTransform((255-b).astype(np.uint8), cv2.DIST_L2, 3)
            ea = cv2.Canny(a, 50, 150)
            ys, xs = np.where(ea > 0)
            if len(xs) == 0: return 1e6
            return float(np.mean(dt[ys, xs]))
        return max(chamfer(maskA, maskB), chamfer(maskB, maskA))

def _match_orb(descQ: Optional[np.ndarray], descR: Optional[np.ndarray]) -> float:
    """Return 1 - inlier_ratio (so lower is better)."""
    if descQ is None or descR is None:
        return 1.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descQ, descR, k=2)
    good = 0
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75*m[1].distance:
            good += 1
    ratio = good / max(1, len(matches))
    return 1.0 - float(ratio)

def pack_for_db(feats: ShapeFeatures, svg: Optional[str]=None) -> Dict[str, object]:
    return {
        "width": feats.width, "height": feats.height,
        "contour": feats.contour, "bbox": feats.mask_bbox,
        "efd": feats.efd, "hu": feats.hu, "zernike": feats.zernike,
        "bitting": feats.bitting, "orb_kp": feats.orb_kp,
        # Store ORB as list for JSON; app can skip if too big
        "orb_desc": None if feats.orb_desc is None else feats.orb_desc.tolist(),
        "svg": svg
    }

def unpack_from_db(d: Dict[str, object]) -> ShapeFeatures:
    return ShapeFeatures(
        width=int(d["width"]), height=int(d["height"]),
        contour=d["contour"],
        mask_bbox=tuple(d.get("bbox", (0,0,int(d["width"]),int(d["height"])))),
        efd=d["efd"], hu=d["hu"], zernike=d.get("zernike"),
        bitting=d["bitting"],
        orb_kp=int(d.get("orb_kp", 0)),
        orb_desc=None if d.get("orb_desc") is None else np.array(d["orb_desc"], dtype=np.uint8)
    )

def match_score(query: ShapeFeatures, candidate: ShapeFeatures,
                stats: Optional[Dict[str, Tuple[float,float]]] = None) -> float:
    """
    Lower is better. Combines:
    - EFD L2
    - Hu L2
    - Zernike L2 (if both present)
    - Hausdorff/Chamfer on masks reconstructed from contours
    - DTW on bitting
    - ORB mismatch (1 - inlier_ratio)
    """
    # Reconstruct masks from contours for Hausdorff
    def mask_from_contour(sf: ShapeFeatures, out_h: int = 280) -> np.ndarray:
        scale = out_h / sf.height
        out_w = max(1, int(sf.width * scale))
        m = np.zeros((out_h, out_w), np.uint8)
        cnt = (np.array(sf.contour, np.int32) * scale).astype(np.int32)
        cv2.drawContours(m, [cnt], -1, 255, thickness=-1)
        return m

    efd_d = np.linalg.norm(np.array(query.efd) - np.array(candidate.efd))
    hu_d  = np.linalg.norm(np.array(query.hu)  - np.array(candidate.hu))
    if query.zernike is not None and candidate.zernike is not None:
        zern_d = np.linalg.norm(np.array(query.zernike) - np.array(candidate.zernike))
    else:
        zern_d = 0.0

    mQ = mask_from_contour(query)
    mR = mask_from_contour(candidate)
    haus = _hausdorff(mQ, mR)

    dtw = _dtw(np.array(query.bitting), np.array(candidate.bitting), w=20)

    orb_mis = _match_orb(query.orb_desc, candidate.orb_desc)

    # Normalize (z-score) w.r.t. provided stats if any
    comps = {
        "efd": efd_d, "hu": hu_d, "zern": zern_d,
        "haus": haus, "dtw": dtw, "orb": orb_mis
    }
    if stats:
        for k in comps:
            mu, sd = stats.get(k, (0.0, 1.0))
            comps[k] = _zscore(comps[k], mu, sd)

    # Weighted fusion (tweakable; good defaults)
    w = {"efd": 0.30, "hu": 0.12, "zern": 0.08, "haus": 0.30, "dtw": 0.15, "orb": 0.05}
    score = sum(w[k]*comps[k] for k in w)
    return float(score)