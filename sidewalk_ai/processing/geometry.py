# sidewalk_ai/processing/geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2, os
import numpy as np


# --------------------------------------------------------------------------- #
# 0) Typed results – easy to validate/serialize
# --------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class WidthResult:
    width_m: float           # best estimate
    margin_m: float          # ± error band (25 % default)
    n_pixels: int            # inlier count used in the fit


@dataclass(slots=True, frozen=True)
class ClearanceResult:
    label: str
    L_m: float               # free space on the left  of the obstacle
    R_m: float               # free space on the right
    total_m: float           # L+R+obstacle width


# --------------------------------------------------------------------------- #
# 1)  Core helpers -- largely ported from the research prototype
# --------------------------------------------------------------------------- #

def _scale_from_ground(
    sidewalk: np.ndarray,
    depth:    np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H_cam: float = 2.70,
    RANSAC_N: int = 8_000,
    frac_good: float = 0.08,
) -> float:
    """
    Fit a plane through random sidewalk pixels via RANSAC and
    derive an *α* scale factor that converts pixel units into metres.

    Ported – with small clean-ups – from `scale_from_ground` :contentReference[oaicite:0]{index=0}.
    """
    ys, xs = np.where(sidewalk)
    if xs.size < 50:               # not enough data
        return 1.0

    rng  = np.random.default_rng(0)
    sel  = rng.choice(xs.size, size=min(RANSAC_N, xs.size), replace=False)
    u, v = xs[sel], ys[sel]
    Zr   = depth[v, u].astype(np.float32)

    Xr = (u - cx) * Zr / fx
    Yr = (v - cy) * Zr / fy
    P  = np.column_stack([Xr, Yr, Zr])

    # robust distance threshold = 1 % of IQR
    iqr = np.subtract(*np.percentile(Zr, [75, 25]))
    eps = 0.01 * iqr if iqr > 1e-3 else 0.005

    best_cnt = 0
    best_d   = 1.0

    for _ in range(1_500):
        a, b, c = P[rng.choice(P.shape[0], 3, replace=False)]
        n   = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            continue
        n /= n_norm
        d  = -np.dot(n, a)
        cnt = np.count_nonzero(np.abs(P @ n + d) < eps)

        if cnt > best_cnt:
            best_cnt, best_d = cnt, d
            if cnt > frac_good * P.shape[0]:
                break

    return abs(H_cam / best_d)


def _largest_dense_cluster(
    xs: np.ndarray,
    gap_thresh: float = 0.20,
) -> np.ndarray:
    """
    Returns the densely packed subset with the most points along the 1-D axis,
    exactly as in the thesis prototype :contentReference[oaicite:1]{index=1}.
    """
    if xs.size == 0:
        return xs
    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= gap_thresh:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return np.array(max(clusters, key=len))


# --------------------------------------------------------------------------- #
# 2)  Public API – what the pipeline will call
# --------------------------------------------------------------------------- #

def compute_width(
    sidewalk:      np.ndarray,          # bool mask (H×W)
    depth:         np.ndarray,          # float32 depth map (H×W)
    *,                                  # ← force keyword-only below
    metric_depth:  bool = False,        # ← NEW (True for ZoeDepth)
    FOV_deg:       float = 75.0,        # empirical optimum
    band_frac:     tuple[float, float] = (0.50, 1.00),  # lower 50 % of the image
    trim_pct:      float = 2.0,         # outlier trim for density cluster
    err_pct:       float = 25.0,        # propagated error %
) -> WidthResult:
    """
    Replicates the geometric steps from Section 4.4 of the paper while
    staying side-effect-free.  All parameters are overridable for experiments.
    """
    H, W = sidewalk.shape
    y0 = int(H * band_frac[0])
    y1 = int(H * band_frac[1])

    # 1) pixels we trust for width
    ys, xs = np.where(sidewalk[y0:y1])
    if xs.size < 20:
        return WidthResult(0.0, 0.0, 0)

    Z = depth[y0 + ys, xs]

    # 1) ensure depth is in metres  ────────────────────────────────────
    fx = W / (2.0 * np.tan(np.radians(FOV_deg / 2)))
    fy = fx
    cx = W / 2
    cy = H / 2
    if not metric_depth:
        alpha = _scale_from_ground(sidewalk, depth, fx, fy, cx, cy)
        if os.getenv("SWAI_FORCE_FALLBACK") == "1":
            alpha = float(os.getenv("SWAI_FALLBACK_SCALE", "0.075"))
        Z = Z * alpha

    # 2) pin-hole intrinsics from FOV (reuse fx computed above)
    X = (xs - W / 2) * Z / fx

    # 3) trim outliers and keep densest cluster
    lo, hi = np.percentile(X, [trim_pct, 100 - trim_pct])
    X_in   = X[(X >= lo) & (X <= hi)]
    stable = _largest_dense_cluster(np.sort(X_in))

    if stable.size < 2:
        return WidthResult(0.0, 0.0, 0)

    width = stable[-1] - stable[0]
    margin = err_pct / 100.0 * width
    return WidthResult(width, margin, int(stable.size))


def compute_clearances(
    sidewalk:   np.ndarray,                 # bool H×W
    obstacles:  Sequence[tuple[str, np.ndarray]],   # (label, bool-mask)
    depth:      np.ndarray,
    *,                                  # ← force keyword-only below
    metric_depth:  bool = False,        # ← NEW (True for ZoeDepth)
    FOV_deg:    float = 70.0,
    h_up_px:    int = 12,
    contact_gap: int = 25,
) -> list[ClearanceResult]:
    """
    Vectorised re-implementation of `clearance_m()` :contentReference[oaicite:2]{index=2}
    returning structured results for **all** obstacles at once.
    """
    H, W = sidewalk.shape
    fx = W / (2.0 * np.tan(np.radians(FOV_deg / 2)))
    cx = W / 2
    fy = fx
    cy = H / 2

    α = _scale_from_ground(sidewalk, depth, fx, fy, cx, cy)

    results: list[ClearanceResult] = []
    for label, obs in obstacles:
        band = _contact_band(sidewalk, obs, h_up_px, contact_gap)
        if band is None:
            continue
        px = _px_clearance(sidewalk, band)
        if px is None:
            continue
        l_px, r_px, t_px = px
        rows = np.unique(np.where(band)[0])
        Z_band = α * depth[rows].mean()
        scale  = Z_band / fx
        results.append(
            ClearanceResult(label, l_px * scale, r_px * scale, t_px * scale)
        )
    return results or [ClearanceResult("none", -1.0, -1.0, -1.0)]


# --------------------------------------------------------------------------- #
# 3)  Internal helpers (kept private)
# --------------------------------------------------------------------------- #

def _contact_band(S: np.ndarray, O: np.ndarray,
                  h_up: int, max_gap: int) -> np.ndarray | None:
    """
    Identical logic to `contact_band()` in the prototype :contentReference[oaicite:3]{index=3}.
    """
    H, _ = S.shape
    band = np.zeros_like(O, bool)
    for x in np.unique(np.where(O)[1]):
        y_bot = np.where(O[:, x])[0].max()
        for dy in range(max_gap + 1):
            y = y_bot + dy
            if y >= H:
                break
            if S[y, x]:
                band[max(0, y_bot - h_up + 1): y_bot + 1, x] = True
                break
    return band if band.any() else None


def _px_clearance(S: np.ndarray, band: np.ndarray) -> tuple[int, int, int] | None:
    """
    Pixel-domain clearance measurement adapted from `clearance_px_scan` :contentReference[oaicite:4]{index=4}
    but without Python loops inside the hotspot.
    """
    free = (S & ~band).astype(np.uint8)
    distL = cv2.distanceTransform(free, cv2.DIST_L1, 3)
    distR = cv2.distanceTransform(free[:, ::-1], cv2.DIST_L1, 3)[:, ::-1]

    rows = np.unique(np.where(band)[0])
    if rows.size == 0:
        return None

    l_px = np.inf
    r_px = np.inf
    t_px = np.inf
    for y in rows:
        walk = np.where(S[y])[0]
        obs  = np.where(band[y])[0]
        if walk.size < 2 or obs.size == 0:
            continue
        l_px = min(l_px, obs.min()  - walk.min())
        r_px = min(r_px, walk.max() - obs.max())
        t_px = min(t_px, l_px + r_px + (obs.max() - obs.min() + 1))

    return None if np.isinf(l_px) else (int(l_px), int(r_px), int(t_px))