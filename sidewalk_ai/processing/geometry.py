# sidewalk_ai/processing/geometry.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple
import matplotlib.pyplot as plt

import cv2, os
import numpy as np

ORIG_SIZE = (600, 400)      # Street-View static API
CROP_BOTTOM = 20            # logo strip that we remove
CAM_HEIGHT_M = 1.75

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
    obs_width: float | None = None  # width of the obstacle in meters
    L_pixel: Optional[tuple[int, int]] = None  # (x, y)
    R_pixel: Optional[tuple[int, int]] = None


# --------------------------------------------------------------------------- #
# 1)  Core helpers -- largely ported from the research prototype
# --------------------------------------------------------------------------- #

def project_line_to_ground(m, b, fx, fy, cx, cy, pitch_deg=0.0):
    """
    Converte y = m·x + b  (px) para   Z = a·X + c  (m) no plano do solo.
    """
    # Gere ~50 pontos ao longo da linha na imagem
    xs = np.linspace(0, ORIG_SIZE[0]-1, 50)
    ys = m * xs + b
    X, Z = _ground_intersection(xs, ys, fx, fy, cx, cy, pitch_deg)
    good = np.isfinite(X) & np.isfinite(Z)
    if good.sum() < 10:
        raise RuntimeError("curb line proj. failed")
    # Ajuste Z = a·X + c em coordenadas do solo
    a, c = np.polyfit(X[good], Z[good], 1)
    return a, c            # forma   Z = a·X + c


def _scale_from_ground(
    sidewalk, depth, fx, fy, cx, cy,
    H_cam=CAM_HEIGHT_M, rows_from_bottom=20,
    RANSAC_N=10_000, min_inliers=200
) -> float:
    """α robusto ─ usa só as últimas `rows_from_bottom` linhas da calçada."""
    H, _ = sidewalk.shape
    band = np.arange(max(0, H - rows_from_bottom), H)
    ys, xs = np.where(sidewalk[band])
    if xs.size < min_inliers:
        return 1.0                                   # deixa passar em branco

    ys = ys + band[0]                               # re-alinha índice
    rng = np.random.default_rng(0)
    sel = rng.choice(xs.size, size=min(RANSAC_N, xs.size), replace=False)
    u, v = xs[sel], ys[sel]
    Zr = depth[v, u].astype(np.float32)

    Xr = (u - cx) * Zr / fx
    Yr = (v - cy) * Zr / fy
    P  = np.column_stack([Xr, Yr, Zr])

    # threshold = 2·MAD
    mad = 1.4826 * np.median(np.abs(Zr - np.median(Zr)))
    eps = 2.0 * max(mad, 0.01)

    best_cnt = 0
    best_d   = 1.0
    for _ in range(2000):
        a, b, c = P[rng.choice(P.shape[0], 3, replace=False)]
        n = np.cross(b - a, c - a); n_norm = np.linalg.norm(n)
        if n_norm < 1e-6: continue
        n /= n_norm; d = -np.dot(n, a)
        cnt = np.count_nonzero(np.abs(P @ n + d) < eps)
        if cnt > best_cnt:
            best_cnt, best_d = cnt, d
            if cnt > 0.15 * P.shape[0]: break

    if best_cnt < min_inliers:                       # falhou → neutro
        return 1.0
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


def _ground_intersection(u, v, fx, fy, cx, cy, pitch_deg=0.0,
                         H_cam=CAM_HEIGHT_M):
    """
    Vectorised ray–plane intersection for pixels below the horizon.
    """
    # shift the horizon by camera pitch
    v_h = cy - fy * np.tan(np.radians(pitch_deg))
    denom = (v - v_h).astype(np.float32)
    valid = denom > 1.0
    Z = np.full_like(denom, np.nan, np.float32)
    X = np.full_like(denom, np.nan, np.float32)
    Z[valid] = H_cam * fy / denom[valid]
    X[valid] = (u[valid] - cx) * Z[valid] / fx
    return X, Z


def _intrinsics_after_crop(W: int = 600, H_crop: int = 380,
                           crop_bottom: int = 20, fov_deg: float = 75.0) -> tuple[float, float, float, float]:
    """
    Returns fx, fy, cx, cy *in cropped coordinates* but referenced to the
    original optical centre (cy = 200 px).
    """
    fx = W / (2 * np.tan(np.radians(fov_deg / 2)))
    fy = fx
    cx = W / 2
    cy_orig = ORIG_SIZE[1] / 2                      # 200.0
    cy = cy_orig                                   # same row survives the crop
    return fx, fy, cx, cy


def _has_two_curbs(mask: np.ndarray, min_gap_px=50) -> bool:
    """
    Returns True when the mask shows *two* distinct sidewalk regions
    separated by at least `min_gap_px` columns → likely a parallel view.
    """
    cols = np.unique(np.where(mask)[1])
    if cols.size == 0:         # nothing detected
        return False
    split = np.where(np.diff(np.sort(cols)) > min_gap_px)[0]
    return split.size >= 1

# --------------------------------------------------------------------------- #
# 2)  Public API – what the pipeline will call
# --------------------------------------------------------------------------- #

def compute_width(
    sidewalk:      np.ndarray,          # bool mask (H×W)
    depth:         np.ndarray | None = None,
    *,                                  # keyword-only
    pitch_deg:     float = -10.0,
    #metric_depth:  bool = False,
    #use_geom:      bool = True,
    skip_if_dual:  bool = True,
    FOV_deg:       float = 75.0,
    band_frac:     tuple[float, float] = (0.50, 1.00),
    #trim_pct:      float = 2.0,
    err_pct:       float = 25.0,
) -> WidthResult:

    H, W = sidewalk.shape
    y0 = int(H * band_frac[0])
    y1 = int(H * band_frac[1])

    # 0) intrínsecos já no começo ──────────────────────────────────────
    fx, fy, cx, cy = _intrinsics_after_crop(W, H, CROP_BOTTOM, FOV_deg)

    # 1) horizonte real → garante banda abaixo dele
    v_h = cy - fy * np.tan(np.radians(pitch_deg))
    y0  = max(y0, int(v_h + 5))
    if y0 >= y1 - 5:
        return WidthResult(0.0, 0.0, 0)

    # 2) descarta cena com duas calçadas paralelas → deixa p/ Estágio A
    if skip_if_dual and _has_two_curbs(sidewalk):
        return WidthResult(0.0, 0.0, 0)

    # 3) pixels candidatos
    ys, xs = np.where(sidewalk[y0:y1])
    if xs.size < 20:
        return WidthResult(0.0, 0.0, 0)

    # 4) medida por fileira – primeiro px da calçada em cada lado
    widths = []
    v_ref   = np.arange(y0, y1)
    α = 1.0
    if depth is not None:
        α = _scale_from_ground(sidewalk, depth, fx, fy, cx, cy)  # usa o novo

    for v in v_ref:
        cols = np.where(sidewalk[v])[0]
        if cols.size < 2:      # calçada não visível nesta linha
            continue

        uL, uR = cols[0], cols[-1]
        ZL = (depth[v, uL] * α) if depth is not None else None
        ZR = (depth[v, uR] * α) if depth is not None else None

        # se não temos depth confiável, cai para intersecção geométrica
        if ZL is None or ZR is None or ZL <= 0 or ZR <= 0:
            XL, ZL = _ground_intersection(uL, v, fx, fy, cx, cy, pitch_deg)
            XR, ZR = _ground_intersection(uR, v, fx, fy, cx, cy, pitch_deg)
        else:
            XL = (uL - cx) * ZL / fx
            XR = (uR - cx) * ZR / fx

        if not np.isfinite(XL) or not np.isfinite(XR):
            continue
        widths.append(abs(XR - XL))

    if len(widths) < 5:
        return WidthResult(0.0, 0.0, 0)

    widths = np.array(widths)
    width  = float(np.median(widths))
    margin = err_pct / 100 * width

    # print(dict(
    # y0=y0, y1=y1,
    # width=width, margin=margin,
    # v_h=v_h,
    # finite= np.isfinite(width),
    # band_Z= (α * depth[y0:y1].mean()) if depth is not None else None,
    # chosen_band=(y0, y1),
    # n_pixels=len(widths),
    # ))
    
    return WidthResult(width, margin, len(widths))

def bottom_percent_mask(mask: np.ndarray, percent: float = 5.0, min_pixels: int = 6) -> np.ndarray:
    """Select bottom `percent` of True pixels in mask by y-coordinate."""
    mask_bool = np.asarray(mask) != 0
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    th = np.percentile(ys, 100.0 - float(percent))
    sel = (ys >= th)
    if sel.sum() < min_pixels:
        return np.zeros_like(mask_bool, dtype=bool)
    out = np.zeros_like(mask_bool, dtype=bool)
    out[ys[sel], xs[sel]] = True
    return out

def compute_clearances(
    sidewalk:   np.ndarray,                 
    top_mask:   Tuple[float, float],          
    bot_mask:   Tuple[float, float],          
    obstacles:  Sequence[tuple[str, np.ndarray]],
    sidewalk_width_m: float,
    bottom_percent: float = 5.0,
    min_cand_pixels: int = 6,
    return_candidates: bool = False
) -> list[ClearanceResult]:
    
    results = []
    base_candidate_masks = []

    for label, omask in obstacles:
        omask_bool = omask.astype(bool)
        if omask_bool.sum() == 0:
            results.append(ClearanceResult(label, 0.0, 0.0, 0.0, None, None))
            base_candidate_masks.append(np.zeros_like(omask_bool))
            continue

        # Select candidate base pixels (bottom % of obstacle)
        overlap = omask_bool & sidewalk.astype(bool)
        cand_mask = bottom_percent_mask(overlap, bottom_percent, min_cand_pixels)
        if cand_mask.sum() == 0:
            ys, xs = np.nonzero(omask_bool)
            vmax = int(np.max(ys))
            band_threshold = max(0, vmax - 8)
            cand_mask = np.zeros_like(omask_bool)
            cand_mask[ys[ys >= band_threshold], xs[ys >= band_threshold]] = True

        base_candidate_masks.append(cand_mask.copy())

        if cand_mask.sum() == 0:
            results.append(ClearanceResult(label, 0.0, 0.0, 0.0, None, None))
            continue

        # Get leftmost and rightmost base pixels
        cand_v, cand_u = np.nonzero(cand_mask)
        left_idx = np.argmin(cand_u)
        right_idx = np.argmax(cand_u)
        L_pixel_img = (int(cand_u[left_idx]), int(cand_v[left_idx]))
        R_pixel_img = (int(cand_u[right_idx]), int(cand_v[right_idx]))
        
        # Compute clearance percentages
        xL, yL = L_pixel_img
        xR, yR = R_pixel_img
        
        # Calculate sidewalk edges at obstacle's y-position
        top_x = (yL - top_mask[1]) / top_mask[0] if abs(top_mask[0]) > 1e-5 else 0
        bot_x = (yL - bot_mask[1]) / bot_mask[0] if abs(bot_mask[0]) > 1e-5 else 0
        left_curb = min(top_x, bot_x)
        right_curb = max(top_x, bot_x)
        
        # Calculate widths in pixels
        total_width = right_curb - left_curb
        left_clearance = xL - left_curb
        right_clearance = right_curb - xR
        
        # Convert to percentages
        if total_width > 1:
            left_percent = (left_clearance / total_width)
            right_percent = (right_clearance / total_width)
        else:
            left_percent, right_percent = 0.0, 0.0

        # print(f"L: {left_percent * 100:.2f} % R: {right_percent * 100:.2f} % ")

        results.append(ClearanceResult(
            label=label,
            L_m=left_percent*sidewalk_width_m,
            R_m=right_percent*sidewalk_width_m,
            total_m=left_percent*sidewalk_width_m + right_percent*sidewalk_width_m,
            obs_width=sidewalk_width_m - (left_percent + right_percent) * sidewalk_width_m,
            L_pixel=L_pixel_img,
            R_pixel=R_pixel_img
        ))

    return (results, base_candidate_masks) if return_candidates else results

# --------------------------------------------------------------------------- #
# 3)  Internal helpers (kept private)
# --------------------------------------------------------------------------- #

def compute_width_from_curbs(
    mask: np.ndarray,
    top: tuple[float, float],
    bot: tuple[float, float],
    *,
    pitch_deg: float = -10.0,
    FOV_deg:  float = 75.0,
) -> WidthResult:
    """Mede a largura da calçada a partir das guias já refinadas."""
    H, W = mask.shape
    fx, fy, cx, cy = _intrinsics_after_crop(W, H, CROP_BOTTOM, FOV_deg)

    try:
        a1, c1 = project_line_to_ground(*top, fx, fy, cx, cy, pitch_deg)
        a2, c2 = project_line_to_ground(*bot, fx, fy, cx, cy, pitch_deg)
    except RuntimeError:
        # projeção falhou (linha acima do horizonte, etc.)
        return WidthResult(0.0, 0.0, 0)

    width  = ortho_distance(a1, c1, a2, c2)
    margin = 0.10 * width
    return WidthResult(width, margin, int(mask.sum()))


def ortho_distance(a1,c1, a2,c2):
    """
    w = |c2 - c1| / sqrt(1 + a^2)   (a1≈a2→use média)
    """
    a = 0.5*(a1 + a2)
    return abs(c2 - c1) / np.sqrt(1 + a*a)
