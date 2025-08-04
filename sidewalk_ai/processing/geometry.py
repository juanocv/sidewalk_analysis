# sidewalk_ai/processing/geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


def _safe_col_at_row(m_img, b_img, row, *, mask_row, side, W):
    """
    Devolve a coluna (int) que melhor representa a guia no 'row':
    1. Calcula u0 = intersecção reta × linha.
    2. Se u0 está dentro do quadro e mask_row[u0]==1 → OK.
    3. Caso contrário, procura o pixel 1 mais próximo **dentro do quadro**
       caminhando para dentro (→ se 'side' == "right", ← se "left").
       Se não encontrar, devolve None.
    """
    if abs(m_img) < 1e-6:                    # reta horizontal → faz skip
        return None

    u0 = int(round((row - b_img) / m_img))
    u0 = np.clip(u0, 0, W - 1)               # ainda dentro dos limites

    if mask_row[u0]:                         # bateu direto na calçada
        return u0

    if side == "left":
        # anda para a direita até achar um pixel 1
        for u in range(u0 + 1, W):
            if mask_row[u]:
                return u
    else:  # "right"
        # anda para a esquerda
        for u in range(u0 - 1, -1, -1):
            if mask_row[u]:
                return u
    return None                              # nada encontrado
    

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