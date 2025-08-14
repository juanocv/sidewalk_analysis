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
    sidewalk:   np.ndarray,                 # bool H×W
    top_mask:   Tuple[float, float],          
    bot_mask:   Tuple[float, float],          
    obstacles:  Sequence[tuple[str, np.ndarray]],   # (label, bool-mask)
    depth:      np.ndarray,
    pitch_deg:  float = -10.0,
    FOV_deg:    float = 75.0,
    bottom_percent: float = 5.0,
    min_cand_pixels: int = 6,
    return_candidates: bool = False
) -> list[ClearanceResult]:

    H, W = sidewalk.shape[:2]
    fx, fy, cx, cy = _intrinsics_after_crop(W, H, CROP_BOTTOM, FOV_deg)

    a_top, c_top = project_line_to_ground(*top_mask, fx, fy, cx, cy, pitch_deg)
    a_bot, c_bot = project_line_to_ground(*bot_mask, fx, fy, cx, cy, pitch_deg)

    denom_top = math.sqrt(a_top * a_top + 1.0)
    denom_bot = math.sqrt(a_bot * a_bot + 1.0)

    alpha = 1.0
    if depth is not None:
        alpha = _scale_from_ground(sidewalk, depth, fx, fy, cx, cy)

    results = []
    base_candidate_masks = []

    def project_pixel_to_ground(u_px, v_px):
        if depth is not None and depth[v_px, u_px] > 0:
            Zm = depth[v_px, u_px] * float(alpha)
            Xm = (u_px - cx) * Zm / fx
            return Xm, Zm
        else:
            Xg, Zg = _ground_intersection(
                np.array([u_px]), np.array([v_px]), fx, fy, cx, cy, pitch_deg
            )
            return Xg[0], Zg[0]

    for label, omask in obstacles:
        omask_bool = omask.astype(bool)
        if omask_bool.sum() == 0:
            results.append(ClearanceResult(label, 0.0, 0.0, 0.0, None, None))
            base_candidate_masks.append(np.zeros_like(omask_bool))
            continue

        # Candidate mask = bottom % of overlap with sidewalk
        overlap = omask_bool & sidewalk.astype(bool)
        cand_mask = np.zeros_like(omask_bool)
        if overlap.sum() > 0:
            cand_mask = bottom_percent_mask(overlap, bottom_percent, min_cand_pixels)
        if cand_mask.sum() == 0:
            # Fallback: bottom-most band of obstacle
            ys, xs = np.nonzero(omask_bool)
            vmax = int(np.max(ys))
            band_threshold = max(0, vmax - 8 + 1)  # default 8px band
            band_idx = (ys >= band_threshold)
            cand_mask[ys[band_idx], xs[band_idx]] = True

        base_candidate_masks.append(cand_mask.copy())

        if cand_mask.sum() == 0:
            results.append(ClearanceResult(label, 0.0, 0.0, 0.0, None, None))
            continue

        # Pick leftmost and rightmost pixels in candidate base
        cand_v, cand_u = np.nonzero(cand_mask)
        left_idx = np.argmin(cand_u)
        right_idx = np.argmax(cand_u)
        L_pixel_img = (int(cand_u[left_idx]), int(cand_v[left_idx]))
        R_pixel_img = (int(cand_u[right_idx]), int(cand_v[right_idx]))

        # Project them to ground
        LX, LZ = project_pixel_to_ground(*L_pixel_img)
        RX, RZ = project_pixel_to_ground(*R_pixel_img)

        # Distances from each pixel to each curb
        dL_top = abs(a_top * LX - LZ + c_top) / denom_top
        dL_bot = abs(a_bot * LX - LZ + c_bot) / denom_bot
        dR_top = abs(a_top * RX - RZ + c_top) / denom_top
        dR_bot = abs(a_bot * RX - RZ + c_bot) / denom_bot

        # Which curb is left/right? Compare curb X positions at median Z
        Zv_med = float(np.median([LZ, RZ]))
        def curb_X(a, c, Zq):
            return (Zq - c) / a if abs(a) > 1e-9 else float("inf")
        Xtop_med = curb_X(a_top, c_top, Zv_med)
        Xbot_med = curb_X(a_bot, c_bot, Zv_med)

        if Xtop_med <= Xbot_med:
            L_m = dL_top
            R_m = dR_bot
        else:
            L_m = dL_bot
            R_m = dR_top

        obs_width = abs(RX - LX)
        total_m = L_m + R_m + obs_width

        results.append(ClearanceResult(label, L_m, R_m, total_m, L_pixel_img, R_pixel_img))

    if return_candidates:
        return results, base_candidate_masks
    return results


def plot_clearance_overlay_debug(
    image: np.ndarray,
    top: Tuple[float, float],
    bot: Tuple[float, float],
    obstacles: Sequence[tuple[str, np.ndarray]],
    clearance_results: Sequence,
    sidewalk: Optional[np.ndarray] = None,
    base_candidate_masks: Optional[Sequence[np.ndarray]] = None,
    BASE_BAND_PIXELS: int = 1,
    assume_rgb: bool = True,
    figsize=(12,8)
):
    """
    More advanced overlay for debugging:
      - draws candidate base components (magenta)
      - draws chosen L_pixel (yellow) and R_pixel (cyan)
      - draws perpendicular line from chosen pixel to image curb line (white)
      - shows labels and values
    If base_candidate_masks is None the function recomputes candidates with the same
    rules used in compute_clearances (so you can call it even if you didn't request
    candidates to be returned).
    """
    H, W = image.shape[:2]
    if assume_rgb:
        draw_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    else:
        draw_img = image.copy()

    # draw curb lines (image-space)
    def draw_line_image(m, b, color, thickness=2):
        x0, x1 = 0, W-1
        y0 = int(round(m * x0 + b))
        y1 = int(round(m * x1 + b))
        cv2.line(draw_img, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)

    draw_line_image(top[0], top[1], (0,0,255), 2)   # red
    draw_line_image(bot[0], bot[1], (0,255,0), 2)   # green

    for idx, ((label, omask), res) in enumerate(zip(obstacles, clearance_results)):
        om = omask.astype(bool)
        if om.sum() == 0:
            continue

        # shade obstacle
        #blue = np.array([255,0,0], dtype=np.uint8)  # BGR
        #alpha = 0.45
        #draw_img[om] = (draw_img[om].astype(np.float32)*(1-alpha) + blue.astype(np.float32)*alpha).astype(np.uint8)

        # get or compute candidate mask
        if base_candidate_masks is not None:
            try:
                cand_mask = base_candidate_masks[idx].astype(bool)
            except Exception:
                cand_mask = None
        else:
            cand_mask = None

        if cand_mask is None or cand_mask.sum() == 0:
            # compute same rule as compute_clearances: overlap with sidewalk if available, else bottom band
            cand_mask = np.zeros_like(om)
            if sidewalk is not None:
                overlap = om & sidewalk
                if overlap.sum() > 0:
                    cand_mask = overlap.copy()
                else:
                    ys, xs = np.nonzero(om)
                    vmax = int(np.max(ys))
                    band_threshold = max(0, vmax - BASE_BAND_PIXELS + 1)
                    band_mask = (ys >= band_threshold)
                    cand_mask[ys[band_mask], xs[band_mask]] = True
            else:
                ys, xs = np.nonzero(om)
                vmax = int(np.max(ys))
                band_threshold = max(0, vmax - BASE_BAND_PIXELS + 1)
                band_mask = (ys >= band_threshold)
                cand_mask[ys[band_mask], xs[band_mask]] = True

        # optional: compute components and draw outlines for each component
        if cand_mask.sum() > 0:
            # find components
            num, comp = cv2.connectedComponents(cand_mask.astype(np.uint8))
            for cidx in range(1, num):
                comp_mask = (comp == cidx)
                # fill with semi-transparent magenta
                mag = np.array([255,0,255], dtype=np.uint8)
                beta = 0.65
                draw_img[comp_mask] = (draw_img[comp_mask].astype(np.float32)*(1-beta) + mag.astype(np.float32)*beta).astype(np.uint8)
                # draw contour outline
                # contours expect uint8 uint8 image
                cont_img = (comp_mask.astype(np.uint8)*255)
                cnts, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(draw_img, cnts, -1, (255,255,255), 1, lineType=cv2.LINE_AA)

        # draw chosen pixels
        if hasattr(res, 'L_pixel') and res.L_pixel is not None:
            xL, yL = int(res.L_pixel[0]), int(res.L_pixel[1])
            cv2.circle(draw_img, (xL, yL), 3, (0,255,255), -1, lineType=cv2.LINE_AA)  # yellow
            cv2.circle(draw_img, (xL, yL), 3, (255,255,255), 1, lineType=cv2.LINE_AA)

            # compute perpendicular projection point on image curb line (top and bot are image-space)
            # pick the line that corresponds to 'left' depending on L_pixel being result of top/bot
            # we will draw perpendiculars to both curbs for clarity
            for m,b,color in ((top[0], top[1], (255,255,255)), (bot[0], bot[1], (200,200,200))):
                # project (xL,yL) to nearest point on image line y = m*x + b:
                if abs(m) < 1e-9:
                    # horizontal line; clamp x to xL
                    xproj = xL
                else:
                    xproj = (xL + m*(yL - b)) / (1 + m*m)
                yproj = int(round(m * xproj + b))
                xproj = int(round(xproj))
                # draw small line
                cv2.line(draw_img, (xL, yL), (xproj, yproj), color, 2, lineType=cv2.LINE_AA)

        if hasattr(res, 'R_pixel') and res.R_pixel is not None:
            xR, yR = int(res.R_pixel[0]), int(res.R_pixel[1])
            cv2.circle(draw_img, (xR, yR), 3, (255,255,0), -1, lineType=cv2.LINE_AA)  # cyan
            cv2.circle(draw_img, (xR, yR), 3, (255,255,255), 1, lineType=cv2.LINE_AA)

            for m,b,color in ((top[0], top[1], (255,255,255)), (bot[0], bot[1], (200,200,200))):
                if abs(m) < 1e-9:
                    xproj = xR
                else:
                    xproj = (xR + m*(yR - b)) / (1 + m*m)
                yproj = int(round(m * xproj + b))
                xproj = int(round(xproj))
                cv2.line(draw_img, (xR, yR), (xproj, yproj), color, 2, lineType=cv2.LINE_AA)

        # write label
        ys, xs = np.nonzero(om)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        Ls = f"{getattr(res, 'L_m', 0.0):.2f}"
        Rs = f"{getattr(res, 'R_m', 0.0):.2f}"
        text = f"{label}: L={Ls}m R={Rs}m"
        text_pos = (max(0, cx - 40), max(0, cy - 10))
        cv2.putText(draw_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)

    # convert back to RGB for display
    if assume_rgb:
        disp = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    else:
        disp = draw_img

    plt.figure(figsize=figsize)
    plt.imshow(disp)
    plt.axis('off')
    plt.show()
    return disp


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
