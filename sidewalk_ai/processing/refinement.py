# sidewalk_ai/processing/refinement.py
"""
Morphological + geometric post-processing utilities for sidewalk masks.

All functions are **pure**: they accept/return numpy arrays (dtype=bool or
uint8) and never touch disk, logging, or global state.  That keeps them
fast to unit-test and easy to reuse in batch jobs, the CLI, or FastAPI.
"""
from __future__ import annotations

from typing import Tuple, List

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter


# --------------------------------------------------------------------------- #
# 0)  Row-wise edge interpolation  (≈ original rowwise_fill_sidewalk)         #
# --------------------------------------------------------------------------- #
def rowwise_fill_sidewalk(
    mask: np.ndarray,
    trim_percent: float = 10.0,
    min_span_px: int = 15,
) -> np.ndarray:
    """
    Reconstruct a continuous sidewalk band by

    1. keeping robust left/right edges per *row*;
    2. linearly interpolating missing rows;
    3. filling the area between the two curves.

    Parameters
    ----------
    mask
        H×W **bool** (or 0/1 uint8) raw mask.
    trim_percent
        Ignore the outer N % of pixels in each row when computing edges
        (robust vs. outliers).
    min_span_px
        Minimum width a row must have to be considered valid.

    Returns
    -------
    filled : H×W uint8   – 1 = sidewalk, 0 = background.
    """
    H, W = mask.shape
    mask = mask.astype(bool)

    edge_L = np.full(H, np.nan)
    edge_R = np.full(H, np.nan)

    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return mask.astype(np.uint8)

    for y in rows:
        xs = np.where(mask[y])[0]
        if xs.size < min_span_px:
            continue
        lo, hi = np.percentile(xs, [trim_percent, 100 - trim_percent])
        xs = xs[(xs >= lo) & (xs <= hi)]
        if xs.size < min_span_px:
            continue
        edge_L[y] = xs.min()
        edge_R[y] = xs.max()

    good = ~np.isnan(edge_L)
    if good.sum() < 2:                # too little evidence → bail
        return mask.astype(np.uint8)

    ys = np.arange(H)
    edge_L = np.interp(ys, ys[good], edge_L[good])
    edge_R = np.interp(ys, ys[good], edge_R[good])

    out = np.zeros_like(mask, dtype=np.uint8)
    y0, y1 = rows.min(), rows.max()
    for y in range(y0, y1 + 1):
        l, r = int(edge_L[y]), int(edge_R[y])
        if r - l >= min_span_px:
            out[y, max(0, l): min(W, r + 1)] = 1
    return out


# --------------------------------------------------------------------------- #
# 1)  Bridge-fill across large occluders  (≈ bridge_fill_between_edges)       #
# --------------------------------------------------------------------------- #
def bridge_fill_between_edges(
    mask: np.ndarray,
    smooth_kernel: int = 9,
    min_valid_cols: int = 30,
    infer_bottom: str = "interp",      # "interp" | "thickness" | "none"
    clamp_to: int | None = None,
) -> np.ndarray:
    """
    Close big gaps (cars, bushes…) by estimating a top & bottom edge for every
    column, then filling the polygon between them :contentReference[oaicite:1]{index=1}.

    * `infer_bottom="interp"` – interpolate missing bottom edge;
    * `infer_bottom="thickness"` – assume median thickness;
    * `infer_bottom="none"` – leave columns without bottom edge untouched.
    """
    h, w = mask.shape
    mask = mask.astype(bool)
    clamp_to = clamp_to or (h - 1)

    top = np.full(w, -1, dtype=int)
    bot = np.full(w, -1, dtype=int)
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            top[x], bot[x] = ys[0], ys[-1]

    have = (top >= 0) & (bot >= 0)
    if have.sum() < min_valid_cols:
        return mask.astype(np.uint8)

    top_s = median_filter(top, size=smooth_kernel)
    bot_s = median_filter(bot, size=smooth_kernel)

    miss_bottom = (top_s >= 0) & (bot_s < 0)
    if miss_bottom.any():
        if infer_bottom == "interp":
            good_x = np.where(bot_s >= 0)[0]
            bot_interp = np.interp(np.arange(w), good_x, bot_s[good_x])
            bot_s[miss_bottom] = np.clip(bot_interp[miss_bottom], 0, clamp_to)
        elif infer_bottom == "thickness":
            thick = bot_s[have] - top_s[have]
            med_t = int(np.median(thick))
            bot_s[miss_bottom] = np.clip(top_s[miss_bottom] + med_t, 0, clamp_to)

    have = (top_s >= 0) & (bot_s >= 0)
    xs = np.arange(w)[have]
    pts = np.vstack(
        [
            np.stack([xs, top_s[have]], axis=1),
            np.stack([xs[::-1], bot_s[have][::-1]], axis=1),
        ]
    ).astype(np.int32)

    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, [pts], 1)
    return np.maximum(mask.astype(np.uint8), filled)


# --------------------------------------------------------------------------- #
# 2)  Fill between two almost-parallel straight lines                         #
# --------------------------------------------------------------------------- #
def fit_line_ransac(
    xs: np.ndarray,
    ys: np.ndarray,
    thresh_px: float = 3.0,
    max_trials: int = 100,
) -> Tuple[float, float]:
    """Robust y = m·x + b fit (RANSAC)."""
    best_inliers: np.ndarray = np.empty(0, dtype=int)
    best_params = (0.0, 0.0)

    for _ in range(max_trials):
        i, j = np.random.choice(len(xs), 2, replace=False)
        if xs[j] == xs[i]:
            continue
        m = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - m * xs[i]
        resid = np.abs(ys - (m * xs + b))
        inl = np.nonzero(resid < thresh_px)[0]
        if inl.size > best_inliers.size:
            best_inliers = inl
            best_params = (m, b)

    if best_inliers.size >= 2:
        return tuple(np.polyfit(xs[best_inliers], ys[best_inliers], 1))
    return best_params


def fill_between_independent_lines(
    mask: np.ndarray,
    min_cols: int = 20,
    ransac_thresh: float = 3.0,
    ransac_trials: int = 100,
    return_lines: bool = False,
) -> np.ndarray:
    """
    1. Extract visible top & bottom edges per **column**;
    2. Fit TOP edge (least-squares), BOTTOM edge (RANSAC) :contentReference[oaicite:2]{index=2};
    3. Rasterise both lines and fill the polygon between them.
    """
    h, w = mask.shape
    mask = mask.astype(bool)

    xs, ys_top, ys_bot = [], [], []
    for x in range(w):
        ys = np.nonzero(mask[:, x])[0]
        if ys.size:
            xs.append(x)
            ys_top.append(ys[0])
            ys_bot.append(ys[-1])

    if len(xs) < min_cols:
        return mask.astype(np.uint8)

    xs = np.asarray(xs)
    ys_top = np.asarray(ys_top)
    ys_bot = np.asarray(ys_bot)

    thickness = ys_bot - ys_top
    med_t = np.median(thickness)

    # ---------- FILTRO 1: espessura mínima ----------------------
    good = thickness >= 0.60 * med_t

    # ---------- FILTRO 2: descartar colunas no rodapé -----------
    margin = 2                                  # px de folga
    good &= ys_bot < h - 1 - margin

    xs_fit     = xs[good]
    ys_fit_bot = ys_bot[good]

    if xs_fit.size < min_cols:
        return mask.astype(np.uint8)            # falha segura (bot_line=None)

    m_top, b_top = np.polyfit(xs, ys_top, 1) # no need to use RANSAC for the top line
    m_bot, b_bot = fit_line_ransac(xs_fit, ys_fit_bot, ransac_thresh, ransac_trials)

    xs_full = np.arange(w)
    y_top = np.clip((m_top * xs_full + b_top).astype(int), 0, h - 1)
    y_bot = np.clip((m_bot * xs_full + b_bot).astype(int), 0, h - 1)
    y_bot = np.maximum(y_bot, y_top + 1)  # ensure bottom below top

    pts = np.vstack(
        [
            np.stack([xs_full, y_top], axis=1),
            np.stack([xs_full[::-1], y_bot[::-1]], axis=1),
        ]
    ).astype(np.int32)

    fill = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(fill, [pts], 1)
    # opcional: se quiser preservar a máscara original, troque por
    # filled = np.maximum(mask.astype(np.uint8), fill)
    filled = fill

    # (m, b) no sistema imagem:  y = m·x + b
    lines = ((m_top, b_top), (m_bot, b_bot))
    return (filled, lines) if return_lines else filled

def shave_above_top_envelope(
    m: np.ndarray,
    max_above_px: int | None = None,
    smooth_kernel: int = 15,
    min_cols: int = 30,
) -> np.ndarray:
    """
    Zera pixels acima do invólucro superior (top envelope) do cluster
    original, com uma margem de segurança.

    - Se max_above_px=None, a margem é adaptativa: ~8% da espessura mediana.
    """
    h, w = m.shape
    m = (m > 0).astype(np.uint8)

    # top/bottom por coluna
    top = np.full(w, -1, dtype=int)
    bot = np.full(w, -1, dtype=int)
    for x in range(w):
        ys = np.flatnonzero(m[:, x])
        if ys.size:
            top[x], bot[x] = ys[0], ys[-1]

    have = (top >= 0) & (bot >= 0)
    if have.sum() < min_cols:
        return m  # pouco sinal → devolve como está

    # suaviza o contorno superior para robustez
    top_s = median_filter(top, size=smooth_kernel)

    # margem: absoluta (se dada) ou adaptativa pela espessura mediana
    if max_above_px is None:
        thick = (bot[have] - top[have]).astype(float)
        med_t = np.median(thick)
        margin_t = max(2, int(0.08 * med_t))  # ~8% da espessura típica
    else:
        margin_t = int(max(1, max_above_px))

    out = m.copy()
    for x in range(w):
        if top_s[x] >= 0:
            y_cut = max(0, top_s[x] - margin_t)
            out[:y_cut, x] = 0
    return out

# --------------------------------------------------------------------------- #
# 3)  High-level pipeline  (≈ refine_sidewalk_mask)                           #
# --------------------------------------------------------------------------- #
def refine_sidewalk_mask(
    mask: np.ndarray,
    bands: List[Tuple[float, float, int]] | None = None,
    kernel_height: int = 5,
    close_iter: int = 1,
    max_gap_x: int = 24,
    max_gap_y: int = 6,
    min_keep_area_px: int = 5_000,
    bf_kwargs: dict | None = None,
    pl_kwargs: dict | None = None,
) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Composite refinement used by the original prototype :contentReference[oaicite:3]{index=3}:

        1. band-wise closing,
        2. anisotropic dilation bound,
        3. bridge-fill across large occluders,
        4. hole-fill + speckle removal,
        5. straight-line (parallel curb) infill.

    All steps are parameterised so you can tune behaviour per dataset.
    """
    h, w = mask.shape
    m0 = mask.astype(bool)
    m = np.zeros_like(m0, dtype=np.uint8)

    print(f"Refining sidewalk mask: {m0.sum()} px positive")
    cv2.imwrite("debug_0_raw.png", (m0 * 255).astype(np.uint8))

    # 0) shave above top envelope (remove overhanging patches, etc)
    m0 = shave_above_top_envelope(
        m0.astype(np.uint8),
        max_above_px=None,        # adaptative (~8% thickness)
        smooth_kernel=15,
        min_cols=30,
    ).astype(bool)

    print(f" After shaving: {m0.sum()} px positive")
    cv2.imwrite("debug_1_shaved.png", (m0 * 255).astype(np.uint8))

    # 1) band-wise closing (deals with perspective foreshortening)
    if bands is None:
        bands = [(0.00, 0.35, 25), (0.35, 0.70, 15), (0.70, 1.00, 9)]
    for y0f, y1f, kx in bands:
        y0, y1 = int(h * y0f), int(h * y1f)
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, kernel_height))
        m[y0:y1] = cv2.morphologyEx(
            m0[y0:y1].astype(np.uint8), cv2.MORPH_CLOSE, ker, iterations=close_iter
        )

    print(f" After band-wise closing: {m.sum()} px positive")
    cv2.imwrite("debug_2_bandclosed.png", (m * 255).astype(np.uint8))
                
    # 2) anisotropic bound: dilate raw mask, then keep only what intersects band
    dil = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * max_gap_x + 1, 2 * max_gap_y + 1)
    )
    allowed = cv2.dilate(m0.astype(np.uint8), dil)
    m &= allowed

    print(f" After anisotropic dilation bound: {m.sum()} px positive")
    cv2.imwrite("debug_3_dilbound.png", (m * 255).astype(np.uint8))

    # 3) bridge-fill car/bush occlusions
    m = bridge_fill_between_edges(
        m,
        **(bf_kwargs or dict(smooth_kernel=5, min_valid_cols=2, clamp_to=h - 30)),
    )

    print(f" After bridge-fill: {m.sum()} px positive")
    cv2.imwrite("debug_4_bridgefill.png", (m * 255).astype(np.uint8))

    # 4) hole fill + remove tiny speckles
    m = ndimage.binary_fill_holes(m.astype(bool)).astype(np.uint8)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_keep_area_px:
            keep[lbl == i] = 1

    print(f" After hole-fill + speckle removal: {keep.sum()} px positive")
    cv2.imwrite("debug_5_holefill.png", (keep * 255).astype(np.uint8))
                
    # 5) two-line infill (parallel curbs)
    mask, (top_line, bot_line) = fill_between_independent_lines(
        keep,
        **(pl_kwargs or dict(min_cols=20, ransac_thresh=4.0, ransac_trials=200)),
        return_lines=True,
    )

    print(f" After two-line infill: {mask.sum()} px positive")
    cv2.imwrite("debug_6_twoline.png", (mask * 255).astype(np.uint8))

    return mask, (top_line, bot_line)


# --------------------------------------------------------------------------- #
# NOTE:  mask-fusion (`fuse_sidewalk_masks`) is now in `processing/fusion.py` #
# --------------------------------------------------------------------------- #