import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter

def read_rgbimg(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return IOError(f"Failed to load {img_path}")
    #img = img[:400-20,:] # resize img to crop out google's logo which may interfere
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 'imread' loads img as BGR so it must be converted to RGB

def rowwise_fill_sidewalk(mask: np.ndarray,
                          trim_percent: float = 10.0,
                          min_span_px: int   = 15) -> np.ndarray:
    """
    Reconstruct a continuous sidewalk mask by
      • keeping robust left/right edges per row,
      • linearly interpolating edges where rows had no data,
      • filling between those edges.

    Parameters
    ----------
    mask : H×W binary uint8.
    trim_percent : float
        Drop the outer trim_percent % of sidewalk pixels in each row
        before keeping the extrema (robust to outliers).
    min_span_px : int
        Ignore rows whose span is narrower than this (likely noise).

    Returns
    -------
    A new filled mask (uint8, 0/1).
    """
    H, W = mask.shape
    edge_left  = np.full(H, np.nan, dtype=float)
    edge_right = np.full(H, np.nan, dtype=float)

    # ------------------------------------------------------------------ gather
    rows_with = np.where(mask.any(axis=1))[0]
    if rows_with.size == 0:
        return mask.copy()

    for y in rows_with:
        xs = np.where(mask[y])[0]
        if xs.size < min_span_px:
            continue
        lo, hi = np.percentile(xs, [trim_percent, 100 - trim_percent])
        xs = xs[(xs >= lo) & (xs <= hi)]
        if xs.size < min_span_px:
            continue
        edge_left[y]  = xs.min()
        edge_right[y] = xs.max()

    valid = ~np.isnan(edge_left)
    if valid.sum() < 2:           # not enough info – fall back
        return mask.copy()

    # ------------------------------------------------------------- interpolate
    ys = np.arange(H)
    edge_left  = np.interp(ys, ys[valid], edge_left[valid])
    edge_right = np.interp(ys, ys[valid], edge_right[valid])

    # --------------------------------------------------------------- construct
    filled = np.zeros_like(mask, dtype=np.uint8)
    for y in range(rows_with.min(), rows_with.max() + 1):
        l = int(edge_left[y])
        r = int(edge_right[y])
        if r - l >= min_span_px:
            l = np.clip(l, 0, W - 1)
            r = np.clip(r, 0, W - 1)
            filled[y, l:r + 1] = 1

    return filled

def bridge_fill_between_edges(mask: np.ndarray,
                              smooth_kernel: int = 9,
                              min_valid_columns: int = 30,
                              infer_bottom: str = "interp",   # "interp" | "thickness" | "none"
                              clamp_to: int | None = None) -> np.ndarray:

    h, w = mask.shape
    clamp_to = clamp_to or (h - 1)

    top = np.full(w, -1, dtype=np.int32)
    bot = np.full(w, -1, dtype=np.int32)

    # 1) scan each column
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            top[x] = ys[0]
            bot[x] = ys[-1]

    have_both = (top >= 0) & (bot >= 0)
    if have_both.sum() < min_valid_columns:
        return mask

    # 2) smooth edges we *do* have
    top_s = median_filter(top, size=smooth_kernel)
    bot_s = median_filter(bot, size=smooth_kernel)

    # 3) fill missing bottoms
    miss_bot = (top_s >= 0) & (bot_s < 0)

    if infer_bottom == "interp":
        good_x   = np.where(bot_s >= 0)[0]
        good_bot = bot_s[good_x]
        if good_x.size >= 2:
            bot_interp = np.interp(np.arange(w), good_x, good_bot).astype(np.int32)
            bot_s[miss_bot] = np.clip(bot_interp[miss_bot], 0, clamp_to)

    elif infer_bottom == "thickness":
        thickness = bot_s[have_both] - top_s[have_both]
        med_t     = int(np.median(thickness))
        bot_s[miss_bot] = np.clip(top_s[miss_bot] + med_t, 0, clamp_to)

    # (if infer_bottom == "none": do nothing)

    # 4) rebuild polygon
    have_both = (top_s >= 0) & (bot_s >= 0)
    xs = np.arange(w)[have_both]
    pts = np.vstack([
        np.stack([xs,               top_s[have_both]], axis=1),
        np.stack([xs[::-1], bot_s[have_both][::-1]],  axis=1)
    ]).astype(np.int32)

    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, [pts], 1)

    return np.maximum(mask, filled)

def fill_between_independent_lines(mask: np.ndarray,
                                   min_cols: int = 20,
                                   ransac_thresh: float = 3.0,
                                   ransac_trials: int = 100) -> np.ndarray:
    """
    1) Extract visible top & bottom edges per column.
    2) Fit TOP = least-squares on (xs, ys_top).
    3) Fit BOTTOM = RANSAC on (xs, ys_bot).
    4) Rasterize both lines, fill polygon between them.
    """
    h, w = mask.shape
    xs, ys_top, ys_bot = [], [], []

    for x in range(w):
        ys = np.nonzero(mask[:, x])[0]
        if ys.size:
            xs.append(x)
            ys_top.append(ys[0])
            ys_bot.append(ys[-1])

    if len(xs) < min_cols:
        return mask

    xs = np.array(xs)
    ys_top = np.array(ys_top)
    ys_bot = np.array(ys_bot)

    # 2) fit top edge (LS)
    m_top, b_top = np.polyfit(xs, ys_top, 1)

    # 3) fit bottom edge (RANSAC)
    m_bot, b_bot = fit_line_ransac(xs, ys_bot,
                                   residual_thresh=ransac_thresh,
                                   max_trials=ransac_trials)

    # 4) rasterize both
    xs_full = np.arange(w)
    y_top   = (m_top * xs_full + b_top).astype(np.int32)
    y_bot   = (m_bot * xs_full + b_bot).astype(np.int32)

    # clip & ensure bottom > top+1
    y_top = np.clip(y_top, 0, h - 1)
    y_bot = np.clip(y_bot, 0, h - 1)
    y_bot = np.maximum(y_bot, y_top + 1)

    # 5) fill polygon
    pts = np.vstack([
        np.stack([xs_full,       y_top], axis=1),
        np.stack([xs_full[::-1], y_bot[::-1]], axis=1)
    ]).astype(np.int32)

    fill = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(fill, [pts], 1)
    return fill

def fit_line_ransac(xs: np.ndarray,
                    ys: np.ndarray,
                    residual_thresh: float = 3.0,
                    max_trials: int = 100) -> tuple[float,float]:
    """
    Robustly fit y = m x + b to (xs, ys) via RANSAC:
      - sample 2 points, compute candidate m,b
      - count inliers whose |yi - (m xi + b)| < residual_thresh
      - keep best consensus, then refit on all its inliers
    """
    best_inliers = np.array([], dtype=int)
    best_params = (0.0, 0.0)

    for _ in range(max_trials):
        # pick two random distinct indices
        i, j = np.random.choice(len(xs), 2, replace=False)
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[j], ys[j]
        if x1 == x0:
            continue
        m_cand = (y1 - y0) / (x1 - x0)
        b_cand = y0 - m_cand * x0

        residuals = np.abs(ys - (m_cand * xs + b_cand))
        inliers = np.nonzero(residuals < residual_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = (m_cand, b_cand)

    # final least-squares fit on all inliers
    if len(best_inliers) >= 2:
        m, b = np.polyfit(xs[best_inliers], ys[best_inliers], 1)
    else:
        m, b = best_params

    return m, b

def refine_sidewalk_mask(
    mask: np.ndarray,
    bands: list[tuple[float, float, int]] | None = None,
    kernel_height: int       = 5,
    close_iter: int          = 1,
    max_gap_x: int           = 24,
    max_gap_y: int           = 6,
    min_keep_area_px: int    = 5_000,
    bf_kwargs: dict | None   = None,   # bridge-fill params
    pl_kwargs: dict | None   = None,   # parallel-line fill params
) -> np.ndarray:
    """
    Full pipeline:
      1) band-wise closing
      2) anisotropic bounding
      3) bridge-fill large occluders
      4) hole fill + speckle removal
      5) fit & fill between two straight boundary lines
    Returns a binary mask (0/1).
    """
    # ─── 0) prepare defaults & binarize ──────────────────────────────────────
    h, w = mask.shape
    m0 = (mask > 0).astype(np.uint8)
    m  = np.zeros_like(m0)

    if bands is None:
        bands = [
            (0.00, 0.35, 25),   # bottom 35%
            (0.35, 0.70, 15),   # mid 35%
            (0.70, 1.00,  9)    # top 30%
        ]
    bf_kwargs = bf_kwargs or {
        "smooth_kernel":     5,
        "min_valid_columns": 2,
        "infer_bottom":      "interp",
        "clamp_to":          h - 30
    }
    pl_kwargs = pl_kwargs or {
        "min_cols":      20,
        "lower_offset":   2.0
    }

    # ─── 1) band-wise closing ─────────────────────────────────────────────────
    for y0f, y1f, kx in bands:
        y0, y1 = int(h * y0f), int(h * y1f)
        if y1 <= y0:
            continue
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, kernel_height))
        m[y0:y1] = cv2.morphologyEx(
            m0[y0:y1], cv2.MORPH_CLOSE, ker, iterations=close_iter
        )

    # ─── 2) anisotropic bounding ──────────────────────────────────────────────
    dil = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2*max_gap_x + 1, 2*max_gap_y + 1)
    )
    allowed = cv2.dilate(m0, dil)
    m &= allowed

    # ─── 3) bridge-fill large occluders ──────────────────────────────────────
    m = bridge_fill_between_edges(m, **bf_kwargs)

    # ─── 4) hole fill + remove tiny blobs ────────────────────────────────────
    m = ndimage.binary_fill_holes(m.astype(bool)).astype(np.uint8)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cleaned = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_keep_area_px:
            cleaned[lbl == i] = 1

    # ─── 5) fit & fill between straight lines ────────────────────────────────
    final = fill_between_independent_lines(
    cleaned, 
    min_cols=15,         # lower if large occluders block many columns
    ransac_thresh=4.0,   # tolerance in pixels for RANSAC inliers
    ransac_trials=200    # more trials → more chance to find the true curb
    )

    return final

def fuse_sidewalk_masks(mask1, mask2, method="or"):
    """
    Fuse two sidewalk masks using logical operators OR or AND.
    
    Args:
        mask1, mask2: Binary mask Máscara binária do OneFormer (H×W)
        method: "or" (union) ou "and" (intersection)
    
    Returns:
        Fused mask (H×W uint8)
    """
    import numpy as np
    
    # Ensure that both masks have the same size (HxW)
    if mask1.shape != mask2.shape:
        print(f"Warning: Mask shapes differ - 1: {mask1.shape}, 2: {mask2.shape}")
        # Resize to match
        from PIL import Image
        mask2_pil = Image.fromarray(mask2.astype(np.uint8))
        mask2_pil = mask2_pil.resize(
            (mask1.shape[1], mask2.shape[0]), 
            Image.NEAREST
        )
        mask2 = np.array(mask2_pil)
    
    # Convert into boolean for logical operations
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    # Apply fusion
    if method.lower() == "or":
        fused_mask = np.logical_or(mask1_bool, mask2_bool)
    elif method.lower() == "and":
        fused_mask = np.logical_and(mask1_bool, mask2_bool)
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'or' or 'and'.")
    
    # DEBUG: Show fusion statistics
    print(f"Mask1 pixels: {np.sum(mask1_bool)}")
    print(f"Mask2 pixels: {np.sum(mask2_bool)}")
    print(f"Fused ({method}) pixels: {np.sum(fused_mask)}")
    
    return fused_mask.astype(np.uint8)