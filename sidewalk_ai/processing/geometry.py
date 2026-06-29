# sidewalk_ai/processing/geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt

import cv2, os
import numpy as np

from sidewalk_ai.log import debug_enabled, debug_event, get_logger

logger = get_logger(__name__)


def _swai_log(tag, payload):
    if debug_enabled():
        debug_event(logger, tag, payload)


# ------------------ Camera and image parameters ------------------ #
ORIG_SIZE = (600, 400)  # Street-View static API
CROP_BOTTOM = 0  # logo strip that we remove
CAM_HEIGHT_M = 1.75

# --------------------------------------------------------------------------- #
# 0) Typed results – easy to validate/serialize
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class WidthResult:
    width_m: float  # best estimate
    margin_m: float  # ± error band (25 % default)
    n_pixels: int  # inlier count used in the fit


@dataclass(slots=True, frozen=True)
class ClearanceResult:
    label: str
    L_m: float  # free space on the left  of the obstacle
    R_m: float  # free space on the right
    total_m: float  # L+R+obstacle width
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
    xs = np.linspace(0, ORIG_SIZE[0] - 1, 50)
    ys = m * xs + b
    X, Z = _ground_intersection(xs, ys, fx, fy, cx, cy, pitch_deg)
    good = np.isfinite(X) & np.isfinite(Z)
    if good.sum() < 10:
        raise RuntimeError("curb line proj. failed")
    # Ajuste Z = a·X + c em coordenadas do solo
    a, c = np.polyfit(X[good], Z[good], 1)
    return a, c  # forma   Z = a·X + c


def _scale_from_ground(
    sidewalk,
    depth,
    fx,
    fy,
    cx,
    cy,
    H_cam=CAM_HEIGHT_M,
    rows_from_bottom=20,
    RANSAC_N=10_000,
    min_inliers=200,
) -> float:
    """α robusto ─ usa só as últimas `rows_from_bottom` linhas da calçada."""
    H, _ = sidewalk.shape
    band = np.arange(max(0, H - rows_from_bottom), H)
    ys, xs = np.where(sidewalk[band])
    if xs.size < min_inliers:
        return 1.0  # deixa passar em branco

    ys = ys + band[0]  # re-alinha índice
    rng = np.random.default_rng(0)
    sel = rng.choice(xs.size, size=min(RANSAC_N, xs.size), replace=False)
    u, v = xs[sel], ys[sel]
    Zr = depth[v, u].astype(np.float32)

    Xr = (u - cx) * Zr / fx
    Yr = (v - cy) * Zr / fy
    P = np.column_stack([Xr, Yr, Zr])

    # threshold = 2·MAD
    mad = 1.4826 * np.median(np.abs(Zr - np.median(Zr)))
    eps = 2.0 * max(mad, 0.01)

    best_cnt = 0
    best_d = 1.0
    for _ in range(2000):
        a, b, c = P[rng.choice(P.shape[0], 3, replace=False)]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            continue
        n /= n_norm
        d = -np.dot(n, a)
        cnt = np.count_nonzero(np.abs(P @ n + d) < eps)
        if cnt > best_cnt:
            best_cnt, best_d = cnt, d
            if cnt > 0.15 * P.shape[0]:
                break

    if best_cnt < min_inliers:  # falhou → neutro
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


def _ground_intersection(u, v, fx, fy, cx, cy, pitch_deg=0.0, H_cam=CAM_HEIGHT_M):
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


def _intrinsics_after_crop(
    W: int = 600, fov_deg: float = 90.0
) -> tuple[float, float, float, float]:
    """
    Returns fx, fy, cx, cy *in cropped coordinates* but referenced to the
    original optical centre (cy = 200 px).
    """
    fx = W / (2 * np.tan(np.radians(fov_deg / 2)))
    fy = fx
    cx = W / 2
    cy_orig = ORIG_SIZE[1] / 2  # 200.0
    cy = cy_orig  # same row survives the crop
    return fx, fy, cx, cy


def _has_two_curbs(
    mask: np.ndarray,
    min_gap_px: int = 50,
    min_frac_rows: float = 0.3,
    band: tuple[float, float] = (0.3, 0.9),
) -> bool:
    H, W = mask.shape
    y0 = int(H * band[0])
    y1 = int(H * band[1])
    if y1 <= y0:
        return False

    rows = range(y0, y1)
    gap_mids = []

    for y in rows:
        xs = np.where(mask[y])[0]
        if xs.size < 2:
            continue
        xs = np.sort(xs)
        diffs = np.diff(xs)
        j = int(np.argmax(diffs))
        if diffs[j] > min_gap_px:
            gap_mids.append(0.5 * (xs[j] + xs[j + 1]))

    if not gap_mids:
        return False

    # gap tem que ser frequente (persistente)
    frac = len(gap_mids) / max(1, len(list(rows)))
    if frac < min_frac_rows:
        return False

    # gap tem que estar perto do centro da imagem
    mid_gap = float(np.median(gap_mids))
    if abs(mid_gap - W / 2) > 0.15 * W:
        return False

    # e as duas “calçadas” têm que ficar de lados opostos do centro
    cols = np.unique(np.where(mask)[1])
    cols = np.sort(cols)
    diffs = np.diff(cols)
    k = int(np.argmax(diffs))
    left_cols = cols[: k + 1]
    right_cols = cols[k + 1 :]
    if not (left_cols.size and right_cols.size):
        return False

    c1 = left_cols.mean()
    c2 = right_cols.mean()
    return (c1 < W / 2) and (c2 > W / 2)


def aggregate_headings(widths_per_heading, min_k=5):
    """
    widths_per_heading: lista de floats (uma por heading válida)
    Retorna (width, low, high) com recorte por MAD.
    """
    w = np.array([x for x in widths_per_heading if np.isfinite(x) and x > 0], float)
    if w.size < min_k:
        return 0.0, 0.0, 0.0
    med = np.median(w)
    mad = np.median(np.abs(w - med)) + 1e-6
    keep = np.abs(w - med) <= 2.5 * 1.4826 * mad  # ~99% para normal
    w2 = w[keep] if keep.any() else w
    med2 = float(np.median(w2))
    iqr2 = float(np.percentile(w2, 75) - np.percentile(w2, 25))
    # intervalo de confiança simples
    lo = med2 - 0.5 * iqr2
    hi = med2 + 0.5 * iqr2
    return med2, lo, hi


# --------------------------------------------------------------------------- #
# 2)  Public API – what the pipeline will call
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _WidthDebugRow:
    v: int
    uL: int
    uR: int
    du: int
    width_geom: Optional[float] = None
    width_depth: Optional[float] = None
    near_perp: bool = False
    skipped_du: bool = False
    from_retry: bool = False
    bridged_gap: bool = False
    used_depth: bool = False
    parallax: Optional[float] = None


def compute_width(
    sidewalk: np.ndarray,  # bool mask (H×W)
    depth: np.ndarray | None = None,
    pitch_deg: float = -10.0,
    fov_deg: float = 90.0,
    *,  # keyword-only
    band_frac: tuple[float, float] = (0.50, 1.00),
    err_pct: float = 25.0,
    # --- knobs (quality gating & robustness) ---
    band_mode: str = "adaptive",  # "adaptive" | "fixed"
    adaptive_pct: tuple[float, float] = (0.60, 0.95),
    du_range_px: tuple[int, int] = (20, 220),  # ↓ mais estrito
    parallax_range: tuple[float, float] = (0.05, 0.45),  # ↓ mais estrito
    min_valid_rows: int = 7,  # ↑
    use_data_driven_margin: bool = True,
    divergence_pct: float = 0.25,  # ↓
    bottom_ignore_px: int = 20,  # ignora a faixa com a logo
    # debug helpers
    debug: bool = False,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "compute_width",
) -> WidthResult:
    """
    Robust width estimation with:
      • adaptive band selection (mask-driven, horizon-aware);
      • frame quality gating (Δu, parallax, continuity);
      • automatic mixing of geometric and depth-assisted paths;
      • data-driven uncertainty (IQR) when available.
    """
    H, W = sidewalk.shape
    H_eff = max(1, int(H) - int(bottom_ignore_px))  # ignora rodapé (logo)
    sidewalk = sidewalk.astype(bool)

    # heurísticas suaves para frames sem parallax
    DEPTH_MED = (
        float(np.median(depth[np.isfinite(depth)]))
        if (depth is not None and np.isfinite(depth).any())
        else None
    )
    SOFT_Z_CAP_FACTOR = 1.6
    SOFT_Z_CAP_GLOBAL = 5.5
    # Para evitar platô baixo em diagonais quando parallax==0:
    SOFT_TARGET_DU_MIN, SOFT_TARGET_DU_MAX = 170, 190
    TARGET_DU_NEAR = 170
    Z_CAP_HARD = float(np.clip((DEPTH_MED if DEPTH_MED else 3.2) * 1.45, 4.6, 6.0))

    # contadores e coletores
    near_perp_flag_rows = 0
    rows_seen = 0
    skip_du = 0

    # coletores por linha (geom “bruto” + DU de cada linha)
    widths_geom_raw: list[float] = []
    du_list: list[int] = []

    # profundidade por linha (somente quando parallax válido)
    widths_depth: list[float] = []
    debug_rows: list[_WidthDebugRow] = []

    parallax_valid_count = 0
    total_rows_considered = 0
    depth_good_rows = 0

    # 0) intrinsics and horizon
    fx, fy, cx, cy = _intrinsics_after_crop(W, fov_deg)
    v_h = cy - fy * np.tan(np.radians(pitch_deg))

    _swai_log(
        "intrinsics",
        {
            "W": int(W),
            "H": int(H),
            "FOV_deg": float(fov_deg),
            "pitch_deg": float(pitch_deg),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "v_h": float(v_h),
        },
    )

    # 1) choose band  ──────────────────────────────────────────────────────────────
    # Novo critério: pega o maior trecho contíguo de linhas com máscara abaixo do horizonte,
    # garantindo altura mínima mais generosa. Fallback para percentis amplos e banda fixa.
    MIN_BAND_PX = max(20, int(0.08 * H_eff))
    y_floor = max(int(v_h) + 5, 0)
    y_ceiling = H_eff

    if y_floor >= y_ceiling - 2:
        return WidthResult(0.0, 0.0, 0)

    ys_mask = np.where(sidewalk)[0]
    if band_mode == "fixed":
        y0 = max(y_floor, int(H * band_frac[0]))
        y1 = min(y_ceiling, int(H * band_frac[1]))
        if (y1 - y0) < MIN_BAND_PX:
            y1 = min(y_ceiling, y0 + MIN_BAND_PX)
    else:
        cov_rows = sidewalk[y_floor:y_ceiling].mean(axis=1) if y_ceiling > y_floor else np.array([])
        # maior run contíguo de linhas com cobertura > 0
        if cov_rows.size and np.any(cov_rows > 0):
            nz = np.where(cov_rows > 0)[0]
            best_start = nz[0]
            best_len = 1
            cur_start = nz[0]
            cur_len = 1
            for a, b in zip(nz, nz[1:]):
                if b == a + 1:
                    cur_len += 1
                else:
                    if cur_len > best_len:
                        best_start, best_len = cur_start, cur_len
                    cur_start, cur_len = b, 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            y0 = y_floor + best_start
            y1 = y0 + best_len
        else:
            y0 = y_floor
            y1 = min(y_ceiling, y_floor + MIN_BAND_PX)

        # expanda para garantir altura mínima
        if (y1 - y0) < MIN_BAND_PX:
            grow = (MIN_BAND_PX - (y1 - y0) + 1) // 2
            y0 = max(y_floor, y0 - grow)
            y1 = min(y_ceiling, y1 + grow)
            if (y1 - y0) < MIN_BAND_PX:
                y1 = min(y_ceiling, y0 + MIN_BAND_PX)

        # fallback largo baseado em percentis se a banda continua curta ou sem máscara
        if (ys_mask.size >= 5) and ((y1 - y0) < max(MIN_BAND_PX, int(0.10 * H_eff))):
            yq_lo = int(np.percentile(ys_mask, 20))
            yq_hi = int(np.percentile(ys_mask, 98))
            y0 = max(y_floor, yq_lo)
            y1 = min(y_ceiling, max(y0 + MIN_BAND_PX, yq_hi))
            if (y1 - y0) < MIN_BAND_PX:
                y1 = min(y_ceiling, y0 + MIN_BAND_PX)

    band = sidewalk[y0:y1].astype(np.uint8)
    band_cov = float(band.sum()) / float(band.size) if band.size else 0.0
    cov_norm = float(np.clip((band_cov - 0.10) / 0.25, 0.0, 1.0))

    _swai_log(
        "band",
        {"mode": band_mode, "y0": int(y0), "y1": int(y1), "height": int(y1 - y0), "cov": band_cov},
    )

    # ── DU gating adaptativo ─────────────────────────────────────────────────────
    DU_LO_FLOOR = 6  # não deixe Δu exigir mais que isso em casos extremos
    du_lo_base, du_hi = du_range_px  # ex.: (20, 220)

    # prior simples de Δu a partir de um Z "saudável"
    Z_soft = min(DEPTH_MED * SOFT_Z_CAP_FACTOR, SOFT_Z_CAP_GLOBAL) if DEPTH_MED is not None else 5.0
    du_prior = (float(fx) / max(1.0, Z_soft)) * 2.1  # px para ~2.1m

    # quando a cobertura é baixa (cov_norm→0), aceite Δu menor
    du_lo_eff = int(
        np.clip(
            np.interp(
                cov_norm,
                [0.0, 0.5, 1.0],
                [max(DU_LO_FLOOR, 0.35 * du_prior), max(10.0, 0.5 * du_prior), du_lo_base],
            ),
            DU_LO_FLOOR,
            du_lo_base,
        )
    )

    _swai_log(
        "du_gating",
        {
            "du_lo_eff": int(du_lo_eff),
            "du_hi": int(du_hi),
            "du_prior": float(du_prior),
            "cov_norm": float(cov_norm),
        },
    )

    if y0 >= y1 - 5:
        return WidthResult(0.0, 0.0, 0)

    # 2) quick rejects: dual sidewalks (parallel view)
    # if skip_if_dual and _has_two_curbs(sidewalk, min_gap_px=max_gap_cols):
    #    return WidthResult(0.0, 0.0, 0)

    # 3) continuity score on the band
    def _continuity_score(m, y0, y1):
        b = m[y0:y1].astype(np.uint8)
        if b.sum() == 0:
            return 0.0, 9999
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if num <= 1:
            return 0.0, 9999
        areas = stats[1:, cv2.CC_STAT_AREA]
        dominant = areas.max() if areas.size else 0
        frac = float(dominant) / float(areas.sum()) if areas.sum() > 0 else 0.0

        cols = np.unique(np.where(b)[1])
        if cols.size <= 1:
            gap = 9999
        else:
            cs = np.sort(cols)
            diffs = np.diff(cs)
            gap = int(diffs.max()) if diffs.size else 0
        return frac, gap

    frac_dom, gap_cols = _continuity_score(sidewalk, y0, y1)
    _swai_log("continuity", {"frac_dom": float(frac_dom), "gap_cols": int(gap_cols)})

    # thresholds adaptativos:
    # - quanto MENOR a cobertura, MENOR a exigência de fração dominante
    # - NÃO descartamos por gap (oclusão é esperada); exigimos só linhas válidas depois
    cont_min = 0.25 + 0.25 * cov_norm  # 0.30–0.60
    if frac_dom < cont_min:
        _swai_log("continuity_fail", {"band_cov": float(band_cov), "cont_min": float(cont_min)})
        return WidthResult(0.0, 0.0, 0)

    # 4) per-row computation with gating
    du_hi = du_range_px[1]
    par_lo, par_hi = parallax_range

    v_rows = np.arange(y0, y1)
    was_near_perp_count = 0
    for v in v_rows:
        cols = np.where(sidewalk[v])[0]
        if cols.size < 2:
            continue
        gap_max = None
        bridged = False
        # --- bridge virtual: unifica dois clusters separados por um gap plausível de oclusão ---
        if cols.size >= 2:
            cs = np.sort(cols)
            diffs = np.diff(cs)
            if diffs.size > 0:
                g = int(diffs.max())
                gap_max = g
                # janela padrão de oclusão (ajuste fino se precisar)
                #   • low coverage → aceite gaps maiores
                gap_lo = 40
                gap_hi = int(np.interp(cov_norm, [0.0, 1.0], [160, 120]))  # 160→120 px
                if g >= gap_lo and g <= gap_hi:
                    idx = int(np.argmax(diffs))
                    left_end = cs[idx]
                    right_beg = cs[idx + 1]
                    bridge = np.arange(left_end, right_beg + 1, dtype=int)
                    cs = np.concatenate([cs[: idx + 1], bridge, cs[idx + 1 :]])
                    cols = cs
                    bridged = True
        # q adaptativo: mais inclusivo quando a linha é contínua (gap pequeno)
        if gap_max is not None and gap_max <= 25:
            q = float(np.interp(cov_norm, [0.0, 0.5, 1.0], [0.18, 0.14, 0.12]))
        else:
            q = 0.22 + 0.08 * cov_norm
        uL_q = int(np.quantile(cols, q))
        uR_q = int(np.quantile(cols, 1.0 - q))

        # fallback se der ruim
        if uR_q - uL_q < 3:  # quase colado
            uL, uR = cols[0], cols[-1]
        else:
            uL, uR = uL_q, uR_q

        # se o quantil recortou demais e a linha é contínua, use a extensão total
        span_full = cols[-1] - cols[0]
        used_full_span = False
        if span_full > 0 and (uR - uL) < 0.85 * span_full:
            if (gap_max is None or gap_max <= 20) and cov_norm >= 0.15:
                uL, uR = cols[0], cols[-1]
                used_full_span = True

        du = uR - uL

        was_near_perp = du > du_hi
        rows_seen += 1
        row_dbg = _WidthDebugRow(
            v=int(v),
            uL=int(uL),
            uR=int(uR),
            du=int(du),
            near_perp=bool(was_near_perp),
            from_retry=False,
            bridged_gap=bridged,
        )
        if du < du_lo_eff:
            skip_du += 1
            row_dbg.skipped_du = True
            debug_rows.append(row_dbg)
            continue

        if du > du_hi:
            # === near-perp: puxe para um DU interno estável ===
            near_perp_flag_rows += 1
            soft_near_perp = (
                (gap_max is not None and gap_max <= 20)
                and (cov_norm >= 0.15)
                and (du <= int(1.4 * du_hi))
            )
            if debug:
                _swai_log(
                    "near_perp_row",
                    {
                        "v": int(v),
                        "gap_max": None if gap_max is None else int(gap_max),
                        "cov_norm": float(cov_norm),
                        "q": float(q),
                        "uL_q": int(uL_q),
                        "uR_q": int(uR_q),
                        "uL": int(uL),
                        "uR": int(uR),
                        "span_full": int(span_full),
                        "used_full_span": bool(used_full_span),
                        "du": int(du),
                        "du_hi": int(du_hi),
                        "soft_near_perp": bool(soft_near_perp),
                    },
                )
            if soft_near_perp:
                total_rows_considered += 1
                XL_g, ZL_g = _ground_intersection(
                    np.array([uL], dtype=np.float32),
                    np.array([v], dtype=np.float32),
                    fx,
                    fy,
                    cx,
                    cy,
                    pitch_deg,
                )
                XR_g, ZR_g = _ground_intersection(
                    np.array([uR], dtype=np.float32),
                    np.array([v], dtype=np.float32),
                    fx,
                    fy,
                    cx,
                    cy,
                    pitch_deg,
                )
                ZL = float(min(ZL_g[0], Z_CAP_HARD)) if np.isfinite(ZL_g[0]) else float(ZL_g[0])
                ZR = float(min(ZR_g[0], Z_CAP_HARD)) if np.isfinite(ZR_g[0]) else float(ZR_g[0])
                XL = (uL - cx) * ZL / fx
                XR = (uR - cx) * ZR / fx
                if np.isfinite(XL) and np.isfinite(XR):
                    widths_geom_raw.append(abs(XR - XL))
                    du_list.append(int(du))
                    row_dbg.width_geom = float(abs(XR - XL))
                    was_near_perp_count += 1
                debug_rows.append(row_dbg)
                continue
            target_near = float(
                np.clip(
                    (float(TARGET_DU_NEAR) * (1.0 + 0.06 * (0.5 - np.random.rand()))), 150.0, 190.0
                )
            )
            excess = du - target_near
            if excess > 0:
                shrink = max(1, excess // 2)
                uL2 = uL + shrink
                uR2 = uR - shrink
                du2 = uR2 - uL2
            else:
                uL2, uR2, du2 = uL, uR, du

            # se ainda ficou grande, trata como extremo: só geom + clamp de Z
            if du2 > du_hi or (uR2 <= uL2 + 5):
                row_dbg.uL = int(uL2)
                row_dbg.uR = int(uR2)
                row_dbg.du = int(du2)
                total_rows_considered += 1
                XL_g, ZL_g = _ground_intersection(
                    np.array([uL], dtype=np.float32),
                    np.array([v], dtype=np.float32),
                    fx,
                    fy,
                    cx,
                    cy,
                    pitch_deg,
                )
                XR_g, ZR_g = _ground_intersection(
                    np.array([uR], dtype=np.float32),
                    np.array([v], dtype=np.float32),
                    fx,
                    fy,
                    cx,
                    cy,
                    pitch_deg,
                )
                ZL = float(min(ZL_g[0], Z_CAP_HARD)) if np.isfinite(ZL_g[0]) else float(ZL_g[0])
                ZR = float(min(ZR_g[0], Z_CAP_HARD)) if np.isfinite(ZR_g[0]) else float(ZR_g[0])
                XL = (uL - cx) * ZL / fx
                XR = (uR - cx) * ZR / fx
                if np.isfinite(XL) and np.isfinite(XR):
                    widths_geom_raw.append(abs(XR - XL))
                    du_list.append(int(du))
                    row_dbg.width_geom = float(abs(XR - XL))
                debug_rows.append(row_dbg)
                continue

            # caso "ajustado": substitui bordas e segue
            uL, uR, du = uL2, uR2, du2
            row_dbg.uL = int(uL)
            row_dbg.uR = int(uR)
            row_dbg.du = int(du)

        total_rows_considered += 1

        # Geometric path (always available below the horizon)
        XL_g, ZL_g = _ground_intersection(
            np.array([uL], dtype=np.float32),
            np.array([v], dtype=np.float32),
            fx,
            fy,
            cx,
            cy,
            pitch_deg,
        )
        XR_g, ZR_g = _ground_intersection(
            np.array([uR], dtype=np.float32),
            np.array([v], dtype=np.float32),
            fx,
            fy,
            cx,
            cy,
            pitch_deg,
        )
        if np.isfinite(XL_g[0]) and np.isfinite(XR_g[0]):
            if was_near_perp:
                # clamp "duro" já existente (Z_CAP ~ 5.0)
                ZL = float(min(ZL_g[0], Z_CAP_HARD)) if np.isfinite(ZL_g[0]) else float(ZL_g[0])
                ZR = float(min(ZR_g[0], Z_CAP_HARD)) if np.isfinite(ZR_g[0]) else float(ZR_g[0])
                XL = (uL - cx) * ZL / fx
                XR = (uR - cx) * ZR / fx
                widths_geom_raw.append(abs(XR - XL))
                du_list.append(int(du))
                row_dbg.width_geom = float(abs(XR - XL))
                was_near_perp_count += 1
            else:
                # --- clamp "suave" guiado pelo Zoe quando parallax for inexistente ---
                if DEPTH_MED is not None:
                    Z_SOFT = min(DEPTH_MED * SOFT_Z_CAP_FACTOR, SOFT_Z_CAP_GLOBAL)
                    # só aplica se o Z geométrico está acima do "saudável"
                    ZLg = float(ZL_g[0])
                    ZRg = float(ZR_g[0])
                    ZL = ZLg if ZLg <= Z_SOFT else Z_SOFT
                    ZR = ZRg if ZRg <= Z_SOFT else Z_SOFT
                    XL = (uL - cx) * ZL / fx
                    XR = (uR - cx) * ZR / fx
                    widths_geom_raw.append(abs(XR - XL))
                    du_list.append(int(du))
                    row_dbg.width_geom = float(abs(XR - XL))
                else:
                    width_geom_val = abs(float(XR_g[0] - XL_g[0]))
                    widths_geom_raw.append(width_geom_val)
                    du_list.append(int(du))
                    row_dbg.width_geom = float(width_geom_val)

        # Depth-assisted path (safe indexing)
        if depth is not None:
            # ensure integer indices and in-bounds
            iv = int(v)
            iuL = int(uL)
            iuR = int(uR)
            H_d, W_d = depth.shape[:2]
            if iv < 0 or iv >= H_d or iuL < 0 or iuL >= W_d or iuR < 0 or iuR >= W_d:
                # skip this row if depth lookup is out-of-bounds
                continue
            ZL = float(depth[iv, iuL])
            ZR = float(depth[iv, iuR])
            if not (np.isfinite(ZL) and np.isfinite(ZR)) or ZL <= 0 or ZR <= 0:
                continue
            # depth sanity for this row
            zmean = 0.5 * (ZL + ZR)
            if 0.5 < zmean < 15.0:
                depth_good_rows += 1
            # parallax score
            par = abs(ZL - ZR) / zmean if zmean > 1e-6 else 0.0
            row_dbg.parallax = float(par)
            if depth_good_rows >= max(3, int(0.5 * min_valid_rows)):
                # quando a profundidade na banda parece boa, aceite também parallax muito baixo
                use_depth_row = par <= par_hi
            else:
                use_depth_row = (par >= par_lo) and (par <= par_hi)
            if use_depth_row:
                parallax_valid_count += 1
                XL_d = (uL - cx) * ZL / fx
                XR_d = (uR - cx) * ZR / fx
                if np.isfinite(XL_d) and np.isfinite(XR_d):
                    depth_width_val = abs(float(XR_d - XL_d))
                    widths_depth.append(depth_width_val)
                    row_dbg.width_depth = float(depth_width_val)
                    row_dbg.used_depth = True

        debug_rows.append(row_dbg)

    # fallback leve: se todas as linhas foram descartadas por Δu, tenta subir o band em 20 px uma vez
    if total_rows_considered == 0 and skip_du == rows_seen and (y0 > int(v_h) + 25):
        y0b = max(int(v_h) + 5, y0 - 20)
        y1b = max(y0b + 1, y1 - 20)
        _swai_log(
            "band_retry",
            {"y0_old": int(y0), "y1_old": int(y1), "y0_new": int(y0b), "y1_new": int(y1b)},
        )
        v_rows = np.arange(y0b, y1b)
        band_b = sidewalk[y0b:y1b].astype(np.uint8)
        band_cov_b = float(band_b.sum()) / float(band_b.size) if band_b.size else 0.0
        cov_norm_b = float(np.clip((band_cov_b - 0.10) / 0.25, 0.0, 1.0))
        # repete o processamento básico (reduzido) só geométrico para este retry
        was_near_perp_count = 0
        for v in v_rows:
            cols = np.where(sidewalk[v])[0]
            if cols.size < 2:
                continue
            gap_max = None
            bridged = False
            # --- bridge virtual: unifica dois clusters separados por um gap plausível de oclusão ---
            if cols.size >= 2:
                cs = np.sort(cols)
                diffs = np.diff(cs)
                if diffs.size > 0:
                    g = int(diffs.max())
                    gap_max = g
                    # janela padrão de oclusão (ajuste fino se precisar)
                    #   • low coverage → aceite gaps maiores
                    gap_lo = 40
                    gap_hi = int(np.interp(cov_norm, [0.0, 1.0], [160, 120]))  # 160→120 px
                    if g >= gap_lo and g <= gap_hi:
                        idx = int(np.argmax(diffs))
                        left_end = cs[idx]
                        right_beg = cs[idx + 1]
                    bridge = np.arange(left_end, right_beg + 1, dtype=int)
                    cs = np.concatenate([cs[: idx + 1], bridge, cs[idx + 1 :]])
                    cols = cs
                    bridged = True
            # quantis internos (iguais ao loop principal)
            if gap_max is not None and gap_max <= 25:
                q = float(np.interp(cov_norm_b, [0.0, 0.5, 1.0], [0.18, 0.14, 0.12]))
            else:
                q = 0.22 + 0.08 * cov_norm_b
            uL_q = int(np.quantile(cols, q))
            uR_q = int(np.quantile(cols, 1.0 - q))
            if uR_q - uL_q < 3:
                uL, uR = cols[0], cols[-1]
            else:
                uL, uR = uL_q, uR_q
            span_full = cols[-1] - cols[0]
            used_full_span = False
            if span_full > 0 and (uR - uL) < 0.85 * span_full:
                if (gap_max is None or gap_max <= 20) and cov_norm_b >= 0.15:
                    uL, uR = cols[0], cols[-1]
                    used_full_span = True
            du = uR - uL
            was_near_perp = du > du_hi
            row_dbg = _WidthDebugRow(
                v=int(v),
                uL=int(uL),
                uR=int(uR),
                du=int(du),
                near_perp=bool(was_near_perp),
                from_retry=True,
                bridged_gap=bridged,
            )
            if du < du_lo_eff:
                row_dbg.skipped_du = True
                debug_rows.append(row_dbg)
                continue
            if du > du_hi:
                # === near-perp: traga as bordas para um DU alvo estável ===
                near_perp_flag_rows += 1
                soft_near_perp = (
                    (gap_max is not None and gap_max <= 20)
                    and (cov_norm_b >= 0.15)
                    and (du <= int(1.4 * du_hi))
                )
                if debug:
                    _swai_log(
                        "near_perp_row_retry",
                        {
                            "v": int(v),
                            "gap_max": None if gap_max is None else int(gap_max),
                            "cov_norm": float(cov_norm_b),
                            "q": float(q),
                            "uL_q": int(uL_q),
                            "uR_q": int(uR_q),
                            "uL": int(uL),
                            "uR": int(uR),
                            "span_full": int(span_full),
                            "used_full_span": bool(used_full_span),
                            "du": int(du),
                            "du_hi": int(du_hi),
                            "soft_near_perp": bool(soft_near_perp),
                        },
                    )
                if soft_near_perp:
                    total_rows_considered += 1
                    XL_g, ZL_g = _ground_intersection(
                        np.array([uL], dtype=np.float32),
                        np.array([v], dtype=np.float32),
                        fx,
                        fy,
                        cx,
                        cy,
                        pitch_deg,
                    )
                    XR_g, ZR_g = _ground_intersection(
                        np.array([uR], dtype=np.float32),
                        np.array([v], dtype=np.float32),
                        fx,
                        fy,
                        cx,
                        cy,
                        pitch_deg,
                    )
                    ZL = float(min(ZL_g[0], Z_CAP_HARD)) if np.isfinite(ZL_g[0]) else float(ZL_g[0])
                    ZR = float(min(ZR_g[0], Z_CAP_HARD)) if np.isfinite(ZR_g[0]) else float(ZR_g[0])
                    XL = (uL - cx) * ZL / fx
                    XR = (uR - cx) * ZR / fx
                    if np.isfinite(XL) and np.isfinite(XR):
                        widths_geom_raw.append(abs(XR - XL))
                        du_list.append(int(du))
                        row_dbg.width_geom = float(abs(XR - XL))
                        was_near_perp_count += 1
                    debug_rows.append(row_dbg)
                    continue
                target_near = float(
                    np.clip(
                        (float(TARGET_DU_NEAR) * (1.0 + 0.06 * (0.5 - np.random.rand()))),
                        150.0,
                        190.0,
                    )
                )
                excess = du - target_near
                if excess > 0:
                    shrink = max(1, excess // 2)
                    uL2 = uL + shrink
                    uR2 = uR - shrink
                    du2 = uR2 - uL2
                else:
                    uL2, uR2, du2 = uL, uR, du

                # Se ainda ficou muito grande, trate como extremo: só geom + clamp de Z
                if du2 > du_hi or (uR2 <= uL2 + 5):
                    row_dbg.uL = int(uL2)
                    row_dbg.uR = int(uR2)
                    row_dbg.du = int(du2)
                    total_rows_considered += 1
                    XL_g, ZL_g = _ground_intersection(
                        np.array([uL], dtype=np.float32),
                        np.array([v], dtype=np.float32),
                        fx,
                        fy,
                        cx,
                        cy,
                        pitch_deg,
                    )
                    XR_g, ZR_g = _ground_intersection(
                        np.array([uR], dtype=np.float32),
                        np.array([v], dtype=np.float32),
                        fx,
                        fy,
                        cx,
                        cy,
                        pitch_deg,
                    )
                    ZL = float(min(ZL_g[0], Z_CAP_HARD)) if np.isfinite(ZL_g[0]) else float(ZL_g[0])
                    ZR = float(min(ZR_g[0], Z_CAP_HARD)) if np.isfinite(ZR_g[0]) else float(ZR_g[0])
                    XL = (uL - cx) * ZL / fx
                    XR = (uR - cx) * ZR / fx
                    if np.isfinite(XL) and np.isfinite(XR):
                        widths_geom_raw.append(abs(XR - XL))
                        du_list.append(int(du))
                        row_dbg.width_geom = float(abs(XR - XL))
                    debug_rows.append(row_dbg)
                    continue

                # Caso “ajustado”: substitui bordas e segue fluxo normal
                uL, uR, du = uL2, uR2, du2
                row_dbg.uL = int(uL)
                row_dbg.uR = int(uR)
                row_dbg.du = int(du)

            # só geométrico no retry
            XL_g, ZL_g = _ground_intersection(
                np.array([uL], dtype=np.float32),
                np.array([v], dtype=np.float32),
                fx,
                fy,
                cx,
                cy,
                pitch_deg,
            )
            XR_g, ZR_g = _ground_intersection(
                np.array([uR], dtype=np.float32),
                np.array([v], dtype=np.float32),
                fx,
                fy,
                cx,
                cy,
                pitch_deg,
            )
            if np.isfinite(XL_g[0]) and np.isfinite(XR_g[0]):
                if was_near_perp:
                    ZL = float(min(ZL_g[0], Z_CAP_HARD))
                    ZR = float(min(ZR_g[0], Z_CAP_HARD))
                    XL = (uL - cx) * ZL / fx
                    XR = (uR - cx) * ZR / fx
                    widths_geom_raw.append(abs(XR - XL))
                    du_list.append(int(du))
                    row_dbg.width_geom = float(abs(XR - XL))
                    was_near_perp_count += 1
                else:
                    width_geom_val = abs(float(XR_g[0] - XL_g[0]))
                    widths_geom_raw.append(width_geom_val)
                    du_list.append(int(du))
                    row_dbg.width_geom = float(width_geom_val)

            debug_rows.append(row_dbg)

    _swai_log(
        "rows_counters",
        {
            "rows_seen": int(rows_seen),
            "rows_considered": int(total_rows_considered),
            "skipped_du": int(skip_du),
            "near_perp_rows": int(near_perp_flag_rows),
            "was_near_perp_frac": float(was_near_perp_count) / float(max(1, rows_seen)),
            "n_geom_rows": len(widths_geom_raw),
            "n_depth_rows": len(widths_depth),
        },
    )

    band_h = y1 - y0

    if band_cov < 0.03:  # banda muito rara
        min_rows_eff = 2
    elif band_cov < 0.08 or band_h <= MIN_BAND_PX:
        min_rows_eff = max(3, int(0.5 * min_valid_rows))
    elif band_cov < 0.15:
        min_rows_eff = max(4, int(0.7 * min_valid_rows))
    else:
        min_rows_eff = min_valid_rows

    # fração de linhas com parallax válido (precisamos disto ANTES das flags)
    par_frac = (
        (parallax_valid_count / max(1, total_rows_considered)) if total_rows_considered else 0.0
    )

    # --- SUAVE: pós-processamento quando parallax == 0 e NÃO é near-perp dominante ---
    near_perp_dominante = (near_perp_flag_rows >= max(3, int(0.3 * max(1, rows_seen)))) and (
        depth_good_rows < min_rows_eff
    )
    widths_geom_final = list(widths_geom_raw)

    if (
        (parallax_valid_count == 0)
        and (not near_perp_dominante)
        and (len(widths_geom_raw) >= min_rows_eff)
    ):
        med_du = float(np.median(du_list)) if len(du_list) else None
        if med_du and np.isfinite(med_du):
            # piso dinâmico sugerido pela profundidade da cena
            Z_soft = (
                min(DEPTH_MED * SOFT_Z_CAP_FACTOR, SOFT_Z_CAP_GLOBAL)
                if DEPTH_MED is not None
                else 5.0
            )
            du_from_depth = (float(fx) / max(1.0, Z_soft)) * 2.1  # largura-prior 2.1 m

            # Ajusta o piso/alvo em DU para o FOV atual (baseline ~75°),
            # de forma que o intervalo [SOFT_TARGET_DU_MIN, SOFT_TARGET_DU_MAX]
            # represente aproximadamente o mesmo intervalo de larguras em metros.
            FOV_BASE_DEG = 75.0
            fx_base = W / (2 * np.tan(np.radians(FOV_BASE_DEG / 2.0)))
            f_scale = float(fx / fx_base) if fx_base > 0 else 1.0
            soft_du_min = SOFT_TARGET_DU_MIN * f_scale
            soft_du_max = SOFT_TARGET_DU_MAX * f_scale

            du_floor = float(np.clip(du_from_depth, soft_du_min, soft_du_max))

            # alvo final: mediana clipada pelo piso
            target_du = float(np.clip(med_du, du_floor, soft_du_max))

            # α adaptativo (0.35–0.85) aumenta quando cobertura baixa e parallax==0
            alpha = float(
                np.clip(0.35 + 0.40 * (1.0 - par_frac) + 0.15 * (1.0 - cov_norm), 0.35, 0.85)
            )
            # ^ aqui (1.0 - 0.0) representa "100% sem parallax"; se você computar par_frac>0, use (1.0 - par_frac)

            _swai_log(
                "soft_adjust_du",
                {
                    "reason": "parallax==0",
                    "med_du": med_du,
                    "du_from_depth": du_from_depth,
                    "target_du": target_du,
                    "alpha": alpha,
                    "band_cov": band_cov,
                    "par_frac": par_frac,
                },
            )

            adj = []
            for w, du0 in zip(widths_geom_raw, du_list):
                if du0 > 0:
                    scale = target_du / float(du0)
                    # mistura: mantém alguma variância por linha
                    adj.append((1.0 - alpha) * w + alpha * (scale * w))
            if len(adj) >= min_rows_eff:
                widths_geom_final = adj
    # Ajuste adicional quando a banda começa alta (calçada próxima, base cortada)
    elif (
        (parallax_valid_count < 2)
        and (not near_perp_dominante)
        and (len(widths_geom_raw) >= min_rows_eff)
    ):
        band_start_high = (y0 - max(int(v_h) + 5, 0)) > max(MIN_BAND_PX, int(0.12 * H_eff))
        if band_start_high:
            med_du = float(np.median(du_list)) if len(du_list) else None
            if med_du and np.isfinite(med_du):
                Z_soft = (
                    min(DEPTH_MED * SOFT_Z_CAP_FACTOR, SOFT_Z_CAP_GLOBAL)
                    if DEPTH_MED is not None
                    else 5.0
                )
                du_from_depth = (float(fx) / max(1.0, Z_soft)) * 2.1  # largura-prior 2.1 m
                FOV_BASE_DEG = 75.0
                fx_base = W / (2 * np.tan(np.radians(FOV_BASE_DEG / 2.0)))
                f_scale = float(fx / fx_base) if fx_base > 0 else 1.0
                soft_du_min = SOFT_TARGET_DU_MIN * f_scale
                soft_du_max = SOFT_TARGET_DU_MAX * f_scale
                du_floor = float(np.clip(du_from_depth, soft_du_min, soft_du_max))
                target_du = float(np.clip(med_du, du_floor, soft_du_max))
                alpha = float(
                    np.clip(0.45 + 0.30 * (1.0 - par_frac) + 0.15 * (1.0 - cov_norm), 0.35, 0.85)
                )
                _swai_log(
                    "soft_adjust_du_high_band",
                    {
                        "reason": "band_start_high",
                        "y0": int(y0),
                        "v_h": float(v_h),
                        "med_du": med_du,
                        "du_from_depth": du_from_depth,
                        "target_du": target_du,
                        "alpha": alpha,
                        "band_cov": band_cov,
                        "par_frac": par_frac,
                    },
                )
                adj = []
                for w, du0 in zip(widths_geom_raw, du_list):
                    if du0 > 0:
                        scale = target_du / float(du0)
                        adj.append((1.0 - alpha) * w + alpha * (scale * w))
                if len(adj) >= min_rows_eff:
                    widths_geom_final = adj

    # 5) agregação robusta AGORA sobre os vetores finais
    def robust_agg(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return np.nan, np.nan
        med = float(np.median(a))
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        return med, iqr

    # Em vistas diagonais, privilegie as linhas com DU maior
    # (projeção mais larga, tipicamente mais próximas da câmera).
    widths_geom_for_agg = widths_geom_final
    if len(widths_geom_final) >= min_rows_eff and len(widths_geom_final) == len(du_list):
        du_arr = np.array(du_list, dtype=float)
        if du_arr.size >= min_rows_eff:
            du_p10 = float(np.percentile(du_arr, 10))
            du_p90 = float(np.percentile(du_arr, 90))
            du_spread = du_p90 - du_p10
            # spread em DU suficientemente grande sugere gradiente de distância na banda
            if du_spread >= 10.0 and parallax_valid_count >= 2:
                # mantém apenas as linhas no terço superior de DU
                du_thresh = float(np.quantile(du_arr, 0.66))
                keep = du_arr >= du_thresh
                if int(keep.sum()) >= max(min_rows_eff, 3):
                    widths_geom_for_agg = [w for w, k in zip(widths_geom_final, keep) if k]
                    _swai_log(
                        "du_focus",
                        {
                            "du_p10": du_p10,
                            "du_p90": du_p90,
                            "du_spread": du_spread,
                            "du_thresh": du_thresh,
                            "kept": int(keep.sum()),
                            "total": int(len(widths_geom_final)),
                        },
                    )

    med_g, iqr_g = robust_agg(widths_geom_for_agg)
    med_d, iqr_d = robust_agg(widths_depth)

    if iqr_g is not None and np.isfinite(iqr_g) and abs(iqr_g) < 1e-3:
        iqr_g = 0.0
    if iqr_d is not None and np.isfinite(iqr_d) and abs(iqr_d) < 1e-3:
        iqr_d = 0.0

    _swai_log(
        "agg_pre",
        {
            "n_geom": len(widths_geom_final),
            "n_depth": len(widths_depth),
            "med_g": None if np.isnan(med_g) else float(med_g),
            "iqr_g": None if np.isnan(iqr_g) else float(iqr_g),
            "med_d": None if np.isnan(med_d) else float(med_d),
            "iqr_d": None if np.isnan(iqr_d) else float(iqr_d),
        },
    )

    # se nada válido, saia (mas com fallback opcional de baixa confiança)
    if (np.isnan(med_g) or len(widths_geom_final) < min_rows_eff) and (
        np.isnan(med_d) or len(widths_depth) < min_rows_eff
    ):

        if len(widths_geom_final) >= 3:
            width_fallback = float(np.median(widths_geom_final))
            margin_fb = max((err_pct / 100.0) * width_fallback, 0.35 * width_fallback)
            _swai_log(
                "result_lowconf",
                {
                    "width": width_fallback,
                    "margin": margin_fb,
                    "nrows": int(len(widths_geom_final)),
                },
            )
            return WidthResult(width_fallback, margin_fb, int(len(widths_geom_final)))

        return WidthResult(0.0, 0.0, 0)

    # se nada válido, saia
    # if (np.isnan(med_g) or len(widths_geom_final) < min_rows_eff) and \
    # (np.isnan(med_d) or len(widths_depth)     < min_rows_eff):
    #    return WidthResult(0.0, 0.0, 0)

    # flags de confiabilidade (agora com par_frac definido)
    depth_reliable_base = (not np.isnan(med_d)) and (
        len(widths_depth) >= max(3, int(0.5 * min_rows_eff))
    )
    depth_reliable = depth_reliable_base and (par_frac >= 0.6)
    geom_reliable = (not np.isnan(med_g)) and (len(widths_geom_final) >= min_rows_eff)

    _swai_log(
        "parallax",
        {
            "parallax_valid_count": int(parallax_valid_count),
            "total_rows_considered": int(total_rows_considered),
            "par_frac": float(par_frac),
        },
    )
    _swai_log(
        "reliability",
        {"depth_reliable": bool(depth_reliable), "geom_reliable": bool(geom_reliable)},
    )

    # preferência por geom quando near-perp domina
    if near_perp_dominante:
        depth_reliable = False
        _swai_log(
            "near_perp_force_geom",
            {"near_perp_rows": int(near_perp_flag_rows), "rows_seen": int(rows_seen)},
        )

    # decisão final
    if depth_reliable and geom_reliable:
        denom = max(1e-6, max(med_d, med_g))
        div = abs(med_d - med_g) / denom
        _swai_log(
            "divergence",
            {
                "med_d": None if np.isnan(med_d) else float(med_d),
                "med_g": None if np.isnan(med_g) else float(med_g),
                "div": float(div),
                "divergence_pct": float(divergence_pct),
            },
        )
        chosen = "depth"
    elif depth_reliable:
        chosen = "depth"
    else:
        chosen = "geom"

    _swai_log("chosen", {"chosen": chosen})
    if chosen == "depth":
        width, iqr, nrows = float(med_d), iqr_d, len(widths_depth)
        if geom_reliable and np.isfinite(med_g):
            G = float(med_g)
            # Piso mais agressivo quando a própria geometria é baixa (casos diagonais),
            # mais conservador quando med_g é grande (quase perpendicular, pode estar inflada).
            if G < 1.8:
                geom_floor_frac = 0.90
            elif G < 2.4:
                geom_floor_frac = 0.80
            else:
                geom_floor_frac = 0.70

            geom_floor = geom_floor_frac * G
            if width < geom_floor:
                _swai_log(
                    "depth_geom_floor",
                    {
                        "med_d": float(med_d),
                        "med_g": float(med_g),
                        "width_before": width,
                        "geom_floor": geom_floor,
                        "frac": geom_floor_frac,
                    },
                )
                width = geom_floor
                if iqr_g is not None and np.isfinite(iqr_g):
                    base_iqr = iqr if (iqr is not None and np.isfinite(iqr)) else 0.0
                    iqr = max(base_iqr, float(iqr_g))
    else:
        width, iqr, nrows = float(med_g), iqr_g, len(widths_geom_final)

    if not np.isfinite(width) or width <= 0:
        return WidthResult(0.0, 0.0, 0)

    # 6) incerteza
    margin = (
        (0.5 * iqr)
        if (use_data_driven_margin and iqr and np.isfinite(iqr) and iqr > 0)
        else (err_pct / 100.0) * width
    )

    if debug:
        try:
            debug_path = _plot_compute_width_debug(
                sidewalk_mask=sidewalk,
                rows_dbg=debug_rows,
                y0=int(y0),
                y1=int(y1),
                width=float(width),
                margin=float(margin),
                chosen=chosen,
                du_lo=int(du_lo_eff),
                du_hi=int(du_hi),
                out_dir=debug_dir,
                prefix=debug_prefix,
            )
            _swai_log("debug_plot", {"path": debug_path})
        except Exception as e:
            _swai_log("debug_plot_error", {"error": str(e)})

    _swai_log("result", {"width": float(width), "margin": float(margin), "nrows": int(nrows)})
    return WidthResult(float(width), float(margin), int(nrows))


def _plot_compute_width_debug(
    sidewalk_mask: np.ndarray,
    rows_dbg: Sequence[_WidthDebugRow],
    y0: int,
    y1: int,
    width: float,
    margin: float,
    chosen: str,
    du_lo: int,
    du_hi: int,
    out_dir: Optional[str] = None,
    prefix: str = "compute_width",
) -> str:
    """
    Gera uma figura de debug mostrando:
      - máscara + banda y0:y1 + bordas por linha (cores por motivo de uso/descarta);
      - larguras por linha (geom/depth) vs y;
      - histograma de Δu com limites;
      - resumo textual.
    """
    H, W = sidewalk_mask.shape
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Mask view with band and per-row segments
    ax0 = axs[0, 0]
    ax0.imshow(sidewalk_mask, cmap="gray", origin="upper")
    ax0.axhspan(y0, y1, color="yellow", alpha=0.15, label="banda")
    for r in rows_dbg:
        if r.skipped_du:
            c = "red"
        elif r.near_perp:
            c = "orange"
        else:
            c = "lime"
        ax0.plot([r.uL, r.uR], [r.v, r.v], color=c, linewidth=1.0, alpha=0.9)
    ax0.set_title("Máscara, banda e bordas por linha")
    ax0.set_xlim(0, W)
    ax0.set_ylim(H, 0)

    # Widths vs row
    ax1 = axs[0, 1]
    ys_geom = [r.v for r in rows_dbg if r.width_geom is not None]
    w_geom = [r.width_geom for r in rows_dbg if r.width_geom is not None]
    ys_depth = [r.v for r in rows_dbg if r.width_depth is not None]
    w_depth = [r.width_depth for r in rows_dbg if r.width_depth is not None]
    if ys_geom:
        ax1.scatter(w_geom, ys_geom, s=12, c="lime", label="geom", alpha=0.7)
    if ys_depth:
        ax1.scatter(w_depth, ys_depth, s=12, c="cyan", label="depth", alpha=0.7, marker="x")
    ax1.axvline(width, color="blue", linestyle="--", label=f"escolhido ({chosen})")
    ax1.axvspan(max(0.0, width - margin), width + margin, color="blue", alpha=0.1, label="margem")
    ax1.invert_yaxis()
    ax1.set_xlabel("largura (m)")
    ax1.set_ylabel("linha (y)")
    ax1.set_title("Larguras por linha")
    if ax1.has_data():
        ax1.legend(loc="best")

    # Δu histogram
    ax2 = axs[1, 0]
    du_vals = [r.du for r in rows_dbg if not r.skipped_du]
    if du_vals:
        ax2.hist(du_vals, bins=20, color="gray", alpha=0.8)
    ax2.axvline(du_lo, color="green", linestyle="--", label=f"Δu min {du_lo}")
    ax2.axvline(du_hi, color="red", linestyle="--", label=f"Δu max {du_hi}")
    ax2.set_xlabel("Δu (px)")
    ax2.set_ylabel("contagem de linhas")
    ax2.set_title("Distribuição de Δu na banda")
    ax2.legend(loc="best")

    # Summary text
    ax3 = axs[1, 1]
    ax3.axis("off")
    total_rows = len([r for r in rows_dbg if not r.skipped_du])
    near_perp_rows = len([r for r in rows_dbg if r.near_perp])
    depth_rows = len([r for r in rows_dbg if r.width_depth is not None])
    bridged_rows = len([r for r in rows_dbg if r.bridged_gap])
    par_vals = [r.parallax for r in rows_dbg if r.parallax is not None]
    summary_lines = [
        f"Banda y0={y0}, y1={y1} (h={y1 - y0}px)",
        f"Linhas válidas: {total_rows} | near-perp: {near_perp_rows} | bridged: {bridged_rows}",
        f"Geom linhas: {len(ys_geom)} | Depth linhas: {depth_rows}",
        f"Width escolhido: {width:.2f} m ± {margin:.2f} m ({chosen})",
    ]
    if par_vals:
        summary_lines.append(f"Parallax mediana: {float(np.median(par_vals)):.3f}")
    ax3.text(0.01, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=10)

    fig.tight_layout()
    if out_dir is None:
        out_path = f"{prefix}.png"
    else:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def bottom_percent_mask(mask: np.ndarray, percent: float = 5.0, min_pixels: int = 6) -> np.ndarray:
    """Select bottom `percent` of True pixels in mask by y-coordinate."""
    mask_bool = np.asarray(mask) != 0
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    th = np.percentile(ys, 100.0 - float(percent))
    sel = ys >= th
    if sel.sum() < min_pixels:
        return np.zeros_like(mask_bool, dtype=bool)
    out = np.zeros_like(mask_bool, dtype=bool)
    out[ys[sel], xs[sel]] = True
    return out


def compute_clearances(
    sidewalk: np.ndarray,
    top_mask: Tuple[float, float],
    bot_mask: Tuple[float, float],
    obstacles: Sequence[tuple[str, np.ndarray]],
    sidewalk_width_m: float,
    bottom_percent: float = 5.0,
    min_cand_pixels: int = 6,
    return_candidates: bool = False,
) -> list[ClearanceResult]:

    results = []
    base_candidate_masks = []
    H, W = sidewalk.shape

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
        if abs(top_mask[0]) > 1e-5:
            top_x = (yL - top_mask[1]) / top_mask[0]
        else:
            top_x = float("nan")
        if abs(bot_mask[0]) > 1e-5:
            bot_x = (yL - bot_mask[1]) / bot_mask[0]
        else:
            bot_x = float("nan")

        curb_candidates = [v for v in (top_x, bot_x) if np.isfinite(v)]
        if len(curb_candidates) < 2:
            # Curbs not well-defined at this row; fall back to conservative zeros.
            results.append(ClearanceResult(label, 0.0, 0.0, 0.0, None, L_pixel_img, R_pixel_img))
            continue

        left_curb = float(min(curb_candidates))
        right_curb = float(max(curb_candidates))

        # Clamp curb positions to image bounds to avoid wild extrapolation.
        left_curb = max(0.0, min(left_curb, float(W - 1)))
        right_curb = max(0.0, min(right_curb, float(W - 1)))

        # Calculate widths in pixels
        total_width = right_curb - left_curb
        if not np.isfinite(total_width) or total_width <= 1.0:
            results.append(ClearanceResult(label, 0.0, 0.0, 0.0, None, L_pixel_img, R_pixel_img))
            continue

        # Clamp obstacle base to lie within the curb interval. This prevents
        # negative clearances or values larger than the sidewalk width when
        # segmentation/refinement slightly overshoots the curb.
        xL_clamped = min(max(float(xL), left_curb), right_curb)
        xR_clamped = min(max(float(xR), left_curb), right_curb)

        left_clearance = xL_clamped - left_curb
        right_clearance = right_curb - xR_clamped

        # Convert to percentages (now guaranteed within [0, 1] up to numerics)
        left_percent = max(0.0, min(left_clearance / total_width, 1.0))
        right_percent = max(0.0, min(right_clearance / total_width, 1.0))

        # print(f"L: {left_percent * 100:.2f} % R: {right_percent * 100:.2f} % ")

        L_m = left_percent * sidewalk_width_m
        R_m = right_percent * sidewalk_width_m
        # Numerical safety: ensure we never exceed the sidewalk width and
        # keep obstacle width non-negative.
        L_m = max(0.0, min(L_m, sidewalk_width_m))
        R_m = max(0.0, min(R_m, sidewalk_width_m))
        obs_w = max(0.0, sidewalk_width_m - (L_m + R_m))

        results.append(
            ClearanceResult(
                label=label,
                L_m=L_m,
                R_m=R_m,
                total_m=L_m + R_m,
                obs_width=obs_w,
                L_pixel=L_pixel_img,
                R_pixel=R_pixel_img,
            )
        )

    return (results, base_candidate_masks) if return_candidates else results


# --------------------------------------------------------------------------- #
# 3)  Internal helpers (kept private)
# --------------------------------------------------------------------------- #

'''
def compute_width_from_curbs(
    mask: np.ndarray,
    top: tuple[float, float],
    bot: tuple[float, float],
    *,
    pitch_deg: float = -10.0,
    FOV_deg:  float = 90.0,
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
'''
