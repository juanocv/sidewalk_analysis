# sidewalk_ai/processing/accessibility.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import numpy as np
import re
import os

# ------------------------- helpers -------------------------

def _safe_array(vals: Iterable[float]) -> np.ndarray:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return arr if arr.size else np.array([], dtype=float)

def _iqr_mask(x: np.ndarray) -> np.ndarray:
    if x.size < 4:
        return np.ones_like(x, dtype=bool)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (x >= lo) & (x <= hi)

def _robust_stats(vals: Iterable[float]) -> Dict[str, float]:
    x = _safe_array(vals)
    if x.size == 0:
        return dict(count=0, mean=np.nan, median=np.nan, p10=np.nan, p90=np.nan)
    m = _iqr_mask(x)
    xr = x[m] if m.sum() >= 2 else x
    return dict(
        count=int(xr.size),
        mean=float(np.mean(xr)),
        median=float(np.median(xr)),
        p10=float(np.percentile(xr, 10)),
        p90=float(np.percentile(xr, 90)),
    )

_LABEL_TYPE_RE = re.compile(r"^([a-zA-Z0-9 _\-]+)")

def _label_to_type(label: str) -> str:
    # "tree#4:base2" -> "tree"
    m = _LABEL_TYPE_RE.match(label)
    return m.group(1).strip().lower() if m else label.lower()

# ------------------------- schemas -------------------------

@dataclass(frozen=True)
class PerTypeMetrics:
    count: int
    free_left_m: Dict[str, float]
    free_right_m: Dict[str, float]
    free_total_m: Dict[str, float]
    # largura média "ocupada" pelo obstáculo, se você quiser reportar também:
    obstacle_width_m: Dict[str, float] | None = None

@dataclass(frozen=True)
class GlobalMetrics:
    total_obstacles: int
    free_left_m: Dict[str, float]       # stats globais
    free_right_m: Dict[str, float]
    free_total_m: Dict[str, float]
    meets_120m_ratio: float             # fração de instâncias com total_m >= 1.20
    rating: str                         # "I"/"II"/"III"
    # Multi-view only (None em single-view):
    avg_obstacles_per_view: float | None = None
    avg_obstacles_per_view_rounded: int | None = None

@dataclass(frozen=True)
class AccessibilityMetrics:
    min_clear_required_m: float
    per_type: Dict[str, PerTypeMetrics]
    global_stats: GlobalMetrics

# ------------------------- core -------------------------

# fator do limiar intermediário (padrão 75% do threshold); pode ser ajustado por ENV
_MID_RATIO = float(os.getenv("SWAI_RANK_MID_RATIO", "0.75"))

def _rating_rank_by_threshold(median_corridor_m: float, threshold_m: float = 1.20) -> str:
    """
    Ranking simples baseado na mediana do corredor (pool L∪R):
      - III (Ideal):      mediana ≥ threshold
      - II (Razoável):    mediana ≥ _MID_RATIO * threshold  (padrão: 0.75 * threshold)
      - I  (Ruim):        caso contrário
    """
    if np.isnan(median_corridor_m):
        return "I"
    if median_corridor_m >= threshold_m:
        return "III"
    if median_corridor_m >= _MID_RATIO * threshold_m:
        return "II"
    return "I"

def compute_single_view_metrics(
    clearances: Iterable,      # Sequence[ClearanceResult]
    *,
    min_clear_required_m: float = 1.20,
    include_obstacle_width: bool = False,
) -> AccessibilityMetrics:
    """
    Gera métricas robustas para UMA imagem (single-view).
    """
    # Materializa listas
    items = list(clearances)

    # Por tipo
    # guardamos tuplas (L, R, total_m_original, obs_width) mas
    # "total" nas métricas passará a significar *corredor (pool L∪R)*
    by_type: Dict[str, List[Tuple[float,float,float,float]]] = {}
    for c in items:
        t = _label_to_type(c.label)
        by_type.setdefault(t, []).append( (c.L_m or 0.0, c.R_m or 0.0, c.total_m or 0.0, c.obs_width or 0.0) )

    per_type: Dict[str, PerTypeMetrics] = {}
    all_total = []
    for t, rows in by_type.items():
        Ls = [r[0] for r in rows]
        Rs = [r[1] for r in rows]
        # “total” nas métricas = CORREDOR (pool L∪R)
        corridors = [v for v in [*Ls, *Rs] if v is not None]
        Ws = [r[3] for r in rows]
        per_type[t] = PerTypeMetrics(
            count=len(rows),
            free_left_m=_robust_stats(Ls),
            free_right_m=_robust_stats(Rs),
            free_total_m=_robust_stats(corridors),  # agora: pool L∪R
            obstacle_width_m=_robust_stats(Ws) if include_obstacle_width else None,
        )
        all_total.extend(corridors)

    # “global total” = CORREDOR (pool L∪R) para todos os tipos
    all_corridors = _safe_array(all_total)
    if all_corridors.size:
        msk = _iqr_mask(all_corridors)
        all_corridors = all_corridors[msk] if msk.sum() >= 2 else all_corridors
        meet_ratio = float(np.mean(all_corridors >= min_clear_required_m))
        ft_stats = _robust_stats(all_corridors)  # stats do corredor global
        glob = GlobalMetrics(
            total_obstacles=int(sum(len(rows) for rows in by_type.values())),
            free_left_m=_robust_stats([r[0] for rows in by_type.values() for r in rows]),
            free_right_m=_robust_stats([r[1] for rows in by_type.values() for r in rows]),
            free_total_m=ft_stats,  # já contém median/p10/p90 do corredor (pool)
            meets_120m_ratio=meet_ratio,
            rating=_rating_rank_by_threshold(
                ft_stats.get("median", float("nan")),
                min_clear_required_m,
            ),
        )
    else:
        # Sem obstáculos → acessibilidade plena.
        # Para evitar NaN no output, fixamos as estatísticas do "corredor"
        # no próprio limiar (ou poderia ser qualquer valor ≥ limiar).
        nan_stats = {"median": float("nan"), "p10": float("nan"), "p90": float("nan"), "mean": float("nan")}
        ft_stats = {
            "median": float(min_clear_required_m),
            "p10":    float(min_clear_required_m),
            "p90":    float(min_clear_required_m),
            "mean":   float(min_clear_required_m),
        }
        glob = GlobalMetrics(
            total_obstacles=0,
            free_left_m=_robust_stats([]),
            free_right_m=_robust_stats([]),
            # keep same shape as the non-empty case: a dict with stats keys
            free_total_m=nan_stats,
            meets_120m_ratio=1.0,   # 100% atendem (não há bloqueio)
            rating="III", # regra simplificada: sem obstáculos = ideal
        )

    return AccessibilityMetrics(
        min_clear_required_m=min_clear_required_m,
        per_type=per_type,
        global_stats=glob,
    )

def compute_multiview_metrics(
    results_left: Iterable,   # Iterable[Result]
    results_right: Iterable,  # Iterable[Result]
    *,
    min_clear_required_m: float = 1.20,
) -> Dict[str, AccessibilityMetrics]:
    """
    Agrega métricas por lado (LEFT/RIGHT) e geral (ALL), concatenando
    todas as instâncias de CLEARANCES, com filtro de outliers por IQR.
    """
    def _collect(res_iter: Iterable) -> List:
        cl = []
        for r in res_iter:
            cl.extend(getattr(r, "clearances", []) or [])
        return cl

    left_cl  = _collect(results_left)
    right_cl = _collect(results_right)
    all_cl   = left_cl + right_cl

    # primeiro, compute métricas “como hoje”
    out = {
        "LEFT":  compute_single_view_metrics(left_cl,  min_clear_required_m=min_clear_required_m),
        "RIGHT": compute_single_view_metrics(right_cl, min_clear_required_m=min_clear_required_m),
        "ALL":   compute_single_view_metrics(all_cl,   min_clear_required_m=min_clear_required_m),
    }

    # helper p/ arredondar 2.5 -> 3 (half-up)
    def _round_half_up(x: float) -> int:
        return int(np.floor(x + 0.5))

    def _counts(res_list):
        return [len(getattr(r, "clearances", []) or []) for r in res_list]

    # LEFT / RIGHT: média por vista daquele lado
    for side, res_list in (("LEFT", results_left), ("RIGHT", results_right)):
        counts = _counts(res_list)
        avg = float(np.mean(counts)) if counts else 0.0
        gm = out[side].global_stats
        out[side] = AccessibilityMetrics(
            min_clear_required_m=out[side].min_clear_required_m,
            per_type=out[side].per_type,
            global_stats=GlobalMetrics(
                total_obstacles=gm.total_obstacles,                 # soma (mantido para auditoria)
                free_left_m=gm.free_left_m,
                free_right_m=gm.free_right_m,
                free_total_m=gm.free_total_m,
                meets_120m_ratio=gm.meets_120m_ratio,
                rating=gm.rating,
                avg_obstacles_per_view=avg,
                avg_obstacles_per_view_rounded=_round_half_up(avg),
            ),
        )

    # ALL: usar todas as vistas de ambos os lados
    counts_all = _counts(results_left) + _counts(results_right)
    avg_all = float(np.mean(counts_all)) if counts_all else 0.0
    gm = out["ALL"].global_stats
    out["ALL"] = AccessibilityMetrics(
        min_clear_required_m=out["ALL"].min_clear_required_m,
        per_type=out["ALL"].per_type,
        global_stats=GlobalMetrics(
            total_obstacles=gm.total_obstacles,
            free_left_m=gm.free_left_m,
            free_right_m=gm.free_right_m,
            free_total_m=gm.free_total_m,
            meets_120m_ratio=gm.meets_120m_ratio,
            rating=gm.rating,
            avg_obstacles_per_view=avg_all,
            avg_obstacles_per_view_rounded=_round_half_up(avg_all),
        ),
    )
    return out
