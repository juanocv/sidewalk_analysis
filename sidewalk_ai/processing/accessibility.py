# sidewalk_ai/processing/accessibility.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import numpy as np
import os

from sidewalk_ai.labels import label_to_type


# ------------------------- helpers públicos -------------------------
def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def types_summary(res_list):
    """
    res_list: lista de Results de um lado (LEFT/RIGHT).
    Retorna {tipo: {"prevalence": float 0..1, "typical_count_when_present": int}}
    - só inclui tipos que aparecem em pelo menos 1 vista (evita zeros “ruins”).
    """
    n = len(res_list) or 0
    if n == 0:
        return {}

    by_type_counts: dict[str, list[int]] = {}  # {tipo: [c0, c1, ... c{n-1}]}

    for i, r in enumerate(res_list):
        # 1) começa assumindo 0 para todos os tipos já vistos
        for t in by_type_counts.keys():
            by_type_counts[t].append(0)

        # 2) conta tipos desta vista
        counts: dict[str, int] = {}
        for c in getattr(r, "clearances", []) or []:
            t = _label_to_type(c.label)
            counts[t] = counts.get(t, 0) + 1

        # 3) para tipos novos, crie histórico de zeros das vistas passadas e
        #    acrescente o valor corrente; para tipos já conhecidos, sobrescreva o 0 recém-apensado
        for t, cnt in counts.items():
            if t not in by_type_counts:
                by_type_counts[t] = [0] * i  # zeros para as i vistas anteriores
                by_type_counts[t].append(cnt)
            else:
                by_type_counts[t][-1] = cnt  # substitui o 0 desta vista pelo cnt

    out = {}
    for t, seq in by_type_counts.items():
        # seq tem tamanho n (uma contagem por vista)
        if max(seq) == 0:
            continue  # não reporta tipos que nunca aparecem (segurança extra)
        prevalence = float(np.mean([1 if v > 0 else 0 for v in seq]))
        cond = [v for v in seq if v > 0]
        p50 = float(np.median(cond))
        out[t] = {
            "prevalence": prevalence,
            "typical_count_when_present": round_half_up(p50),
        }
    return out


def corridor_block(acc_side):
    g = acc_side.global_stats
    # g.free_total_m tem as estatísticas do corredor (pool L∪R)
    return {
        "median_m": g.free_total_m.get("median", float("nan")),
        "meets_ratio": g.meets_ratio,
        "rating": g.rating,
    }


# ------------------------- helpers privados -------------------------
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


_label_to_type = label_to_type  # retrocompatibilidade


# ------------------------- schemas -------------------------


@dataclass(frozen=True)
class PerTypeMetrics:
    count: int
    free_left_m: Dict[str, float]
    free_right_m: Dict[str, float]
    free_total_m: Dict[str, float]
    obs_width_m: Dict[str, float] | None = None

    def to_dict(self, drop_none: bool = True) -> Dict:
        d = {
            "count": self.count,
            "free_left_m": self.free_left_m,
            "free_right_m": self.free_right_m,
            "free_total_m": self.free_total_m,
            "obs_width_m": self.obs_width_m,
        }
        if drop_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


@dataclass(frozen=True)
class GlobalMetrics:
    total_obstacles: int
    free_left_m: Dict[str, float]  # stats globais
    free_right_m: Dict[str, float]
    free_total_m: Dict[str, float]
    meets_ratio: float  # fração de instâncias com total_m >= 1.20
    rating: str  # "I"/"II"/"III"
    # novo: estatísticas globais de largura dos obstáculos
    obs_width_m: Dict[str, float] | None = None
    # Multi-view only (None em single-view):
    avg_obstacles_per_view: float | None = None
    avg_obstacles_per_view_rounded: int | None = None

    def to_dict(self, *, drop_none: bool = True, drop_avgs_if_none: bool = True) -> Dict:
        d = {
            "total_obstacles": self.total_obstacles,
            "free_left_m": self.free_left_m,
            "free_right_m": self.free_right_m,
            "free_total_m": self.free_total_m,
            "meets_ratio": self.meets_ratio,
            "rating": self.rating,
            "obs_width_m": self.obs_width_m,
            "avg_obstacles_per_view": self.avg_obstacles_per_view,
            "avg_obstacles_per_view_rounded": self.avg_obstacles_per_view_rounded,
        }
        if drop_avgs_if_none:
            # esconde automaticamente no single-view
            if d["avg_obstacles_per_view"] is None:
                d.pop("avg_obstacles_per_view")
            if d["avg_obstacles_per_view_rounded"] is None:
                d.pop("avg_obstacles_per_view_rounded")
        if drop_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


@dataclass(frozen=True)
class AccessibilityMetrics:
    min_clear_required_m: float
    per_type: Dict[str, PerTypeMetrics]
    global_stats: GlobalMetrics

    def to_dict(self, drop_none: bool = True) -> Dict:
        return {
            "min_clear_required_m": self.min_clear_required_m,
            "global_stats": self.global_stats.to_dict(drop_none=drop_none, drop_avgs_if_none=True),
            "per_type": {k: v.to_dict(drop_none=drop_none) for k, v in self.per_type.items()},
        }


# ------------------------- core -------------------------

# Fator do limiar intermediário, como fração do threshold.
# Ajustável por ENV (SWAI_RANK_MID_RATIO); padrão 0.50, ou seja, 0.60 m para o
# threshold de 1.20 m da NBR 9050.
_MID_RATIO_DEFAULT = 0.50
MID_RATIO = float(os.getenv("SWAI_RANK_MID_RATIO", str(_MID_RATIO_DEFAULT)))
_MID_RATIO = MID_RATIO  # retrocompatibilidade para chamadores existentes


def _rating_rank_by_threshold(median_corridor_m: float, threshold_m: float = 1.20) -> str:
    """
    Ranking simples baseado na mediana do corredor (pool L∪R):
      - III (Ideal):      mediana ≥ threshold
      - II (Razoável):    mediana ≥ _MID_RATIO * threshold  (padrão: 0.50 * threshold)
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
    clearances: Iterable,  # Sequence[ClearanceResult]
    *,
    min_clear_required_m: float = 1.20,
    include_obstacle_width: bool = True,  # mantido p/ compat., mas agora sempre consideramos
) -> AccessibilityMetrics:
    items = list(clearances)

    # Build a mapping type -> list of raw clearance tuples
    # We keep raw None/NaN values so we can exclude those with invalid total_m
    by_type: Dict[str, List[Tuple[float | None, float | None, float | None, float | None]]] = {}
    for c in items:
        t = _label_to_type(c.label)
        L_val = getattr(c, "L_m", None)
        R_val = getattr(c, "R_m", None)
        total_val = getattr(c, "total_m", None)
        obs_w = getattr(c, "obs_width", None)
        by_type.setdefault(t, []).append((L_val, R_val, total_val, obs_w))

    per_type: Dict[str, PerTypeMetrics] = {}
    all_corridors_all = []
    all_corridors_pos = []
    all_obsw = []

    for t, rows in by_type.items():
        # Only consider clearances with a valid total_m for statistical metrics
        valid_rows = [r for r in rows if (r[2] is not None and np.isfinite(r[2]))]

        Ls = [r[0] for r in valid_rows if (r[0] is not None and np.isfinite(r[0]) and r[0] > 0.0)]
        Rs = [r[1] for r in valid_rows if (r[1] is not None and np.isfinite(r[1]) and r[1] > 0.0)]
        # pool L∪R from valid rows
        corridors = [
            v for r in valid_rows for v in (r[0], r[1]) if (v is not None and np.isfinite(v))
        ]
        corridors_pos = [v for v in corridors if v > 0.0]
        Ws = [float(w) for (_, _, _, w) in rows if (w is not None and np.isfinite(w))]

        per_type[t] = PerTypeMetrics(
            count=len(rows),  # count includes all detected obstacles of this type
            free_left_m=_robust_stats(Ls),
            free_right_m=_robust_stats(Rs),
            free_total_m=_robust_stats(corridors),  # pool L∪R from valid rows
            obs_width_m=_robust_stats(Ws) if Ws else None,
        )

        all_corridors_all.extend(corridors)
        all_corridors_pos.extend(corridors_pos)
        all_obsw.extend(Ws)

    all_corridors_all_arr = _safe_array(all_corridors_all)
    all_corridors_pos_arr = _safe_array(all_corridors_pos)
    if all_corridors_all_arr.size:
        # Estatísticas robustas (mediana, p10, p90, etc.) usando apenas
        # corredores > 0 quando existirem; se todos forem zero,
        # usamos o conjunto completo (mediana=0).
        if all_corridors_pos_arr.size:
            ft_stats = _robust_stats(all_corridors_pos_arr)
        else:
            ft_stats = dict(
                count=int(all_corridors_all_arr.size),
                mean=float(np.mean(all_corridors_all_arr)),
                median=float(np.median(all_corridors_all_arr)),
                p10=float(np.percentile(all_corridors_all_arr, 10)),
                p90=float(np.percentile(all_corridors_all_arr, 90)),
            )

        # meet_ratio considera TODOS os corredores (incluindo zeros), sem
        # filtro de IQR, para não subestimar situações com bloqueios totais.
        meet_ratio = float(np.mean(all_corridors_all_arr >= min_clear_required_m))
        glob = GlobalMetrics(
            total_obstacles=int(sum(len(rows) for rows in by_type.values())),
            free_left_m=_robust_stats([r[0] for rows in by_type.values() for r in rows]),
            free_right_m=_robust_stats([r[1] for rows in by_type.values() for r in rows]),
            free_total_m=ft_stats,
            meets_ratio=meet_ratio,
            rating=_rating_rank_by_threshold(
                ft_stats.get("median", float("nan")), min_clear_required_m
            ),
            obs_width_m=_robust_stats(all_obsw) if all_obsw else None,
        )
    else:
        # sem obstáculos → corredor "cheio" e obs_width_m inexistente
        nan_stats = {
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "mean": float("nan"),
        }
        glob = GlobalMetrics(
            total_obstacles=0,
            free_left_m=_robust_stats([]),
            free_right_m=_robust_stats([]),
            free_total_m=nan_stats,
            meets_ratio=1.0,
            rating="III",
            obs_width_m=None,
        )

    return AccessibilityMetrics(
        min_clear_required_m=min_clear_required_m,
        per_type=per_type,
        global_stats=glob,
    )


def _counts(res_list):
    return [len(getattr(r, "clearances", []) or []) for r in res_list]


def compute_multiview_metrics(
    results_left: Iterable,  # Iterable[Result]
    results_right: Iterable,  # Iterable[Result]
    *,
    min_clear_required_m: float = 1.20,
) -> Dict[str, AccessibilityMetrics]:
    """
    Agrega métricas por lado (LEFT/RIGHT) e geral (ALL), concatenando
    todas as instâncias de CLEARANCES, com filtro de outliers por IQR.

    Importante: vistas cuja largura estimada é inválida (width_m <= 0,
    None ou não finita) são ignoradas no cálculo das métricas de
    acessibilidade multi-view, para evitar que falhas de medição de
    largura distorçam as estatísticas de corredor (mediana, desvio,
    meets_ratio, etc.).
    """

    def _collect(res_iter: Iterable) -> List:
        cl = []
        for r in res_iter:
            # Se o objeto tiver atributo `width`, use-o para filtrar
            # vistas com largura inválida. Para chamadas legadas que
            # passam apenas objetos com `.clearances` (sem `.width`),
            # mantemos o comportamento antigo e não filtramos.
            w_res = getattr(r, "width", Ellipsis)
            if w_res is not Ellipsis:
                w_val = getattr(w_res, "width_m", None)
                if w_val is None or not np.isfinite(w_val) or w_val <= 0:
                    # width_m inválido: ignore clearances desta vista
                    continue
            cl.extend(getattr(r, "clearances", []) or [])
        return cl

    left_cl = _collect(results_left)
    right_cl = _collect(results_right)
    all_cl = left_cl + right_cl

    # primeiro, compute métricas “como hoje”
    out = {
        "LEFT": compute_single_view_metrics(left_cl, min_clear_required_m=min_clear_required_m),
        "RIGHT": compute_single_view_metrics(right_cl, min_clear_required_m=min_clear_required_m),
        "ALL": compute_single_view_metrics(all_cl, min_clear_required_m=min_clear_required_m),
    }

    # LEFT / RIGHT: média por vista daquele lado
    for side, res_list in (("LEFT", results_left), ("RIGHT", results_right)):
        counts = _counts(res_list)
        avg = float(np.mean(counts)) if counts else 0.0
        gm = out[side].global_stats
        out[side] = AccessibilityMetrics(
            min_clear_required_m=out[side].min_clear_required_m,
            per_type=out[side].per_type,
            global_stats=GlobalMetrics(
                total_obstacles=gm.total_obstacles,
                free_left_m=gm.free_left_m,
                free_right_m=gm.free_right_m,
                free_total_m=gm.free_total_m,
                meets_ratio=gm.meets_ratio,
                rating=gm.rating,
                obs_width_m=gm.obs_width_m,  # <<-- preserva
                avg_obstacles_per_view=avg,
                avg_obstacles_per_view_rounded=round_half_up(avg),
            ),
        )

    # ALL
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
            meets_ratio=gm.meets_ratio,
            rating=gm.rating,
            obs_width_m=gm.obs_width_m,  # <<--
            avg_obstacles_per_view=avg_all,
            avg_obstacles_per_view_rounded=round_half_up(avg_all),
        ),
    )
    return out
