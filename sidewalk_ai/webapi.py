from __future__ import annotations
import cv2, os
import base64
import numpy as np
import re

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sidewalk_ai.processing.accessibility import (
    compute_single_view_metrics, compute_multiview_metrics
)
from pydantic import BaseModel, Field
import sidewalk_ai as sw
from sidewalk_ai.models.factory import build_depth 
from sidewalk_ai.api.request import from_api_single, from_api_multi, run_pipeline

# Map ZoeDepth variant names to their canonical forms
NAME_MAP = {
    "ZoeD_N": "zoed_n",
    "ZoeD_K": "zoed_k",
    "ZoeD_NK": "zoed_nk",
    # aliases for backwards compatibility
    "zoed_n": "zoed_n",
    "zoed_k": "zoed_k",
    "zoed_nk": "zoed_nk",
}
VALID_ZOE = ["zoed_n", "zoed_k", "zoed_nk"]

from fastapi.middleware.cors import CORSMiddleware

# define a lifespan context manager to load once
@asynccontextmanager
async def lifespan(app: FastAPI):
    seg = sw.build_segmenter("oneformer")       # reused by every pipe
    sv  = sw.StreetViewClient()  # reused by every pipe
    app.state.sv = sv

    # ---- depth back-ends ------------------------------------------------
    depth_midas = build_depth("midas")
    depth_zoe   = build_depth(
    "zoe",
    variant=NAME_MAP.get(
        os.getenv("SWAI_ZOE_VARIANT", "zoed_n"), "zoed_n"
    ),
    )

    # ---- build four pipelines (depth × refine) -------------------------
    def _make(depth_obj, refine: bool):
        return sw.SidewalkPipeline(
            segmenter=seg, depth=depth_obj, streetview=sv, refine=refine
        )

    app.state.pipes = {
        ("midas", True):  _make(depth_midas, True),
        ("midas", False): _make(depth_midas, False),
        ("zoe",   True):  _make(depth_zoe,   True),
        ("zoe",   False): _make(depth_zoe,   False),
    }
    yield
    # (Optional) Shutdown logic here

app = FastAPI(
    title="Sidewalk-AI",
    version="0.1.0",
    description="Automatic sidewalk width estimation and obstacle detection.",
    lifespan=lifespan,
)

# ── add this block right after app = FastAPI(...) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or ["http://localhost"] if you prefer
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Request / response schemas                                         #
# ------------------------------------------------------------------ #
class AddressSingleReq(BaseModel):
# ─── option A: free-form address ──────────────────────────────
    address: str | None = Field(
        default=None,
        example="Av. Paulista 1578, São Paulo",
        description="Ignored if lat+lon are given",
    )
# ─── option B: explicit Street-View coordinates ───────────────
    lat: float  | None = Field(None, description="Latitude  (decimal deg)")
    lon: float  | None = Field(None, description="Longitude (decimal deg)")
    heading: int = Field(0,  ge=0,   le=359)
    pitch:   int = Field(-10,  ge=-90, le=90)
    fov:     int = Field(90, ge=10,  le=120)

    # Multi-angle vs single-angle control. If true, run the
    # multi-angle (address-mode) analysis which samples several
    # headings around the street center. If false, the request will
    # perform a single-angle analysis using the provided heading.
    # single-view SEMPRE → False (campo omitido no contrato)

# ─── depth selection ──────────────────────────────────────────
    depth: str = Field(
        default=os.getenv("SWAI_DEPTH", "zoe"),
        pattern="^(zoe|midas)$",
        description="'zoe' (default) or 'midas'",
        example="zoe",    
    )
    zoe_variant: str | None = Field(
        default=None,
        description="Override ZoeDepth variant (ZoeD_N, ZoeD_K, ZoeD_NK) "
                    "if depth='zoe'. Ignored otherwise.",
        example="ZoeD_N",
    )

# ─── geometry refinement knobs ────────────────────────────────
    refine: bool = True
    force_fallback: bool  = False
    fallback_scale: float | None = Field(None, gt=0,
                        description="metres-per-px when fallback is used")

# ─── misc ──────────────────────────────────────────────────────
    return_mask: bool = False

# ─── accessibility ─────────────────────────────────────────────
    min_clear: float = Field(1.20, ge=0.0, description="Limiar de caminho livre (m) para rating ABNT/NBR 9050")

class AddressMultiReq(BaseModel):
    # Address OU lat/lon
    address: str | None = Field(default=None, example="Av. Paulista 1578, São Paulo")
    lat: float  | None = Field(None); lon: float | None = Field(None)
    # heading não é usado em multi; pitch/fov são aceitos
    pitch:   int = Field(-10,  ge=-90, le=90)
    fov:     int = Field(90, ge=10,  le=120)
    depth: str = Field(default=os.getenv("SWAI_DEPTH", "zoe"), pattern="^(zoe|midas)$")
    zoe_variant: str | None = Field(
        default=None,
        description="Override ZoeDepth variant (ZoeD_N, ZoeD_K, ZoeD_NK) "
                    "if depth='zoe'. Ignored otherwise.",
        example="ZoeD_N",
    )
    refine: bool = True
    force_fallback: bool  = False
    fallback_scale: float | None = Field(None, gt=0)
    return_mask: bool = False
    min_clear: float = Field(1.20, ge=0.0)

class SingleResp(BaseModel):
    width_m: float
    margin_m: float
    clearances: list["ClearanceItem"] = []
    gsv_png_b64:  str  | None = None
    mask_png_b64:  str | None = None
    overlay_png_b64: str  | None = None
    accessibility: dict | None = None

class MultiSideCorridor(BaseModel):
    median_m: float | None = None
    meets_ratio: float | None = None
    rating: str | None = None

class MultiSideObstacles(BaseModel):
    typical_obstacles_per_view: int | None = None
    types: dict[str, dict] | None = None  # {"tree": {"prevalence": 1.0, "typical_count_when_present": 3}, ...}

class MultiSideSummary(BaseModel):
    median_width: dict | None = None      # {"width_m": ..., "margin_m": ...}
    corridor: MultiSideCorridor | None = None
    obstacles: MultiSideObstacles | None = None

class MultiRespSlim(BaseModel):
    # metadados mínimos (sem redundância)
    multi_metadata: dict | None = None    # {"n_headings": {"left": N, "right": M}}
    # sumarização por lado
    per_side: dict[str, MultiSideSummary] | None = None  # "LEFT"/"RIGHT"
    # sumarização agregada
    all_views: dict | None = None         # {"corridor": {...}, "typical_obstacles_per_view": int}
    # detalhe bruto permanece igual
    per_heading: list[dict] | None = None
    # imagens (opcional)
    samples_left: list[dict] | None = None
    samples_right: list[dict] | None = None
    obstacle_images: list[str] | None = None

class ClearanceItem(BaseModel):
    label: str
    L_m: float | None = None
    R_m: float | None = None
    total_m: float | None = None
    obs_width: float | None = None

def _png_b64(arr: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", arr)[1]).decode()
# ------------------------------------------------------------------ #
# shared helper – runs the pipeline exactly once                     #
# ------------------------------------------------------------------ #
def _pick_depth_pipe(req_depth: str, req_refine: bool, zoe_variant: str | None):
    # Build or pick the appropriate pipeline from app.state.pipes
    key = (req_depth, req_refine)
    if key not in app.state.pipes:
        raise HTTPException(400, f"Depth backend '{req_depth}' not available")

    pipe = app.state.pipes[key]

    # optional on-the-fly Zoe variant override (replace depth in the selected pipe)
    if req_depth == "zoe" and zoe_variant:
        v = NAME_MAP.get(zoe_variant, None)
        if v is None:
            raise HTTPException(400, "zoe_variant must be one of " f"{', '.join(VALID_ZOE)}")
        depth_obj = build_depth("zoe", variant=v)
        app.state.pipes[key] = sw.SidewalkPipeline(segmenter=pipe.segmenter, depth=depth_obj, streetview=app.state.sv, refine=req_refine)
        pipe = app.state.pipes[key]

    return pipe

@app.post("/analyse/single", response_model=SingleResp)
def analyse_single(req: AddressSingleReq):
    pipe = _pick_depth_pipe(req.depth, req.refine, req.zoe_variant)
    # normaliza e roda single-view
    cfg = from_api_single(req)
    try:
        res = run_pipeline(pipe, cfg)  # sempre um Result em single
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))

    # single-view → sempre um Result
    res_obj = res
    gsv_png_b64 = None; mask_png_b64 = None; overlay_png_b64 = None

    if req.return_mask and getattr(res_obj, "rgb_image", None) is not None:
        rgb_bgr = cv2.cvtColor(res_obj.rgb_image, cv2.COLOR_RGB2BGR)
        gsv_png_b64  = _png_b64(rgb_bgr)
        mask_u8      = (res_obj.sidewalk_mask * 255).astype("uint8")
        mask_png_b64 = _png_b64(mask_u8)
        overlay      = rgb_bgr.copy()
        overlay[res_obj.sidewalk_mask.astype(bool)] = (0, 255, 0)
        overlay      = cv2.addWeighted(overlay, 0.4, rgb_bgr, 0.6, 0)
        overlay_png_b64 = _png_b64(overlay)

    # acessibilidade (single)
    accessibility = None
    try:
        acc = compute_single_view_metrics(getattr(res_obj, 'clearances', []) or [],
                                          min_clear_required_m=req.min_clear)
        accessibility = {
            "min_clear_required_m": acc.min_clear_required_m,
            "global_stats": acc.global_stats.__dict__,
            "per_type": {k: v.__dict__ for k, v in acc.per_type.items()},
        }
    except Exception:
        accessibility = None

    clearance_items = [
        ClearanceItem(
            label=c.label,
            L_m=c.L_m if hasattr(c, 'L_m') else None,
            R_m=c.R_m if hasattr(c, 'R_m') else None,
            total_m=c.total_m if hasattr(c, 'total_m') else None,
            obs_width=c.obs_width if hasattr(c, 'obs_width') else None,
        ) for c in getattr(res_obj, 'clearances', [])
    ]

    return SingleResp(
        width_m  = getattr(res_obj.width, 'width_m', 0.0),
        margin_m = getattr(res_obj.width, 'margin_m', 0.0),
        clearances      = clearance_items,
        gsv_png_b64     = gsv_png_b64,
        mask_png_b64    = mask_png_b64,
        overlay_png_b64 = overlay_png_b64,
        accessibility   = accessibility,
    )

_LABEL_TYPE_RE = re.compile(r"^([a-zA-Z0-9 _\-]+)")
def _label_type(s: str) -> str:
    m = _LABEL_TYPE_RE.match(s or "")
    return m.group(1).strip().lower() if m else (s or "").lower()

def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))

def _types_summary(res_list):
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
            t = _label_type(c.label)
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
            "typical_count_when_present": _round_half_up(p50),
        }
    return out

@app.post("/analyse/multi", response_model=MultiRespSlim)
def analyse_multi(req: AddressMultiReq):
    pipe = _pick_depth_pipe(req.depth, req.refine, req.zoe_variant)
    cfg = from_api_multi(req)
    try:
        res = run_pipeline(pipe, cfg)  # dict com 'results', 'metadata', 'per_heading', ...
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))

    if not (isinstance(res, dict) and 'results' in res):
        raise HTTPException(500, "Unexpected pipeline output for multi-view")

    multi_metadata = res.get('metadata') or {}
    per_heading    = res.get('per_heading') or []
    obstacle_images= res.get('obstacle_images') or []
    left, right    = res.get('results', ([], []))

    # ---------- acessibilidade (LEFT/RIGHT/ALL) ----------
    try:
        acc = compute_multiview_metrics(left, right, min_clear_required_m=req.min_clear)
    except Exception:
        raise HTTPException(500, "Failed to compute multi-view accessibility metrics")

    def _corridor_block(acc_side):
        g = acc_side.global_stats
        # g.free_total_m tem as estatísticas do corredor (pool L∪R)
        return {
            "median_m": g.free_total_m.get("median", float("nan")),
            "meets_ratio": g.meets_120m_ratio,
            "rating": g.rating,
        }

    # ---------- sumarização por lado (formato compacto) ----------
    per_side = {}

    if left is not None:
        lg = acc["LEFT"].global_stats
        per_side["LEFT"] = {
            "median_width": None,  # injeta abaixo com meta
            "corridor": _corridor_block(acc["LEFT"]),
            "obstacles": {
                "typical_obstacles_per_view": (
                    lg.avg_obstacles_per_view_rounded
                    or _round_half_up(lg.avg_obstacles_per_view or 0.0)
                ),
                "types": _types_summary(left) or None,  # {"tree": {"prevalence": ..., "typical_count_when_present": ...}, ...}
            },
        }

    if right is not None:
        rg = acc["RIGHT"].global_stats
        per_side["RIGHT"] = {
            "median_width": None,
            "corridor": _corridor_block(acc["RIGHT"]),
            "obstacles": {
                "typical_obstacles_per_view": (
                    rg.avg_obstacles_per_view_rounded
                    or _round_half_up(rg.avg_obstacles_per_view or 0.0)
                ),
                "types": _types_summary(right) or None,
            },
        }

    # injeta medianas de largura por lado (vêm do metadata calculado no helper run_pipeline)
    lm = (multi_metadata or {}).get("left_median")
    rm = (multi_metadata or {}).get("right_median")
    if lm and "LEFT" in per_side:
        per_side["LEFT"]["median_width"]  = {"width_m": float(lm[0]), "margin_m": float(lm[1])}
    if rm and "RIGHT" in per_side:
        per_side["RIGHT"]["median_width"] = {"width_m": float(rm[0]), "margin_m": float(rm[1])}

    # ---------- bloco agregado ALL ----------
    g_all = acc["ALL"].global_stats
    all_views = {
        "corridor": {
            "median_m": g_all.free_total_m.get("median", float("nan")),
            "meets_ratio": g_all.meets_120m_ratio,
            "rating": g_all.rating,
        },
        "typical_obstacles_per_view": (
            g_all.avg_obstacles_per_view_rounded
            or _round_half_up(g_all.avg_obstacles_per_view or 0.0)
        ),
    }

    # ---------- helpers p/ imagens de amostra (mantidos) ----------
    def _png_triplet(result):
        if getattr(result, "rgb_image", None) is None:
            return None
        rgb_bgr = cv2.cvtColor(result.rgb_image, cv2.COLOR_RGB2BGR)
        mask_u8 = (result.sidewalk_mask * 255).astype("uint8")
        overlay = rgb_bgr.copy()
        overlay[result.sidewalk_mask.astype(bool)] = (0,255,0)
        overlay = cv2.addWeighted(overlay, 0.4, rgb_bgr, 0.6, 0)
        return {
            "gsv_png_b64":  _png_b64(rgb_bgr),
            "mask_png_b64": _png_b64(mask_u8),
            "overlay_png_b64": _png_b64(overlay),
        }

    def _sample_indices(n, k=3):
        if n<=0: return []
        if n<=k: return list(range(n))
        return sorted(set([n//4, n//2, (3*n)//4]))[:k]

    samples_left = samples_right = None
    if req.return_mask:
        if left:
            samples_left  = [s for i in _sample_indices(len(left), 3) if (s := _png_triplet(left[i]))]
        if right:
            samples_right = [s for i in _sample_indices(len(right), 3) if (s := _png_triplet(right[i]))]

    # ---------- retorno final ajustado ao schema ----------
    return MultiRespSlim(
        multi_metadata={"n_headings": (multi_metadata or {}).get("n_headings")},
        per_side=per_side,
        all_views=all_views,
        per_heading=per_heading,
        samples_left=samples_left,
        samples_right=samples_right,
        obstacle_images=obstacle_images,
    )

# ------------------------------------------------------------------ #
# GET /ping                                                          #
# ------------------------------------------------------------------ #
@app.get("/ping", tags=["health"])
def ping():
    return {"ok": True}

# Pydantic forward refs (para versões 1 e 2)
try:
    # Pydantic v2
    SingleResp.model_rebuild()
    MultiRespSlim.model_rebuild()
except AttributeError:
    # Pydantic v1
    SingleResp.update_forward_refs()
    MultiRespSlim.update_forward_refs()
