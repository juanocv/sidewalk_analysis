from __future__ import annotations
import cv2, os

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sidewalk_ai.io.image_io import (
    objects_overlay_bgr, png_b64, png_triplet, sample_indices
)
from sidewalk_ai.processing.accessibility import (
    corridor_block, round_half_up, types_summary, compute_single_view_metrics, compute_multiview_metrics
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
    overlay_sidewalk_png_b64: str  | None = None
    overlay_obstacle_png_b64: str  | None = None
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
    width_range_m: dict | None = None     # {"min_m": ..., "max_m": ...}
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

class ClearanceItem(BaseModel):
    label: str
    L_m: float | None = None
    R_m: float | None = None
    total_m: float | None = None
    obs_width: float | None = None

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
    gsv_png_b64 = None; overlay_sidewalk_png_b64 = None; overlay_obstacle_png_b64 = None

    if req.return_mask and getattr(res_obj, "rgb_image", None) is not None:
        rgb_bgr = cv2.cvtColor(res_obj.rgb_image, cv2.COLOR_RGB2BGR)
        gsv_png_b64  = png_b64(rgb_bgr)

        # 1) overlay da calçada → overlay_sidewalk_png_b64
        overlay = rgb_bgr.copy()
        overlay[res_obj.sidewalk_mask.astype(bool)] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.4, rgb_bgr, 0.6, 0)
        overlay_sidewalk_png_b64 = png_b64(overlay)

        # 2) overlay de objetos por tipo → vai em overlay_obstacle_png_b64
        obj_overlay = objects_overlay_bgr(rgb_bgr, getattr(res_obj, "obstacles", None))
        overlay_obstacle_png_b64 = png_b64(obj_overlay)

    # acessibilidade (single)
    accessibility = None
    try:
        acc = compute_single_view_metrics(getattr(res_obj, 'clearances', []) or [],
                                          min_clear_required_m=req.min_clear)
        accessibility = acc.to_dict()
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
        overlay_sidewalk_png_b64 = overlay_sidewalk_png_b64,
        overlay_obstacle_png_b64 = overlay_obstacle_png_b64,
        accessibility   = accessibility,
    )

# ------------------------------------------------------------------ #
# POST /analyse/multi                                               #
# ------------------------------------------------------------------ #
# # helper functions                                                #
# ------------------------------------------------------------------ #

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
    left, right    = res.get('results', ([], []))

    # ---------- acessibilidade (LEFT/RIGHT/ALL) ----------
    try:
        acc = compute_multiview_metrics(left, right, min_clear_required_m=req.min_clear)
    except Exception:
        raise HTTPException(500, "Failed to compute multi-view accessibility metrics")

    # ---------- sumarização por lado (formato compacto) ----------
    per_side = {}

    if left is not None:
        lg = acc["LEFT"].global_stats
        per_side["LEFT"] = {
            "median_width": None,  # injeta abaixo com meta
            "width_range_m": None,
            "corridor": corridor_block(acc["LEFT"]),
            "obstacles": {
                "typical_obstacles_per_view": (
                    lg.avg_obstacles_per_view_rounded
                    or round_half_up(lg.avg_obstacles_per_view or 0.0)
                ),
                "types": types_summary(left) or None,  # {"tree": {"prevalence": ..., "typical_count_when_present": ...}, ...}
            },
        }

    if right is not None:
        rg = acc["RIGHT"].global_stats
        per_side["RIGHT"] = {
            "median_width": None,
            "width_range_m": None,
            "corridor": corridor_block(acc["RIGHT"]),
            "obstacles": {
                "typical_obstacles_per_view": (
                    rg.avg_obstacles_per_view_rounded
                    or round_half_up(rg.avg_obstacles_per_view or 0.0)
                ),
                "types": types_summary(right) or None,
            },
        }

    # injeta medianas de largura por lado (vêm do metadata calculado no helper run_pipeline)
    lm = (multi_metadata or {}).get("left_median")
    rm = (multi_metadata or {}).get("right_median")
    if lm and "LEFT" in per_side:
        per_side["LEFT"]["median_width"]  = {"width_m": float(lm[0]), "margin_m": float(lm[1])}
    if rm and "RIGHT" in per_side:
        per_side["RIGHT"]["median_width"] = {"width_m": float(rm[0]), "margin_m": float(rm[1])}

    # injeta faixas de largura por lado e global (quando disponíveis)
    lw_range = (multi_metadata or {}).get("left_width_range")
    rw_range = (multi_metadata or {}).get("right_width_range")
    aw_range = (multi_metadata or {}).get("all_width_range")
    if lw_range and "LEFT" in per_side:
        per_side["LEFT"]["width_range_m"] = {
            "min_m": float(lw_range[0]),
            "max_m": float(lw_range[1]),
        }
    if rw_range and "RIGHT" in per_side:
        per_side["RIGHT"]["width_range_m"] = {
            "min_m": float(rw_range[0]),
            "max_m": float(rw_range[1]),
        }

    # ---------- bloco agregado ALL ----------
    g_all = acc["ALL"].global_stats
    all_views = {
        "corridor": {
            "median_m": g_all.free_total_m.get("median", float("nan")),
            "meets_ratio": g_all.meets_ratio,
            "rating": g_all.rating,
        },
        "typical_obstacles_per_view": (
            g_all.avg_obstacles_per_view_rounded
            or round_half_up(g_all.avg_obstacles_per_view or 0.0)
        ),
        "width_range_m": (
            {
                "min_m": float(aw_range[0]),
                "max_m": float(aw_range[1]),
            } if aw_range else None
        ),
    }

    samples_left = samples_right = None
    if req.return_mask:
        if left:
            samples_left  = [s for i in sample_indices(len(left), 3) if (s := png_triplet(left[i]))]
        if right:
            samples_right = [s for i in sample_indices(len(right), 3) if (s := png_triplet(right[i]))]

    # ---------- retorno final ajustado ao schema ----------
    return MultiRespSlim(
        multi_metadata={"n_headings": (multi_metadata or {}).get("n_headings")},
        per_side=per_side,
        all_views=all_views,
        per_heading=per_heading,
        samples_left=samples_left,
        samples_right=samples_right,
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
