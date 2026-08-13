"""
Sidewalk-AI Web API.

Run it with::

    python -m pip install -e ".[api,ml]"
    uvicorn sidewalk_ai.webapi:app --host 127.0.0.1 --port 8000

Endpoints and schemas are documented at ``/docs`` once the server is up; see
also ``docs/webapi.md``.

Concurrency
-----------
The endpoints are synchronous, so Starlette runs them in a worker thread pool
and several requests can overlap. The segmentation and depth models behind them
are neither thread-safe nor cheap in VRAM, so model work is serialised through a
semaphore sized by ``SWAI_API_MAX_CONCURRENCY`` (default 1). Raise it only if the
selected back-ends are known to tolerate concurrent inference on your hardware.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sidewalk_ai as sw
from sidewalk_ai.api.request import from_api_multi, from_api_single, run_pipeline
from sidewalk_ai.io.image_io import objects_overlay_bgr, png_b64, png_triplet, sample_indices
from sidewalk_ai.log import configure_logging, get_logger
from sidewalk_ai.models.factory import build_depth
from sidewalk_ai.processing.accessibility import (
    compute_multiview_metrics,
    compute_single_view_metrics,
    corridor_block,
    round_half_up,
    types_summary,
)

configure_logging(force=False)
logger = get_logger(__name__)

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

DEFAULT_DEPTH = os.getenv("SWAI_DEPTH", "zoe")
DEFAULT_ZOE_VARIANT = NAME_MAP.get(os.getenv("SWAI_ZOE_VARIANT", "zoed_n"), "zoed_n")

# Serialises model work; see the module docstring.
MAX_CONCURRENCY = max(1, int(os.getenv("SWAI_API_MAX_CONCURRENCY", "1")))
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)

# "*" keeps the bundled index.html working when opened from disk. Narrow it with
# a comma-separated list once the front-end has a fixed origin.
CORS_ORIGINS = [o.strip() for o in os.getenv("SWAI_API_CORS_ORIGINS", "*").split(",") if o.strip()]


@contextmanager
def _inference_slot():
    """Hold one of the model-inference slots for the duration of the block."""
    acquired = _inference_slots.acquire(timeout=float(os.getenv("SWAI_API_QUEUE_TIMEOUT_S", "300")))
    if not acquired:
        raise HTTPException(503, "Server busy: inference queue timed out")
    try:
        yield
    finally:
        _inference_slots.release()


class PipelineRegistry:
    """
    Lazily builds and caches one pipeline per (depth back-end, variant, refine).

    The previous implementation kept a fixed dict keyed by (depth, refine) and
    *replaced* an entry whenever a request asked for a different ZoeDepth
    variant. That leaked one request's variant into every later request and
    rebuilt the model on each call. Keying on the variant fixes both.
    """

    def __init__(self, segmenter: Any, streetview: Any) -> None:
        self._segmenter = segmenter
        self._streetview = streetview
        self._depth_models: dict[tuple[str, str | None], Any] = {}
        self._pipes: dict[tuple[str, str | None, bool], Any] = {}
        self._lock = threading.Lock()

    def get(self, backend: str, variant: str | None, refine: bool):
        key = (backend, variant, refine)
        with self._lock:
            pipe = self._pipes.get(key)
            if pipe is not None:
                return pipe

            depth = self._depth_models.get((backend, variant))
            if depth is None:
                logger.info("Loading depth back-end %s (variant=%s)", backend, variant)
                depth = build_depth(backend, variant=variant)
                self._depth_models[(backend, variant)] = depth

            pipe = sw.SidewalkPipeline(
                segmenter=self._segmenter,
                depth=depth,
                streetview=self._streetview,
                refine=refine,
            )
            self._pipes[key] = pipe
            return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Sidewalk AI API (max_concurrency=%s)", MAX_CONCURRENCY)
    segmenter = sw.build_segmenter("oneformer")
    streetview = sw.StreetViewClient()

    app.state.sv = streetview
    app.state.registry = PipelineRegistry(segmenter, streetview)

    # Warm the configured default so the first request does not pay for the
    # model load. Other combinations are built on demand.
    default_variant = DEFAULT_ZOE_VARIANT if DEFAULT_DEPTH == "zoe" else None
    app.state.registry.get(DEFAULT_DEPTH, default_variant, True)
    logger.info("Sidewalk AI API ready (default depth=%s)", DEFAULT_DEPTH)

    yield


app = FastAPI(
    title="Sidewalk-AI",
    version="0.1.0",
    description="Automatic sidewalk width estimation and obstacle detection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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
        json_schema_extra={"example": "Av. Paulista 1578, São Paulo"},
        description="Ignored if lat+lon are given",
    )
    # ─── option B: explicit Street-View coordinates ───────────────
    lat: float | None = Field(None, description="Latitude  (decimal deg)")
    lon: float | None = Field(None, description="Longitude (decimal deg)")
    heading: int = Field(0, ge=0, le=359)
    pitch: int = Field(-10, ge=-90, le=90)
    fov: int = Field(90, ge=10, le=120)

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
        json_schema_extra={"example": "zoe"},
    )
    zoe_variant: str | None = Field(
        default=None,
        description="Override ZoeDepth variant (ZoeD_N, ZoeD_K, ZoeD_NK) "
        "if depth='zoe'. Ignored otherwise.",
        json_schema_extra={"example": "ZoeD_N"},
    )

    # ─── geometry refinement knobs ────────────────────────────────
    refine: bool = True
    force_fallback: bool = False
    fallback_scale: float | None = Field(
        None, gt=0, description="metres-per-px when fallback is used"
    )

    # ─── misc ──────────────────────────────────────────────────────
    return_mask: bool = False

    # ─── accessibility ─────────────────────────────────────────────
    min_clear: float = Field(
        1.20, ge=0.0, description="Limiar de caminho livre (m) para rating ABNT/NBR 9050"
    )


class AddressMultiReq(BaseModel):
    # Address OU lat/lon
    address: str | None = Field(
        default=None, json_schema_extra={"example": "Av. Paulista 1578, São Paulo"}
    )
    lat: float | None = Field(None)
    lon: float | None = Field(None)
    # heading não é usado em multi; pitch/fov são aceitos
    pitch: int = Field(-10, ge=-90, le=90)
    fov: int = Field(90, ge=10, le=120)
    depth: str = Field(default=os.getenv("SWAI_DEPTH", "zoe"), pattern="^(zoe|midas)$")
    zoe_variant: str | None = Field(
        default=None,
        description="Override ZoeDepth variant (ZoeD_N, ZoeD_K, ZoeD_NK) "
        "if depth='zoe'. Ignored otherwise.",
        json_schema_extra={"example": "ZoeD_N"},
    )
    refine: bool = True
    force_fallback: bool = False
    fallback_scale: float | None = Field(None, gt=0)
    return_mask: bool = False
    min_clear: float = Field(1.20, ge=0.0)


class SingleResp(BaseModel):
    width_m: float
    margin_m: float
    clearances: list["ClearanceItem"] = []
    gsv_png_b64: str | None = None
    overlay_sidewalk_png_b64: str | None = None
    overlay_obstacle_png_b64: str | None = None
    accessibility: dict | None = None


class MultiSideCorridor(BaseModel):
    median_m: float | None = None
    meets_ratio: float | None = None
    rating: str | None = None


class MultiSideObstacles(BaseModel):
    typical_obstacles_per_view: int | None = None
    types: dict[str, dict] | None = (
        None  # {"tree": {"prevalence": 1.0, "typical_count_when_present": 3}, ...}
    )


class MultiSideSummary(BaseModel):
    median_width: dict | None = None  # {"width_m": ..., "margin_m": ...}
    width_range_m: dict | None = None  # {"min_m": ..., "max_m": ...}
    corridor: MultiSideCorridor | None = None
    obstacles: MultiSideObstacles | None = None


class MultiRespSlim(BaseModel):
    # metadados mínimos (sem redundância)
    multi_metadata: dict | None = None  # {"n_headings": {"left": N, "right": M}}
    # sumarização por lado
    per_side: dict[str, MultiSideSummary] | None = None  # "LEFT"/"RIGHT"
    # sumarização agregada
    all_views: dict | None = None  # {"corridor": {...}, "typical_obstacles_per_view": int}
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
# shared helpers                                                     #
# ------------------------------------------------------------------ #
def _resolve_variant(depth: str, zoe_variant: str | None) -> str | None:
    """Canonical ZoeDepth variant for a request, or None for other back-ends."""
    if depth != "zoe":
        return None
    if not zoe_variant:
        return DEFAULT_ZOE_VARIANT
    canonical = NAME_MAP.get(zoe_variant)
    if canonical is None:
        raise HTTPException(400, f"zoe_variant must be one of {', '.join(VALID_ZOE)}")
    return canonical


def _pick_depth_pipe(req_depth: str, req_refine: bool, zoe_variant: str | None):
    """Pipeline for this request. Never mutates another request's pipeline."""
    variant = _resolve_variant(req_depth, zoe_variant)
    try:
        return app.state.registry.get(req_depth, variant, req_refine)
    except ModuleNotFoundError as exc:
        raise HTTPException(503, f"Depth backend '{req_depth}' is not installed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _run(pipe, cfg):
    """Execute the pipeline under the inference semaphore, mapping errors."""
    with _inference_slot():
        try:
            return run_pipeline(pipe, cfg)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post("/analyse/single", response_model=SingleResp)
def analyse_single(req: AddressSingleReq):
    pipe = _pick_depth_pipe(req.depth, req.refine, req.zoe_variant)
    res = _run(pipe, from_api_single(req))  # sempre um Result em single

    # single-view → sempre um Result
    res_obj = res
    gsv_png_b64 = None
    overlay_sidewalk_png_b64 = None
    overlay_obstacle_png_b64 = None

    if req.return_mask and getattr(res_obj, "rgb_image", None) is not None:
        rgb_bgr = cv2.cvtColor(res_obj.rgb_image, cv2.COLOR_RGB2BGR)
        gsv_png_b64 = png_b64(rgb_bgr)

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
        acc = compute_single_view_metrics(
            getattr(res_obj, "clearances", []) or [], min_clear_required_m=req.min_clear
        )
        accessibility = acc.to_dict()
    except Exception:
        logger.exception("Failed to compute single-view accessibility metrics")
        accessibility = None

    clearance_items = [
        ClearanceItem(
            label=c.label,
            L_m=c.L_m if hasattr(c, "L_m") else None,
            R_m=c.R_m if hasattr(c, "R_m") else None,
            total_m=c.total_m if hasattr(c, "total_m") else None,
            obs_width=c.obs_width if hasattr(c, "obs_width") else None,
        )
        for c in getattr(res_obj, "clearances", [])
    ]

    return SingleResp(
        width_m=getattr(res_obj.width, "width_m", 0.0),
        margin_m=getattr(res_obj.width, "margin_m", 0.0),
        clearances=clearance_items,
        gsv_png_b64=gsv_png_b64,
        overlay_sidewalk_png_b64=overlay_sidewalk_png_b64,
        overlay_obstacle_png_b64=overlay_obstacle_png_b64,
        accessibility=accessibility,
    )


# ------------------------------------------------------------------ #
# POST /analyse/multi                                               #
# ------------------------------------------------------------------ #
# # helper functions                                                #
# ------------------------------------------------------------------ #


@app.post("/analyse/multi", response_model=MultiRespSlim)
def analyse_multi(req: AddressMultiReq):
    pipe = _pick_depth_pipe(req.depth, req.refine, req.zoe_variant)
    # dict com 'results', 'metadata', 'per_heading', ...
    res = _run(pipe, from_api_multi(req))

    if not (isinstance(res, dict) and "results" in res):
        raise HTTPException(500, "Unexpected pipeline output for multi-view")

    multi_metadata = res.get("metadata") or {}
    per_heading = res.get("per_heading") or []
    left, right = res.get("results", ([], []))

    # ---------- acessibilidade (LEFT/RIGHT/ALL) ----------
    try:
        acc = compute_multiview_metrics(left, right, min_clear_required_m=req.min_clear)
    except Exception as exc:
        logger.exception("Failed to compute multi-view accessibility metrics")
        raise HTTPException(500, "Failed to compute multi-view accessibility metrics") from exc

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
                "types": types_summary(left)
                or None,  # {"tree": {"prevalence": ..., "typical_count_when_present": ...}, ...}
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
        per_side["LEFT"]["median_width"] = {"width_m": float(lm[0]), "margin_m": float(lm[1])}
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
            }
            if aw_range
            else None
        ),
    }

    samples_left = samples_right = None
    if req.return_mask:
        if left:
            samples_left = [s for i in sample_indices(len(left), 3) if (s := png_triplet(left[i]))]
        if right:
            samples_right = [
                s for i in sample_indices(len(right), 3) if (s := png_triplet(right[i]))
            ]

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
