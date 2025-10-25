from __future__ import annotations
import cv2, os
import base64

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sidewalk_ai.processing.accessibility import (
    compute_single_view_metrics, compute_multiview_metrics
)
import numpy as np
from pydantic import BaseModel, Field
import sidewalk_ai as sw
from sidewalk_ai.models.factory import build_depth 
from sidewalk_ai.api.request import from_api_req, run_pipeline

# Map ZoeDepth variant names to their canonical forms
NAME_MAP = {
    "ZoeD_N": "zoed_n",
    "ZoeD_K": "zoed_k",
    "Zoed_NK": "zoed_nk",
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
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Request / response schemas                                         #
# ------------------------------------------------------------------ #
class AddressReq(BaseModel):
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
    pitch:   int = Field(0,  ge=-90, le=90)
    fov:     int = Field(90, ge=10,  le=120)

    # Multi-angle vs single-angle control. If true, run the
    # multi-angle (address-mode) analysis which samples several
    # headings around the street center. If false, the request will
    # perform a single-angle analysis using the provided heading.
    multi_view: bool = Field(True, description="Run multi-angle analysis (default true)")

# ─── depth selection ──────────────────────────────────────────
    depth: str = Field(
        default=os.getenv("SWAI_DEPTH", "midas"),
        pattern="^(midas|zoe)$",
        description="'midas' (default) or 'zoe'",
        example="midas",    
    )
    zoe_variant: str | None = Field(
        default=None,
        description="Override ZoeDepth variant (ZoeD_N, ZoeD_K, ZoeD_NK) "
                    "if depth='zoe'. Ignored otherwise.",
        example="ZoeD_N",
    )

# ─── geometry refinement knobs ────────────────────────────────
    refine: bool = True
    force_fallback: bool  = True
    fallback_scale: float | None = Field(None, gt=0,
                        description="metres-per-px when fallback is used")

# ─── misc ──────────────────────────────────────────────────────
    return_mask: bool = False

# ─── accessibility ─────────────────────────────────────────────
    min_clear: float = Field(1.20, ge=0.0, description="Limiar de caminho livre (m) para rating ABNT/NBR 9050")


class WidthResp(BaseModel):
    width_m: float
    margin_m: float
    clearances: list["ClearanceItem"] = []
    gsv_png_b64:  str  | None = None
    mask_png_b64:  str | None = None
    overlay_png_b64: str  | None = None
    # Optional fields for multi-view responses
    multi_metadata: dict | None = None
    per_heading: list[dict] | None = None
    obstacle_images: list[str] | None = None
    accessibility: dict | None = None


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
def _run_pipeline(req: "AddressReq") -> sw.core.pipeline.Result:
    # Build or pick the appropriate pipeline from app.state.pipes
    key = (req.depth, req.refine)
    if key not in app.state.pipes:
        raise HTTPException(400, f"Depth backend '{req.depth}' not available")

    pipe = app.state.pipes[key]

    # optional on-the-fly Zoe variant override (replace depth in the selected pipe)
    if req.depth == "zoe" and req.zoe_variant:
        v = NAME_MAP.get(req.zoe_variant, None)
        if v is None:
            raise HTTPException(400, "zoe_variant must be one of " f"{', '.join(VALID_ZOE)}")
        depth_obj = build_depth("zoe", variant=v)
        app.state.pipes[key] = sw.SidewalkPipeline(segmenter=pipe.segmenter, depth=depth_obj, streetview=app.state.sv, refine=req.refine)
        pipe = app.state.pipes[key]

    # normalize request and run via helper
    cfg = from_api_req(req)
    try:
        return run_pipeline(pipe, cfg)
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.post("/analyse", response_model=WidthResp)
def analyse(req: AddressReq):
    """
    Returns JSON by default.
    """
    # ---------------- run pipeline ONCE ---------------------------
    try:
        res   = _run_pipeline(req)
        # build Base-64 images only if the client asked for them
        gsv_png_b64     = None
        mask_png_b64    = None
        overlay_png_b64 = None
        if req.return_mask:
            # The pipeline may return a single Result or a tuple (left_list, right_list)
            # In the multi-view case pick a representative Result to build images.
            rep = None
            if isinstance(res, tuple) and len(res) == 2:
                left, right = res
                # choose the middle estimate of left if available, else right
                if left:
                    rep = left[len(left) // 2]
                elif right:
                    rep = right[len(right) // 2]
            else:
                rep = res

            if rep is not None and rep.rgb_image is not None:
                rgb_bgr = cv2.cvtColor(rep.rgb_image, cv2.COLOR_RGB2BGR)
                gsv_png_b64  = _png_b64(rgb_bgr)

                mask_u8  = (rep.sidewalk_mask * 255).astype("uint8")
                mask_png_b64 = _png_b64(mask_u8)

                mask_bool = rep.sidewalk_mask.astype(bool)
                # create an overlay image with the sidewalk mask
                overlay = rgb_bgr.copy()
                overlay[mask_bool] = (0, 255, 0)
                overlay = cv2.addWeighted(overlay, 0.4, rgb_bgr, 0.6, 0)
                overlay_png_b64 = _png_b64(overlay)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    # map dataclass (slots=True) -> Pydantic model for single-result
    multi_metadata = None
    per_heading = None
    obstacle_images = None

    if isinstance(res, dict) and 'results' in res:
        # Rich multi-view return from helper
        multi_metadata = res.get('metadata')
        per_heading = res.get('per_heading')
        obstacle_images = res.get('obstacle_images')
        # choose a representative Result object for width/clearances
        inner = None
        r = res.get('results')
        if isinstance(r, tuple) and len(r) == 2:
            left, right = r
            if left:
                inner = left[len(left)//2]
            elif right:
                inner = right[len(right)//2]
        elif isinstance(r, list) and r:
            inner = r[len(r)//2]
        else:
            inner = None
        if inner is None:
            raise HTTPException(404, "No estimates found in multi-view result")
        res_obj = inner
        # ----- Accessibility (MULTI) -----
        accessibility = None
        try:
            left_list, right_list = res.get('results', ([], []))
            acc = compute_multiview_metrics(left_list, right_list, min_clear_required_m=req.min_clear)
            accessibility = {
                "LEFT": {
                    "min_clear_required_m": acc["LEFT"].min_clear_required_m,
                    "global_stats": acc["LEFT"].global_stats.__dict__,
                    "per_type": {k: v.__dict__ for k, v in acc["LEFT"].per_type.items()},
                },
                "RIGHT": {
                    "min_clear_required_m": acc["RIGHT"].min_clear_required_m,
                    "global_stats": acc["RIGHT"].global_stats.__dict__,
                    "per_type": {k: v.__dict__ for k, v in acc["RIGHT"].per_type.items()},
                },
                "ALL": {
                    "min_clear_required_m": acc["ALL"].min_clear_required_m,
                    "global": acc["ALL"].global_stats.__dict__,
                    "per_type": {k: v.__dict__ for k, v in acc["ALL"].per_type.items()},
                },
            }
        except Exception:
            accessibility = None
    else:
        res_obj = res
        # ----- Accessibility (SINGLE) -----
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
        )
        for c in getattr(res_obj, 'clearances', [])
    ]

    return WidthResp(
        width_m  = getattr(res_obj.width, 'width_m', 0.0),
        margin_m = getattr(res_obj.width, 'margin_m', 0.0),
        clearances      = clearance_items,
        gsv_png_b64     = gsv_png_b64,
        mask_png_b64    = mask_png_b64,
        overlay_png_b64 = overlay_png_b64,
        multi_metadata  = multi_metadata,
        per_heading     = per_heading,
        obstacle_images = obstacle_images,
        accessibility   = accessibility
    )


# ------------------------------------------------------------------ #
# GET /ping                                                          #
# ------------------------------------------------------------------ #
@app.get("/ping", tags=["health"])
def ping():
    return {"ok": True}