from __future__ import annotations
import cv2, os
import base64

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import sidewalk_ai as sw
from sidewalk_ai.models.factory import build_depth 

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
    sv  = sw.StreetViewClient()
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


class WidthResp(BaseModel):
    width_m: float
    margin_m: float
    gsv_png_b64:  str  | None = None
    mask_png_b64:  str | None = None
    overlay_png_b64: str  | None = None

def _png_b64(arr: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", arr)[1]).decode()
# ------------------------------------------------------------------ #
# shared helper – runs the pipeline exactly once                     #
# ------------------------------------------------------------------ #
def _run_pipeline(req: "AddressReq") -> sw.core.pipeline.Result:
    # pick the pre-built pipeline
    key = (req.depth, req.refine)
    if key not in app.state.pipes:
        raise HTTPException(400, f"Depth backend '{req.depth}' not available")
    pipe = app.state.pipes[key]
 
    # optional on-the-fly Zoe variant override
    if req.depth == "zoe" and req.zoe_variant:
        # build once then cache in the dict
        v = NAME_MAP.get(req.zoe_variant, None)
        if v is None:
            raise HTTPException(400, "zoe_variant must be one of "
                                       f"{', '.join(VALID_ZOE)}")
        depth_obj = build_depth("zoe", variant=v)
        app.state.pipes[key] = sw.SidewalkPipeline(
            segmenter=pipe.segmenter,
            depth=depth_obj,
            streetview=app.state.sv,      
            refine=req.refine,
        )
        pipe = app.state.pipes[key]

        
    # expose fallback knobs to refinement code
    if req.fallback_scale is not None:
        os.environ["SWAI_FALLBACK_SCALE"] = str(req.fallback_scale)
    if req.force_fallback:
        os.environ["SWAI_FORCE_FALLBACK"] = "1"

    # address vs lat/lon logic
    if req.lat is not None and req.lon is not None:
        return pipe.analyse_coords(
            lat=req.lat, lon=req.lon,
            heading=req.heading, pitch=req.pitch, fov=req.fov
        )
    if req.address:
        return pipe.analyse_address(req.address)

    raise HTTPException(422, "Either address or lat+lon required")

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
            rgb_bgr = cv2.cvtColor(res.rgb_image, cv2.COLOR_RGB2BGR)
            gsv_png_b64  = _png_b64(rgb_bgr)

            mask_u8  = (res.sidewalk_mask * 255).astype("uint8")
            mask_png_b64 = _png_b64(mask_u8)

            mask_bool = res.sidewalk_mask.astype(bool)
            # create an overlay image with the sidewalk mask
            overlay = rgb_bgr.copy()
            overlay[mask_bool] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.4, rgb_bgr, 0.6, 0)
            overlay_png_b64 = _png_b64(overlay)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    return WidthResp(
    width_m  = res.width.width_m,
    margin_m = res.width.margin_m,
    gsv_png_b64     = gsv_png_b64,
    mask_png_b64    = mask_png_b64,
    overlay_png_b64 = overlay_png_b64
    )


# ------------------------------------------------------------------ #
# GET /ping                                                          #
# ------------------------------------------------------------------ #
@app.get("/ping", tags=["health"])
def ping():
    return {"ok": True}