from __future__ import annotations
from pathlib import Path
import base64

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import sidewalk_ai as sw

app = FastAPI(
    title="Sidewalk-AI",
    version="0.1.0",
    description="Automatic sidewalk width estimation and obstacle detection.",
)

# ------------------------------------------------------------------ #
# initialise heavy stuff once on startup                             #
# ------------------------------------------------------------------ #
@app.on_event("startup")
def _load():
    seg   = sw.build_segmenter("oneformer")
    depth = sw.MidasEstimator()
    sv    = sw.StreetViewClient()
    app.state.pipe = sw.SidewalkPipeline(segmenter=seg, depth=depth, streetview=sv)


# ------------------------------------------------------------------ #
# Request / response schemas                                         #
# ------------------------------------------------------------------ #
class AddressReq(BaseModel):
    address: str = Field(example="Av. Paulista 1578, São Paulo")
    return_mask: bool = False


class WidthResp(BaseModel):
    width_m: float
    margin_m: float
    mask_png_b64: str | None = None


# ------------------------------------------------------------------ #
# POST /analyse                                                      #
# ------------------------------------------------------------------ #
@app.post("/analyse", response_model=WidthResp)
def analyse(req: AddressReq):
    try:
        res = app.state.pipe.analyse_address(req.address)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    png_b64 = None
    if req.return_mask:
        import cv2
        import numpy as np

        mask_png = cv2.imencode(".png", (res.sidewalk_mask * 255).astype("uint8"))[1]
        png_b64 = base64.b64encode(mask_png).decode()

    return WidthResp(
        width_m=res.width.width_m,
        margin_m=res.width.margin_m,
        mask_png_b64=png_b64,
    )


# ------------------------------------------------------------------ #
# GET /ping                                                          #
# ------------------------------------------------------------------ #
@app.get("/ping", tags=["health"])
def ping():
    return {"ok": True}