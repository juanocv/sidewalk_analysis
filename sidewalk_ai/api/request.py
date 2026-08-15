from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import os
import base64
import cv2
import numpy as np
from sidewalk_ai.core.pipeline import DepthScale
from sidewalk_ai.io.streetview import ImageRequest
from sidewalk_ai.log import get_logger

logger = get_logger(__name__)


@dataclass
class RequestConfig:
    multi_view: bool
    address: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    heading: int
    pitch: int
    fov: int
    depth: str
    zoe_variant: Optional[str]
    refine: bool
    force_fallback: bool
    fallback_scale: Optional[float]
    return_mask: bool


def from_cli_args(args) -> RequestConfig:
    return RequestConfig(
        multi_view=getattr(args, "multi_view", True),
        address=getattr(args, "address", None),
        lat=getattr(args, "lat", None),
        lon=getattr(args, "lon", None),
        heading=getattr(args, "heading", 0),
        pitch=getattr(args, "pitch", 0),
        fov=getattr(args, "fov", 90),
        depth=getattr(args, "depth", os.getenv("SWAI_DEPTH", "midas")),
        zoe_variant=getattr(args, "zoe_variant", None),
        refine=getattr(args, "refine", True),
        force_fallback=getattr(args, "force_fallback", False),
        fallback_scale=getattr(args, "fallback_scale", None),
        return_mask=getattr(args, "return_mask", False),
    )


def from_api_single(req) -> RequestConfig:
    """Normaliza um request *single-view* para o run_pipeline."""
    return RequestConfig(
        multi_view=False,
        address=getattr(req, "address", None),
        lat=getattr(req, "lat", None),
        lon=getattr(req, "lon", None),
        heading=int(getattr(req, "heading", 0) or 0),
        pitch=int(getattr(req, "pitch", 0) or 0),
        fov=int(getattr(req, "fov", 90) or 90),
        depth=getattr(req, "depth", os.getenv("SWAI_DEPTH", "midas")),
        zoe_variant=getattr(req, "zoe_variant", None),
        refine=bool(getattr(req, "refine", True)),
        force_fallback=bool(getattr(req, "force_fallback", False)),
        fallback_scale=getattr(req, "fallback_scale", None),
        return_mask=bool(getattr(req, "return_mask", False)),
    )


def from_api_multi(req) -> RequestConfig:
    """Normaliza um request *multi-view* para o run_pipeline."""
    return RequestConfig(
        multi_view=True,
        address=getattr(req, "address", None),
        lat=getattr(req, "lat", None),
        lon=getattr(req, "lon", None),
        heading=0,  # ignorado em multi-view
        pitch=int(getattr(req, "pitch", 0) or 0),
        fov=int(getattr(req, "fov", 90) or 90),
        depth=getattr(req, "depth", os.getenv("SWAI_DEPTH", "midas")),
        zoe_variant=getattr(req, "zoe_variant", None),
        refine=bool(getattr(req, "refine", True)),
        force_fallback=bool(getattr(req, "force_fallback", False)),
        fallback_scale=getattr(req, "fallback_scale", None),
        return_mask=bool(getattr(req, "return_mask", False)),
    )


def run_pipeline(pipe, cfg: RequestConfig):
    """
    Run the given SidewalkPipeline according to the normalized RequestConfig.

    This helper centralizes the branching between multi-view address-mode and
    single-view (explicit heading) execution. It also applies geocoding when
    an address is provided and latitude/longitude are needed.
    """
    # Depth-scale knobs travel as an explicit argument. They used to be written
    # into os.environ, which leaked one request's settings into every other
    # request served by the same process.
    depth_scale = DepthScale(
        fallback_scale=cfg.fallback_scale,
        force_fallback=cfg.force_fallback,
    )

    # Multi-view: prefer address or lat/lon; pipeline handles sampling
    if cfg.multi_view:
        if cfg.lat is not None and cfg.lon is not None:
            # novo pipeline: use multi_view=True em analyse_coords
            out = pipe.analyse_coords(
                lat=cfg.lat,
                lon=cfg.lon,
                pitch=cfg.pitch,
                fov=cfg.fov,
                multi_view=True,
                depth_scale=depth_scale,
            )
        elif cfg.address:
            # novo método específico de multi-view por endereço
            out = pipe.analyse_address_multiview(
                cfg.address, pitch=cfg.pitch, fov=cfg.fov, depth_scale=depth_scale
            )
        else:
            raise ValueError("Either address or lat+lon required for multi-view")

        # Expect out to be (left_list, right_list) or similar. Build metadata
        left, right = out if isinstance(out, tuple) and len(out) == 2 else ([], [])

        def _median_of_estimates(estimates):
            if not estimates:
                return None
            # cole pares, mas só mantenha os que têm width finita e >0
            pairs = [(e.width.width_m, e.width.margin_m) for e in estimates]
            pairs = [(w, m) for (w, m) in pairs if w is not None and np.isfinite(w) and w > 0]
            if not pairs:
                return None
            widths = [w for (w, m) in pairs]  # já estão filtradas
            margins = [m for (w, m) in pairs if m is not None and np.isfinite(m)]
            med_w = float(np.median(widths)) if widths else float("nan")
            med_m = float(np.median(margins)) if margins else float("nan")
            return med_w, med_m

        def _width_range_of_estimates(estimates):
            """
            Retorna uma faixa robusta [lo, hi] de larguras prováveis para um conjunto
            de estimativas multi-view. Usa percentis 10–90 quando há dados suficientes
            e min/max nos casos com poucas amostras.
            """
            if not estimates:
                return None
            widths = [
                float(e.width.width_m)
                for e in estimates
                if getattr(e, "width", None) is not None
                and getattr(e.width, "width_m", None) is not None
                and np.isfinite(e.width.width_m)
                and e.width.width_m > 0
            ]
            if not widths:
                return None
            x = np.asarray(widths, dtype=float)
            if x.size >= 4:
                lo = float(np.percentile(x, 10))
                hi = float(np.percentile(x, 90))
            else:
                lo = float(np.min(x))
                hi = float(np.max(x))
            return lo, hi

        meta = {
            "left_median": _median_of_estimates(left),
            "right_median": _median_of_estimates(right),
            "left_width_range": _width_range_of_estimates(left),
            "right_width_range": _width_range_of_estimates(right),
            "all_width_range": _width_range_of_estimates((left or []) + (right or [])),
            "n_headings": {"left": len(left), "right": len(right)},
        }

        # Build per-heading summaries and obstacle images (base64) for any
        # estimate that has clearances (considered obstacles).
        per_heading = []
        obstacle_images = []

        for side_name, lst in (("left", left), ("right", right)):
            for i, est in enumerate(lst):
                ch = {
                    "side": side_name,
                    "index": i,
                    "heading_deg": getattr(est, "heading", None),
                    "width_m": getattr(est.width, "width_m", None),
                    "margin_m": getattr(est.width, "margin_m", None),
                    "n_clearances": (
                        len(est.clearances) if getattr(est, "clearances", None) is not None else 0
                    ),
                    "clearances": [
                        {
                            "label": c.label,
                            "total_m": getattr(c, "total_m", None),
                            "obs_width": getattr(c, "obs_width", None),
                            "L_m": getattr(c, "L_m", None),
                            "R_m": getattr(c, "R_m", None),
                        }
                        for c in getattr(est, "clearances", [])
                    ],
                }
                per_heading.append(ch)

                # If this estimate reports any clearances, produce a simple
                # overlay image (sidewalk mask + notice) and return base64.
                if getattr(est, "clearances", None) and len(est.clearances) > 0:
                    if getattr(est, "rgb_image", None) is not None:
                        rgb = est.rgb_image
                        try:
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            overlay = bgr.copy()
                            mask_bool = getattr(est, "sidewalk_mask", None)
                            if mask_bool is not None:
                                mask_bool = mask_bool.astype(bool)
                                overlay[mask_bool] = (0, 255, 0)
                            # draw a red header bar indicating obstacles
                            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 24), (0, 0, 255), -1)
                            cv2.putText(
                                overlay,
                                f"OBSTACLES: {len(est.clearances)}",
                                (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            obstacle_images.append(
                                base64.b64encode(cv2.imencode(".png", overlay)[1]).decode()
                            )
                        except Exception as exc:
                            # The overlay is a convenience; the numbers for
                            # this heading are already in per_heading.
                            logger.debug(
                                "Could not render the overlay for %s#%s: %s", side_name, i, exc
                            )

        result = {
            "results": out,
            "metadata": meta,
            "per_heading": per_heading,
            "obstacle_images": obstacle_images,
        }
        return result

    # Single-view: fetch a single Street View image for the requested heading
    # and run the pipeline analysis on that image. This avoids the
    # multi-heading sampling performed in `analyse_coords`.
    if cfg.lat is not None and cfg.lon is not None:
        req = ImageRequest(
            lat=cfg.lat, lon=cfg.lon, heading=cfg.heading, pitch=cfg.pitch, fov=cfg.fov
        )
        img_path = pipe.sv.fetch(req)
        return pipe.analyse_image(
            img_path,
            pitch=cfg.pitch,
            fov=cfg.fov,
            heading=cfg.heading,
            depth_scale=depth_scale,
        )

    if cfg.address:
        lat, lon = pipe.sv.geocode(cfg.address)
        req = ImageRequest(lat=lat, lon=lon, heading=cfg.heading, pitch=cfg.pitch, fov=cfg.fov)
        img_path = pipe.sv.fetch(req)
        return pipe.analyse_image(
            img_path,
            pitch=cfg.pitch,
            fov=cfg.fov,
            heading=cfg.heading,
            depth_scale=depth_scale,
        )

    raise ValueError("Either address or lat+lon required")
