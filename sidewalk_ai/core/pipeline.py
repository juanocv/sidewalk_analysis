# sidewalk_ai/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from sidewalk_ai.io.image_io import read_rgb
from sidewalk_ai.io.streetview import StreetViewClient, ImageRequest
from sidewalk_ai.processing.refinement import refine_sidewalk_mask
from sidewalk_ai.processing.geometry import (
     WidthResult,
     ClearanceResult,
     compute_width,
     compute_clearances,
     compute_width_from_curbs,
 )
from sidewalk_ai.processing.fusion import logical_fuse
from sidewalk_ai.models.base import Segmenter
from sidewalk_ai.models.midas import MidasEstimator


# --------------------------------------------------------------------------- #
# 0)  Public result dataclass                                                  #
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class Result:
    """What the caller gets back from `SidewalkPipeline`."""
    width: WidthResult
    clearances: Sequence[ClearanceResult]
    sidewalk_mask: np.ndarray          # H×W  uint8  (0/1)
    seg_map: np.ndarray | None = None  # panoptic id map (optional)
    img_path: Path | None = None
    rgb_image: np.ndarray | None = None  # H×W×3  uint8 (RGB)


# --------------------------------------------------------------------------- #
# 1)  Main orchestration class                                                #
# --------------------------------------------------------------------------- #
class SidewalkPipeline:
    """
    High-level, dependency-injected pipeline.

    Parameters
    ----------
    segmenter
        Any object that fulfils the :class:`~sidewalk_ai.models.base.Segmenter`
        protocol (Detectron2, OneFormer, DeepLab, ensemble…).
    depth
        A :class:`~sidewalk_ai.models.midas.MidasEstimator` (or any object that
        implements ``predict(img_rgb) -> np.ndarray``).
    streetview
        A :class:`~sidewalk_ai.io.streetview.StreetViewClient`.  Create **one**
        and reuse it for the lifetime of your app to benefit from HTTP keep-
        alive and the on-disk cache.
    refine
        If *True* (default) apply :func:`refine_sidewalk_mask` to the raw
        sidewalk mask produced by `segmenter`.
    fuse_method
        If `segmenter.segment()` returns *multiple* candidate masks you can
        pass them as an iterable to `logical_fuse()` via this parameter.  For
        normal single-mask back-ends leave it on "none".
    """

    def __init__(
        self,
        *,
        segmenter: Segmenter,
        depth: MidasEstimator,
        streetview: StreetViewClient,
        refine: bool = True,
        fuse_method: str | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.depth_est = depth
        self.sv = streetview
        self.refine = refine
        self.fuse_method = fuse_method

    # ------------------------------------------------------------------ #
    # Convenience overloads                                              #
    # ------------------------------------------------------------------ #
    def analyse_address(self, address: str) -> Result:
        """
        The call your **web app** or CLI will use 99 % of the time.
        """
        img_path = self.sv.fetch(address)
        return self._analyse_path(img_path)

    def analyse_coords(
        self,
        lat: float, lon: float,
        heading: int = 0,
        pitch:   int = 0,               # NEW
        fov:     int = 90,              # optional, keeps default
    ) -> Result:
        req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
        img_path = self.sv.fetch(req)
        return self._analyse_path(img_path)

    # ------------------------------------------------------------------ #
    # Core implementation (private)                                      #
    # ------------------------------------------------------------------ #
    def _analyse_path(self, img_path: Path) -> Result:
        img_rgb = read_rgb(img_path)

        # -------- Segmentation ---------------------------------------- #
        sidewalk_mask, seg_map, _ = self.segmenter.segment(img_rgb)

        # Some back-ends (ensemble) may return a tuple of masks
        if isinstance(sidewalk_mask, Iterable) and not isinstance(sidewalk_mask, np.ndarray):
            sidewalk_mask = logical_fuse(list(sidewalk_mask), method=self.fuse_method or "or")

        if self.refine:
            sidewalk_mask, edge_top, edge_bot = refine_sidewalk_mask(sidewalk_mask)

        # -------- Depth ------------------------------------------------ #
        depth_map = self.depth_est.predict(img_rgb)
        metric = getattr(self.depth_est, "is_metric", False)
        #print(f"Depth map shape: {depth_map.shape}, metric={metric}")
        #use_geom = getattr(self.depth_est, "use_geometry", False)
        width_res = compute_width(sidewalk_mask, depth_map)
        #width_res_1 = compute_width(sidewalk_mask, depth_map, metric_depth=metric, use_geom=True)
        #width_res_2 = compute_width(sidewalk_mask, depth_map, metric_depth=metric, use_geom=False)
        #width_res_3 = compute_width_from_curbs(sidewalk_mask, edge_top, edge_bot)
        #print(
        #    f"Width results:\n"
        #    f"compute_width={width_res_1.width_m:.2f}m (Use_geom = True);\n"
        #    f"compute_width={width_res_2.width_m:.2f}m (Use_geom = False);\n"
        #    f"compute_width_from_curbs={width_res_3.width_m:.2f}m"
        #)
        #if width_res_2.width_m < width_res_1.width_m and width_res_2.width_m < width_res_3.width_m:
        #    width_res = width_res_2
        #elif width_res_3.width_m < width_res_1.width_m and width_res_3.width_m < width_res_2.width_m:
        #    width_res = width_res_3
        #else:
        #    width_res = width_res_1
        # -------- Geometry --------------------------------------------- #
        # Optionally compute obstacle clearances (pass empty list if none)
        clearances = compute_clearances(
            sidewalk_mask,
            obstacles=[],            # supply obstacle masks here when ready
            depth=depth_map,
            metric_depth=metric 
        )

        self._last_rgb = img_rgb  # for debugging

        return Result(
            width=width_res,
            clearances=clearances,
            sidewalk_mask=sidewalk_mask,
            seg_map=seg_map,
            img_path=img_path,
            rgb_image=img_rgb
        )