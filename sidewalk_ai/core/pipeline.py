# sidewalk_ai/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Sequence, Literal

from matplotlib import pyplot as plt
import numpy as np

from sidewalk_ai.processing.refinement import refine_sidewalk_mask
from sidewalk_ai.io.image_io import read_rgb
from sidewalk_ai.io.streetview import StreetViewClient, ImageRequest
from sidewalk_ai.models._obstacles import extract_obstacles
from sidewalk_ai.processing.geometry import (
     WidthResult,
     ClearanceResult,
     _has_two_curbs,
     compute_width,
     compute_clearances,
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
    refined_mask: np.ndarray | None = None  # H×W  uint8  (0/1)
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
        lat, lon = self.sv.geocode(address)
        center_heading = self._find_street_center(lat=lat, lon=lon)
        self._generate_heading_ranges(center_heading)
        left_headings, right_headings = self._generate_heading_ranges(center_heading)

        left_estimates = []
        right_estimates = []
        for heading in left_headings:
            req = ImageRequest(lat, lon, heading=heading)
            img_path = self.sv.fetch(req)
            left_estimates.append(self._analyse_path(img_path))

        for heading in right_headings:
            req = ImageRequest(lat, lon, heading=heading)
            img_path = self.sv.fetch(req)
            right_estimates.append(self._analyse_path(img_path))

        median_left = self._calculate_median_width(left_estimates)
        median_right = self._calculate_median_width(right_estimates)

        return left_estimates, right_estimates

    def analyse_coords(
        self,
        lat: float, lon: float,
        heading: int = 0,
        pitch:   int = 0,               # NEW
        fov:     int = 90,              # optional, keeps default
    ) -> Result:
        center_heading = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
        self._generate_heading_ranges(center_heading)
        left_headings, right_headings = self._generate_heading_ranges(center_heading)

        left_estimates = []
        right_estimates = []
        for heading in left_headings:
            req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
            img_path = self.sv.fetch(req)
            left_estimates.append(self._analyse_path(img_path))

        for heading in right_headings:
            req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
            img_path = self.sv.fetch(req)
            right_estimates.append(self._analyse_path(img_path))

        median_left = self._calculate_median_width(left_estimates)
        median_right = self._calculate_median_width(right_estimates)

        return left_estimates, right_estimates
    
    def _calculate_median_width(
        self,
        estimates: Sequence[Result],
    ) -> Result:
        pass  # TODO: implement median width calculation from multiple estimates

    def _find_street_center(
        self,
        lat: float | None = None,
        lon: float | None = None,
        address: str | None = None,
        pitch: int = 0,
        fov: int = 90,
        test_angles: list[int] | None = None,
    ) -> int | None:
        """
        Find the heading that shows the street center (two sidewalks visible).
        
        Parameters
        ----------
        lat, lon : float
            Location coordinates
        pitch, fov : int
            Camera parameters
        test_angles : list[int], optional
            Headings to test. Defaults to [0, 90, 180, 270]
        
        Returns
        -------
        int or None
            Best heading showing street center, or None if not found
        """
        if test_angles is None:
            test_angles = [0, 90, 180, 270]
        
        print(f"Testing {len(test_angles)} angles to find street center")
        
        best_heading = None
        best_score = -1
        
        for heading in test_angles:
            try:
                # Fetch image for this heading
                if address is None:
                    req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
                    img_path = self.sv.fetch(req)
                    img_rgb = read_rgb(img_path)
                elif lat is None or lon is None:
                    img_path = self.sv.fetch(address)
                    img_rgb = read_rgb(img_path)

                # Quick segmentation (just need the mask)
                out = self.segmenter.segment(img_rgb)
                sidewalk_mask = out[0]
                
                # Handle ensemble outputs
                if isinstance(sidewalk_mask, Iterable) and not isinstance(sidewalk_mask, np.ndarray):
                    sidewalk_mask = logical_fuse(list(sidewalk_mask), method=self.fuse_method or "or")
                
                # Check if we have two curbs (street center indicator)
                has_two = _has_two_curbs(sidewalk_mask, min_gap_px=50)
                
                # Score this heading
                if has_two:
                    # Additional scoring: prefer more balanced sidewalk coverage
                    cols = np.where(sidewalk_mask)[1]
                    if cols.size > 0:
                        left_coverage = np.sum(cols < sidewalk_mask.shape[1] // 2)
                        right_coverage = np.sum(cols >= sidewalk_mask.shape[1] // 2)
                        balance = min(left_coverage, right_coverage) / max(left_coverage, right_coverage, 1)
                        score = balance
                    else:
                        score = 0.5
                    
                    print(f"Heading {heading}°: Center found (score={score:.2f})")
                    
                    if score > best_score:
                        best_score = score
                        best_heading = heading
                else:
                    print(f"Heading {heading}°: No center detected")
                    
            except Exception as e:
                print(f"Heading {heading}°: Error - {e}")
                continue
        
        if best_heading is not None:
            print(f"Best center heading: {best_heading}° (score={best_score:.2f})")
        else:
            print("No street center found in test angles")
        
        return best_heading

    def _generate_heading_ranges(
        self,
        center_heading: int,
        angle_step: int = 10,
        max_deviation: int = 40,
    ) -> tuple[list[int], list[int]]:
        """
        Generate heading ranges for left and right sides from center.
        
        Parameters
        ----------
        center_heading : int
            The heading showing street center (from find_street_center)
        angle_step : int
            Step size in degrees for generating headings
        max_deviation : int
            Maximum deviation from center in each direction
        
        Returns
        -------
        left_headings, right_headings : tuple[list[int], list[int]]
            Lists of headings for left and right sides
        """
        # Left side: subtract angles (counterclockwise)
        left_headings = [
            (center_heading - angle) % 360
            for angle in range(angle_step, max_deviation + angle_step, angle_step)
        ]
        
        # Right side: add angles (clockwise)
        right_headings = [
            (center_heading + angle) % 360
            for angle in range(angle_step, max_deviation + angle_step, angle_step)
        ]
        
        return left_headings, right_headings


    # ------------------------------------------------------------------ #
    # Core implementation (private)                                      #
    # ------------------------------------------------------------------ #
    def _analyse_path(self, img_path: Path) -> Result:
        img_rgb = read_rgb(img_path)

        #initial_time = time.time()
        # -------- Mask Segmentation -------- #
        obstacles = []
        out = self.segmenter.segment(img_rgb)
        if len(out) == 3:
            sidewalk_mask, seg_map, seg_info = out # for either detectron2 or oneformer
        elif len(out) == 4:
            sidewalk_mask, seg_map, seg_info, obstacles = out # for deeplab

        # Some back-ends (ensemble) may return a tuple of masks
        if isinstance(sidewalk_mask, Iterable) and not isinstance(sidewalk_mask, np.ndarray):
            sidewalk_mask = logical_fuse(list(sidewalk_mask), method=self.fuse_method or "or")

        #print(f"Segmentation took {time.time() - initial_time:.4f} seconds")

        # -------- Mask Refinement -------- #
        refined_mask, (edge_top, edge_bot) = refine_sidewalk_mask(sidewalk_mask)
        #print(f"Mask refinement took {time.time() - initial_time:.4f} seconds")

        # -------- Obstacle Extraction -------- #
        if not obstacles:
            obstacles = extract_obstacles(seg_map, seg_info, refined_mask)
        #print(f"Obstacle extraction took {time.time() - initial_time:.4f} seconds")

        # -------- Depth ------------------------------------------------ #
        depth_map = self.depth_est.predict(img_rgb)
        metric = getattr(self.depth_est, "is_metric", False)
        width_res = compute_width(sidewalk_mask, depth_map)

        #print(f"Width estimation took {time.time() - initial_time:.4f} seconds")

        # -------- Geometry --------------------------------------------- #
        # Optionally compute obstacle clearances (pass empty list if none)
        print([lbl for lbl, _ in obstacles])
        clearances = compute_clearances(
            sidewalk_mask,
            obstacles=obstacles,            
            top_mask=edge_top,
            bot_mask=edge_bot,
            sidewalk_width_m=width_res.width_m,
            return_candidates=False,
        )

        #print(f"Clearance estimation took {time.time() - initial_time:.4f} seconds")

        # -------- Return Result --------------------------------------- #
        self._last_rgb = img_rgb  # for debugging

        return Result(
            width=width_res,
            clearances=clearances,
            sidewalk_mask=sidewalk_mask,
            refined_mask=refined_mask,
            seg_map=seg_map,
            img_path=img_path,
            rgb_image=img_rgb
        )