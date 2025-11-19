# sidewalk_ai/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from sidewalk_ai.processing.refinement import refine_sidewalk_mask
from sidewalk_ai.processing.refinement import RefinementError
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


WIDTH_PARAMS = {
    "band_mode": "adaptive",
    "adaptive_pct": (0.60, 0.95),   # ↓ faixa mais estreita (mais perto do observador)
    "du_range_px": (20, 220),       # ↓ corta near-perpendicular extremo
    "parallax_range": (0.05, 0.45), # ↓ evita “paralaxe exuberante” instável
    "min_valid_rows": 7,            # ↑ mediana mais robusta
    "divergence_pct": 0.25,         # ↓ troca p/ geom mais cedo quando divergir
    "use_data_driven_margin": True,
    "bottom_ignore_px": 20,         # ignora a faixa com a logo
}

# --------------------------------------------------------------------------- #
# 0)  Public result dataclass                                                 #
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class Result:
    """What the caller gets back from `SidewalkPipeline`."""
    width: WidthResult
    clearances: Sequence[ClearanceResult]
    sidewalk_mask: np.ndarray          # H×W  uint8  (0/1)
    refined_mask: np.ndarray | None = None  # H×W  uint8  (0/1)
    seg_map: np.ndarray | None = None  # panoptic id map (optional)
    seg_info: list | None = None       # list of SegmentInfo tuples (id,name)
    img_path: Path | None = None
    rgb_image: np.ndarray | None = None  # H×W×3  uint8 (RGB)
    obstacles: list[tuple[str, np.ndarray]] | None = None
    heading: int | None = None


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
        initial_time: float = None,
    ) -> None:
        self.segmenter = segmenter
        self.depth_est = depth
        self.sv = streetview
        self.refine = refine
        self.fuse_method = fuse_method
        self.initial_time = initial_time

    # ------------------------------------------------------------------ #
    # Convenience overloads                                              #
    # ------------------------------------------------------------------ #
    def analyse_address(
        self,
        address: str,
        *,
        heading: int | None = None,
        pitch: int = 0,
        fov: int = 90,
        initial_time: float = None,
    ) -> Result:
        """
        Single-view por endereço. Se heading não for dado, tenta achar o centro
        (usando pitch/fov) e usa 0° como fallback.
        """
        lat, lon = self.sv.geocode(address)
        if heading is None:
            center = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
            heading = center if center is not None else 0

        req = ImageRequest(lat, lon, heading=int(heading), pitch=pitch, fov=fov)
        t0 = time.time()
        img_path = self.sv.fetch(req)
        print(f"Image acquisition took {time.time() - t0:.4f} seconds")
        return self._analyse_path(
            img_path,
            pitch=pitch,
            fov=fov,
            heading=int(heading),
            initial_time=initial_time or time.time(),
        )

    def analyse_address_multiview(
        self,
        address: str,
        *,
        pitch: int = 0,
        fov: int = 90,
        max_per_side: int = 4,   # limite prático p/ tempo de execução
    ) -> tuple[list["Result"], list["Result"]]:
        """
        Multi-view por endereço: amostra ângulos à esquerda/direita do centro.
        Usa 0° como fallback se o centro não for encontrado.
        """
        lat, lon = self.sv.geocode(address)
        center_heading = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
        if center_heading is None:
            center_heading = 0  # fallback robusto

        left_headings, right_headings = self._generate_heading_ranges(center_heading)
        if max_per_side is not None and max_per_side > 0:
            left_headings  = left_headings[:max_per_side]
            right_headings = right_headings[:max_per_side]

        left_estimates: list[Result] = []
        right_estimates: list[Result] = []
        for h in left_headings:
            print(f"[MULTI-VIEW] analysing LEFT heading {h}°")
            req = ImageRequest(lat, lon, heading=h, pitch=pitch, fov=fov)
            t0 = time.time()
            img_path = self.sv.fetch(req)
            print(f"Image acquisition took {time.time() - t0:.4f} seconds")
            try:
                left_estimates.append(
                    self._analyse_path(
                        img_path,
                        pitch=pitch,
                        fov=fov,
                        heading=h,
                        initial_time=time.time(),
                    )
                )
            except RefinementError as e:
                print(f"Skipping heading {h} (left): {e}")

        for h in right_headings:
            print(f"[MULTI-VIEW] analysing RIGHT heading {h}°")
            req = ImageRequest(lat, lon, heading=h, pitch=pitch, fov=fov)
            t0 = time.time()
            img_path = self.sv.fetch(req)
            print(f"Image acquisition took {time.time() - t0:.4f} seconds")
            try:
                right_estimates.append(
                    self._analyse_path(
                        img_path,
                        pitch=pitch,
                        fov=fov,
                        heading=h,
                        initial_time=time.time(),
                    )
                )
            except RefinementError as e:
                print(f"Skipping heading {h} (right): {e}")

        return left_estimates, right_estimates

    def analyse_coords(
        self,
        lat: float, lon: float,
        heading: int | None = None,
        pitch:   int = 0,
        fov:     int = 90,
        *,
        multi_view: bool = False,
        max_per_side: int = 4,  # novo knob
    ) -> Result | tuple[list["Result"], list["Result"]]:
        if not multi_view:
            use_heading = heading
            if use_heading is None:
                center = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
                use_heading = center if center is not None else 0
            req = ImageRequest(lat, lon, heading=int(use_heading), pitch=pitch, fov=fov)
            t0 = time.time()
            img_path = self.sv.fetch(req)
            print(f"Image acquisition took {time.time() - t0:.4f} seconds")
            return self._analyse_path(
                img_path,
                pitch=pitch,
                fov=fov,
                heading=int(use_heading),
                initial_time=time.time(),
            )

        center_heading = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
        if center_heading is None:
            center_heading = heading if heading is not None else 0  # fallback

        left_headings, right_headings = self._generate_heading_ranges(center_heading)
        if max_per_side is not None and max_per_side > 0:
            left_headings  = left_headings[:max_per_side]
            right_headings = right_headings[:max_per_side]

        left_estimates: list[Result] = []
        right_estimates: list[Result] = []
        for h in left_headings:
            print(f"[MULTI-VIEW] analysing LEFT heading {h}°")
            req = ImageRequest(lat, lon, heading=h, pitch=pitch, fov=fov)
            img_path = self.sv.fetch(req)
            try:
                left_estimates.append(
                    self._analyse_path(
                        img_path,
                        pitch=pitch,
                        fov=fov,
                        heading=h,
                        initial_time=time.time(),
                    )
                )
            except RefinementError as e:
                print(f"Skipping heading {h} (left): {e}")

        for h in right_headings:
            print(f"[MULTI-VIEW] analysing RIGHT heading {h}°")
            req = ImageRequest(lat, lon, heading=h, pitch=pitch, fov=fov)
            img_path = self.sv.fetch(req)
            try:
                right_estimates.append(
                    self._analyse_path(
                        img_path,
                        pitch=pitch,
                        fov=fov,
                        heading=h,
                        initial_time=time.time(),
                    )
                )
            except RefinementError as e:
                print(f"Skipping heading {h} (right): {e}")

        return left_estimates, right_estimates
    
    # ------------------------------------------------------------------ #
    # Core implementation (private)                                      #
    def _find_street_center(
        self,
        lat: float,
        lon: float,
        *,
        pitch: int = 0,
        fov: int = 90,
        initial_time: float = None,
        test_angles: list[int] | None = None,
    ) -> int | None:
        """
        Busca um heading que mostre o centro da rua (duas guias visíveis).
        Retorna None se não achar em `test_angles`.
        """
        
        if test_angles is None:
            test_angles = [0, 90, 180, 270]

        if initial_time is None:
            # Prefer pipeline-wide start time when available; otherwise start now.
            initial_time = self.initial_time or time.time()

        print(f"Testing {len(test_angles)} angles to find street center")

        best_heading = None
        best_score = -1.0

        for heading in test_angles:
            try:
                req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
                t0 = time.time()
                img_path = self.sv.fetch(req)
                print(f"Image acquisition took {time.time() - t0:.4f} seconds")
                img_rgb = read_rgb(img_path)

                out = self.segmenter.segment(img_rgb)
                sidewalk_mask = out[0]
                if isinstance(sidewalk_mask, Iterable) and not isinstance(sidewalk_mask, np.ndarray):
                    sidewalk_mask = logical_fuse(list(sidewalk_mask), method=self.fuse_method or "or")

                has_two = _has_two_curbs(sidewalk_mask, min_gap_px=50)
                if has_two:
                    cols = np.where(sidewalk_mask)[1]
                    if cols.size > 0:
                        left_cov  = np.sum(cols <  sidewalk_mask.shape[1] // 2)
                        right_cov = np.sum(cols >= sidewalk_mask.shape[1] // 2)
                        balance = min(left_cov, right_cov) / max(left_cov, right_cov, 1)
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

        print(f"Finding street center took {time.time() - initial_time:.4f} seconds")
        return best_heading

    def _generate_heading_ranges(
        self,
        center_heading: int,
        initial_offset: int = 40,
        angle_step: int = 15,
        max_deviation: int = 90,
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
        # Start at an initial offset (e.g., 40°) then step by angle_step up to
        # max_deviation. This avoids sampling headings that are too near the
        # street-center view which may still show both curbs and add noise.
        def seq_offsets(sign: int) -> list[int]:
            offsets = []
            angle = initial_offset
            while angle <= max_deviation:
                offsets.append(sign * angle)
                angle += angle_step
            return offsets

        # Left side: subtract offsets (counterclockwise)
        left_headings = [((center_heading + off) % 360) for off in seq_offsets(-1)]

        # Right side: add offsets (clockwise)
        right_headings = [((center_heading + off) % 360) for off in seq_offsets(1)]
        
        return left_headings, right_headings


    # ------------------------------------------------------------------ #
    # Core implementation (private)                                      #
    # ------------------------------------------------------------------ #
    def _analyse_path(
        self,
        img_path: Path,
        *,
        pitch: int = 0,
        fov: int = 90,
        heading: int | None = None,
        initial_time: float = None,
    ) -> Result:
        if initial_time is None:
            initial_time = self.initial_time
        if initial_time is None:
            initial_time = time.time()
        img_rgb = read_rgb(img_path)

        # -------- Mask Segmentation -------- #
        obstacles = []
        out = self.segmenter.segment(img_rgb)
        if len(out) == 3:
            sidewalk_mask, seg_map, seg_info = out # for either detectron2 or oneformer
        elif len(out) == 4:
            sidewalk_mask, seg_map, seg_info, obstacles = out # for deeplab

        #print(f"Segmentation map {seg_map}")
        #print(f"Segmentation info {seg_info}")

        # Some back-ends (ensemble) may return a tuple of masks
        if isinstance(sidewalk_mask, Iterable) and not isinstance(sidewalk_mask, np.ndarray):
            sidewalk_mask = logical_fuse(list(sidewalk_mask), method=self.fuse_method or "or")

        print(f"Segmentation took {time.time() - initial_time:.4f} seconds")

        # -------- Mask Refinement -------- #
        refined_mask, (edge_top, edge_bot) = refine_sidewalk_mask(sidewalk_mask)
        print(f"Mask refinement took {time.time() - initial_time:.4f} seconds")

        # -------- Obstacle Extraction (base-only) -------- #
        # Sempre derive obstáculos pela BASE (contato com a calçada) a partir
        # do mapa panóptico – robusto contra copas coladas:
        if seg_map is not None and seg_info is not None:
            obstacles = extract_obstacles(seg_map, seg_info, refined_mask)
        # caso extremo: sem panoptic disponível, mantém os do segmenter
        print(f"Obstacle extraction took {time.time() - initial_time:.4f} seconds")

        # -------- Depth ------------------------------------------------ #
        depth_map = self.depth_est.predict(img_rgb)
        
        #metric = getattr(self.depth_est, "is_metric", False)
        #m_cov = float(sidewalk_mask.mean())
        #d_min, d_med, d_max = float(depth_map.min()), float(np.median(depth_map)), float(depth_map.max())
        
        #print("[SWAI][frame]", {"img": str(img_path), "mask_coverage": m_cov,
        #                    "depth_min": d_min, "depth_med": d_med, "depth_max": d_max,
        #                    "depth_metric": bool(metric)})

        # -------- Width ------------------------------------------------ #
        params = dict(WIDTH_PARAMS)
        #params.update(kw)  # sobrescreve com overrides de ambiente, se houver
        width_res = compute_width(sidewalk_mask, depth_map, pitch_deg=pitch, fov_deg=fov, **params)
       
        #print(f"Width estimation {width_res}")
        print(f"Width estimation took {time.time() - initial_time:.4f} seconds")

        # -------- Geometry --------------------------------------------- #
        # Optionally compute obstacle clearances (pass empty list if none)
        # print([lbl for lbl, _ in obstacles])
        clearances = compute_clearances(
            sidewalk_mask,
            obstacles=obstacles,            
            top_mask=edge_top,
            bot_mask=edge_bot,
            sidewalk_width_m=width_res.width_m,
            return_candidates=False,
        )

        print(f"Clearance estimation took {time.time() - initial_time:.4f} seconds")

        # -------- Return Result --------------------------------------- #
        self._last_rgb = img_rgb  # for debugging

        return Result(
            width=width_res,
            clearances=clearances,
            sidewalk_mask=sidewalk_mask,
            refined_mask=refined_mask,
            seg_map=seg_map,
            seg_info=seg_info,
            img_path=img_path,
            rgb_image=img_rgb,
            obstacles=obstacles,
            heading=heading,
        )
