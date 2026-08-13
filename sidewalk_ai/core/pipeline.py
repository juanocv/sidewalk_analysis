# sidewalk_ai/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

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
    to_metric_depth,
)
from sidewalk_ai.processing.fusion import logical_fuse
from sidewalk_ai.models.base import Segmenter
from sidewalk_ai.log import debug_event, get_logger

logger = get_logger(__name__)

_LOGO_BAR_PX = 20  # faixa inferior com a barra/logo do Google
try:
    from sidewalk_ai.io.image_io import get_google_bar_height_px as _swai_get_bar_height

    _LOGO_BAR_PX = int(_swai_get_bar_height())
except Exception:
    # fallback silencioso para manter compatibilidade se o helper nao existir
    pass


WIDTH_PARAMS = {
    "band_mode": "adaptive",
    "adaptive_pct": (0.60, 0.95),  # ↓ faixa mais estreita (mais perto do observador)
    "du_range_px": (20, 220),  # ↓ corta near-perpendicular extremo
    "parallax_range": (0.05, 0.45),  # ↓ evita “paralaxe exuberante” instável
    "min_valid_rows": 7,  # ↑ mediana mais robusta
    "divergence_pct": 0.25,  # ↓ troca p/ geom mais cedo quando divergir
    "use_data_driven_margin": True,
    "bottom_ignore_px": _LOGO_BAR_PX,  # ignora a faixa com a logo
    # debug output
    # Relative to the working directory, matching the CLI's --outdir default.
    # An absolute path built from __file__ would write inside site-packages.
    "debug_dir": Path("debug_out"),
    "debug_prefix": "frame_",
}


# --------------------------------------------------------------------------- #
# 0)  Public result dataclass                                                 #
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class DepthScale:
    """
    How to turn a non-metric depth map into metres.

    Only consulted for back-ends whose ``is_metric`` is False (MiDaS and
    friends); ZoeDepth already reports metres and is passed through untouched.

    fallback_scale
        Constant metres-per-unit used when the ground-plane fit finds too
        little support. ``None`` means "no fallback": the depth path is then
        dropped for that frame rather than reported in an arbitrary unit.
    force_fallback
        Skip the ground-plane fit entirely and always use *fallback_scale*.
    """

    fallback_scale: float | None = None
    force_fallback: bool = False


@dataclass(slots=True, frozen=True)
class Result:
    """What the caller gets back from `SidewalkPipeline`."""

    width: WidthResult
    clearances: Sequence[ClearanceResult]
    sidewalk_mask: np.ndarray  # H×W  uint8  (0/1) – raw segmenter output
    # Mask actually used downstream (obstacle extraction). Equal to
    # ``sidewalk_mask`` when the pipeline was built with ``refine=False``.
    refined_mask: np.ndarray | None = None  # H×W  uint8  (0/1)
    seg_map: np.ndarray | None = None  # panoptic id map (optional)
    seg_info: list | None = None  # list of SegmentInfo tuples (id,name)
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
    depth_scale
        Default :class:`DepthScale` for non-metric depth back-ends.  Every
        ``analyse_*`` method accepts a per-call override, so a server can serve
        different settings per request without touching shared state.
    """

    def __init__(
        self,
        *,
        segmenter: Segmenter,
        depth: Any,
        streetview: StreetViewClient,
        refine: bool = True,
        args=None,
        fuse_method: str | None = None,
        initial_time: float | None = None,
        depth_scale: DepthScale | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.depth_est = depth
        self.sv = streetview
        self.args = args
        self.refine = refine
        self.fuse_method = fuse_method
        self.initial_time = initial_time
        self.depth_scale = depth_scale or DepthScale()

    # ------------------------------------------------------------------ #
    # Depth helper                                                       #
    # ------------------------------------------------------------------ #
    def _predict_depth_without_logo(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Executa o estimador de profundidade ignorando a faixa inferior
        com a logo do Google, mas devolvendo um mapa H×W alinhado com
        a imagem original.

        A segmentação continua enxergando a imagem inteira (incluindo
        a barra), mas o modelo de profundidade nunca recebe esses pixels.
        """
        H, W = img_rgb.shape[:2]
        bar = _LOGO_BAR_PX

        if bar > 0 and H > bar:
            core_img = img_rgb[:-bar, :, :]
        else:
            core_img = img_rgb

        depth_core = self.depth_est.predict(core_img)
        depth_core = np.asarray(depth_core, dtype=np.float32)

        # MiDaS/Zoe já devolvem mesma resolução da entrada, mas mantemos
        # uma checagem defensiva.
        if depth_core.shape[:2] != core_img.shape[:2]:
            raise RuntimeError(
                f"Depth estimator returned shape {depth_core.shape} for input {core_img.shape}"
            )

        # Sem recorte? Basta devolver o resultado direto.
        if core_img is img_rgb or bar <= 0 or H <= bar:
            return depth_core

        # Com recorte: remonta um mapa HxW, deixando a faixa da logo como NaN.
        depth_full = np.full((H, W), np.nan, dtype=np.float32)
        h_core, w_core = depth_core.shape[:2]
        depth_full[:h_core, :w_core] = depth_core
        return depth_full

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
        initial_time: float | None = None,
        depth_scale: DepthScale | None = None,
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
        img_path = self.sv.fetch(req)
        return self._analyse_path(
            img_path,
            pitch=pitch,
            fov=fov,
            heading=int(heading),
            initial_time=initial_time or time.time(),
            depth_scale=depth_scale,
        )

    def analyse_image(
        self,
        img_path: Path | str,
        *,
        pitch: int = 0,
        fov: int = 90,
        heading: int | None = None,
        depth_scale: DepthScale | None = None,
    ) -> Result:
        """
        Analyse an image that is already on disk, skipping Street View entirely.

        Use this for local files and for frames a caller fetched itself; *pitch*
        and *fov* must describe how the image was captured, since the width
        geometry depends on the horizon they imply.
        """
        return self._analyse_path(
            Path(img_path),
            pitch=pitch,
            fov=fov,
            heading=heading,
            depth_scale=depth_scale,
        )

    def _analyse_heading_series(
        self,
        *,
        lat: float,
        lon: float,
        headings: Sequence[int],
        side_label: str,
        pitch: int,
        fov: int,
        depth_scale: DepthScale | None = None,
    ) -> list[Result]:
        """Analyse a list of headings for one sidewalk side."""
        estimates: list[Result] = []
        label = side_label.upper()

        for heading in headings:
            logger.info("Analysing %s heading %s°", label, heading)
            req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
            img_path = self.sv.fetch(req)
            try:
                estimates.append(
                    self._analyse_path(
                        img_path,
                        pitch=pitch,
                        fov=fov,
                        heading=heading,
                        initial_time=time.time(),
                        depth_scale=depth_scale,
                    )
                )
            except RefinementError as exc:
                logger.warning("Skipping heading %s (%s): %s", heading, side_label.lower(), exc)

        return estimates

    def analyse_address_multiview(
        self,
        address: str,
        *,
        pitch: int = 0,
        fov: int = 90,
        max_per_side: int = 4,  # limite prático p/ tempo de execução
        depth_scale: DepthScale | None = None,
    ) -> tuple[list["Result"], list["Result"]]:
        """
        Multi-view por endereço: amostra ângulos à esquerda/direita do centro.
        Usa 0° como fallback se o centro não for encontrado.
        """
        lat, lon = self.sv.geocode(address)
        center_heading = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
        if center_heading is None:
            center_heading = 0  # fallback robusto

        left_headings, right_headings = self._generate_heading_ranges(
            center_heading, max_per_side=max_per_side
        )

        left_estimates = self._analyse_heading_series(
            lat=lat,
            lon=lon,
            headings=left_headings,
            side_label="left",
            pitch=pitch,
            fov=fov,
            depth_scale=depth_scale,
        )
        right_estimates = self._analyse_heading_series(
            lat=lat,
            lon=lon,
            headings=right_headings,
            side_label="right",
            pitch=pitch,
            fov=fov,
            depth_scale=depth_scale,
        )

        return left_estimates, right_estimates

    def analyse_coords(
        self,
        lat: float,
        lon: float,
        heading: int | None = None,
        pitch: int = 0,
        fov: int = 90,
        *,
        multi_view: bool = False,
        max_per_side: int = 4,  # novo knob
        depth_scale: DepthScale | None = None,
    ) -> Result | tuple[list["Result"], list["Result"]]:
        if not multi_view:
            use_heading = heading
            if use_heading is None:
                center = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
                use_heading = center if center is not None else 0
            req = ImageRequest(lat, lon, heading=int(use_heading), pitch=pitch, fov=fov)
            img_path = self.sv.fetch(req)
            return self._analyse_path(
                img_path,
                pitch=pitch,
                fov=fov,
                heading=int(use_heading),
                initial_time=time.time(),
                depth_scale=depth_scale,
            )

        center_heading = self._find_street_center(lat=lat, lon=lon, pitch=pitch, fov=fov)
        if center_heading is None:
            center_heading = heading if heading is not None else 0  # fallback

        left_headings, right_headings = self._generate_heading_ranges(
            center_heading, max_per_side=max_per_side
        )

        left_estimates = self._analyse_heading_series(
            lat=lat,
            lon=lon,
            headings=left_headings,
            side_label="left",
            pitch=pitch,
            fov=fov,
            depth_scale=depth_scale,
        )
        right_estimates = self._analyse_heading_series(
            lat=lat,
            lon=lon,
            headings=right_headings,
            side_label="right",
            pitch=pitch,
            fov=fov,
            depth_scale=depth_scale,
        )

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

        logger.info("Testing %s angles to find street center", len(test_angles))

        best_heading = None
        best_score = -1.0

        for heading in test_angles:
            try:
                req = ImageRequest(lat, lon, heading=heading, pitch=pitch, fov=fov)
                # t0 = time.time()
                img_path = self.sv.fetch(req)
                img_rgb = read_rgb(img_path)

                out = self.segmenter.segment(img_rgb)
                sidewalk_mask = out[0]
                if isinstance(sidewalk_mask, Iterable) and not isinstance(
                    sidewalk_mask, np.ndarray
                ):
                    sidewalk_mask = logical_fuse(
                        list(sidewalk_mask), method=self.fuse_method or "or"
                    )

                sidewalk_mask = sidewalk_mask.astype(bool)

                # Pré-processamento leve: fechar grandes oclusões (carros, arbustos)
                # usando o mesmo tipo de bridge-fill da etapa de refinamento.
                """
                try:
                    bridged = bridge_fill_between_edges(
                        sidewalk_mask.astype(np.uint8),
                        smooth_kernel=5,
                        min_valid_cols=5,
                        clamp_to=sidewalk_mask.shape[0] - 30,
                    ).astype(bool)
                    mask_for_center = bridged
                except Exception as bf_exc:
                    print(f"Heading {heading}°: bridge-fill failed ({bf_exc}), using raw mask")
                    mask_for_center = sidewalk_mask
                """

                has_two = _has_two_curbs(sidewalk_mask, min_gap_px=50)
                if has_two:
                    h, w = sidewalk_mask.shape
                    cols = np.where(sidewalk_mask)[1]
                    if cols.size > 0:
                        # 1) equilíbrio esquerda/direita em área de calçada
                        left_cov = np.sum(cols < w // 2)
                        right_cov = np.sum(cols >= w // 2)
                        balance = min(left_cov, right_cov) / max(left_cov, right_cov, 1)

                        # 2) continuidade vertical em ambas as calçadas
                        y0 = int(0.30 * h)
                        y1 = int(0.90 * h)
                        if y1 <= y0:
                            y0, y1 = 0, h
                        band_h = max(1, y1 - y0)
                        left_rows = np.any(sidewalk_mask[y0:y1, : w // 2], axis=1)
                        right_rows = np.any(sidewalk_mask[y0:y1, w // 2 :], axis=1)
                        frac_left_rows = float(left_rows.sum()) / band_h
                        frac_right_rows = float(right_rows.sum()) / band_h
                        row_score = min(frac_left_rows, frac_right_rows)

                        # score final combina equilíbrio e continuidade
                        score = balance * row_score
                    else:
                        score = 0.0

                    logger.info("Heading %s°: center found (score=%.2f)", heading, score)
                    if score > best_score:
                        best_score = score
                        best_heading = heading
                else:
                    logger.info("Heading %s°: no center detected", heading)
            except Exception as e:
                logger.warning("Heading %s°: failed while finding center: %s", heading, e)
                continue

        if best_heading is not None:
            logger.info("Best center heading: %s° (score=%.2f)", best_heading, best_score)
        else:
            logger.info("No street center found in test angles")

        logger.info("Finding street center took %.4f seconds", time.time() - initial_time)
        return best_heading

    def _generate_heading_ranges(
        self,
        center_heading: int,
        initial_offset: int = 70,
        angle_step: int = 5,
        max_deviation: int = 120,
        max_per_side: int | None = None,
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

        # Gera offsets começando mais perto de `max_deviation` (perpendicular)
        # e voltando em passos de `angle_step` até `initial_offset`. Assim,
        # quando limitamos via `max_per_side`, priorizamos sempre os ângulos
        # mais próximos de 90° (perpendicular e quase-perpendicular).
        def seq_offsets(sign: int) -> list[int]:
            """Produce offsets starting at `initial_offset` and increasing by
            `angle_step` up to `max_deviation`.

            This yields offsets in the order: initial_offset,
            initial_offset+angle_step, initial_offset+2*angle_step, ...
            which matches the desired capture sequence (closest to
            perpendicular first is controlled by initial_offset).
            """
            offsets = []
            # defensive: ensure sensible parameters
            if angle_step <= 0 or max_deviation < initial_offset:
                return offsets
            angle = initial_offset
            while angle <= max_deviation:
                offsets.append(sign * angle)
                if max_per_side is not None and len(offsets) >= max_per_side:
                    break
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
        initial_time: float | None = None,
        depth_scale: DepthScale | None = None,
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
            sidewalk_mask, seg_map, seg_info = out  # detectron2, oneformer
        elif len(out) == 4:
            sidewalk_mask, seg_map, seg_info, obstacles = out  # deeplab, ensemble
        else:
            # Falling through used to leave every name below unbound, so the
            # real failure surfaced 20 lines later as a confusing NameError.
            raise TypeError(
                f"{type(self.segmenter).__name__}.segment() returned {len(out)} values; "
                "expected (mask, seg_map, seg_info) or (mask, seg_map, seg_info, obstacles)"
            )

        # Some back-ends (ensemble) may return a tuple of masks
        if isinstance(sidewalk_mask, Iterable) and not isinstance(sidewalk_mask, np.ndarray):
            sidewalk_mask = logical_fuse(list(sidewalk_mask), method=self.fuse_method or "or")

        logger.info("Segmentation took %.4f seconds", time.time() - initial_time)

        # -------- Mask Refinement -------- #
        # The fitted curb lines are no longer consumed here: clearances now read
        # the sidewalk span directly from the mask (see compute_clearances).
        if self.refine:
            refined_mask, _curb_lines = refine_sidewalk_mask(sidewalk_mask)
            logger.info("Mask refinement took %.4f seconds", time.time() - initial_time)
        else:
            # refine=False keeps the raw segmenter mask downstream. Note this
            # also disables the RefinementError path that lets multi-view skip
            # unusable headings.
            refined_mask = np.asarray(sidewalk_mask).astype(np.uint8)
            logger.debug("Mask refinement disabled (refine=False)")

        # -------- Obstacle Extraction (base-only) -------- #
        # Sempre derive obstáculos pela BASE (contato com a calçada) a partir
        # do mapa panóptico – robusto contra copas coladas:
        if seg_map is not None and seg_info is not None:
            obstacles = extract_obstacles(seg_map, seg_info, refined_mask)
        # caso extremo: sem panoptic disponível, mantém os do segmenter
        logger.info("Obstacle extraction took %.4f seconds", time.time() - initial_time)

        # -------- Depth ------------------------------------------------ #
        depth_map = self._predict_depth_without_logo(img_rgb)

        metric = getattr(self.depth_est, "is_metric", False)
        if not metric:
            # Relative back-ends are only defined up to a factor; compute_width
            # reads the map as metres, so recover that factor first.
            scale_cfg = depth_scale or self.depth_scale
            depth_map, alpha, alpha_source = to_metric_depth(
                depth_map,
                sidewalk_mask,
                fov_deg=fov,
                fallback_scale=scale_cfg.fallback_scale,
                force_fallback=scale_cfg.force_fallback,
            )
            if alpha is None:
                logger.warning(
                    "No metric scale for a non-metric depth back-end; "
                    "dropping the depth path for this frame "
                    "(set --fallback-scale to keep it)"
                )
                depth_map = None
            else:
                logger.debug("Metric scale alpha=%.5f (source=%s)", alpha, alpha_source)

        m_cov = float(sidewalk_mask.mean())
        if depth_map is not None and np.isfinite(depth_map).any():
            d_min = float(np.nanmin(depth_map))
            d_med = float(np.nanmedian(depth_map))
            d_max = float(np.nanmax(depth_map))
        else:
            d_min = d_med = d_max = float("nan")

        debug_event(
            logger,
            "depth",
            {
                "img": str(img_path),
                "mask_coverage": m_cov,
                "depth_min": d_min,
                "depth_med": d_med,
                "depth_max": d_max,
                "is_metric": metric,
            },
        )

        # -------- Width ------------------------------------------------ #
        params = dict(WIDTH_PARAMS)
        debug_flag = bool(getattr(self.args, "debug", False))
        if debug_flag:
            params["debug_dir"] = getattr(self.args, "outdir", params.get("debug_dir"))
            params["debug_prefix"] = params.get("debug_prefix", "frame_")
        else:
            params.pop("debug_dir", None)
            params.pop("debug_prefix", None)

        width_res = compute_width(
            sidewalk_mask,
            depth_map,
            pitch_deg=pitch,
            fov_deg=fov,
            debug=debug_flag,
            **params,
        )

        # Aviso quando não há suporte suficiente de pixels para medir largura.
        if width_res.width_m <= 0.0 and m_cov < 0.05:
            logger.warning(
                "No sidewalk support for width estimation",
                extra={
                    "event": "no_sidewalk_support",
                    "payload": {
                        "mask_coverage": m_cov,
                        "width_n_pixels": int(width_res.n_pixels),
                    },
                },
            )

        # print(f"Width estimation {width_res}")
        # print(f"Width estimation took {time.time() - initial_time:.4f} seconds")

        # -------- Geometry --------------------------------------------- #
        # Optionally compute obstacle clearances (pass empty list if none)
        # print([lbl for lbl, _ in obstacles])
        clearances = compute_clearances(
            sidewalk_mask,
            obstacles=obstacles,
            sidewalk_width_m=width_res.width_m,
            return_candidates=False,
        )

        logger.info("Clearance estimation took %.4f seconds", time.time() - initial_time)

        # -------- Return Result --------------------------------------- #
        # The RGB frame travels on the Result rather than on the pipeline: a
        # `self._last_rgb` here is shared mutable state, and the API serves many
        # requests from one pipeline instance. The CLI sets it explicitly when
        # the debug sheet needs a fallback.
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
