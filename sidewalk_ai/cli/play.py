#!/usr/bin/env python
"""
Command-line front-end for the sidewalk pipeline.

Examples
--------
  1 Address mode (OneFormer + CUDA):
        sidewalk-ai "Av. Paulista 1578, Sao Paulo"

  2 Local image, Detectron2 on CPU:
        sidewalk-ai --image generic/images/frame.jpg --seg detectron2 --device cpu
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sidewalk_ai as sw
from sidewalk_ai.api.request import from_cli_args, run_pipeline
from sidewalk_ai.cli._argparse import build_parser
from sidewalk_ai.cli._builder import build_segmenter
from sidewalk_ai.core.pipeline import DepthScale
from sidewalk_ai.io.image_io import read_rgb
from sidewalk_ai.log import configure_logging, get_logger
from sidewalk_ai.models.factory import build_depth
from sidewalk_ai.processing.accessibility import (
    MID_RATIO,
    compute_multiview_metrics,
    compute_single_view_metrics,
)

logger = get_logger(__name__)


def _write_debug_sheet(*debug_args, **debug_kwargs):
    try:
        from sidewalk_ai.cli._debug_viz import write_debug_sheet
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional debug dependency"
        raise RuntimeError(
            "Debug sheet generation requires optional ML/debug dependencies. "
            'Install the ML extra with `python -m pip install -e ".[ml]"` and '
            "install backend-specific packages such as Detectron2 when using "
            "`--debug` with those visualizations. Missing module: "
            f"{missing}"
        ) from exc

    return write_debug_sheet(*debug_args, **debug_kwargs)


# --------------------------------------------------------------------------- #
# Pipeline construction and execution                                         #
# --------------------------------------------------------------------------- #
def _build(args):
    """Build the segmenter and the pipeline described by *args*."""
    segmenter = build_segmenter(
        args.seg,
        ckpt=args.ckpt,
        dl_model=args.deeplab_model,
        device=args.device,
        method=args.ensemble_method,
    )
    depth = build_depth(args.depth, variant=args.zoe_variant, device=args.device)
    pipe = sw.SidewalkPipeline(
        segmenter=segmenter,
        depth=depth,
        args=args,
        streetview=sw.StreetViewClient(),
        refine=args.refine,
        depth_scale=DepthScale(
            fallback_scale=args.fallback_scale,
            force_fallback=args.force_fallback,
        ),
    )
    return segmenter, pipe


def _run(args, pipe):
    """
    Execute the requested analysis.

    Returns ``(result, multi_view_meta)`` where *multi_view_meta* is the rich
    dict produced for multi-view runs, or ``None`` for single-view ones.
    """
    if args.image:
        # pitch/fov must be forwarded: the width geometry derives the horizon
        # from them, and letting them default silently used the wrong one.
        result = pipe.analyse_image(
            args.image.resolve(),
            pitch=args.pitch,
            fov=args.fov,
            heading=args.heading,
        )
        return result, None

    out = run_pipeline(pipe, from_cli_args(args))

    # Multi-view runs come back as a rich dict; unwrap the estimates and keep
    # the metadata for the summary and the JSON export.
    if isinstance(out, dict) and "results" in out:
        return out["results"], out
    return out, None


# --------------------------------------------------------------------------- #
# Printing                                                                    #
# --------------------------------------------------------------------------- #
def _print_multi_view_summary(meta):
    if not meta:
        return
    left_median = meta.get("left_median")
    right_median = meta.get("right_median")
    counts = meta.get("n_headings", {})

    print("\nMULTI-VIEW SUMMARY:")
    for label, median, key in (
        ("LEFT ", left_median, "left"),
        ("RIGHT", right_median, "right"),
    ):
        if median:
            print(
                f"  {label} median width = {median[0]:.2f} "
                f"± {median[1]:.2f} m  (headings={counts.get(key, 0)})"
            )
        else:
            print(f"  {label} no median (headings={counts.get(key, 0)})")


def _print_clearance(clearance):
    value = clearance.obs_width
    if value is None or not np.isfinite(value):
        value = float("nan")
    left = getattr(clearance, "L_m", None)
    right = getattr(clearance, "R_m", None)
    if left is not None and right is not None:
        print(f"CLEAR  {clearance.label:<8} {value:.2f} m  L={left:.2f}  R={right:.2f}")
    else:
        print(f"CLEAR  {clearance.label:<8} {value:.2f} m")


def _accessibility_to_dict(metrics):
    return {
        "min_clear_required_m": metrics.min_clear_required_m,
        "global": metrics.global_stats.__dict__,
        "per_type": {k: v.__dict__ for k, v in metrics.per_type.items()},
    }


def _print_result(result, args):
    """Print a single-view result (or the middle estimate of a side)."""
    if result is None:
        print("No result")
        return

    if isinstance(result, tuple) and len(result) == 2:
        left, right = result
        if left:
            chosen = left[len(left) // 2]
        elif right:
            chosen = right[len(right) // 2]
        else:
            print("No estimates returned for either side")
            return
    else:
        chosen = result

    print(f"WIDTH  {chosen.width.width_m:.2f} ± {chosen.width.margin_m:.2f} m")
    for clearance in chosen.clearances:
        _print_clearance(clearance)

    # ── Accessibility (single-view) ───────────────────────────────────
    try:
        threshold = float(getattr(args, "min_clear", 1.20))
        mid_threshold = MID_RATIO * threshold
        metrics = compute_single_view_metrics(chosen.clearances, min_clear_required_m=threshold)
        stats = metrics.global_stats
        median = stats.free_total_m.get("median", float("nan"))
        print(
            f"  Obstacles={stats.total_obstacles} | "
            f"Median corridor={median:.2f} m | "
            f"Rank={stats.rating} (II>={mid_threshold:.2f} m, III>={threshold:.2f} m)"
        )
        if metrics.per_type:
            print("  Per-type corridor medians (m):")
            for name, per_type in metrics.per_type.items():
                left_median = per_type.free_left_m.get("median", float("nan"))
                right_median = per_type.free_right_m.get("median", float("nan"))
                print(f"    - {name}: left={left_median:.2f}  right={right_median:.2f}")

        if getattr(args, "metrics_json", None):
            Path(args.metrics_json).write_text(
                json.dumps(_accessibility_to_dict(metrics), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception as exc:
        logger.warning("Failed to compute accessibility metrics for single-view: %s", exc)


def _width_range(estimates):
    """Robust [lo, hi] band of plausible widths across multi-view estimates."""
    values = [
        float(e.width.width_m)
        for e in estimates
        if getattr(e, "width", None) is not None
        and getattr(e.width, "width_m", None) is not None
        and np.isfinite(e.width.width_m)
        and e.width.width_m > 0
    ]
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    if x.size >= 4:
        return float(np.percentile(x, 10)), float(np.percentile(x, 90))
    return float(np.min(x)), float(np.max(x))


def _print_side(estimates, side_name, args, pipe, segmenter):
    """Per-heading widths and clearances for one side, plus optional debug sheets."""
    aggregated = defaultdict(list)  # label -> obstacle widths

    for index, result in enumerate(estimates):
        heading = getattr(result, "heading", None)
        heading_str = f" ({heading}°)" if heading is not None else ""
        print(
            f"\n{side_name} heading #{index}{heading_str}  "
            f"WIDTH {result.width.width_m:.2f} ± {result.width.margin_m:.2f} m"
        )
        for clearance in result.clearances:
            aggregated[clearance.label].append(clearance.obs_width)
            _print_clearance(clearance)

        if args.debug:
            # write_debug_sheet builds its filename from args.image.
            previous_image = getattr(args, "image", None)
            try:
                args.image = getattr(result, "img_path", None) or Path(f"{side_name}_{index}.png")
                _write_debug_sheet(result, pipe, args, segmenter)
            except Exception as exc:
                logger.warning("Failed to write debug sheet for %s#%s: %s", side_name, index, exc)
            finally:
                args.image = previous_image

    if not aggregated:
        print(f"\n{side_name} AGGREGATED CLEARANCES: none")
        return

    print(f"\n{side_name} AGGREGATED CLEARANCES:")
    for label, values in aggregated.items():
        finite = [v for v in values if v is not None and np.isfinite(v)]
        mean = float(np.mean(finite)) if finite else float("nan")
        median = float(np.median(finite)) if finite else float("nan")
        print(f"  {label:<12} count={len(values):2d}  mean={mean:.2f} m  median={median:.2f} m")


def _print_tuple_results(result, args, pipe, segmenter, multi_view_meta):
    """Print both sides of a multi-view run."""
    left, right = result

    ranges = {
        "LEFT": _width_range(left),
        "RIGHT": _width_range(right),
        "ALL": _width_range(list(left) + list(right)),
    }
    print("\nESTIMATED SIDEWALK WIDTH RANGE (multi-view):")
    for label in ("LEFT", "RIGHT", "ALL"):
        band = ranges[label]
        if band:
            print(f"  {label:<5} ~ {band[0]:.2f} to {band[1]:.2f} m")

    _print_side(left, "LEFT", args, pipe, segmenter)
    _print_side(right, "RIGHT", args, pipe, segmenter)

    # ── Accessibility (multi-view) ────────────────────────────────────
    try:
        threshold = float(getattr(args, "min_clear", 1.20))
        mid_threshold = MID_RATIO * threshold

        metrics = compute_multiview_metrics(left, right, min_clear_required_m=threshold)
        print(f"\n[ACCESSIBILITY] threshold={threshold:.2f} m")
        for side in ("LEFT", "RIGHT", "ALL"):
            stats = metrics[side].global_stats
            average = (
                stats.avg_obstacles_per_view_rounded
                or stats.avg_obstacles_per_view
                or stats.total_obstacles
            )
            median = stats.free_total_m.get("median", float("nan"))
            print(
                f"  {side:<5} -> Obstacles~{average} (avg/view) | "
                f"Median corridor={median:.2f} m | "
                f"Rank={stats.rating} (II>={mid_threshold:.2f} m, III>={threshold:.2f} m)"
            )

        if getattr(args, "metrics_json", None):
            _write_multiview_json(args, metrics, ranges, multi_view_meta)
    except Exception as exc:
        logger.warning("Failed to compute accessibility metrics for multi-view: %s", exc)


def _write_multiview_json(args, metrics, ranges, multi_view_meta):
    payload = {side: _accessibility_to_dict(metrics[side]) for side in ("LEFT", "RIGHT", "ALL")}

    meta = multi_view_meta.get("metadata") if multi_view_meta else None
    meta = meta if isinstance(meta, dict) else {}

    block = {}
    if "n_headings" in meta:
        block["n_headings"] = meta["n_headings"]

    # Prefer the ranges computed by the pipeline helper; fall back to the local ones.
    width_range = {}
    for side, key in (
        ("LEFT", "left_width_range"),
        ("RIGHT", "right_width_range"),
        ("ALL", "all_width_range"),
    ):
        band = meta.get(key) or ranges.get(side)
        if band:
            width_range[side] = {"min_m": float(band[0]), "max_m": float(band[1])}
    if width_range:
        block["width_range_m"] = width_range

    if block:
        payload["multi_view"] = block

    Path(args.metrics_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_final_debug_sheet(result, args, pipe, segmenter):
    """Composite debug sheet for single-view runs."""
    if getattr(result, "rgb_image", None) is not None:
        pipe._last_rgb = result.rgb_image
    elif getattr(args, "image", None):
        try:
            pipe._last_rgb = read_rgb(args.image)
        except Exception:
            pipe._last_rgb = getattr(pipe, "_last_rgb", None)

    previous_image = getattr(args, "image", None)
    try:
        if previous_image is None:
            args.image = getattr(result, "img_path", None) or Path("singleview.png")
        _write_debug_sheet(result, pipe, args, segmenter)
    except Exception as exc:
        logger.warning("Failed to write debug sheet: %s", exc)
    finally:
        args.image = previous_image


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    started = time.time()

    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.image or args.address or (args.lat is not None and args.lon is not None)):
        parser.error("provide an address, --lat/--lon, or --image")

    # A Windows console is cp1252 by default, where an unencodable character
    # raises UnicodeEncodeError mid-print. That used to abort the whole
    # accessibility block -- the NBR 9050 rating included -- leaving only a
    # logged warning. Output text is ASCII now; this keeps a future slip
    # degrading to "?" instead of losing a section.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass

    configure_logging(
        debug=args.debug,
        level=args.log_level,
        fmt=args.log_format,
        log_file=args.log_file,
    )
    args.outdir.mkdir(exist_ok=True, parents=True)
    logger.info("CLI argument parsing took %.4f seconds", time.time() - started)

    # geometry.py and refinement.py consult SWAI_DEBUG for their own artifacts.
    if args.debug:
        os.environ["SWAI_DEBUG"] = "1"
    else:
        os.environ.pop("SWAI_DEBUG", None)

    segmenter, pipe = _build(args)
    logger.info("Pipeline building took %.4f seconds", time.time() - started)

    result, multi_view_meta = _run(args, pipe)
    if multi_view_meta is not None:
        _print_multi_view_summary(multi_view_meta.get("metadata"))
    logger.info("Pipeline run took %.4f seconds", time.time() - started)

    if isinstance(result, tuple) and len(result) == 2:
        _print_tuple_results(result, args, pipe, segmenter, multi_view_meta)
    else:
        _print_result(result, args)
        if args.debug:
            _write_final_debug_sheet(result, args, pipe, segmenter)
    logger.info("Total run took %.4f seconds", time.time() - started)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
