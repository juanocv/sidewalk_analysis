#!/usr/bin/env python
"""
Play-ground for the new pipeline.

Examples
--------
  1 Address mode (OneFormer+CUDA):
        python -m sidewalk_ai.play "Av. Paulista 1578, São Paulo"

  2 Local image, Detectron2 on CPU, label synonyms:
        python -m sidewalk_ai.play --image generic/images/frame.jpg \
                                    --seg detectron2 --device cpu \
                                    --label sidewalk,pavement,path
"""

from flask import json
import sidewalk_ai as sw
from sidewalk_ai.cli._builder import build_segmenter
from sidewalk_ai.cli._debug_viz import write_debug_sheet
from sidewalk_ai.cli._argparse import build_parser
from sidewalk_ai.models.factory import build_depth 
import numpy as np
import time
from pathlib import Path
from sidewalk_ai.api.request import from_cli_args, run_pipeline
from sidewalk_ai.processing.accessibility import (
   compute_single_view_metrics,
   compute_multiview_metrics,
)

initial_time = time.time()
# ───────────────────────── CLI args ────────────────────────────────
args = build_parser().parse_args()
args.outdir.mkdir(exist_ok=True, parents=True)
print(f"CLI argument parsing took {time.time() - initial_time:.4f} seconds")

# ──────────────────────── Debug output ─────────────────────────────
def log(msg: str):
    if args.debug:
        print("[DBG]", msg)

# Propaga flag de debug para o restante do pipeline (ex.: geometry._swai_log)
import os
if args.debug:
    os.environ["SWAI_DEBUG"] = "1"
else:
    os.environ.pop("SWAI_DEBUG", None)

# ――― expose the fallback scale so geometry.compute_width() can read it
os.environ.setdefault("SWAI_FALLBACK_SCALE", str(args.fallback_scale))
if args.force_fallback:
    os.environ["SWAI_FORCE_FALLBACK"] = "1"

# ──────────────────────── Build pipeline ───────────────────────────
segmenter = build_segmenter(args.seg, ckpt=args.ckpt,
                            dl_model=args.deeplab_model,
                            device=args.device,
                            method=args.ensemble_method)
depth = build_depth(args.depth, variant=args.zoe_variant, device=args.device)
streetview = sw.StreetViewClient()
pipe       = sw.SidewalkPipeline(segmenter=segmenter,
                                 depth=depth,
                                 args=args,
                                 streetview=streetview)
print(f"Pipeline building took {time.time() - initial_time:.4f} seconds")

# ── run ────────────────────────────────────────────────────────────
if args.image:
    from sidewalk_ai.core.pipeline import SidewalkPipeline
    res = SidewalkPipeline._analyse_path(pipe, args.image.resolve())
elif args.lat is not None and args.lon is not None:
    lat, lon = args.lat, args.lon
    # analyse_coords returns (left_estimates, right_estimates)
    # Use the shared helper to apply CLI flags consistently
    cfg = from_cli_args(args)
    res = run_pipeline(pipe, cfg)
else:
    if not args.address:
        raise ValueError("address or --image required")
    cfg = from_cli_args(args)
    res = run_pipeline(pipe, cfg)

# If the shared helper returned a rich dict for multi-view runs, extract
# the original 'results' (tuple or list) so the rest of the CLI (printing
# helpers and debug sheet writer) works unchanged. Keep the metadata in
# `multi_view_meta` for later use if needed.
multi_view_meta = None
if isinstance(res, dict) and 'results' in res:
    multi_view_meta = res
    res = res['results']

# If we have multi-view metadata, print a short summary and optionally
# write any obstacle images produced by the helper into args.outdir when
# --debug is set.
if multi_view_meta is not None:
    meta = multi_view_meta.get('metadata')
    if meta:
        lm = meta.get('left_median')
        rm = meta.get('right_median')
        counts = meta.get('n_headings', {})
        print("\nMULTI-VIEW SUMMARY:")
        if lm:
            print(f"  LEFT  median width = {lm[0]:.2f} ± {lm[1]:.2f} m  (headings={counts.get('left',0)})")
        else:
            print(f"  LEFT  no median (headings={counts.get('left',0)})")
        if rm:
            print(f"  RIGHT median width = {rm[0]:.2f} ± {rm[1]:.2f} m  (headings={counts.get('right',0)})")
        else:
            print(f"  RIGHT no median (headings={counts.get('right',0)})")

print(f"Pipeline run took {time.time() - initial_time:.4f} seconds")

'''
    if args.debug:
        imgs = multi_view_meta.get('obstacle_images', []) or []
        perh = multi_view_meta.get('per_heading', []) or []
        for i, b64 in enumerate(imgs):
            try:
                data = base64.b64decode(b64)
                # try to get a friendly name from per_heading when available
                label = None
                if i < len(perh):
                    p = perh[i]
                    label = f"{p.get('side')}_{p.get('index')}"
                fname = args.outdir / (f"obstacle_{label}.png" if label else f"obstacle_{i}.png")
                fname.write_bytes(data)
                print(f"WROTE DEBUG {fname}")
            except Exception as e:
                print(f"Failed to write obstacle image #{i}: {e}")
'''
# ──────────────────────────── Print results ─────────────────────────
# `analyse_coords` and `analyse_address` may return tuples of lists
# (left_estimates, right_estimates). Normalize to a single printable
# result for the CLI: prefer the median of the left side if available,
# otherwise the right side, otherwise the single `res` object.
def _print_result(obj):
    if obj is None:
        print("No result")
        return
    # obj may be a tuple (left_list, right_list)
    if isinstance(obj, tuple) and len(obj) == 2:
        left, right = obj
        # prefer left median
        chosen = None
        if left:
            chosen = left[len(left) // 2]
        elif right:
            chosen = right[len(right) // 2]
        else:
            print("No estimates returned for either side")
            return
    else:
        chosen = obj

    # Print width and clearances
    print(f"WIDTH  {chosen.width.width_m:.2f} ± {chosen.width.margin_m:.2f} m")
    for c in chosen.clearances:
        val = c.obs_width if c.obs_width is not None else float("nan")
        L = getattr(c, "L_m", None)
        R = getattr(c, "R_m", None)
        if L is not None and R is not None:
            print(f"CLEAR  {c.label:<8} {val:.2f} m  L={L:.2f}  R={R:.2f}")
        else:
            print(f"CLEAR  {c.label:<8} {val:.2f} m")

    # ── Accessibility (single-view) ───────────────────────────────────
    try:
        thr = float(getattr(args, "min_clear", 1.20))
        # o mesmo fator usado no accessibility.py (padrão 0.50)
        mid_ratio = float(os.getenv("SWAI_RANK_MID_RATIO", "0.50"))
        mid_thr = mid_ratio * thr
        acc = compute_single_view_metrics(chosen.clearances, min_clear_required_m=thr)
        g = acc.global_stats
        med = g.free_total_m.get('median', float('nan'))
        print(f"  Obstacles={g.total_obstacles} | "
              f"Median corridor={med:.2f} m | "
              f"Rank={g.rating} (II≥{mid_thr:.2f} m, III≥{thr:.2f} m)")
        # opcional: ainda pode mostrar a fração ≥ threshold, mas sem sugerir que afeta o ranking
        # print(f"  Share of corridors ≥{thr:.2f} m = {g.meets_ratio:.0%}")
        if acc.per_type:
            print("  Per-type corridor medians (m):")
            for t, m in acc.per_type.items():
                lm = m.free_left_m.get('median', float('nan'))
                rm = m.free_right_m.get('median', float('nan'))
                #tm = m.free_total_m.get('median', float('nan'))
                print(f"    - {t}: left={lm:.2f}  right={rm:.2f}")
        # optional JSON
        if getattr(args, "metrics_json", None):
            out = {"min_clear_required_m": acc.min_clear_required_m,
                   "global": g.__dict__,
                   "per_type": {k: v.__dict__ for k, v in acc.per_type.items()}}
            Path(args.metrics_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[WARN] failed to compute accessibility metrics for single-view: {e}")

def _print_tuple_results(obj):
    """Print median for each side and all clearances per heading.

    obj is (left_list, right_list)
    """
    left, right = obj
    #lef_med = _median_of_estimates(left)
    #rig_med = _median_of_estimates(right)

    # Estimated sidewalk width range (multi-view)
    def _width_range(estimates):
        vals = [
            getattr(e.width, "width_m", None)
            for e in estimates
            if getattr(e, "width", None) is not None
            and getattr(e.width, "width_m", None) is not None
            and np.isfinite(e.width.width_m)
            and e.width.width_m > 0
        ]
        if not vals:
            return None
        x = np.asarray(vals, dtype=float)
        if x.size >= 4:
            lo = float(np.percentile(x, 10))
            hi = float(np.percentile(x, 90))
        else:
            lo = float(np.min(x))
            hi = float(np.max(x))
        return lo, hi

    left_range = _width_range(left)
    right_range = _width_range(right)
    all_range = _width_range(list(left) + list(right))
    print("\nESTIMATED SIDEWALK WIDTH RANGE (multi-view):")
    if left_range:
        print(f"  LEFT  ≈ {left_range[0]:.2f} to {left_range[1]:.2f} m")
    if right_range:
        print(f"  RIGHT ≈ {right_range[0]:.2f} to {right_range[1]:.2f} m")
    if all_range:
        print(f"  ALL   ≈ {all_range[0]:.2f} to {all_range[1]:.2f} m")

    # Print per-heading clearances and optionally write debug sheets
    def _print_and_debug_list(lst, side_name):
        # Print per-heading widths but aggregate clearances across headings
        from collections import defaultdict

        agg = defaultdict(list)  # label -> list of obs_widths
        for i, res in enumerate(lst):
            heading_deg = getattr(res, "heading", None)
            heading_str = f" ({heading_deg}°)" if heading_deg is not None else ""
            print(f"\n{side_name} heading #{i}{heading_str}  WIDTH {res.width.width_m:.2f} ± {res.width.margin_m:.2f} m")
            for c in res.clearances:
                agg[c.label].append(c.obs_width)
                val = c.obs_width 
                L = getattr(c, 'L_m', None)
                R = getattr(c, 'R_m', None)
                if val is None:
                    val = float("nan")
                if L is not None and R is not None:
                    print(f"CLEAR  {c.label:<8} {val:.2f} m  L={L:.2f}  R={R:.2f}")
                else:
                    print(f"CLEAR  {c.label:<8} {val:.2f} m")
            if args.debug:
                # Ensure args.image exists for write_debug_sheet (it uses args.image.name)
                old_image = getattr(args, 'image', None)
                try:
                    img_path = getattr(res, 'img_path', None)
                    if img_path is None:
                        # create a synthetic Path so write_debug_sheet can build a filename
                        img_path = Path(f"{side_name}_{i}.png")
                    args.image = img_path
                    write_debug_sheet(res, pipe, args, segmenter)
                except Exception as e:
                    print(f"Failed to write debug sheet for {side_name}#{i}: {e}")
                finally:
                    # restore original args.image
                    args.image = old_image

        # Print aggregated clearances summary for this side
        if agg:
            print(f"\n{side_name} AGGREGATED CLEARANCES:")
            for label, vals in agg.items():
                vals_num = [v for v in vals if v is not None and np.isfinite(v)]
                mean_v = float(np.mean(vals_num)) if vals_num else float("nan")
                med_v = float(np.median(vals_num)) if vals_num else float("nan")
                cnt = len(vals)
                print(f"  {label:<12} count={cnt:2d}  mean={mean_v:.2f} m  median={med_v:.2f} m")
        else:
            print(f"\n{side_name} AGGREGATED CLEARANCES: none")

    _print_and_debug_list(left, "LEFT")
    _print_and_debug_list(right, "RIGHT")


   # ── Accessibility (multi-view) ────────────────────────────────────
    try:
       thr = float(getattr(args, "min_clear", 1.20))
       import os
       mid_ratio = float(os.getenv("SWAI_RANK_MID_RATIO", "0.50"))
       mid_thr = mid_ratio * thr

       acc = compute_multiview_metrics(left, right, min_clear_required_m=thr)
       print(f"\n[ACCESSIBILITY] threshold={thr:.2f} m")
       for side in ("LEFT", "RIGHT", "ALL"):
           g = acc[side].global_stats
           avg = g.avg_obstacles_per_view_rounded or g.avg_obstacles_per_view or g.total_obstacles
           med = g.free_total_m.get('median', float('nan'))
           print(f"  {side:<5} → Obstacles≈{avg} (avg/view) | "
                 f"Median corridor={med:.2f} m | "
                 f"Rank={g.rating} (II≥{mid_thr:.2f} m, III≥{thr:.2f} m)")
           # opcional:
           # print(f"           Share of corridors ≥{thr:.2f} m = {g.meets_ratio:.0%}")
       # optional JSON
       if getattr(args, "metrics_json", None):
           def _acc_to_dict(a):
               return {
                   "min_clear_required_m": a.min_clear_required_m,
                   "global": a.global_stats.__dict__,
                   "per_type": {k: v.__dict__ for k, v in a.per_type.items()},
               }
           out = {
               "LEFT":  _acc_to_dict(acc["LEFT"]),
               "RIGHT": _acc_to_dict(acc["RIGHT"]),
               "ALL":   _acc_to_dict(acc["ALL"]),
           }
           # Bloco extra: resumo multi-view (largura + nº de headings)
           mv_block = {}
           # metadata retornado por run_pipeline (se existir)
           meta = multi_view_meta.get("metadata") if multi_view_meta is not None else None
           # n_headings: {"left": N, "right": M}
           if isinstance(meta, dict) and "n_headings" in meta:
               mv_block["n_headings"] = meta["n_headings"]
           # Faixas de largura: preferir o metadata, cair para o cálculo local
           def _range_to_dict(r):
               return {"min_m": float(r[0]), "max_m": float(r[1])}
           lw = meta.get("left_width_range") if isinstance(meta, dict) else None
           rw = meta.get("right_width_range") if isinstance(meta, dict) else None
           aw = meta.get("all_width_range") if isinstance(meta, dict) else None
           # fallback para as faixas calculadas acima na função
           if lw is None:
               lw = left_range
           if rw is None:
               rw = right_range
           if aw is None:
               aw = all_range
           width_range = {}
           if lw:
               width_range["LEFT"] = _range_to_dict(lw)
           if rw:
               width_range["RIGHT"] = _range_to_dict(rw)
           if aw:
               width_range["ALL"] = _range_to_dict(aw)
           if width_range:
               mv_block["width_range_m"] = width_range
           # Só adiciona o bloco se tiver algo
           if mv_block:
               out["multi_view"] = mv_block
           Path(args.metrics_json).write_text(
               json.dumps(out, ensure_ascii=False, indent=2)
           )
    except Exception as e:
       print(f"[WARN] failed to compute accessibility metrics for multi-view: {e}")

# Choose printing method based on result type
if isinstance(res, tuple) and len(res) == 2:
    _print_tuple_results(res)
else:
    _print_result(res)
print(f"Result printing took {time.time() - initial_time:.4f} seconds")
    
# ───────────────────────── Debug sheet ────────────────────────────
# Always write a debug sheet when --debug is set. For file-based runs the
# existing code used args.image; for single-view coordinate runs we may
# not have args.image, so prefer res.img_path/rgb_image and temporarily
# set args.image so the debug writer can build a sensible filename.
if args.debug:
    # ensure pipeline has an RGB image available
    if getattr(res, 'rgb_image', None) is not None:
        pipe._last_rgb = res.rgb_image
    else:
        # fallback to args.image when present
        if getattr(args, 'image', None):
            try:
                pipe._last_rgb = sw.io.image_io.read_rgb(args.image)
            except Exception:
                pipe._last_rgb = getattr(pipe, '_last_rgb', None)

    # Temporarily ensure args.image exists for filename/header construction
    old_image = getattr(args, 'image', None)
    try:
        if getattr(args, 'image', None) is None:
            if getattr(res, 'img_path', None) is not None:
                args.image = res.img_path
            else:
                args.image = Path("singleview.png")
        write_debug_sheet(res, pipe, args, segmenter)
    except Exception as e:
        print(f"Failed to write debug sheet: {e}")
    finally:
        args.image = old_image
print(f"Debug sheet writing took {time.time() - initial_time:.4f} seconds")
