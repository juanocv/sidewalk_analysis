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

import sidewalk_ai as sw
from sidewalk_ai.cli._builder import build_segmenter
from sidewalk_ai.cli._debug_viz import write_debug_sheet
from sidewalk_ai.io.image_io import read_rgb
from sidewalk_ai.cli._argparse import build_parser
from sidewalk_ai.models.factory import build_depth 


# ───────────────────────── CLI args ────────────────────────────────
args = build_parser().parse_args()
args.outdir.mkdir(exist_ok=True, parents=True)

# ──────────────────────── Debug output ─────────────────────────────
def log(msg: str):
    if args.debug:
        print("[DBG]", msg)

# ――― expose the fallback scale so geometry.compute_width() can read it
import os
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
                                 streetview=streetview)

# ── run ────────────────────────────────────────────────────────────
if args.image:
    from sidewalk_ai.core.pipeline import SidewalkPipeline
    res = SidewalkPipeline._analyse_path(pipe, args.image.resolve())
else:
    if not args.address:
        raise ValueError("address or --image required")
    res = pipe.analyse_address(args.address)

# ──────────────────────────── Print results ─────────────────────────
print(f"WIDTH  {res.width.width_m:.2f} ± {res.width.margin_m:.2f} m")
for c in res.clearances:
    print(f"CLEAR  {c.label:<8} {c.obs_width:.2f} m  L={c.L_m:.2f}  R={c.R_m:.2f}")
    
# ───────────────────────── Debug sheet ────────────────────────────
if args.debug and args.image:
    pipe._last_rgb = sw.io.image_io.read_rgb(args.image)  # store once
    write_debug_sheet(res, pipe, args, segmenter)