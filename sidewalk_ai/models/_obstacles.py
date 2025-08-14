# sidewalk_ai/models/_obstacles.py
from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------- #
SIDEWALK_LABELS = ("sidewalk", "pavement")          # synonyms
IGNORE_LABELS   = {
    "road", "route", "building", "wall", "sky", "terrain",
    "floor", "ground", "ceiling", "bridge", "car", "bus",
    "truck", "train", "motorcycle", "bicycle", "person",
    "sign", "street lamp", "fence", "stairs"
}

MIN_INST_AREA_PX   = 30          # reject very tiny noise blobs
MAX_INST_FRAC      = 0.50        # reject background-size regions
MIN_OVERLAP_PX     = 10          # at least this many pixels on sidewalk
MIN_OVERLAP_RATIO  = 0.01        # ≥ 1 % of the instance must sit on sidewalk


def extract_obstacles(
    seg_map:      np.ndarray,
    seg_info:     Sequence[Tuple[int, str]],
    sidewalk_mask: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """
    Return   [(label, bool-mask), …]   for every instance that overlaps
    the sidewalk mask “enough”.  No heavy geometry tests – fast & simple.
    """
    obstacles: list[Tuple[str, np.ndarray]] = []
    H, W      = seg_map.shape
    #sidewalk_area = sidewalk_mask.sum().astype(float) + 1e-6

    #print(f"Sidewalk area: {sidewalk_area} px")

    for seg_id, raw_lbl in seg_info:
        lbl = raw_lbl.lower().strip()

        # 1) skip labels we don’t care about
        if any(lbl.startswith(s) for s in IGNORE_LABELS) or any(lbl.startswith(s) for s in SIDEWALK_LABELS):
            #print(f"Skipping {raw_lbl} ({lbl})")
            continue

        inst_mask   = (seg_map == seg_id)
        inst_area   = inst_mask.sum()

        # 2) basic sanity filters
        if (inst_area < MIN_INST_AREA_PX):
            #print(f"Skipping {raw_lbl} ({lbl}) due to area {inst_area}")
            continue

        # 3) overlap test
        overlap_px   = (inst_mask & sidewalk_mask).sum()
        overlap_r    = overlap_px / inst_area
        if overlap_px < MIN_OVERLAP_PX or overlap_r < MIN_OVERLAP_RATIO:
            #print(f"Skipping {raw_lbl} ({lbl}) due to overlap {overlap_px} px / {overlap_r:.2%}")
            continue
        if overlap_px >= MIN_OVERLAP_PX and overlap_r >= MIN_OVERLAP_RATIO:
            #print(f"Adding {raw_lbl} ({lbl}) with {overlap_px} px overlap")
            obstacles.append( (raw_lbl, inst_mask.astype(bool)) )

    return obstacles