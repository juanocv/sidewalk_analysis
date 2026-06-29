# sidewalk_ai/models/ensemble.py
from __future__ import annotations
from typing import Literal

from .base import Segmenter


class EnsembleSegmenter(Segmenter):
    """
    Combine **any two** Segmenters with “or” / “and” fusion.
    """

    def __init__(
        self,
        seg1: Segmenter,
        seg2: Segmenter,
        *,
        method: Literal["or", "and"] = "or",
    ):
        self.a = seg1
        self.b = seg2
        if method not in {"or", "and"}:
            raise ValueError("method must be 'or' or 'and'")
        self.method = method

    def segment(self, img_rgb, target_label="sidewalk", *, device=None):
        out1 = self.a.segment(img_rgb, target_label)
        out2 = self.b.segment(img_rgb, target_label)
        # normalize to (mask, seg_map, seg_info, obstacles)
        if len(out1) == 3:
            m1, sm1, si1 = out1
            obs1 = []
        else:
            m1, sm1, si1, obs1 = out1
        if len(out2) == 3:
            m2, sm2, si2 = out2
            obs2 = []
        else:
            m2, sm2, si2, obs2 = out2
        fuse = m1 | m2 if self.method == "or" else m1 & m2
        # we don't have a meaningful seg_map/seg_info for the fused result
        return fuse, None, None, []
