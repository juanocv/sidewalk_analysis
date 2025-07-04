# sidewalk_ai/models/ensemble.py
from __future__ import annotations
import numpy as np
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
        m1, _, _ = self.a.segment(img_rgb, target_label)
        m2, _, _ = self.b.segment(img_rgb, target_label)
        fuse = m1 | m2 if self.method == "or" else m1 & m2
        return fuse, None, None