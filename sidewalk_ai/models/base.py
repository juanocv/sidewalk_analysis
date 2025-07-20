# sidewalk_ai/models/base.py
from __future__ import annotations
import numpy as np
from typing import Protocol, Tuple, List


class SegmentInfo(Tuple[int, str]):  # (id, label)
    """Lightweight copy of the HF / Detectron segment dict."""


class Segmenter(Protocol):
    """
    Common contract for *all* panoptic back-ends.

    Returns
    -------
    mask         : (H,W)  bool  – pixels that belong to the target class
    seg_map      : (H,W)  int16 – panoptic id map   (optional for callers)
    seg_info     : list[SegmentInfo]               (optional)
    """
    def segment(
        self,
        img_rgb: np.ndarray,
        target_label: str | list[str] = "sidewalk",
        *,
        device: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, list[SegmentInfo] | None]:
        ...

class DepthEstimator(Protocol):
    """
    Common behaviour for depth back-ends.

    `is_metric`
        *True*  -> returned depth is already in **metres** (ZoeDepth).  
        *False* -> needs ground-plane scaling (MiDaS, etc.).
    """
    is_metric: bool = False

    def predict(self, img_rgb: np.ndarray) -> np.ndarray: ...