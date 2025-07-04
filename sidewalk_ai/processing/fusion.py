# sidewalk_ai/processing/fusion.py
"""
Mask-fusion helpers.

All functions are **pure** (NumPy in → NumPy out, no prints, no I/O) so
they slot cleanly into unit tests and the SidewalkPipeline.
"""
from __future__ import annotations

from typing import Literal, Sequence, Tuple

import cv2
import numpy as np


__all__ = [
    "resize_like",
    "logical_fuse",
    "weighted_soft_fuse",
]


# --------------------------------------------------------------------------- #
# 0)  Resize helper – keeps boilerplate out of the public API                 #
# --------------------------------------------------------------------------- #
def resize_like(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Return *src* resized to the height/width of *ref* with **nearest**
    interpolation.  Supports boolean / uint8 without type surprises.
    """
    if src.shape == ref.shape:
        return src

    h, w = ref.shape[:2]
    resized = cv2.resize(
        src.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(src.dtype)


# --------------------------------------------------------------------------- #
# 1)  Logical fusion (crisp masks)                                            #
# --------------------------------------------------------------------------- #
def logical_fuse(
    masks: Sequence[np.ndarray],
    method: Literal["or", "and", "majority"] = "or",
) -> np.ndarray:
    """
    Combine an arbitrary list of **binary masks** with three strategies:

    * **"or"**       – union of positives (default, tolerant);
    * **"and"**      – intersection of positives (strict);
    * **"majority"** – pixel is 1 iff >½ of masks vote 1 (robust).

    The function auto-resizes all masks to the shape of the first one.
    All inputs must be 2-D and convertible to bool.
    """
    if not masks:
        raise ValueError("No masks provided")

    ref = masks[0]
    stack = []

    for m in masks:
        if m.ndim != 2:
            raise ValueError("Masks must be 2-D (H×W)")
        stack.append(resize_like(m, ref).astype(bool))

    arr = np.stack(stack, axis=0)            # (N, H, W)

    if method == "or":
        fused = arr.any(axis=0)
    elif method == "and":
        fused = arr.all(axis=0)
    elif method == "majority":
        fused = arr.sum(axis=0) > (len(masks) // 2)
    else:                                   # pragma: no cover
        raise ValueError(f"Unknown method: {method}")

    return fused.astype(np.uint8)            # 0/1 uint8


# --------------------------------------------------------------------------- #
# 2)  Soft fusion (confidence maps)                                           #
# --------------------------------------------------------------------------- #
def weighted_soft_fuse(
    soft_masks: Sequence[Tuple[np.ndarray, float]],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    **Optional** advanced routine for when your back-ends output *probability*
    maps instead of crisp masks.

    Parameters
    ----------
    soft_masks
        Sequence of *(H×W float32, weight)*.
    threshold
        Pixel is positive if the weighted average ≥ *threshold*.

    Returns
    -------
    (H×W) uint8
    """
    if not soft_masks:
        raise ValueError("No soft masks given")

    ref = soft_masks[0][0]
    acc  = np.zeros_like(ref, dtype="float32")
    norm = 0.0

    for sm, w in soft_masks:
        acc  += resize_like(sm, ref).astype("float32") * w
        norm += w

    fused = (acc / norm) >= threshold
    return fused.astype(np.uint8)