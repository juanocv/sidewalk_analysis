# sidewalk_ai/models/ensemble.py
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from sidewalk_ai.processing.fusion import logical_fuse

from .base import SegmentationOutput, Segmenter, SegmentInfo

FuseMethod = Literal["or", "and", "majority"]
_METHODS: tuple[str, ...] = ("or", "and", "majority")


class EnsembleSegmenter(Segmenter):
    """
    Combine two or more Segmenters into one.

    Only the **sidewalk mask** is fused, through
    :func:`~sidewalk_ai.processing.fusion.logical_fuse`, so all three fusion
    rules ("or", "and", "majority") are available.

    The panoptic map is *not* fusable: segment ids come from unrelated label
    spaces per back-end. The map of the first member that supplies one is
    passed through instead, which keeps obstacle extraction working — the
    pipeline intersects those segments with the fused sidewalk mask, so the
    ensemble still decides *where* the sidewalk is.

    Note that ``"majority"`` only differs from ``"and"`` with three or more
    members: across two masks a strict majority already requires both to agree.
    """

    def __init__(self, *segmenters: Segmenter, method: FuseMethod = "or") -> None:
        members = list(segmenters)
        if len(members) < 2:
            raise ValueError("EnsembleSegmenter needs at least two segmenters")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {', '.join(_METHODS)}; got {method!r}")
        self.members = members
        self.method = method

    def segment(self, img_rgb, target_label="sidewalk", *, device=None):
        outputs = [
            SegmentationOutput.coerce(
                member.segment(img_rgb, target_label), source=type(member).__name__
            )
            for member in self.members
        ]

        fused = logical_fuse([out.mask for out in outputs], method=self.method).astype(bool)

        seg_map, seg_info = _first_panoptic(outputs, fused.shape)

        # With a panoptic map available the pipeline rebuilds obstacles from it
        # and ignores this list, so only bother when there is none.
        obstacles = [] if seg_map is not None else _merge_obstacles(outputs, fused)

        return SegmentationOutput(fused, seg_map, seg_info, obstacles)


def _first_panoptic(
    outputs: Sequence[SegmentationOutput],
    shape: tuple[int, ...],
) -> tuple[np.ndarray | None, list[SegmentInfo] | None]:
    """
    Panoptic map of the first member that has one matching the fused shape.

    A member whose map is a different size was resized during fusion, so its
    segment ids would no longer line up with the fused mask; skip it rather
    than emit a misaligned map.
    """
    for out in outputs:
        if out.seg_map is None or out.seg_info is None:
            continue
        if tuple(np.shape(out.seg_map)[:2]) != tuple(shape):
            continue
        return out.seg_map, out.seg_info
    return None, None


def _merge_obstacles(
    outputs: Sequence[SegmentationOutput],
    fused: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """
    Pool the members' own obstacle lists, keeping those that touch the fused
    sidewalk. Used only when no member exposes a panoptic map; two members may
    report the same physical object, so counts from this path are upper bounds.
    """
    merged: list[tuple[str, np.ndarray]] = []
    for out in outputs:
        for label, mask in out.obstacles:
            m = np.asarray(mask).astype(bool)
            if m.shape != fused.shape:
                continue
            if not (m & fused).any():
                continue
            merged.append((label, m))
    return merged
