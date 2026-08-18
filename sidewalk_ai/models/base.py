# sidewalk_ai/models/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, Tuple

import numpy as np


class SegmentInfo(Tuple[int, str]):  # (id, label)
    """Lightweight copy of the HF / Detectron segment dict."""


@dataclass(slots=True, frozen=True)
class SegmentationOutput:
    """
    What a segmentation back-end hands back.

    This used to be a bare tuple of three or four elements, which callers told
    apart with ``len(out)``. Falling through that check left every name unbound
    and surfaced far away as a ``NameError``, and neither the arity nor the
    order was written down anywhere a reader would look.

    mask
        H×W bool, the pixels belonging to the target class.
    seg_map
        H×W panoptic/semantic id map. ``None`` when the back-end has none, which
        makes the pipeline skip obstacle extraction entirely.
    seg_info
        ``(id, label)`` pairs describing *seg_map*.
    obstacles
        ``(label, H×W bool)`` pairs the back-end found itself. Only consulted
        when *seg_map* is ``None``; otherwise the pipeline derives obstacles
        from the map so every back-end goes through the same rules.
    ignore_labels, sidewalk_labels
        Which names in *seg_info* are background and which are the target.
        They travel with the map because they describe it: ADE20K's 150
        classes and Cityscapes' 19 do not share a vocabulary. ``None`` means
        the ADE20K-shaped defaults in ``models._obstacles``.
    """

    mask: np.ndarray
    seg_map: np.ndarray | None = None
    seg_info: list[SegmentInfo] | None = None
    obstacles: list[tuple[str, np.ndarray]] = field(default_factory=list)
    ignore_labels: frozenset[str] | None = None
    sidewalk_labels: frozenset[str] | None = None

    @classmethod
    def coerce(cls, value: Any, *, source: str | None = None) -> "SegmentationOutput":
        """
        Accept this type, or the legacy 3-/4-tuple a third-party back-end returns.

        *source* names the offending segmenter in the error, so a bad return
        value points at its author rather than at the pipeline.
        """
        if isinstance(value, cls):
            return value

        origin = f"{source}.segment()" if source else "segment()"
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(
                f"{origin} returned {type(value).__name__}; expected SegmentationOutput"
            )

        if len(value) == 3:
            mask, seg_map, seg_info = value
            obstacles: Sequence = ()
        elif len(value) == 4:
            mask, seg_map, seg_info, obstacles = value
        else:
            raise TypeError(
                f"{origin} returned {len(value)} values; expected a SegmentationOutput, "
                "or (mask, seg_map, seg_info[, obstacles])"
            )
        return cls(mask, seg_map, seg_info, list(obstacles or []))


class Segmenter(Protocol):
    """Common contract for *all* segmentation back-ends."""

    def segment(
        self,
        img_rgb: np.ndarray,
        target_label: str | list[str] = "sidewalk",
        *,
        device: str | None = None,
    ) -> SegmentationOutput: ...


class DepthEstimator(Protocol):
    """
    Common behaviour for depth back-ends.

    `is_metric`
        *True*  -> returned depth is already in **metres** (ZoeDepth).
        *False* -> needs ground-plane scaling (MiDaS, etc.).
    """

    is_metric: bool = False

    def predict(self, img_rgb: np.ndarray) -> np.ndarray: ...
